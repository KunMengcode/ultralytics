---
comments: true
description: Jointly train one Ultralytics YOLO detection model on multiple datasets at once, with one detection head per dataset and an optional shared-backbone-only design.
keywords: Ultralytics, YOLO, multi-dataset training, joint training, multi-head, shared backbone, shared neck, MultiDatasetDetectionTrainer, detection
---

# Multi-Dataset Training

!!! warning "Experimental"

    Multi-dataset joint training is experimental. The `MultiDatasetDetectionTrainer`, the `share` argument, and the
    multi-output export described on this page may change without notice.

Multi-dataset joint training trains **one** detection model on several datasets at the same time. Instead of
[fine-tuning](finetuning-guide.md) each dataset one after another, which can cause the model to forget earlier data,
every dataset feeds batches to the same optimizer and the model learns all class sets together.

Every dataset keeps its own **detection head** (its own number of classes), so different label spaces never collide at
the classification layer. The `share` argument selects how much of the network is shared between datasets:

| `share`         | Shared weights  | Per-dataset weights    |
| --------------- | --------------- | ---------------------- |
| `all` (default) | Backbone + neck | Detection heads        |
| `backbone`      | Backbone        | Neck + detection heads |

## How It Works

Ultralytics builds a `MultiHeadDetectionModel` from the model YAML: the backbone (and optionally the neck) is built
once, while the detection head becomes a `MultiDetect` container holding one `Detect` head per dataset. With
`share="backbone"`, every neck layer after the backbone is replicated into a per-dataset `DatasetBranch` so each
dataset owns an independent neck as well.

Batches must come from a single dataset at a time. The `MultiDatasetConcatDataset` tags every sample with its source
`dataset_idx`, and the `MultiDatasetBatchSampler` draws each batch from one dataset in proportion to its size. The
model then runs only the head of the batch's dataset, and the loss routes to that head's criterion. This keeps the
per-head training math identical to single-dataset training.

During validation the trainer evaluates each dataset with its own head and its own class names, so metrics, plots, and
confusion matrices stay local to each dataset. The mean fitness across datasets selects the best checkpoint.

## Dataset YAML

Point `data` at a multi-dataset YAML that lists one standard dataset YAML per entry:

```yaml
# multi.yaml
train:
    - african-wildlife.yaml
    - HomeObjects-3K.yaml
val: # optional; defaults to the train list (each dataset's own val split)
    - african-wildlife.yaml
    - HomeObjects-3K.yaml
```

Rules:

- `train` must be a non-empty list of dataset YAML files.
- `val` is optional; when omitted, every dataset validates its own validation split. When set, it must list exactly one
  YAML per train entry so datasets pair with heads by index.
- Relative paths are resolved against the directory of the multi-dataset YAML.
- Pass a single YAML path as `data`. A list of datasets is treated as sequential fine-tuning, not joint training.

## Train

Multi-dataset training needs the `MultiDatasetDetectionTrainer`:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import MultiDatasetDetectionTrainer

model = YOLO("yolo11n.pt")
model.train(
    data="multi.yaml",
    trainer=MultiDatasetDetectionTrainer,
    share="backbone",  # "all" (default) shares backbone+neck; "backbone" keeps the neck per-dataset
    epochs=100,
    imgsz=640,
)
```

Use `share="all"` to share the backbone and neck, and `share="backbone"` when the datasets differ enough that a single
neck would be a bottleneck. Per-dataset training losses are printed after each epoch, and per-dataset, per-epoch
validation metrics are written to `metrics.json` in the run directory.

## Predict and Export

A trained multi-head checkpoint predicts with `MultiHeadDetectionPredictor`, which runs NMS on each dataset head,
offsets the class indices into the model's flattened namespace, and merges the detections:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import MultiHeadDetectionPredictor

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict("image.jpg", predictor=MultiHeadDetectionPredictor)
```

To score a single dataset with its own head, use `MultiDatasetDetectionValidator` and set `head_index`:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import MultiDatasetDetectionValidator

model = YOLO("runs/detect/train/weights/best.pt")
validator = MultiDatasetDetectionValidator(args=dict(data="african-wildlife.yaml"))
validator.head_index = 0
metrics = model.val(data="african-wildlife.yaml", validator=validator)
```

Export produces one output tensor per dataset head, so a two-dataset model exports as a graph with `output0` and
`output1`:

```python
model.export(format="onnx")
```

## Limitations

- `compile=True` is not supported for multi-dataset training; the trainer raises an error.
- AutoBatch (`batch=-1`) is not supported because batches cannot mix dataset sizes.
- The `share` argument only affects `MultiDatasetDetectionTrainer`; single-dataset training ignores it.
