# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import math
import random
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ultralytics.data import MultiDatasetBatchSampler, MultiDatasetConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, MultiHeadDetectionModel, load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK, YAML, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first, unwrap_model

from .val import MultiDatasetDetectionValidator


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Move a batch to the device, normalize images, and apply optional multi-scale resizing.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Plot the class and bounding box label distributions of the training dataset.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            device=self.device,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Move a batch to the device, normalize images to [0, 1], and apply optional multi-scale resizing.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

    def set_model_names_for_load(self, model):
        """Set target dataset names before loading weights so cls heads can remap by name."""
        if getattr(self.args, "cls_remap", True) and self.data.get("names"):
            model.names = self.data["names"]
        return model

    def get_class_counts(self):
        """Return per-class instance counts from the training dataset labels."""
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        return np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)

    def compute_class_weights(self, class_counts):
        """Return inverse-frequency weights raised to cls_pw, giving classes without train labels the mean weight."""
        present = class_counts > 0
        weights = (1.0 / np.where(present, class_counts, 1.0)) ** self.args.cls_pw
        return np.where(present, weights, weights[present].mean())

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens; values are restricted to the range [0, 1]).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        class_counts = self.get_class_counts()
        if not class_counts.any():  # nothing counted (e.g. missing/unreadable masks); keep default weights
            return
        weights = self.compute_class_weights(class_counts)
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model  # distillation: the student model builds the loss criterion
        model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str | dict, optional): Model configuration file path or dictionary.
            weights (torch.nn.Module, optional): Pretrained model whose weights are loaded into the new model.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = self.set_model_names_for_load(
            DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _build_train_pipeline(self):
        """Build the training pipeline and align the default detection limit with observed dataset object counts."""
        super()._build_train_pipeline()
        if self.args.task in {"detect", "segment", "pose", "obb"}:
            datasets = {"train": self.train_loader.dataset, "val": self.test_loader.dataset}
            yolo.detect.DetectionValidator._check_max_det(self.args, datasets)
            unwrap_model(self.model).set_head_attr(max_det=self.args.max_det)

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Plot the class and bounding box label distributions of the training dataset."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)


def _unique_labels(stems: list[str]) -> list[str]:
    """Return display names for the datasets, disambiguating repeated YAML stems with an index suffix."""
    labels, seen = [], {}
    for stem in stems:
        seen[stem] = seen.get(stem, 0) + 1
        labels.append(stem if seen[stem] == 1 else f"{stem}-{seen[stem]}")
    return labels


def _display_names(entries: list) -> list[str]:
    """Return a display name per dataset YAML, preferring the parent directory for generic file names."""
    names = []
    for entry in entries:
        path = Path(str(entry))
        stem = path.stem
        names.append(path.parent.name if stem.lower() in {"data", "dataset"} and path.parent.name else stem)
    return _unique_labels(names)


def _flat_names(labels: list[str], names_list: list[dict[int, str]]) -> dict[int, str]:
    """Flatten per-dataset class names into one namespace, prefixing names that repeat across datasets."""
    flat, used = {}, set()
    offset = 0
    for label, names in zip(labels, names_list):
        for j in sorted(names):
            name = names[j]
            key = f"{label}/{name}" if name in used else name
            used.add(name)
            flat[offset + j] = key
        offset += len(names)
    return flat


class MultiDatasetDetectionTrainer(DetectionTrainer):
    """Detection trainer for joint training on multiple datasets with one detection head per dataset.

    The `data` argument points to a multi-dataset YAML listing one standard dataset YAML per entry:

    ```yaml
    train:
      - /data/ds1/data.yaml
      - /data/ds2/data.yaml
    val: # optional; defaults to the train list (each dataset's own validation split)
      - /data/ds1/data.yaml
      - /data/ds2/data.yaml
    ```

    The backbone and neck are shared while each dataset owns a `Detect` head, and `MultiDatasetBatchSampler` draws
    each batch from a single dataset in proportion to its size. Per-dataset training losses and validation metrics
    are logged separately; the mean fitness across datasets selects the best checkpoint.

    Attributes:
        dataset_cfgs (list[dict]): Checked dataset dictionaries, one per training dataset (head order).
        val_cfgs (list[dict]): Checked dataset dictionaries, one per validation dataset (paired with heads by index).
        dataset_labels (list[str]): Display names of the datasets.
        val_loaders (list[DataLoader]): One validation loader per dataset.
        per_dataset_metrics (dict): Epoch -> {dataset name: metrics} for every validation pass.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.models.yolo.detect import MultiDatasetDetectionTrainer
        >>> model = YOLO("ultralytics/cfg/models/11/yolo11n.yaml")
        >>> model.train(data="multi.yaml", trainer=MultiDatasetDetectionTrainer, epochs=100)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize the multi-dataset trainer and register per-dataset logging callbacks.

        Args:
            cfg (dict, optional): Default configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Callback functions to run during training.
        """
        self.dataset_cfgs = []
        self.val_cfgs = []
        self.dataset_labels = []
        self.val_loaders = []
        self.per_dataset_metrics = {}
        self._loss_sum_by_dataset = {}
        self._loss_count_by_dataset = {}
        self._batch_dataset_idx = None
        super().__init__(cfg, overrides, _callbacks)
        if self.args.compile:
            raise ValueError("Multi-dataset multi-head training does not support compile=True; set compile=False.")
        self.add_callback("on_train_batch_end", self._accumulate_per_dataset_loss)
        self.add_callback("on_train_epoch_end", self._log_per_dataset_losses)

    def get_dataset(self) -> dict[str, Any]:
        """Load one checked dataset per entry of the multi-dataset YAML and merge their metadata.

        Returns:
            (dict[str, Any]): Merged dataset dictionary with per-dataset `nc`/`names` metadata.

        Raises:
            TypeError: If `data` is a list instead of a multi-dataset YAML path.
            SyntaxError: If the YAML does not list one dataset YAML per entry under `train`.
        """
        data = self.args.data
        if isinstance(data, (list, tuple)):
            raise TypeError(
                "Multi-dataset joint training expects a single multi-dataset YAML (e.g. data=multi.yaml); "
                "a data list is treated as sequential fine-tuning by Model.train()."
            )
        path = Path(str(data))
        multi = YAML.load(check_file(path))

        def load(entry, split=""):
            p = Path(str(entry))
            p = p if p.is_absolute() else (path.parent / p)
            if p.suffix not in {".yaml", ".yml"}:
                raise SyntaxError(f"multi-dataset entries must be dataset YAML files, but got '{entry}'")
            return check_det_dataset(str(p), split=split)

        train_entries = multi.get("train")
        if not isinstance(train_entries, (list, tuple)) or not train_entries:
            raise SyntaxError(
                f"{path.name} must map 'train' to a non-empty list of dataset YAMLs, but got {train_entries!r}"
            )
        val_entries = multi.get("val")
        val_entries = list(train_entries) if val_entries is None or val_entries == "" else val_entries
        if not isinstance(val_entries, (list, tuple)):
            val_entries = [val_entries]
        if len(val_entries) != len(train_entries):
            raise SyntaxError(
                f"{path.name} must list one val YAML per train YAML to pair datasets with heads, "
                f"but got {len(train_entries)} train and {len(val_entries)} val entries"
            )
        self.dataset_cfgs = [load(e) for e in train_entries]
        self.val_cfgs = [load(e, split=self.args.split) for e in val_entries]
        self.dataset_labels = _display_names(train_entries)

        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            for cfg in (*self.dataset_cfgs, *self.val_cfgs):
                cfg["names"] = {0: "item"}
                cfg["nc"] = 1
        else:
            for label, train_cfg, val_cfg in zip(self.dataset_labels, self.dataset_cfgs, self.val_cfgs):
                if train_cfg["names"] != val_cfg["names"]:
                    LOGGER.warning(
                        f"Validation dataset '{label}' class names differ from its training head; "
                        "metrics use the validation class names but predictions come from the training head."
                    )
        return {
            "train": [cfg["train"] for cfg in self.dataset_cfgs],
            "val": [cfg[self.args.split] for cfg in self.val_cfgs],
            "nc": sum(cfg["nc"] for cfg in self.dataset_cfgs),  # flattened total for shared trainer math
            "dataset_nc": [cfg["nc"] for cfg in self.dataset_cfgs],  # one entry per dataset head
            "names": _flat_names(self.dataset_labels, [cfg["names"] for cfg in self.dataset_cfgs]),
            "channels": self.dataset_cfgs[0].get("channels", 3),
            "path": path.parent,
        }

    def build_dataset(self, img_path: str | list, mode: str = "train", batch: int | None = None):
        """Build the mixed training dataset, or a single validation dataset in val mode.

        Args:
            img_path (str | list): Dataset image path used in val mode; ignored for the mixed train dataset.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): MultiDatasetConcatDataset in train mode, YOLODataset in val mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            datasets = [
                build_yolo_dataset(self.args, cfg["train"], batch, cfg, mode=mode, stride=gs, rect=False)
                for cfg in self.dataset_cfgs
            ]
            return MultiDatasetConcatDataset(datasets)
        cfg = self.val_cfgs[0]
        return build_yolo_dataset(self.args, cfg[self.args.split], batch, cfg, mode=mode, rect=True, stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return the mixed training loader, or build and return one validation loader per dataset.

        Args:
            dataset_path (str): Unused; datasets come from the parsed multi-dataset YAML.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for the mixed loader, 'val' for the per-dataset loaders.

        Returns:
            (DataLoader): The training loader, or the first validation loader (all live in `val_loaders`).
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        if mode == "train":
            with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            sampler = MultiDatasetBatchSampler(
                dataset.dataset_sizes,
                batch_size,
                num_replicas=self.world_size,
                rank=max(rank, 0),
                seed=int(self.args.seed) + 1,
            )
            loader = build_dataloader(
                dataset, batch=batch_size, workers=self.args.workers, batch_sampler=sampler, device=self.device
            )
            # DataLoader locks sampler/batch_size after init; iteration uses batch_sampler, while these keep the
            # DDP epoch sync (train_loader.sampler.set_epoch) and downstream batch-size math working.
            object.__setattr__(loader, "sampler", sampler)
            object.__setattr__(loader, "batch_size", batch_size)
            return loader
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        self.val_loaders = []
        for cfg in self.val_cfgs:
            with torch_distributed_zero_first(rank):
                ds = build_yolo_dataset(
                    self.args, cfg[self.args.split], batch_size, cfg, mode=mode, rect=True, stride=gs
                )
            self.val_loaders.append(
                build_dataloader(
                    ds, batch=batch_size, workers=self.args.workers, shuffle=False, rank=rank, device=self.device
                )
            )
        return self.val_loaders[0]

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
        """Return a multi-head YOLO detection model with one head per dataset.

        Args:
            cfg (str | dict, optional): Model configuration file path or dictionary.
            weights (torch.nn.Module, optional): Pretrained model whose weights seed the new model.
            verbose (bool): Whether to display model information.

        Returns:
            (MultiHeadDetectionModel): Multi-head detection model.
        """
        share = getattr(self.args, "share", "all")
        if share not in {"all", "backbone"}:
            raise ValueError(f"share must be 'all' or 'backbone', not {share!r}")
        model = MultiHeadDetectionModel(
            cfg or "yolo11n.yaml",
            ch=self.data["channels"],
            nc=self.data["dataset_nc"],
            share=share,
            verbose=verbose and RANK == -1,
        )
        model.names = self.data["names"]
        model.dataset_names = [dict(c["names"]) for c in self.dataset_cfgs]
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Attach the flattened class namespace and per-dataset metadata to the model."""
        self.model.nc = self.data["nc"]
        self.model.nc_list = list(self.data["dataset_nc"])
        self.model.names = self.data["names"]
        self.model.dataset_names = [dict(cfg["names"]) for cfg in self.dataset_cfgs]
        self.model.args = self.args

    def set_class_weights(self):
        """Compute per-head class weights from each dataset's labels when cls_pw > 0."""
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        weights = []
        for cfg, ds in zip(self.dataset_cfgs, self.train_loader.dataset.datasets):
            if not len(ds.labels):
                return
            counts = np.bincount(
                np.concatenate([lb["cls"].flatten() for lb in ds.labels]).astype(int), minlength=cfg["nc"]
            ).astype(np.float32)
            w = (1.0 / np.where(counts == 0, 1.0, counts)) ** self.args.cls_pw
            weights.append(torch.from_numpy(w / w.mean()).to(self.device))
        unwrap_model(self.model).per_head_class_weights = weights
        LOGGER.info(
            "Per-dataset class weights: "
            + ", ".join(f"{label}={w.cpu().numpy().round(3)}" for label, w in zip(self.dataset_labels, weights))
        )

    def plot_training_labels(self):
        """Plot label statistics of every dataset into its own subdirectory."""
        for label, cfg, ds in zip(self.dataset_labels, self.dataset_cfgs, self.train_loader.dataset.datasets):
            if not len(ds.labels):
                continue
            boxes = np.concatenate([lb["bboxes"] for lb in ds.labels], 0)
            cls = np.concatenate([lb["cls"] for lb in ds.labels], 0)
            path = self.save_dir / f"labels_{label}"
            path.mkdir(parents=True, exist_ok=True)
            plot_labels(boxes, cls.squeeze(), names=cfg["names"], save_dir=path, on_plot=self.on_plot)

    def auto_batch(self):
        """Raise: AutoBatch cannot mix batches of different sizes across dataset heads."""
        raise ValueError(
            "'batch=-1' (AutoBatch) is not supported for multi-dataset training; set an explicit batch size."
        )

    def preprocess_batch(self, batch):
        """Record the batch's source dataset before moving tensors to the device.

        Args:
            batch (dict): Batch dictionary that carries the `dataset_idx` tag in multi-dataset training.

        Returns:
            (dict): Preprocessed batch.
        """
        ds = batch.get("dataset_idx")
        if ds is not None and ds.numel():
            self._batch_dataset_idx = int(ds[0])
        return super().preprocess_batch(batch)

    def _accumulate_per_dataset_loss(self, trainer):
        """Accumulate each step's loss components under the dataset of the batch."""
        idx = self._batch_dataset_idx
        if idx is None or trainer.loss_items is None:
            return
        sums = self._loss_sum_by_dataset.setdefault(idx, {})
        for k, v in trainer.loss_items.items():
            v = v.detach()
            sums[k] = v if k not in sums else sums[k] + v
        self._loss_count_by_dataset[idx] = self._loss_count_by_dataset.get(idx, 0) + 1

    def _log_per_dataset_losses(self, trainer):
        """Log per-dataset mean training losses at the end of each epoch."""
        if RANK not in {-1, 0} or not self._loss_sum_by_dataset:
            return
        lines = []
        for i in sorted(self._loss_sum_by_dataset):
            label = self.dataset_labels[i] if i < len(self.dataset_labels) else f"dataset{i}"
            n = max(self._loss_count_by_dataset.get(i, 1), 1)
            losses = "  ".join(f"{k} {float(v) / n:.4g}" for k, v in self._loss_sum_by_dataset[i].items())
            lines.append(f"  {label}: batches {self._loss_count_by_dataset.get(i, 0)}  {losses}")
        LOGGER.info("train per-dataset losses:\n" + "\n".join(lines))
        self._loss_sum_by_dataset, self._loss_count_by_dataset = {}, {}

    def validate(self):
        """Validate every dataset with its own head and return mean metrics and mean fitness.

        Returns:
            (tuple): Mean metrics dict across datasets and mean fitness, or (None, None) without validation data.
        """
        if not self.val_loaders:
            return None, None
        per_dataset, all_metrics = {}, []
        for i, (label, loader, cfg) in enumerate(zip(self.dataset_labels, self.val_loaders, self.val_cfgs)):
            self.validator.dataloader = loader
            self.validator.head_index = i
            self.validator.dataset_names = dict(cfg["names"])
            self.validator.save_dir = self.save_dir / f"val_{label}"
            self.validator.save_dir.mkdir(parents=True, exist_ok=True)
            saved_data = self.data
            self.data = cfg  # BaseValidator installs trainer.data for metric names and label paths
            try:
                results = self.validator(self)
            finally:
                self.data = saved_data
            if results is None:
                continue
            per_dataset[label] = results
            all_metrics.append(results)
            LOGGER.info(
                colorstr("val: ")
                + f"{label}: "
                + ", ".join(f"{k} {v:.4g}" for k, v in results.items() if k != "fitness")
            )
        if not all_metrics:
            return None, None
        mean = {}
        for k in all_metrics[0]:
            if k == "fitness":
                continue
            vals = [m[k] for m in all_metrics if k in m and m[k] is not None]
            mean[k] = float(np.mean(vals)) if vals else 0.0
        fitnesses = [m["fitness"] for m in all_metrics if m.get("fitness") is not None]
        mean["fitness"] = float(np.mean(fitnesses)) if fitnesses else mean.get("metrics/mAP50-95(B)", 0.0)
        self.per_dataset_metrics[getattr(self, "_metrics_epoch_key", self.epoch)] = per_dataset
        self._write_per_dataset_metrics()
        fitness = mean.pop("fitness")
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return mean, fitness

    def _write_per_dataset_metrics(self):
        """Persist the per-dataset, per-epoch metrics next to the run outputs."""
        try:
            (self.save_dir / "metrics.json").write_text(
                json.dumps(self.per_dataset_metrics, indent=2), encoding="utf-8"
            )
        except OSError as e:
            LOGGER.warning(f"Failed to write metrics.json: {e}")

    def get_validator(self):
        """Return the validator that scores one dataset at a time with its matching head."""
        return MultiDatasetDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def final_eval(self):
        """Validate the best checkpoint on every dataset with its matching head."""
        best = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if best:
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if not best:
            return
        LOGGER.info(f"\nValidating {best}...")
        model, _ = load_checkpoint(best)
        model = model.to(self.device)
        model.args = self.args
        model.names = self.data["names"]
        model.criterion = None
        self.model = model
        self.ema = None
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self._metrics_epoch_key = "best"  # record the best-checkpoint pass separately from the epoch rows
        self.metrics, self.fitness = self.validate()
        del self._metrics_epoch_key
        self.epoch += 1  # log best metrics at step epochs+1, not overwriting the last epoch row
        self.run_callbacks("on_fit_epoch_end")
        self.epoch -= 1
