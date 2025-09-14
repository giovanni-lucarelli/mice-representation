"""Data loading and splitting utilities for image classification.

This module defines `DataManager`, a utility to:
- load an `ImageFolder` dataset and human-readable labels,
- create train/val/test splits with stratification,
- apply appropriate augmentations and normalization,
- expose ready-to-use PyTorch `DataLoader`s.
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from typing import Optional, Dict, Any

import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from pathlib import Path

from .mini_imagenet import MiniImageNet, AsTupleDataset

#? -------------------------------------------------------------- #
#?                         Data Manager                           #
#? -------------------------------------------------------------- #

class IndexedDataset(Dataset):
    """Wrap a dataset to also return its own index as first element.

    This provides stable per-sample indices in [0, len(dataset)) independent of any
    underlying base dataset indices, which is ideal for memory banks.
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if isinstance(item, tuple):
            if len(item) == 2:
                image, label = item
                return index, image, label
            else:
                return (index,) + item
        return index, item, -1


class DataManager():
    """Manage MiniImageNet dataset, splits, transforms, and data loaders."""
    def __init__(self,
                 data_path      : str,
                 batch_size     : int                           = 512,
                 train_transform: Optional[transforms.Compose]  = None,
                 eval_transform : Optional[transforms.Compose]  = None,
                 num_workers    : int                           = 8,
                 train_split    : float                         = 0.7,
                 val_split      : float                         = 0.15,
                 split_seed     : int                           = 42,
                 use_cuda       : bool                          = True,
                 return_indices : bool                          = False):
        """Initialize manager configuration and default transforms.

        - train_split and val_split are fractions of the whole dataset;
          test size is inferred as 1 - (train_split + val_split).
        - If no transforms are provided, sensible defaults for 224x224 images are used.
        - split_seed controls reproducible shuffling in dataset splits.
        """
        self.data_path = Path(data_path)
            
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.split_seed = split_seed
        self.use_cuda = use_cuda
        self.return_indices = return_indices
        
        if train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224), # Standard per ImageNet
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.train_transform = train_transform

        if eval_transform is None:
            self.eval_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.eval_transform = eval_transform

        # Ensure transforms always output tensors (not dicts)
        def _ensure_tensor_transform(t):
            if t is None:
                return None
            def _call(inp: Any):
                out = t(inp)
                if isinstance(out, dict):
                    return out.get('imgs', out.get('image', out))
                return out
            return _call

        self.train_transform = _ensure_tensor_transform(self.train_transform)
        self.eval_transform = _ensure_tensor_transform(self.eval_transform)
            
        # Dataset attributes
        self.dataset        : Optional[ImageFolder]     = None
        self._base_dataset  : Optional[ImageFolder]     = None
        self.train_dataset  : Optional[Subset]          = None
        self.val_dataset    : Optional[Subset]          = None
        self.test_dataset   : Optional[Subset]          = None
        
        # DataLoader attributes
        self.train_loader   : Optional[DataLoader] = None
        self.val_loader     : Optional[DataLoader] = None
        self.test_loader    : Optional[DataLoader] = None
        
        # Dataset info
        self.num_classes : Optional[int]    = None
        
    def load_data(self):
        """Create base MiniImageNet dataset and populate class metadata."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path {self.data_path} does not exist")

        print(f"Loading ImageFolder from {self.data_path}")
        # If the dataset path contains predefined splits (train/val or train/test),
        # use the train split as the base dataset to avoid treating split folders as classes.
        split_train = self.data_path / "train"
        split_val = self.data_path / "val"
        split_test = self.data_path / "test"
        if split_train.is_dir() and (split_val.is_dir() or split_test.is_dir()):
            base_root = split_train
        else:
            base_root = self.data_path

        self._base_dataset = ImageFolder(root=base_root, transform=None)
        self.dataset = self._base_dataset

        self.num_classes = len(self.dataset.classes)

        print(f"Dataset loaded: {len(self.dataset)} samples, {self.num_classes} classes")

    def _has_predefined_splits(self) -> bool:
        """Return True if dataset directory contains train/val or train/test folders."""
        root: Path = self.data_path
        return (root / "train").is_dir() and ((root / "val").is_dir() or (root / "test").is_dir())

    def _make_split_dataset(self, split: str, transform):
        """Create a MiniImageNet split dataset wrapped to return (image, label)."""
        ds = MiniImageNet(
            root=str(self.data_path),
            split=split,
            transform=transform,
            verbose=False,
        )
        return AsTupleDataset(ds)
        
    def split_data(self):
        """Create train/val/test datasets: use predefined splits when available, else stratified split."""

        if self._base_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        if self._has_predefined_splits():
            print("Using predefined dataset splits (train/val/test)")
            self.train_dataset = self._make_split_dataset("train", self.train_transform)
            self.val_dataset = self._make_split_dataset("val", self.eval_transform)
            self.test_dataset = self._make_split_dataset("test", self.eval_transform)

            print(f"Train dataset: {len(self.train_dataset)} samples")
            print(f"Val dataset: {len(self.val_dataset)} samples")
            print(f"Test dataset: {len(self.test_dataset)} samples")
            return

        print(f"Splitting dataset into train, val, and test sets")

        indices = list(range(len(self._base_dataset)))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=1 - self.train_split,
            stratify=[self._base_dataset.targets[i] for i in indices],
            shuffle=True,
            random_state=self.split_seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=self.val_split / (1 - self.train_split),
            stratify=[self._base_dataset.targets[i] for i in temp_idx],
            shuffle=True,
            random_state=self.split_seed
        )

        train_full = ImageFolder(root=self.data_path, transform=self.train_transform)
        eval_full = ImageFolder(root=self.data_path, transform=self.eval_transform)

        self.train_dataset = Subset(train_full, train_idx)
        self.val_dataset = Subset(eval_full, val_idx)
        self.test_dataset = Subset(eval_full, test_idx)

        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
        
    def _seed_worker(self, worker_id: int):
        """Ensure deterministic seeding for each DataLoader worker."""
        import random
        import numpy as _np
        worker_seed = self.split_seed + worker_id
        random.seed(worker_seed)
        _np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def create_loaders(self):
        """Instantiate PyTorch `DataLoader`s with safe defaults.

        - `effective_workers` is clamped to avoid shared-memory exhaustion.
        - `persistent_workers` disabled for stability across restarts.
        - `prefetch_factor` configurable via env var when workers > 0.
        """
        use_cuda = self.use_cuda and torch.cuda.is_available()
        pin = use_cuda
        effective_workers = min(self.num_workers, int(os.environ.get("NUM_WORKERS_OVERRIDE", "8")))
        persistent = False
        prefetch = int(os.environ.get("PREFETCH_FACTOR", "2")) if effective_workers > 0 else None

        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise ValueError("Train, val, and test datasets not provided")

        g = torch.Generator()
        g.manual_seed(self.split_seed)
        
        train_ds = self.train_dataset
        if self.return_indices:
            train_ds = IndexedDataset(train_ds)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=effective_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            worker_init_fn=self._seed_worker,
            generator=g,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=effective_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            worker_init_fn=self._seed_worker,
            generator=g,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=effective_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            worker_init_fn=self._seed_worker,
            generator=g,
        )
        
    #? ------------------------ Setup -------------------------
        
    def setup(self):
        """Convenience one-liner: load, split, and create loaders."""
        self.load_data()
        self.split_data()
        self.create_loaders()
        return self.train_loader, self.val_loader, self.test_loader
        
        
        