"""Dataset loading, stratified nested splitting, transforms, and EDA for Oxford-IIIT Pet."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Official 12 cat breeds in Oxford-IIIT Pet (the remaining 25 are dog breeds)
CAT_BREEDS_NORMALIZED = {
    "abyssinian",
    "bengal",
    "birman",
    "bombay",
    "british shorthair",
    "egyptian mau",
    "maine coon",
    "persian",
    "ragdoll",
    "russian blue",
    "siamese",
    "sphynx",
}


def is_cat_breed(class_name: str) -> bool:
    norm = class_name.replace("_", " ").strip().lower()
    return norm in CAT_BREEDS_NORMALIZED


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, eval_transform) using standard ImageNet preprocessing."""
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def _extract_labels(ds: datasets.OxfordIIITPet) -> np.ndarray:
    """Fast label extraction without decoding images."""
    if hasattr(ds, "_labels"):
        return np.asarray(ds._labels, dtype=np.int64)
    return np.asarray([ds[i][1] for i in range(len(ds))], dtype=np.int64)


def get_fixed_train_val_indices(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically split official trainval into (train_pool, fixed_val)."""
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=split_seed
    )
    train_pool_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(train_pool_idx), np.sort(val_idx)


def get_nested_stratified_subset(
    train_pool_idx: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    subset_seed: int = 42,
) -> np.ndarray:
    """Return a stratified subset of train_pool_idx.

    For a given subset_seed, subsets are strictly nested:
        subset(0.05) ⊂ subset(0.10) ⊂ subset(0.25) ⊂ subset(0.50) ⊂ subset(1.00).
    Across different subset_seeds, the 10% subset resamples different examples per class,
    so multi-seed runs honestly capture both data-sampling and training variance.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        return train_pool_idx.copy()

    pool_labels = labels[train_pool_idx]
    classes = np.unique(pool_labels)
    selected: List[int] = []

    for c in classes:
        cls_positions = train_pool_idx[pool_labels == c]
        rng = np.random.default_rng(subset_seed * 10007 + int(c) * 97 + 13)
        permuted = rng.permutation(cls_positions)
        k = max(1, int(round(len(permuted) * fraction)))
        selected.extend(permuted[:k].tolist())

    return np.asarray(sorted(selected), dtype=np.int64)


def prepare_dataloaders(
    data_dir: str | Path,
    fraction: float = 1.0,
    seed: int = 42,
    split_seed: int = 42,
    val_ratio: float = 0.2,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 2,
    img_size: int = 224,
    download: bool = True,
) -> Dict[str, Any]:
    """Build train / val / test DataLoaders plus metadata for Oxford-IIIT Pet."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = get_transforms(img_size=img_size)

    # Two handles over trainval so train gets augmentations and val stays deterministic
    trainval_aug = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=download,
    )
    trainval_eval = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="trainval",
        target_types="category",
        transform=eval_tf,
        download=False,
    )
    test_ds = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="test",
        target_types="category",
        transform=eval_tf,
        download=download,
    )

    trainval_labels = _extract_labels(trainval_aug)
    test_labels = _extract_labels(test_ds)
    train_pool_idx, val_idx = get_fixed_train_val_indices(
        trainval_labels, val_ratio=val_ratio, split_seed=split_seed
    )
    train_sub_idx = get_nested_stratified_subset(
        train_pool_idx, trainval_labels, fraction=fraction, subset_seed=seed
    )

    # Sanity check: zero overlap between train and val
    assert len(set(train_sub_idx.tolist()) & set(val_idx.tolist())) == 0

    train_subset = Subset(trainval_aug, train_sub_idx.tolist())
    val_subset = Subset(trainval_eval, val_idx.tolist())

    pin_memory = torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=min(batch_size, len(train_subset)),
        shuffle=True,
        drop_last=False,
        generator=g,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    class_names = list(trainval_aug.classes)
    species_map = {name: ("Cat" if is_cat_breed(name) else "Dog") for name in class_names}

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "num_classes": len(class_names),
        "species_map": species_map,
        "counts": {
            "train_100": int(len(train_pool_idx)),
            "train_used": int(len(train_sub_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_ds)),
        },
        "train_indices": train_sub_idx,
        "val_indices": val_idx,
        "trainval_labels": trainval_labels,
        "test_labels": test_labels,
        "test_dataset": test_ds,
    }


def compute_eda_summary(
    data_dir: str | Path,
    split_seed: int = 42,
    val_ratio: float = 0.2,
    download: bool = True,
) -> Dict[str, Any]:
    """Compute class distribution statistics for Requirement 1 (EDA)."""
    data_dir = Path(data_dir)
    trainval_ds = datasets.OxfordIIITPet(
        root=str(data_dir), split="trainval", target_types="category", download=download
    )
    test_ds = datasets.OxfordIIITPet(
        root=str(data_dir), split="test", target_types="category", download=download
    )

    tv_labels = _extract_labels(trainval_ds)
    test_labels = _extract_labels(test_ds)
    class_names = list(trainval_ds.classes)

    train_pool_idx, val_idx = get_fixed_train_val_indices(
        tv_labels, val_ratio=val_ratio, split_seed=split_seed
    )
    train_10_idx = get_nested_stratified_subset(
        train_pool_idx, tv_labels, fraction=0.10, subset_seed=split_seed
    )

    c_train100 = Counter(tv_labels[train_pool_idx].tolist())
    c_train10 = Counter(tv_labels[train_10_idx].tolist())
    c_val = Counter(tv_labels[val_idx].tolist())
    c_test = Counter(test_labels.tolist())

    per_class = []
    for idx, name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": name,
                "species": "Cat" if is_cat_breed(name) else "Dog",
                "train_100": c_train100[idx],
                "train_10": c_train10[idx],
                "val": c_val[idx],
                "test": c_test[idx],
                "total": c_train100[idx] + c_val[idx] + c_test[idx],
            }
        )

    n_cats = sum(1 for row in per_class if row["species"] == "Cat")
    n_dogs = sum(1 for row in per_class if row["species"] == "Dog")

    return {
        "dataset": "Oxford-IIIT Pet",
        "num_classes": len(class_names),
        "num_cat_breeds": n_cats,
        "num_dog_breeds": n_dogs,
        "total_images": int(len(tv_labels) + len(test_labels)),
        "splits": {
            "official_trainval": int(len(tv_labels)),
            "train_100": int(len(train_pool_idx)),
            "train_10": int(len(train_10_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_labels)),
        },
        "per_class": per_class,
    }
