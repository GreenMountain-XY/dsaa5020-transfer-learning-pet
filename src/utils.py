"""Utility functions: reproducibility, device handling, accurate timing, and I/O."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN can slow down slightly; benchmark=False ensures stability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_device: str = "auto") -> torch.device:
    """Select CUDA > MPS > CPU automatically unless explicitly specified."""
    if prefer_device != "auto":
        return torch.device(prefer_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize accelerator before wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def timed_now(device: torch.device) -> float:
    """Return synchronized perf_counter timestamp."""
    sync_device(device)
    return time.perf_counter()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_gpu_info(device: torch.device) -> Dict[str, Any]:
    """Collect hardware metadata for fair timing interpretation."""
    info: Dict[str, Any] = {"device_type": device.type}
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_mem_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["cuda_version"] = torch.version.cuda
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon MPS"
    else:
        info["gpu_name"] = "CPU"
    info["torch_version"] = torch.__version__
    return info


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
