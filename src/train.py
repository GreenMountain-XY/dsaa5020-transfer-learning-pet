"""Training, evaluation, accurate GPU timing, and embedding extraction."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import prepare_dataloaders
from src.models import (
    DEFAULT_EPOCHS,
    build_model,
    get_classifier_layer,
    set_module_train_mode,
)
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    get_gpu_info,
    save_json,
    set_seed,
    timed_now,
)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on a DataLoader and return loss, accuracy, macro-F1, preds, targets."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        probs = torch.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.max(dim=1).values.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_conf = np.concatenate(all_probs)

    return {
        "loss": float(total_loss / max(1, total_samples)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_pred": y_pred,
        "y_true": y_true,
        "y_conf": y_conf,
    }


@torch.no_grad()
def extract_penultimate_features(
    model: nn.Module,
    backbone: str,
    loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings via a forward pre-hook on the classifier."""
    model.eval()
    feats_buf: List[np.ndarray] = []
    labels_buf: List[np.ndarray] = []

    captured: List[torch.Tensor] = []

    def _pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().float().cpu())

    clf_layer = get_classifier_layer(model, backbone)
    handle = clf_layer.register_forward_hook(lambda m, inp, out: _pre_hook(m, inp))

    collected = 0
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            captured.clear()
            _ = model(images)
            batch_feats = captured[0].numpy()
            feats_buf.append(batch_feats)
            labels_buf.append(targets.numpy())
            collected += len(targets)
            if max_samples is not None and collected >= max_samples:
                break
    finally:
        handle.remove()

    feats = np.concatenate(feats_buf, axis=0)
    labels = np.concatenate(labels_buf, axis=0)
    if max_samples is not None:
        feats = feats[:max_samples]
        labels = labels[:max_samples]
    return feats, labels


def run_single_experiment(
    backbone: str = "resnet18",
    strategy: str = "frozen",
    fraction: float = 0.10,
    seed: int = 42,
    data_dir: str | Path = "data",
    output_dir: str | Path = "results",
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    weight_decay: float = 1e-4,
    backbone_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    uniform_lr: bool = False,
    num_workers: int = 2,
    save_embeddings: bool = False,
    save_checkpoint: bool = False,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute one complete training & evaluation run and persist all artifacts."""
    pct_tag = int(round(fraction * 100))
    run_name = f"{strategy}_f{pct_tag:03d}_s{seed}"
    run_dir = ensure_dir(Path(output_dir) / backbone / run_name)
    metrics_path = run_dir / "metrics.json"

    if skip_existing and metrics_path.exists():
        import json

        with open(metrics_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if verbose:
            print(
                f"[SKIP] {backbone}/{run_name} already exists "
                f"(Test Acc={cached['test_accuracy']*100:.2f}%, F1={cached['test_macro_f1']*100:.2f}%)"
            )
        return cached

    set_seed(seed)
    device = get_device("auto")
    use_amp = device.type == "cuda"

    default_max_ep, default_pat = DEFAULT_EPOCHS[strategy]
    max_epochs = max_epochs if max_epochs is not None else default_max_ep
    patience = patience if patience is not None else default_pat
    if batch_size is None:
        batch_size = 32 if fraction <= 0.25 else 64

    data_bundle = prepare_dataloaders(
        data_dir=data_dir,
        fraction=fraction,
        seed=seed,
        split_seed=42,
        val_ratio=0.2,
        batch_size=batch_size,
        eval_batch_size=64,
        num_workers=num_workers,
        download=True,
    )
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]

    model, param_groups = build_model(
        backbone=backbone,
        strategy=strategy,
        num_classes=data_bundle["num_classes"],
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        uniform_lr=uniform_lr,
    )
    model = model.to(device)
    trainable_params, total_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history_rows: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = 0
    best_state_dict = None
    best_train_acc_at_best = 0.0
    best_train_loss_at_best = 0.0
    best_val_loss_at_best = 0.0
    no_improve = 0
    cumulative_train_time = 0.0
    time_to_best_val = 0.0

    if verbose:
        print(
            f"\n>>> Running {backbone} | {strategy} | {pct_tag}% data "
            f"({data_bundle['counts']['train_used']} imgs) | seed={seed} | device={device}"
        )

    epoch_iter = tqdm(
        range(1, max_epochs + 1),
        desc=f"{backbone}/{run_name}",
        leave=False,
        disable=not verbose,
    )

    for epoch in epoch_iter:
        # ---- Train step (timed strictly on training pass) ----
        t0 = timed_now(device)
        set_module_train_mode(model, backbone, strategy)

        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = targets.size(0)
            tr_loss_sum += loss.item() * bs
            tr_correct += (logits.argmax(dim=1) == targets).sum().item()
            tr_total += bs

        scheduler.step()
        epoch_train_time = timed_now(device) - t0
        cumulative_train_time += epoch_train_time

        train_loss = tr_loss_sum / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)

        # ---- Validation step ----
        val_res = evaluate_loader(model, val_loader, criterion, device, use_amp=use_amp)
        val_loss = val_res["loss"]
        val_acc = val_res["accuracy"]
        val_f1 = val_res["macro_f1"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "generalization_gap": train_acc - val_acc,
                "epoch_train_time_sec": epoch_train_time,
                "cum_train_time_sec": cumulative_train_time,
                "lr_head": optimizer.param_groups[-1]["lr"],
            }
        )

        epoch_iter.set_postfix(
            tr_acc=f"{train_acc*100:.1f}%",
            val_acc=f"{val_acc*100:.1f}%",
            val_f1=f"{val_f1*100:.1f}%",
        )

        # Early stopping criterion on Val Macro-F1 (tie-break by Val Acc)
        improved = (val_f1 > best_val_f1 + 1e-4) or (
            abs(val_f1 - best_val_f1) <= 1e-4 and val_acc > best_val_acc + 1e-4
        )
        if improved:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_acc_at_best = train_acc
            best_train_loss_at_best = train_loss
            best_val_loss_at_best = val_loss
            time_to_best_val = cumulative_train_time
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best validation checkpoint for final test evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_res = evaluate_loader(model, test_loader, criterion, device, use_amp=use_amp)

    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )

    # Save history CSV
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(run_dir / "history.csv", index=False)

    # Save test predictions for confusion matrix & error analysis
    np.savez_compressed(
        run_dir / "test_predictions.npz",
        y_true=test_res["y_true"],
        y_pred=test_res["y_pred"],
        y_conf=test_res["y_conf"],
    )

    # Optionally save penultimate test embeddings for t-SNE visualization
    if save_embeddings:
        feats, feat_labels = extract_penultimate_features(
            model, backbone, test_loader, device, max_samples=1500
        )
        np.savez_compressed(
            run_dir / "test_embeddings.npz",
            features=feats,
            labels=feat_labels,
        )

    if save_checkpoint and best_state_dict is not None:
        torch.save(best_state_dict, run_dir / "best_model.pt")

    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "backbone": backbone,
        "strategy": strategy,
        "fraction": float(fraction),
        "fraction_pct": pct_tag,
        "seed": int(seed),
        "train_samples": data_bundle["counts"]["train_used"],
        "val_samples": data_bundle["counts"]["val"],
        "test_samples": data_bundle["counts"]["test"],
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": float(trainable_params / total_params),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "epochs_ran": int(len(history_rows)),
        "best_epoch": int(best_epoch),
        "best_train_acc": float(best_train_acc_at_best),
        "best_train_loss": float(best_train_loss_at_best),
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss_at_best),
        "final_train_acc": float(history_rows[-1]["train_acc"]),
        "final_val_acc": float(history_rows[-1]["val_acc"]),
        "gen_gap_at_best": float(best_train_acc_at_best - best_val_acc),
        "gen_gap_at_final": float(
            history_rows[-1]["train_acc"] - history_rows[-1]["val_acc"]
        ),
        "test_loss": float(test_res["loss"]),
        "test_accuracy": float(test_res["accuracy"]),
        "test_macro_f1": float(test_res["macro_f1"]),
        "total_train_time_sec": float(cumulative_train_time),
        "time_to_best_val_sec": float(time_to_best_val),
        "avg_epoch_time_sec": float(cumulative_train_time / max(1, len(history_rows))),
        "peak_gpu_mem_mb": peak_mem_mb,
        "hardware": get_gpu_info(device),
    }

    save_json(metrics, metrics_path)

    if verbose:
        print(
            f"[DONE] {backbone}/{run_name} | "
            f"BestEp={best_epoch}/{len(history_rows)} | "
            f"ValAcc={best_val_acc*100:.2f}% | "
            f"TestAcc={test_res['accuracy']*100:.2f}% | "
            f"TestF1={test_res['macro_f1']*100:.2f}% | "
            f"TrainTime={cumulative_train_time:.1f}s (to best: {time_to_best_val:.1f}s)"
        )

    return metrics
