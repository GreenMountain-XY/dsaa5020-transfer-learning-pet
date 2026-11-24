"""CLI orchestrator for Core, Extended, and Multi-Backbone transfer learning experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.dataset import compute_eda_summary
from src.train import run_single_experiment
from src.utils import ensure_dir, save_json

DEFAULT_SEEDS = [42, 123, 2026]
EMBEDDING_TARGETS = {
    ("resnet18", "scratch", 0.10, 42),
    ("resnet18", "frozen", 0.10, 42),
    ("resnet18", "partial_ft", 0.10, 42),
    ("resnet18", "full_ft", 0.10, 42),
    ("resnet18", "frozen", 1.00, 42),
    ("resnet18", "full_ft", 1.00, 42),
}


def build_experiment_plan(
    mode: str,
    seeds: List[int],
    curve_seeds: List[int],
) -> List[Tuple[str, str, float, int]]:
    """Return ordered list of (backbone, strategy, fraction, seed) runs."""
    plan: List[Tuple[str, str, float, int]] = []
    seen = set()

    def _add(backbone: str, strategy: str, frac: float, seed: int) -> None:
        key = (backbone, strategy, round(frac, 4), seed)
        if key not in seen:
            seen.add(key)
            plan.append((backbone, strategy, frac, seed))

    if mode == "quick_test":
        for strat in ("scratch", "frozen", "full_ft"):
            _add("resnet18", strat, 0.10, 42)
        _add("resnet18", "frozen", 1.00, 42)
        return plan

    # 1. Core Required Experiments (Requirements 2 & 3)
    for seed in seeds:
        for strat in ("scratch", "frozen", "full_ft"):
            _add("resnet18", strat, 0.10, seed)
        _add("resnet18", "frozen", 1.00, seed)

    if mode == "core":
        return plan

    # 2. Extended 2x4 Factorial Matrix (10% & 100% across all 4 strategies)
    for seed in seeds:
        _add("resnet18", "partial_ft", 0.10, seed)
        for strat in ("scratch", "partial_ft", "full_ft"):
            _add("resnet18", strat, 1.00, seed)

    # 3. Extended Data Efficiency Curve (5%, 25%, 50% intermediate points)
    for seed in curve_seeds:
        for frac in (0.05, 0.25, 0.50):
            for strat in ("scratch", "frozen", "partial_ft", "full_ft"):
                _add("resnet18", strat, frac, seed)

    if mode == "extended":
        return plan

    # 4. Optional Second Lightweight Backbone Validation
    if mode == "full_with_mobilenet":
        for strat in ("scratch", "frozen", "partial_ft", "full_ft"):
            _add("mobilenet_v3_small", strat, 0.10, seeds[0])
            _add("mobilenet_v3_small", strat, 1.00, seeds[0])
        return plan

    raise ValueError(f"Unknown mode: {mode}")


def collect_all_metrics(output_dir: str | Path) -> pd.DataFrame:
    """Scan output_dir for all metrics.json files and compile into a DataFrame."""
    import json

    output_dir = Path(output_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/*/metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        # Flatten hardware info
        hw = m.pop("hardware", {})
        m["device_type"] = hw.get("device_type", "unknown")
        m["gpu_name"] = hw.get("gpu_name", "unknown")
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Transfer Learning vs. Scratch experiments on Oxford-IIIT Pet."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="extended",
        choices=["core", "extended", "full_with_mobilenet", "quick_test"],
        help="Experiment suite to run (default: extended).",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds for primary 10% and 100% experiments.",
    )
    parser.add_argument(
        "--curve_seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for intermediate data-efficiency curve points (5%, 25%, 50%).",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-run experiments even if metrics.json already exists.",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save best_model.pt for every run (uses extra disk space).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    # Step 1: Compute and save EDA summary (Requirement 1)
    print("=== Step 1: Computing Oxford-IIIT Pet EDA Summary ===")
    eda = compute_eda_summary(data_dir=args.data_dir, split_seed=42, val_ratio=0.2)
    save_json(eda, output_dir / "eda_summary.json")
    print(
        f"Dataset: {eda['dataset']} | Total={eda['total_images']} imgs | "
        f"Classes={eda['num_classes']} ({eda['num_cat_breeds']} cats, {eda['num_dog_breeds']} dogs)\n"
        f"Splits: train_100={eda['splits']['train_100']}, "
        f"train_10={eda['splits']['train_10']}, "
        f"val={eda['splits']['val']}, test={eda['splits']['test']}"
    )

    # Step 2: Execute planned experiments
    plan = build_experiment_plan(args.mode, args.seeds, args.curve_seeds)
    print(f"\n=== Step 2: Running {len(plan)} Experiments (mode={args.mode}) ===")

    for idx, (backbone, strategy, frac, seed) in enumerate(plan, start=1):
        print(f"\n--- [{idx}/{len(plan)}] {backbone} | {strategy} | {int(frac*100)}% | seed={seed} ---")
        save_emb = (backbone, strategy, round(frac, 2), seed) in EMBEDDING_TARGETS
        max_ep = 2 if args.mode == "quick_test" else None
        run_single_experiment(
            backbone=backbone,
            strategy=strategy,
            fraction=frac,
            seed=seed,
            data_dir=args.data_dir,
            output_dir=output_dir,
            max_epochs=max_ep,
            num_workers=args.num_workers,
            save_embeddings=save_emb,
            save_checkpoint=args.save_checkpoints,
            skip_existing=not args.no_skip,
            verbose=True,
        )

    # Step 3: Aggregate all completed runs
    df = collect_all_metrics(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    if not df.empty:
        df.to_csv(tables_dir / "all_runs.csv", index=False)
        print(f"\nSaved aggregated metrics ({len(df)} runs) to {tables_dir / 'all_runs.csv'}")


if __name__ == "__main__":
    main()
