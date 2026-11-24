"""Generate all summary tables and publication-grade figures for Requirements 1-5."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score

from src.dataset import is_cat_breed
from src.run_experiments import collect_all_metrics
from src.utils import ensure_dir, load_json

STRATEGY_ORDER = ["scratch", "frozen", "partial_ft", "full_ft"]
STRATEGY_LABELS = {
    "scratch": "From Scratch",
    "frozen": "Frozen Feature Extraction",
    "partial_ft": "Partial Fine-Tuning (L4+FC)",
    "full_ft": "Full Fine-Tuning",
}
STRATEGY_SHORT = {
    "scratch": "Scratch",
    "frozen": "Frozen",
    "partial_ft": "Partial FT",
    "full_ft": "Full FT",
}
STRATEGY_COLORS = {
    "scratch": "#D62728",     # Red
    "frozen": "#1F77B4",      # Blue
    "partial_ft": "#FF7F0E",  # Orange
    "full_ft": "#2CA02C",     # Green
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def build_summary_tables(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Aggregate mean ± std per (backbone, strategy, fraction_pct) and export CSV + Markdown."""
    grouped = (
        df.groupby(["backbone", "fraction_pct", "strategy"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            train_samples=("train_samples", "mean"),
            trainable_params=("trainable_params", "first"),
            test_acc_mean=("test_accuracy", "mean"),
            test_acc_std=("test_accuracy", "std"),
            test_f1_mean=("test_macro_f1", "mean"),
            test_f1_std=("test_macro_f1", "std"),
            val_acc_mean=("best_val_acc", "mean"),
            val_acc_std=("best_val_acc", "std"),
            train_acc_best_mean=("best_train_acc", "mean"),
            gen_gap_best_mean=("gen_gap_at_best", "mean"),
            gen_gap_final_mean=("gen_gap_at_final", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            time_to_best_mean=("time_to_best_val_sec", "mean"),
            total_time_mean=("total_train_time_sec", "mean"),
            total_time_std=("total_train_time_sec", "std"),
            peak_mem_mb=("peak_gpu_mem_mb", "mean"),
        )
    )
    grouped["strategy_rank"] = grouped["strategy"].map(
        {s: i for i, s in enumerate(STRATEGY_ORDER)}
    )
    grouped = grouped.sort_values(
        ["backbone", "fraction_pct", "strategy_rank"]
    ).drop(columns=["strategy_rank"])

    grouped.to_csv(tables_dir / "summary_by_condition.csv", index=False)

    # Pretty Markdown table for direct use in final report
    def _fmt(mean: float, std: float, scale: float = 100.0, suffix: str = "%") -> str:
        if pd.isna(std):
            return f"{mean * scale:.2f}{suffix}"
        return f"{mean * scale:.2f} ± {std * scale:.2f}{suffix}"

    md_lines = [
        "# Experiment Results Summary",
        "",
        "| Backbone | Data % | Strategy | Seeds | Trainable Params | Test Accuracy | Test Macro-F1 | Train-Val Gap (Best) | Best Ep | Time to Best (s) | Total Time (s) |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in grouped.iterrows():
        params_str = (
            f"{r['trainable_params']/1e6:.2f}M"
            if r["trainable_params"] >= 1e5
            else f"{r['trainable_params']/1e3:.1f}K"
        )
        md_lines.append(
            f"| {r['backbone']} | {int(r['fraction_pct'])}% | "
            f"{STRATEGY_SHORT.get(r['strategy'], r['strategy'])} | {int(r['n_seeds'])} | "
            f"{params_str} | "
            f"{_fmt(r['test_acc_mean'], r['test_acc_std'])} | "
            f"{_fmt(r['test_f1_mean'], r['test_f1_std'])} | "
            f"{r['gen_gap_best_mean']*100:+.2f}% | "
            f"{r['best_epoch_mean']:.1f} | "
            f"{r['time_to_best_mean']:.1f}s | "
            f"{_fmt(r['total_time_mean'], r['total_time_std'], scale=1.0, suffix='s')} |"
        )

    (tables_dir / "summary_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return grouped


def plot_fig1_eda(eda_path: Path, fig_dir: Path) -> None:
    """Requirement 1: Oxford-IIIT Pet class distribution & split breakdown."""
    if not eda_path.exists():
        print("[WARN] eda_summary.json not found; skipping Fig 1.")
        return
    eda = load_json(eda_path)
    pc = pd.DataFrame(eda["per_class"]).sort_values(["species", "train_100"], ascending=[True, False])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3.2, 1.1]}
    )

    colors = ["#E69F00" if s == "Cat" else "#56B4E9" for s in pc["species"]]
    x = np.arange(len(pc))
    ax1.bar(x, pc["train_100"], color=colors, alpha=0.88, label="Train 100% (~80/class)")
    ax1.bar(x, pc["train_10"], color="#222222", alpha=0.85, width=0.5, label="Train 10% (~8/class)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pc["class_name"], rotation=75, ha="right", fontsize=7.5)
    ax1.set_ylabel("Number of Training Images")
    ax1.set_title(
        f"Oxford-IIIT Pet Per-Breed Training Distribution "
        f"({eda['num_cat_breeds']} Cat Breeds in Orange, {eda['num_dog_breeds']} Dog Breeds in Blue)"
    )
    ax1.legend(loc="upper right")

    # Right panel: split totals
    splits = eda["splits"]
    split_names = ["Train\n(10%)", "Train\n(100%)", "Val\n(Fixed)", "Test\n(Official)"]
    split_vals = [splits["train_10"], splits["train_100"], splits["val"], splits["test"]]
    bar_cols = ["#222222", "#1F77B4", "#2CA02C", "#9467BD"]
    bars = ax2.bar(split_names, split_vals, color=bar_cols, alpha=0.9)
    for b, v in zip(bars, split_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 70,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_ylabel("Images")
    ax2.set_title("Dataset Split Summary")
    ax2.set_ylim(0, max(split_vals) * 1.15)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_eda_class_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_model_comparison_10pct(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Requirement 2: 10% subset strategy comparison on ResNet-18."""
    sub = summary[(summary["backbone"] == "resnet18") & (summary["fraction_pct"] == 10)].copy()
    if sub.empty:
        return
    strategies = [s for s in STRATEGY_ORDER if s in set(sub["strategy"])]
    sub = sub.set_index("strategy").loc[strategies].reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(sub))
    w = 0.34

    acc_m = sub["test_acc_mean"].values * 100
    acc_s = sub["test_acc_std"].fillna(0).values * 100
    f1_m = sub["test_f1_mean"].values * 100
    f1_s = sub["test_f1_std"].fillna(0).values * 100

    b1 = ax.bar(x - w / 2, acc_m, w, yerr=acc_s, capsize=4, label="Test Accuracy (%)", color="#1F77B4", alpha=0.9)
    b2 = ax.bar(x + w / 2, f1_m, w, yerr=f1_s, capsize=4, label="Test Macro-F1 (%)", color="#FF7F0E", alpha=0.9)

    for bars, vals, errs in ((b1, acc_m, acc_s), (b2, f1_m, f1_s)):
        for bar, v, e in zip(bars, vals, errs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + e + 1.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ResNet-18 Performance on 10% Labeled Data (~8 imgs/class, Mean ± Std)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_model_comparison_10pct.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_data_efficiency(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Requirement 3 + Extension: Data Efficiency Curve across 5%–100% + 10% vs 100% Frozen highlight."""
    sub = summary[summary["backbone"] == "resnet18"].copy()
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Left: Full Data Efficiency Curve (Test Macro-F1 vs Data %)
    for strat in STRATEGY_ORDER:
        s_df = sub[sub["strategy"] == strat].sort_values("fraction_pct")
        if s_df.empty:
            continue
        x = s_df["fraction_pct"].values
        y = s_df["test_f1_mean"].values * 100
        yerr = s_df["test_f1_std"].fillna(0).values * 100
        ax1.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            capsize=3.5,
            color=STRATEGY_COLORS[strat],
            label=STRATEGY_LABELS[strat],
        )

    ax1.set_xlabel("Labeled Training Data (%)")
    ax1.set_ylabel("Test Macro-F1 (%)")
    ax1.set_xticks(sorted(sub["fraction_pct"].unique()))
    ax1.set_ylim(0, 100)
    ax1.set_title("Data Efficiency Curve (ResNet-18)")
    ax1.legend(loc="lower right")

    # Right: Direct comparison of 10% vs 100% across available strategies
    comp = sub[sub["fraction_pct"].isin([10, 100])].copy()
    strategies = [s for s in STRATEGY_ORDER if s in set(comp["strategy"])]
    x = np.arange(len(strategies))
    w = 0.34

    vals_10, err_10, vals_100, err_100 = [], [], [], []
    for s in strategies:
        r10 = comp[(comp["strategy"] == s) & (comp["fraction_pct"] == 10)]
        r100 = comp[(comp["strategy"] == s) & (comp["fraction_pct"] == 100)]
        vals_10.append(r10["test_acc_mean"].values[0] * 100 if len(r10) else np.nan)
        err_10.append(r10["test_acc_std"].fillna(0).values[0] * 100 if len(r10) else 0.0)
        vals_100.append(r100["test_acc_mean"].values[0] * 100 if len(r100) else np.nan)
        err_100.append(r100["test_acc_std"].fillna(0).values[0] * 100 if len(r100) else 0.0)

    b1 = ax2.bar(x - w / 2, vals_10, w, yerr=err_10, capsize=4, label="10% Data", color="#6BAED6")
    b2 = ax2.bar(x + w / 2, vals_100, w, yerr=err_100, capsize=4, label="100% Data", color="#08519C")
    for bars, vals, errs in ((b1, vals_10, err_10), (b2, vals_100, err_100)):
        for bar, v, e in zip(bars, vals, errs):
            if not np.isnan(v):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + e + 1.2,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

    ax2.set_xticks(x)
    ax2.set_xticklabels([STRATEGY_SHORT[s] for s in strategies])
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Effect of 10× More Labeled Data (10% vs. 100%)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_data_efficiency_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_generalization(output_dir: Path, summary: pd.DataFrame, fig_dir: Path) -> None:
    """Requirement 4: Learning curves (Train vs Val) + Generalization Gap analysis."""
    # Pick seed 42 histories on 10% data for all available strategies
    strategies = [
        s
        for s in STRATEGY_ORDER
        if (output_dir / "resnet18" / f"{s}_f010_s42" / "history.csv").exists()
    ]
    if not strategies:
        return

    n = len(strategies)
    fig = plt.figure(figsize=(4.3 * n, 7.6))
    gs = fig.add_gridspec(2, n, height_ratios=[1.2, 1.0], hspace=0.38, wspace=0.25)

    for i, strat in enumerate(strategies):
        hist = pd.read_csv(output_dir / "resnet18" / f"{strat}_f010_s42" / "history.csv")
        ax = fig.add_subplot(gs[0, i])
        ax.plot(hist["epoch"], hist["train_acc"] * 100, "--", lw=2, color=STRATEGY_COLORS[strat], label="Train Acc")
        ax.plot(hist["epoch"], hist["val_acc"] * 100, "-", lw=2.2, color=STRATEGY_COLORS[strat], label="Val Acc")
        ax.fill_between(
            hist["epoch"],
            hist["val_acc"] * 100,
            hist["train_acc"] * 100,
            color=STRATEGY_COLORS[strat],
            alpha=0.12,
        )
        ax.set_title(f"{STRATEGY_SHORT[strat]} (10% Data)")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=8.5)

    # Bottom spanning subplot: Generalization Gap (Train Acc - Val Acc) at Final vs Best Epoch
    ax_gap = fig.add_subplot(gs[1, :])
    sub10 = summary[(summary["backbone"] == "resnet18") & (summary["fraction_pct"] == 10)].copy()
    sub10 = sub10.set_index("strategy").loc[[s for s in STRATEGY_ORDER if s in set(sub10["strategy"])]].reset_index()

    x = np.arange(len(sub10))
    w = 0.34
    gap_best = sub10["gen_gap_best_mean"].values * 100
    gap_final = sub10["gen_gap_final_mean"].values * 100

    b1 = ax_gap.bar(x - w / 2, gap_best, w, label="Gap at Best Val Epoch", color="#4C72B0")
    b2 = ax_gap.bar(x + w / 2, gap_final, w, label="Gap at Final Epoch", color="#DD8452")
    for bars, vals in ((b1, gap_best), (b2, gap_final)):
        for bar, v in zip(bars, vals):
            ax_gap.text(
                bar.get_x() + bar.get_width() / 2,
                v + (1.0 if v >= 0 else -3.0),
                f"{v:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax_gap.axhline(0, color="black", lw=0.8)
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels([STRATEGY_LABELS[s] for s in sub10["strategy"]])
    ax_gap.set_ylabel("Train Acc − Val Acc (%)")
    ax_gap.set_title("Overfitting Quantification: Generalization Gap on 10% Data (Lower = Better Generalization)")
    ax_gap.legend(loc="upper right")

    plt.savefig(fig_dir / "fig4_generalization_curves_and_gap.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig5_tsne(output_dir: Path, fig_dir: Path) -> None:
    """Requirement 4: t-SNE visualization of penultimate representations on test set."""
    candidates = [
        ("scratch_f010_s42", "From Scratch (10%)"),
        ("frozen_f010_s42", "Frozen Pretrained (10%)"),
        ("full_ft_f010_s42", "Full Fine-Tuning (10%)"),
        ("full_ft_f100_s42", "Full Fine-Tuning (100%)"),
    ]
    available = [
        (run, title, output_dir / "resnet18" / run / "test_embeddings.npz")
        for run, title in candidates
        if (output_dir / "resnet18" / run / "test_embeddings.npz").exists()
    ]
    if not available:
        print("[INFO] No test_embeddings.npz found; skipping Fig 5 (t-SNE).")
        return

    # Visualize 10 selected classes clearly so clusters are crisp and readable
    selected_classes = list(range(10))
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4.4))
    if len(available) == 1:
        axes = [axes]

    for ax, (_, title, emb_path) in zip(axes, available):
        data = np.load(emb_path)
        feats = data["features"]
        labels = data["labels"]

        mask = np.isin(labels, selected_classes)
        feats_sub = feats[mask]
        labels_sub = labels[mask]

        perp = min(30, max(5, len(feats_sub) // 4))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
        emb_2d = tsne.fit_transform(feats_sub)
        sil = silhouette_score(feats_sub, labels_sub)

        for idx, c in enumerate(selected_classes):
            cm = labels_sub == c
            ax.scatter(
                emb_2d[cm, 0],
                emb_2d[cm, 1],
                s=14,
                alpha=0.78,
                color=cmap(idx),
            )
        ax.set_title(f"{title}\n(Silhouette = {sil:.3f})", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "t-SNE of ResNet-18 Penultimate Representations (First 10 Pet Breeds on Test Set)",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_tsne_representations.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig6_tradeoff(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Requirement 5: Performance vs. Training Time & Data Cost Pareto Trade-off."""
    sub = summary[
        (summary["backbone"] == "resnet18") & (summary["fraction_pct"].isin([10, 100]))
    ].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    markers = {10: "o", 100: "D"}
    sizes = {10: 130, 100: 210}

    for _, r in sub.iterrows():
        strat = r["strategy"]
        pct = int(r["fraction_pct"])
        x = r["total_time_mean"]
        y = r["test_f1_mean"] * 100
        ax.scatter(
            x,
            y,
            s=sizes[pct],
            marker=markers[pct],
            color=STRATEGY_COLORS[strat],
            edgecolors="black",
            linewidths=0.9,
            alpha=0.9,
            zorder=4,
        )
        ax.annotate(
            f"{STRATEGY_SHORT[strat]} ({pct}%)\n{y:.1f}%, {x:.0f}s",
            (x, y),
            textcoords="offset points",
            xytext=(8, -4 if pct == 10 else 6),
            fontsize=8.5,
            fontweight="bold",
        )

    # Custom legend for color (strategy) and shape (data size)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=STRATEGY_COLORS[s], markersize=9, label=STRATEGY_LABELS[s])
        for s in STRATEGY_ORDER
        if s in set(sub["strategy"])
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="10% Labeled Data"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="100% Labeled Data"),
    ]

    ax.set_xscale("log")
    ax.set_xlabel("Total Training Time (seconds, log scale)")
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Trade-off: Classification Performance vs. Computational Cost & Labeled Data Size")
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, ncol=2)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig6_efficiency_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig7_error_analysis(output_dir: Path, data_dir: Path, fig_dir: Path) -> None:
    """Requirement 5 Extension: Top confused breed pairs + Cat vs. Dog breakdown."""
    # Choose best available resnet18 prediction file (prefer full_ft_f100_s42, else full_ft_f010_s42, else frozen_f010_s42)
    preferred = [
        ("full_ft_f100_s42", "Full FT (100%)"),
        ("full_ft_f010_s42", "Full FT (10%)"),
        ("frozen_f100_s42", "Frozen (100%)"),
        ("frozen_f010_s42", "Frozen (10%)"),
    ]
    chosen_run, chosen_label, pred_path = None, None, None
    for run, label in preferred:
        p = output_dir / "resnet18" / run / "test_predictions.npz"
        if p.exists():
            chosen_run, chosen_label, pred_path = run, label, p
            break
    if pred_path is None:
        return

    eda_path = output_dir / "eda_summary.json"
    if not eda_path.exists():
        return
    eda = load_json(eda_path)
    class_names = [row["class_name"] for row in sorted(eda["per_class"], key=lambda r: r["class_id"])]

    preds_npz = np.load(pred_path)
    y_true = preds_npz["y_true"]
    y_pred = preds_npz["y_pred"]

    # Top confused pairs
    mistakes = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            mistakes[(class_names[t], class_names[p])] += 1
    top_pairs = mistakes.most_common(12)

    # Cat vs Dog per-class Macro-F1 across 10% strategies
    cat_ids = [i for i, n in enumerate(class_names) if is_cat_breed(n)]
    dog_ids = [i for i, n in enumerate(class_names) if not is_cat_breed(n)]

    species_rows = []
    for strat in STRATEGY_ORDER:
        p = output_dir / "resnet18" / f"{strat}_f010_s42" / "test_predictions.npz"
        if not p.exists():
            continue
        d = np.load(p)
        per_cls_f1 = f1_score(d["y_true"], d["y_pred"], labels=list(range(len(class_names))), average=None, zero_division=0)
        species_rows.append(
            {
                "strategy": STRATEGY_SHORT[strat],
                "Cat Breeds (12)": float(np.mean(per_cls_f1[cat_ids]) * 100),
                "Dog Breeds (25)": float(np.mean(per_cls_f1[dog_ids]) * 100),
            }
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1.0]})

    if top_pairs:
        pair_labels = [f"{a} → {b}" for (a, b), _ in top_pairs][::-1]
        pair_counts = [c for _, c in top_pairs][::-1]
        ax1.barh(pair_labels, pair_counts, color="#C44E52", alpha=0.88)
        ax1.set_xlabel("Misclassification Count on Test Set")
        ax1.set_title(f"Top-12 Confused Breed Pairs ({chosen_label})")

    if species_rows:
        sp_df = pd.DataFrame(species_rows)
        x = np.arange(len(sp_df))
        w = 0.35
        ax2.bar(x - w / 2, sp_df["Cat Breeds (12)"], w, label="Cat Breeds (12)", color="#E69F00")
        ax2.bar(x + w / 2, sp_df["Dog Breeds (25)"], w, label="Dog Breeds (25)", color="#56B4E9")
        ax2.set_xticks(x)
        ax2.set_xticklabels(sp_df["strategy"])
        ax2.set_ylabel("Mean Class F1 (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title("Cat vs. Dog Fine-Grained Macro-F1 (10% Data)")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "fig7_error_analysis.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig8_backbone_comparison(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Optional Extension: Cross-backbone validation (ResNet-18 vs MobileNetV3-Small)."""
    backbones = summary["backbone"].unique().tolist()
    if len(backbones) < 2:
        return

    sub = summary[summary["fraction_pct"] == 10].copy()
    strategies = [s for s in STRATEGY_ORDER if s in set(sub["strategy"])]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(strategies))
    w = 0.8 / len(backbones)
    palette = ["#1F77B4", "#2CA02C", "#9467BD"]

    for bi, bb in enumerate(backbones):
        bb_df = sub[sub["backbone"] == bb].set_index("strategy")
        vals = [bb_df.loc[s, "test_f1_mean"] * 100 if s in bb_df.index else np.nan for s in strategies]
        offset = (bi - (len(backbones) - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w * 0.92, label=bb, color=palette[bi % len(palette)], alpha=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies])
    ax.set_ylabel("Test Macro-F1 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Cross-Backbone Validation on 10% Labeled Data")
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "fig8_backbone_comparison.png", bbox_inches="tight")
    plt.close(fig)


def generate_all_outputs(output_dir: str | Path = "results", data_dir: str | Path = "data") -> None:
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    fig_dir = ensure_dir(output_dir / "figures")

    df = collect_all_metrics(output_dir)
    if df.empty:
        print(f"[WARN] No metrics.json files found under {output_dir}. Run experiments first.")
        return

    df.to_csv(tables_dir / "all_runs.csv", index=False)
    summary = build_summary_tables(df, tables_dir)
    print(f"Summary table written to {tables_dir / 'summary_table.md'}")

    plot_fig1_eda(output_dir / "eda_summary.json", fig_dir)
    plot_fig2_model_comparison_10pct(summary, fig_dir)
    plot_fig3_data_efficiency(summary, fig_dir)
    plot_fig4_generalization(output_dir, summary, fig_dir)
    plot_fig5_tsne(output_dir, fig_dir)
    plot_fig6_tradeoff(summary, fig_dir)
    plot_fig7_error_analysis(output_dir, data_dir, fig_dir)
    plot_fig8_backbone_comparison(summary, fig_dir)

    generated = sorted(p.name for p in fig_dir.glob("*.png"))
    print(f"Generated {len(generated)} figures in {fig_dir}: {', '.join(generated)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables and plots from experiment results.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    generate_all_outputs(output_dir=args.output_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
