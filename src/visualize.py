"""Standalone, Torch-free script to reproduce all summary tables and figures (fig1–fig8)
directly from saved experiment artifacts in results/.

Examples:
    # Reproduce all tables and all 8 figures
    python -m src.visualize --output_dir results

    # Reproduce a single figure (or a subset)
    python -m src.visualize --output_dir results --figures fig6
    python -m src.visualize --output_dir results --figures fig1 fig3 fig7
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, silhouette_score

if __package__:
    from . import plot_style
else:
    import plot_style

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
BREED_SHORT = {
    "American Bulldog": "Amer. Bulldog",
    "American Pit Bull Terrier": "Amer. Pit Bull",
    "British Shorthair": "Brit. Shorth.",
    "English Cocker Spaniel": "Eng. Cocker",
    "German Shorthaired": "Ger. Shorth.",
    "Great Pyrenees": "Gr. Pyrenees",
    "Miniature Pinscher": "Mini Pinscher",
    "Staffordshire Bull Terrier": "Staff. Bull T.",
    "Wheaten Terrier": "Wheaten Terr.",
    "Yorkshire Terrier": "Yorkshire T.",
}
BACKBONE_LABELS = {
    "resnet18": "ResNet-18",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}

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

ALL_FIGURE_KEYS = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8")


def _ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_cat_breed(class_name: str) -> bool:
    return class_name.replace("_", " ").strip().lower() in CAT_BREEDS_NORMALIZED


def collect_all_metrics(output_dir: str | Path) -> pd.DataFrame:
    """Scan output_dir/*/*/metrics.json and compile into a DataFrame."""
    output_dir = Path(output_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/*/metrics.json")):
        m = _load_json(path)
        hw = m.pop("hardware", {})
        m["device_type"] = hw.get("device_type", "unknown")
        m["gpu_name"] = hw.get("gpu_name", "unknown")
        rows.append(m)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_summary_tables(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Reproduce Table 2: mean ± std per (backbone, strategy, fraction_pct) as CSV + Markdown."""
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


def build_species_f1_table(output_dir: Path, tables_dir: Path) -> pd.DataFrame:
    """Reproduce Table 3: 3-seed Cat (12 breeds) vs. Dog (25 breeds) mean class F1 on ResNet-18."""
    eda_path = output_dir / "eda_summary.json"
    if not eda_path.exists():
        return pd.DataFrame()
    eda = _load_json(eda_path)
    n_classes = eda["num_classes"]
    cat_ids = [r["class_id"] for r in eda["per_class"] if r["species"] == "Cat"]
    dog_ids = [r["class_id"] for r in eda["per_class"] if r["species"] == "Dog"]

    rows = []
    for pct in (10, 100):
        for strat in STRATEGY_ORDER:
            cat_vals, dog_vals = [], []
            for pred_path in sorted((output_dir / "resnet18").glob(f"{strat}_f{pct:03d}_s*/test_predictions.npz")):
                d = np.load(pred_path)
                per_cls_f1 = f1_score(
                    d["y_true"], d["y_pred"], labels=list(range(n_classes)), average=None, zero_division=0
                )
                cat_vals.append(float(np.mean(per_cls_f1[cat_ids]) * 100))
                dog_vals.append(float(np.mean(per_cls_f1[dog_ids]) * 100))
            if not cat_vals:
                continue
            rows.append(
                {
                    "fraction_pct": pct,
                    "strategy": strat,
                    "n_seeds": len(cat_vals),
                    "cat_f1_mean": float(np.mean(cat_vals)),
                    "cat_f1_std": float(np.std(cat_vals, ddof=1)) if len(cat_vals) > 1 else np.nan,
                    "dog_f1_mean": float(np.mean(dog_vals)),
                    "dog_f1_std": float(np.std(dog_vals, ddof=1)) if len(dog_vals) > 1 else np.nan,
                    "dog_minus_cat_pp": float(np.mean(dog_vals) - np.mean(cat_vals)),
                }
            )

    sp_df = pd.DataFrame(rows)
    if sp_df.empty:
        return sp_df
    sp_df.to_csv(tables_dir / "species_f1_table.csv", index=False)

    md_lines = [
        "# Species-Level Mean Class F1 (ResNet-18, Cat vs. Dog)",
        "",
        "| Data % | Strategy | Seeds | Cat F1 (12 cls) | Dog F1 (25 cls) | Dog − Cat |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    for _, r in sp_df.iterrows():
        c_str = f"{r['cat_f1_mean']:.2f} ± {r['cat_f1_std']:.2f}%" if not pd.isna(r["cat_f1_std"]) else f"{r['cat_f1_mean']:.2f}%"
        d_str = f"{r['dog_f1_mean']:.2f} ± {r['dog_f1_std']:.2f}%" if not pd.isna(r["dog_f1_std"]) else f"{r['dog_f1_mean']:.2f}%"
        md_lines.append(
            f"| {int(r['fraction_pct'])}% | {STRATEGY_SHORT[r['strategy']]} | {int(r['n_seeds'])} | "
            f"{c_str} | {d_str} | {r['dog_minus_cat_pp']:+.2f} pp |"
        )
    (tables_dir / "species_f1_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return sp_df


@plot_style.paper_style(two_columns=True)
def plot_fig1_eda(eda_path: Path, fig_dir: Path) -> None:
    """Fig. 1 (Req 1): Oxford-IIIT Pet per-breed class distribution & split summary."""
    if not eda_path.exists():
        print("[WARN] eda_summary.json not found; skipping Fig 1.")
        return
    eda = _load_json(eda_path)
    pc = pd.DataFrame(eda["per_class"]).sort_values(["species", "train_100"], ascending=[True, False])

    fig, ax1 = plt.subplots(
        figsize=plot_style.print_size(2.10, width_fraction=0.72, two_columns=True),
        layout="constrained",
    )

    x = np.arange(len(pc))
    species = pc["species"].to_numpy()
    light_colors = {"Cat": "#FFF1AA", "Dog": "#CBDAFA"}
    for name, background in zip(("Cat", "Dog"), plot_style.GROUP_BACKGROUNDS):
        positions = x[species == name]
        if positions.size:
            left = -0.9 if name == "Cat" else positions[0] - 0.5
            right = len(pc) - 0.1 if name == "Dog" else positions[-1] + 0.5
            ax1.axvspan(left, right, color=background, lw=0)
            ax1.text(np.mean(positions), 91, f"{name} ({len(positions)})",
                     color="#465771", fontweight="bold", ha="center", va="center")
    ax1.bar(x, pc["train_100"], color=[plot_style.SPECIES_COLORS[name] for name in species],
            edgecolor="black", linewidth=0.45, width=0.84, zorder=3)
    ax1.bar(x, pc["train_10"], color=[light_colors[name] for name in species],
            edgecolor="black", linewidth=0.45, width=0.84, zorder=4)
    ax1.set_xticks(x, [BREED_SHORT.get(name, name) for name in pc["class_name"]],
                   rotation=55, ha="right", rotation_mode="anchor", fontsize=4.6)
    ax1.set_ylabel("Training images", labelpad=2)
    ax1.set(xlim=(-0.9, len(pc) - 0.1), ylim=(0, 100), yticks=[0, 20, 40, 60, 80])
    plot_style.style_axes(ax1, grid="y")
    ax1.tick_params(axis="x", pad=1.5)
    ax1.tick_params(axis="y", pad=2.5)

    fig.legend(
        handles=[plot_style.legend_swatch(color, label) for color, label in (
            (plot_style.SPECIES_COLORS["Cat"], "Cat (Full)"),
            (light_colors["Cat"], "Cat (Sub)"),
            (plot_style.SPECIES_COLORS["Dog"], "Dog (Full)"),
            (light_colors["Dog"], "Dog (Sub)"),
        )],
        loc="outside upper center", ncol=4, fontsize=5.5,
        handlelength=0.9, columnspacing=0.7, borderpad=0.22,
    )

    plot_style.save_figure(fig, fig_dir / "fig1_eda_class_distribution")
    plt.close(fig)


@plot_style.paper_style()
def plot_fig2_model_comparison_10pct(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Fig. 2 (Req 2): 10% subset strategy comparison on ResNet-18 (3-seed mean ± std)."""
    sub = summary[(summary["backbone"] == "resnet18") & (summary["fraction_pct"] == 10)].copy()
    if sub.empty:
        return
    present = set(sub["strategy"])
    strategies = [strategy for strategy in STRATEGY_ORDER if strategy in present]
    sub = sub.set_index("strategy").loc[strategies]

    fig, ax = plt.subplots(
        figsize=plot_style.print_size(1.64, width_fraction=0.86), layout="constrained",
    )
    x = np.arange(len(strategies))
    width = 0.36
    f1_hatch = "////"
    backgrounds = dict(zip(STRATEGY_ORDER, plot_style.GROUP_BACKGROUNDS))
    plot_style.group_bands(ax, x, [backgrounds[strategy] for strategy in strategies])
    plot_style.style_axes(ax, grid="y")
    ax.tick_params(axis="both", pad=2.5)
    for metric, offset, palette, hatch in (
        ("test_acc", -width / 2, plot_style.STRATEGY_COLORS, None),
        ("test_f1", width / 2, plot_style.STRATEGY_LIGHT_COLORS, f1_hatch),
    ):
        means = sub[f"{metric}_mean"].to_numpy() * 100
        std = sub[f"{metric}_std"].to_numpy() * 100
        positions = x + offset
        ax.bar(
            positions, means, width, color=[palette[strategy] for strategy in strategies],
            edgecolor="black", linewidth=plot_style.MARK_EDGE_WIDTH, hatch=hatch, zorder=3,
        )
        plot_style.error_bars(ax, positions, means, std)
        for position, mean, deviation in zip(positions, means, np.nan_to_num(std)):
            ax.text(position, mean + deviation + 1.8, f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=4.8)

    ax.set_xticks(x, [STRATEGY_SHORT[strategy] for strategy in strategies], fontsize=6)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylabel("Score (%)", labelpad=2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(
        handles=[
            plot_style.legend_swatch(plot_style.STRATEGY_COLORS["scratch"], "Accuracy"),
            plot_style.legend_swatch(plot_style.STRATEGY_LIGHT_COLORS["scratch"], "Macro-F1",
                                    hatch=f1_hatch),
        ],
        loc="upper left", ncol=1, fontsize=5.3, handlelength=0.85,
        borderpad=0.22, labelspacing=0.22,
    )
    plot_style.save_figure(fig, fig_dir / "fig2_model_comparison_10pct")
    plt.close(fig)


@plot_style.paper_style(two_columns=True)
def plot_fig3_data_efficiency(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Fig. 3 (Req 3 + Extension): Data Efficiency Curve (5%–100%) & 10% vs 100% bars."""
    sub = summary[summary["backbone"] == "resnet18"].copy()
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=plot_style.print_size(2.08, width_fraction=0.78, two_columns=True),
        layout="constrained",
    )
    fractions = sorted(sub["fraction_pct"].unique())
    positions = {fraction: index for index, fraction in enumerate(fractions)}

    strat_markers = {"scratch": "s", "frozen": "o", "partial_ft": "^", "full_ft": "D"}
    strat_styles = {"scratch": "-", "frozen": "-", "partial_ft": "--", "full_ft": ":"}
    strat_sizes = {"scratch": 3.0, "frozen": 3.6, "partial_ft": 3.4, "full_ft": 2.9}
    for strat in STRATEGY_ORDER:
        s_df = sub[sub["strategy"] == strat].sort_values("fraction_pct")
        if s_df.empty:
            continue
        x = s_df["fraction_pct"].map(positions).to_numpy()
        y = s_df["test_f1_mean"].to_numpy() * 100
        repeated = s_df["n_seeds"].to_numpy() > 1
        color = plot_style.STRATEGY_COLORS[strat]
        marker = strat_markers[strat]
        linestyle = strat_styles[strat]
        ms = strat_sizes[strat]
        ax1.plot(x, y, color=color, linestyle=linestyle, label=STRATEGY_SHORT[strat])
        plot_style.error_bars(ax1, x[repeated], y[repeated],
                              s_df["test_f1_std"].to_numpy()[repeated] * 100)
        for mask, facecolor in ((repeated, color), (~repeated, "white")):
            ax1.plot(x[mask], y[mask], linestyle="none", marker=marker, markersize=ms,
                     markerfacecolor=facecolor, markeredgecolor=color, markeredgewidth=0.9,
                     zorder=5)

    ax1.set_xlabel("Labeled training data (%)", labelpad=1.5)
    ax1.set_ylabel("Test Macro-F1 (%)", labelpad=1.5)
    ax1.set_xticks(range(len(fractions)), fractions)
    ax1.set(xlim=(-0.22, len(fractions) - 0.78), ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    plot_style.style_axes(ax1)
    ax1.tick_params(axis="both", pad=2.5)
    seed_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=3.2,
               markerfacecolor=facecolor, markeredgecolor="black", label=label)
        for facecolor, label in (("white", "1 seed"), ("black", "3 seeds"))
    ]
    strat_line_handles = [
        Line2D([], [], color=plot_style.STRATEGY_COLORS[s], linestyle=strat_styles[s],
               marker=strat_markers[s], markersize=3.0, label=STRATEGY_SHORT[s])
        for s in STRATEGY_ORDER if not sub[sub["strategy"] == s].empty
    ]
    ax1.legend(handles=[*strat_line_handles, *seed_handles],
               loc="center left", bbox_to_anchor=(0.02, 0.48), ncol=2,
               fontsize=5.2, columnspacing=0.6, handletextpad=0.35)

    comp = sub[sub["fraction_pct"].isin([10, 100])].copy()
    strategies = [s for s in STRATEGY_ORDER if s in set(comp["strategy"])]
    x = np.arange(len(strategies))
    width = 0.38
    gap = 0.0
    backgrounds = dict(zip(STRATEGY_ORDER, plot_style.GROUP_BACKGROUNDS))
    plot_style.group_bands(ax2, x, [backgrounds[strategy] for strategy in strategies])
    handles = []
    for fraction, offset, bar_color in (
        (10, -(width + gap) / 2, "#87A1EB"),
        (100, (width + gap) / 2, "#1843BA"),
    ):
        rows = comp[comp["fraction_pct"] == fraction].set_index("strategy").reindex(strategies)
        values = rows["test_acc_mean"].to_numpy() * 100
        std = rows["test_acc_std"].to_numpy() * 100
        ax2.bar(x + offset, values, width, color=bar_color,
                edgecolor="black", zorder=3)
        plot_style.error_bars(ax2, x + offset, values, std)
        for position, value, deviation in zip(x + offset, values, np.nan_to_num(std)):
            if np.isfinite(value):
                ax2.annotate(f"{value:.1f}", (position, value + deviation), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom", fontsize=5.0)
        handles.append(plot_style.legend_swatch(bar_color, f"{fraction}% data"))

    ax2.set_xticks(x, [STRATEGY_SHORT[s] for s in strategies], fontsize=6)
    ax2.set_xlabel("Fine-tuning strategy", labelpad=1.5)
    ax2.set_ylabel("Test Accuracy (%)", labelpad=1.5)
    ax2.set(xlim=(-0.5, max(0.5, len(x) - 0.5)), ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    plot_style.style_axes(ax2, grid="y")
    ax2.tick_params(axis="both", pad=2.5)
    ax2.legend(handles=handles, loc="upper left", ncol=1, fontsize=5.3,
               borderpad=0.22, labelspacing=0.22)

    plot_style.save_figure(fig, fig_dir / "fig3_data_efficiency_curve")
    plt.close(fig)


@plot_style.paper_style(two_columns=True)
def plot_fig4_generalization(output_dir: Path, summary: pd.DataFrame, fig_dir: Path) -> None:
    """Fig. 4a & 4b: 2x2 learning curves and Generalization Gap bar chart."""
    strategies = [
        s
        for s in STRATEGY_ORDER
        if (output_dir / "resnet18" / f"{s}_f010_s42" / "history.csv").exists()
    ]
    if not strategies:
        return

    # Fig. 4a: 2x2 four-panel learning curves with single-row 6-item legend, no titles, unified xticks
    fig_a, axes_a = plt.subplots(
        2, 2, figsize=plot_style.print_size(2.15, width_fraction=0.72, two_columns=True),
        sharey=True, layout="constrained",
    )
    for i, (ax, strat) in enumerate(zip(axes_a.flat, strategies)):
        hist = pd.read_csv(output_dir / "resnet18" / f"{strat}_f010_s42" / "history.csv")
        color = plot_style.STRATEGY_COLORS[strat]
        ax.plot(hist["epoch"], hist["train_acc"] * 100, "--", color=color)
        ax.plot(hist["epoch"], hist["val_acc"] * 100, "-", color=color)
        row, col = divmod(i, 2)
        if row == 1:
            ax.set_xlabel("Epoch", labelpad=1.5)
        if strat != "scratch":
            ax.set_xlim(0, 21)
            ax.set_xticks([5, 10, 15, 20])
        ax.set(ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
        plot_style.style_axes(ax)
        ax.tick_params(axis="x", pad=2.5)
        ax.tick_params(axis="y", labelleft=(col == 0), pad=2.5)
    fig_a.supylabel("Accuracy (%)", fontsize=8, fontweight="bold")
    strat_handles = [
        plot_style.legend_swatch(plot_style.STRATEGY_COLORS[s], STRATEGY_SHORT[s])
        for s in strategies
    ]
    line_handles = [
        Line2D([], [], color="black", linestyle=linestyle, label=label)
        for linestyle, label in (("--", "Train"), ("-", "Validation"))
    ]
    fig_a.get_layout_engine().set(h_pad=0.015)
    fig_a.legend(
        handles=[*strat_handles, *line_handles],
        loc="outside upper center", ncol=6, fontsize=5.0,
        handlelength=1.1, columnspacing=0.5, borderpad=0.20, borderaxespad=0.05,
    )
    plot_style.save_figure(fig_a, fig_dir / "fig4a_generalization_curves")
    plt.close(fig_a)

    # Fig. 4b: Generalization Gap bar chart (2 unified bar colors, clear pair gap, tighter group spacing)
    sub10 = summary[(summary["backbone"] == "resnet18") & (summary["fraction_pct"] == 10)].copy()
    sub10 = sub10.set_index("strategy").loc[[s for s in STRATEGY_ORDER if s in set(sub10["strategy"])]].reset_index()

    with plot_style.paper_style(two_columns=False):
        fig_b, ax_gap = plt.subplots(
            figsize=plot_style.print_size(1.30, width_fraction=0.95),
            layout="constrained",
        )
        step = 0.72
        x = np.arange(len(sub10)) * step
        width = 0.25
        gap = 0.0
        cutoff = 30
        cap_h = 2.6
        backgrounds = dict(zip(STRATEGY_ORDER, plot_style.GROUP_BACKGROUNDS))
        for center, strategy in zip(x, sub10["strategy"]):
            ax_gap.axvspan(center - step / 2, center + step / 2,
                           color=backgrounds[strategy], linewidth=0, zorder=0)
        handles = []
        for metric, offset, bar_color, label in (
            ("gen_gap_best_mean", -(width + gap) / 2, "#3F6BE2", "Best val. epoch"),
            ("gen_gap_final_mean", (width + gap) / 2, "#cbd6f5", "Final epoch"),
        ):
            values = sub10[metric].to_numpy() * 100
            heights = [cutoff + cap_h if (np.isfinite(v) and v > cutoff) else v for v in values]
            ax_gap.bar(x + offset, heights, width, color=bar_color, edgecolor="black", zorder=3)
            for position, value in zip(x + offset, values):
                if not np.isfinite(value):
                    continue
                truncated = value > cutoff
                top_y = cutoff + cap_h if truncated else value
                if truncated:
                    ax_gap.bar(position, cap_h, width, bottom=cutoff, color="none",
                               edgecolor="black", linewidth=0, hatch="////////", zorder=4)
                ax_gap.annotate(f"{value:+.1f}", (position, top_y),
                                xytext=(0, 1.4 if value >= 0 else -1.4), textcoords="offset points",
                                ha="center", va="bottom" if value >= 0 else "top", fontsize=3.9,
                                fontweight="bold" if truncated else "normal", zorder=5)
            handles.append(plot_style.legend_swatch(bar_color, label))
        lower = min(0, sub10[["gen_gap_best_mean", "gen_gap_final_mean"]].min().min() * 100 - 4)
        ax_gap.set_xticks(x, [STRATEGY_SHORT[s] for s in sub10["strategy"]], fontsize=6)
        ax_gap.set(xlim=(-step / 2, x[-1] + step / 2), ylim=(lower, 38),
                   yticks=[0, 10, 20, 30])
        ax_gap.set_ylabel("Gap (pp)", labelpad=2)
        plot_style.style_axes(ax_gap, grid="y")
        ax_gap.tick_params(pad=2.5)
        ax_gap.legend(handles=handles, loc="upper right", ncol=2, fontsize=5.2,
                      handlelength=0.85, columnspacing=0.6, borderpad=0.2, labelspacing=0.2)

        plot_style.save_figure(fig_b, fig_dir / "fig4b_generalization_gap")
        plt.close(fig_b)


@plot_style.paper_style(two_columns=True)
def plot_fig5_tsne(output_dir: Path, fig_dir: Path) -> None:
    """Fig. 5 (Req 4): t-SNE of 512-dim penultimate test representations + Silhouette score."""
    candidates = [
        ("scratch_f010_s42", "Scratch (10%)"),
        ("frozen_f010_s42", "Frozen (10%)"),
        ("full_ft_f010_s42", "Full FT (10%)"),
        ("full_ft_f100_s42", "Full FT (100%)"),
    ]
    available = [
        (run, title, output_dir / "resnet18" / run / "test_embeddings.npz")
        for run, title in candidates
        if (output_dir / "resnet18" / run / "test_embeddings.npz").exists()
    ]
    if not available:
        print("[INFO] No test_embeddings.npz found; skipping Fig 5 (t-SNE).")
        return

    eda_path = output_dir / "eda_summary.json"
    if eda_path.exists():
        breeds = sorted(_load_json(eda_path)["per_class"], key=lambda row: row["class_id"])[:10]
        class_names = {row["class_id"]: row["class_name"] for row in breeds}
    else:
        class_names = dict(enumerate((
            "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
            "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
        )))
    colors = ("#1843BA", "#be113c", "#FF9A35", "#64748D", "#21ef58")
    classes = []
    for is_cat, marker in ((True, "o"), (False, "^")):
        group = [(key, name) for key, name in class_names.items() if _is_cat_breed(name) == is_cat]
        classes.extend((key, name, marker, color)
                       for (key, name), color in zip(group, colors, strict=True))

    fig, axes = plt.subplots(
        1, len(available), figsize=plot_style.print_size(1.62, width_fraction=0.96, two_columns=True),
        squeeze=False, layout="constrained",
    )
    for ax, (_, title, emb_path) in zip(axes.flat, available):
        data = np.load(emb_path)
        feats = data["features"]
        labels = data["labels"]

        mask = np.isin(labels, list(class_names))
        feats_sub = feats[mask]
        labels_sub = labels[mask]

        perp = min(30, max(5, len(feats_sub) // 4))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
        emb_2d = tsne.fit_transform(feats_sub)
        sil = silhouette_score(feats_sub, labels_sub)

        for class_id, _, marker, color in classes:
            selected = labels_sub == class_id
            ax.scatter(emb_2d[selected, 0], emb_2d[selected, 1], s=4.5, marker=marker,
                       alpha=0.75, color=color, linewidths=0, rasterized=True)
        ymin, ymax = float(emb_2d[:, 1].min()), float(emb_2d[:, 1].max())
        yrange = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - 0.06 * yrange, ymax + 0.25 * yrange)
        ax.set_title(title)
        ax.text(0.97, 0.96, f"Sil. {sil:.3f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=5.4, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=plot_style.LEGEND_EDGE, lw=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        plot_style.style_axes(ax, grid=None)

    tsne_short = {
        "Abyssinian": "Abyss.",
        "American Bulldog": "Am. Bulldog",
        "American Pit Bull Terrier": "Am. Pit Bull",
        "Basset Hound": "Basset H.",
        "Beagle": "Beagle",
        "Bengal": "Bengal",
        "Birman": "Birman",
        "Bombay": "Bombay",
        "Boxer": "Boxer",
        "British Shorthair": "Brit. Sh.",
    }
    legend_order = [*classes[:5], *classes[5:]]
    fig.legend(
        handles=[Line2D([], [], linestyle="none", marker=marker, color=color, markersize=3.2,
                        markeredgewidth=0, label=tsne_short.get(name, BREED_SHORT.get(name, name)))
                 for _, name, marker, color in legend_order],
        loc="outside lower center", ncol=10, fontsize=4.5,
        handletextpad=0.15, columnspacing=0.4, borderpad=0.25,
    )
    plot_style.save_figure(fig, fig_dir / "fig5_tsne_representations")
    plt.close(fig)


@plot_style.paper_style()
def plot_fig6_tradeoff(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Fig. 6 (Req 5): Performance vs. Total Training Time Pareto Frontier."""
    sub = summary[
        (summary["backbone"] == "resnet18") & (summary["fraction_pct"].isin([10, 100]))
    ].dropna(subset=["total_time_mean", "test_f1_mean"])
    sub = sub[sub["total_time_mean"] > 0]
    if sub.empty:
        return

    fig, ax = plt.subplots(
        figsize=plot_style.print_size(1.63, width_fraction=0.88), layout="constrained",
    )
    markers = {10: "o", 100: "D"}
    callouts = {
        ("frozen", 10): ((3, 7), "left", "bottom"),
        ("full_ft", 100): ((9, -7), "left", "center"),
    }
    ax.set_facecolor(plot_style.GROUP_BACKGROUNDS[0])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([30, 100, 300, 1000]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.margins(x=0.10)
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Total training time (s, log)", labelpad=2)
    ax.set_ylabel("Test Macro-F1 (%)", labelpad=2)
    plot_style.style_axes(ax, log_x=True)
    ax.tick_params(axis="both", pad=2.5)

    pts = sorted(
        zip(sub["total_time_mean"], sub["test_f1_mean"] * 100),
        key=lambda point: (point[0], -point[1]),
    )
    pareto, best_score = [], -np.inf
    for time, score in pts:
        if score > best_score:
            pareto.append((time, score))
            best_score = score
    frontier, = ax.plot(
        *zip(*pareto), linestyle="--", color="black", zorder=2, label="Pareto",
    )

    for row in sub.itertuples():
        strategy, fraction = row.strategy, row.fraction_pct
        time, score = row.total_time_mean, row.test_f1_mean * 100
        color = plot_style.STRATEGY_COLORS[strategy]
        frozen_full_data = (strategy, fraction) == ("frozen", 100)
        ax.scatter(
            time, score, marker=markers[fraction], s=58 if frozen_full_data else 22,
            facecolors="none" if frozen_full_data else color,
            edgecolors=color if frozen_full_data else "black",
            linewidths=0.85 if frozen_full_data else plot_style.MARK_EDGE_WIDTH,
            zorder=3 if frozen_full_data else 4,
        )
        if (strategy, fraction) in callouts:
            offset, ha, va = callouts[strategy, fraction]
            plot_style.badge(ax, f"{score:.1f}%", (time, score), offset,
                             color=color, ha=ha, va=va)

    present_strategies = set(sub["strategy"])
    strategy_handles = [
        plot_style.legend_swatch(plot_style.STRATEGY_COLORS[strategy], STRATEGY_SHORT[strategy])
        for strategy in STRATEGY_ORDER if strategy in present_strategies
    ]
    present_fractions = set(sub["fraction_pct"])
    data_handles = [
        Line2D(
            [], [], linestyle="none", marker=marker, markersize=4.7,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=plot_style.MARK_EDGE_WIDTH, label=f"{fraction}% data",
        )
        for fraction, marker in markers.items() if fraction in present_fractions
    ]
    unified_legend = ax.legend(
        handles=[*strategy_handles, *data_handles, frontier],
        loc="lower right", ncol=2, fontsize=5.0, handlelength=1.0,
        columnspacing=0.6, borderpad=0.22, labelspacing=0.22,
    )
    for label in unified_legend.get_texts():
        if label.get_text() == STRATEGY_SHORT["frozen"]:
            label.set_color(plot_style.STRATEGY_COLORS["frozen"])
            label.set_fontweight("bold")

    plot_style.save_figure(fig, fig_dir / "fig6_efficiency_tradeoff")
    plt.close(fig)


@plot_style.paper_style(two_columns=True)
def plot_fig7_error_analysis(output_dir: Path, fig_dir: Path) -> None:
    """Fig. 7 (Extension): Top-12 confused breed pairs + Cat vs. Dog Macro-F1 breakdown."""
    preferred = [
        ("full_ft_f100_s42", "Full FT (100%)"),
        ("full_ft_f010_s42", "Full FT (10%)"),
        ("frozen_f100_s42", "Frozen (100%)"),
        ("frozen_f010_s42", "Frozen (10%)"),
    ]
    chosen_label, pred_path = None, None
    for run, label in preferred:
        p = output_dir / "resnet18" / run / "test_predictions.npz"
        if p.exists():
            chosen_label, pred_path = label, p
            break
    if pred_path is None:
        return

    eda_path = output_dir / "eda_summary.json"
    if not eda_path.exists():
        return
    eda = _load_json(eda_path)
    class_names = [row["class_name"] for row in sorted(eda["per_class"], key=lambda r: r["class_id"])]

    preds_npz = np.load(pred_path)
    y_true = preds_npz["y_true"]
    y_pred = preds_npz["y_pred"]

    mistakes = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            mistakes[(class_names[t], class_names[p])] += 1
    top_pairs = mistakes.most_common(12)

    cat_ids = [i for i, n in enumerate(class_names) if _is_cat_breed(n)]
    dog_ids = [i for i, n in enumerate(class_names) if not _is_cat_breed(n)]

    species_rows = []
    for strat in STRATEGY_ORDER:
        p = output_dir / "resnet18" / f"{strat}_f010_s42" / "test_predictions.npz"
        if not p.exists():
            continue
        d = np.load(p)
        per_cls_f1 = f1_score(
            d["y_true"], d["y_pred"], labels=list(range(len(class_names))), average=None, zero_division=0
        )
        species_rows.append(
            {
                "strategy": strat,
                "Cat": float(np.mean(per_cls_f1[cat_ids]) * 100),
                "Dog": float(np.mean(per_cls_f1[dog_ids]) * 100),
            }
        )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=plot_style.print_size(1.72, width_fraction=0.72, two_columns=True),
        gridspec_kw={"width_ratios": [0.8, 1.2]}, layout="constrained",
    )
    plot_style.style_axes(ax1, grid="x")
    ax1.tick_params(axis="y", labelsize=5.5, pad=3)

    if top_pairs:
        pairs = top_pairs[::-1]
        pair_labels = [f"{BREED_SHORT.get(true, true)} › {BREED_SHORT.get(pred, pred)}"
                       for (true, pred), _ in pairs]
        pair_counts = [count for _, count in pairs]
        ax1.barh(pair_labels, pair_counts, color="#b5b7c4",
                 edgecolor="black", height=0.78, zorder=3)
        ax1.set_xlabel("Test errors")
        ax1.set_title(chosen_label)
        ax1.set_xlim(left=0)

    plot_style.style_axes(ax2, grid="y")
    if species_rows:
        sp_df = pd.DataFrame(species_rows)
        x = np.arange(len(sp_df))
        width = 0.38
        backgrounds = dict(zip(STRATEGY_ORDER, plot_style.GROUP_BACKGROUNDS))
        plot_style.group_bands(ax2, x, [backgrounds[strategy] for strategy in sp_df["strategy"]])
        handles = []
        for species, count, offset in (("Cat", len(cat_ids), -width / 2), ("Dog", len(dog_ids), width / 2)):
            color = plot_style.SPECIES_COLORS[species]
            ax2.bar(x + offset, sp_df[species], width, color=color, edgecolor="black", zorder=3)
            handles.append(plot_style.legend_swatch(color, f"{species} ({count})"))
        ax2.set_xticks(x, [STRATEGY_SHORT[s].replace(" ", "\n") for s in sp_df["strategy"]],
                       fontsize=6)
        ax2.set_ylabel("Mean Class F1 (%)")
        ax2.set(xlim=(-0.5, len(x) - 0.5), ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
        ax2.tick_params(pad=2.5)
        ax2.set_title("10% labeled data")
        ax2.legend(handles=handles, loc="upper left", ncol=1, fontsize=4.6,
                   borderpad=0.18, labelspacing=0.18, handlelength=0.85, borderaxespad=0.28)

    plot_style.save_figure(fig, fig_dir / "fig7_error_analysis")
    plt.close(fig)


@plot_style.paper_style()
def plot_fig8_backbone_comparison(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Fig. 8 (Extension): Cross-backbone validation (ResNet-18 vs. MobileNetV3-Small)."""
    sub = summary[summary["fraction_pct"] == 10]
    backbones = [name for name in BACKBONE_LABELS if name in set(sub["backbone"])]
    if len(backbones) < 2:
        return

    strategies = [s for s in STRATEGY_ORDER if s in set(sub["strategy"])]

    fig, ax = plt.subplots(
        figsize=plot_style.print_size(1.24, width_fraction=0.92), layout="constrained",
    )
    x = np.arange(len(strategies))
    width = 0.62 / len(backbones)
    backgrounds = dict(zip(STRATEGY_ORDER, plot_style.GROUP_BACKGROUNDS))
    plot_style.group_bands(ax, x, [backgrounds[strategy] for strategy in strategies])
    plot_style.style_axes(ax, grid="y")

    handles = []
    backbone_colors = ["#3F6BE2", "#cbd6f5", "#FF9A35"]
    for index, backbone in enumerate(backbones):
        rows = sub[sub["backbone"] == backbone].set_index("strategy").reindex(strategies)
        values = rows["test_f1_mean"].to_numpy() * 100
        bar_color = backbone_colors[index]
        hatch = "////" if index == 2 else None
        offset = (index - (len(backbones) - 1) / 2) * width
        ax.bar(x + offset, values, width, color=bar_color,
               edgecolor="black", hatch=hatch, zorder=3)
        for position, value in zip(x + offset, values):
            if not np.isfinite(value):
                continue
            if index == 1 and position < 0.5 and value < 1.0:
                ax.annotate(
                    f"{value:.1f}", xy=(position, value + 0.5),
                    xytext=(position, 14.0), ha="center", va="bottom",
                    fontsize=5.2,
                    arrowprops=dict(arrowstyle="<-", color="black", lw=0.6, shrinkA=0, shrinkB=1),
                )
            else:
                ax.annotate(f"{value:.1f}", (position, value), xytext=(0, 1.5),
                            textcoords="offset points", ha="center", va="bottom", fontsize=5.2)
        short_backbone = BACKBONE_LABELS[backbone].replace("MobileNetV3-Small", "MobileNetV3-S")
        handles.append(plot_style.legend_swatch(bar_color, short_backbone,
                                               hatch=hatch))

    ax.set_xticks(x, [STRATEGY_SHORT[s] for s in strategies], fontsize=6)
    ax.set_ylabel("Test Macro-F1 (%)", labelpad=2)
    ax.set(xlim=(-0.5, len(x) - 0.5), ylim=(0, 100), yticks=[0, 25, 50, 75, 100])
    ax.tick_params(pad=2.5)
    ax.legend(handles=handles, loc="upper left", ncol=1, fontsize=4.1,
              borderpad=0.26, labelspacing=0.14, handlelength=0.7,
              handletextpad=0.35, borderaxespad=0.40)

    plot_style.save_figure(fig, fig_dir / "fig8_backbone_comparison")
    plt.close(fig)


def generate_all_outputs(
    output_dir: str | Path = "results",
    data_dir: str | Path = "data",  # kept for backward CLI compatibility
    figures: Sequence[str] = ("all",),
) -> None:
    del data_dir
    output_dir = Path(output_dir)
    tables_dir = _ensure_dir(output_dir / "tables")
    fig_dir = _ensure_dir(output_dir / "figures")

    df = collect_all_metrics(output_dir)
    if df.empty:
        print(f"[WARN] No metrics.json files found under {output_dir}. Run experiments first.")
        return

    df.to_csv(tables_dir / "all_runs.csv", index=False)
    summary = build_summary_tables(df, tables_dir)
    build_species_f1_table(output_dir, tables_dir)
    print(f"Tables written to {tables_dir} (summary_table.md, species_f1_table.md)")

    selected = set(ALL_FIGURE_KEYS) if "all" in figures else set(figures)
    dispatch = {
        "fig1": lambda: plot_fig1_eda(output_dir / "eda_summary.json", fig_dir),
        "fig2": lambda: plot_fig2_model_comparison_10pct(summary, fig_dir),
        "fig3": lambda: plot_fig3_data_efficiency(summary, fig_dir),
        "fig4": lambda: plot_fig4_generalization(output_dir, summary, fig_dir),
        "fig5": lambda: plot_fig5_tsne(output_dir, fig_dir),
        "fig6": lambda: plot_fig6_tradeoff(summary, fig_dir),
        "fig7": lambda: plot_fig7_error_analysis(output_dir, fig_dir),
        "fig8": lambda: plot_fig8_backbone_comparison(summary, fig_dir),
    }
    for key in ALL_FIGURE_KEYS:
        if key in selected:
            dispatch[key]()

    generated = sorted(p.name for p in fig_dir.glob("*.png"))
    print(f"Generated/verified {len(generated)} figures in {fig_dir}: {', '.join(generated)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all summary tables and figures (fig1–fig8) directly from saved results/."
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_FIGURE_KEYS],
        help="Subset of figures to regenerate (default: all).",
    )
    args = parser.parse_args()
    generate_all_outputs(output_dir=args.output_dir, data_dir=args.data_dir, figures=args.figures)


if __name__ == "__main__":
    main()
