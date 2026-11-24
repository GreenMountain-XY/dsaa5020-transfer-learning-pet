"""Shared Matplotlib style for figures drawn at ACM sigconf print size."""

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

COLUMN_WIDTH = 3.33
TEXT_WIDTH = 7.0
FONT_DIR = Path(__file__).resolve().parents[1] / "static" / "fonts"
FONT_STEM = "IntoneMonoNerdFontMono"
FALLBACK_FONT = "DejaVu Sans Mono"

STRATEGY_COLORS = {
    "scratch": "#64748D",
    "frozen": "#be113c",
    "partial_ft": "#FF9A35",
    "full_ft": "#3F6BE2",
}
STRATEGY_LIGHT_COLORS = {
    "scratch": "#b5b7c4",
    "frozen": "#fb7284",
    "partial_ft": "#FFBA73",
    "full_ft": "#cbd6f5",
}
SPECIES_COLORS = {"Cat": "#FFDA45", "Dog": "#87A1EB"}
GROUP_BACKGROUNDS = ("#f1f3e8", "#e9eae4", "#e1e2dc", "#d9dad5")
GRID_COLOR = "#bdc5d0"
LEGEND_EDGE = "#c9cbc8"
MARK_EDGE_WIDTH = 0.5


@lru_cache(maxsize=None)
def register_fonts(font_dir=FONT_DIR):
    """Register the bundled faces with Matplotlib and return their family."""
    regular = font_dir / f"{FONT_STEM}-Regular.ttf"
    for path in sorted(font_dir.glob(f"{FONT_STEM}-*.ttf")):
        font_manager.fontManager.addfont(path)
    if regular.is_file():
        return font_manager.FontProperties(fname=regular).get_name()
    return FALLBACK_FONT


def print_size(height, *, width_fraction=1.0, two_columns=False):
    """Return the canvas size in inches for a LaTeX width fraction."""
    width = TEXT_WIDTH if two_columns else COLUMN_WIDTH
    return width * width_fraction, height


@contextmanager
def paper_style(*, two_columns=False):
    """Apply print-sized typography without changing the caller's style."""
    tick_size = 7 if two_columns else 6.5
    label_size = 8 if two_columns else 7.5
    params = {
        "font.family": list(dict.fromkeys((register_fonts(), FALLBACK_FONT))),
        "font.size": tick_size,
        "axes.labelsize": label_size,
        "axes.labelweight": "bold",
        "axes.labelpad": 3,
        "axes.titlesize": label_size,
        "axes.titleweight": "bold",
        "axes.axisbelow": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.75 if two_columns else 0.65,
        "axes.grid": False,
        "axes.facecolor": "white",
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "lines.linewidth": 1.2 if two_columns else 1.0,
        "patch.linewidth": MARK_EDGE_WIDTH,
        "hatch.linewidth": 0.4,
        "legend.fontsize": tick_size,
        "legend.title_fontsize": tick_size,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 1,
        "legend.facecolor": "white",
        "legend.edgecolor": LEGEND_EDGE,
        "legend.handlelength": 1.0,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.45,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.h_pad": 0.035,
        "figure.constrained_layout.w_pad": 0.045,
        "savefig.dpi": 450,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
    }
    with mpl.rc_context(params):
        yield


def style_axes(ax, *, grid="both", log_x=False):
    """Draw four black spines and apply the paper tick and grid styles."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.tick_params(axis="both", which="both", length=0, pad=6, colors="black")
    if log_x:
        ax.tick_params(axis="x", which="major", direction="in", length=2.5, width=0.6)
        ax.tick_params(axis="x", which="minor", direction="in", length=1.3, width=0.5)
    ax.grid(False, which="both")
    if grid is not None:
        ax.grid(True, axis=grid, which="major")


def group_bands(ax, centers, colors=GROUP_BACKGROUNDS):
    """Shade consecutive unit-width categories behind their grid and bars."""
    for center, color in zip(centers, colors, strict=True):
        ax.axvspan(center - 0.5, center + 0.5, color=color, linewidth=0, zorder=0)


def error_bars(ax, x, y, std):
    """Draw standard deviations where the estimate is available."""
    available = np.isfinite(std) & np.isfinite(y)
    if available.any():
        ax.errorbar(
            np.asarray(x)[available], np.asarray(y)[available],
            yerr=np.asarray(std)[available], fmt="none", ecolor="#111111",
            elinewidth=0.65, capthick=0.55, capsize=2.2, zorder=4,
        )


def legend_swatch(color, label, *, hatch=None):
    return Patch(facecolor=color, edgecolor="black", linewidth=MARK_EDGE_WIDTH,
                 hatch=hatch, label=label)


def badge(ax, text, xy, offset, *, color="black", ha="left", va="center"):
    """Annotate a point using a compact badge and a short leader."""
    return ax.annotate(
        text, xy, xytext=offset, textcoords="offset points",
        ha=ha, va=va, fontsize=6, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LEGEND_EDGE, lw=0.5),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=1, shrinkB=3),
        zorder=6,
    )


def save_figure(fig, output_stem, formats=("png", "pdf", "svg")):
    """Export the same full canvas in raster and vector formats."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path)
        print(f"Saved {path}")
