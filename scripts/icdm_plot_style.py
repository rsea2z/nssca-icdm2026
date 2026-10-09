from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FIG_WIDTH = 7.1
TEAL = "#127C78"
AMBER = "#D49A2A"
BLUE = "#3F6F93"
RED = "#B84A3A"
GRAPHITE = "#242625"
AXIS = "#6F7775"
GRID = "#D8D3C7"
SPINE = "#BFC5C1"
LIGHT = "#F7F8F6"


def set_icdm_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def finish_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, alpha=0.7, linewidth=0.65)
    ax.tick_params(colors=AXIS, length=2.5, width=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.7)


def panel_label(ax: plt.Axes, label: str, text: str | None = None) -> None:
    content = label if text is None else f"{label}. {text}"
    ax.text(
        0.0,
        1.03,
        content,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=GRAPHITE,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix.lower() != ".png":
        fig.savefig(path.with_suffix(".png"))
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
