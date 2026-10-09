from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from icdm_plot_style import (
    AMBER,
    AXIS,
    BLUE,
    FIG_WIDTH,
    GRID,
    TEAL,
    finish_axes,
    panel_label,
    save_figure,
    set_icdm_style,
)


PALETTE = {
    "image_auroc": TEAL,
    "pro_auc": BLUE,
    "normal_fpr": AMBER,
    "reference": AXIS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot alpha and prototype ablations from MVTec summary metrics.")
    parser.add_argument("--summary", default="runs/summary.json")
    parser.add_argument("--output", default="figures/calibration_ablation.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    set_icdm_style()

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 3.55), constrained_layout=True)
    ax_alpha_top, ax_alpha_bottom = axes[0]
    ax_proto_top, ax_proto_bottom = axes[1]

    groups = [item for item in summary.get("groups", []) if item.get("kind") == "calibration"]
    zero = next((item for item in summary.get("records", []) if item.get("kind") == "zero_shot"), None)

    alpha_groups = sorted(
        [g for g in groups if g["shot"] == 1 and g["max_prototypes"] == 256 and g["alpha"] is not None],
        key=lambda g: float(g["alpha"]),
    )
    proto_groups = sorted(
        [g for g in groups if g["shot"] == 1 and g["alpha"] == 0.25 and g["max_prototypes"] is not None],
        key=lambda g: int(g["max_prototypes"]),
    )

    plot_metric_lines(
        ax_alpha_top,
        alpha_groups,
        x_key="alpha",
        x_label="Fusion weight $\\alpha$",
        x_formatter=lambda x: f"{x:.2f}".rstrip("0").rstrip("."),
        y_limits=(66, 87),
        zero=zero,
        show_reference=False,
        selected=0.25,
    )
    ax_alpha_top.set_ylabel("Score (%)")
    panel_label(ax_alpha_top, "A", "fusion sweep")

    plot_single_metric(
        ax_alpha_bottom,
        alpha_groups,
        x_key="alpha",
        y_key="normal_fpr_at_95_tpr_mean",
        yerr_key="normal_fpr_at_95_tpr_std",
        color=PALETTE["normal_fpr"],
        label="Normal FPR",
        x_formatter=lambda x: f"{x:.2f}".rstrip("0").rstrip("."),
        x_label="Fusion weight $\\alpha$",
        y_label="FPR@95TPR (%)",
        y_limits=(64, 88),
        zero=zero,
        reference_value="normal_fpr_at_95_tpr",
        selected=0.25,
    )
    panel_label(ax_alpha_bottom, "B", "fusion boundary")

    plot_metric_lines(
        ax_proto_top,
        proto_groups,
        x_key="max_prototypes",
        x_label="Prototype budget per class",
        x_formatter=lambda x: f"{int(x):d}",
        y_limits=(72, 87),
        zero=zero,
        show_reference=False,
        log_x=True,
        selected=512,
    )
    ax_proto_top.set_ylabel("Score (%)")
    panel_label(ax_proto_top, "C", "prototype sweep")

    plot_single_metric(
        ax_proto_bottom,
        proto_groups,
        x_key="max_prototypes",
        y_key="normal_fpr_at_95_tpr_mean",
        yerr_key="normal_fpr_at_95_tpr_std",
        color=PALETTE["normal_fpr"],
        label="Normal FPR",
        x_formatter=lambda x: f"{int(x):d}",
        x_label="Prototype budget per class",
        y_label="FPR@95TPR (%)",
        y_limits=(64, 88),
        zero=zero,
        reference_value="normal_fpr_at_95_tpr",
        log_x=True,
        selected=512,
    )
    panel_label(ax_proto_bottom, "D", "prototype boundary")

    for ax in axes.flat:
        finish_axes(ax)

    save_figure(fig, out)
    print(out)
    print(out.with_suffix(".png"))


def plot_metric_lines(
    ax: plt.Axes,
    groups: list[dict[str, object]],
    *,
    x_key: str,
    x_label: str,
    x_formatter,
    y_limits: tuple[float, float],
    zero: dict[str, object] | None,
    show_reference: bool,
    log_x: bool = False,
    selected: float | None = None,
) -> None:
    x = np.asarray([float(g[x_key]) for g in groups], dtype=float)
    image = np.asarray([float(g["image_auroc_mean"]) for g in groups], dtype=float) * 100
    image_std = np.asarray([float(g["image_auroc_std"]) for g in groups], dtype=float) * 100
    pro = np.asarray([float(g["pro_auc_mean"]) for g in groups], dtype=float) * 100
    pro_std = np.asarray([float(g["pro_auc_std"]) for g in groups], dtype=float) * 100

    if selected is not None:
        ax.axvline(selected, color=GRID, linestyle=":", linewidth=0.9, zorder=0)

    ax.errorbar(
        x,
        image,
        yerr=image_std,
        color=PALETTE["image_auroc"],
        marker="o",
        markersize=4.5,
        linewidth=1.4,
        capsize=2.5,
        label="Image AUROC",
        zorder=3,
    )
    ax.errorbar(
        x,
        pro,
        yerr=pro_std,
        color=PALETTE["pro_auc"],
        marker="s",
        markersize=4.5,
        linewidth=1.4,
        capsize=2.5,
        label="PRO",
        zorder=3,
    )

    if zero is not None and show_reference:
        ref = (float(zero["image_auroc"]) if x_key == "alpha" else float(zero["pro_auc"])) * 100
        ax.axhline(ref, color=PALETTE["reference"], linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel(x_label)
    ax.set_ylim(*y_limits)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: x_formatter(value)))
        ax.set_xlim(min(x) / 1.18, max(x) * 1.18)
    else:
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: x_formatter(value)))
        ax.set_xlim(min(x) - (max(x) - min(x)) * 0.05, max(x) + (max(x) - min(x)) * 0.05)

    ax.legend(frameon=False, loc="lower right", handlelength=1.6)


def plot_single_metric(
    ax: plt.Axes,
    groups: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    yerr_key: str,
    color: str,
    label: str,
    x_formatter,
    x_label: str,
    y_label: str,
    y_limits: tuple[float, float],
    zero: dict[str, object] | None,
    reference_value: str | None = None,
    log_x: bool = False,
    selected: float | None = None,
) -> None:
    x = np.asarray([float(g[x_key]) for g in groups], dtype=float)
    y = np.asarray([float(g[y_key]) for g in groups], dtype=float) * 100
    yerr = np.asarray([float(g[yerr_key]) for g in groups], dtype=float) * 100

    if selected is not None:
        ax.axvline(selected, color=GRID, linestyle=":", linewidth=0.9, zorder=0)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=color,
        marker="o",
        markersize=4.5,
        linewidth=1.4,
        capsize=2.5,
        label=label,
        zorder=3,
    )

    if zero is not None and reference_value is not None:
        ax.axhline(
            float(zero[reference_value]) * 100,
            color=PALETTE["reference"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_limits)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: x_formatter(value)))
        ax.set_xlim(min(x) / 1.18, max(x) * 1.18)
    else:
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: x_formatter(value)))
        ax.set_xlim(min(x) - (max(x) - min(x)) * 0.05, max(x) + (max(x) - min(x)) * 0.05)


if __name__ == "__main__":
    main()
