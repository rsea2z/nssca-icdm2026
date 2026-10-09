from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


SHOT_COLORS = {
    1: "#e76f51",
    2: "#f4a261",
    4: "#e9c46a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot image-AUROC vs PRO tradeoff from summary metrics.")
    parser.add_argument("--summary", default="runs/summary.json")
    parser.add_argument("--output", default="figures/pilot_tradeoff.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 5.2))
    records = summary.get("records", [])
    groups = [item for item in summary.get("groups", []) if item.get("kind") == "calibration"]
    groups.sort(key=lambda item: int(item["shot"]))

    zero = next((item for item in records if item.get("kind") == "zero_shot"), None)
    if zero is not None:
        zero_x = float(zero["image_auroc"])
        zero_y = float(zero["pro_auc"])
        plt.scatter(zero_x, zero_y, s=170, color="#2a6f97", edgecolor="white", linewidth=1.2, zorder=4)
        plt.text(zero_x + 0.0015, zero_y + 0.01, "Zero-shot", fontsize=10, color="#24323a")

    for record in records:
        if record.get("kind") != "calibration":
            continue
        shot = int(record["shot"])
        color = SHOT_COLORS.get(shot, "#f4a261")
        plt.scatter(
            float(record["image_auroc"]),
            float(record["pro_auc"]),
            s=70,
            color=color,
            alpha=0.35,
            edgecolor="none",
            zorder=2,
        )

    mean_xs = []
    mean_ys = []
    for group in groups:
        shot = int(group["shot"])
        color = SHOT_COLORS.get(shot, "#f4a261")
        x = float(group["image_auroc_mean"])
        y = float(group["pro_auc_mean"])
        mean_xs.append(x)
        mean_ys.append(y)
        plt.errorbar(
            x,
            y,
            xerr=float(group.get("image_auroc_std") or 0.0),
            yerr=float(group.get("pro_auc_std") or 0.0),
            fmt="o",
            ms=12,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            mec="white",
            mew=1.2,
            zorder=5,
        )
        plt.text(x + 0.0012, y + 0.012, f"{shot}-shot", fontsize=9, color="#24323a")

    if mean_xs:
        plt.plot(mean_xs, mean_ys, color="#7f8c8d", linewidth=1.1, alpha=0.5, zorder=1)

    xs = [float(item["image_auroc"]) for item in records if item.get("image_auroc") is not None]
    ys = [float(item["pro_auc"]) for item in records if item.get("pro_auc") is not None]
    if xs and ys:
        xpad = max(0.003, (max(xs) - min(xs)) * 0.12)
        ypad = max(0.04, (max(ys) - min(ys)) * 0.12)
        plt.xlim(min(xs) - xpad, max(xs) + xpad)
        plt.ylim(max(0.0, min(ys) - ypad), min(1.0, max(ys) + ypad))

    plt.axvline(0.85, color="#c7d0d5", linewidth=0.8, linestyle="--", zorder=0)
    plt.axhline(0.5, color="#c7d0d5", linewidth=0.8, linestyle="--", zorder=0)
    plt.xlabel("Image AUROC")
    plt.ylabel("PRO")
    plt.title("Calibration tradeoff on MVTec AD")
    plt.grid(True, color="#e8ecef", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()
