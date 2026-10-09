from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 200-threshold PRO validation runs.")
    parser.add_argument("--root", default="runs/pro_dense200", help="Dense-PRO result root.")
    parser.add_argument("--output", default="docs/dense_pro_validation.md", help="Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT / args.root
    score_blend = load_score_blend(ROOT / "runs" / "score_blend" / "summary.json")
    lines = [
        "# Dense-PRO Validation",
        "",
        "This audit recomputes PRO with 200 uniformly spaced thresholds for the final top-0.1% rule.",
        "Image AUROC and FPR use the final score-blended image scores when available; PRO@200 is computed from the fused maps and is unaffected by the score blend.",
        "It is a robustness check for the main-paper PRO values, which were generated with 50 thresholds for tractability.",
        "",
        "| Dataset | Setting | Image AUROC | Pixel AUROC | PRO@200 | FPR@95TPR | Runs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    zero_by_dataset: dict[str, dict[str, float]] = {}
    shot_rows: list[dict[str, object]] = []
    for dataset in ("mvtec", "visa_full"):
        dataset_root = root / dataset
        if not dataset_root.exists():
            continue
        zero_path = dataset_root / "zero_topk0.001_pro200" / "metrics.json"
        if zero_path.exists():
            payload = read_json(zero_path)
            zero_fpr = normal_fpr_at_95_tpr(payload["per_sample"])
            zero_by_dataset[dataset] = {
                "pro_auc": float(payload["pro_auc"]),
                "normal_fpr_at_95_tpr": zero_fpr,
            }
            lines.append(
                row(
                    display_dataset(dataset),
                    "zero-shot",
                    payload["image_auroc"],
                    payload["pixel_auroc"],
                    payload["pro_auc"],
                    zero_fpr,
                    1,
                )
            )
        for shot in (1, 2, 4):
            runs = []
            for seed in (0, 1, 2):
                path = dataset_root / f"calib_shot{shot}_seed{seed}_alpha0.25_proto512" / "metrics.json"
                if path.exists():
                    runs.append(read_json(path))
            if not runs:
                continue
            pro_values = [item["pro_auc"] for item in runs]
            score_group = score_blend.get((dataset, shot))
            image_values = [item["image_auroc"] for item in runs]
            pixel_values = [item["pixel_auroc"] for item in runs]
            fpr_values = [normal_fpr_at_95_tpr(item["per_sample"]) for item in runs]
            if score_group is not None:
                image_values = values_from_group(score_group, "image_auroc")
                pixel_values = values_from_group(score_group, "pixel_auroc")
                fpr_values = values_from_group(score_group, "normal_fpr_at_95_tpr")
            shot_rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "pro_auc": pro_values,
                    "normal_fpr_at_95_tpr": fpr_values,
                }
            )
            lines.append(
                row(
                    display_dataset(dataset),
                    f"{shot}-shot",
                    image_values,
                    pixel_values,
                    pro_values,
                    fpr_values,
                    len(runs),
                )
            )
    if shot_rows:
        lines.extend(
            [
                "",
                "## Deltas vs Zero-Shot",
                "",
                "Positive PRO deltas are better; negative FPR deltas are better.",
                "",
                "| Dataset | Setting | Delta PRO@200 | Delta FPR@95TPR |",
                "|---|---|---:|---:|",
            ]
        )
        for item in shot_rows:
            zero = zero_by_dataset[item["dataset"]]
            pro_delta = [float(value) - zero["pro_auc"] for value in item["pro_auc"]]
            fpr_delta = [float(value) - zero["normal_fpr_at_95_tpr"] for value in item["normal_fpr_at_95_tpr"]]
            lines.append(
                f"| {display_dataset(str(item['dataset']))} | {item['shot']}-shot | {fmt(pro_delta)} | {fmt(fpr_delta)} |"
            )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def row(
    dataset: str,
    setting: str,
    image: float | list[float],
    pixel: float | list[float],
    pro: float | list[float],
    fpr: float | list[float] | None,
    runs: int,
) -> str:
    return (
        f"| {dataset} | {setting} | {fmt(image)} | {fmt(pixel)} | "
        f"{fmt(pro)} | {fmt(fpr) if fpr is not None else 'n/a'} | {runs} |"
    )


def fmt(value: float | list[float]) -> str:
    if isinstance(value, list):
        if len(value) == 1:
            return f"{100 * value[0]:.2f}"
        return f"{100 * mean(value):.2f} +/- {100 * stdev(value):.2f}"
    return f"{100 * value:.2f}"


def display_dataset(name: str) -> str:
    return {"mvtec": "MVTec", "visa_full": "VisA"}.get(name, name)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_score_blend(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    payload = read_json(path)
    groups = {}
    for group in payload.get("selected_groups", []):
        groups[(str(group["dataset"]), int(group["shot"]))] = group
    return groups


def values_from_group(group: dict, key: str) -> list[float]:
    mean_value = float(group[f"{key}_mean"])
    std_value = float(group.get(f"{key}_std", 0.0))
    num_runs = int(group.get("num_runs", 1))
    if num_runs <= 1:
        return [mean_value]
    if std_value == 0.0:
        return [mean_value] * num_runs
    # Construct a synthetic sample with the exact sample mean and stdev so the
    # formatter prints the stored score-blend aggregate without needing the raw
    # per-seed records in this PRO-focused audit.
    if num_runs == 3:
        return [mean_value - std_value, mean_value, mean_value + std_value]
    return [mean_value] * num_runs


def normal_fpr_at_95_tpr(per_sample: list[dict[str, object]]) -> float:
    scores = np.asarray([float(item["score"]) for item in per_sample], dtype=float)
    labels = np.asarray([1 if item.get("label") == "anomaly" else 0 for item in per_sample], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return float("nan")
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
