from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.evaluation import image_auroc


CALIB_RE = re.compile(r"calib_shot(\d+)_seed(\d+)_alpha([0-9.]+)_proto(\d+)")
TOPK_RE = re.compile(r"topk_sweep_shot(\d+)_seed(\d+)_alpha([0-9.]+)_proto(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-class MVTec metrics for failure analysis.")
    parser.add_argument("--baseline", required=True, help="Zero-shot metrics JSON with top-level per_sample.")
    parser.add_argument(
        "--baseline-topk-fraction",
        type=float,
        default=None,
        help="If the baseline is a top-k sweep JSON, select this top-k fraction from its results.",
    )
    parser.add_argument("--root", required=True, help="Root directory containing calibration metrics.json files.")
    parser.add_argument("--output-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV table path.")
    parser.add_argument(
        "--calibration-topk-fraction",
        type=float,
        default=None,
        help="If calibration runs are top-k sweep JSON files, select this top-k fraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_per_class(Path(args.baseline), topk_fraction=args.baseline_topk_fraction)
    calibration_runs = load_calibration_runs(Path(args.root), topk_fraction=args.calibration_topk_fraction)

    grouped: dict[int, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for run in calibration_runs:
        shot = int(run["shot"])
        for class_name, stats in run["per_class"].items():
            grouped[shot][class_name].append(stats)

    summary = {
        "baseline_path": str(Path(args.baseline)),
        "root": str(Path(args.root)),
        "shots": {},
    }

    rows: list[dict[str, object]] = []
    for shot in sorted(grouped):
        shot_items = grouped[shot]
        shot_rows = []
        for class_name in sorted(baseline):
            base = baseline[class_name]
            items = shot_items.get(class_name, [])
            if not items:
                continue

            image_values = [float(item["image_auroc"]) for item in items]
            fpr_values = [float(item["normal_fpr_at_95_tpr"]) for item in items]
            delta_image = [value - float(base["image_auroc"]) for value in image_values]
            delta_fpr = [value - float(base["normal_fpr_at_95_tpr"]) for value in fpr_values]

            row = {
                "shot": shot,
                "class_name": class_name,
                "num_runs": len(items),
                "baseline_image_auroc": base["image_auroc"],
                "baseline_normal_fpr_at_95_tpr": base["normal_fpr_at_95_tpr"],
                "calibration_image_auroc_mean": float(np.mean(image_values)),
                "calibration_image_auroc_std": float(np.std(image_values, ddof=1)) if len(image_values) > 1 else 0.0,
                "delta_image_auroc_mean": float(np.mean(delta_image)),
                "delta_image_auroc_std": float(np.std(delta_image, ddof=1)) if len(delta_image) > 1 else 0.0,
                "calibration_normal_fpr_at_95_tpr_mean": float(np.mean(fpr_values)),
                "calibration_normal_fpr_at_95_tpr_std": float(np.std(fpr_values, ddof=1)) if len(fpr_values) > 1 else 0.0,
                "delta_normal_fpr_at_95_tpr_mean": float(np.mean(delta_fpr)),
                "delta_normal_fpr_at_95_tpr_std": float(np.std(delta_fpr, ddof=1)) if len(delta_fpr) > 1 else 0.0,
                "num_normals": int(base["num_normals"]),
                "num_anomalies": int(base["num_anomalies"]),
            }
            shot_rows.append(row)
            rows.append(row)

        summary["shots"][str(shot)] = {
            "num_classes": len(shot_rows),
            "mean_delta_image_auroc": float(np.mean([row["delta_image_auroc_mean"] for row in shot_rows])) if shot_rows else None,
            "mean_delta_normal_fpr_at_95_tpr": float(np.mean([row["delta_normal_fpr_at_95_tpr_mean"] for row in shot_rows])) if shot_rows else None,
            "top_image_improvements": top_rows(shot_rows, "delta_image_auroc_mean", reverse=True),
            "top_fpr_reductions": top_rows(shot_rows, "delta_normal_fpr_at_95_tpr_mean", reverse=False),
            "worst_image_regressions": top_rows(shot_rows, "delta_image_auroc_mean", reverse=False),
            "worst_fpr_regressions": top_rows(shot_rows, "delta_normal_fpr_at_95_tpr_mean", reverse=True),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"baseline": baseline, **summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    print(output_json)
    print(Path(args.output_csv))
    for shot in sorted(grouped):
        shot_summary = summary["shots"][str(shot)]
        print(
            f"shot={shot} mean_delta_image_auroc={shot_summary['mean_delta_image_auroc']:.4f} "
            f"mean_delta_fpr={shot_summary['mean_delta_normal_fpr_at_95_tpr']:.4f}"
        )


def load_per_class(path: Path, *, topk_fraction: float | None = None) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_sample = extract_per_sample(payload, path, topk_fraction=topk_fraction)
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in per_sample:
        by_class[str(item["class_name"])].append(item)

    per_class: dict[str, dict[str, object]] = {}
    for class_name, items in by_class.items():
        labels = [1 if is_anomaly_label(item.get("label")) else 0 for item in items]
        scores = [float(item["score"]) for item in items]
        if len(set(labels)) < 2:
            raise ValueError(f"{path} / {class_name} does not contain both normal and anomaly samples.")
        anomaly_scores = [score for score, label in zip(scores, labels) if label == 1]
        normal_scores = [score for score, label in zip(scores, labels) if label == 0]
        threshold = float(np.quantile(anomaly_scores, 0.05))
        per_class[class_name] = {
            "image_auroc": image_auroc(labels, scores),
            "normal_fpr_at_95_tpr": float(np.mean(np.asarray(normal_scores) >= threshold)),
            "num_normals": int(np.sum(np.asarray(labels) == 0)),
            "num_anomalies": int(np.sum(np.asarray(labels) == 1)),
        }
    return per_class


def is_anomaly_label(label: object) -> bool:
    value = str(label).strip().lower()
    return value not in {"good", "normal", "0", "false"}


def extract_per_sample(payload: dict[str, object], path: Path, *, topk_fraction: float | None) -> list[dict[str, object]]:
    per_sample = payload.get("per_sample")
    if not per_sample:
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{path} does not contain per_sample metrics.")
        if topk_fraction is None and len(results) == 1:
            selected = results[0]
        else:
            matches = [
                result
                for result in results
                if result.get("topk_fraction") is not None and abs(float(result["topk_fraction"]) - float(topk_fraction)) < 1e-12
            ]
            if not matches:
                raise ValueError(f"{path} does not contain topk_fraction={topk_fraction}.")
            selected = matches[0]
        per_sample = selected.get("per_sample")
        if not per_sample:
            raise ValueError(f"{path} selected result does not contain per_sample metrics.")

    return list(per_sample)


def load_calibration_runs(root: Path, *, topk_fraction: float | None = None) -> list[dict[str, object]]:
    runs = []
    for path in sorted(root.glob("*shot*_seed*_alpha*_proto*/metrics.json")):
        match = CALIB_RE.fullmatch(path.parent.name) or TOPK_RE.fullmatch(path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "per_sample" not in payload and "results" not in payload:
            continue
        runs.append(
            {
                "path": str(path),
                "shot": int(match.group(1)),
                "seed": int(match.group(2)),
                "alpha": float(match.group(3)),
                "max_prototypes": int(match.group(4)),
                "per_class": load_per_class(path, topk_fraction=topk_fraction),
            }
        )
    if not runs:
        raise ValueError(f"No calibration metrics found under {root}.")
    return runs


def top_rows(rows: list[dict[str, object]], key: str, *, reverse: bool) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    return [
        {
            "shot": row["shot"],
            "class_name": row["class_name"],
            "value": row[key],
        }
        for row in ordered[:5]
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shot",
        "class_name",
        "num_runs",
        "baseline_image_auroc",
        "baseline_normal_fpr_at_95_tpr",
        "calibration_image_auroc_mean",
        "calibration_image_auroc_std",
        "delta_image_auroc_mean",
        "delta_image_auroc_std",
        "calibration_normal_fpr_at_95_tpr_mean",
        "calibration_normal_fpr_at_95_tpr_std",
        "delta_normal_fpr_at_95_tpr_mean",
        "delta_normal_fpr_at_95_tpr_std",
        "num_normals",
        "num_anomalies",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
