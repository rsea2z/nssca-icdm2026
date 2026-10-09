from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ZERO_RE = re.compile(r"zero_shot_seed_(\d+)")
CALIB_RE = re.compile(r"calib_shot(\d+)_seed(\d+)_alpha([0-9.]+)_proto(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MVTec zero-shot and calibration metrics.")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--output", default="runs/summary.json")
    parser.add_argument(
        "--include-per-sample",
        action="store_true",
        help="Keep per-image scores in the summary output. Disabled by default to keep summaries small.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    records = []
    for path in sorted(root.glob("**/metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = infer_meta(path)
        if meta is None:
            continue
        if "per_sample" in metrics:
            metrics["normal_fpr_at_95_tpr"] = normal_fpr_at_95_tpr(metrics["per_sample"])
        if not args.include_per_sample:
            metrics.pop("per_sample", None)
        records.append({**meta, **metrics, "path": str(path)})

    summary = build_summary(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


def infer_meta(path: Path) -> dict[str, object] | None:
    name = path.parent.name
    match = ZERO_RE.fullmatch(name)
    if match:
        return {
            "kind": "zero_shot",
            "shot": 1,
            "seed": int(match.group(1)),
            "alpha": None,
            "max_prototypes": None,
        }
    match = CALIB_RE.fullmatch(name)
    if match:
        return {
            "kind": "calibration",
            "shot": int(match.group(1)),
            "seed": int(match.group(2)),
            "alpha": float(match.group(3)),
            "max_prototypes": int(match.group(4)),
        }
    if name == "mvtec_zero_shot_shot1_seed0":
        return {
            "kind": "zero_shot_legacy",
            "shot": 1,
            "seed": 0,
            "alpha": None,
            "max_prototypes": None,
        }
    return None


def build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    by_group: dict[tuple[str, int, float | None, int | None], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["kind"]),
            int(record["shot"]),
            None if record["alpha"] is None else float(record["alpha"]),
            None if record["max_prototypes"] is None else int(record["max_prototypes"]),
        )
        by_group[key].append(record)

    groups = []
    for (kind, shot, alpha, max_prototypes), items in sorted(by_group.items()):
        groups.append(
            {
                "kind": kind,
                "shot": shot,
                "alpha": alpha,
                "max_prototypes": max_prototypes,
                "num_runs": len(items),
                "image_auroc_mean": mean_of(items, "image_auroc"),
                "image_auroc_std": std_of(items, "image_auroc"),
                "pixel_auroc_mean": mean_of(items, "pixel_auroc"),
                "pixel_auroc_std": std_of(items, "pixel_auroc"),
                "pro_auc_mean": mean_of(items, "pro_auc"),
                "pro_auc_std": std_of(items, "pro_auc"),
                "normal_fpr_at_95_tpr_mean": mean_of(items, "normal_fpr_at_95_tpr"),
                "normal_fpr_at_95_tpr_std": std_of(items, "normal_fpr_at_95_tpr"),
            }
        )
    return {"num_records": len(records), "groups": groups, "records": records}


def mean_of(items: list[dict[str, object]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def std_of(items: list[dict[str, object]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(values, ddof=1))


def normal_fpr_at_95_tpr(per_sample: list[dict[str, object]]) -> float | None:
    scores = np.asarray([float(item["score"]) for item in per_sample], dtype=float)
    labels = np.asarray([1 if item.get("label") == "anomaly" else 0 for item in per_sample], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return None
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
