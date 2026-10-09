from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TOPK_RE = re.compile(r"topk_sweep_shot(\d+)_seed(\d+)_alpha([0-9.]+)_proto(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize top-k pooling sweep metrics.")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--output", default="runs/topk_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for path in sorted(Path(args.root).glob("topk_sweep_*/metrics.json")):
        meta = infer_meta(path)
        if meta is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            result = dict(result)
            if "per_sample" in result and result.get("normal_fpr_at_95_tpr") is None:
                result["normal_fpr_at_95_tpr"] = normal_fpr_at_95_tpr(result["per_sample"])
            result.pop("per_sample", None)
            records.append(
                {
                    **meta,
                    "score_mode": result.get("score_mode", payload.get("score_mode", "raw")),
                    "topk_fraction": float(result["topk_fraction"]),
                    "num_calibration": payload.get("num_calibration"),
                    "num_test_evaluated": payload.get("num_test_evaluated"),
                    "pro_num_thresholds": payload.get("pro_num_thresholds"),
                    "device": payload.get("device"),
                    **result,
                    "path": str(path),
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"num_records": len(records), "groups": build_groups(records), "records": records}, indent=2),
        encoding="utf-8",
    )
    print(output)


def infer_meta(path: Path) -> dict[str, object] | None:
    match = TOPK_RE.fullmatch(path.parent.name)
    if not match:
        return None
    return {
        "shot": int(match.group(1)),
        "seed": int(match.group(2)),
        "alpha": float(match.group(3)),
        "max_prototypes": int(match.group(4)),
    }


def build_groups(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_group: dict[tuple[int, float, int, float, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["shot"]),
            float(record["alpha"]),
            int(record["max_prototypes"]),
            float(record["topk_fraction"]),
            str(record.get("score_mode", "raw")),
        )
        by_group[key].append(record)

    groups = []
    for (shot, alpha, max_prototypes, topk_fraction, score_mode), items in sorted(by_group.items()):
        groups.append(
            {
                "shot": shot,
                "alpha": alpha,
                "max_prototypes": max_prototypes,
                "topk_fraction": topk_fraction,
                "score_mode": score_mode,
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
    return groups


def mean_of(items: list[dict[str, object]], key: str) -> float | None:
    values = []
    for item in items:
        if item.get(key) is None:
            return None
        values.append(float(item[key]))
    return float(np.mean(values)) if values else None


def std_of(items: list[dict[str, object]], key: str) -> float | None:
    values = []
    for item in items:
        if item.get(key) is None:
            return None
        values.append(float(item[key]))
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
