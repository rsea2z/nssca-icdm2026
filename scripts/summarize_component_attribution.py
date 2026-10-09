from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize component attribution metrics.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for path in sorted(Path(args.root).glob("**/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            result = dict(result)
            result.pop("per_sample", None)
            records.append(
                {
                    "dataset": payload.get("dataset"),
                    "shot": payload.get("shot"),
                    "seed": payload.get("seed"),
                    "alpha": payload.get("alpha"),
                    "max_prototypes": payload.get("max_prototypes"),
                    "num_calibration": payload.get("num_calibration"),
                    "num_test_evaluated": payload.get("num_test_evaluated"),
                    "path": str(path),
                    **result,
                }
            )

    output = {
        "num_records": len(records),
        "groups": build_groups(records),
        "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


def build_groups(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            record["dataset"],
            int(record["shot"]),
            str(record["mode"]),
            float(record["topk_fraction"]),
            float(record["alpha"]),
            int(record["max_prototypes"]),
        )
        grouped[key].append(record)

    groups = []
    for (dataset, shot, mode, topk, alpha, proto), items in sorted(grouped.items()):
        groups.append(
            {
                "dataset": dataset,
                "shot": shot,
                "mode": mode,
                "topk_fraction": topk,
                "alpha": alpha,
                "max_prototypes": proto,
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
    values = finite_values(items, key)
    return float(np.mean(values)) if values else None


def std_of(items: list[dict[str, object]], key: str) -> float | None:
    values = finite_values(items, key)
    if not values:
        return None
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def finite_values(items: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return values


if __name__ == "__main__":
    main()
