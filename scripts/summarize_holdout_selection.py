from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize holdout-class hyperparameter selection runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    selections = []
    for path in sorted(Path(args.root).glob("**/selection.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("folds", []):
            selected = row["selected"]
            record = {
                "dataset": payload["dataset"],
                "shot": payload["shot"],
                "seed": payload["seed"],
                "fold": row["fold"],
                "alpha": selected["alpha"],
                "max_prototypes": selected["max_prototypes"],
                "topk_fraction": selected["topk_fraction"],
                "dev_image_auroc": row["dev_metrics"]["image_auroc"],
                "dev_fpr": row["dev_metrics"]["normal_fpr_at_95_tpr"],
                "heldout_image_auroc": row["heldout_metrics"]["image_auroc"],
                "heldout_fpr": row["heldout_metrics"]["normal_fpr_at_95_tpr"],
                "path": str(path),
            }
            records.append(record)
            selections.append((selected["alpha"], selected["max_prototypes"], selected["topk_fraction"]))

    output = {
        "num_records": len(records),
        "selection_counts": [
            {"alpha": key[0], "max_prototypes": key[1], "topk_fraction": key[2], "count": value}
            for key, value in sorted(Counter(selections).items())
        ],
        "groups": build_groups(records),
        "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


def build_groups(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], int(record["shot"]))].append(record)
    groups = []
    for (dataset, shot), items in sorted(grouped.items()):
        groups.append(
            {
                "dataset": dataset,
                "shot": shot,
                "num_folds": len(items),
                "heldout_image_auroc_mean": mean_of(items, "heldout_image_auroc"),
                "heldout_image_auroc_std": std_of(items, "heldout_image_auroc"),
                "heldout_fpr_mean": mean_of(items, "heldout_fpr"),
                "heldout_fpr_std": std_of(items, "heldout_fpr"),
                "dev_image_auroc_mean": mean_of(items, "dev_image_auroc"),
                "dev_fpr_mean": mean_of(items, "dev_fpr"),
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
