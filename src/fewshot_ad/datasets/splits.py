from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .common import ImageRecord, records_to_dicts, write_json


def make_fewshot_split(
    records: Iterable[ImageRecord],
    shots: int,
    seed: int,
    classes: Iterable[str] | None = None,
) -> dict[str, object]:
    """Build a normal-only calibration split and a test split."""

    selected = set(classes) if classes else None
    records = [r for r in records if selected is None or r.class_name in selected]
    train_normals: dict[str, list[ImageRecord]] = defaultdict(list)
    test_records: list[ImageRecord] = []

    for record in records:
        if record.split == "train" and not record.is_anomaly:
            train_normals[record.class_name].append(record)
        elif record.split == "test":
            test_records.append(record)

    rng = random.Random(seed)
    calibration: list[ImageRecord] = []
    per_class_counts: dict[str, int] = {}
    for class_name, candidates in sorted(train_normals.items()):
        candidates = sorted(candidates, key=lambda r: r.image_path)
        rng.shuffle(candidates)
        chosen = candidates[:shots]
        calibration.extend(chosen)
        per_class_counts[class_name] = len(chosen)

    return {
        "meta": {
            "shots": shots,
            "seed": seed,
            "classes": sorted(train_normals),
            "per_class_counts": per_class_counts,
            "num_test": len(test_records),
        },
        "calibration": records_to_dicts(sorted(calibration, key=lambda r: (r.class_name, r.image_path))),
        "test": records_to_dicts(sorted(test_records, key=lambda r: (r.class_name, r.image_path))),
    }


def save_split(split: dict[str, object], output_path: str | Path) -> None:
    write_json(Path(output_path), split)

