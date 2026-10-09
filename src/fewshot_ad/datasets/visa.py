from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .common import ImageRecord
from .mvtec import index_mvtec


def index_visa(
    root: str | Path,
    classes: Iterable[str] | None = None,
    split_csv: str | Path | None = None,
) -> list[ImageRecord]:
    """Index VisA from either a split CSV or an MVTec-style converted folder.

    Anomalib converts VisA to an MVTec-style layout under ``visa_pytorch``.
    That is the preferred path for this project. The CSV reader is intentionally
    permissive because VisA split CSV columns differ across tools.
    """

    root = Path(root)
    if split_csv:
        return _index_visa_csv(root, Path(split_csv), classes=classes)
    return index_mvtec(root, classes=classes)


def _index_visa_csv(
    root: Path,
    split_csv: Path,
    classes: Iterable[str] | None = None,
) -> list[ImageRecord]:
    selected = set(classes) if classes else None
    records: list[ImageRecord] = []

    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_value = _first_present(row, ["image", "image_path", "img_path", "path"])
            if not image_value:
                continue

            class_name = _first_present(row, ["class_name", "class", "category", "object", "cls"])
            image_path = _resolve_path(root, image_value)
            if not class_name:
                class_name = _infer_class_from_path(root, image_path)
            if selected is not None and class_name not in selected:
                continue

            label_value = _first_present(row, ["label", "anomaly", "is_anomaly", "defect"])
            label = _normalize_label(label_value)
            split = _first_present(row, ["split", "phase", "subset"]) or _infer_split(image_path)
            mask_value = _first_present(row, ["mask", "mask_path", "ground_truth"])
            mask_path = str(_resolve_path(root, mask_value)) if mask_value else None

            records.append(
                ImageRecord(
                    image_path=str(image_path),
                    class_name=class_name,
                    split=split,
                    label=label,
                    mask_path=mask_path,
                )
            )

    return records


def _first_present(row: dict[str, str | None], keys: list[str]) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value:
            return value.strip()
    return None


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _infer_class_from_path(root: Path, image_path: Path) -> str:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        return image_path.parent.name
    return relative.parts[0] if relative.parts else image_path.parent.name


def _infer_split(image_path: Path) -> str:
    parts = {part.lower() for part in image_path.parts}
    if "train" in parts:
        return "train"
    if "val" in parts or "validation" in parts:
        return "val"
    return "test"


def _normalize_label(value: str | None) -> str:
    if value is None:
        return "normal"
    normalized = value.strip().lower()
    if normalized in {"0", "normal", "good", "false", "negative"}:
        return "normal"
    return "anomaly"

