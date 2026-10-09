from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """One indexed image and its optional pixel mask."""

    image_path: str
    class_name: str
    split: str
    label: str
    mask_path: str | None = None

    @property
    def is_anomaly(self) -> bool:
        return self.label != "normal"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def sorted_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if is_image(p))


def records_to_dicts(records: Iterable[ImageRecord]) -> list[dict[str, str | None]]:
    return [record.to_dict() for record in records]


def record_from_dict(data: dict[str, str | None]) -> ImageRecord:
    return ImageRecord(
        image_path=str(data["image_path"]),
        class_name=str(data["class_name"]),
        split=str(data["split"]),
        label=str(data["label"]),
        mask_path=str(data["mask_path"]) if data.get("mask_path") else None,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))

