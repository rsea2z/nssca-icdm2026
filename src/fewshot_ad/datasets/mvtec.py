from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .common import ImageRecord, sorted_images


def index_mvtec(root: str | Path, classes: Iterable[str] | None = None) -> list[ImageRecord]:
    """Index an MVTec-style anomaly dataset.

    Expected layout:
        root/class/train/good/*.png
        root/class/test/good/*.png
        root/class/test/defect_type/*.png
        root/class/ground_truth/defect_type/*_mask.png
    """

    root = Path(root)
    selected = set(classes) if classes else None
    records: list[ImageRecord] = []

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if selected is not None and class_name not in selected:
            continue

        train_good = class_dir / "train" / "good"
        for image_path in sorted_images(train_good):
            records.append(
                ImageRecord(
                    image_path=str(image_path),
                    class_name=class_name,
                    split="train",
                    label="normal",
                )
            )

        test_dir = class_dir / "test"
        if not test_dir.exists():
            continue
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            label = "normal" if defect_dir.name == "good" else "anomaly"
            for image_path in sorted_images(defect_dir):
                mask_path = None
                if label == "anomaly":
                    mask_path = _find_mvtec_mask(class_dir, defect_dir.name, image_path)
                records.append(
                    ImageRecord(
                        image_path=str(image_path),
                        class_name=class_name,
                        split="test",
                        label=label,
                        mask_path=str(mask_path) if mask_path else None,
                    )
                )

    return records


def _find_mvtec_mask(class_dir: Path, defect_name: str, image_path: Path) -> Path | None:
    mask_dir = class_dir / "ground_truth" / defect_name
    if not mask_dir.exists():
        return None

    candidates = []
    for suffix in [".png", ".bmp", ".tif", ".tiff"]:
        candidates.append(mask_dir / f"{image_path.stem}_mask{suffix}")
        candidates.append(mask_dir / f"{image_path.stem}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(mask_dir.glob(f"{image_path.stem}*"))
    return matches[0] if matches else None

