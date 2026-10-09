from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.datasets import index_mvtec, index_visa
from fewshot_ad.datasets.common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write AnomalyCLIP-compatible meta.json for MVTec/VisA.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--root", required=True, help="Dataset root where meta.json will be written.")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--split-csv", default=None, help="Optional VisA split CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if args.dataset == "mvtec":
        records = index_mvtec(root, classes=args.classes)
    else:
        records = index_visa(root, classes=args.classes, split_csv=args.split_csv)

    meta: dict[str, dict[str, list[dict[str, object]]]] = {"train": defaultdict(list), "test": defaultdict(list)}
    for record in records:
        if record.split not in {"train", "test"}:
            continue
        image_path = Path(record.image_path).resolve()
        mask_path = Path(record.mask_path).resolve() if record.mask_path else None
        relative_image = image_path.relative_to(root).as_posix()
        relative_mask = mask_path.relative_to(root).as_posix() if mask_path else ""
        meta[record.split][record.class_name].append(
            {
                "img_path": relative_image,
                "mask_path": relative_mask,
                "cls_name": record.class_name,
                "specie_name": _infer_specie(root, image_path),
                "anomaly": 1 if record.is_anomaly else 0,
            }
        )

    serializable = {
        phase: {class_name: items for class_name, items in sorted(class_items.items())}
        for phase, class_items in meta.items()
    }
    output_path = root / "meta.json"
    write_json(output_path, serializable)
    print(f"wrote {output_path}")
    for phase, class_items in serializable.items():
        count = sum(len(items) for items in class_items.values())
        print(f"{phase}: classes={len(class_items)} images={count}")


def _infer_specie(root: Path, image_path: Path) -> str:
    try:
        relative = image_path.relative_to(root)
    except ValueError:
        return image_path.parent.name
    parts = relative.parts
    if len(parts) >= 3:
        return parts[2]
    return image_path.parent.name


if __name__ == "__main__":
    main()

