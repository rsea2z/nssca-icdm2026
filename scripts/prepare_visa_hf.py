from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
import pyarrow.parquet as pq


FILE_RE = re.compile(r"(?P<class_name>[a-z0-9_]+)\.(?P<split>train|test)-00000-of-00001\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Hugging Face VisA mirror to an MVTec-style folder.")
    parser.add_argument("--repo-id", default="BrachioLab/visa")
    parser.add_argument("--output-root", required=True, help="Destination root, e.g. datasets/visa_pytorch")
    parser.add_argument("--download-dir", default=None, help="Optional download/cache folder for parquet files.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--skip-download", action="store_true", help="Reuse an existing downloaded parquet tree.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    download_dir = Path(args.download_dir).resolve() if args.download_dir else output_root.parent / "_hf_visa"
    if not args.skip_download:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=download_dir,
            allow_patterns=["data/*.parquet"],
            max_workers=args.max_workers,
        )

    selected = set(args.classes) if args.classes else None
    parquet_files = sorted(download_dir.glob("data/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found in {download_dir}")

    counts = {"train": 0, "test": 0}
    for parquet_path in parquet_files:
        match = FILE_RE.fullmatch(parquet_path.name)
        if not match:
            continue
        class_name = match.group("class_name")
        split = match.group("split")
        if selected is not None and class_name not in selected:
            continue

        table = pq.read_table(parquet_path)
        records = table.to_pylist()
        for idx, row in enumerate(records):
            image_bytes = row["image"]["bytes"]
            label = int(row["label"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if split == "train":
                if label != 0:
                    continue
                image_path = output_root / class_name / "train" / "good" / f"{idx:06d}.png"
                _save_image(image, image_path)
                counts["train"] += 1
                continue

            if label == 0:
                image_path = output_root / class_name / "test" / "good" / f"{idx:06d}.png"
                _save_image(image, image_path)
            else:
                mask_bytes = row["mask"]["bytes"]
                mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
                image_path = output_root / class_name / "test" / "anomaly" / f"{idx:06d}.png"
                mask_path = output_root / class_name / "ground_truth" / "anomaly" / f"{idx:06d}_mask.png"
                _save_image(image, image_path)
                _save_image(mask, mask_path)
            counts["test"] += 1

    print(f"converted VisA mirror to {output_root}")
    print(f"train normals: {counts['train']}")
    print(f"test images: {counts['test']}")


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


if __name__ == "__main__":
    main()
