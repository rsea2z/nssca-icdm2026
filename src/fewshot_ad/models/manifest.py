from __future__ import annotations

import json
from pathlib import Path


def load_manifest(path: str | Path) -> dict[str, str]:
    """Load JSONL image-to-feature mapping."""

    path = Path(path)
    mapping: dict[str, str] = {}
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_path = str(item["image_path"])
            feature_path = Path(item["feature_path"])
            if not feature_path.is_absolute():
                feature_path = base_dir / feature_path
            mapping[image_path] = str(feature_path)
            mapping[str(Path(image_path))] = str(feature_path)
    return mapping


def resolve_feature_path(image_path: str, mapping: dict[str, str]) -> str:
    candidates = [image_path, str(Path(image_path))]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    raise KeyError(f"No feature file in manifest for image_path={image_path!r}.")

