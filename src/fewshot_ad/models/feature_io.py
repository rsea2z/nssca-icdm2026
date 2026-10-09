from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureSample:
    path: str
    features: np.ndarray
    base_map: np.ndarray | None = None
    mask: np.ndarray | None = None
    image_path: str | None = None
    class_name: str | None = None


def load_feature_sample(path: str | Path) -> FeatureSample:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data:
            raise KeyError(f"{path} does not contain required key 'features'.")
        features = np.asarray(data["features"], dtype=np.float32)
        base_map = np.asarray(data["base_map"], dtype=np.float32) if "base_map" in data else None
        mask = np.asarray(data["mask"], dtype=np.uint8) if "mask" in data else None
        image_path = _optional_string(data, "image_path")
        class_name = _optional_string(data, "class_name")
    return FeatureSample(
        path=str(path),
        features=features,
        base_map=base_map,
        mask=mask,
        image_path=image_path,
        class_name=class_name,
    )


def _optional_string(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())

