from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrototypeBank:
    """A per-class memory bank of normalized normal patch features."""

    class_name: str
    prototypes: np.ndarray

    @classmethod
    def build(
        cls,
        class_name: str,
        feature_tensors: list[np.ndarray],
        max_prototypes: int = 256,
        seed: int = 0,
    ) -> "PrototypeBank":
        if not feature_tensors:
            raise ValueError(f"No features provided for class {class_name!r}.")

        flattened = [_flatten_features(features) for features in feature_tensors]
        features = np.concatenate(flattened, axis=0)
        features = features[np.isfinite(features).all(axis=1)]
        if features.size == 0:
            raise ValueError(f"All features are invalid for class {class_name!r}.")

        if len(features) > max_prototypes:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(features), size=max_prototypes, replace=False)
            features = features[indices]

        return cls(class_name=class_name, prototypes=l2_normalize(features))

    def distance_map(self, features: np.ndarray) -> np.ndarray:
        spatial_shape = features.shape[:-1]
        query = l2_normalize(_flatten_features(features))
        similarities = query @ self.prototypes.T
        nearest = np.max(similarities, axis=1)
        distances = 1.0 - nearest
        return distances.reshape(spatial_shape)


def fuse_anomaly_maps(base_map: np.ndarray | None, calibration_map: np.ndarray, alpha: float) -> np.ndarray:
    """Fuse zero-shot map and prototype-distance map after min-max scaling."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    calibration_map = minmax_scale(calibration_map)
    if base_map is None:
        return calibration_map
    base_map = minmax_scale(base_map)
    if calibration_map.shape != base_map.shape:
        calibration_map = resize_map(calibration_map, base_map.shape)
    return alpha * base_map + (1.0 - alpha) * calibration_map


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)


def minmax_scale(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum = float(np.nanmin(array))
    maximum = float(np.nanmax(array))
    if maximum - minimum < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def resize_map(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D map to ``target_shape`` with bilinear interpolation."""

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("resize_map expects a 2D array.")
    src_h, src_w = array.shape
    dst_h, dst_w = target_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return array
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError("array shapes must be positive.")

    y = np.linspace(0.0, src_h - 1, dst_h, dtype=np.float32)
    x = np.linspace(0.0, src_w - 1, dst_w, dtype=np.float32)

    y0 = np.floor(y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    wy = (y - y0).astype(np.float32)

    top = (1.0 - wy)[:, None] * array[y0, :] + wy[:, None] * array[y1, :]

    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wx = (x - x0).astype(np.float32)

    resized = (1.0 - wx)[None, :] * top[:, x0] + wx[None, :] * top[:, x1]
    return resized.astype(np.float32)


def _flatten_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim < 2:
        raise ValueError("features must have shape (..., dim)")
    return features.reshape(-1, features.shape[-1])
