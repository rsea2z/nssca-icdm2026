from __future__ import annotations

from collections import deque

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC with tie-aware average ranks."""

    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    rank_sum_pos = float(ranks[positives].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def image_auroc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    return roc_auc(np.asarray(labels), np.asarray(scores))


def pixel_auroc(masks: list[np.ndarray], anomaly_maps: list[np.ndarray]) -> float:
    if len(masks) != len(anomaly_maps):
        raise ValueError("masks and anomaly_maps must have the same length.")
    flat_masks = np.concatenate([np.asarray(mask).astype(np.uint8).reshape(-1) for mask in masks])
    flat_scores = np.concatenate([np.asarray(score, dtype=np.float32).reshape(-1) for score in anomaly_maps])
    return roc_auc(flat_masks, flat_scores)


def pro_auc(
    masks: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> float:
    """Approximate per-region-overlap AUC up to ``max_fpr``.

    This implementation is intentionally dependency-light for early experiments.
    For final reporting, cross-check with the official MVTec evaluation script.
    """

    if len(masks) != len(anomaly_maps):
        raise ValueError("masks and anomaly_maps must have the same length.")
    if not masks:
        return float("nan")

    maps = [np.asarray(score, dtype=np.float32) for score in anomaly_maps]
    masks_bool = [np.asarray(mask).astype(bool) for mask in masks]
    all_scores = np.concatenate([score.reshape(-1) for score in maps])
    finite_scores = all_scores[np.isfinite(all_scores)]
    if finite_scores.size == 0:
        return float("nan")

    thresholds = np.linspace(float(finite_scores.max()), float(finite_scores.min()), num_thresholds)
    points: list[tuple[float, float]] = []
    total_normal_pixels = sum(int((~mask).sum()) for mask in masks_bool)
    if total_normal_pixels == 0:
        return float("nan")

    components_by_image = [_connected_components(mask) for mask in masks_bool]
    if sum(len(components) for components in components_by_image) == 0:
        return float("nan")

    for threshold in thresholds:
        false_positive_pixels = 0
        overlaps = []
        for mask, score, components in zip(masks_bool, maps, components_by_image):
            prediction = score >= threshold
            false_positive_pixels += int(np.logical_and(prediction, ~mask).sum())
            for component in components:
                if component.size == 0:
                    continue
                overlaps.append(float(prediction[component[:, 0], component[:, 1]].mean()))
        fpr = false_positive_pixels / total_normal_pixels
        if fpr <= max_fpr:
            points.append((fpr, float(np.mean(overlaps))))

    if len(points) < 2:
        return float("nan")
    points = sorted(points)
    fpr = np.asarray([p[0] for p in points], dtype=np.float64)
    pro = np.asarray([p[1] for p in points], dtype=np.float64)
    return float(np.trapz(pro, fpr) / max_fpr)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask).astype(bool)
    try:
        from scipy import ndimage
    except ImportError:
        return _connected_components_python(mask)

    labels, count = ndimage.label(mask)
    components: list[np.ndarray] = []
    for index in range(1, count + 1):
        coords = np.argwhere(labels == index)
        if coords.size:
            components.append(coords.astype(np.int64, copy=False))
    return components


def _connected_components_python(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask).astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            coords = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(np.asarray(coords, dtype=np.int64))
    return components
