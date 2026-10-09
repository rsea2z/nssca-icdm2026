from __future__ import annotations

import numpy as np

from fewshot_ad.evaluation import image_auroc, pixel_auroc, topk_mean


def test_image_auroc_perfect_ordering() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert image_auroc(labels, scores) == 1.0


def test_pixel_auroc_perfect_ordering() -> None:
    masks = [np.array([[0, 1], [0, 1]], dtype=np.uint8)]
    maps = [np.array([[0.1, 0.8], [0.2, 0.9]], dtype=np.float32)]
    assert pixel_auroc(masks, maps) == 1.0


def test_topk_mean() -> None:
    anomaly_map = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    assert topk_mean(anomaly_map, fraction=0.5) == 2.5

