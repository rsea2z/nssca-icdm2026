from __future__ import annotations

import numpy as np


def topk_mean(anomaly_map: np.ndarray, fraction: float = 0.01) -> float:
    """Image-level anomaly score from the top fraction of pixels/tokens."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    values = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(values.size * fraction)))
    partition = np.partition(values, values.size - k)
    return float(partition[-k:].mean())

