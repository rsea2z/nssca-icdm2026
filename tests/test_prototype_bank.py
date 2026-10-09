from __future__ import annotations

import numpy as np

from fewshot_ad.calibration import PrototypeBank


def test_prototype_distance_separates_far_patch() -> None:
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 0] = 1.0
    bank = PrototypeBank.build("x", [normal], max_prototypes=16, seed=0)

    sample = normal.copy()
    sample[1, 1] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    distance = bank.distance_map(sample)

    assert distance[1, 1] > 0.9
    assert float(np.median(distance)) < 1e-4


def test_fuse_anomaly_maps_aligns_shapes() -> None:
    base_map = np.zeros((4, 4), dtype=np.float32)
    calibration_map = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    from fewshot_ad.calibration.prototype_bank import fuse_anomaly_maps

    output = fuse_anomaly_maps(base_map, calibration_map, alpha=0.5)

    assert output.shape == base_map.shape
    assert np.isfinite(output).all()
    assert output[0, 0] == 0.0
