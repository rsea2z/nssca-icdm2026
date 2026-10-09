from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.calibration import PrototypeBank, fuse_anomaly_maps
from fewshot_ad.evaluation import image_auroc, pixel_auroc, pro_auc, topk_mean


def main() -> None:
    rng = np.random.default_rng(0)
    normal_center = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    anomaly_center = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    calibration = [normal_center + 0.01 * rng.normal(size=(8, 8, 4)).astype(np.float32) for _ in range(4)]
    bank = PrototypeBank.build("synthetic", calibration, max_prototypes=64, seed=0)

    labels = []
    scores = []
    masks = []
    maps = []
    for index in range(20):
        features = normal_center + 0.01 * rng.normal(size=(8, 8, 4)).astype(np.float32)
        mask = np.zeros((8, 8), dtype=np.uint8)
        label = 0
        if index >= 10:
            features[2:5, 2:5] = anomaly_center + 0.01 * rng.normal(size=(3, 3, 4)).astype(np.float32)
            mask[2:5, 2:5] = 1
            label = 1
        anomaly_map = fuse_anomaly_maps(None, bank.distance_map(features), alpha=0.0)
        labels.append(label)
        scores.append(topk_mean(anomaly_map, fraction=0.10))
        masks.append(mask)
        maps.append(anomaly_map)

    result = {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(masks, maps),
        "pro_auc": pro_auc(masks, maps),
    }
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

    if result["image_auroc"] < 0.99 or result["pixel_auroc"] < 0.99:
        raise SystemExit("smoke test failed")


if __name__ == "__main__":
    main()
