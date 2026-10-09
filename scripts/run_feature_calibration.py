from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.calibration import PrototypeBank, fuse_anomaly_maps
from fewshot_ad.datasets.common import record_from_dict, read_json, write_json
from fewshot_ad.evaluation import image_auroc, pixel_auroc, pro_auc, topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run few-shot normal prototype calibration on exported features.")
    parser.add_argument("--split", required=True, help="Few-shot split JSON from build_fewshot_splits.py.")
    parser.add_argument("--feature-manifest", required=True, help="JSONL mapping image_path to feature_path.")
    parser.add_argument("--output", required=True, help="Output metrics JSON path.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-prototypes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--pro-max-fpr", type=float, default=0.30)
    parser.add_argument("--pro-num-thresholds", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="Device for prototype distance computation: cpu, cuda, or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = read_json(Path(args.split))
    calibration_records = [record_from_dict(item) for item in split["calibration"]]
    test_records = [record_from_dict(item) for item in split["test"]]
    feature_paths = load_manifest(Path(args.feature_manifest))

    features_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in calibration_records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        features_by_class[record.class_name].append(sample.features)

    banks = {
        class_name: PrototypeBank.build(
            class_name,
            tensors,
            max_prototypes=args.max_prototypes,
            seed=args.seed,
        )
        for class_name, tensors in features_by_class.items()
    }
    device_banks = build_device_banks(banks, args.device)

    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_masks: list[np.ndarray] = []
    pixel_maps: list[np.ndarray] = []
    per_sample: list[dict[str, object]] = []

    for record in test_records:
        if record.class_name not in banks:
            continue
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        calibration_map = distance_map(
            banks[record.class_name],
            sample.features,
            device_banks.get(record.class_name),
        )
        anomaly_map = fuse_anomaly_maps(sample.base_map, calibration_map, alpha=args.alpha)
        score = topk_mean(anomaly_map, fraction=args.topk_fraction)

        image_labels.append(1 if record.is_anomaly else 0)
        image_scores.append(score)
        if sample.mask is not None:
            pixel_masks.append((sample.mask > 0).astype(np.uint8))
            pixel_maps.append(anomaly_map)

        per_sample.append(
            {
                "image_path": record.image_path,
                "class_name": record.class_name,
                "label": record.label,
                "score": score,
            }
        )

    metrics = {
        "image_auroc": image_auroc(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_maps) if pixel_masks else None,
        "pro_auc": pro_auc(
            pixel_masks,
            pixel_maps,
            max_fpr=args.pro_max_fpr,
            num_thresholds=args.pro_num_thresholds,
        ) if pixel_masks else None,
        "num_calibration": len(calibration_records),
        "num_test_evaluated": len(image_scores),
        "alpha": args.alpha,
        "max_prototypes": args.max_prototypes,
        "topk_fraction": args.topk_fraction,
        "pro_num_thresholds": args.pro_num_thresholds,
        "device": args.device,
        "per_sample": per_sample,
    }
    write_json(Path(args.output), metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}, indent=2))


def build_device_banks(banks: dict[str, PrototypeBank], device: str) -> dict[str, object]:
    if device == "cpu":
        return {}

    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device!r}, but CUDA is not available.")

    return {
        class_name: torch.as_tensor(bank.prototypes, dtype=torch.float32, device=torch_device)
        for class_name, bank in banks.items()
    }


def distance_map(bank: PrototypeBank, features: np.ndarray, device_prototypes: object | None) -> np.ndarray:
    if device_prototypes is None:
        return bank.distance_map(features)

    import torch

    prototypes = device_prototypes
    spatial_shape = features.shape[:-1]
    query = torch.as_tensor(
        features.reshape(-1, features.shape[-1]),
        dtype=torch.float32,
        device=prototypes.device,
    )
    with torch.no_grad():
        query = torch.nn.functional.normalize(query, dim=-1, eps=1e-12)
        similarities = query @ prototypes.T
        distances = 1.0 - similarities.max(dim=1).values
    return distances.reshape(spatial_shape).detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    main()
