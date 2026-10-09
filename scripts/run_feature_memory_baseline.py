from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.calibration import PrototypeBank, fuse_anomaly_maps
from fewshot_ad.calibration.prototype_bank import l2_normalize
from fewshot_ad.datasets.common import record_from_dict, read_json, write_json
from fewshot_ad.evaluation import image_auroc, pixel_auroc, pro_auc, topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path
from run_feature_calibration import build_device_banks, distance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run same-feature memory baselines on exported features.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", choices=["nearest", "diag_gaussian"], required=True)
    parser.add_argument("--max-prototypes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument("--pro-max-fpr", type=float, default=0.30)
    parser.add_argument("--pro-num-thresholds", type=int, default=50)
    parser.add_argument("--compute-pixel-metrics", action="store_true")
    parser.add_argument("--device", default="cpu")
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

    if args.baseline == "nearest":
        scorer = NearestScorer(features_by_class, max_prototypes=args.max_prototypes, seed=args.seed, device=args.device)
    else:
        scorer = DiagGaussianScorer(features_by_class, max_patches=args.max_prototypes, seed=args.seed)

    image_labels: list[int] = []
    scores_by_fraction: dict[float, list[float]] = {fraction: [] for fraction in args.topk_fractions}
    per_sample_by_fraction: dict[float, list[dict[str, object]]] = {fraction: [] for fraction in args.topk_fractions}
    pixel_masks: list[np.ndarray] = []
    pixel_maps: list[np.ndarray] = []

    for record in test_records:
        if record.class_name not in scorer.class_names:
            continue
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        raw_map = scorer.score_map(record.class_name, sample.features)
        anomaly_map = fuse_anomaly_maps(sample.base_map, raw_map, alpha=0.0)
        image_label = 1 if record.is_anomaly else 0
        image_labels.append(image_label)
        for fraction in args.topk_fractions:
            score = topk_mean(anomaly_map, fraction=fraction)
            scores_by_fraction[fraction].append(score)
            per_sample_by_fraction[fraction].append(
                {
                    "image_path": record.image_path,
                    "class_name": record.class_name,
                    "label": record.label,
                    "score": score,
                }
            )

        if args.compute_pixel_metrics and sample.mask is not None:
            pixel_masks.append((sample.mask > 0).astype(np.uint8))
            pixel_maps.append(anomaly_map)

    pixel_auc = pixel_auroc(pixel_masks, pixel_maps) if args.compute_pixel_metrics and pixel_masks else None
    pro = (
        pro_auc(pixel_masks, pixel_maps, max_fpr=args.pro_max_fpr, num_thresholds=args.pro_num_thresholds)
        if args.compute_pixel_metrics and pixel_masks
        else None
    )

    results = []
    for fraction in args.topk_fractions:
        per_sample = per_sample_by_fraction[fraction]
        results.append(
            {
                "topk_fraction": fraction,
                "image_auroc": image_auroc(image_labels, scores_by_fraction[fraction]),
                "pixel_auroc": pixel_auc,
                "pro_auc": pro,
                "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(per_sample),
                "per_sample": per_sample,
            }
        )

    output = {
        "baseline": args.baseline,
        "max_prototypes": args.max_prototypes,
        "seed": args.seed,
        "num_calibration": len(calibration_records),
        "num_test_evaluated": len(image_labels),
        "pro_num_thresholds": args.pro_num_thresholds,
        "device": args.device,
        "results": results,
    }
    write_json(Path(args.output), output)
    print(json.dumps({**output, "results": [{k: v for k, v in item.items() if k != "per_sample"} for item in results]}, indent=2))


class NearestScorer:
    def __init__(self, features_by_class: dict[str, list[np.ndarray]], *, max_prototypes: int, seed: int, device: str) -> None:
        self.banks = {
            class_name: PrototypeBank.build(class_name, tensors, max_prototypes=max_prototypes, seed=seed)
            for class_name, tensors in features_by_class.items()
        }
        self.device_banks = build_device_banks(self.banks, device)

    @property
    def class_names(self) -> set[str]:
        return set(self.banks)

    def score_map(self, class_name: str, features: np.ndarray) -> np.ndarray:
        return distance_map(self.banks[class_name], features, self.device_banks.get(class_name))


@dataclass
class DiagStats:
    mean: np.ndarray
    inv_std: np.ndarray


class DiagGaussianScorer:
    def __init__(self, features_by_class: dict[str, list[np.ndarray]], *, max_patches: int, seed: int) -> None:
        self.stats = {
            class_name: build_diag_stats(tensors, max_patches=max_patches, seed=seed)
            for class_name, tensors in features_by_class.items()
        }

    @property
    def class_names(self) -> set[str]:
        return set(self.stats)

    def score_map(self, class_name: str, features: np.ndarray) -> np.ndarray:
        stats = self.stats[class_name]
        spatial_shape = features.shape[:-1]
        query = l2_normalize(features.reshape(-1, features.shape[-1]))
        z = (query - stats.mean) * stats.inv_std
        score = np.mean(z * z, axis=1)
        return score.reshape(spatial_shape).astype(np.float32)


def build_diag_stats(feature_tensors: list[np.ndarray], *, max_patches: int, seed: int) -> DiagStats:
    flattened = [features.reshape(-1, features.shape[-1]) for features in feature_tensors]
    features = np.concatenate(flattened, axis=0).astype(np.float32)
    features = features[np.isfinite(features).all(axis=1)]
    if len(features) > max_patches:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(features), size=max_patches, replace=False)
        features = features[indices]
    features = l2_normalize(features)
    mean = np.mean(features, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(features, axis=0, dtype=np.float64).astype(np.float32)
    return DiagStats(mean=mean, inv_std=(1.0 / np.maximum(std, 1e-3)).astype(np.float32))


def normal_fpr_at_95_tpr(per_sample: list[dict[str, object]]) -> float | None:
    scores = np.asarray([float(item["score"]) for item in per_sample], dtype=float)
    labels = np.asarray([1 if str(item.get("label")).lower() == "anomaly" else 0 for item in per_sample], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return None
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
