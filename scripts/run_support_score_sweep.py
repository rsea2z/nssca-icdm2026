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
from run_feature_calibration import build_device_banks, distance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate support-normalized image scores over top-k fractions.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-prototypes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument("--score-modes", nargs="+", choices=["raw", "centered"], default=["raw", "centered"])
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
    support_means = compute_support_means(args, calibration_records, feature_paths, banks, device_banks)

    image_labels: list[int] = []
    scores: dict[tuple[str, float], list[float]] = {
        (mode, fraction): [] for mode in args.score_modes for fraction in args.topk_fractions
    }
    per_sample: dict[tuple[str, float], list[dict[str, object]]] = {
        (mode, fraction): [] for mode in args.score_modes for fraction in args.topk_fractions
    }
    pixel_masks: list[np.ndarray] = []
    pixel_maps: list[np.ndarray] = []

    for record in test_records:
        if record.class_name not in banks:
            continue

        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        anomaly_map = calibrated_map(args, record.class_name, sample.features, sample.base_map, banks, device_banks)
        image_label = 1 if record.is_anomaly else 0
        image_labels.append(image_label)

        for fraction in args.topk_fractions:
            raw_score = topk_mean(anomaly_map, fraction=fraction)
            for mode in args.score_modes:
                score = raw_score
                if mode == "centered":
                    score = raw_score - support_means.get((record.class_name, fraction), 0.0)
                key = (mode, fraction)
                scores[key].append(score)
                per_sample[key].append(
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
    for mode in args.score_modes:
        for fraction in args.topk_fractions:
            key = (mode, fraction)
            samples = per_sample[key]
            results.append(
                {
                    "score_mode": mode,
                    "topk_fraction": fraction,
                    "image_auroc": image_auroc(image_labels, scores[key]),
                    "pixel_auroc": pixel_auc,
                    "pro_auc": pro,
                    "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(samples),
                    "per_sample": samples,
                }
            )

    output = {
        "alpha": args.alpha,
        "max_prototypes": args.max_prototypes,
        "seed": args.seed,
        "num_calibration": len(calibration_records),
        "num_test_evaluated": len(image_labels),
        "pro_num_thresholds": args.pro_num_thresholds,
        "device": args.device,
        "support_means": {
            f"{class_name}|{fraction:g}": value
            for (class_name, fraction), value in sorted(support_means.items())
        },
        "results": results,
    }
    write_json(Path(args.output), output)
    print(json.dumps({**output, "results": [{k: v for k, v in item.items() if k != "per_sample"} for item in results]}, indent=2))


def compute_support_means(
    args: argparse.Namespace,
    records,
    feature_paths: dict[str, Path],
    banks: dict[str, PrototypeBank],
    device_banks: dict[str, object],
) -> dict[tuple[str, float], float]:
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for record in records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        anomaly_map = calibrated_map(args, record.class_name, sample.features, sample.base_map, banks, device_banks)
        for fraction in args.topk_fractions:
            values[(record.class_name, fraction)].append(topk_mean(anomaly_map, fraction=fraction))
    return {key: float(np.mean(items)) for key, items in values.items()}


def calibrated_map(
    args: argparse.Namespace,
    class_name: str,
    features: np.ndarray,
    base_map: np.ndarray,
    banks: dict[str, PrototypeBank],
    device_banks: dict[str, object],
) -> np.ndarray:
    calibration_map = distance_map(
        banks[class_name],
        features,
        device_banks.get(class_name),
    )
    return fuse_anomaly_maps(base_map, calibration_map, alpha=args.alpha)


def normal_fpr_at_95_tpr(samples: list[dict[str, object]]) -> float | None:
    scores = np.asarray([float(item["score"]) for item in samples], dtype=float)
    labels = np.asarray([1 if item.get("label") == "anomaly" else 0 for item in samples], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return None
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
