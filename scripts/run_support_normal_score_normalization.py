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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fewshot_ad.calibration import PrototypeBank, fuse_anomaly_maps
from fewshot_ad.datasets.common import record_from_dict, read_json, write_json
from fewshot_ad.evaluation import image_auroc, pixel_auroc, pro_auc, topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path
from run_feature_calibration import build_device_banks, distance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate support-only normal score normalization.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-prototypes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-fraction", type=float, default=0.001)
    parser.add_argument("--zero-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--normalization",
        choices=["none", "support_center", "support_z"],
        nargs="+",
        default=["none", "support_center", "support_z"],
    )
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
    support_stats = compute_support_stats(args, calibration_records, feature_paths, banks, device_banks)

    image_labels: list[int] = []
    per_sample_by_key: dict[tuple[str, float], list[dict[str, object]]] = {
        (mode, weight): [] for mode in args.normalization for weight in args.zero_weights
    }
    pixel_masks: list[np.ndarray] = []
    pixel_maps: list[np.ndarray] = []

    for record in test_records:
        if record.class_name not in banks:
            continue
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        base_score, fused_score, anomaly_map = score_sample(args, record.class_name, sample, banks, device_banks)
        image_labels.append(1 if record.is_anomaly else 0)
        for weight in args.zero_weights:
            raw_score = blend_score(base_score, fused_score, weight)
            for mode in args.normalization:
                score = normalize_score(raw_score, record.class_name, weight, mode, support_stats)
                per_sample_by_key[(mode, weight)].append(
                    {
                        "image_path": record.image_path,
                        "class_name": record.class_name,
                        "label": record.label,
                        "score": score,
                        "raw_score": raw_score,
                        "zero_score": base_score,
                        "fused_score": fused_score,
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
    for mode in args.normalization:
        for weight in args.zero_weights:
            samples = per_sample_by_key[(mode, weight)]
            scores = [float(item["score"]) for item in samples]
            results.append(
                {
                    "score_mode": mode,
                    "zero_weight": weight,
                    "topk_fraction": args.topk_fraction,
                    "image_auroc": image_auroc(image_labels, scores),
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
        "topk_fraction": args.topk_fraction,
        "pro_num_thresholds": args.pro_num_thresholds,
        "device": args.device,
        "support_stats": {
            f"{class_name}|{weight:g}": {
                "mean": values["mean"],
                "pooled_std": values["pooled_std"],
                "num_support": values["num_support"],
            }
            for (class_name, weight), values in sorted(support_stats.items())
        },
        "results": results,
    }
    write_json(Path(args.output), output)
    print(json.dumps({**output, "results": [{k: v for k, v in item.items() if k != "per_sample"} for item in results]}, indent=2))


def compute_support_stats(
    args: argparse.Namespace,
    records,
    feature_paths: dict[str, Path],
    banks: dict[str, PrototypeBank],
    device_banks: dict[str, object],
) -> dict[tuple[str, float], dict[str, float]]:
    scores_by_class_weight: dict[tuple[str, float], list[float]] = defaultdict(list)
    scores_by_weight: dict[float, list[float]] = defaultdict(list)
    for record in records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        base_score, fused_score, _ = score_sample(args, record.class_name, sample, banks, device_banks)
        for weight in args.zero_weights:
            score = blend_score(base_score, fused_score, weight)
            scores_by_class_weight[(record.class_name, weight)].append(score)
            scores_by_weight[weight].append(score)

    pooled_std = {
        weight: safe_std(values)
        for weight, values in scores_by_weight.items()
    }
    stats = {}
    for key, values in scores_by_class_weight.items():
        class_name, weight = key
        stats[(class_name, weight)] = {
            "mean": float(np.mean(values)),
            "pooled_std": pooled_std[weight],
            "num_support": float(len(values)),
        }
    return stats


def score_sample(
    args: argparse.Namespace,
    class_name: str,
    sample,
    banks: dict[str, PrototypeBank],
    device_banks: dict[str, object],
) -> tuple[float, float, np.ndarray]:
    if sample.base_map is None:
        raise KeyError(f"{sample.path} does not contain required key 'base_map'.")
    calibration_map = distance_map(
        banks[class_name],
        sample.features,
        device_banks.get(class_name),
    )
    anomaly_map = fuse_anomaly_maps(sample.base_map, calibration_map, alpha=args.alpha)
    base_score = topk_mean(sample.base_map, fraction=args.topk_fraction)
    fused_score = topk_mean(anomaly_map, fraction=args.topk_fraction)
    return base_score, fused_score, anomaly_map


def blend_score(base_score: float, fused_score: float, zero_weight: float) -> float:
    return float(zero_weight * base_score + (1.0 - zero_weight) * fused_score)


def normalize_score(
    score: float,
    class_name: str,
    zero_weight: float,
    mode: str,
    support_stats: dict[tuple[str, float], dict[str, float]],
) -> float:
    if mode == "none":
        return float(score)
    stats = support_stats[(class_name, zero_weight)]
    centered = float(score) - float(stats["mean"])
    if mode == "support_center":
        return centered
    if mode == "support_z":
        return centered / float(stats["pooled_std"])
    raise ValueError(f"Unknown normalization mode: {mode}")


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    std = float(np.std(np.asarray(values, dtype=float)))
    return std if std >= 1e-8 else 1.0


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
