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
    parser = argparse.ArgumentParser(description="Run component attribution controls on exported features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-prototypes", type=int, default=512)
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
    rotated_class = build_rotated_class_map(sorted(banks))

    mode_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    mode_labels: dict[str, list[int]] = defaultdict(list)
    mode_masks: dict[str, list[np.ndarray]] = defaultdict(list)
    scores_by_mode_fraction: dict[tuple[str, float], list[float]] = defaultdict(list)
    samples_by_mode_fraction: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)

    for record in test_records:
        if record.class_name not in banks:
            continue
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        if sample.base_map is None:
            raise KeyError(f"{sample.path} does not contain required key 'base_map'.")

        own_proto_map = distance_map(
            banks[record.class_name],
            sample.features,
            device_banks.get(record.class_name),
        )
        other_class = rotated_class[record.class_name]
        other_proto_map = distance_map(
            banks[other_class],
            sample.features,
            device_banks.get(other_class),
        )
        maps = {
            "zero_shot_topk": sample.base_map,
            "prototype_only": fuse_anomaly_maps(sample.base_map, own_proto_map, alpha=0.0),
            "fused": fuse_anomaly_maps(sample.base_map, own_proto_map, alpha=args.alpha),
            "other_class_fused": fuse_anomaly_maps(sample.base_map, other_proto_map, alpha=args.alpha),
        }

        label = 1 if record.is_anomaly else 0
        for mode, anomaly_map in maps.items():
            mode_labels[mode].append(label)
            if args.compute_pixel_metrics and sample.mask is not None:
                mode_masks[mode].append((sample.mask > 0).astype(np.uint8))
                mode_maps[mode].append(anomaly_map)
            for fraction in args.topk_fractions:
                score = topk_mean(anomaly_map, fraction=fraction)
                key = (mode, fraction)
                scores_by_mode_fraction[key].append(score)
                samples_by_mode_fraction[key].append(
                    {
                        "image_path": record.image_path,
                        "class_name": record.class_name,
                        "label": record.label,
                        "score": score,
                    }
                )

    pixel_by_mode = {}
    pro_by_mode = {}
    for mode in mode_labels:
        if args.compute_pixel_metrics and mode_masks[mode]:
            pixel_by_mode[mode] = pixel_auroc(mode_masks[mode], mode_maps[mode])
            pro_by_mode[mode] = pro_auc(
                mode_masks[mode],
                mode_maps[mode],
                max_fpr=args.pro_max_fpr,
                num_thresholds=args.pro_num_thresholds,
            )
        else:
            pixel_by_mode[mode] = None
            pro_by_mode[mode] = None

    results = []
    for (mode, fraction), scores in sorted(scores_by_mode_fraction.items()):
        per_sample = samples_by_mode_fraction[(mode, fraction)]
        results.append(
            {
                "mode": mode,
                "topk_fraction": fraction,
                "image_auroc": image_auroc(mode_labels[mode], scores),
                "pixel_auroc": pixel_by_mode[mode],
                "pro_auc": pro_by_mode[mode],
                "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(per_sample),
                "per_sample": per_sample,
            }
        )

    output = {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "alpha": args.alpha,
        "max_prototypes": args.max_prototypes,
        "topk_fractions": args.topk_fractions,
        "num_calibration": len(calibration_records),
        "num_test_evaluated": len(next(iter(mode_labels.values()))) if mode_labels else 0,
        "pro_num_thresholds": args.pro_num_thresholds,
        "device": args.device,
        "results": results,
    }
    write_json(Path(args.output), output)
    print(json.dumps({**output, "results": [{k: v for k, v in item.items() if k != "per_sample"} for item in results]}, indent=2))


def build_rotated_class_map(class_names: list[str]) -> dict[str, str]:
    if len(class_names) < 2:
        raise ValueError("Need at least two classes for other-class prototype control.")
    return {name: class_names[(index + 1) % len(class_names)] for index, name in enumerate(class_names)}


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
