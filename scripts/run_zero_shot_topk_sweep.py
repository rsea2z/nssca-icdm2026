from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.datasets.common import record_from_dict, read_json, write_json
from fewshot_ad.evaluation import image_auroc, pixel_auroc, pro_auc, topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot anomaly maps over several top-k fractions.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--pro-max-fpr", type=float, default=0.30)
    parser.add_argument("--pro-num-thresholds", type=int, default=50)
    parser.add_argument(
        "--skip-pixel-metrics",
        action="store_true",
        help="Only compute image AUROC and normal FPR. Pixel AUROC and PRO are top-k invariant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = read_json(Path(args.split))
    test_records = [record_from_dict(item) for item in split["test"]]
    feature_paths = load_manifest(Path(args.feature_manifest))

    image_labels: list[int] = []
    scores_by_fraction: dict[float, list[float]] = {fraction: [] for fraction in args.topk_fractions}
    per_sample_by_fraction: dict[float, list[dict[str, object]]] = {fraction: [] for fraction in args.topk_fractions}
    pixel_masks: list[np.ndarray] = []
    pixel_maps: list[np.ndarray] = []

    for record in test_records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        if sample.base_map is None:
            raise KeyError(f"{sample.path} does not contain required key 'base_map'.")

        anomaly_map = sample.base_map
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

        if not args.skip_pixel_metrics and sample.mask is not None:
            pixel_masks.append((sample.mask > 0).astype(np.uint8))
            pixel_maps.append(anomaly_map)

    pixel_auc = pixel_auroc(pixel_masks, pixel_maps) if pixel_masks else None
    pro = pro_auc(pixel_masks, pixel_maps, max_fpr=args.pro_max_fpr, num_thresholds=args.pro_num_thresholds) if pixel_masks else None

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
        "kind": "zero_shot",
        "num_test_evaluated": len(image_labels),
        "pro_num_thresholds": args.pro_num_thresholds,
        "results": results,
    }
    write_json(Path(args.output), output)
    print(json.dumps({**output, "results": [{k: v for k, v in item.items() if k != "per_sample"} for item in results]}, indent=2))


def normal_fpr_at_95_tpr(per_sample: list[dict[str, object]]) -> float | None:
    scores = np.asarray([float(item["score"]) for item in per_sample], dtype=float)
    labels = np.asarray([1 if item.get("label") == "anomaly" else 0 for item in per_sample], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return None
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
