from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.datasets.common import record_from_dict, read_json, write_json
from fewshot_ad.evaluation import image_auroc, pixel_auroc, pro_auc, topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exported zero-shot anomaly maps.")
    parser.add_argument("--split", required=True, help="Few-shot split JSON; only the test section is used.")
    parser.add_argument("--feature-manifest", required=True, help="JSONL mapping image_path to feature_path.")
    parser.add_argument("--output", required=True, help="Output metrics JSON path.")
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--pro-max-fpr", type=float, default=0.30)
    parser.add_argument("--pro-num-thresholds", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = read_json(Path(args.split))
    test_records = [record_from_dict(item) for item in split["test"]]
    feature_paths = load_manifest(Path(args.feature_manifest))

    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_masks = []
    pixel_maps = []
    per_sample = []

    for record in test_records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        if sample.base_map is None:
            raise KeyError(f"{sample.path} does not contain required key 'base_map'.")
        anomaly_map = sample.base_map
        score = topk_mean(anomaly_map, fraction=args.topk_fraction)

        image_labels.append(1 if record.is_anomaly else 0)
        image_scores.append(score)
        if sample.mask is not None:
            pixel_masks.append((sample.mask > 0).astype("uint8"))
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
        "pro_auc": pro_auc(pixel_masks, pixel_maps, max_fpr=args.pro_max_fpr, num_thresholds=args.pro_num_thresholds) if pixel_masks else None,
        "num_test_evaluated": len(image_scores),
        "topk_fraction": args.topk_fraction,
        "pro_num_thresholds": args.pro_num_thresholds,
        "per_sample": per_sample,
    }
    write_json(Path(args.output), metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
