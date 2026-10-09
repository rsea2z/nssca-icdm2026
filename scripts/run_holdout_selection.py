from __future__ import annotations

import argparse
import itertools
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
from fewshot_ad.evaluation import image_auroc, topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path
from run_feature_calibration import build_device_banks, distance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select global calibration hyperparameters on held-in classes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--max-prototypes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = read_json(Path(args.split))
    calibration_records = [record_from_dict(item) for item in split["calibration"]]
    test_records = [record_from_dict(item) for item in split["test"]]
    feature_paths = load_manifest(Path(args.feature_manifest))

    class_names = sorted({record.class_name for record in test_records})
    folds = {
        fold: [name for index, name in enumerate(class_names) if index % args.folds == fold]
        for fold in range(args.folds)
    }

    features_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in calibration_records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        features_by_class[record.class_name].append(sample.features)

    banks_by_proto = {
        max_proto: {
            class_name: PrototypeBank.build(
                class_name,
                tensors,
                max_prototypes=max_proto,
                seed=args.seed,
            )
            for class_name, tensors in features_by_class.items()
        }
        for max_proto in args.max_prototypes
    }
    device_banks_by_proto = {
        max_proto: build_device_banks(banks, args.device)
        for max_proto, banks in banks_by_proto.items()
    }

    candidate_samples: dict[tuple[float, int, float], list[dict[str, object]]] = {
        candidate: []
        for candidate in itertools.product(args.alphas, args.max_prototypes, args.topk_fractions)
    }

    for record in test_records:
        if record.class_name not in features_by_class:
            continue
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        for max_proto, banks in banks_by_proto.items():
            proto_map = distance_map(
                banks[record.class_name],
                sample.features,
                device_banks_by_proto[max_proto].get(record.class_name),
            )
            for alpha in args.alphas:
                anomaly_map = fuse_anomaly_maps(sample.base_map, proto_map, alpha=alpha)
                for topk in args.topk_fractions:
                    candidate_samples[(alpha, max_proto, topk)].append(
                        {
                            "image_path": record.image_path,
                            "class_name": record.class_name,
                            "label": record.label,
                            "score": topk_mean(anomaly_map, fraction=topk),
                        }
                    )

    fold_rows = []
    for fold, heldout_classes in folds.items():
        heldout = set(heldout_classes)
        dev_classes = [name for name in class_names if name not in heldout]
        scored_candidates = []
        for candidate, samples in candidate_samples.items():
            dev_metrics = compute_metrics([item for item in samples if item["class_name"] in dev_classes])
            scored_candidates.append((candidate, dev_metrics))
        selected, dev_metrics = select_candidate(scored_candidates)
        heldout_metrics = compute_metrics(
            [item for item in candidate_samples[selected] if item["class_name"] in heldout]
        )
        fold_rows.append(
            {
                "fold": fold,
                "dev_classes": dev_classes,
                "heldout_classes": heldout_classes,
                "selected": {
                    "alpha": selected[0],
                    "max_prototypes": selected[1],
                    "topk_fraction": selected[2],
                },
                "dev_metrics": dev_metrics,
                "heldout_metrics": heldout_metrics,
            }
        )

    output = {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "fold_count": args.folds,
        "class_names": class_names,
        "candidate_grid": {
            "alphas": args.alphas,
            "max_prototypes": args.max_prototypes,
            "topk_fractions": args.topk_fractions,
        },
        "selection_rule": "maximize dev image AUROC, then minimize dev FPR@95TPR, then prefer smaller prototype budget and lower top-k",
        "folds": fold_rows,
    }
    write_json(Path(args.output), output)
    print(json.dumps(output, indent=2))


def select_candidate(
    scored_candidates: list[tuple[tuple[float, int, float], dict[str, float | int | None]]]
) -> tuple[tuple[float, int, float], dict[str, float | int | None]]:
    def key(item: tuple[tuple[float, int, float], dict[str, float | int | None]]) -> tuple[float, float, int, float]:
        candidate, metrics = item
        image_auc = finite_or(metrics.get("image_auroc"), low=True)
        fpr = finite_or(metrics.get("normal_fpr_at_95_tpr"), low=False)
        return (-image_auc, fpr, int(candidate[1]), float(candidate[2]))

    return sorted(scored_candidates, key=key)[0]


def finite_or(value: object, *, low: bool) -> float:
    if value is None:
        return -float("inf") if low else float("inf")
    value = float(value)
    if not np.isfinite(value):
        return -float("inf") if low else float("inf")
    return value


def compute_metrics(samples: list[dict[str, object]]) -> dict[str, float | int | None]:
    labels = np.asarray([1 if str(item["label"]).lower() == "anomaly" else 0 for item in samples], dtype=int)
    scores = np.asarray([float(item["score"]) for item in samples], dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        image_auc = None
    else:
        image_auc = image_auroc(labels, scores)
    return {
        "num_samples": int(labels.size),
        "num_anomaly": int(labels.sum()) if labels.size else 0,
        "num_normal": int((labels == 0).sum()) if labels.size else 0,
        "image_auroc": image_auc,
        "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(samples),
    }


def normal_fpr_at_95_tpr(samples: list[dict[str, object]]) -> float | None:
    scores = np.asarray([float(item["score"]) for item in samples], dtype=float)
    labels = np.asarray([1 if str(item["label"]).lower() == "anomaly" else 0 for item in samples], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return None
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
