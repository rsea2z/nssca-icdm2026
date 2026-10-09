from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.calibration import PrototypeBank, fuse_anomaly_maps
from fewshot_ad.calibration.prototype_bank import minmax_scale
from fewshot_ad.datasets.common import record_from_dict, read_json
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot zero-shot vs calibrated qualitative MVTec cases.")
    parser.add_argument("--split", default="splits/mvtec/shot_1_seed_0.json")
    parser.add_argument("--feature-manifest", default="features/manifest.jsonl")
    parser.add_argument("--zero-metrics", default="runs/zero_shot_topk0.001/metrics.json")
    parser.add_argument(
        "--calibration-metrics",
        default="runs/mvtec_final512/calib_shot1_seed0_alpha0.25_proto512/metrics.json",
    )
    parser.add_argument("--zero-topk-fraction", type=float, default=0.001)
    parser.add_argument("--calibration-topk-fraction", type=float, default=None)
    parser.add_argument("--output", default="figures/qualitative_cases.png")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max-prototypes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-normals", type=int, default=2)
    parser.add_argument("--num-anomalies", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = read_json(Path(args.split))
    feature_paths = load_manifest(Path(args.feature_manifest))
    banks = build_banks(split, feature_paths, args.max_prototypes, args.seed)

    zero = keyed_samples(Path(args.zero_metrics), topk_fraction=args.zero_topk_fraction)
    calibrated = keyed_samples(Path(args.calibration_metrics), topk_fraction=args.calibration_topk_fraction)
    selected = select_cases(zero, calibrated, args.num_normals, args.num_anomalies)
    if not selected:
        raise RuntimeError("No overlapping samples found between zero-shot and calibration metrics.")

    rows = []
    for item in selected:
        sample = load_feature_sample(resolve_feature_path(item["image_path"], feature_paths))
        bank = banks[item["class_name"]]
        calibration_map = bank.distance_map(sample.features)
        fused_map = fuse_anomaly_maps(sample.base_map, calibration_map, alpha=args.alpha)
        base_map = minmax_scale(sample.base_map)
        rows.append(
            {
                **item,
                "image": load_image(item["image_path"], base_map.shape),
                "mask": (sample.mask > 0).astype(np.float32) if sample.mask is not None else np.zeros_like(base_map),
                "base_map": base_map,
                "fused_map": fused_map,
            }
        )

    plot_rows(rows, Path(args.output))


def build_banks(split: dict[str, object], feature_paths: dict[str, str], max_prototypes: int, seed: int) -> dict[str, PrototypeBank]:
    features_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in split["calibration"]:
        record = record_from_dict(item)
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        features_by_class[record.class_name].append(sample.features)
    return {
        class_name: PrototypeBank.build(class_name, tensors, max_prototypes=max_prototypes, seed=seed)
        for class_name, tensors in features_by_class.items()
    }


def keyed_samples(path: Path, *, topk_fraction: float | None = None) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_sample = payload.get("per_sample")
    if per_sample is None:
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{path} does not contain per_sample or top-k results.")
        selected = select_topk_result(results, path, topk_fraction=topk_fraction)
        per_sample = selected.get("per_sample")
        if per_sample is None:
            raise ValueError(f"{path} selected result does not contain per_sample.")
    return {str(item["image_path"]): item for item in per_sample}


def select_topk_result(results: list[dict[str, object]], path: Path, *, topk_fraction: float | None) -> dict[str, object]:
    if topk_fraction is None and len(results) == 1:
        return results[0]
    matches = [
        result
        for result in results
        if result.get("topk_fraction") is not None
        and topk_fraction is not None
        and abs(float(result["topk_fraction"]) - topk_fraction) < 1e-12
    ]
    if not matches:
        raise ValueError(f"{path} does not contain topk_fraction={topk_fraction}.")
    return matches[0]


def select_cases(
    zero: dict[str, dict[str, object]],
    calibrated: dict[str, dict[str, object]],
    num_normals: int,
    num_anomalies: int,
) -> list[dict[str, object]]:
    rows = []
    for image_path, zero_item in zero.items():
        if image_path not in calibrated:
            continue
        calib_item = calibrated[image_path]
        rows.append(
            {
                "image_path": image_path,
                "class_name": zero_item["class_name"],
                "label": zero_item["label"],
                "zero_score": float(zero_item["score"]),
                "calibrated_score": float(calib_item["score"]),
                "score_delta": float(zero_item["score"]) - float(calib_item["score"]),
            }
        )

    normals = [row for row in rows if row["label"] == "good"]
    anomalies = [row for row in rows if row["label"] != "good"]
    normals = sorted(normals, key=lambda row: (row["score_delta"], row["zero_score"]), reverse=True)
    anomalies = sorted(anomalies, key=lambda row: row["calibrated_score"], reverse=True)
    return normals[:num_normals] + anomalies[:num_anomalies]


def load_image(path: str, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def plot_rows(rows: list[dict[str, object]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 1.75 * len(rows)), squeeze=False)
    titles = ["Image", "GT mask", "AnomalyCLIP", "Calibrated"]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title)

    for row_idx, row in enumerate(rows):
        image = row["image"]
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 1].imshow(row["mask"], cmap="gray", vmin=0, vmax=1)
        overlay(axes[row_idx, 2], image, row["base_map"])
        overlay(axes[row_idx, 3], image, row["fused_map"])

        label = f"{row['class_name']} / {row['label']}\n{row['zero_score']:.3f}->{row['calibrated_score']:.3f}"
        axes[row_idx, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=30)
        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    print(output)
    print(pdf_output)


def overlay(ax: plt.Axes, image: np.ndarray, anomaly_map: np.ndarray) -> None:
    ax.imshow(image)
    ax.imshow(minmax_scale(anomaly_map), cmap="magma", alpha=0.58, vmin=0, vmax=1)


if __name__ == "__main__":
    main()
