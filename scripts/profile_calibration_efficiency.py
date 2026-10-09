from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.calibration import PrototypeBank, fuse_anomaly_maps
from fewshot_ad.datasets.common import record_from_dict, read_json
from fewshot_ad.evaluation import topk_mean
from fewshot_ad.models import load_feature_sample, load_manifest, resolve_feature_path
from run_feature_calibration import build_device_banks, distance_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile few-shot calibration memory and throughput.")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-prototypes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--topk-fraction", type=float, default=0.001)
    parser.add_argument("--num-images", type=int, default=300)
    parser.add_argument("--warmup-images", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_paths = load_manifest(Path(args.feature_manifest))
    rows = []
    for shot in args.shots:
        for seed in args.seeds:
            split_path = Path(args.splits_dir) / f"shot_{shot}_seed_{seed}.json"
            if not split_path.exists():
                print(f"[skip] missing split: {split_path}")
                continue
            split = read_json(split_path)
            calibration_records = [record_from_dict(item) for item in split["calibration"]]
            test_records = [record_from_dict(item) for item in split["test"]]
            for max_proto in args.max_prototypes:
                rows.append(
                    profile_one(
                        calibration_records,
                        test_records,
                        feature_paths,
                        dataset=Path(args.splits_dir).name,
                        shot=shot,
                        seed=seed,
                        max_prototypes=max_proto,
                        alpha=args.alpha,
                        topk_fraction=args.topk_fraction,
                        num_images=args.num_images,
                        warmup_images=args.warmup_images,
                        device=args.device,
                    )
                )

    payload = {"rows": rows, "groups": summarize(rows)}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    print(output_json)
    print(Path(args.output_csv))
    for group in payload["groups"]:
        print(
            f"{group['dataset']} shot={group['shot']} proto={group['max_prototypes']} "
            f"throughput={group['images_per_second_mean']:.2f}/s "
            f"memory={group['prototype_memory_mb_mean']:.2f}MB"
        )


def profile_one(
    calibration_records,
    test_records,
    feature_paths: dict[str, str],
    *,
    dataset: str,
    shot: int,
    seed: int,
    max_prototypes: int,
    alpha: float,
    topk_fraction: float,
    num_images: int,
    warmup_images: int,
    device: str,
) -> dict[str, object]:
    features_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in calibration_records:
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        features_by_class[record.class_name].append(sample.features)

    build_start = time.perf_counter()
    banks = {
        class_name: PrototypeBank.build(
            class_name,
            tensors,
            max_prototypes=max_prototypes,
            seed=seed,
        )
        for class_name, tensors in features_by_class.items()
    }
    device_banks = build_device_banks(banks, device)
    build_seconds = time.perf_counter() - build_start

    selected_records = [record for record in test_records if record.class_name in banks]
    selected_records = selected_records[: max(num_images + warmup_images, warmup_images)]
    reset_cuda_peak(device)
    timings = []
    for index, record in enumerate(selected_records):
        sample = load_feature_sample(resolve_feature_path(record.image_path, feature_paths))
        start = time.perf_counter()
        calibration_map = distance_map(
            banks[record.class_name],
            sample.features,
            device_banks.get(record.class_name),
        )
        anomaly_map = fuse_anomaly_maps(sample.base_map, calibration_map, alpha=alpha)
        _ = topk_mean(anomaly_map, fraction=topk_fraction)
        elapsed = time.perf_counter() - start
        if index >= warmup_images:
            timings.append(elapsed)

    total_eval_seconds = float(sum(timings))
    num_measured = len(timings)
    prototype_count = int(sum(bank.prototypes.shape[0] for bank in banks.values()))
    prototype_dim = int(next(iter(banks.values())).prototypes.shape[1]) if banks else 0
    prototype_bytes = int(sum(bank.prototypes.nbytes for bank in banks.values()))
    return {
        "dataset": dataset,
        "shot": shot,
        "seed": seed,
        "max_prototypes": max_prototypes,
        "alpha": alpha,
        "topk_fraction": topk_fraction,
        "device": device,
        "num_classes": len(banks),
        "prototype_count": prototype_count,
        "prototype_dim": prototype_dim,
        "prototype_memory_mb": prototype_bytes / (1024**2),
        "build_seconds": build_seconds,
        "num_measured_images": num_measured,
        "eval_seconds_total": total_eval_seconds,
        "seconds_per_image_mean": float(np.mean(timings)) if timings else None,
        "seconds_per_image_std": float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0,
        "images_per_second": float(num_measured / total_eval_seconds) if total_eval_seconds > 0 else None,
        "cuda_peak_memory_mb": cuda_peak_memory_mb(device),
    }


def reset_cuda_peak(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def cuda_peak_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        return None


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["shot"]), int(row["max_prototypes"]))].append(row)
    groups = []
    for (dataset, shot, max_prototypes), items in sorted(grouped.items()):
        groups.append(
            {
                "dataset": dataset,
                "shot": shot,
                "max_prototypes": max_prototypes,
                "num_runs": len(items),
                "prototype_memory_mb_mean": mean(items, "prototype_memory_mb"),
                "build_seconds_mean": mean(items, "build_seconds"),
                "seconds_per_image_mean": mean(items, "seconds_per_image_mean"),
                "images_per_second_mean": mean(items, "images_per_second"),
                "cuda_peak_memory_mb_mean": mean(items, "cuda_peak_memory_mb"),
            }
        )
    return groups


def mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
