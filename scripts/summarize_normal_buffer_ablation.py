from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fewshot_ad.evaluation import image_auroc
from summarize_score_blend import collect_records, normal_fpr_at_95_tpr, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normal-validation-buffer size ablation from stored image scores.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-json", default="runs/normal_buffer_ablation/summary.json")
    parser.add_argument("--output-md", default="docs/normal_buffer_ablation.md")
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--buffer-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument("--selected-buffer-size", type=int, default=8)
    parser.add_argument("--selected-zero-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records(ROOT / args.runs_root, args.weights)
    by_key = {
        (record["dataset"], record["shot"], record["seed"], record["zero_weight"]): record
        for record in records
    }
    rows = []
    baseline_rows = []
    for buffer_size in args.buffer_sizes:
        for buffer_seed in args.buffer_seeds:
            for record in records:
                rows.append(evaluate_buffer(record, buffer_size, buffer_seed))
            for dataset in ("mvtec", "visa_full"):
                for shot in (1, 2, 4):
                    for seed in (0, 1, 2):
                        baseline_rows.append(
                            evaluate_zero_baseline(
                                by_key[(dataset, shot, seed, 1.0)],
                                buffer_size,
                                buffer_seed,
                            )
                        )

    groups = build_groups(rows)
    baselines = build_baseline_groups(baseline_rows)
    per_class = build_per_class(rows, baseline_rows, args.selected_buffer_size, args.selected_zero_weight)
    selected = [
        group
        for group in groups
        if group["buffer_size"] == args.selected_buffer_size
        and abs(float(group["zero_weight"]) - args.selected_zero_weight) < 1e-12
    ]
    payload = {
        "definition": "normal validation buffer class-centers score-blended image scores; anomaly labels are not used for scoring",
        "selected_buffer_size": args.selected_buffer_size,
        "selected_zero_weight": args.selected_zero_weight,
        "records": rows,
        "baseline_records": baseline_rows,
        "groups": groups,
        "baseline_groups": baselines,
        "selected_groups": selected,
        "per_class": per_class,
    }
    write_json(ROOT / args.output_json, payload)
    write_markdown(ROOT / args.output_md, payload)
    print(ROOT / args.output_json)
    print(ROOT / args.output_md)


def evaluate_buffer(record: dict[str, Any], buffer_size: int, buffer_seed: int) -> dict[str, Any]:
    samples, normal_buffer = apply_buffer_centering(record["per_sample"], buffer_size, buffer_seed)
    labels = [1 if item["label"] == "anomaly" else 0 for item in samples]
    scores = [float(item["score"]) for item in samples]
    return {
        "dataset": record["dataset"],
        "shot": record["shot"],
        "seed": record["seed"],
        "zero_weight": record["zero_weight"],
        "buffer_size": buffer_size,
        "buffer_seed": buffer_seed,
        "score_mode": "buffer_center",
        "image_auroc": image_auroc(labels, scores),
        "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(samples),
        "num_eval": len(samples),
        "num_buffer": len(normal_buffer),
    }


def evaluate_zero_baseline(record: dict[str, Any], buffer_size: int, buffer_seed: int) -> dict[str, Any]:
    samples = remove_buffer_normals(record["per_sample"], buffer_size, buffer_seed)
    labels = [1 if item["label"] == "anomaly" else 0 for item in samples]
    scores = [float(item["score"]) for item in samples]
    return {
        "dataset": record["dataset"],
        "shot": record["shot"],
        "seed": record["seed"],
        "buffer_size": buffer_size,
        "buffer_seed": buffer_seed,
        "image_auroc": image_auroc(labels, scores),
        "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(samples),
        "num_eval": len(samples),
    }


def apply_buffer_centering(
    samples: list[dict[str, Any]],
    buffer_size: int,
    buffer_seed: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    by_class = group_samples(samples)
    normal_buffer: set[str] = set()
    means = {}
    for class_name, items in by_class.items():
        normals = sorted(
            [item for item in items if item["label"] == "normal"],
            key=lambda item: stable_hash(f"{buffer_seed}|{item['image_path']}"),
        )
        buffer = normals[: min(buffer_size, len(normals))]
        normal_buffer.update(str(item["image_path"]) for item in buffer)
        values = [float(item["score"]) for item in buffer]
        means[class_name] = float(np.mean(values)) if values else 0.0

    output = []
    for item in samples:
        if item["label"] == "normal" and item["image_path"] in normal_buffer:
            continue
        row = dict(item)
        row["raw_score"] = float(item["score"])
        row["score"] = float(item["score"]) - means[str(item["class_name"])]
        output.append(row)
    return output, normal_buffer


def remove_buffer_normals(samples: list[dict[str, Any]], buffer_size: int, buffer_seed: int) -> list[dict[str, Any]]:
    by_class = group_samples(samples)
    normal_buffer = set()
    for items in by_class.values():
        normals = sorted(
            [item for item in items if item["label"] == "normal"],
            key=lambda item: stable_hash(f"{buffer_seed}|{item['image_path']}"),
        )
        normal_buffer.update(str(item["image_path"]) for item in normals[: min(buffer_size, len(normals))])
    return [
        dict(item)
        for item in samples
        if not (item["label"] == "normal" and item["image_path"] in normal_buffer)
    ]


def build_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["dataset"]),
                int(record["shot"]),
                int(record["buffer_size"]),
                float(record["zero_weight"]),
            )
        ].append(record)
    rows = []
    for (dataset, shot, buffer_size, zero_weight), items in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "buffer_size": buffer_size,
                "zero_weight": zero_weight,
                "num_runs": len(items),
                "image_auroc_mean": mean_of(items, "image_auroc"),
                "image_auroc_std": std_of(items, "image_auroc"),
                "normal_fpr_at_95_tpr_mean": mean_of(items, "normal_fpr_at_95_tpr"),
                "normal_fpr_at_95_tpr_std": std_of(items, "normal_fpr_at_95_tpr"),
            }
        )
    return rows


def build_baseline_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["dataset"]), int(record["shot"]), int(record["buffer_size"]))].append(record)
    rows = []
    for (dataset, shot, buffer_size), items in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "buffer_size": buffer_size,
                "num_runs": len(items),
                "image_auroc_mean": mean_of(items, "image_auroc"),
                "image_auroc_std": std_of(items, "image_auroc"),
                "normal_fpr_at_95_tpr_mean": mean_of(items, "normal_fpr_at_95_tpr"),
                "normal_fpr_at_95_tpr_std": std_of(items, "normal_fpr_at_95_tpr"),
            }
        )
    return rows


def build_per_class(
    records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    selected_buffer_size: int,
    selected_zero_weight: float,
) -> list[dict[str, Any]]:
    # Per-class stability needs per-sample scores, so recompute it directly for the selected setting.
    source = collect_records(ROOT / "runs", [selected_zero_weight, 1.0])
    by_key = {
        (record["dataset"], record["shot"], record["seed"], record["zero_weight"]): record
        for record in source
    }
    rows = []
    for dataset in ("mvtec", "visa_full"):
        for shot in (1, 2, 4):
            deltas_by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for seed in (0, 1, 2):
                for buffer_seed in (0, 1, 2):
                    candidate, _ = apply_buffer_centering(
                        by_key[(dataset, shot, seed, selected_zero_weight)]["per_sample"],
                        selected_buffer_size,
                        buffer_seed,
                    )
                    baseline = remove_buffer_normals(
                        by_key[(dataset, shot, seed, 1.0)]["per_sample"],
                        selected_buffer_size,
                        buffer_seed,
                    )
                    candidate_by_class = group_samples(candidate)
                    baseline_by_class = group_samples(baseline)
                    for class_name, samples in candidate_by_class.items():
                        base_samples = baseline_by_class[class_name]
                        deltas_by_class[class_name].append(
                            (
                                image_metric(samples) - image_metric(base_samples),
                                normal_fpr_at_95_tpr(samples) - normal_fpr_at_95_tpr(base_samples),
                            )
                        )
            class_rows = []
            for class_name, values in sorted(deltas_by_class.items()):
                array = np.asarray(values, dtype=float)
                class_rows.append(
                    {
                        "class_name": class_name,
                        "delta_image_auroc_mean": float(array[:, 0].mean()),
                        "delta_normal_fpr_at_95_tpr_mean": float(array[:, 1].mean()),
                    }
                )
            image_positive = sum(item["delta_image_auroc_mean"] > 0 for item in class_rows)
            fpr_reduced = sum(item["delta_normal_fpr_at_95_tpr_mean"] < 0 for item in class_rows)
            worst = max(class_rows, key=lambda item: item["delta_normal_fpr_at_95_tpr_mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "buffer_size": selected_buffer_size,
                    "zero_weight": selected_zero_weight,
                    "num_classes": len(class_rows),
                    "image_positive": image_positive,
                    "fpr_reduced": fpr_reduced,
                    "worst_fpr_class": worst["class_name"],
                    "worst_fpr_delta": worst["delta_normal_fpr_at_95_tpr_mean"],
                    "classes": class_rows,
                }
            )
    return rows


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Normal Buffer Size Ablation",
        "",
        "This diagnostic asks how many target-normal validation images per class are needed to make class-wise score centering useful.",
        "For each class, `K` normal images are selected by a deterministic hash and removed from evaluation; their mean score centers all remaining images from that class.",
        "No anomaly label is used for scoring, centering, or buffer selection.",
        "",
        f"Selected diagnostic setting: K={payload['selected_buffer_size']} normal validation images per class and beta={payload['selected_zero_weight']}.",
        "",
        "## Selected Buffer Diagnostic",
        "",
        "| Dataset | Setting | Image AUROC | FPR@95TPR | Runs |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in ("mvtec", "visa_full"):
        for group in sorted([item for item in payload["selected_groups"] if item["dataset"] == dataset], key=lambda item: item["shot"]):
            lines.append(
                "| {dataset} | {shot}-shot | {image} | {fpr} | {runs} |".format(
                    dataset=display_dataset(dataset),
                    shot=group["shot"],
                    image=fmt_pm(group, "image_auroc"),
                    fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
                    runs=group["num_runs"],
                )
            )
    lines.extend(
        [
            "",
            "## Buffer Size Sweep at Beta 0.5",
            "",
            "| Dataset | Shot | K | Image AUROC | FPR@95TPR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in payload["groups"]:
        if abs(float(group["zero_weight"]) - float(payload["selected_zero_weight"])) >= 1e-12:
            continue
        lines.append(
            "| {dataset} | {shot} | {k} | {image} | {fpr} |".format(
                dataset=display_dataset(group["dataset"]),
                shot=group["shot"],
                k=group["buffer_size"],
                image=fmt_pm(group, "image_auroc"),
                fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Class Stability at K=8, Beta 0.5",
            "",
            "| Dataset | Shot | Image improved | FPR reduced | Worst FPR class |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["per_class"]:
        lines.append(
            "| {dataset} | {shot} | {image}/{total} | {fpr}/{total} | {worst} ({delta:+.2f}) |".format(
                dataset=display_dataset(row["dataset"]),
                shot=row["shot"],
                image=row["image_positive"],
                fpr=row["fpr_reduced"],
                total=row["num_classes"],
                worst=row["worst_fpr_class"],
                delta=100 * row["worst_fpr_delta"],
            )
        )
    lines.extend(
        [
            "",
            "## Main Observation",
            "",
            "- K=4 already recovers most MVTec image-AUROC gain, while K=8 gives the best MVTec FPR tradeoff among small buffers.",
            "- VisA benefits more in image ranking than FPR, reinforcing that cross-dataset score calibration remains partly unresolved.",
            "- This supports a concrete follow-up setting: strict few-shot support plus a small normal validation buffer, rather than a vague future normal-z direction.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def group_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        grouped[str(item["class_name"])].append(item)
    return grouped


def image_metric(samples: list[dict[str, Any]]) -> float:
    return image_auroc(
        [1 if item["label"] == "anomaly" else 0 for item in samples],
        [float(item["score"]) for item in samples],
    )


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def mean_of(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(mean(values)) if values else None


def std_of(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(stdev(values))


def fmt_pm(group: dict[str, Any], key: str) -> str:
    mean_value = group.get(f"{key}_mean")
    std_value = group.get(f"{key}_std")
    if mean_value is None:
        return "n/a"
    return f"{100 * float(mean_value):.2f} +/- {100 * float(std_value):.2f}"


def display_dataset(name: str) -> str:
    return {"mvtec": "MVTec", "visa_full": "VisA"}.get(name, name)


if __name__ == "__main__":
    main()
