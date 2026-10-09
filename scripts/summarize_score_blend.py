from __future__ import annotations

import argparse
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

from fewshot_ad.evaluation import image_auroc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize zero/fused score-level blend diagnostics.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-json", default="runs/score_blend/summary.json")
    parser.add_argument("--output-md", default="docs/score_blend_validation.md")
    parser.add_argument("--zero-weight", type=float, default=0.25)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = ROOT / args.runs_root
    records = collect_records(runs_root, args.weights)
    groups = build_groups(records)
    selected = [group for group in groups if abs(group["zero_weight"] - args.zero_weight) < 1e-12]
    per_class = build_per_class_summary(records, args.zero_weight)
    write_selected_metric_files(runs_root, records, args.zero_weight)
    light_records = [{key: value for key, value in record.items() if key != "per_sample"} for record in records]
    payload = {
        "definition": "score = zero_weight * zero_shot_topk_score + (1 - zero_weight) * fused_map_topk_score",
        "selected_zero_weight": args.zero_weight,
        "records": light_records,
        "groups": groups,
        "selected_groups": selected,
        "per_class": per_class,
    }
    write_json(ROOT / args.output_json, payload)
    write_markdown(ROOT / args.output_md, payload)
    print(ROOT / args.output_json)
    print(ROOT / args.output_md)


def collect_records(runs_root: Path, weights: list[float]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in ("mvtec", "visa_full"):
        zero_samples = load_zero_samples(runs_root, dataset)
        for shot in (1, 2, 4):
            for seed in (0, 1, 2):
                calibrated = load_calibrated_result(runs_root, dataset, shot, seed)
                calibrated_samples = calibrated["per_sample"]
                for zero_weight in weights:
                    samples = blend_samples(zero_samples, calibrated_samples, zero_weight)
                    records.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "zero_weight": zero_weight,
                            "image_auroc": image_metric(samples),
                            "pixel_auroc": calibrated.get("pixel_auroc"),
                            "pro_auc": calibrated.get("pro_auc"),
                            "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(samples),
                            "num_test_evaluated": len(samples),
                            "per_sample": samples,
                        }
                    )
    return records


def load_zero_samples(runs_root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    path = runs_root / "zero_shot_topk_sweep" / dataset / "metrics.json"
    payload = load_json(path)
    result = find_result(payload["results"], topk_fraction=0.001)
    return {str(item["image_path"]): item for item in result["per_sample"]}


def load_calibrated_result(runs_root: Path, dataset: str, shot: int, seed: int) -> dict[str, Any]:
    if dataset == "mvtec":
        path = (
            runs_root
            / "mvtec_final512"
            / f"calib_shot{shot}_seed{seed}_alpha0.25_proto512"
            / "metrics.json"
        )
        return load_json(path)

    path = (
        runs_root
        / "visa_full"
        / f"topk_sweep_shot{shot}_seed{seed}_alpha0.25_proto512"
        / "metrics.json"
    )
    payload = load_json(path)
    return find_result(payload["results"], topk_fraction=0.001)


def find_result(results: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    for result in results:
        if all(result.get(key) == value for key, value in filters.items()):
            return result
    raise KeyError(f"No result matching {filters}")


def blend_samples(
    zero_samples: dict[str, dict[str, Any]],
    calibrated_samples: list[dict[str, Any]],
    zero_weight: float,
) -> list[dict[str, Any]]:
    blended = []
    for item in calibrated_samples:
        zero = zero_samples[str(item["image_path"])]
        score = zero_weight * float(zero["score"]) + (1.0 - zero_weight) * float(item["score"])
        blended.append(
            {
                "image_path": item["image_path"],
                "class_name": item["class_name"],
                "label": item["label"],
                "score": score,
                "zero_score": float(zero["score"]),
                "fused_score": float(item["score"]),
            }
        )
    return blended


def build_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["dataset"]), int(record["shot"]), float(record["zero_weight"]))
        by_group[key].append(record)

    groups = []
    for (dataset, shot, zero_weight), items in sorted(by_group.items()):
        groups.append(
            {
                "dataset": dataset,
                "shot": shot,
                "zero_weight": zero_weight,
                "num_runs": len(items),
                "image_auroc_mean": mean_of(items, "image_auroc"),
                "image_auroc_std": std_of(items, "image_auroc"),
                "pixel_auroc_mean": mean_of(items, "pixel_auroc"),
                "pixel_auroc_std": std_of(items, "pixel_auroc"),
                "pro_auc_mean": mean_of(items, "pro_auc"),
                "pro_auc_std": std_of(items, "pro_auc"),
                "normal_fpr_at_95_tpr_mean": mean_of(items, "normal_fpr_at_95_tpr"),
                "normal_fpr_at_95_tpr_std": std_of(items, "normal_fpr_at_95_tpr"),
            }
        )
    return groups


def build_per_class_summary(records: list[dict[str, Any]], zero_weight: float) -> list[dict[str, Any]]:
    rows = []
    selected = [record for record in records if abs(float(record["zero_weight"]) - zero_weight) < 1e-12]
    for dataset in ("mvtec", "visa_full"):
        zero_samples = load_zero_samples(ROOT / "runs", dataset)
        for shot in (1, 2, 4):
            deltas_by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for record in selected:
                if record["dataset"] != dataset or record["shot"] != shot:
                    continue
                candidate_by_class = group_samples(record["per_sample"])
                zero_by_class = group_samples([zero_samples[item["image_path"]] for item in record["per_sample"]])
                for class_name, samples in candidate_by_class.items():
                    candidate_image = image_metric(samples)
                    candidate_fpr = normal_fpr_at_95_tpr(samples)
                    zero_image = image_metric(zero_by_class[class_name])
                    zero_fpr = normal_fpr_at_95_tpr(zero_by_class[class_name])
                    if any(value is None for value in (candidate_fpr, zero_fpr)):
                        continue
                    deltas_by_class[class_name].append((candidate_image - zero_image, candidate_fpr - zero_fpr))
            class_rows = []
            for class_name, deltas in sorted(deltas_by_class.items()):
                values = np.asarray(deltas, dtype=float)
                class_rows.append(
                    {
                        "class_name": class_name,
                        "delta_image_auroc_mean": float(values[:, 0].mean()),
                        "delta_normal_fpr_at_95_tpr_mean": float(values[:, 1].mean()),
                    }
                )
            image_positive = sum(row["delta_image_auroc_mean"] > 0 for row in class_rows)
            fpr_reduced = sum(row["delta_normal_fpr_at_95_tpr_mean"] < 0 for row in class_rows)
            worst = max(class_rows, key=lambda item: item["delta_normal_fpr_at_95_tpr_mean"])
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "num_classes": len(class_rows),
                    "image_positive": image_positive,
                    "fpr_reduced": fpr_reduced,
                    "worst_fpr_class": worst["class_name"],
                    "worst_fpr_delta": worst["delta_normal_fpr_at_95_tpr_mean"],
                    "classes": class_rows,
                }
            )
    return rows


def write_selected_metric_files(runs_root: Path, records: list[dict[str, Any]], zero_weight: float) -> None:
    beta_text = format_weight(zero_weight)
    for record in records:
        if abs(float(record["zero_weight"]) - zero_weight) >= 1e-12:
            continue
        output = (
            runs_root
            / "score_blend"
            / str(record["dataset"])
            / f"score_blend_shot{record['shot']}_seed{record['seed']}_beta{beta_text}"
            / "metrics.json"
        )
        payload = {
            "score_mode": "zero_fused_blend",
            "zero_weight": record["zero_weight"],
            "shot": record["shot"],
            "seed": record["seed"],
            "image_auroc": record["image_auroc"],
            "pixel_auroc": record["pixel_auroc"],
            "pro_auc": record["pro_auc"],
            "normal_fpr_at_95_tpr": record["normal_fpr_at_95_tpr"],
            "num_test_evaluated": record["num_test_evaluated"],
            "per_sample": record["per_sample"],
        }
        write_json(output, payload)


def format_weight(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def group_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        grouped[str(item["class_name"])].append(item)
    return grouped


def image_metric(samples: list[dict[str, Any]]) -> float:
    labels = np.asarray([1 if item.get("label") == "anomaly" else 0 for item in samples], dtype=int)
    scores = np.asarray([float(item["score"]) for item in samples], dtype=float)
    return image_auroc(labels, scores)


def normal_fpr_at_95_tpr(samples: list[dict[str, Any]]) -> float:
    labels = np.asarray([1 if item.get("label") == "anomaly" else 0 for item in samples], dtype=int)
    scores = np.asarray([float(item["score"]) for item in samples], dtype=float)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return float("nan")
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_groups"]
    lines = [
        "# Score-Blend Validation",
        "",
        "This diagnostic keeps the fused anomaly map for pixel metrics, but changes the image score to",
        "",
        "`score = beta * zero_shot_topk_score + (1 - beta) * fused_map_topk_score`,",
        "",
        f"with `beta={payload['selected_zero_weight']}` selected by the MVTec sweep and frozen for VisA.",
        "No VisA labels are used to choose the blend weight.",
        "",
        "## Selected Blend",
        "",
        "| Dataset | Setting | Image AUROC | Pixel AUROC | PRO | FPR@95TPR | Runs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("mvtec", "visa_full"):
        for group in sorted([item for item in selected if item["dataset"] == dataset], key=lambda item: item["shot"]):
            lines.append(
                "| {dataset} | {shot}-shot | {image} | {pixel} | {pro} | {fpr} | {runs} |".format(
                    dataset=display_dataset(dataset),
                    shot=group["shot"],
                    image=fmt_pm(group, "image_auroc"),
                    pixel=fmt_pm(group, "pixel_auroc"),
                    pro=fmt_pm(group, "pro_auc"),
                    fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
                    runs=group["num_runs"],
                )
            )
    lines.extend(
        [
            "",
            "## Blend-Weight Sweep",
            "",
            "| Dataset | Shot | beta | Image AUROC | FPR@95TPR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for group in payload["groups"]:
        lines.append(
            "| {dataset} | {shot} | {beta:.2f} | {image} | {fpr} |".format(
                dataset=display_dataset(group["dataset"]),
                shot=group["shot"],
                beta=group["zero_weight"],
                image=fmt_pm(group, "image_auroc"),
                fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Class Stability At Selected Blend",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_pm(group: dict[str, Any], key: str) -> str:
    mean_value = group.get(f"{key}_mean")
    std_value = group.get(f"{key}_std")
    if mean_value is None:
        return "n/a"
    if std_value is None:
        return f"{100 * float(mean_value):.2f}"
    return f"{100 * float(mean_value):.2f} +/- {100 * float(std_value):.2f}"


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


def display_dataset(name: str) -> str:
    return {"mvtec": "MVTec", "visa_full": "VisA"}.get(name, name)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
