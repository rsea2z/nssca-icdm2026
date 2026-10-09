from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot score and FPR failure diagnostics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-topk-fraction", type=float, default=None)
    parser.add_argument("--candidate-topk-fraction", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_per_sample(Path(args.baseline), args.baseline_topk_fraction)
    candidate = load_per_sample(Path(args.candidate), args.candidate_topk_fraction)
    paired = align_by_image_path(baseline, candidate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sample_csv(output_dir / f"{args.dataset}_paired_scores.csv", paired)
    rows = per_class_rows(paired)
    write_rows_csv(output_dir / f"{args.dataset}_per_class_fpr_delta.csv", rows)
    plot_histogram(output_dir / f"{args.dataset}_score_histogram.png", args.dataset, paired)
    plot_fpr_delta(output_dir / f"{args.dataset}_fpr_delta.png", args.dataset, rows)
    print(output_dir)


def load_per_sample(path: Path, topk_fraction: float | None) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload:
        if topk_fraction is None:
            if len(payload["results"]) != 1:
                raise ValueError(f"{path} is a sweep; provide --baseline-topk-fraction or --candidate-topk-fraction.")
            return payload["results"][0]["per_sample"]
        for result in payload["results"]:
            if abs(float(result["topk_fraction"]) - float(topk_fraction)) < 1e-12:
                return result["per_sample"]
        raise ValueError(f"topk_fraction={topk_fraction} not found in {path}.")
    if "per_sample" not in payload:
        raise KeyError(f"{path} does not contain per_sample.")
    return payload["per_sample"]


def align_by_image_path(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_by_path = {str(item["image_path"]): item for item in baseline}
    cand_by_path = {str(item["image_path"]): item for item in candidate}
    common = sorted(set(base_by_path) & set(cand_by_path))
    if not common:
        raise ValueError("No overlapping image_path values.")
    paired = []
    for image_path in common:
        base = base_by_path[image_path]
        cand = cand_by_path[image_path]
        paired.append(
            {
                "image_path": image_path,
                "class_name": base["class_name"],
                "label": base["label"],
                "baseline_score": float(base["score"]),
                "candidate_score": float(cand["score"]),
                "score_delta": float(cand["score"]) - float(base["score"]),
            }
        )
    return paired


def write_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_class_rows(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in paired:
        by_class[str(item["class_name"])].append(item)
    rows = []
    for class_name, items in sorted(by_class.items()):
        baseline_fpr = normal_fpr_at_95_tpr(items, "baseline_score")
        candidate_fpr = normal_fpr_at_95_tpr(items, "candidate_score")
        rows.append(
            {
                "class_name": class_name,
                "num_samples": len(items),
                "baseline_fpr": baseline_fpr,
                "candidate_fpr": candidate_fpr,
                "fpr_delta": None if baseline_fpr is None or candidate_fpr is None else candidate_fpr - baseline_fpr,
                "normal_score_delta_mean": mean_score_delta(items, "normal"),
                "anomaly_score_delta_mean": mean_score_delta(items, "anomaly"),
            }
        )
    return rows


def normal_fpr_at_95_tpr(items: list[dict[str, object]], key: str) -> float | None:
    scores = np.asarray([float(item[key]) for item in items], dtype=float)
    labels = np.asarray([1 if str(item["label"]).lower() == "anomaly" else 0 for item in items], dtype=int)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return None
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


def mean_score_delta(items: list[dict[str, object]], label: str) -> float | None:
    values = [float(item["score_delta"]) for item in items if str(item["label"]).lower() == label]
    return float(np.mean(values)) if values else None


def plot_histogram(path: Path, dataset: str, paired: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal = [item for item in paired if str(item["label"]).lower() == "normal"]
    anomaly = [item for item in paired if str(item["label"]).lower() == "anomaly"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), constrained_layout=True)
    for axis, items, title in ((axes[0], normal, "Normal"), (axes[1], anomaly, "Anomaly")):
        axis.hist([float(item["baseline_score"]) for item in items], bins=40, alpha=0.55, label="zero-shot")
        axis.hist([float(item["candidate_score"]) for item in items], bins=40, alpha=0.55, label="fused")
        axis.set_title(f"{dataset} {title}")
        axis.set_xlabel("image score")
        axis.set_ylabel("count")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fpr_delta(path: Path, dataset: str, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in rows if row["fpr_delta"] is not None]
    rows = sorted(rows, key=lambda row: float(row["fpr_delta"]))
    names = [str(row["class_name"]) for row in rows]
    deltas = [float(row["fpr_delta"]) for row in rows]
    colors = ["#2f6f5f" if value <= 0 else "#b34b4b" for value in deltas]
    height = max(3.0, 0.28 * len(rows))
    fig, axis = plt.subplots(figsize=(7.0, height), constrained_layout=True)
    axis.barh(names, deltas, color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_title(f"{dataset} per-class FPR delta")
    axis.set_xlabel("candidate FPR - zero-shot FPR")
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
