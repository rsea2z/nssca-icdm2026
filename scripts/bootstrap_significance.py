from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.evaluation import image_auroc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison for image-level metrics.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--baseline-topk-fraction", type=float, default=None)
    parser.add_argument("--candidate-topk-fraction", type=float, default=None)
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_metric_files(
        Path(args.baseline),
        Path(args.candidate),
        name=args.name,
        baseline_topk_fraction=args.baseline_topk_fraction,
        candidate_topk_fraction=args.candidate_topk_fraction,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        ci=args.ci,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def compare_metric_files(
    baseline_path: Path,
    candidate_path: Path,
    *,
    name: str | None = None,
    baseline_topk_fraction: float | None = None,
    candidate_topk_fraction: float | None = None,
    num_bootstrap: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, object]:
    baseline = load_per_sample(baseline_path, topk_fraction=baseline_topk_fraction)
    candidate = load_per_sample(candidate_path, topk_fraction=candidate_topk_fraction)
    paired = align_samples(baseline, candidate)
    result = paired_bootstrap(
        paired,
        num_bootstrap=num_bootstrap,
        seed=seed,
        ci=ci,
    )
    return {
        "name": name or f"{candidate_path.parent.name}_vs_{baseline_path.parent.name}",
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_topk_fraction": baseline_topk_fraction,
        "candidate_topk_fraction": candidate_topk_fraction,
        "num_samples": len(paired["labels"]),
        "num_bootstrap": num_bootstrap,
        "seed": seed,
        "ci": ci,
        **result,
    }


def load_per_sample(path: Path, *, topk_fraction: float | None = None) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_sample = payload.get("per_sample")
    if per_sample:
        return list(per_sample)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} has neither per_sample nor results.")
    if topk_fraction is None and len(results) == 1:
        selected = results[0]
    else:
        matches = [
            result
            for result in results
            if result.get("topk_fraction") is not None
            and topk_fraction is not None
            and abs(float(result["topk_fraction"]) - float(topk_fraction)) < 1e-12
        ]
        if not matches:
            raise ValueError(f"{path} does not contain topk_fraction={topk_fraction}.")
        selected = matches[0]

    per_sample = selected.get("per_sample")
    if not per_sample:
        raise ValueError(f"{path} selected result does not contain per_sample.")
    return list(per_sample)


def align_samples(
    baseline: Iterable[dict[str, object]],
    candidate: Iterable[dict[str, object]],
) -> dict[str, np.ndarray]:
    baseline_by_path = {str(item["image_path"]): item for item in baseline}
    candidate_by_path = {str(item["image_path"]): item for item in candidate}
    common = sorted(set(baseline_by_path) & set(candidate_by_path))
    if not common:
        raise ValueError("No overlapping image_path values between baseline and candidate.")

    labels = []
    baseline_scores = []
    candidate_scores = []
    for image_path in common:
        base = baseline_by_path[image_path]
        cand = candidate_by_path[image_path]
        base_label = label_to_int(base.get("label"))
        cand_label = label_to_int(cand.get("label"))
        if base_label != cand_label:
            raise ValueError(f"Label mismatch for {image_path}: {base_label} != {cand_label}")
        labels.append(base_label)
        baseline_scores.append(float(base["score"]))
        candidate_scores.append(float(cand["score"]))

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "baseline_scores": np.asarray(baseline_scores, dtype=np.float64),
        "candidate_scores": np.asarray(candidate_scores, dtype=np.float64),
    }


def label_to_int(label: object) -> int:
    value = str(label).strip().lower()
    return 0 if value in {"good", "normal", "0", "false"} else 1


def paired_bootstrap(
    paired: dict[str, np.ndarray],
    *,
    num_bootstrap: int,
    seed: int,
    ci: float,
) -> dict[str, object]:
    labels = paired["labels"]
    base_scores = paired["baseline_scores"]
    cand_scores = paired["candidate_scores"]

    base_metrics = compute_metrics(labels, base_scores)
    cand_metrics = compute_metrics(labels, cand_scores)
    observed_delta = {
        key: cand_metrics[key] - base_metrics[key]
        for key in base_metrics
    }

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("Bootstrap requires both positive and negative samples.")

    rng = np.random.default_rng(seed)
    draws = {key: [] for key in base_metrics}
    for _ in range(num_bootstrap):
        sampled_pos = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        sampled_neg = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        sample_labels = labels[sampled]
        sample_base = compute_metrics(sample_labels, base_scores[sampled])
        sample_cand = compute_metrics(sample_labels, cand_scores[sampled])
        for key in draws:
            draws[key].append(sample_cand[key] - sample_base[key])

    alpha = (1.0 - ci) / 2.0
    intervals = {}
    p_values = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=np.float64)
        intervals[key] = {
            "mean_delta": float(np.mean(arr)),
            "std_delta": float(np.std(arr, ddof=1)),
            "ci_low": float(np.quantile(arr, alpha)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        }
        obs = float(observed_delta[key])
        if obs >= 0:
            tail = float(np.mean(arr <= 0.0))
        else:
            tail = float(np.mean(arr >= 0.0))
        p_values[key] = min(1.0, 2.0 * tail)

    return {
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "observed_delta": observed_delta,
        "bootstrap": intervals,
        "two_sided_p_value_vs_zero": p_values,
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "image_auroc": image_auroc(labels, scores),
        "normal_fpr_at_95_tpr": normal_fpr_at_95_tpr(labels, scores),
    }


def normal_fpr_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    anomaly_scores = scores[labels == 1]
    normal_scores = scores[labels == 0]
    if anomaly_scores.size == 0 or normal_scores.size == 0:
        return float("nan")
    threshold = float(np.quantile(anomaly_scores, 0.05))
    return float(np.mean(normal_scores >= threshold))


if __name__ == "__main__":
    main()
