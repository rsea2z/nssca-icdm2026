from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fixed-threshold false-positive mass movement.")
    parser.add_argument("--output-csv", default="docs/fp_mass_decomposition.csv")
    parser.add_argument("--output-md", default="docs/fp_mass_decomposition.md")
    parser.add_argument("--topk-fraction", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [
        (
            "MVTec",
            ROOT / "runs/zero_shot_topk0.001/metrics.json",
            ROOT / "runs/mvtec_final512/calib_shot{shot}_seed{seed}_alpha0.25_proto512/metrics.json",
        ),
        (
            "VisA",
            ROOT / "runs/visa_full/zero_shot_topk_sweep/metrics.json",
            ROOT / "runs/visa_full/topk_sweep_shot{shot}_seed{seed}_alpha0.25_proto512/metrics.json",
        ),
    ]

    rows: list[dict[str, object]] = []
    for dataset, zero_path, candidate_pattern in specs:
        zero_samples = load_samples(zero_path, args.topk_fraction)
        threshold = threshold_at_95_tpr(zero_samples)
        for shot in (1, 2, 4):
            for seed in (0, 1, 2):
                candidate_path = Path(str(candidate_pattern).format(shot=shot, seed=seed))
                if not candidate_path.exists():
                    continue
                candidate_samples = load_samples(candidate_path, args.topk_fraction)
                rows.append(compute_row(dataset, shot, seed, threshold, zero_samples, candidate_samples, candidate_path))

    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)
    print(Path(args.output_csv))
    print(Path(args.output_md))


def load_samples(path: Path, topk_fraction: float) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_sample = payload.get("per_sample")
    if per_sample is None:
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"{path} does not contain per_sample or results.")
        matches = [
            result
            for result in results
            if result.get("topk_fraction") is not None and abs(float(result["topk_fraction"]) - topk_fraction) < 1e-12
        ]
        if not matches:
            raise ValueError(f"{path} does not contain topk_fraction={topk_fraction}.")
        per_sample = matches[0].get("per_sample")
    if per_sample is None:
        raise ValueError(f"{path} selected result does not contain per_sample.")
    return {str(item["image_path"]): item for item in per_sample}


def threshold_at_95_tpr(samples: dict[str, dict[str, object]]) -> float:
    anomaly_scores = [
        float(item["score"])
        for item in samples.values()
        if str(item["label"]).strip().lower() not in {"good", "normal", "0", "false"}
    ]
    if not anomaly_scores:
        raise ValueError("No anomaly scores found for zero-shot threshold.")
    return float(np.quantile(np.asarray(anomaly_scores, dtype=float), 0.05))


def compute_row(
    dataset: str,
    shot: int,
    seed: int,
    zero_threshold: float,
    zero_samples: dict[str, dict[str, object]],
    candidate_samples: dict[str, dict[str, object]],
    candidate_path: Path,
) -> dict[str, object]:
    common = sorted(set(zero_samples) & set(candidate_samples))
    normal_paths = [
        path
        for path in common
        if str(zero_samples[path]["label"]).strip().lower() in {"good", "normal", "0", "false"}
    ]
    if not normal_paths:
        raise ValueError(f"No normal samples overlap for {candidate_path}.")

    corrected = introduced = persistent = stable = 0
    for path in normal_paths:
        base_fp = float(zero_samples[path]["score"]) >= zero_threshold
        cand_fp = float(candidate_samples[path]["score"]) >= zero_threshold
        if base_fp and not cand_fp:
            corrected += 1
        elif not base_fp and cand_fp:
            introduced += 1
        elif base_fp and cand_fp:
            persistent += 1
        else:
            stable += 1

    total = len(normal_paths)
    return {
        "dataset": dataset,
        "shot": shot,
        "seed": seed,
        "zero_threshold": zero_threshold,
        "num_normals": total,
        "corrected_zero_fp": corrected,
        "introduced_new_fp": introduced,
        "persistent_fp": persistent,
        "stable_true_negative": stable,
        "corrected_rate": corrected / total,
        "introduced_rate": introduced / total,
        "net_fixed_threshold_fp_delta": (introduced - corrected) / total,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["shot"]))].append(row)

    lines = [
        "# False-Positive Mass Decomposition",
        "",
        "This diagnostic fixes the zero-shot 95%-TPR threshold and counts how normal samples move under the fused score.",
        "`corrected_zero_fp` means a zero-shot false positive becomes non-FP at the fixed zero-shot threshold; `introduced_new_fp` means the reverse.",
        "It complements, but does not replace, FPR@95TPR because FPR@95TPR recomputes a threshold per scoring rule.",
        "",
        "| Dataset | Shot | Corrected rate | Introduced rate | Net FP delta | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, shot), items in sorted(grouped.items()):
        corrected = mean(items, "corrected_rate")
        introduced = mean(items, "introduced_rate")
        net = mean(items, "net_fixed_threshold_fp_delta")
        lines.append(
            f"| {dataset} | {shot} | {pct(corrected)} | {pct(introduced)} | {pct(net)} | {len(items)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Run Rows",
            "",
            "| Dataset | Shot | Seed | Corrected | Introduced | Persistent | Stable | Net FP delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {shot} | {seed} | {corrected_zero_fp} | {introduced_new_fp} | "
            "{persistent_fp} | {stable_true_negative} | {net} |".format(
                dataset=row["dataset"],
                shot=row["shot"],
                seed=row["seed"],
                corrected_zero_fp=row["corrected_zero_fp"],
                introduced_new_fp=row["introduced_new_fp"],
                persistent_fp=row["persistent_fp"],
                stable_true_negative=row["stable_true_negative"],
                net=pct(float(row["net_fixed_threshold_fp_delta"])),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def pct(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


if __name__ == "__main__":
    main()
