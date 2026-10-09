from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bootstrap_significance import compare_metric_files


RUN_RE = re.compile(r".*shot(?P<shot>\d+)_seed(?P<seed>\d+).*proto(?P<proto>\d+)")
SCORE_BLEND_RE = re.compile(r"score_blend_shot(?P<shot>\d+)_seed(?P<seed>\d+)_beta(?P<beta>[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard bootstrap comparisons for the paper.")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--output-json", default="runs/statistics/bootstrap_summary.json")
    parser.add_argument("--output-csv", default="runs/statistics/bootstrap_summary.csv")
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    comparisons = build_comparisons(root)
    results = []
    for index, item in enumerate(comparisons):
        result = compare_metric_files(
            item["baseline"],
            item["candidate"],
            name=item["name"],
            baseline_topk_fraction=item.get("baseline_topk_fraction"),
            candidate_topk_fraction=item.get("candidate_topk_fraction"),
            num_bootstrap=args.num_bootstrap,
            seed=args.seed + index,
        )
        result["dataset"] = item["dataset"]
        result["variant"] = item["variant"]
        result["shot"] = item.get("shot")
        result["seed_id"] = item.get("seed_id")
        result["max_prototypes"] = item.get("max_prototypes")
        results.append(result)

    summary = summarize(results)
    payload = {"comparisons": results, "groups": summary}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(Path(args.output_csv), results)
    print(output_json)
    print(Path(args.output_csv))
    for group in summary:
        print(
            f"{group['dataset']} {group['variant']} shot={group['shot']} "
            f"image_delta={group['image_auroc_delta_mean']:.4f} "
            f"fpr_delta={group['normal_fpr_at_95_tpr_delta_mean']:.4f}"
        )


def build_comparisons(root: Path) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    mvtec_baseline = root / "zero_shot_topk0.001" / "metrics.json"
    visa_baseline = root / "visa_full" / "zero_shot_topk_sweep" / "metrics.json"

    for path in sorted((root / "mvtec_final512").glob("calib_shot*_seed*_alpha0.25_proto512/metrics.json")):
        meta = parse_run(path.parent.name)
        comparisons.append(
            {
                "dataset": "mvtec",
                "variant": "fused_vs_zero_shot",
                "name": f"mvtec_fused_shot{meta['shot']}_seed{meta['seed']}_vs_zero",
                "baseline": mvtec_baseline,
                "candidate": path,
                "shot": meta["shot"],
                "seed_id": meta["seed"],
                "max_prototypes": meta["proto"],
            }
        )

    for path in sorted((root / "score_blend" / "mvtec").glob("score_blend_shot*_seed*_beta*/metrics.json")):
        meta = parse_score_blend_run(path.parent.name)
        fused = root / "mvtec_final512" / f"calib_shot{meta['shot']}_seed{meta['seed']}_alpha0.25_proto512" / "metrics.json"
        comparisons.append(
            {
                "dataset": "mvtec",
                "variant": "score_blend_vs_zero_shot",
                "name": f"mvtec_score_blend_shot{meta['shot']}_seed{meta['seed']}_vs_zero",
                "baseline": mvtec_baseline,
                "candidate": path,
                "shot": meta["shot"],
                "seed_id": meta["seed"],
                "max_prototypes": 512,
            }
        )
        if fused.exists():
            comparisons.append(
                {
                    "dataset": "mvtec",
                    "variant": "score_blend_vs_fused",
                    "name": f"mvtec_score_blend_shot{meta['shot']}_seed{meta['seed']}_vs_fused",
                    "baseline": fused,
                    "candidate": path,
                    "shot": meta["shot"],
                    "seed_id": meta["seed"],
                    "max_prototypes": 512,
                }
            )

    for path in sorted((root / "mvtec_memory_only512_image").glob("topk_sweep_shot*_seed*_alpha0.0_proto512/metrics.json")):
        meta = parse_run(path.parent.name)
        candidate = root / "mvtec_final512" / f"calib_shot{meta['shot']}_seed{meta['seed']}_alpha0.25_proto512" / "metrics.json"
        if not candidate.exists():
            continue
        comparisons.append(
            {
                "dataset": "mvtec",
                "variant": "fused_vs_memory_only",
                "name": f"mvtec_fused_shot{meta['shot']}_seed{meta['seed']}_vs_memory",
                "baseline": path,
                "candidate": candidate,
                "baseline_topk_fraction": 0.001,
                "shot": meta["shot"],
                "seed_id": meta["seed"],
                "max_prototypes": meta["proto"],
            }
        )

    for path in sorted((root / "visa_full").glob("topk_sweep_shot*_seed*_alpha0.25_proto512/metrics.json")):
        meta = parse_run(path.parent.name)
        comparisons.append(
            {
                "dataset": "visa_full",
                "variant": "fused_vs_zero_shot",
                "name": f"visa_full_fused_shot{meta['shot']}_seed{meta['seed']}_vs_zero",
                "baseline": visa_baseline,
                "candidate": path,
                "baseline_topk_fraction": 0.001,
                "candidate_topk_fraction": 0.001,
                "shot": meta["shot"],
                "seed_id": meta["seed"],
                "max_prototypes": meta["proto"],
            }
        )

    for path in sorted((root / "score_blend" / "visa_full").glob("score_blend_shot*_seed*_beta*/metrics.json")):
        meta = parse_score_blend_run(path.parent.name)
        fused = root / "visa_full" / f"topk_sweep_shot{meta['shot']}_seed{meta['seed']}_alpha0.25_proto512" / "metrics.json"
        comparisons.append(
            {
                "dataset": "visa_full",
                "variant": "score_blend_vs_zero_shot",
                "name": f"visa_full_score_blend_shot{meta['shot']}_seed{meta['seed']}_vs_zero",
                "baseline": visa_baseline,
                "candidate": path,
                "baseline_topk_fraction": 0.001,
                "shot": meta["shot"],
                "seed_id": meta["seed"],
                "max_prototypes": 512,
            }
        )
        if fused.exists():
            comparisons.append(
                {
                    "dataset": "visa_full",
                    "variant": "score_blend_vs_fused",
                    "name": f"visa_full_score_blend_shot{meta['shot']}_seed{meta['seed']}_vs_fused",
                    "baseline": fused,
                    "candidate": path,
                    "baseline_topk_fraction": 0.001,
                    "shot": meta["shot"],
                    "seed_id": meta["seed"],
                    "max_prototypes": 512,
                }
            )

    missing = [item for item in comparisons if not Path(item["baseline"]).exists() or not Path(item["candidate"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs for {len(missing)} comparisons.")
    return comparisons


def parse_run(name: str) -> dict[str, int]:
    match = RUN_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Cannot parse run name: {name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def parse_score_blend_run(name: str) -> dict[str, int | float]:
    match = SCORE_BLEND_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Cannot parse score-blend run name: {name}")
    return {
        "shot": int(match.group("shot")),
        "seed": int(match.group("seed")),
        "beta": float(match.group("beta")),
    }


def summarize(results: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for item in results:
        key = (str(item["dataset"]), str(item["variant"]), int(item["shot"]))
        groups.setdefault(key, []).append(item)

    summary = []
    for (dataset, variant, shot), items in sorted(groups.items()):
        image_delta = [float(item["observed_delta"]["image_auroc"]) for item in items]
        fpr_delta = [float(item["observed_delta"]["normal_fpr_at_95_tpr"]) for item in items]
        summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "shot": shot,
                "num_runs": len(items),
                "image_auroc_delta_mean": mean(image_delta),
                "image_auroc_delta_min": min(image_delta),
                "image_auroc_delta_max": max(image_delta),
                "normal_fpr_at_95_tpr_delta_mean": mean(fpr_delta),
                "normal_fpr_at_95_tpr_delta_min": min(fpr_delta),
                "normal_fpr_at_95_tpr_delta_max": max(fpr_delta),
            }
        )
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_csv(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "variant",
        "shot",
        "seed_id",
        "name",
        "baseline_image_auroc",
        "candidate_image_auroc",
        "delta_image_auroc",
        "image_auroc_ci_low",
        "image_auroc_ci_high",
        "baseline_fpr",
        "candidate_fpr",
        "delta_fpr",
        "fpr_ci_low",
        "fpr_ci_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "variant": item["variant"],
                    "shot": item["shot"],
                    "seed_id": item["seed_id"],
                    "name": item["name"],
                    "baseline_image_auroc": item["baseline_metrics"]["image_auroc"],
                    "candidate_image_auroc": item["candidate_metrics"]["image_auroc"],
                    "delta_image_auroc": item["observed_delta"]["image_auroc"],
                    "image_auroc_ci_low": item["bootstrap"]["image_auroc"]["ci_low"],
                    "image_auroc_ci_high": item["bootstrap"]["image_auroc"]["ci_high"],
                    "baseline_fpr": item["baseline_metrics"]["normal_fpr_at_95_tpr"],
                    "candidate_fpr": item["candidate_metrics"]["normal_fpr_at_95_tpr"],
                    "delta_fpr": item["observed_delta"]["normal_fpr_at_95_tpr"],
                    "fpr_ci_low": item["bootstrap"]["normal_fpr_at_95_tpr"]["ci_low"],
                    "fpr_ci_high": item["bootstrap"]["normal_fpr_at_95_tpr"]["ci_high"],
                }
            )


if __name__ == "__main__":
    main()
