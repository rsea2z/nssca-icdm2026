from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print reproducible commands for the few-shot experiment grid.")
    parser.add_argument("--splits-dir", default="splits/mvtec")
    parser.add_argument("--feature-manifest", default="features/manifest.jsonl")
    parser.add_argument("--output-dir", default="runs/mvtec_grid")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-prototypes", nargs="+", type=int, default=[64, 256])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("# Zero-shot baseline")
    for seed in args.seeds:
        split = Path(args.splits_dir) / f"shot_1_seed_{seed}.json"
        output = Path(args.output_dir) / f"zero_shot_seed_{seed}" / "metrics.json"
        print(
            "python scripts\\evaluate_feature_baseline.py "
            f"--split {split} "
            f"--feature-manifest {args.feature_manifest} "
            f"--output {output}"
        )

    print("\n# Few-shot normal prototype calibration")
    for shot, seed, alpha, max_prototypes in product(args.shots, args.seeds, args.alphas, args.max_prototypes):
        split = Path(args.splits_dir) / f"shot_{shot}_seed_{seed}.json"
        run_name = f"shot{shot}_seed{seed}_alpha{alpha:g}_proto{max_prototypes}"
        output = Path(args.output_dir) / run_name / "metrics.json"
        print(
            "python scripts\\run_feature_calibration.py "
            f"--split {split} "
            f"--feature-manifest {args.feature_manifest} "
            f"--output {output} "
            f"--alpha {alpha:g} "
            f"--max-prototypes {max_prototypes} "
            f"--seed {seed}"
        )


if __name__ == "__main__":
    main()

