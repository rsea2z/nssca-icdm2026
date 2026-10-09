from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fewshot_ad.datasets import index_mvtec, index_visa, make_fewshot_split, save_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normal-only few-shot calibration splits.")
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--root", required=True, help="Dataset root path.")
    parser.add_argument("--output", required=True, help="Output directory for split JSON files.")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--split-csv", default=None, help="Optional VisA split CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "mvtec":
        records = index_mvtec(args.root, classes=args.classes)
    else:
        records = index_visa(args.root, classes=args.classes, split_csv=args.split_csv)

    output_dir = Path(args.output)
    for shot in args.shots:
        split = make_fewshot_split(records, shots=shot, seed=args.seed, classes=args.classes)
        output_path = output_dir / f"shot_{shot}_seed_{args.seed}.json"
        save_split(split, output_path)
        meta = split["meta"]
        print(
            f"wrote {output_path} | classes={len(meta['classes'])} "
            f"calibration={sum(meta['per_class_counts'].values())} test={meta['num_test']}"
        )


if __name__ == "__main__":
    main()
