from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


DATASETS = {
    "mvtec": {
        "seed_dir": "seeds_mvtec",
        "data_link": Path("anomaly_detection/mvtec_anomaly_detection"),
        "train_glob": "train/good/*.png",
    },
    "visa": {
        "seed_dir": "seeds_visa",
        "data_link": Path("DATA/anomaly_detection/VisA_pytorch/1cls"),
        "train_glob": "train/good/*.*",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PromptAD seed files from this project's few-shot splits.")
    parser.add_argument("--promptad-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    promptad_root = Path(args.promptad_root)
    project_root = Path(args.project_root)
    data_root = Path(args.data_root)
    splits_dir = project_root / args.splits_dir
    config = DATASETS[args.dataset]

    ensure_link(promptad_root / config["data_link"], data_root)
    class_to_lines: dict[str, list[str]] = {}
    for shot in args.shots:
        split_path = splits_dir / f"shot_{shot}_seed_{args.seed}.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        by_class: dict[str, list[str]] = {}
        for item in payload["calibration"]:
            by_class.setdefault(item["class_name"], []).append(item["image_path"])
        for class_name, paths in sorted(by_class.items()):
            train_paths = promptad_train_paths(data_root, class_name, config["train_glob"])
            index_by_path = {normalize(path): index for index, path in enumerate(train_paths)}
            indices = []
            for image_path in paths:
                key = normalize(Path(image_path))
                if key not in index_by_path:
                    raise KeyError(f"{image_path} not found in PromptAD train order for {class_name}.")
                indices.append(index_by_path[key])
            class_to_lines.setdefault(class_name, []).append(f"#{shot}: {' '.join(str(index) for index in indices)}")

    seed_root = promptad_root / "datasets" / config["seed_dir"]
    for class_name, lines in class_to_lines.items():
        class_dir = seed_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "selected_samples_per_run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"prepared {args.dataset} seed files for seed {args.seed}")


def ensure_link(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return
        if link_path.is_symlink():
            link_path.unlink()
        else:
            return
    link_path.symlink_to(target.resolve(), target_is_directory=True)


def promptad_train_paths(data_root: Path, class_name: str, pattern: str) -> list[Path]:
    paths = [Path(path) for path in glob.glob(str(data_root / class_name / pattern))]
    if not paths:
        raise FileNotFoundError(f"No PromptAD train images found for {class_name} under {data_root}.")
    return paths


def normalize(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


if __name__ == "__main__":
    main()
