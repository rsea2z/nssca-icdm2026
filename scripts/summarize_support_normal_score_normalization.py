from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize support-only normal score normalization runs.")
    parser.add_argument("--root", default="runs/support_normal_score")
    parser.add_argument("--output-json", default="runs/support_normal_score/summary.json")
    parser.add_argument("--output-md", default="docs/support_normal_score_normalization.md")
    parser.add_argument("--selected-mode", default="support_center")
    parser.add_argument("--selected-zero-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records(ROOT / args.root)
    groups = build_groups(records)
    selected = [
        group
        for group in groups
        if group["score_mode"] == args.selected_mode
        and abs(float(group["zero_weight"]) - args.selected_zero_weight) < 1e-12
    ]
    payload = {
        "definition": "support-only class centering/z-normalization of score-blended image scores",
        "selected_mode": args.selected_mode,
        "selected_zero_weight": args.selected_zero_weight,
        "records": records,
        "groups": groups,
        "selected_groups": selected,
    }
    write_json(ROOT / args.output_json, payload)
    write_markdown(ROOT / args.output_md, payload)
    print(ROOT / args.output_json)
    print(ROOT / args.output_md)


def collect_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("**/metrics.json")):
        dataset = dataset_from_path(path)
        payload = load_json(path)
        shot, seed = parse_shot_seed(path)
        for result in payload["results"]:
            records.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "seed": seed,
                    "score_mode": result["score_mode"],
                    "zero_weight": result["zero_weight"],
                    "topk_fraction": result["topk_fraction"],
                    "num_calibration": payload["num_calibration"],
                    "num_test_evaluated": payload["num_test_evaluated"],
                    "image_auroc": result["image_auroc"],
                    "pixel_auroc": result.get("pixel_auroc"),
                    "pro_auc": result.get("pro_auc"),
                    "normal_fpr_at_95_tpr": result["normal_fpr_at_95_tpr"],
                    "path": str(path),
                }
            )
    return records


def dataset_from_path(path: Path) -> str:
    parts = path.parts
    for name in ("mvtec", "visa_full"):
        if name in parts:
            return name
    return "unknown"


def parse_shot_seed(path: Path) -> tuple[int, int]:
    for part in path.parts:
        if part.startswith("support_normal_shot"):
            match = re.search(r"support_normal_shot(?P<shot>\d+)_seed(?P<seed>\d+)", part)
            if match:
                return int(match.group("shot")), int(match.group("seed"))
    raise ValueError(f"Cannot parse shot/seed from {path}")


def build_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset"]),
            int(record["shot"]),
            str(record["score_mode"]),
            float(record["zero_weight"]),
        )
        by_group[key].append(record)

    groups = []
    for (dataset, shot, mode, zero_weight), items in sorted(by_group.items()):
        groups.append(
            {
                "dataset": dataset,
                "shot": shot,
                "score_mode": mode,
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


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Support-Only Normal Score Normalization",
        "",
        "This experiment uses only the 1/2/4 normal support images to estimate class-wise score offsets.",
        "For 1-shot stability, the selected support-only rule subtracts the class support mean and keeps a global score scale; no anomaly label is used for scoring or normalization.",
        "",
        f"Selected setting: `{payload['selected_mode']}`, `beta={payload['selected_zero_weight']}`.",
        "",
        "## Selected Support-Only Variant",
        "",
        "| Dataset | Setting | Image AUROC | Pixel AUROC | PRO | FPR@95TPR | Runs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("mvtec", "visa_full"):
        rows = sorted([item for item in payload["selected_groups"] if item["dataset"] == dataset], key=lambda item: item["shot"])
        for group in rows:
            lines.append(result_row(dataset, group))

    lines.extend(
        [
            "",
            "## Sweep",
            "",
            "| Dataset | Shot | Mode | beta | Image AUROC | FPR@95TPR |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for group in payload["groups"]:
        lines.append(
            "| {dataset} | {shot} | {mode} | {beta:.2f} | {image} | {fpr} |".format(
                dataset=display_dataset(str(group["dataset"])),
                shot=group["shot"],
                mode=group["score_mode"],
                beta=group["zero_weight"],
                image=fmt_pm(group, "image_auroc"),
                fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_row(dataset: str, group: dict[str, Any]) -> str:
    return "| {dataset} | {shot}-shot | {image} | {pixel} | {pro} | {fpr} | {runs} |".format(
        dataset=display_dataset(dataset),
        shot=group["shot"],
        image=fmt_pm(group, "image_auroc"),
        pixel=fmt_pm(group, "pixel_auroc"),
        pro=fmt_pm(group, "pro_auc"),
        fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
        runs=group["num_runs"],
    )


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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
