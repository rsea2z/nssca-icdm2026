from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final experiment summaries for paper writing.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-md", default="docs/paper_results.md")
    parser.add_argument("--output-tex", default="paper/generated/results_tables.tex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    snapshot = build_snapshot(runs_root)
    write_markdown(Path(args.output_md), snapshot)
    write_tex(Path(args.output_tex), snapshot)
    print(args.output_md)
    print(args.output_tex)


def build_snapshot(runs_root: Path) -> dict[str, Any]:
    mvtec_zero_sweep = optional_load_json(runs_root / "zero_shot_topk_sweep" / "mvtec" / "metrics.json")
    visa_zero_sweep = optional_load_json(runs_root / "zero_shot_topk_sweep" / "visa_full" / "metrics.json")
    if visa_zero_sweep is None:
        visa_zero_sweep = optional_load_json(runs_root / "visa_full" / "zero_shot_topk_sweep" / "metrics.json")

    mvtec_zero = load_json(runs_root / "zero_shot_topk0.001" / "metrics.json")
    visa_zero = find_result(load_json(runs_root / "visa_full" / "zero_shot_topk_sweep" / "metrics.json"), topk_fraction=0.001)
    score_blend = load_json(runs_root / "score_blend" / "summary.json")
    mvtec_main = [group for group in score_blend["selected_groups"] if group["dataset"] == "mvtec"]
    visa_main = [group for group in score_blend["selected_groups"] if group["dataset"] == "visa_full"]
    same_feature = {
        "mvtec_nearest": load_json(runs_root / "baselines" / "mvtec_nearest" / "topk_summary.json")["groups"],
        "mvtec_diag_gaussian": load_json(runs_root / "baselines" / "mvtec_diag_gaussian" / "topk_summary.json")[
            "groups"
        ],
        "visa_nearest": load_json(runs_root / "baselines" / "visa_full_nearest" / "topk_summary.json")["groups"],
        "visa_diag_gaussian": load_json(
            runs_root / "baselines" / "visa_full_diag_gaussian" / "topk_summary.json"
        )["groups"],
    }
    bootstrap = load_json(runs_root / "statistics" / "bootstrap_summary.json")["groups"]
    efficiency = {
        "mvtec": load_json(runs_root / "efficiency" / "mvtec_efficiency.json")["groups"],
        "visa": load_json(runs_root / "efficiency" / "visa_full_efficiency.json")["groups"],
    }
    proto1024 = {
        "mvtec": load_json(runs_root / "proto1024" / "mvtec" / "summary.json")["groups"],
        "visa": [
            group
            for group in load_json(runs_root / "proto1024" / "visa_full" / "topk_summary.json")["groups"]
            if group["topk_fraction"] == 0.001
        ],
    }
    visa_oracle = load_json(runs_root / "visa_oracle_diagnostic" / "topk_summary.json")["groups"]
    component_attribution = optional_load_json(runs_root / "component_attribution" / "summary.json")
    holdout_selection = optional_load_json(runs_root / "holdout_selection" / "mvtec" / "summary.json")

    return {
        "mvtec_zero": mvtec_zero,
        "visa_zero": visa_zero,
        "score_blend": score_blend,
        "zero_sweeps": {
            "mvtec": mvtec_zero_sweep,
            "visa_full": visa_zero_sweep,
        },
        "mvtec_main": sorted(mvtec_main, key=lambda item: item["shot"]),
        "visa_main": sorted(visa_main, key=lambda item: item["shot"]),
        "same_feature": same_feature,
        "bootstrap": sorted(bootstrap, key=lambda item: (item["dataset"], item["variant"], item["shot"])),
        "efficiency": efficiency,
        "proto1024": proto1024,
        "visa_oracle": best_visa_oracle(visa_oracle),
        "component_attribution": component_attribution,
        "holdout_selection": holdout_selection,
        "failure_diagnostics": diagnostic_files(runs_root),
    }


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def optional_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path)


def find_result(payload: dict[str, Any], **filters: Any) -> dict[str, Any]:
    for result in payload.get("results", []):
        if all(result.get(key) == value for key, value in filters.items()):
            return result
    raise KeyError(f"No result matching {filters}")


def best_visa_oracle(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for shot in [1, 2, 4]:
        candidates = [group for group in groups if group["shot"] == shot]
        best_image = max(candidates, key=lambda item: item["image_auroc_mean"])
        best_fpr = min(candidates, key=lambda item: item["normal_fpr_at_95_tpr_mean"])
        rows.append(
            {
                "shot": shot,
                "best_image": best_image,
                "best_fpr": best_fpr,
            }
        )
    return rows


def write_markdown(path: Path, snapshot: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Paper Result Snapshot")
    lines.append("")
    lines.append("Generated from `runs/` artifacts by `scripts/collect_paper_results.py`.")
    lines.append("")
    lines.append("Final rule: `alpha=0.25`, `beta=0.25`, `max_prototypes=512`, `topk_fraction=0.001`.")
    lines.append("")

    lines.append("## MVTec Main")
    lines.append("")
    lines.append("| Setting | image AUROC | pixel AUROC | PRO | FPR@95TPR |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(metric_row("Zero-shot AnomalyCLIP", snapshot["mvtec_zero"]))
    for group in snapshot["mvtec_main"]:
        lines.append(summary_row(f"{group['shot']}-shot calibration", group))
    lines.append("")

    lines.append("## Full VisA Transfer")
    lines.append("")
    lines.append("| Setting | image AUROC | pixel AUROC | PRO | FPR@95TPR |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(metric_row("Zero-shot AnomalyCLIP", snapshot["visa_zero"]))
    for group in snapshot["visa_main"]:
        lines.append(summary_row(f"{group['shot']}-shot calibration", group))
    lines.append("")

    lines.append("## Same-Feature Controls")
    lines.append("")
    lines.append("| Dataset | Control | 1-shot image/FPR | 2-shot image/FPR | 4-shot image/FPR |")
    lines.append("|---|---|---:|---:|---:|")
    for dataset, control, key in [
        ("MVTec", "Nearest", "mvtec_nearest"),
        ("MVTec", "Diag-Gaussian", "mvtec_diag_gaussian"),
        ("VisA", "Nearest", "visa_nearest"),
        ("VisA", "Diag-Gaussian", "visa_diag_gaussian"),
    ]:
        lines.append(control_row(dataset, control, snapshot["same_feature"][key]))
    lines.append("")

    write_zero_sweep_section(lines, snapshot["zero_sweeps"])
    write_component_section(lines, snapshot["component_attribution"])
    write_holdout_section(lines, snapshot["holdout_selection"])

    lines.append("## Bootstrap Deltas")
    lines.append("")
    lines.append("Positive image deltas mean the fused rule is better. Negative FPR deltas mean fewer normal false positives.")
    lines.append("")
    lines.append("| Dataset | Comparison | Shot | image AUROC delta | FPR delta |")
    lines.append("|---|---|---:|---:|---:|")
    for group in snapshot["bootstrap"]:
        lines.append(
            "| {dataset} | {variant} | {shot} | {image} | {fpr} |".format(
                dataset=group["dataset"],
                variant=group["variant"],
                shot=group["shot"],
                image=fmt(group["image_auroc_delta_mean"]),
                fpr=fmt(group["normal_fpr_at_95_tpr_delta_mean"]),
            )
        )
    lines.append("")

    lines.append("## Prototype 1024 Diagnostic")
    lines.append("")
    lines.append("| Setting | image AUROC | pixel AUROC | PRO | FPR@95TPR |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset, groups in [("MVTec", snapshot["proto1024"]["mvtec"]), ("VisA", snapshot["proto1024"]["visa"])]:
        for group in sorted(groups, key=lambda item: item["shot"]):
            lines.append(summary_row(f"{dataset} {group['shot']}-shot", group))
    lines.append("")

    lines.append("## VisA Oracle Diagnostic")
    lines.append("")
    lines.append(
        "This diagnostic tunes alpha/prototype/top-k on VisA itself. It is not a valid main-paper selection rule; it checks whether the FPR limitation is merely the MVTec-selected setting."
    )
    lines.append("")
    lines.append("| Shot | best image setting | image AUROC/FPR | best FPR setting | image AUROC/FPR |")
    lines.append("|---:|---|---:|---|---:|")
    for item in snapshot["visa_oracle"]:
        best_image = item["best_image"]
        best_fpr = item["best_fpr"]
        lines.append(
            "| {shot} | alpha={a1}, proto={p1}, topk={t1} | {m1} / {f1} | alpha={a2}, proto={p2}, topk={t2} | {m2} / {f2} |".format(
                shot=item["shot"],
                a1=best_image["alpha"],
                p1=best_image["max_prototypes"],
                t1=best_image["topk_fraction"],
                m1=fmt(best_image["image_auroc_mean"]),
                f1=fmt(best_image["normal_fpr_at_95_tpr_mean"]),
                a2=best_fpr["alpha"],
                p2=best_fpr["max_prototypes"],
                t2=best_fpr["topk_fraction"],
                m2=fmt(best_fpr["image_auroc_mean"]),
                f2=fmt(best_fpr["normal_fpr_at_95_tpr_mean"]),
            )
        )
    lines.append("")

    lines.append("## Efficiency")
    lines.append("")
    lines.append("| Dataset | Proto | memory MB | sec/image | img/s | peak MB |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset, groups in [("MVTec", snapshot["efficiency"]["mvtec"]), ("VisA", snapshot["efficiency"]["visa"])]:
        for group in sorted([item for item in groups if item["shot"] == 1], key=lambda item: item["max_prototypes"]):
            lines.append(
                "| {dataset} | {proto} | {mem} | {sec} | {ips} | {peak} |".format(
                    dataset=dataset,
                    proto=group["max_prototypes"],
                    mem=fmt(group["prototype_memory_mb_mean"], 2),
                    sec=fmt(group["seconds_per_image_mean"], 6),
                    ips=fmt(group["images_per_second_mean"], 2),
                    peak=fmt(group["cuda_peak_memory_mb_mean"], 2),
                )
            )
    lines.append("")

    write_failure_diagnostics_section(lines, snapshot["failure_diagnostics"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, snapshot: dict[str, Any]) -> None:
    lines = [
        "% Generated by scripts/collect_paper_results.py",
        "\\newcommand{\\MvtecZeroImage}{%s}" % fmt(snapshot["mvtec_zero"]["image_auroc"]),
        "\\newcommand{\\MvtecZeroFpr}{%s}" % fmt(snapshot["mvtec_zero"]["normal_fpr_at_95_tpr"]),
        "\\newcommand{\\VisaZeroImage}{%s}" % fmt(snapshot["visa_zero"]["image_auroc"]),
        "\\newcommand{\\VisaZeroFpr}{%s}" % fmt(snapshot["visa_zero"]["normal_fpr_at_95_tpr"]),
    ]
    for prefix, groups in [("Mvtec", snapshot["mvtec_main"]), ("Visa", snapshot["visa_main"])]:
        for group in groups:
            shot = group["shot"]
            lines.extend(
                [
                    "\\newcommand{\\%sShot%sImage}{%s}" % (prefix, shot, fmt_pm(group, "image_auroc")),
                    "\\newcommand{\\%sShot%sPixel}{%s}" % (prefix, shot, fmt_pm(group, "pixel_auroc")),
                    "\\newcommand{\\%sShot%sPro}{%s}" % (prefix, shot, fmt_pm(group, "pro_auc")),
                    "\\newcommand{\\%sShot%sFpr}{%s}" % (prefix, shot, fmt_pm(group, "normal_fpr_at_95_tpr")),
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    return "| {label} | {image} | {pixel} | {pro} | {fpr} |".format(
        label=label,
        image=fmt(metrics["image_auroc"]),
        pixel=fmt(metrics["pixel_auroc"]),
        pro=fmt(metrics["pro_auc"]),
        fpr=fmt(metrics["normal_fpr_at_95_tpr"]),
    )


def summary_row(label: str, group: dict[str, Any]) -> str:
    return "| {label} | {image} | {pixel} | {pro} | {fpr} |".format(
        label=label,
        image=fmt_pm(group, "image_auroc"),
        pixel=fmt_pm(group, "pixel_auroc"),
        pro=fmt_pm(group, "pro_auc"),
        fpr=fmt_pm(group, "normal_fpr_at_95_tpr"),
    )


def control_row(dataset: str, control: str, groups: list[dict[str, Any]]) -> str:
    by_shot = {group["shot"]: group for group in groups if group["topk_fraction"] == 0.001}
    return "| {dataset} | {control} | {s1} | {s2} | {s4} |".format(
        dataset=dataset,
        control=control,
        s1=image_fpr(by_shot[1]),
        s2=image_fpr(by_shot[2]),
        s4=image_fpr(by_shot[4]),
    )


def image_fpr(group: dict[str, Any]) -> str:
    return f"{fmt(group['image_auroc_mean'])} / {fmt(group['normal_fpr_at_95_tpr_mean'])}"


def write_zero_sweep_section(lines: list[str], sweeps: dict[str, Any | None]) -> None:
    rows = []
    for dataset, payload in sweeps.items():
        if not payload:
            continue
        for result in payload.get("results", []):
            rows.append((dataset, result))
    if not rows:
        return

    lines.append("## Zero-Shot Top-k Sweep")
    lines.append("")
    lines.append(
        "This isolates the pooling effect from prototype calibration. Pixel metrics are top-k invariant; missing entries mean the supplemental run skipped recomputing them."
    )
    lines.append("")
    lines.append("| Dataset | top-k | image AUROC | pixel AUROC | PRO | FPR@95TPR |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset, result in sorted(rows, key=lambda item: (item[0], float(item[1]["topk_fraction"]))):
        lines.append(
            "| {dataset} | {topk} | {image} | {pixel} | {pro} | {fpr} |".format(
                dataset=dataset,
                topk=result["topk_fraction"],
                image=fmt(result.get("image_auroc")),
                pixel=fmt(result.get("pixel_auroc")),
                pro=fmt(result.get("pro_auc")),
                fpr=fmt(result.get("normal_fpr_at_95_tpr")),
            )
        )
    lines.append("")


def write_component_section(lines: list[str], payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    groups = [group for group in payload.get("groups", []) if abs(float(group["topk_fraction"]) - 0.001) < 1e-12]
    if not groups:
        return

    lines.append("## Component Attribution")
    lines.append("")
    lines.append("Rows use the final `alpha=0.25`, 512-prototype setting and top-0.1% pooling.")
    lines.append("")
    lines.append("| Dataset | Component | 1-shot image/FPR | 2-shot image/FPR | 4-shot image/FPR |")
    lines.append("|---|---|---:|---:|---:|")
    for dataset in sorted({str(group["dataset"]) for group in groups}):
        dataset_groups = [group for group in groups if str(group["dataset"]) == dataset]
        for mode in ["zero_shot_topk", "prototype_only", "fused", "other_class_fused"]:
            mode_groups = [group for group in dataset_groups if group["mode"] == mode]
            if mode_groups:
                lines.append(component_row(dataset, mode, mode_groups))
    lines.append("")


def component_row(dataset: str, mode: str, groups: list[dict[str, Any]]) -> str:
    by_shot = {int(group["shot"]): group for group in groups}
    return "| {dataset} | {mode} | {s1} | {s2} | {s4} |".format(
        dataset=dataset,
        mode=mode.replace("_", " "),
        s1=image_fpr_or_na(by_shot.get(1)),
        s2=image_fpr_or_na(by_shot.get(2)),
        s4=image_fpr_or_na(by_shot.get(4)),
    )


def image_fpr_or_na(group: dict[str, Any] | None) -> str:
    if group is None:
        return "NA"
    return image_fpr(group)


def write_holdout_section(lines: list[str], payload: dict[str, Any] | None) -> None:
    if not payload:
        return

    lines.append("## Holdout-Class Hyperparameter Selection")
    lines.append("")
    lines.append(
        "MVTec classes are split into held-in development classes and held-out evaluation classes. This is a diagnostic for top-k/prototype selection sensitivity, not the main no-retuning VisA protocol."
    )
    lines.append("")
    lines.append("| Shot | folds | heldout image AUROC | heldout FPR@95TPR | dev image AUROC | dev FPR@95TPR |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for group in payload.get("groups", []):
        lines.append(
            "| {shot} | {folds} | {himg} +/- {himgstd} | {hfpr} +/- {hfprstd} | {dimg} | {dfpr} |".format(
                shot=group["shot"],
                folds=group["num_folds"],
                himg=fmt(group.get("heldout_image_auroc_mean")),
                himgstd=fmt(group.get("heldout_image_auroc_std")),
                hfpr=fmt(group.get("heldout_fpr_mean")),
                hfprstd=fmt(group.get("heldout_fpr_std")),
                dimg=fmt(group.get("dev_image_auroc_mean")),
                dfpr=fmt(group.get("dev_fpr_mean")),
            )
        )
    lines.append("")
    counts = payload.get("selection_counts", [])
    if counts:
        lines.append("Selection counts:")
        lines.append("")
        lines.append("| alpha | prototypes | top-k | count |")
        lines.append("|---:|---:|---:|---:|")
        for item in counts:
            lines.append(
                "| {alpha} | {proto} | {topk} | {count} |".format(
                    alpha=item["alpha"],
                    proto=item["max_prototypes"],
                    topk=item["topk_fraction"],
                    count=item["count"],
                )
            )
        lines.append("")


def write_failure_diagnostics_section(lines: list[str], files: list[str]) -> None:
    if not files:
        return
    lines.append("## Failure Diagnostic Artifacts")
    lines.append("")
    for file in files:
        lines.append(f"- `{file}`")
    lines.append("")


def diagnostic_files(runs_root: Path) -> list[str]:
    figures_root = runs_root.parent / "figures" / "failure_diagnostics"
    if not figures_root.exists():
        return []
    return sorted(str(path.as_posix()) for path in figures_root.glob("*") if path.is_file())


def fmt_pm(group: dict[str, Any], key: str) -> str:
    mean = group.get(f"{key}_mean")
    std = group.get(f"{key}_std")
    if mean is None:
        return "NA"
    if std is None:
        return fmt(mean)
    return f"{fmt(mean)} +/- {fmt(std)}"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
