from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


EVENT_RE = re.compile(
    r"^\[(?P<event>start|done)\]\s+"
    r"(?P<timestamp>\S+)\s+"
    r"(?P<dataset>\S+)\s+shot=(?P<shot>\d+)\s+seed=(?P<seed>\d+)\s+class=(?P<class_name>\S+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize wall-clock runtime from PromptAD aligned-run logs.")
    parser.add_argument("--log-dir", default="runs/logs")
    parser.add_argument("--output-json", default="runs/strong_baselines/promptad/runtime_summary.json")
    parser.add_argument("--output-md", default="docs/promptad_runtime_context.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_logs(Path(args.log_dir))
    summary = summarize(rows)
    write_json(Path(args.output_json), rows, summary)
    write_markdown(Path(args.output_md), rows, summary)
    print(args.output_json)
    print(args.output_md)


def parse_logs(log_dir: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], dict[str, object]] = {}
    for path in sorted(log_dir.glob("promptad_aligned*.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = EVENT_RE.match(line.strip())
            if not match:
                continue
            key = (
                match.group("dataset"),
                int(match.group("shot")),
                int(match.group("seed")),
            )
            item = grouped.setdefault(
                key,
                {
                    "dataset": key[0],
                    "shot": key[1],
                    "seed": key[2],
                    "log_files": set(),
                    "start_times": [],
                    "done_times": [],
                    "classes": set(),
                },
            )
            item["log_files"].add(path.name)
            timestamp = datetime.fromisoformat(match.group("timestamp"))
            if match.group("event") == "start":
                item["start_times"].append(timestamp)
            else:
                item["done_times"].append(timestamp)
                item["classes"].add(match.group("class_name"))

    rows = []
    for item in grouped.values():
        start_times = item["start_times"]
        done_times = item["done_times"]
        if not start_times or not done_times:
            continue
        start_time = min(start_times)
        end_time = max(done_times)
        rows.append(
            {
                "dataset": item["dataset"],
                "shot": item["shot"],
                "seed": item["seed"],
                "num_classes": len(item["classes"]),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "wall_clock_hours": (end_time - start_time).total_seconds() / 3600.0,
                "log_files": sorted(item["log_files"]),
            }
        )
    return sorted(rows, key=lambda row: (str(row["dataset"]), int(row["shot"]), int(row["seed"])))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["shot"]))].append(float(row["wall_clock_hours"]))

    summary = []
    for (dataset, shot), values in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "shot": shot,
                "num_runs": len(values),
                "wall_clock_hours_mean": mean(values),
                "wall_clock_hours_std": pstdev(values) if len(values) > 1 else 0.0,
                "wall_clock_hours_min": min(values),
                "wall_clock_hours_max": max(values),
            }
        )
    return summary


def write_json(path: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "groups": summary}, indent=2), encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    lines = [
        "# PromptAD Runtime Context",
        "",
        "Generated from local aligned-run logs in `runs/logs/promptad_aligned*.log`.",
        "",
        "These are wall-clock durations for our public-code aligned reruns, not canonical numbers from the PromptAD paper.",
        "",
        "## Aggregate",
        "",
        "| Dataset | Shot | Runs | Wall-clock hours |",
        "|---|---:|---:|---:|",
    ]
    for group in summary:
        lines.append(
            "| {dataset} | {shot} | {runs} | {mean:.2f} +/- {std:.2f} |".format(
                dataset=group["dataset"],
                shot=group["shot"],
                runs=group["num_runs"],
                mean=group["wall_clock_hours_mean"],
                std=group["wall_clock_hours_std"],
            )
        )

    lines.extend(
        [
            "",
            "## Per Run",
            "",
            "| Dataset | Shot | Seed | Classes | Wall-clock hours |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {shot} | {seed} | {classes} | {hours:.2f} |".format(
                dataset=row["dataset"],
                shot=row["shot"],
                seed=row["seed"],
                classes=row["num_classes"],
                hours=row["wall_clock_hours"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
