from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_MAIN_VALUES = [
    "86.34",
    "90.29",
    "30.22",
    "65.52",
    "89.68",
    "89.64",
    "79.19",
    "49.61",
    "89.78",
    "89.90",
    "77.85",
    "51.53",
    "89.89",
    "90.02",
    "76.13",
    "47.47",
    "73.67",
    "92.63",
    "68.38",
    "79.00",
    "80.97",
    "93.34",
    "75.43",
    "76.26",
    "80.72",
    "93.47",
    "75.90",
    "74.77",
    "80.70",
    "93.21",
    "74.31",
    "77.62",
    "92.76",
    "94.50",
    "23.22",
    "93.19",
    "95.01",
    "21.78",
    "94.08",
    "95.41",
    "19.63",
    "87.73",
    "95.55",
    "45.45",
]


def main() -> None:
    paper_results = read("docs/paper_results.md")
    generated = read("paper/generated/results_tables.tex")
    manuscript = read("paper_icme/main.tex")
    promptad = read("docs/promptad_direct_baseline.md")
    dense_pro = read("docs/dense_pro_validation.md")
    score_blend = read("docs/score_blend_validation.md")
    normal_z = read("docs/normal_score_normalization.md")
    normal_buffer = read("docs/normal_buffer_ablation.md")
    lightweight = read("docs/lightweight_neighbor_boundary.md")

    check_generated_macros(generated, paper_results)
    check_main_values(manuscript)
    check_score_blend_values(manuscript, score_blend)
    check_per_class_counts(manuscript)
    check_promptad_values(manuscript, promptad)
    check_dense_pro_values(manuscript, dense_pro)
    check_normal_z_values(manuscript, normal_z)
    check_normal_buffer_values(manuscript, normal_buffer)
    check_lightweight_neighbor_values(manuscript, lightweight)
    print("ICME consistency check passed.")


def check_generated_macros(generated: str, paper_results: str) -> None:
    macros = dict(re.findall(r"\\newcommand\{\\([^}]+)\}\{([^}]+)\}", generated))
    required = {
        "MvtecZeroImage": "0.8634",
        "MvtecZeroFpr": "0.6552",
        "VisaZeroImage": "0.7367",
        "VisaZeroFpr": "0.7900",
        "MvtecShot1Image": "0.8968 +/- 0.0014",
        "MvtecShot4Fpr": "0.4747 +/- 0.0333",
        "VisaShot1Image": "0.8097 +/- 0.0012",
        "VisaShot4Fpr": "0.7762 +/- 0.0104",
    }
    for key, value in required.items():
        actual = macros.get(key)
        if actual != value:
            raise AssertionError(f"{key} mismatch: expected {value}, got {actual}")
        if value not in paper_results:
            raise AssertionError(f"{value} from {key} not found in docs/paper_results.md")


def check_main_values(manuscript: str) -> None:
    missing = [value for value in EXPECTED_MAIN_VALUES if value not in manuscript]
    if missing:
        raise AssertionError(f"main.tex is missing expected result values: {', '.join(missing)}")


def check_promptad_values(manuscript: str, promptad: str) -> None:
    for value in ("0.9276", "0.9450", "0.2322", "0.8773", "0.9555", "0.4545"):
        if value not in promptad:
            raise AssertionError(f"PromptAD direct baseline doc missing {value}")
    for value in ("92.76", "94.50", "23.22", "87.73", "95.55", "45.45"):
        if value not in manuscript:
            raise AssertionError(f"PromptAD value {value} missing from main.tex")


def check_score_blend_values(manuscript: str, score_blend: str) -> None:
    if "docs/score_blend_validation.md" not in manuscript:
        raise AssertionError("main.tex does not reference the score-blend validation artifact.")
    for value in (
        "89.68",
        "89.78",
        "89.89",
        "49.61",
        "51.53",
        "47.47",
        "80.97",
        "80.72",
        "80.70",
        "76.26",
        "74.77",
        "77.62",
    ):
        if value not in score_blend:
            raise AssertionError(f"Score-blend validation doc missing {value}")
        if value not in manuscript:
            raise AssertionError(f"Score-blend value {value} missing from main.tex")


def check_dense_pro_values(manuscript: str, dense_pro: str) -> None:
    if "docs/dense_pro_validation.md" not in manuscript:
        raise AssertionError("main.tex does not reference dense PRO validation artifact.")
    rows = parse_markdown_tables(dense_pro)
    main_rows = [row for row in rows if "PRO@200" in row and "Setting" in row]
    delta_rows = [row for row in rows if "Delta PRO@200" in row and "Setting" in row]
    values = {(row["Dataset"], row["Setting"]): row for row in main_rows}
    expected = {
        ("MVTec", "zero-shot"): 56.48,
        ("MVTec", "1-shot"): 83.40,
        ("MVTec", "2-shot"): 83.30,
        ("MVTec", "4-shot"): 84.28,
        ("VisA", "zero-shot"): 78.66,
        ("VisA", "1-shot"): 81.44,
        ("VisA", "2-shot"): 82.15,
        ("VisA", "4-shot"): 83.55,
    }
    for key, expected_pro in expected.items():
        row = values.get(key)
        if row is None:
            raise AssertionError(f"dense PRO table missing row: {key}")
        assert_close(parse_metric_cell(row["PRO@200"]), expected_pro, f"dense PRO {key}")
    deltas = {(row["Dataset"], row["Setting"]): row for row in delta_rows}
    for dataset in ("MVTec", "VisA"):
        zero = parse_metric_cell(values[(dataset, "zero-shot")]["PRO@200"])
        for shot in ("1-shot", "2-shot", "4-shot"):
            row = deltas.get((dataset, shot))
            if row is None:
                raise AssertionError(f"dense PRO delta table missing row: {(dataset, shot)}")
            observed_delta = parse_metric_cell(row["Delta PRO@200"])
            expected_delta = parse_metric_cell(values[(dataset, shot)]["PRO@200"]) - zero
            assert_close(observed_delta, expected_delta, f"dense PRO delta {(dataset, shot)}", tolerance=0.02)


def check_normal_z_values(manuscript: str, normal_z: str) -> None:
    for value in ("91.77", "91.93", "83.73", "83.84", "72.66", "76.26"):
        if value not in normal_z:
            raise AssertionError(f"normal score normalization doc missing {value}")


def check_normal_buffer_values(manuscript: str, normal_buffer: str) -> None:
    if "docs/normal_buffer_ablation.md" not in manuscript:
        raise AssertionError("main.tex does not reference normal buffer ablation artifact.")
    for value in ("91.44", "49.09", "50.98", "82.75", "83.03", "75.37", "76.35"):
        if value not in normal_buffer:
            raise AssertionError(f"normal buffer ablation doc missing {value}")
        if value not in manuscript:
            raise AssertionError(f"normal buffer ablation value {value} missing from main.tex")


def check_lightweight_neighbor_values(manuscript: str, lightweight: str) -> None:
    if "docs/lightweight_neighbor_boundary.md" not in manuscript:
        raise AssertionError("main.tex does not reference the lightweight neighbor boundary artifact.")
    for term in ("PA-CLIP", "MRAD", "ProtoAD", "GCR"):
        if term not in lightweight:
            raise AssertionError(f"lightweight neighbor boundary missing {term}")
    for value in ("82.05", "70.45", "69.31", "74.09", "89.68", "49.61", "80.97", "76.26"):
        if value not in lightweight:
            raise AssertionError(f"lightweight neighbor boundary missing value {value}")


def check_per_class_counts(manuscript: str) -> None:
    payload = load_json("runs/score_blend/summary.json")
    display = {"mvtec": "MVTec", "visa_full": "VisA"}
    for row in payload["per_class"]:
        dataset = display[row["dataset"]]
        expected = (
            f"{dataset} & {row['shot']} & "
            f"{row['image_positive']}/{row['num_classes']} & "
            f"{row['fpr_reduced']}/{row['num_classes']}"
        )
        if expected not in manuscript:
            raise AssertionError(f"main.tex missing per-class stability row: {expected}")
    for artifact in ("docs/score_blend_validation.md", "runs/score_blend/summary.json"):
        escaped = artifact.replace("_", r"\_")
        if artifact not in manuscript and escaped not in manuscript:
            raise AssertionError(f"main.tex missing artifact path: {artifact}")


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def load_json(relative_path: str) -> dict:
    import json

    return json.loads(read(relative_path))


def parse_markdown_tables(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            headers = None
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if headers is None:
            headers = cells
            continue
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def parse_metric_cell(cell: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
    if not match:
        raise AssertionError(f"Could not parse metric cell: {cell!r}")
    return float(match.group(0))


def assert_close(actual: float, expected: float, label: str, tolerance: float = 0.01) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label} mismatch: expected {expected:.2f}, got {actual:.2f}")


if __name__ == "__main__":
    main()
