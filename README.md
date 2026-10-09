# Training-Free Normal-Support Score Calibration for Frozen Vision-Language Anomaly Detection

Official code release for the ICDM 2026 Applied Track paper
**Training-Free Normal-Support Score Calibration for Frozen Vision-Language Anomaly Detection**.

This repository contains the calibration/evaluation code, configuration templates,
selected result snapshots, validation reports, and commands used to produce the paper
results. It does not redistribute either benchmark dataset or the external AnomalyCLIP
weights; all public data and external checkpoints are obtained from their official
sources.

## Repository Layout

```text
README.md                         installation, data, commands, and result map
requirements-reproducibility.txt  tested Python dependencies
configs/                          public path templates and final calibration settings
docs/                             validation, provenance, and reproducibility-checklist reports
scripts/                          export, calibration, evaluation, and summary code
src/fewshot_ad/                   reusable calibration and evaluation library
runs/                             selected metrics, summaries, and statistics
tests/                            unit and synthetic smoke tests
MANIFEST.sha256                   SHA-256 inventory of repository files
```

Raw images, exported `.npz` features, external model repositories, and model weights are
intentionally excluded to keep the repository lightweight and to respect upstream
licenses. The commands below regenerate them from public sources.

## Tested Environment

- Python: 3.10.9
- Core Python packages are pinned in `requirements-reproducibility.txt`.
- PyTorch 2.7.0+cu128 was used for CUDA acceleration and AnomalyCLIP feature export.
- CPU evaluation is supported by the calibration code; use `--device cpu`.

Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-reproducibility.txt
python -m pip install -e .[dev]
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/smoke_test.py
```

The last two commands should report 5 passing tests and synthetic image/pixel/PRO scores
near 1.0. AnomalyCLIP feature export additionally requires the dependencies specified by
the official AnomalyCLIP repository.

## Datasets and Preprocessing

### MVTec AD

1. Download MVTec AD from <https://www.mvtec.com/company/research/datasets/mvtec-ad>.
2. Set `MVTec_ROOT` below to the directory containing the 15 category directories.
3. Build deterministic 1/2/4-shot normal-support splits for seeds 0, 1, and 2:

```bash
for seed in 0 1 2; do
  python scripts/build_fewshot_splits.py \
    --dataset mvtec --root "$MVTec_ROOT" --output splits/mvtec \
    --shots 1 2 4 --seed "$seed"
done
```

The split JSON files contain only paths and labels. Calibration images are drawn from
the normal training split; all labeled test images are used for evaluation. No test image
is moved into calibration.

### VisA

1. Download VisA through the official project page
   <https://github.com/amazon-science/spot-diff> or AWS Open Data registry
   <https://registry.opendata.aws/visa/>.
2. Convert it to the MVTec-style layout expected by the loader, or use an existing
   Anomalib conversion. The expected root contains `<category>/train/good`,
   `<category>/test/good`, `<category>/test/bad`, and `ground_truth` masks.
3. Build the VisA splits:

```bash
for seed in 0 1 2; do
  python scripts/build_fewshot_splits.py \
    --dataset visa --root "$VISA_ROOT" --output splits/visa \
    --shots 1 2 4 --seed "$seed"
done
```

The paper uses all 12 VisA categories and 2,162 test images. The optional
`scripts/prepare_visa_hf.py` can convert the Hugging Face parquet distribution to the
MVTec-style tree.

## Frozen AnomalyCLIP Feature Export

Clone the official AnomalyCLIP repository into `external/AnomalyCLIP-main` and follow its
instructions to obtain the released prompt checkpoints:

- MVTec export uses `checkpoints/9_12_4_multiscale/epoch_15.pth`.
- VisA export uses `checkpoints/9_12_4_multiscale_visa/epoch_15.pth`.
- CLIP ViT-L/14@336px is downloaded automatically or can be placed in `--clip-cache-dir`.

Example MVTec export:

```bash
python scripts/build_anomalyclip_meta.py \
  --dataset mvtec --root "$MVTec_ROOT"

python scripts/export_anomalyclip_features.py \
  --dataset mvtec --data-path "$MVTec_ROOT" \
  --output-dir features/mvtec_anomalyclip \
  --manifest features/mvtec_manifest.jsonl \
  --anomalyclip-root external/AnomalyCLIP-main \
  --checkpoint-path external/AnomalyCLIP-main/checkpoints/9_12_4_multiscale/epoch_15.pth \
  --clip-cache-dir external/clip_cache \
  --splits splits/mvtec/shot_1_seed_0.json splits/mvtec/shot_2_seed_0.json splits/mvtec/shot_4_seed_0.json
```

Repeat with `--dataset visa`, the VisA root, VisA checkpoint, and VisA splits. Each
record is stored as one `.npz` file with:

| Key | Required | Meaning |
|---|---:|---|
| `features` | yes | `(H, W, D)` or `(N, D)` frozen patch features |
| `base_map` | yes | zero-shot AnomalyCLIP anomaly map |
| `mask` | yes for test anomalies | pixel ground truth |
| `image_path` | no | source image path |
| `class_name` | no | category name |

The JSONL manifest maps each `image_path` to its exported `feature_path`.

## Final Settings

All proposed-method rows use:

- fusion weight `alpha = 0.25`
- zero-shot score-blend weight `beta = 0.25`
- prototype budget `M = 512`
- image pooling fraction `rho = 0.001`
- calibration seeds `{0, 1, 2}`
- support sizes `{1, 2, 4}`
- 50 uniformly spaced PRO thresholds in main tables
- 200-threshold PRO in the robustness audit

Hyperparameter ranges examined in the stored diagnostics were:

- `alpha`: `{0.00, 0.25, 0.50, 0.75, 1.00}`
- prototype budget: `{32, 64, 128, 256, 512}`, with a separate 1,024 diagnostic
- score-blend `beta`: `{0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00}`
- top-`k`: `{0.001, 0.005, 0.010, 0.020, 0.050}`

The final values were selected by a labeled MVTec sweep, then frozen for VisA. A
class-heldout MVTec selection diagnostic and support-only normal-z audit are included as
boundary checks. The paper explicitly states that this protocol is not fully
anomaly-free hyperparameter selection.

## Reproducing the Main Results

The shell variables below assume the split and manifest paths from the previous steps.

### Zero-shot baselines

```bash
python scripts/run_zero_shot_topk_sweep.py \
  --split splits/mvtec/shot_1_seed_0.json \
  --feature-manifest features/mvtec_manifest.jsonl \
  --output runs/zero_shot_topk_sweep/mvtec/metrics.json \
  --topk-fractions 0.001 0.005 0.01 0.02 0.05

python scripts/run_zero_shot_topk_sweep.py \
  --split splits/visa/shot_1_seed_0.json \
  --feature-manifest features/visa_manifest.jsonl \
  --output runs/zero_shot_topk_sweep/visa_full/metrics.json \
  --topk-fractions 0.001 0.005 0.01 0.02 0.05
```

### MVTec fused calibration

For every `shot in 1 2 4` and `seed in 0 1 2`:

```bash
python scripts/run_feature_calibration.py \
  --split "splits/mvtec/shot_${shot}_seed_${seed}.json" \
  --feature-manifest features/mvtec_manifest.jsonl \
  --output "runs/mvtec_final512/calib_shot${shot}_seed${seed}_alpha0.25_proto512/metrics.json" \
  --alpha 0.25 --max-prototypes 512 --topk-fraction 0.001 \
  --pro-num-thresholds 50 --device cuda
```

### Full VisA transfer

For every `shot in 1 2 4` and `seed in 0 1 2`:

```bash
python scripts/run_topk_sweep.py \
  --split "splits/visa/shot_${shot}_seed_${seed}.json" \
  --feature-manifest features/visa_manifest.jsonl \
  --output "runs/visa_full/topk_sweep_shot${shot}_seed${seed}_alpha0.25_proto512/metrics.json" \
  --alpha 0.25 --max-prototypes 512 \
  --topk-fractions 0.001 0.005 0.01 0.02 0.05 \
  --compute-pixel-metrics --device cuda
```

### Conservative score blend

This command reads the zero-shot and fused-map per-sample scores, evaluates the complete
beta sweep, writes selected beta=0.25 metrics, and creates the paper summary:

```bash
python scripts/summarize_score_blend.py \
  --runs-root runs \
  --output-json runs/score_blend/summary.json \
  --output-md docs/score_blend_validation.md \
  --zero-weight 0.25 \
  --weights 0.0 0.1 0.25 0.5 0.75 0.9 1.0
```

### Statistics, PRO audit, efficiency, and tables

```bash
python scripts/run_bootstrap_suite.py \
  --root runs --num-bootstrap 2000 --seed 0 \
  --output-json runs/statistics/bootstrap_summary.json \
  --output-csv runs/statistics/bootstrap_summary.csv

python scripts/summarize_dense_pro.py \
  --root runs/pro_dense200 \
  --output docs/dense_pro_validation.md

python scripts/profile_calibration_efficiency.py \
  --splits-dir splits/mvtec \
  --feature-manifest features/mvtec_manifest.jsonl \
  --output-json runs/efficiency/mvtec_efficiency.json \
  --output-csv runs/efficiency/mvtec_efficiency.csv \
  --max-prototypes 128 256 512 1024 --device cuda

python scripts/collect_paper_results.py \
  --runs-root runs \
  --output-md docs/paper_results.md \
  --output-tex paper/generated/results_tables.tex
```

The 200-threshold PRO snapshots in `runs/pro_dense200/` use the same final settings with
`--pro-num-thresholds 200`. They can be regenerated by replacing the PRO argument in the
calibration commands and preserving the directory names used by
`scripts/summarize_dense_pro.py`.

## Result Snapshot

Values are percentages. `FPR` is normal FPR@95TPR. Mean and standard deviation are over
three support seeds.

| Dataset | Setting | Image AUROC | Pixel AUROC | PRO | FPR |
|---|---|---:|---:|---:|---:|
| MVTec | Zero-shot | 86.34 | 90.29 | 30.22 | 65.52 |
| MVTec | 1-shot | 89.68 +/- 0.14 | 89.64 +/- 0.39 | 79.19 +/- 2.22 | 49.61 +/- 0.12 |
| MVTec | 2-shot | 89.78 +/- 0.20 | 89.90 +/- 0.27 | 77.85 +/- 1.04 | 51.53 +/- 1.22 |
| MVTec | 4-shot | 89.89 +/- 0.36 | 90.02 +/- 0.16 | 76.13 +/- 1.06 | 47.47 +/- 3.33 |
| VisA | Zero-shot | 73.67 | 92.63 | 68.38 | 79.00 |
| VisA | 1-shot | 80.97 +/- 0.12 | 93.34 +/- 0.36 | 75.43 +/- 8.22 | 76.26 +/- 0.42 |
| VisA | 2-shot | 80.72 +/- 0.43 | 93.47 +/- 0.49 | 75.90 +/- 7.99 | 74.77 +/- 0.52 |
| VisA | 4-shot | 80.70 +/- 0.24 | 93.21 +/- 0.13 | 74.31 +/- 9.31 | 77.62 +/- 1.04 |

The corresponding raw snapshots are:

- `runs/mvtec_final512/summary.json`
- `runs/score_blend/summary.json`
- `runs/statistics/bootstrap_summary.json`
- `runs/pro_dense200/`
- `runs/efficiency/`
- `runs/failure_analysis/`
- `docs/paper_results.md`

## Metrics

- **Image AUROC**: tie-aware rank AUROC over image-level anomaly scores.
- **Pixel AUROC**: tie-aware rank AUROC over all flattened pixels and masks.
- **PRO**: area under mean connected-component overlap versus pixel false-positive rate,
  normalized by `max_fpr=0.30`. Main tables use 50 thresholds; the audit uses 200.
- **FPR@95TPR**: fraction of normal images scoring at or above the score threshold whose
  anomaly true-positive rate is 95%. It is an evaluation diagnostic, not a deployment
  threshold-selection rule.

## Runtime and Run Counts

- Proposed calibration: no gradient update, prompt training, backbone update, or test-time
  optimization.
- Main calibration evaluations: 18 runs (2 datasets x 3 support sizes x 3 seeds).
- Zero-shot baselines: 2 full-dataset runs.
- Dense-PRO audit snapshots: 20 runs (2 zero-shot + 18 calibrated).
- Paired bootstrap: 2,000 resamples per comparison, seed 0.
- At 512 prototypes on MVTec, the stored efficiency snapshot reports 22.50 MiB prototype
  memory and approximately 345 images/s after feature export on the GPU listed above.

## Notes on External Baselines

The proposed method is compared under two regimes:

1. **Same-feature controls**: zero-shot top-k, prototype-only, diagonal Gaussian, fused,
   and wrong-class controls all consume the same exported AnomalyCLIP features.
2. **Trained prompt-adaptation context**: public PromptAD code is run separately. Its
   original protocol and training cost are retained; the comparison is explicitly a
   deployment boundary rather than a claim that the training-free head surpasses it.

`docs/baseline_provenance.md`, `docs/promptad_direct_baseline.md`, and
`docs/promptad_runtime_context.md` record the relevant provenance and caveats.

## License and Attribution

This package is supplied for reproducibility of the ICDM 2026 paper. MVTec AD and VisA
remain governed by their respective upstream licenses. AnomalyCLIP and PromptAD remain
governed by their upstream repositories. Please cite the paper and the original datasets
and methods when using this package.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026trainingfree,
  title     = {Training-Free Normal-Support Score Calibration for Frozen Vision-Language Anomaly Detection},
  author    = {Liu, Junjie and Wang, Jingnan and Zhang, Boqiang and Li, Kunyu and Zhang, Lichao},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

## License

This repository is released under the [MIT License](LICENSE).
Third-party artifacts (AnomalyCLIP, CLIP, MVTec AD, VisA) remain under their own
licenses and are not redistributed here.
