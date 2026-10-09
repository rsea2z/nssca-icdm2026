# Paper Result Snapshot

Generated from `runs/` artifacts by `scripts/collect_paper_results.py`.

Final rule: `alpha=0.25`, `beta=0.25`, `max_prototypes=512`, `topk_fraction=0.001`.

## MVTec Main

| Setting | image AUROC | pixel AUROC | PRO | FPR@95TPR |
|---|---:|---:|---:|---:|
| Zero-shot AnomalyCLIP | 0.8634 | 0.9029 | 0.3022 | 0.6552 |
| 1-shot calibration | 0.8968 +/- 0.0014 | 0.8964 +/- 0.0039 | 0.7919 +/- 0.0222 | 0.4961 +/- 0.0012 |
| 2-shot calibration | 0.8978 +/- 0.0020 | 0.8990 +/- 0.0027 | 0.7785 +/- 0.0104 | 0.5153 +/- 0.0122 |
| 4-shot calibration | 0.8989 +/- 0.0036 | 0.9002 +/- 0.0016 | 0.7613 +/- 0.0106 | 0.4747 +/- 0.0333 |

## Full VisA Transfer

| Setting | image AUROC | pixel AUROC | PRO | FPR@95TPR |
|---|---:|---:|---:|---:|
| Zero-shot AnomalyCLIP | 0.7367 | 0.9263 | 0.6838 | 0.7900 |
| 1-shot calibration | 0.8097 +/- 0.0012 | 0.9334 +/- 0.0036 | 0.7543 +/- 0.0822 | 0.7626 +/- 0.0042 |
| 2-shot calibration | 0.8072 +/- 0.0043 | 0.9347 +/- 0.0049 | 0.7590 +/- 0.0799 | 0.7477 +/- 0.0052 |
| 4-shot calibration | 0.8070 +/- 0.0024 | 0.9321 +/- 0.0013 | 0.7431 +/- 0.0931 | 0.7762 +/- 0.0104 |

## Same-Feature Controls

| Dataset | Control | 1-shot image/FPR | 2-shot image/FPR | 4-shot image/FPR |
|---|---|---:|---:|---:|
| MVTec | Nearest | 0.8205 / 0.7045 | 0.8161 / 0.7066 | 0.8213 / 0.7081 |
| MVTec | Diag-Gaussian | 0.6931 / 0.7409 | 0.6856 / 0.7452 | 0.6939 / 0.7209 |
| VisA | Nearest | 0.6597 / 0.8863 | 0.6690 / 0.9002 | 0.6695 / 0.8947 |
| VisA | Diag-Gaussian | 0.5568 / 0.9418 | 0.5481 / 0.9425 | 0.5445 / 0.9407 |

## Zero-Shot Top-k Sweep

This isolates the pooling effect from prototype calibration. Pixel metrics are top-k invariant; missing entries mean the supplemental run skipped recomputing them.

| Dataset | top-k | image AUROC | pixel AUROC | PRO | FPR@95TPR |
|---|---:|---:|---:|---:|---:|
| mvtec | 0.001 | 0.8634 | 0.9029 | 0.3022 | 0.6552 |
| mvtec | 0.005 | 0.8631 | 0.9029 | 0.3022 | 0.6381 |
| mvtec | 0.01 | 0.8604 | 0.9029 | 0.3022 | 0.6531 |
| mvtec | 0.02 | 0.8550 | 0.9029 | 0.3022 | 0.6574 |
| mvtec | 0.05 | 0.8453 | 0.9029 | 0.3022 | 0.6767 |
| visa_full | 0.001 | 0.7367 | 0.9263 | 0.6838 | 0.7900 |
| visa_full | 0.005 | 0.7243 | 0.9263 | 0.6838 | 0.8077 |
| visa_full | 0.01 | 0.7124 | 0.9263 | 0.6838 | 0.8170 |
| visa_full | 0.02 | 0.7000 | 0.9263 | 0.6838 | 0.8254 |
| visa_full | 0.05 | 0.6822 | 0.9263 | 0.6838 | 0.8420 |

## Component Attribution

Rows use the final `alpha=0.25`, 512-prototype setting and top-0.1% pooling.

| Dataset | Component | 1-shot image/FPR | 2-shot image/FPR | 4-shot image/FPR |
|---|---|---:|---:|---:|
| mvtec | zero shot topk | 0.8634 / 0.6552 | 0.8634 / 0.6552 | 0.8634 / 0.6552 |
| mvtec | prototype only | 0.8205 / 0.7045 | 0.8161 / 0.7066 | 0.8213 / 0.7081 |
| mvtec | fused | 0.8779 / 0.5332 | 0.8790 / 0.5268 | 0.8804 / 0.5054 |
| mvtec | other class fused | 0.5923 / 0.8787 | 0.5938 / 0.8515 | 0.5996 / 0.8501 |
| visa_full | zero shot topk | 0.7367 / 0.7900 | 0.7367 / 0.7900 | 0.7367 / 0.7900 |
| visa_full | prototype only | 0.6597 / 0.8863 | 0.6690 / 0.9002 | 0.6695 / 0.8947 |
| visa_full | fused | 0.7637 / 0.8337 | 0.7585 / 0.8396 | 0.7616 / 0.8385 |
| visa_full | other class fused | 0.5386 / 0.9082 | 0.5361 / 0.9002 | 0.5313 / 0.9078 |

## Holdout-Class Hyperparameter Selection

MVTec classes are split into held-in development classes and held-out evaluation classes. This is a diagnostic for top-k/prototype selection sensitivity, not the main no-retuning VisA protocol.

| Shot | folds | heldout image AUROC | heldout FPR@95TPR | dev image AUROC | dev FPR@95TPR |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 0.8649 +/- 0.0547 | 0.5875 +/- 0.1801 | 0.8841 | 0.5286 |
| 2 | 9 | 0.8697 +/- 0.0484 | 0.5886 +/- 0.1612 | 0.8888 | 0.5168 |
| 4 | 9 | 0.8615 +/- 0.0490 | 0.5874 +/- 0.1375 | 0.8877 | 0.5191 |

Selection counts:

| alpha | prototypes | top-k | count |
|---:|---:|---:|---:|
| 0.25 | 256 | 0.001 | 2 |
| 0.25 | 256 | 0.005 | 7 |
| 0.25 | 512 | 0.001 | 6 |
| 0.25 | 512 | 0.005 | 2 |
| 0.25 | 1024 | 0.001 | 8 |
| 0.25 | 1024 | 0.005 | 2 |

## Bootstrap Deltas

Positive image deltas mean the fused rule is better. Negative FPR deltas mean fewer normal false positives.

| Dataset | Comparison | Shot | image AUROC delta | FPR delta |
|---|---|---:|---:|---:|
| mvtec | fused_vs_zero_shot | 1 | 0.0145 | -0.1221 |
| mvtec | fused_vs_zero_shot | 2 | 0.0156 | -0.1285 |
| mvtec | fused_vs_zero_shot | 4 | 0.0170 | -0.1499 |
| mvtec | score_blend_vs_fused | 1 | 0.0190 | -0.0371 |
| mvtec | score_blend_vs_fused | 2 | 0.0189 | -0.0114 |
| mvtec | score_blend_vs_fused | 4 | 0.0186 | -0.0307 |
| mvtec | score_blend_vs_zero_shot | 1 | 0.0335 | -0.1592 |
| mvtec | score_blend_vs_zero_shot | 2 | 0.0345 | -0.1399 |
| mvtec | score_blend_vs_zero_shot | 4 | 0.0356 | -0.1806 |
| visa_full | fused_vs_zero_shot | 1 | 0.0270 | 0.0437 |
| visa_full | fused_vs_zero_shot | 2 | 0.0219 | 0.0495 |
| visa_full | fused_vs_zero_shot | 4 | 0.0249 | 0.0485 |
| visa_full | score_blend_vs_fused | 1 | 0.0460 | -0.0710 |
| visa_full | score_blend_vs_fused | 2 | 0.0487 | -0.0918 |
| visa_full | score_blend_vs_fused | 4 | 0.0454 | -0.0624 |
| visa_full | score_blend_vs_zero_shot | 1 | 0.0730 | -0.0274 |
| visa_full | score_blend_vs_zero_shot | 2 | 0.0706 | -0.0423 |
| visa_full | score_blend_vs_zero_shot | 4 | 0.0704 | -0.0139 |

## Prototype 1024 Diagnostic

| Setting | image AUROC | pixel AUROC | PRO | FPR@95TPR |
|---|---:|---:|---:|---:|
| MVTec 1-shot | 0.8753 +/- 0.0059 | 0.8954 +/- 0.0027 | 0.7704 +/- 0.0237 | 0.5275 +/- 0.0087 |
| MVTec 2-shot | 0.8854 +/- 0.0088 | 0.8998 +/- 0.0033 | 0.7553 +/- 0.0130 | 0.5168 +/- 0.0344 |
| MVTec 4-shot | 0.8808 +/- 0.0055 | 0.8999 +/- 0.0009 | 0.7512 +/- 0.0097 | 0.5332 +/- 0.0162 |
| VisA 1-shot | 0.7647 +/- 0.0118 | 0.9329 +/- 0.0013 | 0.7927 +/- 0.0864 | 0.8164 +/- 0.0313 |
| VisA 2-shot | 0.7673 +/- 0.0120 | 0.9336 +/- 0.0008 | 0.7434 +/- 0.0961 | 0.8261 +/- 0.0185 |
| VisA 4-shot | 0.7686 +/- 0.0098 | 0.9368 +/- 0.0016 | 0.7273 +/- 0.0903 | 0.8139 +/- 0.0341 |

## VisA Oracle Diagnostic

This diagnostic tunes alpha/prototype/top-k on VisA itself. It is not a valid main-paper selection rule; it checks whether the FPR limitation is merely the MVTec-selected setting.

| Shot | best image setting | image AUROC/FPR | best FPR setting | image AUROC/FPR |
|---:|---|---:|---|---:|
| 1 | alpha=0.5, proto=512, topk=0.001 | 0.7817 / 0.7987 | alpha=0.5, proto=512, topk=0.001 | 0.7817 / 0.7987 |
| 2 | alpha=0.5, proto=512, topk=0.001 | 0.7719 / 0.8188 | alpha=0.5, proto=256, topk=0.001 | 0.7709 / 0.8091 |
| 4 | alpha=0.5, proto=512, topk=0.001 | 0.7722 / 0.8216 | alpha=0.5, proto=256, topk=0.001 | 0.7596 / 0.7983 |

## Efficiency

| Dataset | Proto | memory MB | sec/image | img/s | peak MB |
|---|---:|---:|---:|---:|---:|
| MVTec | 128 | 5.62 | 0.002848 | 351.55 | 21.78 |
| MVTec | 256 | 11.25 | 0.002865 | 349.07 | 27.40 |
| MVTec | 512 | 22.50 | 0.002896 | 345.32 | 40.03 |
| MVTec | 1024 | 45.00 | 0.002956 | 338.31 | 63.35 |
| VisA | 128 | 4.50 | 0.002898 | 345.33 | 20.65 |
| VisA | 256 | 9.00 | 0.002973 | 336.40 | 25.15 |
| VisA | 512 | 18.00 | 0.002984 | 335.08 | 35.53 |
| VisA | 1024 | 36.00 | 0.003054 | 327.53 | 54.35 |

## Failure Diagnostic Artifacts

- `figures/failure_diagnostics/mvtec_fpr_delta.png`
- `figures/failure_diagnostics/mvtec_paired_scores.csv`
- `figures/failure_diagnostics/mvtec_per_class_fpr_delta.csv`
- `figures/failure_diagnostics/mvtec_score_histogram.png`
- `figures/failure_diagnostics/visa_full_fpr_delta.png`
- `figures/failure_diagnostics/visa_full_paired_scores.csv`
- `figures/failure_diagnostics/visa_full_per_class_fpr_delta.csv`
- `figures/failure_diagnostics/visa_full_score_histogram.png`

