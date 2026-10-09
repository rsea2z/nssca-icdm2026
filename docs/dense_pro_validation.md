# Dense-PRO Validation

This audit recomputes PRO with 200 uniformly spaced thresholds for the final top-0.1% rule.
Image AUROC and FPR use the final score-blended image scores when available; PRO@200 is computed from the fused maps and is unaffected by the score blend.
It is a robustness check for the main-paper PRO values, which were generated with 50 thresholds for tractability.

| Dataset | Setting | Image AUROC | Pixel AUROC | PRO@200 | FPR@95TPR | Runs |
|---|---|---:|---:|---:|---:|---:|
| MVTec | zero-shot | 86.34 | 90.29 | 56.48 | 65.52 | 1 |
| MVTec | 1-shot | 89.68 +/- 0.14 | 89.64 +/- 0.39 | 83.40 +/- 1.79 | 49.61 +/- 0.12 | 3 |
| MVTec | 2-shot | 89.78 +/- 0.20 | 89.90 +/- 0.27 | 83.30 +/- 1.21 | 51.53 +/- 1.22 | 3 |
| MVTec | 4-shot | 89.89 +/- 0.36 | 90.02 +/- 0.16 | 84.28 +/- 1.71 | 47.47 +/- 3.33 | 3 |
| VisA | zero-shot | 73.67 | 92.63 | 78.66 | 79.00 | 1 |
| VisA | 1-shot | 80.97 +/- 0.12 | 93.34 +/- 0.36 | 81.44 +/- 1.34 | 76.26 +/- 0.42 | 3 |
| VisA | 2-shot | 80.72 +/- 0.43 | 93.47 +/- 0.49 | 82.15 +/- 1.73 | 74.77 +/- 0.52 | 3 |
| VisA | 4-shot | 80.70 +/- 0.24 | 93.21 +/- 0.13 | 83.55 +/- 1.78 | 77.62 +/- 1.04 | 3 |

## Deltas vs Zero-Shot

Positive PRO deltas are better; negative FPR deltas are better.

| Dataset | Setting | Delta PRO@200 | Delta FPR@95TPR |
|---|---|---:|---:|
| MVTec | 1-shot | 26.92 +/- 1.79 | -15.92 +/- 0.12 |
| MVTec | 2-shot | 26.82 +/- 1.21 | -13.99 +/- 1.22 |
| MVTec | 4-shot | 27.80 +/- 1.71 | -18.06 +/- 3.33 |
| VisA | 1-shot | 2.78 +/- 1.34 | -2.74 +/- 0.42 |
| VisA | 2-shot | 3.49 +/- 1.73 | -4.23 +/- 0.52 |
| VisA | 4-shot | 4.89 +/- 1.78 | -1.39 +/- 1.04 |
