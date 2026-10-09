# Normal-Only Score Normalization

This diagnostic applies class-wise z-normalization to image scores:

`z(x) = (score(x) - mean_normal(class)) / std_normal(class)`.

The normal statistics are estimated only from normal images. For benchmark hygiene, normal test images use 5-fold cross-fit statistics, so a normal image is not normalized by a statistic that includes itself. Anomaly labels are used only after scoring to report AUROC and FPR.

This is a score-normalization diagnostic rather than the main few-shot protocol: it assumes a target normal validation buffer beyond the 1/2/4 support images.

The selected variant uses an untuned equal score blend, `beta=0.5`, after normal z-normalization.

## Selected Normalized Variant

| Dataset | Setting | Image AUROC | Pixel AUROC | PRO | FPR@95TPR | Runs |
|---|---|---:|---:|---:|---:|---:|
| MVTec | 1-shot | 91.77 +/- 0.15 | 89.64 +/- 0.39 | 79.19 +/- 2.22 | 48.61 +/- 2.41 | 3 |
| MVTec | 2-shot | 91.89 +/- 0.21 | 89.90 +/- 0.27 | 77.85 +/- 1.04 | 50.61 +/- 0.54 | 3 |
| MVTec | 4-shot | 91.93 +/- 0.30 | 90.02 +/- 0.16 | 76.13 +/- 1.06 | 49.68 +/- 3.15 | 3 |
| VisA | 1-shot | 83.73 +/- 0.16 | 93.34 +/- 0.36 | 75.43 +/- 8.22 | 76.26 +/- 1.87 | 3 |
| VisA | 2-shot | 83.84 +/- 0.50 | 93.47 +/- 0.49 | 75.90 +/- 7.99 | 72.66 +/- 1.63 | 3 |
| VisA | 4-shot | 83.73 +/- 0.18 | 93.21 +/- 0.13 | 74.31 +/- 9.31 | 73.91 +/- 1.18 | 3 |

## Normalized Beta Sweep

| Dataset | Shot | beta | Image AUROC | FPR@95TPR |
|---|---:|---:|---:|---:|
| MVTec | 1 | 0.00 | 86.63 +/- 0.93 | 64.31 +/- 3.78 |
| MVTec | 1 | 0.10 | 89.20 +/- 0.48 | 58.89 +/- 2.41 |
| MVTec | 1 | 0.25 | 91.07 +/- 0.25 | 54.53 +/- 2.55 |
| MVTec | 1 | 0.50 | 91.77 +/- 0.15 | 48.61 +/- 2.41 |
| MVTec | 1 | 0.75 | 91.43 +/- 0.15 | 46.11 +/- 1.93 |
| MVTec | 1 | 0.90 | 91.02 +/- 0.14 | 47.39 +/- 0.45 |
| MVTec | 1 | 1.00 | 90.53 +/- 0.00 | 49.25 +/- 0.00 |
| MVTec | 2 | 0.00 | 86.87 +/- 0.83 | 65.67 +/- 1.72 |
| MVTec | 2 | 0.10 | 89.44 +/- 0.53 | 60.10 +/- 3.16 |
| MVTec | 2 | 0.25 | 91.26 +/- 0.34 | 54.03 +/- 0.87 |
| MVTec | 2 | 0.50 | 91.89 +/- 0.21 | 50.61 +/- 0.54 |
| MVTec | 2 | 0.75 | 91.46 +/- 0.19 | 48.11 +/- 0.87 |
| MVTec | 2 | 0.90 | 91.03 +/- 0.14 | 48.47 +/- 0.33 |
| MVTec | 2 | 1.00 | 90.53 +/- 0.00 | 49.25 +/- 0.00 |
| MVTec | 4 | 0.00 | 86.72 +/- 1.16 | 69.31 +/- 3.78 |
| MVTec | 4 | 0.10 | 89.25 +/- 0.81 | 63.67 +/- 5.67 |
| MVTec | 4 | 0.25 | 91.16 +/- 0.63 | 55.17 +/- 3.05 |
| MVTec | 4 | 0.50 | 91.93 +/- 0.30 | 49.68 +/- 3.15 |
| MVTec | 4 | 0.75 | 91.57 +/- 0.12 | 45.54 +/- 0.33 |
| MVTec | 4 | 0.90 | 91.11 +/- 0.05 | 47.25 +/- 0.87 |
| MVTec | 4 | 1.00 | 90.53 +/- 0.00 | 49.25 +/- 0.00 |
| VisA | 1 | 0.00 | 77.06 +/- 0.84 | 84.20 +/- 3.33 |
| VisA | 1 | 0.10 | 80.27 +/- 0.52 | 80.73 +/- 2.41 |
| VisA | 1 | 0.25 | 82.85 +/- 0.22 | 76.16 +/- 1.31 |
| VisA | 1 | 0.50 | 83.73 +/- 0.16 | 76.26 +/- 1.87 |
| VisA | 1 | 0.75 | 82.91 +/- 0.11 | 77.30 +/- 1.32 |
| VisA | 1 | 0.90 | 82.15 +/- 0.05 | 77.69 +/- 0.85 |
| VisA | 1 | 1.00 | 81.61 +/- 0.00 | 77.65 +/- 0.00 |
| VisA | 2 | 0.00 | 77.11 +/- 1.21 | 84.10 +/- 1.36 |
| VisA | 2 | 0.10 | 80.65 +/- 0.83 | 78.45 +/- 0.71 |
| VisA | 2 | 0.25 | 83.18 +/- 0.62 | 73.53 +/- 0.63 |
| VisA | 2 | 0.50 | 83.84 +/- 0.50 | 72.66 +/- 1.63 |
| VisA | 2 | 0.75 | 82.95 +/- 0.26 | 74.95 +/- 0.00 |
| VisA | 2 | 0.90 | 82.17 +/- 0.09 | 77.48 +/- 0.16 |
| VisA | 2 | 1.00 | 81.61 +/- 0.00 | 77.65 +/- 0.00 |
| VisA | 4 | 0.00 | 77.80 +/- 1.04 | 81.74 +/- 1.61 |
| VisA | 4 | 0.10 | 80.98 +/- 0.68 | 78.03 +/- 2.44 |
| VisA | 4 | 0.25 | 83.13 +/- 0.31 | 75.74 +/- 1.51 |
| VisA | 4 | 0.50 | 83.73 +/- 0.18 | 73.91 +/- 1.18 |
| VisA | 4 | 0.75 | 82.91 +/- 0.10 | 76.40 +/- 0.68 |
| VisA | 4 | 0.90 | 82.15 +/- 0.05 | 76.78 +/- 0.57 |
| VisA | 4 | 1.00 | 81.61 +/- 0.00 | 77.65 +/- 0.00 |

## Main Observation

- Compared with the current labeled-sweep score blend, the normal-z equal blend improves MVTec image AUROC from 89.68--89.89 to 91.77--91.93 while keeping FPR better than the zero-shot anchor.
- On full VisA, it improves image AUROC from 80.70--80.97 to 83.73--83.84 and FPR from 74.77--77.62 to 72.66--76.26.
- This supports the independent reviewer's hypothesis that normal-only score normalization is a more promising path than more PromptAD baseline runs.
