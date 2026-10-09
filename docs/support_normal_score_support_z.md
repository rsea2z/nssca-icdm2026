# Support-Only Normal Score Normalization

This experiment uses only the 1/2/4 normal support images to estimate class-wise score offsets.
For 1-shot stability, the selected support-only rule subtracts the class support mean and keeps a global score scale; no anomaly label is used for scoring or normalization.

Selected setting: `support_z`, `beta=0.5`.

## Selected Support-Only Variant

| Dataset | Setting | Image AUROC | Pixel AUROC | PRO | FPR@95TPR | Runs |
|---|---|---:|---:|---:|---:|---:|
| MVTec | 1-shot | 89.69 +/- 1.81 | 89.64 +/- 0.39 | 79.19 +/- 2.22 | 57.53 +/- 11.84 | 3 |
| MVTec | 2-shot | 90.92 +/- 1.53 | 89.90 +/- 0.27 | 77.85 +/- 1.04 | 52.39 +/- 11.05 | 3 |
| MVTec | 4-shot | 91.44 +/- 0.90 | 90.02 +/- 0.16 | 76.13 +/- 1.06 | 47.89 +/- 6.14 | 3 |
| VisA | 1-shot | 80.23 +/- 0.89 | 93.34 +/- 0.36 | 75.43 +/- 8.22 | 73.63 +/- 3.69 | 3 |
| VisA | 2-shot | 82.19 +/- 0.74 | 93.47 +/- 0.49 | 75.90 +/- 7.99 | 74.88 +/- 2.87 | 3 |
| VisA | 4-shot | 83.18 +/- 0.41 | 93.21 +/- 0.13 | 74.31 +/- 9.31 | 72.45 +/- 1.23 | 3 |

## Sweep

| Dataset | Shot | Mode | beta | Image AUROC | FPR@95TPR |
|---|---:|---|---:|---:|---:|
| MVTec | 1 | support_z | 0.00 | 85.52 +/- 0.65 | 62.31 +/- 8.36 |
| MVTec | 1 | support_z | 0.25 | 89.59 +/- 0.82 | 54.82 +/- 6.68 |
| MVTec | 1 | support_z | 0.50 | 89.69 +/- 1.81 | 57.53 +/- 11.84 |
| MVTec | 1 | support_z | 0.75 | 88.67 +/- 2.32 | 66.60 +/- 11.17 |
| MVTec | 1 | support_z | 1.00 | 87.61 +/- 2.56 | 71.38 +/- 10.28 |
| MVTec | 2 | support_z | 0.00 | 87.40 +/- 1.04 | 61.74 +/- 2.73 |
| MVTec | 2 | support_z | 0.25 | 91.02 +/- 1.02 | 53.10 +/- 4.01 |
| MVTec | 2 | support_z | 0.50 | 90.92 +/- 1.53 | 52.39 +/- 11.05 |
| MVTec | 2 | support_z | 0.75 | 89.80 +/- 1.82 | 59.96 +/- 12.90 |
| MVTec | 2 | support_z | 1.00 | 88.61 +/- 1.93 | 65.52 +/- 13.62 |
| MVTec | 4 | support_z | 0.00 | 87.95 +/- 1.38 | 59.60 +/- 7.30 |
| MVTec | 4 | support_z | 0.25 | 91.47 +/- 1.04 | 48.04 +/- 9.04 |
| MVTec | 4 | support_z | 0.50 | 91.44 +/- 0.90 | 47.89 +/- 6.14 |
| MVTec | 4 | support_z | 0.75 | 90.28 +/- 0.99 | 56.32 +/- 10.49 |
| MVTec | 4 | support_z | 1.00 | 88.99 +/- 0.93 | 63.38 +/- 8.66 |
| VisA | 1 | support_z | 0.00 | 73.80 +/- 1.20 | 84.89 +/- 1.56 |
| VisA | 1 | support_z | 0.25 | 79.27 +/- 0.59 | 79.14 +/- 2.31 |
| VisA | 1 | support_z | 0.50 | 80.23 +/- 0.89 | 73.63 +/- 3.69 |
| VisA | 1 | support_z | 0.75 | 79.58 +/- 1.87 | 76.02 +/- 4.92 |
| VisA | 1 | support_z | 1.00 | 78.55 +/- 2.18 | 78.52 +/- 4.28 |
| VisA | 2 | support_z | 0.00 | 75.86 +/- 1.02 | 84.30 +/- 1.04 |
| VisA | 2 | support_z | 0.25 | 81.27 +/- 0.78 | 76.20 +/- 2.73 |
| VisA | 2 | support_z | 0.50 | 82.19 +/- 0.74 | 74.88 +/- 2.87 |
| VisA | 2 | support_z | 0.75 | 81.52 +/- 0.74 | 74.71 +/- 1.93 |
| VisA | 2 | support_z | 1.00 | 80.45 +/- 0.80 | 74.22 +/- 2.46 |
| VisA | 4 | support_z | 0.00 | 76.52 +/- 0.43 | 83.85 +/- 2.89 |
| VisA | 4 | support_z | 0.25 | 82.30 +/- 0.42 | 76.13 +/- 2.03 |
| VisA | 4 | support_z | 0.50 | 83.18 +/- 0.41 | 72.45 +/- 1.23 |
| VisA | 4 | support_z | 0.75 | 82.35 +/- 0.31 | 73.53 +/- 2.34 |
| VisA | 4 | support_z | 1.00 | 81.19 +/- 0.24 | 75.64 +/- 3.19 |
