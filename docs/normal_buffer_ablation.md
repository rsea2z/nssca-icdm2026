# Normal Buffer Size Ablation

This diagnostic asks how many target-normal validation images per class are needed to make class-wise score centering useful.
For each class, `K` normal images are selected by a deterministic hash and removed from evaluation; their mean score centers all remaining images from that class.
No anomaly label is used for scoring, centering, or buffer selection.

Selected diagnostic setting: K=8 normal validation images per class and beta=0.5.

## Selected Buffer Diagnostic

| Dataset | Setting | Image AUROC | FPR@95TPR | Runs |
|---|---|---:|---:|---:|
| MVTec | 1-shot | 91.44 +/- 0.36 | 49.09 +/- 2.98 | 9 |
| MVTec | 2-shot | 91.44 +/- 0.28 | 50.98 +/- 3.28 | 9 |
| MVTec | 4-shot | 91.44 +/- 0.45 | 49.57 +/- 3.51 | 9 |
| VisA | 1-shot | 82.75 +/- 0.35 | 76.35 +/- 3.51 | 9 |
| VisA | 2-shot | 83.03 +/- 0.39 | 75.37 +/- 3.17 | 9 |
| VisA | 4-shot | 82.84 +/- 0.35 | 75.96 +/- 3.08 | 9 |

## Buffer Size Sweep at Beta 0.5

| Dataset | Shot | K | Image AUROC | FPR@95TPR |
|---|---:|---:|---:|---:|
| MVTec | 1 | 1 | 89.40 +/- 0.67 | 62.02 +/- 5.60 |
| MVTec | 1 | 2 | 90.28 +/- 0.91 | 58.23 +/- 3.89 |
| MVTec | 1 | 4 | 91.10 +/- 0.74 | 52.33 +/- 4.65 |
| MVTec | 1 | 8 | 91.44 +/- 0.36 | 49.09 +/- 2.98 |
| MVTec | 1 | 16 | 91.55 +/- 0.45 | 48.39 +/- 2.92 |
| MVTec | 1 | 32 | 90.99 +/- 0.61 | 49.23 +/- 5.01 |
| MVTec | 2 | 1 | 89.53 +/- 0.59 | 63.32 +/- 2.75 |
| MVTec | 2 | 2 | 90.39 +/- 0.83 | 58.58 +/- 4.19 |
| MVTec | 2 | 4 | 91.12 +/- 0.85 | 52.58 +/- 4.10 |
| MVTec | 2 | 8 | 91.44 +/- 0.28 | 50.98 +/- 3.28 |
| MVTec | 2 | 16 | 91.56 +/- 0.49 | 50.31 +/- 3.39 |
| MVTec | 2 | 32 | 90.95 +/- 0.61 | 53.24 +/- 6.29 |
| MVTec | 4 | 1 | 89.31 +/- 0.46 | 62.73 +/- 3.51 |
| MVTec | 4 | 2 | 90.33 +/- 0.78 | 56.98 +/- 4.37 |
| MVTec | 4 | 4 | 91.22 +/- 0.60 | 52.25 +/- 4.00 |
| MVTec | 4 | 8 | 91.44 +/- 0.45 | 49.57 +/- 3.51 |
| MVTec | 4 | 16 | 91.58 +/- 0.66 | 48.10 +/- 3.03 |
| MVTec | 4 | 32 | 91.05 +/- 0.69 | 49.23 +/- 7.48 |
| VisA | 1 | 1 | 80.46 +/- 1.69 | 81.16 +/- 4.72 |
| VisA | 1 | 2 | 81.72 +/- 0.57 | 78.73 +/- 3.81 |
| VisA | 1 | 4 | 82.28 +/- 0.32 | 76.79 +/- 1.94 |
| VisA | 1 | 8 | 82.75 +/- 0.35 | 76.35 +/- 3.51 |
| VisA | 1 | 16 | 82.93 +/- 0.21 | 76.72 +/- 2.14 |
| VisA | 1 | 32 | 83.01 +/- 0.27 | 75.82 +/- 2.27 |
| VisA | 2 | 1 | 80.52 +/- 1.59 | 79.25 +/- 5.74 |
| VisA | 2 | 2 | 81.82 +/- 0.60 | 77.28 +/- 4.33 |
| VisA | 2 | 4 | 82.50 +/- 0.28 | 74.75 +/- 2.62 |
| VisA | 2 | 8 | 83.03 +/- 0.39 | 75.37 +/- 3.17 |
| VisA | 2 | 16 | 83.15 +/- 0.33 | 74.91 +/- 2.39 |
| VisA | 2 | 32 | 83.17 +/- 0.41 | 74.68 +/- 1.51 |
| VisA | 4 | 1 | 80.17 +/- 1.84 | 81.09 +/- 3.06 |
| VisA | 4 | 2 | 81.70 +/- 0.64 | 78.49 +/- 3.83 |
| VisA | 4 | 4 | 82.18 +/- 0.49 | 77.13 +/- 2.56 |
| VisA | 4 | 8 | 82.84 +/- 0.35 | 75.96 +/- 3.08 |
| VisA | 4 | 16 | 83.07 +/- 0.26 | 76.97 +/- 1.84 |
| VisA | 4 | 32 | 83.09 +/- 0.29 | 76.41 +/- 1.55 |

## Per-Class Stability at K=8, Beta 0.5

| Dataset | Shot | Image improved | FPR reduced | Worst FPR class |
|---|---:|---:|---:|---|
| MVTec | 1 | 10/15 | 5/15 | capsule (+28.89) |
| MVTec | 2 | 10/15 | 8/15 | capsule (+28.15) |
| MVTec | 4 | 10/15 | 8/15 | zipper (+18.98) |
| VisA | 1 | 9/12 | 8/12 | chewinggum (+15.61) |
| VisA | 2 | 9/12 | 7/12 | pipe_fryum (+16.93) |
| VisA | 4 | 9/12 | 7/12 | chewinggum (+13.23) |

## Main Observation

- K=4 already recovers most MVTec image-AUROC gain, while K=8 gives the best MVTec FPR tradeoff among small buffers.
- VisA benefits more in image ranking than FPR, reinforcing that cross-dataset score calibration remains partly unresolved.
- This supports a concrete follow-up setting: strict few-shot support plus a small normal validation buffer, rather than a vague future normal-z direction.
