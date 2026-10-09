# False-Positive Mass Decomposition

This diagnostic fixes the zero-shot 95%-TPR threshold and counts how normal samples move under the fused score.
`corrected_zero_fp` means a zero-shot false positive becomes non-FP at the fixed zero-shot threshold; `introduced_new_fp` means the reverse.
This is intentionally not a main-paper evidence table.
It tests direct threshold transfer across different score scales, while FPR@95TPR recomputes a high-recall threshold per scoring rule.
The introduced-FP pattern below therefore warns that the fused score should not inherit the zero-shot operating threshold without recalibration; it does not contradict the reported FPR@95TPR ranking results.

| Dataset | Shot | Corrected rate | Introduced rate | Net FP delta | Seeds |
|---|---:|---:|---:|---:|---:|
| MVTec | 1 | +0.00% | +34.48% | +34.48% | 3 |
| MVTec | 2 | +0.00% | +34.48% | +34.48% | 3 |
| MVTec | 4 | +0.00% | +34.48% | +34.48% | 3 |
| VisA | 1 | +0.00% | +21.00% | +21.00% | 3 |
| VisA | 2 | +0.00% | +21.00% | +21.00% | 3 |
| VisA | 4 | +0.00% | +21.00% | +21.00% | 3 |

## Per-Run Rows

| Dataset | Shot | Seed | Corrected | Introduced | Persistent | Stable | Net FP delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| MVTec | 1 | 0 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 1 | 1 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 1 | 2 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 2 | 0 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 2 | 1 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 2 | 2 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 4 | 0 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 4 | 1 | 0 | 161 | 306 | 0 | +34.48% |
| MVTec | 4 | 2 | 0 | 161 | 306 | 0 | +34.48% |
| VisA | 1 | 0 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 1 | 1 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 1 | 2 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 2 | 0 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 2 | 1 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 2 | 2 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 4 | 0 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 4 | 1 | 0 | 202 | 760 | 0 | +21.00% |
| VisA | 4 | 2 | 0 | 202 | 760 | 0 | +21.00% |
