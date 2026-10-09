# Selected Blend vs K=8 Buffer

This note compares the main labeled-MVTec-selected score blend with the K=8 normal-buffer diagnostic.
It is a boundary table, not a new method claim.

## Aggregate Stability

| Dataset | Shot | Selected blend: image+ / FPR reduced | K=8 buffer: image+ / FPR reduced | Net change in FPR reduced |
|---|---:|---:|---:|---:|
| MVTec | 1 | 8/15 / 7/15 | 10/15 / 5/15 | -2 |
| MVTec | 2 | 9/15 / 8/15 | 10/15 / 8/15 | 0 |
| MVTec | 4 | 9/15 / 8/15 | 10/15 / 8/15 | 0 |
| VisA | 1 | 7/12 / 4/12 | 9/12 / 8/12 | +4 |
| VisA | 2 | 7/12 / 5/12 | 9/12 / 7/12 | +2 |
| VisA | 4 | 7/12 / 6/12 | 9/12 / 7/12 | +1 |

## Reading

- The K=8 buffer clearly strengthens VisA class-level FPR stability.
- MVTec remains the stronger source benchmark for the selected blend.
- The table supports a narrow follow-up story: if a small extra normal buffer is available, the calibration becomes more stable on VisA, but this does not replace the strict 1/2/4-shot setting.
