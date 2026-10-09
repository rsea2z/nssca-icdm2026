# Lightweight Neighbor Boundary

This note answers the reviewer-style question: what is the closest direct comparison to no-training or lightweight alternatives such as PA-CLIP, MRAD, ProtoAD, or GCR?

The named methods are not treated as same-protocol reruns because they differ in backbone, memory construction, score definition, or continual-routing assumptions. Instead, the paper uses frozen-export proxy controls on the same AnomalyCLIP feature cache. This is the direct evidence boundary.

## Mapping From Neighbor Families To Direct Proxies

| Neighbor family | Reviewer example | Direct frozen-export proxy in this project | Why this proxy is fair |
|---|---|---|---|
| Decision-stage zero-shot scoring | PA-CLIP PAD decision | zero-shot top-k score and score-blend audit | isolates whether a decision-stage score rule over the frozen map explains the gain |
| Retrieval-style memory | MRAD | nearest-prototype support retrieval | uses only support memory and nearest matching over exported patch features |
| Prototype / density memory | ProtoAD, PaDiM-like controls | prototype-only and diagonal-Gaussian memory | tests whether normal-memory scoring alone is enough |
| Routing / class-memory fragility | GCR-style routing concern | wrong-class prototype control and class-heldout selection | tests whether class-matched memory is required and whether global choices survive heldout classes |
| Top-k / normalization fragility | fixed top-k and per-image normalization concern | zero-shot top-k sweep, support-only normal-z, and K=8 normal-buffer diagnostic | checks whether the selected scoring rule is only a brittle pooling artifact |

## Same-Protocol Proxy Results

Values are image AUROC / FPR@95TPR percentages. All direct proxies use the same exported AnomalyCLIP features and the same MVTec/VisA splits as the final method.

| Direct proxy | Design axis | MVTec 1/2/4-shot | VisA 1/2/4-shot | Main reading |
|---|---|---:|---:|---|
| Zero-shot top-k | decision-only anchor | 86.34 / 65.52 | 73.67 / 79.00 | top-k pooling alone is not enough |
| Nearest prototype | MRAD-style retrieval proxy | 82.05 / 70.45; 81.61 / 70.66; 82.13 / 70.81 | 65.97 / 88.63; 66.90 / 90.02; 66.95 / 89.47 | retrieval-only memory is weaker than the final score blend |
| Diagonal Gaussian | lightweight density-memory proxy | 69.31 / 74.09; 68.56 / 74.52; 69.39 / 72.09 | 55.68 / 94.18; 54.81 / 94.25; 54.45 / 94.07 | simple density memory collapses image ranking |
| Fused map only | prototype memory plus map fusion | 87.79 / 53.32; 87.90 / 52.68; 88.04 / 50.54 | 76.37 / 83.37; 75.85 / 83.96; 76.16 / 83.85 | fusion helps MVTec but needs the score anchor for VisA FPR |
| Final score blend | proposed frozen-export rule | 89.68 / 49.61; 89.78 / 51.53; 89.89 / 47.47 | 80.97 / 76.26; 80.72 / 74.77; 80.70 / 77.62 | best strict 1/2/4-shot main rule |
| Support-only normal-z | anomaly-free support normalization diagnostic | 89.69 / 57.53; 90.92 / 52.39; 91.44 / 47.89 | 80.23 / 73.63; 82.19 / 74.88; 83.18 / 72.45 | removes extra-buffer assumption but remains MVTec 1-shot sensitive |
| K=8 normal buffer | small extra normal-buffer diagnostic | 91.44 / 49.09; 91.44 / 50.98; 91.44 / 49.57 | 82.75 / 76.35; 83.03 / 75.37; 82.84 / 75.96 | improves stability when a small target-normal validation buffer exists |

## Takeaway

- The closest direct lightweight proxies have already been evaluated under the same frozen-export protocol.
- The gain is not explained by retrieval-only memory, diagonal-Gaussian memory, top-k pooling alone, or wrong-class prototype routing.
- The strict support-only normal-z diagnostic partially addresses anomaly-free score normalization, but it is still a diagnostic rather than a replacement for the labeled-MVTec-selected main rule.
- The K=8 buffer result is the cleanest follow-up path if the deployment setting allows a small extra target-normal validation buffer.

