# External Baseline Provenance

Last checked: 2026-05-09.

GitHub reachability spot-check: `git ls-remote` succeeded for AnomalyCLIP, PromptAD, AdaptCLIP, and CLIP-AD on 2026-05-09.

Core extracted few-shot values are recorded in `docs/external_baseline_results.md`.

## Decision Rule

1. Use paper-published values when the paper reports the same dataset, shot count, and metric.
2. If a needed cell is absent and public code exists, run the method in our protocol.
3. If neither a paper value nor runnable public code exists, leave the cell blank.
4. Do not infer missing metrics from nearby protocols. In particular, `FPR@95TPR` is treated as our-run-only unless a paper explicitly reports it.

Published external values must be marked as literature/protocol context when the method uses a different backbone, training data, prompt-learning procedure, or image-score definition from our frozen AnomalyCLIP calibration protocol.

## Main Paper Baselines

| Method or family | Paper result status | Code status | Current action |
|---|---|---|---|
| AnomalyCLIP | Paper reports MVTec AD and VisA zero-shot results. | Official repo: https://github.com/zqhang/AnomalyCLIP | Use our reproduced zero-shot maps for main anchor; cite paper values only as context. |
| PaDiM / PatchCore | Papers report standard MVTec AD anomaly detection/localization results, but not our frozen AnomalyCLIP feature protocol. | PatchCore repo: https://github.com/amazon-research/patchcore-inspection | Keep our same-feature nearest and diag-Gaussian controls as direct evidence; cite original results only as standard full-data context. |
| WinCLIP / WinCLIP+ | Paper reports MVTec AD and VisA zero-shot and few-normal-shot results. | Public implementations exist; paper also uses OpenCLIP. | Use published 0/1/2/4-shot paper values where the requested metric exists; run code only for missing metrics such as `FPR@95TPR`. |
| PromptAD | Paper reports MVTec and VisA 1/2/4-shot image-level and pixel-level results. | Official repo: https://github.com/FuNz-0/PromptAD | Published values remain external context. Direct aligned MVTec image/FPR reproduction is complete for 1/2/4-shot seeds 0/1/2; pixel AUROC is not usable from the classification route. |
| DevPrompt | Paper reports one-normal-shot MVTecAD and VISA results. | No public repo located in local paper scan or quick web search. | Use paper values for 1-shot only; leave 2/4-shot or missing metrics blank unless code appears. |
| CLIP-AD | Paper reports zero-shot MVTec AD and VisA results. | Public repo found: https://github.com/ByChelsea/CLIP-AD | Use zero-shot paper values as ZSAD context; do not fill few-shot cells unless code is adapted and evaluated. |
| AdaCLIP | Paper reports zero-shot MVTec AD and VisA results. | Official repo: https://github.com/caoyunkang/AdaCLIP | Use zero-shot paper values as backbone context; few-shot cells stay blank unless evaluated. |
| AA-CLIP | Paper reports zero-shot MVTec AD and VisA results. | Public repo found: https://github.com/Mwxinnn/AA-CLIP | Use zero-shot paper values as backbone context; missing metrics stay blank unless evaluated. |
| GenCLIP | Paper reports zero-shot MVTec and VisA results. | No public repo located in local paper scan or quick web search. | Use zero-shot paper values only; leave missing metrics blank. |
| AdaptCLIP | Paper reports zero-shot and one/few-shot industrial-domain results including MVTec and VisA. | Official repo: https://github.com/gaobb/AdaptCLIP | Use paper values where dataset, shot, and metric match; run code only for missing exact metrics. |
| FastRef | Paper reports MVTec, VisA, MPDD, and RealIAD 1/2/4-shot results. | No public repo located in local paper scan or quick web search. | Use published few-shot values directly; leave `FPR@95TPR` blank unless code becomes available. |

## Direct Baseline Reruns

| Method | Dataset / split | Completed local evidence | Missing evidence | Use in paper |
|---|---|---|---|---|
| PromptAD | MVTec aligned 1/2/4-shot seeds 0/1/2 | `runs/strong_baselines/promptad/mvtec/k_*/csv/Seed_*-results.csv`; 1-shot AUROC/FPR 92.73 +/- 0.41 / 23.30 +/- 3.17; 2-shot 93.19 +/- 0.43 / 21.78 +/- 1.22; 4-shot 94.08 +/- 1.10 / 19.63 +/- 0.69 | Usable pixel AUROC from segmentation path | Treat as a strong-baseline warning; PromptAD is stronger on MVTec image AUROC and FPR. |

## Closest Lightweight Context

These are the nearest no-training / lightweight neighbors to cite when the review asks for direct context around memory, pseudo-anomaly, or routing design.

| Method | Paper result status | Code status | Current action |
|---|---|---|---|
| PA-CLIP | Paper reports zero-shot MVTec AD and VisA results with pseudo-anomaly awareness. | No direct aligned rerun yet. | Cite as adjacent zero-shot context; do not infer our few-shot FPR from its protocol. |
| MRAD | Paper reports train-free zero-shot MVTec AD and VisA results via memory-driven retrieval. | No direct aligned rerun yet. | Cite as adjacent train-free retrieval context; use our same-feature nearest-prototype support retrieval as the direct proxy. |
| ProtoAD | Paper reports prototype-based anomaly localization on MVTec-class benchmarks. | No aligned frozen-AnomalyCLIP rerun yet. | Cite as the closest prototype-memory neighbor; use prototype-only and diagonal-Gaussian controls as direct proxies. |
| GCR | Paper reports task-agnostic continual anomaly detection with routing across prototype banks. | No aligned frozen-AnomalyCLIP rerun yet. | Cite as the closest routing-style neighbor; use wrong-class prototypes and class-heldout selection as direct routing-fragility proxies. |

The proxy-to-neighbor mapping and key numbers are consolidated in
`docs/lightweight_neighbor_boundary.md`.

## Optional Expansion Candidates

These local papers also contain MVTec/VisA-style results and can be added to a broader comparison table if space allows.

| Method | Paper result status | Code status | Action if added |
|---|---|---|---|
| APRIL-GAN / VAND challenge method | Reports zero-shot and 1/2/4-shot MVTec AD and VisA results. | https://github.com/ByChelsea/VAND-APRIL-GAN | Use paper values first. |
| InCTRL | Reports few-shot MVTec AD and VisA results. | https://github.com/mala-lab/InCTRL | Use paper values first; code can fill missing metrics if needed. |
| AnoPLe | Reports few-shot multi-class MVTec AD, VisA, and Real-IAD results. | https://github.com/YoojLee/AnoPLe | Use paper values first; mark as multi-class protocol context. |
| SubspaceAD | Reports 1/2/4-shot MVTec AD and VisA results. | https://github.com/CLendering/SubspaceAD | Use paper values first; code can fill missing metrics if needed. |
| UniVAD | Reports one-normal-shot MVTec AD, VisA, and MVTec LOCO results. | https://github.com/FantasticGNU/UniVAD | Use 1-shot paper values; leave other shot counts blank unless evaluated. |
