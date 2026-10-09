# ML Reproducibility Checklist v2.0 Responses

Paper: **Training-Free Normal-Support Score Calibration for Frozen Vision-Language
Anomaly Detection**

Answers use **Yes**, **No**, **Partial**, or **Not applicable**. File and section
references are relative to this reproducibility package; paper references use the
submitted manuscript.

## Models And Algorithms

1. **Is there a clear description of the mathematical setting, algorithm, and/or model?**
   **Yes.** The paper's Method section defines the normal-support setup, cosine-distance
   prototype bank, map fusion, top-k image pooling, and conservative score blend. The
   implementation is in `src/fewshot_ad/calibration/`, `src/fewshot_ad/evaluation/`, and
   `scripts/run_feature_calibration.py`.

2. **Is there a clear explanation of any assumptions?**
   **Yes.** The paper assumes only a small set of target-normal images is available at
   deployment, no labeled anomalies are used to construct the calibration head, and the
   frozen detector export remains fixed. It also states that support images must cover
   the relevant normal modes and that fully anomaly-free hyperparameter selection is out
   of scope.

3. **Is there an analysis of the complexity (time, space, sample size) of the algorithm?**
   **Yes.** The paper gives the `O(Md)` memory and `O(NMd)` cosine-comparison cost, where
   `M` is prototypes per class, `d` feature dimension, and `N` patch tokens. It reports
   22.50 MiB for 512 MVTec prototypes and approximately 345 images/s after export.
   `runs/efficiency/` contains the measured snapshots.

## Theoretical Claims

4. **For theoretical claims, is the claim clearly stated?**
   **Not applicable.** The paper makes no theorem-level theoretical contribution. Its
   fixed-threshold decomposition is stated as a diagnostic identity and interpreted
   empirically.

5. **For theoretical claims, is there a complete proof?**
   **Not applicable.** See item 4. The fixed-threshold score decomposition is derived
   directly in the Method section and does not rely on an external theorem.

## Datasets

6. **Are relevant statistics, such as number of examples, provided?**
   **Yes.** The paper and `README.md` report 15 MVTec categories, 3,629 training images,
   1,725 test images, 12 VisA categories, and 2,162 VisA test images. Per-run counts are
   stored in the result JSON files.

7. **Are train/validation/test splits described?**
   **Yes.** Normal calibration images come from the standard normal training split; the
   full public test split is used for evaluation. Seeds 0/1/2 generate three deterministic
   1/2/4-shot support choices. The class-heldout selection diagnostic is described
   separately as a diagnostic rather than the main protocol.

8. **Is preprocessing explained, including excluded data?**
   **Yes.** The package describes MVTec's original layout, the MVTec-style VisA
   conversion, AnomalyCLIP feature/map export, masks, and JSONL manifests. No benchmark
   images are excluded from the reported test sets.

9. **Is there a link to a downloadable dataset or simulation environment?**
   **Yes.** `README.md` links to the official MVTec AD site and the official VisA
   project/AWS Open Data pages.

10. **For newly collected data, is the collection and annotation process described?**
    **Not applicable.** The paper uses only public benchmarks and does not introduce a new
    collected dataset.

## Shared Code

11. **Are dependencies specified?**
    **Yes.** `requirements-reproducibility.txt` pins the tested Python environment, and
    `README.md` identifies Python, CUDA/PyTorch, and the external AnomalyCLIP dependency.

12. **Is training code included?**
    **Not applicable for the proposed calibration head**, because it performs no gradient
    update or prompt training. Feature export, calibration, evaluation, baseline-control,
    summary, and statistical-analysis code are included. PromptAD remains an external
    trained baseline; provenance and aligned-evaluation scripts/reports are included.

13. **Is evaluation code included?**
    **Yes.** `scripts/run_zero_shot_topk_sweep.py`,
    `scripts/run_feature_calibration.py`, `scripts/run_topk_sweep.py`,
    `scripts/summarize_score_blend.py`, `scripts/run_bootstrap_suite.py`, and
    `src/fewshot_ad/evaluation/` implement the reported evaluations.

14. **Are pretrained models included?**
    **Partial.** External AnomalyCLIP prompt checkpoints and CLIP weights are not
    redistributed because of size and upstream licensing. `README.md` gives exact paths,
    official repository/source instructions, and the export command for each dataset. The
    calibration head is constructed deterministically from support features, so it has no
    separately learned weights.

15. **Does the README include a table of results with precise commands?**
    **Yes.** `README.md` includes the main result table and command blocks for split
    construction, feature export, zero-shot evaluation, fused calibration, score blending,
    bootstrap statistics, dense-PRO auditing, efficiency profiling, and result collection.

## Experimental Results

16. **Is the range of hyperparameters, selection method, and final configuration given?**
    **Yes.** `README.md` lists the alpha, prototype-budget, beta, and top-k ranges; states
    that final values were selected on a labeled MVTec sweep and frozen for VisA; and
    identifies the final setting as `alpha=0.25`, `beta=0.25`, `M=512`, and
    `top-k=0.001`. Stored heldout and support-only diagnostics expose the boundary of this
    selection protocol.

17. **Is the exact number of training and evaluation runs given?**
    **Yes.** The proposed calibration has no training runs. `README.md` reports 18 main
    calibration evaluations (2 datasets x 3 support sizes x 3 seeds), 2 zero-shot
    full-dataset runs, 20 dense-PRO snapshots, and 2,000 bootstrap resamples per
    comparison.

18. **Is there a clear definition of each reported measure or statistic?**
    **Yes.** `README.md` defines tie-aware image and pixel AUROC, the PRO integral and its
    0.30 maximum-FPR limit, threshold counts, and normal FPR@95TPR. The paper also states
    that FPR@95TPR is an evaluation diagnostic rather than a deployable threshold rule.

19. **Are results reported with central tendency and variation?**
    **Yes.** Main proposed-method rows report mean and standard deviation over three
    support seeds. Bootstrap confidence intervals and paired score-level deltas are stored
    in `runs/statistics/` and summarized in the paper.

20. **Is average runtime or estimated energy cost reported?**
    **Partial.** The paper reports post-export calibration runtime, memory, and PromptAD
    wall-clock training cost where available. Energy consumption was not measured. The
    hardware and software environment are stated in `README.md`.

21. **Is the computing infrastructure described?**
    **Yes.** `README.md` records the Windows host, RTX 3070 Laptop GPU with 8 GiB memory,
    driver version, Python version, and tested package versions. CPU execution is also
    supported for calibration evaluation.
