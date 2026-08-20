Completed and committed the crosswalk as [ddm_dio1_dion3_crosswalk_20260814.md](/Users/adpena/Projects/pact/.omx/research/ddm_dio1_dion3_crosswalk_20260814.md) in commit `c70d21085440b465beb951ed805139e3ca724439`.

Key conclusion: the [Dion3 paper](https://arxiv.org/html/2608.11612v1) and [Torch repository](https://github.com/microsoft/dion) target large NVIDIA/FSDP optimizer walls that Pact does not have. Current-scale Gram-NS, custom kernels, and megabatching were folded. Row-subset updates with error feedback remain worth a scorer-free shadow replay under the existing Muon-finisher owner.

Measured:

- Old V9 NS5: 9.689 GFLOP/epoch; CPU toy bracket 0.0274% of the real epoch median.
- Current TR1 NS5: 1.789 GFLOP/epoch; CPU toy bracket 0.0615%.
- Inputs, outputs, repeat outputs, timings, and source are retained in the [SSD receipt](/Volumes/VertigoDataTier/pact/ddm_dio1_20260814/retained/ns5_toy_bracket_receipt.json).

Not measured: direct MLX/Metal timing, Dion3 convergence on Pact, scorer effects, or an exact contest row. No training, scorer, paid job, or archive build ran.

Effective frontier remains MC36 Variant C: **S=0.1619344578804448 @ 186,269 B `[contest-CUDA T4, n600]`**. Own-vehicle frontier remains LC2: **S=0.16959899569230852 @ 187,226 B `[contest-CUDA T4, n600]`**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER:** Owner: WP1/JD1 Muon-finisher owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_dio1_20260814/retained/`. Fire trigger: storage preflight passes and a governed scorer-free trace contains real per-step gradients and pre-step momentum for all six renderer matrices; then run update-RMS-matched shadow replay without changing weights.
- **FOLDED-INTO-EXISTING-OWNER, CONTINGENT:** Owner: JS1/#982 owner. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/`. Fire trigger: RX2 harvest and #982 preflight complete, with eligible matrices and retained gradient/momentum traces; reuse the same shadow protocol.

## LIVE-HYPOTHESES

- Top-L1 row selection may reduce simultaneous renderer-row interference while error feedback preserves omitted momentum; the existing six-block Muon finisher makes this directly testable.
- Dion3’s inverse-square-root fraction scaling may provide a useful initial bracket, but measured full-tensor update RMS remains authoritative.
- Stable Gram-NS could matter for numerical quality if a future larger trunk develops ill-conditioned polar inputs, even though no current speed wall exists.

## DEAD-ENDS

- Importing Dion3’s 6× speed headline is closed at the current scale: it measures optimizer-only GH200/FSDP workloads, while Pact NS5 remains far below the 5% build gate.
- CuteDSL kernels, CUDA graphs, and FSDP megabatching are closed for the current MLX/Metal line because their runtime and communication bottlenecks are absent.
- Direct reuse of the Torch/DTensor package is closed; only the smaller selector/error-feedback idea has a plausible consumer.
- RX2 is closed as a Dion3 consumer because it is a fixed-token rate harvest.
- Direct ambient row masking of #552/#556 factors is closed because it does not preserve their tangent, SPD, and product-metric contracts.
- Importing paper fractions or learning rates as Pact constants is closed because their evidence is task- and scale-specific.