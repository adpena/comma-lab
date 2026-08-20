The charter is fully executed and committed as `016c2a62dd`.

Artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_sr1_implicit_edge_conditioning_20260811.md)
- [Reproducible pre-probe](/Users/adpena/Projects/pact/experiments/ddm_sr1_implicit_edge_preprobe.py)
- [Retained analysis receipt](/Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811/ANALYSIS_RESULT.json)

Measured `[macOS-CPU advisory, scorer-free n600 token entropy]` across 117,964,800 events:

- Causal-edge conditioning: 114,704 B, only **−2 B / −0.00174%** after table and framing charges.
- Pose cross-stream conditioning: 114,749 B, **+43 B**.
- Both decoded exactly with empty terminal states, 25 checkpoints, and byte-identical deterministic repeats.

The preregistered 1% falsifier fired. Standalone additive edge/pose probability calibration is closed at FORMULATION scope. The surviving route is distortion-side joint conditioning: use decoder-derived edge state inside js1’s semantic/pose optimization without transmitting a mask.

Not measured: `d_seg`, `d_pose`, rendered output, complete archive bytes, or an exact score. Effective composed pointer remains `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

Validation passed: payload-retention scan, table round trips, Ruff, compilation, artifact/SHA verification, two review-tracker passes, and serializer post-commit verification.

## NEXT_IF_RESUMED

- `implicit_joint_distortion_conditioning` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1 stage-1 / #995 successor; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/`; fire trigger: ps135 `SOLVE` has landed, js1 stage 0 retained the shipping-base per-edge decomposition, no explicit edge mask ships, and every complete semantic/pose/archive candidate is retained and jointly priced against an equal-parameter control.

## LIVE-HYPOTHESES

- Decoder-derived edge state can still improve segmentation at fixed bytes when it changes joint proposal generation and capacity allocation; sr1 tested only post-hoc probability calibration.
- Boundary-aware precision routing may outperform uniform precision because segmentation debt is localized while total counted model bytes can remain fixed.
- Rich carrier-to-semantic modulation may help where scalar pose sign failed because the scalar probe discarded nearly all cross-stream geometry.

## DEAD-ENDS

- Standalone causal-edge calibration: closed because it saved only 2 charged bytes and had weak held-out support.
- Scalar pose-sign/delta-sign calibration: closed because it cost 43 bytes and was held-out negative.
- Explicit PE3/contour/mask transmission: prior evidence was scorer-negative on every tested pair and cost 74,408 B.
- Causal partition-codec replacement: CPC1 already measured 255,288 B, over twice the live semantic stream.
- Checkerboard/reordering alone: it introduces no new information beyond HPAC’s causal group context.
- Transferring CR1 support-object savings or foreign-paper percentages to the full wire: invalid because they measure different objects and receivers.

