# DDM DA1 telemetry-decomposition DAG feed — 2026-07-28

Status: `DECOMPOSED_FIVE_WALLS_MEASURED_n600_ADVISORY_POINTER_UNMOVED`.
Arm `ddm_da1`, worktree off `main@d41cba1b10`. Axis `[macOS-CPU advisory]` — real LZMA1/WebP coders +
frozen CPU-torch SegNet/PoseNet + plug-in conditional entropy over decoder-derivable cells. NO
byte-closed `upstream/evaluate.py` row. Pointer UNMOVED `0.19108` custody / official `0.172`.
`score_claim=false · promotion_eligible=false · rank_or_kill_eligible=false`.

Every fc1/r2s/oc1 AGGREGATE taken apart; the disaggregation IS the signal:

- **D1 support 421,366 B** (reproduced EXACTLY): Road 213,581 (50.7%) + Lane 145,738 (34.6%) = **85%
  Road/Lane separatrix**. Temporal FALSIFIER TRIGGERED — XOR-delta 582,960 (1.38× WORSE), static-freq
  predictor empty (68 B). Boundary drifts **1.41 px median** pair-to-pair → redundancy is CONTOUR, not
  mask. verdict_scope INSTANCE/FORMULATION (mask-granularity delta-coding on copy base).

- **D2 frame_0 2.7 MB** (4-Q n600 SegNet+PoseNet+range(A)): pose_term CATASTROPHIC at every rung
  (0.75 @ 4.65 MB → 1.04 @ 2.70 MB, ≥6× banked 0.127, ≥4.4× the 0.172 bar); support grows 1.19–1.53×;
  **56.5–57.3% of crush error is scorer-invisible (ker A)**. REFRAME: the 2.7 MB prices STORED-real-f0;
  banked pose uses a store-nothing WARP carrier (0.127 @ ~0 B) → WebP-stored frame_0 is pose+rate
  DOMINATED, a fallback not a floor. verdict_scope FAMILY.

- **D3 label 41,392 B** (reproduced 41,358 floor): **73.9% of bytes in TOP-10 cells, ALL Road-adj-Lane
  boundary cells**. Determinism is a BOUNDARY-DISTANCE property (bd0 = 98% deterministic), NOT margin
  (margin-alone H stays 1.7–2.0; charter margin-threshold hypothesis FALSIFIED). Lever = restrict
  support to low-bdist annulus, derive the far tail. verdict_scope INSTANCE.

- **D4 values 10 MB** (60-pair, 100,596 flip sites): range(A) residual flips 98.87% of sites; minimal
  uint8 amplitude **median 1.11, p90 7.78, 64% ≤2 steps**. Implied alphabet **~1.7–4 b (+sign)** vs
  int8×3's 24 b → r2s 10.06 MB is OVER-PRECISION (H0≈8 smooth-signal signature per #532), not
  incompressibility. Reprice via amplitude+sign+context. verdict_scope INSTANCE (encoding over-precise).

- **D5 pose 0.127**: STOPPED-DESCENT artifact. R1 run stopped ep1130, seg plateaued (~0.0046) while
  d_pose still dropping **−1.26%/epoch (log)**. Carrier ladder (proximity law): exact 9.3e-10→0 · PR130
  2.33e-5→0.015 · banked 0.00161→0.127 · box 0.0166→0.408. **~0.095 S left on the table at ZERO added
  bytes** (~217 unchanged-byte epochs banked→solved-plane). verdict_scope FAMILY.

Receipts: `/Volumes/VertigoDataTier/pact/ddm_da1_20260728/{d1_support_decomp_n600,d1c_d3_boundary_htable_n600,
d2_frame0_coupling_n600,d4_minimal_amplitude,d5_pose_decomp}.json`. Drivers `experiments/ddm_da1_{d1,d1c_d3,
d2,d4}*.py` (committed 194224ed5b, 9028dcab2b). Memo `.omx/research/ddm_da1_telemetry_decomposition_20260728.md`.

No launch, exact score, fire, reseal, paid dispatch, promotion, or pointer movement. MAIN landing review.
