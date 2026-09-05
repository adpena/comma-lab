# CHARTER ddm_ane2 — ENGINEER the fp16 drift out of the ANE scorer path (operator 2026-09-05: "Drift can be engineered to fix") — selective precision, the fp32 split-point ladder, and a realized exact-argmax hybrid

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm (apparatus/acceleration). Spawned 2026-09-05 ~15:00Z. Parent: ane1 (`.omx/research/ddm_ane1_ane_screening_lane_20260905.md`,
commits 397e10038…addf3814c): placement PROVED 100% fp16 ops; fp32 places 0 ops on the ANE; fp16 SegNet flips 4.818e-5 of argmax pixels (authority bar 3.3e-5 —
1.46× over), fp16 PoseNet self-MSE 1.125e-2 vs d_pose 7.77e-6 (1,448× over; 99.96% of the damage in the |≈31| dimension at 0.83% relative error); ANE screening of
the pose sweep refused (4/39); `coreml_cpu_fp32` bit-exact at 3.28× (SegNet) / 5.12× (PoseNet); hybrid priced GO on area (flip band 0.357% of pixels, 89× headroom
under the 3× bar) but NO-GO on the 07-13 lane's tile-recompute REALIZATION (4.27× the dense pass). MAIN's error to undo: ane1's verdict was recorded as "the ANE is
closed"; per the runtime-lift grant ([[m163]]: device locks are PORTING ITEMS, never walls) an unengineered drift is a to-do, not a closure.

## PRIOR-LAW PREDICTION (owed line, with arithmetic)
fp16 error is RELATIVE (~2^-11 ≈ 4.9e-4 per op, compounding to ~1e-3 over a trunk); the contest's pose term is ABSOLUTE: d_pose 7.77e-6 ⇒ per-dim tolerance
√(7.77e-6) ≈ 2.8e-3 on outputs of magnitude up to |31|. A head-only fp32 split cannot cure PoseNet (feature error 1e-3 × |W·f| ≈ 0.03 ≫ 2.8e-3); the fp32 split
must move EARLY enough that the fp16 prefix's error, propagated through the remaining fp32 suffix, lands ≤ 2.8e-3 per dim. PREDICTION: the PoseNet split-point
ladder (last k blocks fp32) reaches the d_pose bar at a split leaving ≥ 50% of FLOPs in fp16 on the ANE, for ≥ 3× end-to-end — OR it does not, and the measured
sensitivity profile says where the error is born (a layer whose activations have large dynamic range: LayerNorm/GELU tails), which is then curable by per-op fp32
for THAT op alone (CoreML selective precision). For SegNet the flip rate is 1.46× the bar: per-op fp32 on the final decoder + logits (a few % of FLOPs) is predicted
to reach ≤ 3.3e-5, i.e. bit-exact-enough argmax at ≥ 10× trunk speed. FALSIFIER: no split point with ≥ 30% fp16 FLOPs reaches either bar → drift is not engineerable
by precision placement on this hardware; then price the ONE remaining route (fp16 forward + fp32 recompute restricted to the margin band / the pose head's
sensitive subspace) and say plainly if it too fails the 3× bar.

## Scope (in order; all MEASURED on n600 real inputs; ane1's mlpackages + fidelity harness are the instrument — extend, never fork)
1. **Per-op sensitivity profile** (both scorers): with CoreML fp32-CPU as the reference, flip ONE op/block to fp16 at a time (coremltools `compute_precision`
   op-selector) and measure the output error it induces at n120 stratified (then n600 for the finalists): which ops carry the drift?
2. **Split-point ladder**: fp16 prefix on the ANE (placement PROVED per op via MLComputePlan) + fp32 suffix (CPU/GPU) at k = last {1, 2, 4, 8, …} blocks;
   per rung: placement fractions, SegNet flip rate vs 3.3e-5, PoseNet Δ√(10·MSE) vs the sweep's adoption deltas AND vs 2.8e-3/dim, end-to-end ms/forward vs
   1-thread CPU torch (the 76.8 ms torch render+preprocess overhead is OUTSIDE the scorer — report scorer-only AND end-to-end).
3. **Selective per-op fp32** for the ops step 1 names (the minimal set), same measurements; combine with the split if needed.
4. **Realized exact-argmax hybrid (SegNet)**: fp16 ANE dense pass + fp32 recompute restricted to the margin band (top-2 logit margin below the measured fp16 noise
   floor; ane1's census: 0.357% of pixels) with halo — a CROP-batched fp32 CoreML pass, not the 07-13 tile scheme; bar: 0 argmax flips at n600 AND ≥ 3× end-to-end.
5. **Wire what passes** behind ane1's `--scorer-backend` (new values e.g. `ane_split_kN`, `ane_hybrid_exact`) with the same CPU-confirm contract for any
   non-bit-exact backend; replay fs1's 39-point sweep on the best pose backend (rank agreement + wall-clock); ≥ 20 tests. Memo
   `.omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md` with an "Equations leg (`tac.canonical_equations`)" line (extend `scorer_fp16_drift_by_axis_v1`
   with the split-point anchors); lane registered.

## Cost + admission
$0; CPU + ANE (+GPU for fp32 suffix if CoreML places it there); ONE process at a time; detached via `tools/launch_detached_process.py --done-receipt <distinct>`
(the launcher's `--artifact-budget-gib` EXISTS, line 1203); the Metal is cl2's/md3's — CoreML GPU placement is light but declare and measure it; the memory
watchdog is report-only and names the actor.

## OPTIMAL FORM
Reference form = ane1's instrument (real n600 inputs, MLComputePlan placement proof, per-scorer bars) + CoreML selective-precision conversion as documented by
coremltools (grep its API; never guess a kwarg). No mechanism reduction: every drift number is a measured n600 (or stratified n120 → n600 finalist) figure against
the CPU-torch fp32 authority form.

## Rules that bind
NO-FAKE (no "ANE" claim without per-op placement proof); ALWAYS KEEP THE PAYLOAD (mlpackages per rung, per-pair fidelity tables, receipts with sha256);
`upstream/` READ-ONLY (convert copies); no Modal, no burn-cell Metal use; commits ONLY via the serializer with post-edit shas and `[no-triality] [p0-ledger-ok]`;
NO co-author trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints every 10 tool uses (`--subagent-id ddm_ane2`); never
invent flags (grep argparse first); no `/tmp` evidence; register the lane first (`lane_ddm_ane2_engineer_precision_drift_20260905`); persist records before bulk
saves; label MEASURED/DERIVED/INFERRED. `docs/operating_manual_craft_handoff.md` binds. End with `fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
