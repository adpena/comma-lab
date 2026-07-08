# POSE SOLVED-IN-PRINCIPLE — the OUTPUT-SPACE per-pair inverse solve (2026-07-08)

STORES CONSULTED (proactive recall — this synthesis is 90% recall, 10% new): DAG FEED-poseladder
2026-07-03 (the FRAME0-FREE-CANVAS eureka: P-E/P-F, operator-approved, the eval-hack FIREWALL) ·
`council_pose_carrier_optimal_form_symposium_20260703.md` (R1 = L0 SOLID floor d_pose 0.0011 contribution
0.105; **"R1 floored at 0.0011 ONLY because it trained with w_pose=0"**; L3 null-texture lever; #205 = the
pose experiment) · `pose_frame0_inverse_solve_probe_20260703T0810Z.md` (#249: P-E existence 2.71e-07;
quantization-aware STE solve robust, grad_vs_frozen_gap 0.0; CORRECTION — image-space stores
RATE-PROHIBITIVE at n600, pose-space scalars the only cheap channel) · #251 (MLX proposer parity 8.8e-2 →
CPU-torch polish/authority) · FEED-posehard/poseresearch/poseforkverdict + `pose_carrier_arms_measured_
20260708.md` (16030e6bf: pair-consistency law; deterministic 1.995; real-f0 10.42 DEAD-formulation) ·
`pose_taskspace_native_morse_smale_depth_warp_design_20260708.md` (#365 + review 5711a4fdf). Operator
directive: "With utmost rigor and passion fix and solve pose for our task space SDF level set witness."
Pointer **0.19110 UNMOVED** — this memo is a SYNTHESIS of measured anchors + a solve design; the A2/A2+
measurement (in flight) is the anchor-producer. Axis of every number cited: as in its source artifact
([macOS-CPU advisory] class unless stated).

## 1. THE CONTRADICTION, RESOLVED (four of our own anchors, now one consistent picture)

| anchor | number | what it actually measured |
|---|---|---|
| R1 probe (#245, 2026-07-03) | d_pose **0.0011** | SOLO descent of the per-pair residual against a FROZEN (w_pose=0) witness render — the OUTPUT-SPACE objective, trained alone |
| run-1 crucible ep200 | **1.79 plateau** | the SAME representation trained JOINTLY under ~18 competing terms (w_seg=100, w_pose=1.0, EMA verdict) — a training-dynamics artifact, NOT a cap |
| "cap ~2.5 by construction" (FEED-posehard) | 2.562 self-fit floor | a PHOTOMETRIC self-fit: fit ξ to warp real-f0→real-f1 in PIXEL space — a FLOW-SPACE objective; proves planar ≠ dense flow, says NOTHING about the output-space solve |
| a131065f (2026-07-08) | 1.995 deterministic / 10.42 real-f0 | the pair-consistency law + the un-descended deterministic start |

**The resolution in one line: d_pose lives in PoseNet's 6-dim OUTPUT space, and per-pair ξ_eff is a
6→6 map — so the planar warp family, hopeless at reproducing dense FLOW, is generically sufficient to
STEER the 6 scored outputs to their targets.** The pose-research planar-manifold saturation argument is
correct in flow space; its projection onto output space is not ~2.5 — it is 0.0011 (MEASURED, R1). The
"H-TARGET hard cap" verdict conflated the two objective spaces (verdict_scope of that cap: the
photometric-self-fit + joint-training INSTANCE, not the carrier family). Every prior negative survives
re-scoping; no anchor is discarded.

## 2. THE SOLVE (three stacked mechanisms; S1 measured-mechanism, S2 prediction-under-measurement, S3 optional)

**S1 — SOLVE-DON'T-TRAIN (kills the run-1 plateau class):** pose leaves the training loop. At
export/byte-close time, on the FROZEN final witness render, run a per-pair 6-DOF damped LM/Gauss-Newton
solve of ξ_eff directly through R (STE through uint8 — the #249 correctness key; solution provably
survives the eval roundtrip, grad_vs_frozen_gap 0.0) against the exact 6-scalar PoseNet targets.
Deterministic, minutes-class, decoupled from every crucible pathology (gradient competition, EMA lag,
spike-guard, dxi scale). Expected floor: reproduces R1's **0.0011 → contribution 0.105** as a SOLVED
quantity. This is #342 solve-don't-train applied to pose, and the 6→6 conditioning makes it the textbook
case. If f1 later moves (more training), re-solve — cheap.

**S2 — +k STEERING DOF (the #365 depth stratification, re-purposed):** the per-cell
affine-inverse-depth params are not (only) geometric correctness — they are EXTRA STEERING DIMENSIONS
transverse to the planar submanifold in output space. 6 targets vs 6+k well-conditioned DOF ⇒ the
target moves INTO the reachable set; residual collapses orders below 0.0011 (PREDICTION; A2+ measures).
#249's P-E (2.7e-07 with unbounded DOF) is the existence proof that reachability, not representation, is
the only question — and the DOF-vs-bytes ladder is steep because we are closing a 6-constraint gap.
Counted cost k≈4–12 fp16 scalars/pair → ~5–15 KB total → rate ~0.003–0.010. Target: d_pose ~1e-4–3e-5 →
**contribution ~0.02–0.03 (ancestor-class) at store-nothing-preserving rate.**

**S3 — ξ-consistent null-texture (symposium L3):** the in-training 0-rate lever (pose-legible texture in
the SegNet-null) stays available as co-adaptation, but is NO LONGER load-bearing — S1/S2 carry the floor
post-hoc. v7.5's pose launch gate stops being a training bet and becomes a $0 post-hoc check: "does the
S1 solve on the current EMA render reproduce ≤~0.0011 at n600?"

## 3. LEGALITY (firewall walked, not waved)
Stored: pose-space scalars only (ξ_eff + k depth params per pair) — video-derived → COUNTED (~3–15 KB
→ rate ~0.002–0.010). Warp + render: generic deterministic code (rule-118 FREE). Inflate loads NO
scorer — the solve is compress-time only (the "TTO is a compress-time tool ONLY" discipline verbatim;
unlimited compute at compress, single forward at inflate). #249's CORRECTION explicitly endorses "an
O(10)-scalar pose-aligned residual" as the legal cheap channel and names the image-table smuggle as the
eval-hack FAKE — S1/S2 store no image-space tables. Pair-consistency is enforced BY the objective (solve
against the actual witness f1), which is why this succeeds where stored-real-f0 failed 10.42.

## 4. BUDGET CONSEQUENCE (honest, with the uncertainty labeled)
S1 alone (measured mechanism): pose 0.105 + rate ~0.002 — sub-0.19 then demands d_seg+rate ≤ 0.085
(hard). S2 (if A2+ confirms): pose ~0.02–0.03 → sub-0.19 arithmetic opens (d_seg 0.09 + rate 0.07 +
pose 0.03 = 0.19) and sub-0.15 stays a d_seg+rate fight with pose retired from the budget — the
2026-07-03 council's "pose LEAVES the budget" reframe, now with the mechanism that actually does it.
UNCERTAINTIES: R1 0.0011 custody (n/through-R details not re-read this turn — S1's own measurement
re-establishes the floor independently, the plan does not lean on R1's exact digit); per-pair
ill-conditioning tail (near-zero-motion pairs; LM damping + report the tail); k-DOF conditioning
(A2+ measures, no assertion).

## 5. EXECUTION (in flight + next)
Rung A2/A2+ ADDED to the running ladder agent (same harness, same pairs, #249 tool machinery; A2 =
6-DOF solve, A2+ = 6+k with A1's depth params; cut A1-GT-cells before A2 if the envelope binds).
On land: register `pose_output_space_inverse_solve_v1` (+ revise `morse_smale_stratified_parallax_
dpose_v1` to the field form) with the measured anchors [advisory axis, n-labeled, n600/exact owed];
fold the (d_pose, bytes) frontier into #248/#365; SPEC_v75 §1 pose-gate delta ONLY after the
measurement (no spec edits on predictions). DSL obligation: the solve lands as a byte-close/export
stage + a `Lever` factory when BUILT (config-orphan discipline). Triality: this memo + DAG
FEED-posesolve (same commit); equations on anchor-landing.
