# POSE SOLVED-IN-PRINCIPLE — the OUTPUT-SPACE per-pair inverse solve (2026-07-08)

## ⛔ 0a. MEASURED REFUTATION (2026-07-08, pose_mladder_depthwarp_measured, 70649531f) — THE CENTRAL CLAIM OF THIS MEMO IS REFUTED AT THE FORMULATION LEVEL. Read this FIRST; the derivation below is kept append-only as the (wrong) hypothesis it corrects.
The A2/A2+ ladder MEASURED (n8–24, real byte-close, frozen CPU-torch, positive control 1.2e-12) that a
per-pair 6-DOF (and +6 oracle-mask off-plane, 12-DOF) inverse solve **does NOT reach d_pose 0.0011**. It
floors at **~1.2–1.5** — orders above the ~0.019 target:
- A0 (0-DOF global ground-H warp) = **1.685**; A2 (6-DOF pose-space LM solve) = **1.486**; A2+ (12-DOF,
  ORACLE GT mask) = **1.223**. Same-8-pair: 1.580 → 1.362 → 1.223 (monotone but shallow, floors ~1.2).
- **WHY my synthesis was WRONG (the specific error):** I claimed "ξ_eff is a 6→6 map → generically steers
  the 6 outputs to target." FALSE for a fixed render: a rate-cheap 6-scalar warp of the FIXED cartoon
  render spans a LOW-RANK, APPEARANCE-CONSTRAINED manifold whose reachable PoseNet-output set does NOT
  contain the real-pair target. Adversarially confirmed a genuine local min (zero-init + aggressive
  FD/iters plateaus identically 1.2–2.1), not a conditioning artifact.
- **Rung 0 killed the #365 depth premise up front (measured, not assumed):** corr(d_pose, |ego
  translation|) = **NEGATIVE** (−0.446 n24 / −0.676 n8) — d_pose does NOT rise with forward motion;
  off-plane finite-depth parallax MASS ≈ **0.5%** (area 2.7%). Depth stratification can touch ≤~0.5–3% of
  the flow — and A2+ with an ORACLE mask (a strict ceiling) delivered only −10%, empirically confirming it.
- **#249's ~2.7e-7 does NOT transfer:** it solved FREE PIXELS (full-rank, rate-prohibitive per #249's own
  correction), NOT a rate-cheap warp. Different object.
- **R1's cited 0.0011 is NOT reproducible by a post-hoc warp of this fixed render.** IF real, it required
  JOINT pose-descent training that co-adapts the render itself — so it is an UNMEASURED-at-authority anchor
  owed re-validation through byte-close at n600 (#238), NOT a floor a cheap carrier reaches. Note the LIVE
  #205 run (w_pose=1.0 joint) sits at d_pose ~1.75 at ep200 (pre-Muon), also not near 0.0011 yet.
- **verdict_scope: FORMULATION** — REFUTED = "cheap post-hoc 6(+k)-DOF warp of a FIXED render reaches low
  d_pose." NOT killed: joint pose-descent training that co-adapts the render (the L3 null-texture path;
  #205 is the live test), nor the paradigm. Store-real-appearance stays DEAD (10.4 + rate 573).
- **CORRECTED next step:** pose is NOT cheaply solved post-hoc. Either a dedicated pose-descent run makes
  the render pose-legible in the SegNet-null (re-validate R1 at n600 byte-close first — is 0.0011 even
  real?), OR pose is a budget item (~1.7 → contribution √(10·1.7)≈4.1 → kills sub-0.19) and the sub-0.15
  fight rides d_seg + rate with pose held. Eq `pose_output_space_inverse_solve_v1` stays COUNCIL-FLAGGED
  as REFUTED-at-formulation (anchor: A2=1.486, not 0.0011). `morse_smale_stratified_parallax_dpose_v1`
  registered with the measured advisory anchors (the shallow frontier), n600 owed.

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

## 5b. BYTE-CLOSE READINESS (verified 2026-07-08, $0 downstream-blocker check — the solve is byte-close-VIABLE at the 6-DOF floor with EXISTING machinery)
Checked `tools/levelset_byte_close_and_eval.py` + `src/tac/boundary_math/xi_pose_coder.py` against S1/S2. READ, not inferred:
- **S1 (6-DOF solve) → byte-close READY modulo a connector.** The store-nothing-ξ carrier EXISTS + is byte-closed
  (#241): `serialize_pose_carrier_store_nothing` (PCAR store_nothing **v2** = derive-H at decode from coded ξ, NO
  stored H block, n_kf=0) warps the GENERATED witness render by per-pair H (rule-118 free; op-for-op oracle-parity
  with the shipped inflate). `serialize_xi_payload(q, scales, coder)` takes a QUANTIZED per-pair ξ as INPUT — so
  A2's externally-SOLVED ξ_eff feeds straight in (solve → quantize → serialize). **No new format**; the only gap is
  a small connector (hand LM-solved ξ to the serializer + read back realized d_pose on the inflated frames — the
  tool already does realized-through-R read-back). CONFIRMS the 0.0011-class floor is byte-CLOSEABLE, not just
  harness-measurable.
- **Discipline the tool enforces (docstring L56-64), CONSISTENT with the solve:** an inert stored sidecar the render
  does NOT consume is bytes the scorer never reads → does NOT lower realized d_pose (`--fold-pose-sidecar` OFF by
  default, LOUDLY records this). The solve is NOT that — ξ_eff PARAMETRIZES the decode warp → changes the rendered
  frames. Consume-by-warp, not inert-store. A2's number is already realized-through-warp-through-R.
- **S2 (+k depth steering) → byte-close needs a NEW decode stage (the genuine #365 build, gated on A2+ paying).** The
  per-cell affine-inverse-depth warp does NOT exist in the decode yet (only the canonical-equation stub
  `morse_smale_stratified_parallax_dpose_20260708.py` + a gauge mention). When A2+ shows +k pays: extend the
  store-nothing warp to apply per-Morse-cell inverse-depth flow from k stored fp16/pair (COUNTED), oracle-parity
  gated, inlined into the shipped inflate. The dead `warp_real_luma` PCAR (stores real keyframe — 10.42,
  formulation-dead) is NOT the target; warp-GENERATED-source (what store_nothing already is) is.
Net: A2 = near-ready byte-close landing (connector only); A2+ = scoped, gated build. No scramble when the number lands.
