# SPEC_v10 — RECONCILED SSoT (2026-07-19): the capstone as ONE constrained program on the shared scorer plane
<!-- # DUPLICATE_SOT_OK: THE live v10 base SSoT (registry doc_id spec_v10_capstone_reconciled, status=active); coexists BY DESIGN with the registered additive successor SECTION spec_v10_integer_plane_vehicle (which supersedes only stale prose in §§R2/R3/R6/R7) and the registered-superseded historical cold_start doc; waiver added 2026-07-19 at MAIN drift-fix -->

**This document is the single self-consistent successor of
`.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md` (§0–§15)** — the reconciliation the
state review (`v10_capstone_state_review_20260719_codex.md` §5 BUILD-1, tasks #521/#540) named as owed.
It folds, with no contradictions left standing: the base SPEC · the lattice composition law + its n600
closure (`v10_lattice_rate_verdict_and_composition_20260719.md`) · the registered
`f32_receiver_arithmetic_exactness_admissibility_v1` and `pose_plane_proximity_corollary_v1` laws · the
flattened Lagrangian/KKT derivation **as corrected by the 2026-07-19 fresh-eyes verification**
(`spec_v10_reconciliation_and_kkt_verify_20260719_fable.md`, findings V-1..V-8) · the landed #543
production receiver and #548 ŷ R-D ladder · and the 9-step seal path with per-step status.

**Subordination:** NO-FAKE supreme rule > THE GOAL (sub-0.15) > this SPEC. SPEC_v75 §8 OPERATING
CONTRACT binds verbatim. **Pointer honesty: `0.1910828242 [contest-CPU Linux x86_64]` UNMOVED —
everything here is MEANS**; success has exactly one definition (a lower byte-closed
`upstream/evaluate.py` n600 exact row). Bank 0.18804 is NON-SUBMISSION borrowed substrate, never a v10
input. **`launch_ready = false`** (see §R7). Axis honesty: every number tagged; all local rows are
`[macOS-CPU advisory]`, `score_claim=false`, `promotable=false`.

**Naming lock (unchanged):** v9c2 = terminal warm diagnostic (banked ep725 best, advisory n600 d_seg
0.003458) · v9c3 = warm fixed-events dev successor (dev, non-promotable) · **v10 = the cold, fully
seeded capstone — no warm weights ever.** `_dev`/`_prod` maturity axis orthogonal; only `_prod` exact
rows may touch the pointer.

---

## §R1 The settled physics (MEASURED — the representation collapse)

1. **One shared plane.** Both frozen scorers read the SAME 384×512×3 plane: SegNet reads the last-frame
   plane; PoseNet resizes FIRST with the same bilinear `A` then yuv6 (`upstream/modules.py:71-75`)
   [DERIVED-from-source, frozen factorization].
2. **Exact realization is solved and free.** The factor-2 bounded-uint8 lattice solve realizes ANY
   target plane exactly in camera uint8 space. n600 replay COMPLETE (receipt blocker 1 CLOSED): 50/50
   chunks, **114 mismatched pixels / 117,964,800 = d_seg 9.664e-7**, max |A residual| 8.5e-14; every
   mismatch is the fp32 ULP-tie class; clip(round) comparator is 376× worse (the SOLVER buys exactness)
   [MEASURED n600 advisory, lattice memo + aggregate receipt on VertigoDataTier].
3. **One realization buys BOTH scorers.** Pairs (gt_f0, solved_f1): d_seg 0.0 and mean d_pose 9.3e-10
   (n6, full upstream DistortionNet) — seg and pose COLLAPSE onto one axis: fidelity of the shared plane
   ŷ [MEASURED n6 advisory]. uint8-STE / R-roundtrip / AA-washout walls are 0 BY CONSTRUCTION for any
   ŷ-generator (composition law; reproduced at n24 by the #548 ladder: rung B re-realizes the witness's
   advisory d_seg 0.003455).
4. **Direct-plane payloads are rate-DEAD** (verdict_scope: formulation — complete direct-plane/raw-frame
   payloads): 1.70 MB/frame ⇒ n600 rate term ≈ 680; source-exact direct ŷ ≈ 628 [MEASURED/DERIVED,
   ladder rungs A/C/D + lattice memo]. **The compact generator/state description family is OPEN and is
   the whole game:** the ep725 witness-as-ŷ-generator = **83,838 actual bytes** (n600 rate term 0.0558)
   at d_seg 0.003455 / d_pose 63 (frame1-diagnostic pairing) [MEASURED, ladder rung B].
5. **fp32 arithmetic is part of the contract.** Frame-195 / pair-125 class: exact rational plane
   equality can still flip single argmax cells at the ULP-tie surface (pair-125: 1 flip at ZERO
   perturbation, d_seg 5.09e-6). The registered `f32_receiver_arithmetic_exactness_admissibility_v1`
   law (hard oracle or margin > κ·μ_ULP) is load-bearing on every exactness claim; #543 declares
   `native-cpu-torch-f32-first-max-class-index.v1` [MEASURED + registered].
6. **Pose is a corollary of plane proximity** (`pose_plane_proximity_corollary_v1`, registered):
   near-source planes d_pose 5.35e-10..1.14e-9 (per-row means at plane RMSE ≈ 0); the c2-witness plane
   at RMSE 25 gives d_pose 63 at ANY d_seg (verdict_scope: instance). Pose binds only at the marginal
   crossover d_pose* = 2.5e-4 — **~5.7 orders of magnitude** above the solved rows (the earlier
   "~9 orders" wording was an arithmetic error; corrected per verification V-1). The intermediate-RMSE
   regime is UNMEASURED; regime thresholds are ASSUMED pre-registered radii.

## §R2 The optimization object (the corrected flattened Lagrangian/KKT program)

Canonical statement: `v10_flattened_lagrangian_kkt_derivation_20260719.md` **including its 2026-07-19
appended corrections** (V-2..V-6). Summary with corrections baked in:

- **Variables:** plane residuals `r_0, r_1 ∈ R^N`, N = 589,824/frame, vs the source planes. Camera
  realization, uint8 survival, R-roundtrip are OUT (0 by construction, §R1.2-3); ker(A) never enters
  (range-only counting; arbitrary-fill is the measured rate-death).
- **Objective:** `R(r) = bytes(Q(r) vs a rule-118-free decode-side predictor)` — proxy: weighted
  entropy/ℓ1 (convex relaxation); authority: actual brotli-Q11/zstd-19 bytes, never the proxy.
- **Seg constraints:** per-pixel winner/rival half-spaces, first-order:
  `q_{p,c′}·r_1 ≥ −m_{p,c′}` with cached margins (gt_n600; MEASURED distribution: median 5.89,
  2.67% < 1.0, 0.28% < 0.1 — the annulus is a few % of pixels). Band law (implemented,
  `joint_seg_pose_rate.py`): `r_ch = min(r_max, d_feat/(3·Lip_local·|q_ch|))`, `d_feat = scale·m/‖Δw‖`.
  **Scope (V-4):** the box is the inner approximation of each single-pixel DIAGONAL slab; cross-pixel
  receptive-field coupling (ERF r50≈85px) is NOT bounded by it; `Lip_local` is configured custody, not
  yet measured; no positive-band operating point exists yet — the hard oracle is the sole authority.
- **Pose constraint:** `(1/6)‖J_0 r_0 + J_1 r_1‖² ≤ τ_pose` — ONE convex quadratic restricting 6 DOF.
- **KKT structure:** duals λ concentrate on the annulus (complementary slackness — corroborated: 96.9%
  of c2 residual flips in-annulus; margin distribution above). **Pose inactivity (μ=0) at in-band
  solutions is a PRE-REGISTERED FALSIFIABLE PREDICTION, not a theorem** (V-3): the near-source rows are
  the trivial r→0 limit; the only measured large-r row had d_pose 63. If pose binds, the correction is
  6 DOF at ENCODE time whose BYTE cost through the description grammar is measured (the bytes(τ_pose)
  curve), never "O(6) numbers free" (V-2 — a decode-side Jᵀ expansion would need PoseNet-derived,
  video-derived rows: forbidden/counted).
- **Waterfill:** `dS = 100·d(d_seg) + (5/√(10·d_pose))·d(d_pose) + (25/37,545,489)·d(bytes)`; crossover
  d_pose* = 2.5e-4. The #536 waterfill is **2-term PENDING the bindingness harvest** (V-6); the one
  admissible measured point is `INCONCLUSIVE_FLAT_OR_NOISY` (instance scope, #549).
- **Convexity scope (V-5):** first-order + continuous relaxation only. Quarantined non-convexity:
  (a) the Lipschitz validity radius, (b) the fp32 hard-oracle check + repair loop (combinatorial),
  (c) integer/quantized description variables.
- **Channels:** seg bands anisotropic per 1/|q_ch|; PoseNet chroma passes a 2×2 box (fine chroma
  pose-invisible) ⇒ chroma residual pose-cheap by construction; luma carries the pose-sensitive
  component [DERIVED-from-source].

## §R3 Registered laws (equations leg)

| law | status | note |
|---|---|---|
| `bounded_uint8_resize_preimage_cell_feasibility_v1` (factor-2 lattice) | REGISTERED | ratified pre-review; n600 replay now closed |
| `f32_receiver_arithmetic_exactness_admissibility_v1` | REGISTERED | binds every exactness claim (§R1.5) |
| `pose_plane_proximity_corollary_v1` | REGISTERED; **module corrected 2026-07-19** (V-1/V-7: ~5.7 orders; ASSUMED-threshold labels) | **OWED to MAIN: re-emit the registry row** — the appended JSONL payload still embeds the stale "9 orders" summary |
| `segnet_head_rank4_linear_flipdist_v1` | REGISTERED (2026-07-15) | consumed by the band law |
| anisotropic band law (`derive_hyperplane_channel_band`) | IMPLEMENTED, not separately registered | registration owed WITH its first positive-band anchors (#540 discipline: no law without real rows) |
| KKT/waterfill law | BLOCKED on measured curves | do not register without scientific rows (state-review branch-custody ruling) |

## §R4 Built / landed inventory (as of 2026-07-19, all on MAIN)

- **#543 production receiver (LANDED):** scorer-free deterministic factor-2 archive/inflate
  (`ZIP_STORED` 0.bin, canonical header + length-prefixed sections), 12-real-pair byte-close proof
  (double decode, tree-hash equal, 14,155,776/14,155,776 exact numerators), storage preflight,
  write-once pair stages, resume. Limits (honest): raw-y is structurally exact but NOT byte-competitive
  (9.4× reference at n600); `witness-y-stub` is a typed fail-closed placeholder; factor-10 section
  MISSING; DERIVED n600 decode 3.8 min (7.9× headroom vs 30-min budget) is linear projection.
- **#548 ŷ R-D ladder (LANDED, n24 advisory):** 4 rungs, 56,623,104/56,623,104 block solves
  FEASIBLE_EXACT, zero repairs; decisive separation 83,838 B (generator) vs 42,051,900 B (direct
  Brotli-Q11 n24 planes). Frame0 policy = source (not contest-complete); pose interaction debt named.
- **n600 lattice replay (CLOSED, #547 leg 1):** §R1.2.
- **#549 joint-solve instrument (LANDED):** additive solver + bounded runner; zero-band control exact;
  range-coordinate description rate-dead at that point; positive bands BLOCKED on winner/rival VJP +
  PoseNet-6 Jacobian custody — **the running VJP arm's deliverables**.
- **Structural V10 compiler:** 7 counted sections + `Factor2IntegerScorerPlane`; `MISSING_FACTOR_IDS =
  ("10",)`; `launch_ready=false` throughout.
- **KKT derivation memo:** corrected (append-only) per §R2.
- **RUNNING:** VJP-custody positive-bands arm (constraint matrix: q rows, J, Lip field → first
  positive-band curve points; bindingness maps; bytes(τ_pose) curve).

## §R5 SUPERSESSION MAP (base SPEC §0–§15 → disposition; history never deleted)

Legend: **STANDS** (binding as written) · **STANDS-SCOPED** (binding within the named scope) ·
**SUPERSEDED-BY** (successor statement here governs).

| base § | disposition |
|---|---|
| §0 vehicle identity / naming | **STANDS** (v10 = cold start, reserved; §14.1 lock unchanged) |
| §1 quadrilateral (Kolmogorov·projection·realization·completeness) | **STANDS** as doctrine; realization leg now SOLVED-EXACT for the shared plane (§R1.2) |
| §2 P1 seeded static classes | **STANDS-SCOPED**: seeds are now content of the compact ŷ-DESCRIPTION (generator/state), scored through the lattice+receiver path |
| §2 P2 head born solved + gauge | **STANDS-SCOPED** (trained-generator descriptor path); #530 Init≠Fork split binding |
| §2 P3 range(A)-restricted targets (#520) | **SUPERSEDED-BY §R2 range-only counting + lattice realization**: ker(A) is excluded at the PAYLOAD level by construction (range-only byte counting + free preimage solve). #520's render-target projection layer remains relevant ONLY inside a trained-generator descriptor, as a training-efficiency lever — no longer a launch-gate |
| §2 P4 content-priced coder | **STANDS**: applies to whatever compact description ships; authority = archive.zip stat |
| §2 P5 boundary laws (#518) | **STANDS-SCOPED** (trained-generator path). #518 branch MERGED (p0 ledger 2026-07-19); the 8-vs-27 efficacy A/B artifact is still OWED — defaults remain provisional |
| §2 P6 store-nothing pose + joint descent | **SUPERSEDED-BY the pose plane-proximity law (§R1.6, §15)**: for the solve-near-source family, training-side pose forces are the WRONG lever (c2-lineage diagnostics only). The photometric-wall result (post-hoc stored pose DEAD, 5 formulations) STANDS for far-from-source generators; R1 dxi (7.2 KB) remains the fallback bank. Pose disposition per family: solve-near-source → expect slack (pre-registered, bindingness-gated); far generator → pose destroyed regardless |
| §2 P7 per-class carriers + basis cure | **STANDS-SCOPED**: the v8 carrier kit and p0_497 curvelet A/B are candidate COMPACT-ŷ-DESCRIPTION grammars — their byte prices enter the §R2 objective; the ladder is their pricing instrument |
| §2 P8 v9c2 terminal seeds | **STANDS**: witness-as-ŷ is the live measured rung (83,838 B / d_seg 0.003455) — the generator baseline any compact description must beat or refine (needs ~4× better d_seg at similar rate, or a residual-vs-generator payload) |
| §3/§3.1 necessity-certificate table | **STANDS** as carrier-design evidence (per-cell certificates, Lane no-safe-interior, annulus LUMA one-sided, hood-tex 1,759 B) — re-read as pricing the ŷ-description; realization walls are 0 by construction so every certificate is now purely a RATE statement |
| §4 seeds inventory | **STANDS** with the same re-read; counted/free boundary unchanged (rule 118) |
| §5 completeness table | **STANDS**; the "d_seg↔d_pose coupling through shared A" missing-term (§14.11) is now REALIZED IN THE REPRESENTATION (one plane, §R1.3) |
| §6 launch-gating chain | **SUPERSEDED-BY §R6 (the 9-step seal path)** — §6 was written for the trained-generator-only path; its gates survive inside steps 5–7 where that path is active |
| §7 open questions OQ-0..6 | **STANDS-SCOPED**: OQ-0 answered (v9c2 terminal/diagnostic, §14.3); OQ-1 re-scoped by P6 supersession (pose window relevant to the generator path only); OQ-2/3/5/6 open; OQ-4 open (phase efficacy UNTESTED per §14.6) |
| §8/§9 DSL module + provenance | **STANDS**; #529 (real success path) / #530 / #531 remain owed before any compile claims |
| §10 DAG feed | historical record — STANDS |
| §11 round-1 review F1–F8 | **STANDS** (its fixes remain binding) |
| §12 triality/stores | historical record — STANDS |
| §13 session fold (Fisher · triggers · EMA · arms A/B/C/G · §13.13 SOL fixes) | **STANDS-SCOPED**: all measured findings remain valid; the built levers are trained-generator-path boundary-A/B items. #528 (w_pose×score-domain refusal) and #529/#530/#531 remain binding build items. The reconciled PRIORITY puts the inverse-solve/compact-description path FIRST (it is the measured non-dominated arm); the §13 levers fire only where a trained generator remains the descriptor |
| §14.1 naming lock | **STANDS** |
| §14.2 birth-before-phase order law | **STANDS-SCOPED** (trained-generator path); efficacy UNTESTED (§14.6 retraction stands — the per-pixel post-hoc probe was a strawman; the real train-side/#425-receiver test is still owed) |
| §14.3 v9c2 disposition (diagnostic + bank) | **STANDS** (executed: banked ep725) |
| §14.4 organ-B localization | **STANDS** (aiming input for the generator path) |
| §14.5 decision framework | **SUPERSEDED-BY §R6** (same spirit; the cheap-de-risk gate is now the measured ladder + bindingness harvest rather than the phase-efficacy probe alone) |
| §14.6 confirmed corrections | **STANDS** |
| §14.7–14.9 channel/hyperplane-native · level · intrinsic complexity | **STANDS** as design doctrine — and §R1/§R2 ARE these doctrines made mechanical (the plane is the level the score sees; the hyperplane arrangement is now the literal constraint set) |
| §14.10 pools non-additive + $0 gate sprint | **STANDS**; the pool×channel R-D curve instrument is now CONCRETE: bytes(band-scale) per §R2; S_floor ≈0.118 remains PROVISIONAL |
| §14.11 inverse-solve crux + hybrid of forms | **STANDS — and is now the ACTIVE path** (lattice + receiver + ladder + KKT are its realization); "the witness IS the inverse of the flattened scorer" is the reconciled program statement |
| §14.12 build manifest | **SUPERSEDED-BY §R7** (rows carried forward with updated statuses) |
| §15 pose-falls-out addendum | **STANDS with corrections**: "~9 orders" → **~5.7 orders** (V-1); "pose falls out" is a pre-registered prediction pending the bindingness harvest (V-3); carrier statement (solve near source + residual-vs-predictor) unchanged |

**No contradiction left standing:** the three-independent-axis picture, the "pose-shape the generator"
training program as a v10 default, the §6 launch chain, and the §14.12 manifest are the superseded
statements; everything else is scoped, not contradicted.

## §R6 The 9-step seal path (state review §7, statuses as of 2026-07-19)

| # | verb | item | status |
|---:|---|---|---|
| 1 | CONFIRM | #547 n600 lattice replay, all 50 chunks + aggregate | **CLOSED** (d_seg 9.664e-7, ULP class; aggregate receipt on VertigoDataTier). Frame-0-substitution interaction still open (rides the compact-f0 work) |
| 2 | MEASURE | #548 ŷ R-D ladder | **CLOSED at n24 advisory** (4 rungs, both distortions, exact decode, runtime custody). OPEN: compact frame0 description · joint two-plane packet · full-n600 compact-ladder measurement |
| 3 | DERIVE | #540 equations + reconciled SPEC | **PARTIAL→LANDED**: f32 law + pose law registered; KKT derivation corrected (V-1..V-6); **this document is the reconciled SPEC** (pending MAIN review). OPEN: band-law registration with positive anchors; KKT/waterfill law blocked on curves; pose-law registry-row re-emit (V-1) |
| 4 | BUILD | #543 production receiver + inflate | **PARTIAL**: factor-2 production receiver LANDED byte-close-proven; OPEN: compact `witness-y`/descriptor codec (stub fail-closed), factor-10 decision, full production renderer for the chosen description |
| 5 | MEASURE | P1/P2/P3 probes · coherent phase · p0_497 curvelet A/B · #270 · #518 8-vs-27 A/B | **OWED** (all trained-generator-path items; #518 branch merged, efficacy A/B artifact not yet produced) |
| 6 | MEASURE/DERIVE | #535/#536/#541 pool-channel · MDL · KKT waterfill | **RUNNING/OWED**: VJP-custody arm running (q/J/Lip custody → positive-band curves → bindingness maps → bytes(τ_pose)); waterfill INCONCLUSIVE at 1 point; #541 constructive solver is the production consumer |
| 7 | CONFIRM | #537 resumability + receiver/archive gate | **OWED** (continuous==resume equality, per-stage checkpoints preserved, parse-back byte-close, certified cleanup) |
| 8 | CONFIRM | exact evaluator closure — same archive.zip through `upstream/evaluate.py` contest-CPU **and** contest-CUDA | **OWED** (no axis inferred from the other) |
| 9 | SEAL | MAIN review + operator GO (paid only via #381 ≤ $20 governed) | **OWED** — this SPEC grants none of it |

## §R7 LAUNCH-READINESS MANIFEST — `launch_ready = false`

Every leaf named; nothing hides. A GO cannot even be REQUESTED until every row below is closed or
reasoned-N/A with MAIN review:

| leaf | owner | state |
|---|---|---|
| Compact ŷ descriptor/decoder (the video-derived representation of the shared plane; witness-y or per-class-carrier grammar) | #548 successor | **MISSING — the decisive science** |
| Compact frame0 description + joint two-plane packet (pose interaction measured, not source-f0 policy) | #548/#547 | MISSING |
| Positive-band custody: winner/rival VJP sidecar + PoseNet-6 Jacobian + measured Lip field | VJP arm (running) | RUNNING |
| Bindingness maps (is pose inactive at in-band solutions?) + bytes(τ_pose) curve | VJP arm → #536 | OWED (pre-registered prediction, V-3) |
| Band-scale sweep → measured R-D curves → #536 KKT waterfill solved on real rows | #536 | BLOCKED on the two rows above |
| #541 constructive solver at a positive operating point (min R(r) s.t. polytope ∩ ellipsoid, oracle-gated) | #541 | OWED |
| Production receiver for the CHOSEN description (witness-y-stub replaced; factor-10 decision receipt) | #543 successor | OWED |
| Full-n600 compact-ladder measurement through the production receiver | #548 successor | OWED |
| Trained-generator-path gates (only if that path ships content): #528 refusal wired · #529 compiler real success · #530 Init≠Fork · #531 T-quotient · p0_497 basis A/B · #518 efficacy A/B · phase-efficacy real test | §13/§14 owners | OWED |
| #537 full resumability certificate | #537 | OWED |
| Runtime/storage: decode < 30 min measured (not projected) · storage waterfall · certified cleanup | #543 successor | OWED |
| Exact same-byte contest-CPU + contest-CUDA replay | governed evaluator | OWED |
| Registry hygiene: pose-law row re-emit (V-1) · band-law registration with anchors · task/P0 mirror sync | MAIN | OWED |
| Operator GO (explicit; paid via #381 ≤ $20 governed launcher + lane claim) | operator | NOT REQUESTED |

**Binding restated:** no partial tally is an n600 result; no advisory row is a score; the provisional
0.118 floor and the 0.18804 borrowed snapshot have no pointer authority; only a receiver-closed exact
row on both contest axes may be compared against 0.1910828242.

## §R8 Triality + stores + task notes

- **DAG leg:** FEED-v10-reconciled (appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`
  same landing) — records this reconciliation + the V-1 correction.
- **DSL leg:** no new lever (this is a SPEC/reconciliation; the #529 real-success compiler remains the
  DSL build item; `Factor2IntegerScorerPlane` already counted).
- **Equations leg:** §R3 table; corrections landed in the law module + KKT memo appendix;
  `# FORMALIZATION_PENDING:reconciliation-doc-no-new-measured-finding` for this document itself.
- **Canonical-doc registry:** `spec_v10_capstone_cold_start_seeded` → `superseded_by:
  spec_v10_capstone_reconciled`; new active entry points here (per the #533 anti-duplicate-SoT
  contract). Fold future v10 design into THIS file, never a new one.
- **Task notes (MAIN updates ledgers):** #521 (SSoT reconcile) — deliverable LANDED pending MAIN
  review; #540 — reconciled-SPEC leg landed, equation legs per §R3; #547 leg 1 CLOSED; #548 n24 leg
  CLOSED (production adoption open).
- **STORES CONSULTED:** base SPEC §0–§15 · state review 20260719 · lattice composition memo +
  aggregate n600 receipt · ladder memo/JSON/CSV · #543 memo + byte-close receipt · #549 memo +
  receipts · KKT memo (+ corrections) · pose/f32/factor-2/rank-4 registry rows · frozen-scorer +
  fractal factorizations · null-subspace measure · gt_n600.npz (margins re-measured) ·
  verification memo (this arm) · `docs/operating_manual_craft_handoff.md` · CLAUDE.md.

**Pointer 0.1910828242 UNMOVED — this SPEC is MEANS.** The reconciled program exists to land a
byte-closed `upstream/evaluate.py` n600 exact row below it, then toward sub-0.15.

## §R9 PROPOSED_PENDING_MAIN_REVIEW — yhat-native generator and receiver fold (2026-07-19)

This section is an append-only proposed correction from
`lane_yhat_native_generator_20260719`. It has no authority until MAIN reviews and accepts the branch.
It specifically narrows §R1.1-3, §R4, and the corresponding supersession rows; it does not silently
rewrite their historical evidence.

### §R9.1 Measured equivalence boundary

The new n24 real-pair measurement solved and replayed both scorer planes with exact rational
numerator equality on `28,311,552 / 28,311,552` samples and zero failures. All 24 pairs classify as
`EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS`; none is bit-identical through the frozen CPU oracle, and
there are seven SegNet argmax disagreements. Mean yhat-minus-direct deltas were
`Delta d_seg=-2.1175947040319443e-7` and `Delta d_pose=-0.000240008036286099` on the
`[macOS-CPU advisory] NON-PROMOTABLE` axis. Receipt SHA-256:
`1ad1cf84672c696b46f62ca8586bb29d5c70f55de5803902b6c37666e5b85c0f` (revision 2, binding the
tool bytes proposed for this branch).

**Proposed correction to §R1.1/§R1.3:** the scorers share the same spatial resize geometry and
frame-1 plane, but PoseNet consumes a temporal pair `(yhat_f0, yhat_f1)` while SegNet consumes only
`yhat_f1`. One exact realization is required per independently described plane. A single
repeat-frame1 realization does not buy general two-plane Pose closure.

**Proposed correction to §R1.2:** exact rational plane realization is demonstrated for the supplied
feasible planes under the recorded receiver arithmetic. “Exact plane” must not be promoted to
bit-identical frozen-oracle output; `f32_receiver_arithmetic_exactness_admissibility_v1` remains
load-bearing.

### §R9.2 Runtime correction: integer and arbitrary-rational receivers are different lanes

- The §R4 `3.8 min` derivation belongs only to the fast factor-2 **integer-yhat repeat-frame1**
  receiver: n12 wall `4.53 s` for one integer frame-1 plane realized once and copied, linearly
  derived n600 `3.775 min`. It does not measure two independently described Pose planes.
- The new arbitrary-rational exact receiver measured `597.7790400451 s` of solve work across n24,
  `12.4537300009 s` per plane on average. Its n600 two-plane projection is
  `14,944.4760011276 s = 249.0746000188 min`.
- Therefore this implementation on the supplied feasible donor-derived fractional planes has a
  derived n600 projection beyond the 30-minute boundary before generic expansion, packaging, and
  output I/O. Full n600 decode remains unmeasured.

**Proposed correction to §R1.2/§R4:** “realization solved and free” is scoped to the integer-yhat
factor-2 domain, its declared arithmetic, and the measured repeat-frame1 arrangement. A fractional
yhat generator must either (a) learn two independently described integer-uint8 scorer planes and
measure the induced Seg/Pose debt plus two-plane runtime, or (b) use a substantially faster exact
rational solver. Neither child is presently closed.

### §R9.3 Representation correction and proposed head

The incumbent ep725 trunk already evaluates its MLP on the `384x512` grid. A yhat-native head does
not reduce MLP sample count by the camera/scorer coordinate ratio
`(874*1164)/(384*512)=5.1744384765625`. It instead declares the existing three-channel head output to
be the scorer plane, removing the differentiable bicubic-upsample, camera uint8/STE, and bilinear
downsample from the training coordinate. Contest decode still owes camera `874x1164` uint8 output
through an exact receiver.

Proposed training semantics:

```text
G_theta(z)_f0,f1 in R^(384x512x3)
  -> SegNet(frame1)
  -> PoseNet(rgb_to_yuv6(frame0, frame1))

decode: compact z -> deterministic NumPy-fp32 G -> yhat -> exact receiver P -> camera uint8
        with exact A(P(yhat)) replay
```

Initial trunk, codes, FiLM, output head, SDF/palette/carrier state, chroma, EMA, curriculum stages,
and resume checkpoints stay structurally available. Camera-space AA/noise/`ker(A)` terms become
no-op or must be re-derived; Seg, Pose, counted-rate pressure, EMA, and stage-boundary persistence
remain. This is a design, not a trainer landing.

### §R9.4 Proposed counted description and PDW2 dependency

`YhatNativeDescription.v1` is the proposed strict packaging spine:

```text
counted description -> strict canonical parse/re-encode -> deterministic NumPy-fp32 expander
                    -> ordered two-plane yhat -> exact receiver -> camera uint8 -> exact A replay
```

Its header binds version, source/runtime hashes, float32 order/arithmetic, geometry, pair count,
seed, canonical section order/lengths/hashes, and expanded two-plane hash. Counted sections include
PDW2/PDP2 target payload, learned/video-fitted weights, pair/frame codes, fitted seeds/scales/palettes
and entropy parameters, residuals, framing/hashes, and archive overhead. Free rule-118 material is
only generic code/operations, fixed non-video priors, PDW2 evaluation code, the lattice algorithm,
camera assembly, and generic coder code. Any scorer-derived shipped payload counts; scorer weights
and GT tables remain forbidden.

Sibling PDW2 commit `edf47756ba629e079a2a63233bf8f0293cf85f3d` supplies the proposed inner
certificate: margin-preserving `138` raw / `133` Brotli-q11 bytes (20 float32 coefficients) and
partition-only `134` raw / `122` Brotli-q11 bytes (19 scalars). It remains
`TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`; MAIN must review that commit before integration.

### §R9.5 Chroma/Pose and readiness disposition

PoseNet consumes both RGB planes after RGB-to-YUV6; SegNet consumes frame 1. Neither frame nor
chroma can be deleted by a Seg-only argument, and integer projection is a new scorer-debt A/B rather
than an exact reparameterization of the fractional incumbent. Direct dense yhat remains rate-dead.

The n24 receipt gate is closed in the default-OFF `YhatNativeGeneratorPolicy`; receiver/archive
gates remain owed. The policy and keyword-only DSL factory add no argv, epoch, launch, score,
promotion, or pointer authority. Proposed DAG:

```text
compact description -> deterministic expander -> integer-fast OR exact-rational receiver
                    -> exact replay -> archive parse-back -> separate contest CPU/CUDA
```

Open leaves: compact generator, receiver-choice A/B, full n600 measured decode below 30 minutes,
archive parse-back, same-byte contest-CPU, same-byte contest-CUDA, MAIN review. The pointer remains
`0.1910828242 [contest-CPU Linux x86_64]`.

## §R10 SCORER-NATIVE DOCTRINE ADOPTION (2026-07-23 — operator 9-directive wave; canonical detail: `.omx/research/ddm_scorer_native_doctrine_and_synthesis_20260723.md`)

**The vehicle's coordinates, instruments, and pipeline are re-based to the scorer's internal dimensions.** This section BINDS the v10 program; the doctrine doc carries the full 8 points + synthesis. Vehicle-level deltas:

1. **State space.** The constrained program of §R2 now runs measure→solve→represent→realize through scorer-internal coordinates: the exact gaze field λ_k = ∂S/∂z_k (tractable: 6 pose scalars/pair + the §R1 rank-4 head), per-layer Jacobian factors (any interaction by exact composition), layer-relay (multiple-shooting) solves with matching conditions replacing single-shot RGB→argmax inversion. COMPLIANCE UNCHANGED: encode-side only; the archive always carries an RGB-generating description (no scorers at inflate; rule 118 intact).
2. **Stream schema gains PROSODY.** The counted description = symbol layer (worldsheet/grammar/curves) + prosody layer ({amplitude, frequency, phase, contrast, channel-energy, texture statistics} as priced coordinates with their OWN tolerances). Named in-scorer mechanisms: eval-BN frozen statistics (per-layer shift loss) + SE global gating (non-local coupling). Dead frequency bands (exact R transfer + stem Nyquist) priced 0 by construction. Pose: photometric amplitude gradients ARE the PoseNet signal — flat paint pose-blind by construction; the §R9 pose stream must be amplitude-structured + ξ-advected.
3. **THE THREE WALLS (closure-classified; supersedes any "missing data" framing):** W1 realization/paint floor 0.0284 = representation mismatch (cell-grid + wrong prosody; pt1 attacking, 4-way mechanism split owed) · W2 pose code→photometry inverse ABSENT = extraction gap (zero-pose floor 3.09 kills pose-absent packets — e2 FORMULATION-scoped) · W3 description R-D between 706-param/0.0248 and full-lattice/410MB = the §15 tolerance-ladder domain (dr2b charting, prosody-priced).
4. **Instrument chain (modernization owed BEFORE consumption):** sn1 scorer-native diff instrument (reusable tac module) → at1 TOTAL FACTORED INFLUENCE MAP with Phase-0 modernization of the #36 atlas engine + costate organ (atlas = single λ producer; organ re-bases as consumer view — closes OWED pair/site-λ; re-validates the BACKTESTED-FAIL adjoint leg) → rs1 relay solve. Philosophy checklist binds first rows: non-additive pools · uint8-gated amplitudes · NO-FAKE at instruments · n600/P0/freshness · no-recency-floor · first-rungs.
5. **Forward chain (one path, gates named):** sn1+pt1+dr2b landings [each with a DIRECTIVE-CONSUMPTION TABLE] → menu1 realized-flip menu → at1 → rs1 → compose per c1 waterfill (prosody-priced, pose present) → exporter → byte-close → R6 dual-axis exact eval → pointer. #366 fires ONLY on the measured class-(iii) leftover. Closure axiom binding: every wall verdict classifies {extraction | representation | compute} — never data.

**Quadrality sync note (this landing):** DAG ✅ (9 FEEDs 2026-07-23) · tasks ✅ (#656–#660) · equations ✅ today's measured legs (ddm_e2_export_stage_laws_v1 + dr1 anchor on ddm_v17_realized_validity_ratio_uint8_v1); doctrine-level LAWS register on at1/sn1 measurement (principles are not equations until measured — no premature registration) · DSL leg N/A-TODAY-WITH-REASON: no trainer lever changed; prosody-tolerance + relay-DOF Lever objects land WITH their builds (dr2b/rs1) per the lever-at-build rule.
