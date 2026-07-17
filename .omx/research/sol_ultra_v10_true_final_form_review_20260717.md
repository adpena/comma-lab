# SOL-ULTRA fresh-eyes review — V10 true-final-form audit (2026-07-17)

<!-- MAIN annotation (consumption fold, not SOL text): verdict_scope declaration for the gate.
verdict_scope: formulation/implementation/custody — EVERY negative below is one of {FORMULATION (generic-flat-cell
theorem, w_pose composition, T-implicit-in-trunk), COMPILER IMPLEMENTATION (v10 presence-checker), VEHICLE
COMPOSITION (fresh-init vs fork laws), IMPLEMENTATION/CUSTODY (structured-init frac mis-scale, range(A) realized
exactness), CAUSAL/TRAJECTORY (round-6 Force-3/crest reading)}. NONE is a FAMILY or PARADIGM kill: the review
explicitly does NOT reject the generator-first family, linear blind-subspace removal, local projection/trust-region
methods, Fisher-native optimization, per-class carriers, or event-continuation. The round-table's 'FALSIFIED'
tokens are each qualified at their narrowest supported level in-place ('FALSIFIED at generic-flat-cell formulation
scope', etc.). -->

**Review verdict:** **`NOT_TRUE_FINAL / NOT_LAUNCH_CERTIFYING / SPEC_AND_COMPILER_REWRITE_REQUIRED`**.

**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**. This review produced no archive, official evaluator row, launch, promotion, or pointer authority. All measurements below are explicitly advisory.

**Authority and containment:** delegated read/derive/review authority; read-only access to the live c2 and prior run directories; bounded local CPU probes only; no training, no paid/GPU dispatch, no live-run mutation, no checkpoint-weight loading, and no task-artifact edits outside this memo (the launcher-mandated delegation checkpoint ledger is the only bookkeeping write).

**Objects reviewed:** V10 SPEC and compiler branch; V10 buildable-components, #518 resume/warm-up, Fisher Arm A, forces/triggers Arm B, skip/EMA Arm C, curvelet, and phase-carrier branches; V7.5/V8 contracts; frozen upstream scorer/evaluator source; GT cache; factorization/necessity/recursive-fractal/texture/carrier corpus; canonical-equation and DAG surfaces; live and prior telemetry; P0 digest and deferral ledger.

## Executive answer

V10 has the right strategic direction—task-space witnesses, generator-first rate accounting, explicit boundary geometry, train-least discipline, typed gates, and exact evaluator closure—but its current “true final” claim rests on four false or unclosed premises:

1. **The generic flat-cell/`GEOM-ONLY` theorem is contradicted by the strongest measured realization evidence.** A perfect flat realization of the target partition measured `d_seg=0.0416`, while textured realization measured `0.0048`; the sufficient statistic is at least `W=(G,xi,T)`, not `G` alone. Nonlocal VJPs show that interior texture can move remote decisions. V10 currently leaves `T` implicit inside an undefined counted trunk and risks paying twice for “solved” geometry.
2. **The “exact range(A)” consumer is exact only over real-valued tensors.** On a real cached source frame, the float identity held to `1.71e-13`, but uint8 clamp/round changed the scorer-plane input by as much as `62.74`. The current implementation has no lattice-feasible realization proof, scorer equality receipt, or render-loop consumption.
3. **Arm B double-applies the pose marginal.** The live/V10 loss already uses `sqrt(10*d_pose)` under `--score-domain-loss`; multiplying it by `5/sqrt(10*d_pose)` squares the contest marginal. `w_pose=1` is the exact score-domain coefficient.
4. **The V10 “compiler” is a presence-checking blocker reporter, not a compiler.** It validates `hasattr` and `Path.exists`, has no success path, emits no parser-verified argv/LawRef manifest, and does not bind the new build-wave components or semantic receipts. It can fake-pass on empty/stale/adverse artifacts while still being unable to return a config.

**Single highest-impact true-final delta:** make `T` an explicit class-/cell-conditioned residual with unique custody, then train only the quotient residual after deterministic `G`, `xi`, seed, solve, and projection components. This directly addresses the measured `0.0416 -> 0.0048` realization gap and prevents the counted trunk from silently relearning—and double-paying for—the supposedly solved representation. The immediate launch-safety fix is separately mandatory: refuse/remove `PoseMarginalWeightLaw` whenever score-domain loss is enabled.

These are formulation and implementation/custody verdicts. They do **not** reject the generator-first family, linear blind-subspace removal, local projection/trust-region methods, Fisher-native optimization, per-class carriers, or event-continuation as paradigms.

## Evidence and label discipline

- **MEASURED** means a cited real artifact or the bounded local probe in this review.
- **DERIVED** means exact algebra from pinned source or a stated mathematical implication.
- **INFERRED** means a mechanistic reading consistent with measurements but not isolated causally.
- **ASSUMED** means an open premise requiring the named measurement.
- Live/prior run rows are `[macOS-CPU/MLX advisory; non-promotable]` unless the row itself says otherwise.
- `d_seg` training telemetry is not an official contest score. Only exact archive bytes through `upstream/evaluate.py` on the declared contest axis can move the pointer.

**Citation shorthand:** `SPEC` is `51a1feb649:.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md`; `ARM-B` is `c896c99c5b:.omx/research/arm_b_forces_triggers_build_20260717.md`; `LIVE-LAUNCH` is `experiments/results/levelset_n600_witness_20260717T113932Z/launch.sh`; `MAIN-P0` is the canonical main-worktree `.omx/state/operator_p0_ledger.jsonl`, not this review worktree's older snapshot.

### Bounded probes performed here

1. **Range projector realization probe — MEASURED, local macOS-CPU, no scorer forward.** Source: `gt_n600.npz` SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, `gt_f1[0,:,:,0]`. The branch’s exact resize matrices and QR projector gave:

   | realization | `max |A(x')-A(x)|` | mean absolute delta |
   |---|---:|---:|
   | real-valued `P_A x` | `1.7053e-13` | `6.47e-15` |
   | unbounded integer `round(P_A x)` | `0.49823` | `0.15796` |
   | valid uint8 `clip(round(P_A x),0,255)` | **`62.73883`** | **`0.24434`** |

   `P_A x` ranged from `0` to `370.61786`; `598/1,017,336 = 0.05878%` of pixels in that channel were out of gamut. This falsifies only the current **realized exactness** claim, not the fp64 projector theorem.
2. **Structured-init semantic probe — MEASURED, local macOS-CPU, real n600 label cache.** `identify_static_hood_class(lstars)` returned class `4`; class `2` was the top static region and class `4` the bottom hood. The same function therefore cannot implement the declared sky detector. Its `frac_of_frame` diagnostics were `139.4, 3.51, 297.1, 7.43, 152.6`, proving the field is mis-scaled by the frame count.

## Recursive adversarial rounds and challenged assumptions

| round | shared assumption attacked | result |
|---:|---|---|
| 1 | the target argmax partition plus pose is a sufficient statistic; large-margin cells need no texture | **FALSIFIED at generic-flat-cell formulation scope** by the flat-palette floor and nonlocal VJP evidence |
| 2 | scorer feasibility is a composition of exact/convex projections whose fixed point is globally solved | **NOT ESTABLISHED**: nonlinear scorer preimages are nonconvex; uint8 realization breaks the linear identity; local pseudoinverses do not prove a global feasible fixed point |
| 3 | existing modules make all eight pillars operationally buildable | **FALSIFIED at implementation-custody scope**: wrong semantic detectors, unconsumed projector/coder/carriers, and fresh-init/resume-law mismatch |
| 4 | Fisher and score-marginal laws are correctly composed into the training objective | **FALSIFIED for two named formulations**: binary Fisher labeled categorical; output cotangent preconditioner labeled parameter natural gradient; pose marginal applied twice |
| 5 | fail-closed blocker reporting is equivalent to a launch-safe compiler | **FALSIFIED at compiler implementation scope**: existence/symbol probes, no successful return, no parser/LawRef/receipt semantics |
| 6 | the live post-ep725 regression diagnoses Force-3 and a clean pose-conditioning crest | **FALSIFIED as causal/trajectory interpretation**: a no-Force-3 control has the same cold-Muon fingerprint, and later sigma telemetry oscillates/rebounds |

---

## AXIS 1 — task-space equations, representation, carriers, basis, and terms

### CRITICAL A1-C1 — the generic flat-cell certificate drops a measured-necessary `T`

**Claim under review.** V10 calls class interiors Fisher-flat, treats generators/border profiles as sufficient, and assigns Road/Undrivable bulk plus probe-gated Movable interiors to `GEOM-ONLY` generated fills (`51a1feb649:.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md:179-217`). It simultaneously budgets “trained trunk weights” as the main counted item without defining their residual subspace (`:268-289`).

**Contrary evidence.** The measured texture synthesis states that a perfect target partition rendered with flat per-class colors floors at `d_seg=0.0416`, versus `0.0048` for a textured witness, and derives `W=(G,xi,T)` (`.omx/research/fable_synthesis_texture_partition_20260710.md:31-46,58-78`). The real SegNet VJP cure is nonlocal and sign-alternating: only `0.00-0.15` of energy lies within radius 4, up to `83%` of one bucket lies outside radius 36, and flat shifts are nearly orthogonal (`.omx/research/c2_perclass_stratum_carrier_taxonomy_20260716.md:79-94`). The deferral ledger still says the raw nonlocal tail receipt is absent and must precede locality/factorization claims (`.omx/state/deferral_ledger.md:173`).

**Broken assumption.** Pointwise argmax stability is not input-space invariance through a deep nonlinear receptive field. Also, categorical Fisher is small—not identically zero—at finite confidence. A pixel inside one output cell can change a remote boundary.

**Verdict scope.** `FORMULATION: V10 generic flat-cell/exhaustion theorem`. Not a negative on generator-first representations or on any cell individually proven texture-free through the realized receiver.

**Concrete fix.** Restore `T` as a first-class, class-/cell-conditioned stationary texture/metamer residual. Define an ordered quotient/direct-sum contract such as

`x = Decode(G,xi,seeds,solves) + T_class + T_shared + r_irreducible`,

with unique byte ownership and no component allowed to relearn an earlier component. A `GEOM-ONLY` row must pass: real receiver decode, uint8/R, exact Seg/Pose comparison, a nonlocal-radius/tail receipt closing D24, and a remove-`T_c` ablation showing no material evaluator debt. Until then the table should say `PROBE`, not `CERT`.

### CRITICAL A1-C2 — `P_A`, cell generation, and uint8 scorer preservation are different maps

**Evidence.** The range projector implementation correctly derives an orthogonal projector onto `range(A^T)` and proves `A(P_Ax)=A(x)` over fp64 (`35dc3b70f3:src/tac/boundary_math/range_a_projection.py:17-41,70-150`). But its composition law calls the linear projector and Laguerre-cell generator projections onto the same sigma-algebra (`:4-15`), and V10 repeats that equivalence (`SPEC:87-103`). The tests verify only float residual/idempotence and MLX parity; they never clamp, quantize, decode, or score (`35dc3b70f3:src/tac/tests/test_range_a_projection.py:17-91`). The consumer returns unrestricted fp32 and contains no gamut/lattice solver (`range_a_projection.py:131-150,233-253`).

**Measured falsifier.** The local source-frame probe above preserved `A` to `1.71e-13` in reals but incurred a `62.74` maximum scorer-plane difference after valid uint8 realization.

**Broken assumption.** `P_A` is a linear row-space projector; `P_cell` is a nonlinear representation/generator; `Q_uint8` is a discrete feasibility map; `N_seg` and `N_pose` are frozen nonlinear networks. These maps generally do not commute.

**Verdict scope.** `IMPLEMENTATION/REALIZATION: P3 exact score-preserving consumer`. The real-vector-space theorem remains valid and useful as a training prior or null-energy diagnostic.

**Concrete fix.** Type and receipt the chain explicitly:

`x -> P_A -> P_cell -> Q_uint8 -> A -> {N_seg,N_pose}`.

Measure the commutators. Replace unconstrained orthogonal projection with a bounded integer/preimage solve, or apply it only as a soft training regularizer. Admission requires exact decoded bytes, `A(Q(x'))` equality/tolerance, Seg argmax equality, Pose first-six tolerance, and full receiver/archive custody.

### HIGH A1-H1 — “geometry solved; only Dykstra” overstates a local nonconvex inverse

The frozen source explicitly contains nonlinear EfficientNet and FastViT maps (`/Users/adpena/Projects/pact/upstream/modules.py:61-84,103-113`); only resize/YUV stages are linear. The projection memo acknowledges `N^-1` as a local Jacobian pseudoinverse (`.omx/research/projection_unification_and_eight_lenses_20260715.md:85-98`) and lists nonconvex multi-fixed-point behavior as an open lens (`:108-123`). The necessity memo scopes `N^-1` to first order and says exactness holds only in penultimate-patch space (`.omx/research/necessity_solver_inverse_factorization_20260715.md:123-132`).

The preimages of Seg argmax cells and Pose tubes through nonlinear CNNs are generally nonconvex. Classical Dykstra convergence does not follow from the rank-4 **final head** being linear. Reachability of sampled sub-LSB local flips does not prove global intersection feasibility, contraction, uniqueness, or shortest-program selection.

**Broken assumption.** Local linearizability of the frozen networks and convexity of their final heads do not make the end-to-end feasible set globally convex or make alternating projection globally convergent.

**Verdict scope:** `FORMULATION: global convex Dykstra/fixed-point theorem`; local sequential projection/trust-region methods remain open.

**Fix:** rewrite “geometry solved / no training left” as a local sequential-convexification hypothesis. Each iteration must have a trust region, exact realized-through-R acceptance, monotone score/byte receipt, cycle detection on the uint8 lattice, and a failure/empty-intersection certificate. Train only the residual that these certified local solves cannot close.

### HIGH A1-H2 — per-class carriers can double-pay the counted trunk

V10 calls the V8 kit “as built,” generators-first, and awaiting archive composition (`SPEC:144-161`), while the counted trunk remains the dominant undefined payload (`SPEC:275-289`). V8 itself records mask-not-score, boundary theft, Pose seam, and merge/diff/correct risks and requires each carrier to be independently byte-closed and bit-exact before composition (`.omx/research/SPEC_v8_perclass_decomposition_20260708.md:55-86,97-107`). The independent recursive-fractal audit found no DCB encoder/parser/exact alternate archive/full-n600 receiver row and labeled receiver rate `NO_VERDICT_RECEIVER_RATE_CUSTODY` (`.omx/research/codex_findings_recursive_fractal_optimal_representation_v9_20260714_codex.md:55-61,85-105`).

**Broken assumption:** module existence is not a unique-home decomposition or Kolmogorov-minimal carrier proof.

**Verdict scope:** `P7 composition/as-built assertion`, not the per-class-carrier family.

**Fix:** define component order and a unique-home residual projection; publish isolated and composed exact archive bytes, Seg delta, nonlinear Pose contribution, receiver survival, and pairwise interaction signs. The trunk may train only on the residual after deterministic carriers; otherwise it silently duplicates them.

### HIGH A1-H3 — Fisher actuation is useful but mathematically mis-scoped

Arm A’s trainer labels `0.5*sech^2(m/2)` the exact categorical Fisher trace (`443aaeb41a:experiments/train_witness_realized_through_R_mlx.py:1140-1181`), while the canonical helper correctly says it is the **two-class annulus** identity. For five classes, the exact trace is `1-sum_k p_k^2`; a five-way tie has trace `0.8`, whereas the binary surrogate returns `0.5`. Triple saddles are exactly where discarding sub-runner-up mass is least justified.

The backward transform computes a logit-space `g_z^+` cotangent before VJP (`:1184-1221`). For renderer parameters `theta`, this yields `J^T g_z^+ grad_z L`; it is not generally the parameter natural gradient `(J^T g_z J)^+ J^T grad_z L`. The equation module’s narrower “cotangent preconditioner” name is the honest scope. Trace-weighting is importance weighting by curvature, not by itself Riemannian steepest descent.

**Broken assumption.** A binary top-two identity is not the exact K=5 categorical trace, and an output-space Fisher transform is not automatically the pullback natural gradient in renderer-parameter space.

**Verdict scope:** `global-exact K-class Fisher density` and `parameter-natural-gradient` claims. The binary annulus prior and output-space Fisher cotangent arm remain valid experiments.

**Fix:** use full K=5 probabilities for `1-sum p^2` and retain binary target-vs-runner curvature as a labeled approximation/prior. Rename the transform or solve the pullback Fisher with JVP/VJP CG. Keep Euclidean cosine, Fisher cosine, and relative norm together; do not declare either metric universally authoritative without the selected-metric custody row. Apply the already-built Fisher bit allocator in the real coder/receiver before calling the six-surface program complete.

### HIGH A1-H4 — structured-init readiness is semantic fake-green

The structured defaults promise class-self-detection (`35dc3b70f3:src/tac/witness_dsl/spec_v10_structured_init_defaults.py:23-26,60-117`). Yet:

- sky calls `identify_static_hood_class`, whose objective explicitly maximizes bottom concentration (`src/tac/boundary_math/hood_static_component.py:94-127`);
- lane says self-detecting while `build_structured_lane_sdf` defaults to `lane_cls=1` (`src/tac/boundary_math/lane_sdf_component.py:318-342`);
- per-dash anchors point to `fit_lane_line`, whose periodic dash model conflicts with the measured comb refutation;
- the readiness probe checks import/symbol existence only (`spec_v10_structured_init_defaults.py:126-190`), and tests assert strings/types rather than real semantic output (`35dc3b70f3:src/tac/tests/test_spec_v10_structured_init_defaults.py:28-39`).

The real n600 probe returned hood class 4; the top static class was 2. `HoodClassEvidence.frac_of_frame` at `hood_static_component.py:123` omits division by `n`.

**Broken assumption.** Symbol presence and a generic static-class heuristic do not certify that each structured seed has the intended spatial/class semantics on the real video.

**Verdict scope:** `P1 structured-default implementation`, not structured seeding.

**Fix:** implement distinct top-static/sky and bottom-static/hood detectors, data-derived lane-class detection, and an actual nonperiodic per-dash anchor representation. Test distinct class IDs, masks, SDF contribution, and rendered labels on the real cache; correct diagnostic scaling.

### HIGH A1-H5 — content-priced coder is not a receiver-closed archive result

The coder has a real self-describing int8-dequant round trip and deterministic entropy best-of (`35dc3b70f3:src/tac/codec/content_priced_coder.py:184-205,250-342`). But its “better-or-equal” claim compares **content streams** against a baseline number that omits the new manifest/scales; the full blob can be larger (`:27-38`). It excludes all `__cfg` state and the feature bank without proving that the actual receiver reconstructs every necessary generic/config input (`:91-105,345-349`). #336 Fisher allocation is explicitly skipped (`:250-280`). Tests stop at toy/donor checkpoint dequantization and never build an archive, run inflate, reproduce witness pixels, or invoke the frozen evaluators (`35dc3b70f3:src/tac/tests/test_content_priced_coder.py:43-83,111-139`).

**Broken assumption.** A deterministic checkpoint-tensor round trip and a smaller inner stream do not establish a smaller complete counted archive or an executable receiver.

**Verdict scope:** `P4 operational/content-optimal claim`; the entropy codec primitive is real.

**Fix:** integrate it into the counted archive grammar and receiver, include every required video-derived/config byte, prove decoded pixel identity or a measured evaluator tradeoff, and compare **full archive.zip bytes** on the exact same witness. Gate selection by score units per byte, not stream size.

### MEDIUM A1-M1 — “Kolmogorov-optimal” and `K/H=0.47` are labels beyond the measurement

The necessity experiment measured one generator-coded spatial chain representation versus one entropy-style chain code at a stated tolerance (`.omx/research/necessity_solver_inverse_factorization_20260715.md:85-105`). It also says the spatial K-ladder exceeds the current archive and is not a d_seg claim (`:103-105,123-132`). This is a useful description-length upper bound, not computable proof of Kolmogorov complexity or global optimality.

**Broken assumption.** One measured codec-pair ratio is neither Kolmogorov complexity nor a global minimum-description proof.

**Fix:** name it `L_G/H_chain=0.47` for the measured codec pair, and call V10 a **candidate minimum-description task program** until an exact receiver archive establishes its length and score. Keep the philosophical Kolmogorov test as a search discipline, not a proven optimality label.

### MEDIUM A1-M2 — basis takeover needs a clean factorial and the full survivor set

The curvelet branch is fireable, but earlier evidence was sensitive to representation authenticity, receiver batch geometry, and equal-values-not-equal-bytes confounds. V10 correctly keeps the Fourier ban WARN-only and makes the matched-byte A/B decisive (`SPEC:514-525`). The remaining risk is attribution: basis and AA/receiver changes must not move together, and shearlet/step-native remain defensible survivors.

**Broken assumption.** A fireable two-arm basis comparison does not rank the survivor family when receiver, antialiasing, byte identity, or omitted bases can explain the delta.

**Fix:** hold receiver, AA, seed, scalar count, support, and exact archive bytes fixed, or run a basis×AA factorial. Include Fourier, genuine curvelet, compact shearlet, and step-native/local boundary bases. Report per-stratum Seg, Pose, rate, and parse-back survival. Do not infer fresh-start ranking from saved-OFF donor projections.

### MEDIUM A1-M3 — phase-carrier/state rate remains an adoption question

V10 lists phase carriers as TBD (`SPEC:281-283`). Arm G has now built and measured the STORE leg: its object-domain dash codec is `11.3x` smaller than naive raster, but still `16.6x` over the anchor budget; blink-back is `0.787`, and the named next rate crux is a free persistence-class visibility generator (`MAIN-P0:183`, `2026-07-17T21:00:47Z`). This is real progress, but it is not yet a byte-optimal receiver or score result.

**Broken assumption.** Object-domain compression and state-prediction accuracy alone do not establish receiver-closed score-unit-per-byte adoption.

**Fix:** consume the Arm G receipt; implement/test the visibility generator and the named prior-transfer cures; then require the decisive joint through-R `d_seg` A/B at the c2 ep725 EMA boundary (TRAIN event fallback + STORE carrier), exact parse-back, and score-unit-per-byte comparison versus per-dash anchors and persistence. Preserve the built STORE leg without promoting it by its object-domain ratio alone.

### LOW A1-L1 — preserve the correctly scoped representation facts

Frame-0 Seg freedom and the shared resize/YUV factorization follow from frozen scorer source (`/Users/adpena/Projects/pact/upstream/modules.py:35-58,86-113`); Lane’s no-safe-interior exception and formulation-scoped carrier negatives remain explicit in V8 (`.omx/research/SPEC_v8_perclass_decomposition_20260708.md:55-86`).

**Broken assumption.** Correctly scoped invariants and negative scopes can be safely reopened or generalized while repairing adjacent overclaims.

**Fix:** preserve frame-0 Seg freedom; shared resize factorization; BT.601 luma/chroma basis; 2x2 Pose chroma averaging; gauge canonicalization as a precision—not established rate—lever; Lane’s exception; per-pair rather than global boundary asymmetry; and formulation-scoped negatives.

---

## AXIS 2 — compiler, cold-start order, constants, DE/ODE/PDE, and resumability

### CRITICAL A2-C1 — `w_pose(t)` squares the contest marginal under the actual loss

The live c2 config enables `--score-domain-loss` with `--w-pose 1.0` (`experiments/results/levelset_n600_witness_20260717T113932Z/launch.sh:32-34`). The canonical micro-batch loss computes

`pose_term = sqrt(10*pose_l + eps)` and `L = ... + w_pose*pose_term`

(`src/tac/boundary_math/levelset_micro_batch_loss.py:326-329`). Arm B then sets `w_pose=5/sqrt(10*d_pose)` (`c896c99c5b:experiments/train_levelset_witness_realized_through_R_mlx.py:12069-12090`). Therefore

`dL/dd_pose ~= (5/sqrt(10*d_pose))^2`,

not the contest marginal. The tests finite-difference the standalone score derivative and search source strings; they never differentiate the composed loss (`c896c99c5b:src/tac/tests/test_w_pose_marginal_law.py:29-34,102-110`).

**Broken assumption.** The contest marginal may be applied once as an outer weight even when the inner loss already is the exact square-root score term.

**Verdict scope:** `COMPOSITION`. Adaptive pose weighting is not rejected.

**Concrete fix:** compile-refuse `PoseMarginalWeightLaw AND score_domain_loss`. Under score-domain loss, use `w_pose=1` for the exact score objective. The marginal law is admissible only when the underlying pose term is raw `d_pose` (or a measured calibrated proxy), with one composed-gradient regression test.

### CRITICAL A2-C2 — the V10 compiler has no success path and fake-passable gates

The module honestly calls itself a skeleton (`51a1feb649:src/tac/witness_dsl/spec_v10_capstone_20260717.py:11-19`). It probes five levers with `hasattr` (`:162-181`) and eight artifacts with `Path.exists` (`:184-196`). Empty files, stale receipts, an adverse verdict, wrong source hashes, wrong axis, wrong run, or symbol stubs all pass. `spec_v10_status.clear` can be true, but `compile_v10_capstone_launch_config` then unconditionally raises `post_gate_fold_owed`; it never returns config/argv (`:199-230`). This contradicts the SPEC gate that says it returns a zero-blocker config (`SPEC:311-324`).

The claimed “13 blockers” is contextual, not invariant: `5+6+2=13` in an artifact-poor worktree; canonical main already has the GT cache, so the same presence count is 12. Import failure can also collapse five lever blockers into one. More importantly, presence count is not readiness.

**Broken assumption.** Fail-closed presence reporting is equivalent to semantic compilation and launch readiness.

**Verdict scope:** `COMPILER IMPLEMENTATION`, not the fail-closed principle.

**Concrete fix:** emit a typed `WitnessProgram`, resolve every LawRef, compile via the canonical constants compiler, parse with the real trainer parser, and return `(argv, manifest, config_hash)`. Every receipt validator must reopen bytes, verify SHA/schema/provenance/axis/verdict/coverage/producer-consumer identity, and enforce the real semantic consumer.

### CRITICAL A2-C3 — a fresh-init vehicle cannot use fork/resume-only birth laws

V10 explicitly says fresh init and “never weights” from v9c2 (`SPEC:24-29`). Yet P2/P5 depend on `ForkHeadSolve` and `ForkEmaClearance` (`SPEC:73-85,117-128`). The #518 memo says `ForkHeadSolve` requires `--resume-from` (`ead2a13760:.omx/research/p0_resume_warmup_geometry_build_20260717.md:39-49`); `ForkEmaClearance` is a retreatment-fork behavior. These controls cannot bind at a cold birth.

**Broken assumption.** A resume/fork treatment can serve as a cold-start initialization law without inherited state or a distinct fresh-state contract.

**Verdict scope:** `VEHICLE COMPOSITION`.

**Fix:** split fresh-start and fork semantics. Build `InitHeadSolve` against cold structured seeds and a fresh-state head, then initialize optimizer/EMA consistently. Keep `ForkHeadSolve`, resume LR warmup, boundary restoration, and fork EMA clearance exclusive to resume/fork vehicles.

### HIGH A2-H1 — enabled event continuation is not bit-faithfully resumable

The #518 landing names unpersisted `last_boundary_epoch`, pose `engaged_epoch`, BEST/deploy boundary state, and related residuals (`ead2a13760:...p0_resume_warmup_geometry_build_20260717.md:96-118`). Arm B admits detector `mode` is not persisted (`c896c99c5b:.omx/research/arm_b_forces_triggers_build_20260717.md:160-174`). Its score-affecting label-floor/NCDE series and `last_d_pose`/dynamic-weight state are process-local. A mid-event crash can therefore change ramp position, trigger time, or pose weight.

The trainer also catches optimizer-restore exceptions and continues with fresh moments on the #518 branch; silent fresh optimizer state violates deterministic resume unless a typed fork treatment explicitly authorizes it.

**Broken assumption.** Restoring weights while silently resetting controller or optimizer state is a bit-faithful resume.

**Verdict scope:** `RESUME SEMANTICS/IMPLEMENTATION`.

**Fix:** persist controller mode, full detector buffers, event series, latch/fire state, eligibility state, authoritative d_pose source/epoch, dynamic weight, engage/boundary epoch, ramp phase, optimizer kind/state, and EMA selection state. Require continuous-versus-interrupt/resume equality at every transition. Fail closed on optimizer restore unless an explicit recorded fork law applies.

### HIGH A2-H2 — constants are annotated metadata, not executable provenance

`V10_CONSTANTS` stores research measurements but no actual launch constants (`spec_v10_capstone_20260717.py:87-129`). The compiler resolves no LawRefs into argv. The β2 warm-up law’s 27 epochs uses `c=2` and 75 steps/epoch; its own memo labels `c≈2` inferred/provisional and requires the 8-vs-27 A/B (`ead2a13760:...p0_resume_warmup_geometry_build_20260717.md:51-69,108-118`). `MarginStepCap` has no measured margin-field-derived cap at its update site. EMA calibration has useful measured seed-fraction/shadow data, but the terminal law and selection still require the declared measurement.

**Broken assumption.** Attaching provenance prose to a provisional number is equivalent to deriving, resolving, and compiling the constant used by the executable.

**Verdict scope:** `VALUE-PROVENANCE/COMPILE CUSTODY`.

**Fix:** derive warm-up from compiled optimizer beta2 and actual updates per epoch; calibrate `c` by the pre-registered A/B. Derive step cap from a CFL-style bound on maximum boundary/margin displacement per update. Derive EMA decay/finisher from measured update noise, horizon, and terminal shadow-vs-live A/B. Any provisional default must compile as experiment-only, not launch authority.

### HIGH A2-H3 — build-wave branches can disappear semantically at merge

The compiler probes only five #518 symbols and ignores Arm A/B/C/G, curvelet, range projection, coder, structured defaults, and their consumers (`spec_v10_capstone_20260717.py:31-41`). C1 only has a prepared/default-off hook with render-loop consumption owed; C2’s applypass/receiver fold is owed; C3’s compiler composition is owed (`35dc3b70f3:.omx/research/v10_buildable_components_20260717.md:42-55,94-127`). A clean Git merge is not semantic composition.

**Broken assumption.** Branch existence or a conflict-free merge proves that every component is compiled, consumed, resume-safe, and measured at the intended runtime surface.

**Fix:** gate on an exact reviewed-component manifest containing commit hashes, DSL owner, parser flags, runtime consumer, resume keys, equation/LawRef, tests, and measurement receipt. A branch or class name may never satisfy a launch blocker by itself.

### HIGH A2-H4 — the compiler blocker set omits launch-defining scientific gates

At minimum add typed blocker classes for:

1. real post-gate config/argv emission;
2. frozen trainer parse and DSL provenance bijection;
3. unresolved LawRef/constants manifest;
4. receipt schema/hash/axis/verdict/content invalidity;
5. noncanonical repo/artifact root;
6. cold-start use of fork/resume-only machinery;
7. score-domain pose marginal double application;
8. incomplete event/resume state;
9. structured-seed semantic failure;
10. range projection uint8/scorer closure and real consumer;
11. content coder applypass/receiver/archive closure;
12. all required build-wave components consumed at exact reviewed SHAs;
13. interaction matrix unresolved;
14. flat-cell/`T` necessity certificate unresolved, including D24 nonlocal tail;
15. V8/DCB unique-home carrier receiver and composition receipt absent;
16. old-findings-first A/B verdicts absent;
17. #270 cold-vs-warm Muon A/B absent;
18. #497 exact matched-byte basis verdict absent;
19. c2 terminal full-facet/pose/EMA/phase-store harvest absent;
20. D18 code-k90, D21 receiver blind-fill, and D27b terminal-solve triggers unresolved;
21. governed launcher storage/RSS/resume/per-stage-checkpoint receipt absent;
22. exact archive/evaluator custody absent before any promotion claim.

**Broken assumption.** The current blocker enumeration is complete merely because every presently probed path exists.

**Fix:** replace the contextual count with typed blocker identities covering the 22 semantic classes above; each class must carry a validator, evidence schema, and explicit resolution receipt, and promotion blockers remain distinct from build/readiness blockers.

### MEDIUM A2-M1 — the cold-start order is underspecified

The pillars are enumerated, but a mathematically load-bearing order is required because solves, projections, seeds, and training change each other’s residual.

**Broken assumption.** Correct ingredients remain train-least and non-duplicative under an unspecified order of solves, projections, seeding, training, coding, and evaluation.

**Fix / derived order:** validate source/cache and typed seeds -> generate `G`/xi priors -> solve fresh head and gauge -> apply only lattice-safe projection constraints -> initialize `T`/residual carrier -> train the quotient residual -> joint pose stage under the exact score objective -> canonicalize/allocate/coder -> receiver parse-back -> exact evaluator gates. At every boundary, recompute the residual and prevent earlier components from being relearned.

---

## AXIS 3 — forces, triggers, telemetry, and event continuation

### CRITICAL A3-C1 — the live regression fingerprints cold Muon, not Force-3

Live Force-3 run: `d_seg 0.003458@725 -> 0.004294@750 (+24.2%) -> 0.004164@800`; Road `0.004972 -> 0.007484 -> 0.007025` (`/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z/run.log:397,473,668`).

Prior mod32cap control, with no Force-3 but the same cold Muon entry at 726: `0.003414@725 -> 0.004351@750 (+27.45%) -> 0.004163@800` (`/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/run.log:1109,1133,1190`). At ep800 the two runs differ by about `1e-6`.

Arm A’s ep725 readback remains real: phase vs Seg base is Euclidean cosine `-0.149`, Fisher cosine `-0.118`, relative norm `0.627/0.478` (`443aaeb41a:.omx/research/fisher_actuation_arm_a_build_20260717.md:23-54`). It proves local antagonism at one state, not that phase caused the ep725-750 regression.

**Verdict scope:** `cross-run mechanism fingerprint/advisory`, not a randomized causal A/B and not a Force-3 family verdict.

**Broken assumption:** V10 connects late phase antagonism, the regression, and sigma erosion before isolating the optimizer discontinuity (`SPEC:527-542`).

**Fix:** fire #270 from the exact ep725 bytes with Force-3 fixed: cold Muon control versus warm-start momentum + LR decay. Do not relax/remove Force-3 based on the present regression. Preserve exact checkpoints and compare full facets.

### CRITICAL A3-C2 — the “live crest” regression anchor is contradicted by later telemetry

Arm B’s memo/test froze an early six-point rise/decline and extrapolated a fireable crest (`c896c99c5b:.omx/research/arm_b_forces_triggers_build_20260717.md:27-44`). The actual live series later rebounded: median sigma approximately `0.00103@786, 0.00682@794, 0.00865@798, 0.01287@802, 0.00178@806, 0.00110@810, 0.00256@814, 0.01283@818, 0.00180@822`; gate rows remained `DEGENERATE_GUARD_TRIPPED` (`live run.log:580-581,613-614,627-628,646-647,660-661,679-680,693-694,707-708,721-722`). This is an elevated oscillatory band, not an evidenced terminal crest.

**Broken assumption.** An early truncated six-point regression is a stable live crest once the later oscillatory trajectory is observed.

**Verdict scope:** `live-crest regression anchor`, not the crest-detector family.

**Fix:** add the exact ep786-822 sequence as a must-not-fire control. Require local prominence, bounded crest locality, all-window resolution, persistence/no-rebound hysteresis, and consistency with Fisher-annulus mass plus remaining Seg/Pose debt.

### HIGH A3-H1 — beta “event coupling” retains the fixed ep1000 barrier

The implementation defines completion from numeric `anneal_epochs` and suppresses every pose engage signal before that epoch (`c896c99c5b:experiments/train_levelset_witness_realized_through_R_mlx.py:12053-12068`). Calling `beta-anneal-complete -> eligible` an event does not dissolve a constant if completion is not measured from beta dynamics. It also conflicts with the claim that a crest near 802 makes ep1000 suboptimal (`c896c99c5b:.omx/research/arm_b_forces_triggers_build_20260717.md:46-56`).

**Broken assumption.** Renaming a fixed epoch barrier as an event makes the curriculum an event-continuation law.

**Verdict scope:** `EVENT-LAW FORMULATION`.

**Fix:** define beta/tau completion from derivative/residual and stability, not epoch. Treat conditioning signal time and continuation eligibility as separate observables. A/B whether the conditioning event should retime/terminate continuation or merely bank a pending pose engage.

### HIGH A3-H2 — the completeness table is a disposition list, not a composition certificate

V10’s table says `PRESENT/BUILD/EXCLUDED` (`SPEC:293-309,452-458`) but omits scale, metric, trigger, actuator, runtime composition, resume status, authority axis, antagonists, and A/B evidence. It therefore counts default-off/unmeasured/probe-only/unbuilt surfaces as if they closed the system.

Known open cells include: lambda-critical trigger, late phase-weight relaxation, #425 joint receiver/adoption verdict, Fisher surfaces 4-6, sigma_ccprime choice, event-fallback efficacy, Lane skip efficacy, structured `T`, content coder receiver, and multi-carrier interaction.

**Broken assumption.** `PRESENT`, `BUILD`, or `EXCLUDED` alone certifies force completeness without runtime consumption, resume custody, measurement, and interaction disposition.

**Fix:** use the completeness matrix below. A force is `COMPLETE` only if derived, built, compiled, consumed, resume-certified, measured on its declared axis, interaction-audited, and either adopted or explicitly excluded.

### HIGH A3-H3 — sigma_ccprime is demanded but its active law is unresolved

The live config uses scalar `length-weight=0.001` (`LIVE-LAUNCH:55`); V10 says per-class-pair anisotropic tension should replace/demote it (`SPEC:452-458`). Competing canonical derivations yield different Road-Lane values, so “compose #382” is not a settled constant.

**Broken assumption.** The existence of an anisotropic-tension primitive settles its physical normalization, absolute scale, and adoption over the live scalar control.

**Verdict scope:** anisotropic-tension **preset/normalization**, not the family.

**Fix:** run a three-arm realized-through-R A/B: scalar control, Young/Herring-derived matrix, fragility-derived matrix. Preserve the scalar as absolute scale until normalization is measured. Include Lane erosion, other-class leakage, Pose, and rate.

### HIGH A3-H4 — subpixel supervision does not close the event-birth gap

The event-fallback audit measured `26.3%` candidate straddle sites without T1 supervision; Arm B correctly built a distinct fallback force and labels its delta unmeasured (`c896c99c5b:.omx/research/arm_b_forces_triggers_build_20260717.md:11-25`). Any manifest statement that existing subpixel supervision already closes that gap is semantically false: subpixel weights existing active straddles; it does not create supervision at births/pair-0 fallback.

**Broken assumption.** Transport supervision over existing straddles supplies the same support as a birth/fallback force at previously uncovered events.

**Fix:** keep separate telemetry and completeness rows for `transport_straddle_supervision` and `birth_event_fallback`. Measure the fallback A/B before adoption.

### MEDIUM A3-M1 — dual-metric force evidence is too sparse for a schedule law

The ep701 global cosine is Euclidean and very small-sample; ep725 is one n96 reconstructed-orientation checkpoint. Both metrics and relative norm are required, but a single sign/magnitude at one state cannot derive a late-phase trigger.

**Broken assumption.** One or two sparse force-angle snapshots determine a stable continuation or relaxation law across the optimizer transition and terminal basin.

**Fix:** harvest the same exact force map at pre-entry, engagement, pre-Muon, post-Muon, re-descent, and terminal checkpoints; report Euclidean cosine, selected Fisher/Bregman reading, relative norm, per-class terms, and orientation reconstruction sensitivity. Only then derive relax/hold events.

### MEDIUM A3-M2 — dynamic pose control needs source and staleness custody

Most live post-start verdict rows are pose-blind; a dynamic weight or trigger must state whether it used live or EMA d_pose, the measurement epoch, age, selection eligibility, and clamp. Arm B already acknowledges source ambiguity (`ARM-B:168-170`).

**Broken assumption.** Any recently available `d_pose` value is interchangeable authority for a score-affecting dynamic controller.

**Fix:** pin the authoritative source and persist it. Emit `(source, checkpoint_sha, measured_epoch, consumed_epoch, age, d_pose, effective_weight, clamp)` on every change.

### LOW A3-L1 — preserve the good live force provenance

The Arm B force memo records the measured/derived rationale for the class set, band, target, weights, and default-off status (`c896c99c5b:.omx/research/arm_b_forces_triggers_build_20260717.md:11-25,72-111`); Arm A preserves both Euclidean and Fisher readings (`443aaeb41a:.omx/research/fisher_actuation_arm_a_build_20260717.md:23-54`).

**Broken assumption.** Correct provenance and default-off causal controls can be discarded merely because a neighboring trigger interpretation failed.

**Fix:** preserve classes `0,1,2`, band `2.0`, `gt_advected`, `pa_flipmass`, 2x2 chroma Pose safety, explicit term telemetry, structured `fire=unavailable`, both metric readings, and duty-to-measure posture while repairing the named trigger/controller defects.

---

## Adversarial attack on this review's own conclusions

1. **Could the flat-palette result merely indict a bad palette, not prove `T` is necessary?** Yes; it does not prove every class/cell needs a counted texture payload, and a better deterministic metamer generator may synthesize `T` for nearly zero seed bytes. That is why A1-C1 rejects the **generic certificate**, not generator-first realization. The decisive test is per-class/per-cell receiver-realized removal or replacement of `T`, not the historical palette number alone.
2. **Could the range projector still be valuable despite the uint8 falsifier?** Yes. The fp64 theorem is correct, and a renderer trained to stay in gamut—or a bounded lattice preimage solve—may recover most of the capacity benefit. The one-frame/channel probe is a decisive counterexample to universal exactness of the current consumer, not an n600 efficacy verdict or family negative.
3. **Could Arm B's pose law have intended a raw-MSE objective?** The standalone derivation is correct for multiplying raw `d_pose`; the defect arises only in the actual V10/live conjunction with `--score-domain-loss`. The compiler conflict and composed-gradient test are sufficient; the equation need not be deleted.
4. **Could phase still contribute to the regression despite the Muon fingerprint?** Yes. The prior run is not randomized, and the states/other details are not perfectly paired. The near-identical excursion is strong de-attribution evidence, not proof of zero phase effect. The recommended #270 A/B holds Force-3 fixed precisely to decide this.
5. **Is it unfair to call the compiler fake-passable when it deliberately refuses after gates clear?** The final refusal is honest and should remain until a real fold exists. The finding is that `report.clear` and the SPEC's “13 blockers” are not semantic readiness, and the advertised gate 5 cannot succeed. A blocker-report skeleton is useful; it must simply not be treated as the compiler or launch certificate.
6. **Could binary Fisher be the right approximation in a top-two annulus?** Yes, away from multiway junctions. The correction is to label and measure approximation error, use full K=5 trace where logits are available, and preserve the binary prior as an A/B arm.
7. **Could the cold-start vehicle legitimately reuse `Fork*` code paths after a synthetic seed load?** Only if the semantic distinction is explicit and the path no longer requires/pretends a resumed optimizer/trajectory. Reusing implementation is fine; calling resume-only behavior a birth law without fresh-state invariants is not.

These counterarguments narrow, but do not erase, the critical blockers.

---

## V10 system-of-equations completeness matrix

Legend: `CLOSED` = derived+built+compiled+consumed+resume-certified+measured/adopted or excluded; `PARTIAL` = some real surface exists; `MISSING` = demanded term/receipt not represented; `OPEN` = decision correctly deferred.

| force / component | demanded by measured dynamics or exact scorer math | present in SPEC | built / consumed / measured | status and true-final action |
|---|---|---|---|---|
| Base Seg decision loss | exact Seg argmax objective needs a differentiable descent surrogate | yes | live `tau_softplus`; exact score only at verdict | `PARTIAL`: keep, but derive continuation/tau events and judge only through exact argmax |
| Texture/metamer residual `T` | flat palette `0.0416` vs textured `0.0048`; nonlocal sign-alternating VJP | **no explicit first-class row** | hidden in trunk | **MISSING/CRITICAL**: explicit unique-home `T`, quotient residual, remove-T ablations |
| Cell generators `G` | partition geometry/rate decomposition | yes | modules/designs; no receiver-closed composition | `PARTIAL`: exact archive/receiver and interaction custody |
| xi / temporal transport | temporal geometry and phase/pair consistency | yes | live/advisory; #425 STORE codec built+measured | `PARTIAL`: visibility-generator rate cure and joint through-R receiver/adoption A/B open; no generic persistence equivalence inferred |
| Event-birth fallback | 26.3% candidate straddles uncovered | yes in §13 | built default-off; delta unmeasured | `PARTIAL`: highest force A/B after old-findings-first ordering |
| Phase advection | measured subpixel temporal boundary drift | yes | live, binds; ep725 mildly antagonistic | `PARTIAL`: keep while isolating Muon; derive trajectory event before relaxation |
| Margin satisficing | prevent wasted force beyond safe margin | yes | live/binds; isolated efficacy incomplete | `PARTIAL`: dual-metric/term interaction A/B |
| Subpixel boundary / pa_flipmass | uint8/R realization and flip mass | yes | live/binds | `PARTIAL`: preserve; do not claim event-birth coverage |
| Tie-locus / saddle precision | 29.2% sampled saddles sub-LSB; junction fragility | yes | live Force-3 family; causal attribution confounded | `PARTIAL`: isolate Muon, exact terminal receipt |
| Per-pair anisotropic sigma_ccprime | Lane erasure and Gamma-limit pair tensions | yes §13 | primitive built, preset not settled/compiled | `OPEN`: scalar vs Herring vs fragility A/B |
| Scalar length tension | live MCF regularization | implicitly demoted | live `0.001` | `OPEN`: retain as scale/control until sigma matrix wins; no silent removal |
| Lane skip-band force | witness has about 10x GT deficit in measured band | yes §13 | built/binds n24; delta unmeasured | `PARTIAL`: controlled A/B, feature-space upgrade only if needed |
| Illumination-cone gate | night/wet cone modulation | yes | seed/probe only | `OPEN`: semantic seed plus exact receiver probe |
| Mirror transport | hood/wet specular anchored to light, not surface xi | yes probe | not adopted | `OPEN`: P-2 rate-vs-residual and Pose check |
| Joint pose descent | post-hoc pose formulations hit photometric wall | yes | live gate, pose phase not yet terminal | `PARTIAL`: exact score-domain objective; terminal result decides form |
| Pose marginal coefficient | exact score is `sqrt(10*d_pose)` | yes §13 | Arm B built incorrectly composed | **FAIL**: `w_pose=1` under score-domain loss; raw-MSE law only |
| Pose conditioning trigger | sigma geometry should gate joint descent | yes | plateau live; crest arm invalid live anchor | `PARTIAL`: oscillation-safe event, resume state, terminal A/B |
| Muon transition law | optimizer discontinuity matches observed regression fingerprint | yes pending #270 | live cold entry; warm levers built/default-off | `OPEN/HIGH`: old-findings-first #270 A/B from exact ep725 |
| Fisher-density force | decisions concentrate near simplex uncertainty | yes §13 | binary surrogate built; efficacy unmeasured | `PARTIAL`: full K=5 trace vs binary prior A/B |
| Fisher/pullback optimizer | decision geometry may precondition descent | yes §13 | logit cotangent preconditioner built | `PARTIAL`: rename or implement pullback solve; Muon interaction A/B |
| Fisher trigger observable | raw d_seg slope is not decision geometry | yes §13 | sensor exists; controller incomplete | `PARTIAL`: full-K/selected-metric stream + resume custody |
| Fisher bit allocation | measured per-tensor sensitivity should price bytes | yes | #336 exists; coder explicitly skips | **MISSING IN P4**: apply with real RD rows and receiver bytes |
| EMA calibration/selection | full-batch update geometry makes 0.997 suspect; EMA beat live 18.6% at one checkpoint | yes §13 | law/comparator built; terminal open | `PARTIAL`: terminal per-stage shadow/live byte-close, ship winner |
| Head solve and gauge | exact rank-4 head/gauge precision | yes | fork-only solve; gauge measured precision | `PARTIAL`: fresh-init head solve; gauge exact receiver confirmation |
| Range(A) restriction | about half render-difference energy measured in float null space | yes | float primitive; no lattice/consumer closure | **FAIL AS EXACT CONSUMER**: bounded integer preimage and scorer receipt |
| Content-priced coder | shape pricing wastes content structure | yes | checkpoint codec only | `PARTIAL`: full archive receiver, config bytes, Fisher allocation, exact score/bytes |
| Per-class carrier composition | class/edge/saddle residuals have different optimal homes | yes | modules; no unique-home receiver A/B | `PARTIAL`: direct-sum residual + exact composition matrix |
| Persistence-hold force | would fight genuine deaths | explicitly forbidden | excluded | `CLOSED, scoped`: preserve exclusion |
| Symmetric/deep-side pushes | measured `+73%/+98%` harm | forbidden | excluded | `CLOSED, instance/formulation scope` |
| l7/smooth/fixed-beta stages | measured defects/divergence | forbidden | excluded in intended V10 | `CLOSED if compiler proves absence` |
| Blurry/global texture paste | measured harmful formulations | forbidden | excluded | `CLOSED only for tested forms`; do not generalize to `T` family |

## True-final-form delta list

Each delta is the exact SPEC/config/compiler change recommended.

1. **[MEASURED] Restore `W=(G,xi,T)` in the V10 identity.** Replace generic `GEOM-ONLY` rows with `PROBE` until real remove-`T` receiver ablations pass. Add D24 nonlocal-tail closure as a gate.
2. **[DERIVED] Define unique-home quotient custody.** The trunk trains only `r_irreducible` after deterministic generators, seeds, solves, and carriers. Add a no-double-payment receipt and pairwise composition matrix.
3. **[DERIVED + MEASURED falsifier] Separate `P_A`, `P_cell`, `Q_uint8`, `N_seg`, and `N_pose`.** Delete the claim that the first two are the same projection. Gate any consumer on a bounded lattice/preimage and exact decoded scorer receipt.
4. **[DERIVED] Downgrade global Dykstra to local sequential convexification/trust-region unless feasibility/contraction is proved.** Require monotone realized acceptance, cycle detection, and empty-intersection/failure certificates.
5. **[DERIVED] Split cold-start laws from fork laws.** Introduce fresh `InitHeadSolve`/fresh EMA initialization; forbid `Fork*`/resume-only levers in `v10c0`.
6. **[MEASURED] Replace structured-init fake-green semantics.** Distinct hood/sky detectors, data-derived lane class, nonperiodic dash anchors, real-cache/render tests, corrected diagnostic scaling.
7. **[DERIVED] Use `w_pose=1` with score-domain loss.** Add a compiler conflict for the marginal law and a composed-gradient test. Allow adaptive marginal only over raw MSE/calibrated raw proxy.
8. **[DERIVED] Use full K=5 Fisher trace `1-sum p^2`.** Keep `0.5 sech^2(m/2)` as a labeled binary annulus approximation. Report both at multiway junctions.
9. **[DERIVED] Rename Arm A’s transform `LogitFisherCotangentPreconditioner`, or implement `(J^T F J)^+` with JVP/VJP solve.** Do not call the current transform parameter natural gradient.
10. **[MEASURED/OPEN] Isolate #270 cold Muon before changing Force-3.** Exact ep725 checkpoint, Force-3 held fixed, cold vs warm momentum+LR-decay A/B; full facets and exact bytes preserved.
11. **[MEASURED] Replace the clean-crest regression anchor with the actual oscillatory ep786-822 sequence.** It must not fire. Add prominence/locality/persistence/rebound guards and Fisher/debt coupling.
12. **[DERIVED + OPEN] Replace epoch-1000 beta eligibility with measured continuation completion.** Beta/tau derivative/residual and stability are the event; conditioning signal time is a separate event. A/B their coupling.
13. **[MEASURED/OPEN] Keep event-fallback distinct from subpixel transport and measure it.** Do not claim the 26.3% birth gap closed before the A/B.
14. **[DERIVED/OPEN] Treat sigma_ccprime as a three-arm law decision.** Scalar control vs Herring/Young vs fragility; preserve absolute scale and measure Lane/other-class/Pose effects.
15. **[MEASURED] Enforce old-findings-first at the boundary.** Before lower-ranked novelty, adjudicate DsegAwareTaper `78.9%`, HorizonWeightedMargin `47.3%`, StepNativeActivation `34.2%` (`MAIN-P0:172`), then #270, event fallback, curvelet/basis, and the newer Fisher/skip arms, with interaction isolation.
16. **[MEASURED/OPEN] Run the exact matched-byte basis set with receiver/AA held fixed.** Fourier, genuine curvelet, compact shearlet, step-native; exact archive bytes and per-stratum Seg/Pose.
17. **[DERIVED] Make event/controller state complete and crash-replay equal.** Persist every detector buffer/mode/latch, source epoch, dynamic coefficient, boundary/ramp state, optimizer state, and EMA state; fail closed on restore drift.
18. **[DERIVED] Replace the compiler skeleton with a real typed compiler.** Emit parser-verified argv, resolved LawRefs, constants manifest, config hash, exact reviewed component manifest, and semantic receipt validation.
19. **[MEASURED/OPEN] Close P4/P7 receiver custody.** Content coder and DCB/carriers must build a real archive, parse back, reproduce pixels/declared distortion, and report full score units per byte. Apply #336 only from measured RD rows.
20. **[MEASURED/OPEN] Terminal gate remains binding.** Harvest c2 full facets, pose outcome, per-stage/terminal EMA-vs-live, phase-store seed economics, and governed stop before choosing V10 defaults.
21. **[DERIVED] Replace “Kolmogorov-optimal” measurement language with exact program length `L_G` until an archive proves it.** Preserve the shortest-program heuristic, but do not treat a codec-pair ratio as Kolmogorov complexity.
22. **[DERIVED] Rebuild the completeness table as a certificate matrix.** Include scale, metric, trigger, actuator, build/compile/consume/measure state, resume proof, axis, antagonists, and A/B receipt.

## P0 and deferral alignment

| authority row | V10 status after review | required disposition |
|---|---|---|
| `p0_SUPREME_duty_queue_old_findings_first` | §13.5 names the top three but later new arms can still outrun them | make ordering a compiler/merge gate, not prose |
| `p0_lane_three_cruxes` | skip lever measured/built; sub-LSB and dash phase remain multiplicative, not receiver-composed | one joint Lane stack with isolated and composed receipts |
| `p0_fisher_full_leverage` | measurement + partial force/preconditioner built; full-K, pullback optimizer, trigger, rate, EMA selection incomplete | retain P0; correct math labels and close each consumer |
| `p0_ema_calibration` | useful law/18.6% advisory anchor; terminal winner unknown | terminal/per-stage exact shadow-vs-live gate |
| `p0_triggers_forces_review_all_findings` | many surfaces built default-off; crest and pose law contain critical errors | repair, resume-certify, then staggered A/B |
| `p0_realization_limited_not_gradient`, `p0_range_A`, `p0_null_subspace_gauge_kerA` | float geometry is real; uint8 consumer proof absent | lattice-aware preimage and exact scorer/receiver custody |
| `p0_UNIFICATION_projection_preimage_SUPREME` | useful search frame; global convex/fixed-point claim overstates local evidence | local trust-region with feasibility/failure certificates |
| `p0_497_basis_cure` / curvelet takeover | fireable but exact decisive receiver verdict open | controlled matched-byte factorial after old findings |
| `p0_boundary_merge_queue_post_v9c2` | branch inventory not compiler-bound; semantic omission risk | reviewed exact-SHA component manifest and consumer coverage |
| `p0_v10_capstone` | advances architecture, but contradicts measured `T` and is not compilable | adopt the delta list before naming true-final |
| D18 | final k90/code receiver selection not closed | fire only at terminal artifact; coder must consume the result |
| D21a | blind-fill proof exists but receiver does not consume it | keep blocker until n600 bit-identity receiver receipt |
| D24a | nonlocal Jacobian tail receipt absent | mandatory before flat/local factorization certificates |
| D27b | terminal solve readiness receipt absent | keep exact trigger; do not infer terminal readiness |

## What is decidable now vs what remains owed

### Decidable now

- **DERIVED/CONFIRMED:** score-domain pose marginal composition is wrong.
- **MEASURED/CONFIRMED:** unconstrained float range projection does not preserve the resize plane after valid uint8 clamp/round on the tested source frame.
- **CONFIRMED from source:** compiler gates validate existence/symbols only and the compiler has no success path.
- **MEASURED/CONFIRMED:** the sky default invokes a bottom-hood detector; the diagnostic fraction is mis-scaled.
- **MEASURED corpus contradiction:** generic flat-cell texture exhaustion is not established and conflicts with flat-palette/nonlocal-VJP evidence.
- **DERIVED:** rank-4 final head does not make nonlinear scorer preimages convex; local pseudoinverses do not prove global Dykstra convergence.
- **DERIVED:** Arm A density is exact only for the binary reduction; current backward transform is not generally parameter natural gradient.
- **CONFIRMED from source/memos:** cold-start V10 cannot use resume-only `Fork*` birth semantics.
- **CONFIRMED from state wiring:** enabled Arm-B/#518 event paths are not bit-faithfully resumable.
- **MEASURED trajectory:** the claimed clean crest is invalidated by later rebounds.
- **MEASURED cross-run fingerprint:** cold Muon is a stronger explanation for the ep725-800 excursion than Force-3, though not a randomized causal proof.

### Requires c2 terminal or a named measurement

- c2 terminal full-facet result, pose engage/outcome, terminal archive, phase-store harvest, and terminal EMA-vs-live winner;
- exact #270 cold-vs-warm Muon A/B;
- event-fallback delta and late phase hold/relax trajectory A/B;
- old-findings-first isolated A/Bs;
- scalar vs Herring vs fragility sigma_ccprime;
- full-K Fisher density and pullback-preconditioner A/B, including Muon interaction;
- matched-byte Fourier/curvelet/shearlet/step-native receiver A/B;
- per-class `T` necessity/removal and quotient-trunk training experiment;
- bounded lattice range-preimage full-n600 scorer/receiver test;
- 8-vs-27 beta2 warm-up A/B;
- content coder and DCB/per-class carrier exact archive/receiver composition;
- phase-carrier score-unit-per-byte adoption;
- D18/D21a/D24a/D27b trigger-specific receipts;
- exact contest-CPU and contest-CUDA evaluation on the final archive bytes before promotion.

## Triality and landing boundary

- **DAG:** V10’s ready-to-paste FEED is not enough to close these findings. MAIN should append one reviewed FEED only after resolving the branch composition and should include the negative scopes above.
- **DSL:** the current file is a blocker-report skeleton. A real typed, parser-verified, LawRef-resolved compiler is owed.
- **Equations:** correct/qualify the flat-Fisher implication, binary-vs-K-class Fisher law, pose-weight composition, local-vs-global projection statement, and fresh-init/resume law split before registry promotion.
- **MAIN review required:** this memo is an isolated-worktree review artifact. MAIN must review the exact diff and independently decide whether/how to land it. No branch component should be promoted merely because this memo cites it.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; craft handoff and triality manuals; V7.5 §8 and V8 SPEC; V10 SPEC/compiler/naming; all named V10/build-wave branches; upstream `modules.py`, `evaluate.py`, `frame_utils.py`; real `gt_n600.npz`; projection/factorization/necessity/null-space/texture/carrier/recursive-fractal/temporal/night-wet/degraded-lane/dpose corpus; canonical equation list/registry; recent DAG FEED tail and graph recall; live c2 and prior mod32cap run logs/configs; operator P0 digest; deferral ledger; latest prior Codex findings/session summary, council/design memos; live delegation inbox through `2026-07-17T20:49:26Z`.

**Final scoped verdict:** V10 is a high-value **candidate capstone research program** with real primitives and honest default-off work, but it is neither a proved Kolmogorov optimum, a complete system of equations, a semantically compiled launch config, nor a promotion-grade vehicle today. Adopt the critical representation/objective/compiler corrections first; then let the named exact measurements, not the current prose, decide the surviving family choices.
