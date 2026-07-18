# Inverse-solve completeness matrix — term per frozen-forward factor

Date: 2026-07-18
Lane: `completeness_coupling_20260718`
Authority: Task #538, advisory build + `$0` local measurement, no launch
Verdict scope: `V10 INVERSE-SOLVE COMPOSITION / CURRENT-BRANCH CUSTODY`
Pointer: `0.19108` **UNMOVED**
Sacred c2: **READ-ONLY; NOT MUTATED**

## Verdict

**DERIVED — `NOT_COMPLETE_BY_CONSTRUCTION`.** Section 14.11's “ten-factor”
requirement expands to eleven auditable leaves because factor 3 has two distinct
uses of the same resize: camera-resolution preimage control (3a) and the joint
Seg/Pose coupling through shared `A` (3b). The current tree has useful primitives
for most factors, but no row satisfies the strict completion certificate below.
In particular, uint8 feasibility and the joint three-axis waterfill are missing
as consumed V10 terms, and this lane's coupling policy is deliberately advisory
and argv-inert.

`HAVE`, `MISSING`, and `FOLDED` below describe **term disposition**, not launch
readiness:

- `HAVE`: a typed or measured factor-specific surface exists.
- `FOLDED`: the factor is already represented inside a broader frozen map or
  existing term and must not be re-derived, but its V10 composition receipt can
  still be absent.
- `MISSING`: the demanded objective/actuator term does not yet exist as a real
  consumed surface.

A leaf is `COMPLETE` only when all of the following are evidenced together:
derived, built, compiled, consumed, resume-certified, measured on its declared
axis, interaction-audited, and adopted or explicitly excluded. Presence alone
does not pass.

## The sealed 10-factor / 11-leaf matrix

| factor | exact forward/score role | term that must exist | owner / settled authority | disposition now | strict certificate | literal remaining blocker |
|---|---|---|---|---|---|---|
| 1 | payload `decode -> (frame_0, frame_1)` | deterministic generator plus counted-seed custody and receiver parse-back | V10 generator/receiver; `.omx/research/frozen_scorer_exact_factorization_20260715.md` §0 | `FOLDED` | `PARTIAL` | No V10 payload compiler emits and parses back one receiver-closed, resume-certified program with every video-derived seed byte counted. |
| 2 | camera RGB must lie on the finite uint8 lattice before either scorer | bounded lattice/preimage feasibility inside the solve, not real-valued solve followed by rounding | Task #532; SOL true-final review A1-C2 | `MISSING` | `FAIL-AS-EXACT-CONSUMER` | The current real projector preserves `A` in fp64, while valid `clip(round(.))` produced a measured max scorer-plane discrepancy of `62.74`; a lattice-feasible realization plus exact scorer receipt is owed. |
| 3a | camera `(874,1164)` to scorer `(384,512)` through bilinear `A` | camera-resolution subpixel placement and antialias-coverage solve | Task #149; frozen factorization §6.1 | `HAVE` | `PARTIAL` | The camera-resolution actuator is not compiled/consumed in V10 with a uint8-realized n600 interaction receipt against both scorers. |
| 3b | **the same `A` feeds both frozen scorers** | joint Seg/Pose pullback term and score-derived coupling price | Task #538 (this lane); framework-normalized equation candidate `shared_resize_joint_coupling_through_a_v1` | `HAVE` (advisory, argv-inert) | `PARTIAL` | Honest subset coupling is measured locally, but no live trainer consumer, resume state, full-n600 receipt, byte-close archive, or contest-axis adoption exists. |
| 4 | Seg reads RGB; Pose reads BT.601 YUV6 with full luma phases and box-averaged chroma | luma/chroma-coordinate term with joint Seg/Pose guard | Tasks #276/#535; `ChromaBoundaryMatch`; `rgb_at_boundaries_20260715`; `posenet_luma_chroma_asymmetry_20260710` | `HAVE` | `PARTIAL` | Existing channel primitives are not jointly compiled/consumed with factor 3b and lack a receiver-closed n600 interaction/adoption receipt. |
| 5 | fixed deep convolutional maps `N_seg` and `N_pose` | scorer-through-payload residual/TTO; train only the residual not solved by other factors | Task #350 and Task #531 quotient residual `T`; frozen factorization §§2–3 | `FOLDED` | `PARTIAL` | The frozen maps are exact, but V10 has no unique-custody `T` consumer proving earlier solved terms are neither relearned nor double-paid. |
| 6 | SegNet's five logits and rank-4 centered final head end in argmax cells | winner-cell/power-diagram feasibility with satisficing margin | rank-4 equation `segnet_head_rank4_linear_flipdist_v1`; power/Laguerre work; Task #531 for residual custody | `HAVE` | `PARTIAL` | Exact head geometry is not yet a video-fed, byte-closed generator/receiver whose realized uint8 output satisfies the frozen nonlinear preimage and interaction gates. |
| 7 | PoseNet emits 12 values but score reads MSE on the first six | six-scalar pose target/sidecar or joint descent, priced in the correct score domain | pose output-space inverse/sidecar lineage, including Task #366; Task #528 composition correction | `HAVE` | `PARTIAL` | No V10 receiver-closed sidecar/joint consumer proves the six values at acceptable bytes; under score-domain loss the only exact outer coefficient is `1`, not a second pose marginal. |
| 8 | frame 0 has zero Seg obligation but participates in Pose | frame-0 allocation term that spends its freedom on Pose/rate without harming the pair | Task #394; `.omx/research/frame0_chromahf_dofs_20260710.md` | `HAVE` | `PARTIAL` | Structural freedom and advisory measurements exist, but the carrier is not jointly consumed with factors 3b/4/7 in a receiver-closed V10 A/B. |
| 9 | archive bytes price the sufficient statistic; blind complements must cost zero | description-length term plus generic fill for `ker(A)`/blind coordinates and measured content coder | Tasks #401/#519/#520; `.omx/research/null_subspace_rate_measure_20260717.md` | `HAVE` | `PARTIAL` | Float null-space geometry does not establish a legal uint8 receiver or rate win; exact counted bytes, lattice closure, and unique producer/consumer custody remain owed. |
| 10 | `100*d_seg + sqrt(10*d_pose) + 25*bytes/U` chooses the operating point | joint Seg/Pose/rate KKT waterfill over measured residual R-D curves | Task #536 | `MISSING` | `MISSING` | There are no byte-closed local marginal curves for all three axes on one custodied artifact, so the KKT allocation and adopted operating point cannot be solved. |

Count check: ten numbered factors, eleven leaf rows, exactly one row for each
factor except factor 3, which has `3a` and `3b`. Any compiler manifest with a
different key set must refuse.

Task #538 binds factor 3b to the path/hash-verified n8/600 advisory receipt
`05cf34068053a4e2f744dfb35cde729579353686298da3ee1ceaf925f5a71f5f`.
That receipt measures instance-level interference and closes neither factor
3b's live-consumer certificate nor any other leaf. Its B1 input-gradient and
B32 finite-lattice diagnostics are explicitly noncommensurate, so no invented
cross-surface calibration residual is admitted.

## Complete-by-construction compiler contract

**DERIVED.** A future live V10 compiler should accept a sealed manifest keyed by
the eleven leaf IDs above and refuse unless every entry includes:

```text
factor_id, term_id, owner_task, disposition,
derivation_ref, build_sha, compiled_config_hash, consumer_id,
resume_schema_and_replay_ref, measurement_receipt_sha256, authority_axis,
interaction_receipts, adoption_or_scoped_exclusion
```

The gate must reopen referenced receipts and verify bytes/schema/hashes rather
than accepting `Path.exists` or symbol presence. `FOLDED` is admissible only
when `consumer_id` identifies the broader term and the factor-specific
interaction receipt proves the fold. `MISSING` is always refusing. The typed
policy added by Task #538 seals the IDs for future use but intentionally emits
no trainer argv and cannot grant launch or promotion authority.

## Composition laws that prevent double payment

**DERIVED.** These are invariants of the matrix, not optional optimizer choices.

1. `uint8` precedes the shared resize and frozen scorers in every verdict path:
   `payload -> decode -> Q_uint8 -> A -> {N_seg,N_pose}`.
2. `A_seg == A_pose == A`; factor 3a chooses within its preimage while factor 3b
   prices the two scorer responses on the shared visible image. They cannot be
   represented as two independent resizes.
3. Frame 0 is Seg-free, not Pose-free. Fine chroma can be Pose-insensitive only
   at the exact post-`A` scale and away from YUV clamp boundaries.
4. Deterministic generators/solves own their residual first. `T` trains only the
   quotient residual and may not relearn bytes already owned by factors 1, 3,
   6, 7, 8, or 9.
5. Rate allocation uses measured **residual** score-unit-per-byte curves. Pool
   ceilings and multiple levers over one pool are non-additive.
6. Under a score-domain loss already equal to `sqrt(10*d_pose)`, the inner pose
   coefficient is exactly `1`. The local raw-`d_pose` marginal
   `5/sqrt(10*d_pose)` is used only to price a raw pose gradient alongside raw
   Seg distortion, as in the Task #538 coupling equation.

## Triality and authority disposition

- DSL: `shared_resize_joint_coupling_measurement.v2` and
  `shared_resize_joint_coupling_policy.v2` are argv-inert advisory surfaces;
  the distinct future completeness manifest remains
  `inverse_solve_completeness_manifest.v2` and fail-closed.
- DAG: `.omx/research/completeness_coupling_joint_solve_DAG_FEED_20260718.md`.
- Equation: candidate `shared_resize_joint_coupling_through_a_v1`; it does not
  claim a live registry/adoption state. The capital-`A` spelling from the task
  brief is retained only as `NON_RESOLVING_DISPLAY_ALIAS`, because canonical
  equation IDs are lowercase and no alias resolver exists.
- Fourier: **CARGO-CULTED** for this inverse-solve factorization. It is neither
  one of the ten factors nor an admissible missing-term substitute. Natural
  forms remain factor-specific: power diagram/head, channel/chroma,
  AA/preimage, lattice projection, null space, and KKT dual/waterfill.

## Round-1 self-review

1. **Attack:** “Existing primitive” was accidentally equated with COMPLETE.
   **Resolution:** disposition and strict certificate are separate columns; no
   leaf is labeled COMPLETE.
2. **Attack:** The seven ranked missing terms in §14.11 were mistaken for the
   entire factor inventory. **Resolution:** they are the missing subset; the
   matrix also includes decode, frozen conv, rank-4 Seg readout, and Pose-six
   readout from the exact factorization.
3. **Attack:** Shared resize was counted twice without justification.
   **Resolution:** 3a is a preimage/realization DOF; 3b is cross-objective
   coupling on the same output. Their receipts and blockers differ.
4. **Attack:** Advisory local measurements could authorize V10 launch.
   **Resolution:** the policy is argv-inert, the lane remains research-only,
   pointer `0.19108` stays unmoved, and MAIN must review before any landing.
5. **Attack:** A small shared-frame Gram cosine could be promoted into a global
   separability claim. **Resolution:** the structural constrained problem stays
   joint; the measured magnitude/help-harm verdict is scoped only to one
   base-INR checkpoint and deterministic n8/600 support family.
6. **Attack:** A zero equation residual could be misread as zero coupling.
   **Resolution:** the zero binds only exact shared-forward/YUV6 parity. Smooth
   coupling and finite response remain separate diagnostics without a
   cross-surface residual.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and the craft handoff manual.
- V10 SPEC §14.7–§14.11 from `claude/p0_521_spec_v10_capstone_20260717`.
- `.omx/research/frozen_scorer_exact_factorization_20260715.md`.
- `.omx/research/sol_ultra_v10_true_final_form_review_20260717.md`.
- `.omx/research/rgb_at_boundaries_derivation_20260715.md`.
- `.omx/research/frame0_chromahf_dofs_20260710.md`.
- `.omx/research/null_subspace_rate_measure_20260717.md`.
- Relevant canonical-equation and typed-DSL source surfaces named in the table.

## 2026-07-18 V10 compiler / receiver v2 local structural delta

Verdict scope: `V10 COMPILER / RECEIVER / BYTE-CUSTODY STRUCTURAL CERTIFICATE`.
Authority axis: `[local-CPU structural/non-score]`. Pointer delta: `0`.

This dated append supersedes only the local compiler/receiver blocker wording
above; it does not alter the sealed forward-factor semantics. It changes no
row to `COMPLETE` and grants no launch, score, adoption, or promotion authority.
The local compiler's `launch_ready` is exactly `false`.

### Exact paid-route correction

The v2 compiler has seven video-derived paid sections, all counted and
semantically consumed once:

| instruction | exact factor custody | dated structural effect |
|---|---|---|
| `CountedGenerator` | `1` | generator and counted seed are one atomic range |
| `Frame0PoseSixCarrier` | `7,8` | named frame-0/Pose-six consumer exists |
| `InitHeadSolve` | `6` | named cold-only head consumer exists |
| `SharedResizePreimage` | `3a,3b` | both leaves share one exact byte range and one consumer |
| `RgbYuv6Projection` | `4` | named RGB/BT.601/YUV6 consumer exists |
| `BlindFillRateGrammar` | `9` | named blind-fill/rate consumer exists |
| `QuotientResidualT` | `5` only | terminal quotient consumer exists |

Factors `2` and `10` own no section, no byte range, and no receipt. Their rows
remain `MISSING`, their `consumer_id` remains `BLOCKED`, and their measurement
and interaction receipts remain absent.

### Factor 1

Disposition remains `FOLDED`; strict certificate remains `PARTIAL`.
`src/tac/witness_dsl/v10_compiler_receiver.py` emits one canonical counted v2
program. Factor 1 atomically includes generator inputs and seed bytes in one
reopened range; no separate free-seed section exists. The parser binds the
range/hash to the frozen route, the semantic receiver consumes it once, and a
canonical prefix checkpoint is authenticated by deterministic prefix replay.
Interrupted/resumed local output is byte-identical to uninterrupted output.

The earlier literal compiler/parse-back blocker is closed only at this local
structural-reference level. Production renderer parity, archive custody,
uint8-realized scorer interaction, exact contest-CPU/CUDA replay, and adoption
remain owed.

### Factor 5

Disposition remains `FOLDED`; strict certificate remains `PARTIAL`.
`QuotientResidualT` owns factor `5` only. It is terminal, declares exact
class/cell routes, depends on all six predecessor sections, owns one disjoint
parameter group, and freezes all six predecessor parameter groups. Its exact
quotient base is `{1,3a,3b,4,6,7,8,9}`. The factor receipt binds the reopened
factor-5 section SHA and the frozen receiver consumption-receipt SHA.

This is unique structural custody, not evidence that `T` identifies the true
production residual, improves either frozen scorer, survives uint8/resize, or
earns bytes. Those numerical and evaluator-facing claims remain owed.

### Effect on factors 3a/3b/4/6/7/8/9

Only the absence of a structural consumer grammar is unblocked:

- factors `3a/3b` route together through one `SharedResizePreimage` range;
- factor `4` routes through semantic `RgbYuv6Projection` with integer
  BT.601/YUV6 output;
- factor `6` routes through cold-only `InitHeadSolve`;
- factors `7/8` route together through `Frame0PoseSixCarrier`;
- factor `9` routes through `BlindFillRateGrammar`.

These leaves remain `HAVE/PARTIAL`. None is newly measured, adopted,
`COMPLETE`, score-authoritative, or launch-ready. They still owe the production
interaction, byte-close, authority-axis, and adoption receipts named in the
sealed rows.

### #332 and Round-1 honesty

Canonical typed self-recompile/hash provenance is required, but the separate
strict repository-wide #332 audit is preserved as
`dsl_bijection_complete` plus `dsl_bijection_violations`. The Boolean may be
false with explicit debt; neither value can turn `launch_ready` true.

Round 1 rejected the earlier six-route/wrong-factor draft, the factor-5
multi-owner claim, the implied clean #332 bijection, and temporary equation
custody. The corrected route table above and tracked candidate-only equations
at `.omx/research/v10_compiler_receiver_equation_candidates_20260718.jsonl`
resolve those documentation confounds. The corrected controls additionally
refuse a counted-but-semantically- or decoded-frame-inert section, seed bytes
that reach neither generated frame, a paid zero `T` residual, and double
ownership of one frame/index residual. Handler/shared-semantic source digests,
canonical checkpoint base64, and cumulative prefix+suffix resume receipts close
the later custody findings. This is not an independent fresh-eyes clean pass,
and MAIN landing review remains required.

**MEASURED local verification:** `60 passed`; Ruff F/E9/I, `py_compile`, and
`git diff --check` were clean. No launch or score measurement occurred.
