---
title: "#366 direct joint-description descent finishing specification"
utc: 2026-07-21T21:08:58Z
task_id: 366
lane_id: realization_verdict_admits_or_dominated
status: QUEUED_OPERATOR_GO_REQUIRED
execution_allowed: false
research_only: true
---

# #366 direct joint-description descent finishing SPEC

## Ticket state

**QUEUED — operator GO required. No launch or paid dispatch occurred.** This is an exact build and
launch contract, not a runnable argv. A new typed direct-description consumer must land, pass its
tests, storage preflight, resume audit, and governed-launch refusal checks before any GO can become
executable. Raw flags may not be hand-appended.

Success means one **same-artifact** tuple at n600. The build must ingest a full-precision,
SHA-bound official C1 solved-target receipt and derive its thresholds and byte caps at compile
time; rounded values may not enter the launch config. Planning values from the displayed tuple
are `d_seg≈0.00015196`, `d_pose≈0.00010184`, pointer cap 216,223 B, and strict-sub-0.15 cap
154,524 B. These are planning ceilings until the full-precision receipt is bound.

The actual admission contract is:

- `d_seg <= solved_target_receipt.d_seg` and `d_pose <= solved_target_receipt.d_pose`, both
  measured through the integer/uint8 receiver and frozen scorers;
- `pointer_cap_bytes = ceil((pointer_S - nonrate_S_full_precision) / rate_price) - 1` and
  `strict_0_15_cap_bytes = ceil((0.15 - nonrate_S_full_precision) / rate_price) - 1`;
- deterministic parse-back and two independent inflates with identical raw SHA-256;
- exact archive SHA/bytes, runtime, source, config, seed, hardware axis, and scorer custody.

The final authority remains separate exact canonical-evaluator runs on contest-CPU Linux x86_64
and contest-CUDA. This worktree has not established the evaluator path/hash; resolving and binding
that exact surface is a readiness gate. macOS CPU/MLX rows are advisory only.

## Exact typed config skeleton

The builder must extend, not bypass, the existing compiler:

| Field | Frozen specification |
|---|---|
| parent compiler | `tac.witness_dsl.spec_v9_cgauge.compile_v9_cgauge_ideal_mod19_launch_config` |
| pairs | `600` |
| seed | `1234` for every Python/NumPy/MLX/Torch RNG derived from one recorded seed |
| objective | exactly `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489` |
| receiver | C1 deterministic generator -> RGB uint8 -> exact integer factor-2 `R` -> frozen SegNet/PoseNet |
| solved target | required full-precision official C1 receipt plus SHA-256; no rounded fallback |
| description variables | one counted archive description; no post-hoc correction composition by #366 program policy |
| scratch/custody root | `/Volumes/VertigoDataTier/pact/evidence/realization_verdict_20260721/task366_joint_description_descent_<launch_utc>/` |
| schedule | event-driven only; epochs/dwell are fail-safe ceilings, never semantic stage boundaries |

### Descendant composition and custody order

The parent already owns unified semantic form, event tau, the Pose conditioning gate, the
store-nothing pose-carrier flags, `--micro-batch-pairs=1`, and the governed
`legacy_fourier_ab_control` basis. The descendant must not compose duplicate owners.

The exact compiler order is:

1. call `compile_v9_cgauge_ideal_mod19_launch_config(..., flag_custody=False)`;
2. mutate the custody-free typed config with the new direct-description owner and any reviewed
   descendant-only levers;
3. update and validate a descendant expected-active-lever manifest;
4. compile final argv, validate single ownership and all required invariants;
5. derive the constants manifest from final emitted argv and merge every descendant
   lever/component LawRef and constant manifest;
6. build a new descendant DSL program manifest containing the descendant program name, final
   emitted flags, expected-active levers, and an explicitly named
   `pre_custody_typed_config_hash`, then assert manifest-to-argv parity;
7. call `attach_flag_custody(...)` **last**, overwrite the standard
   `dsl_manifest["typed_config_hash"]` with the post-custody `typed.typed_config_hash()`, retain
   the pre-custody hash separately, and reassert final argv byte identity/hash before freezing
   typed/config/argv hashes.

Inherited single owners to reuse:

- `seg_form_unify_tau` for the continuous semantic homotopy;
- `unified_tau_eikonal_hold` for `--tau-advance-mode event`; do not add
  `TauAdvanceEvent()` on top;
- `pose_finish_conditioning_gate` with sigma-min plateau as the primary event and its existing
  positive-epoch fail-safe backstop;
- the inherited store-nothing pose carrier and counted `xi/dxi` state;
- `legacy_fourier_ab_control` for the first matched control run.

Descendant additions, subject to expected-lever and consumer tests:

- `PoseEligibilityOnTauFinalRungV1` (**TO BUILD**) so pose eligibility consumes the persisted
  tau-controller final-rung event rather than an epoch value;
- `PoseBlindComputeGate()`, with compile assertions that the inherited positive
  `--pose-finish-start-epoch` and `--micro-batch-pairs=1` remain true;
- the new `DirectCountedDescriptionJointSolveV1` owner specified below.

The first run may not silently flip the inherited basis. A compact-shearlet or windowed-curvelet
treatment requires a separately preregistered matched A/B, reviewed custody-manifest amendment,
n600 no-regression gate, and operator GO. The legacy basis remains only the governed A/B control;
it is not promoted as the target representation.

Deliberate exclusions:

- `EventTriggeredCurriculum()` is not added; its fixed CE-to-tau handoff surface is inert after
  the form is unified. The inherited `unified_tau_eikonal_hold` owns event tau.
- `PoseMarginalWeightLaw()` is excluded because the exact score-domain pose term is already
  `sqrt(10*d_pose)`; applying the marginal law would square the marginal. The score-domain
  coefficient stays 1.
- `PoseFinishBetaAnnealCoupling()` is excluded from the launchable config: its current consumer
  gates on `anneal_epochs`/`epochs`, not the persisted event-tau final rung. It may be superseded
  only after `PoseEligibilityOnTauFinalRungV1` has a real consumer and tests.
- No new FFT/DCT/rFFT treatment is introduced. No post-hoc additive pool is admitted by the
  #366 program policy.

One new typed component is **TO BUILD** before launch:

`DirectCountedDescriptionJointSolveV1`

Its schema must contain, and its consumer must enforce:

```text
objective_domain = exact_contest_score
rate_source = canonical_archive_parseback_bytes
receiver_domain = integer_uint8_R
candidate_sources = seed_or_conditioning_only
hard_oracle_cadence = every_admission_event_and_checkpoint
optimization_scope = all_description_variables_jointly
post_hoc_composition = forbidden
n64_gate_before_n600 = required
pose_finish_eligibility = persisted_tau_final_rung_event
pose_finish_backstop_cannot_bypass_eligibility = true
```

The DSL addition must resolve to a real consumer and a provenance-complete config manifest;
a lever with no consumer or a hand-invented raw flag is a launch blocker.

## What the generator must learn through uint8

Reuse the already-settled v8/v9 per-class carriers; do not rediscover them:

| Semantic class / score axis | Counted object learned inside the generator | Exact receiver debt |
|---|---|---|
| MyCar | one static image-frame mask/rim generator | preserve the rigid bottom region after uint8/R |
| Undrivable | low-frequency horizon/bulk boundary with slow temporal knots | preserve top-region cells and its Road separatrix |
| Road | ground/bulk field plus oriented one-sided separatrix coefficients | put protected Road cells on the correct argmax side after uint8/R |
| Lane | analytic ground-frame thin bands, curve rhythm, and shallow-side precision | birth and retain Lane cells without Road spill |
| Movable | sparse object islands, births/deaths, and per-object motion parameters | preserve topology events and object cells |
| Pose | jointly trained `xi/dxi` and generator-conditioned frame pair | enter the six-value Pose tube without sacrificing semantic cells |
| Rate | entropy model, symbol order, quantizers, and grammar choices | minimize actual parse-back archive bytes, not a proxy tensor size |

The five semantic-class carriers must cover the four canonical strata
`cell_interior`, `boundary_codim1`, `movable_track`, and `critical_event` wherever applicable.

The I1 chart coefficients, I2 canonical plane/member, I3 description map, g2f trust regions,
coder rungs, and the solved lattice plane may initialize or condition this solve. They may never
be rendered independently and composited afterward. The differentiable inner step must include
the scorer-aligned Jacobian/Fisher/margin geometry and periodic exact secant or finite-difference
checks; every admission decision is made by the hard uint8/R oracle.

## Event-driven run graph

1. **SEED-BOUNDARY event:** validate source hashes, charged/free boundary, quarantine scan,
   parser schema, and complete archive-description variables. Save atomic `seed_bound` checkpoint.
2. **N64 RECEIVER/CUSTODY smoke:** decode a deterministic, stratified 64-pair candidate twice,
   verify byte-identical raw output, require every selected declared semantic cell and Pose tube,
   and cover all five semantic classes across applicable canonical strata. This is advisory and
   non-extrapolative; no n600 global `d_seg` threshold is applied to n64.
3. **TAU-RELAXED events:** advance one geometric tau octave only when the registered per-band
   relaxation detector fires. Save a distinct atomic EMA-shadow checkpoint at every octave.
4. **POSE-CONDITIONED event:** the persisted tau-controller final rung is a hard eligibility
   predicate for **every** engagement signal. Sigma-min plateau is the primary signal. The
   inherited positive start epoch remains a fail-safe backstop, but cannot bypass eligibility:
   if its cap arrives before the final rung, emit `cap_fired_before_event` and keep Pose OFF; only
   after final-rung eligibility may the recorded backstop permit fallback engagement, which is
   not counted as event success. Save pre/post engagement checkpoints and rely only on the
   independent hard oracle for admission.
5. **RATE-KKT events:** retain only changes whose measured same-pool marginal exceeds
   `25/37,545,489 S/byte`; a negative local marginal stops that pool, not an unmeasured family.
6. **N64-SMOKE-PASSED event:** require the receiver/custody smoke above before allocating n600;
   do not call it a score admission.
7. **N600-ADMITTED event:** compile the complete archive, parse it back, double-inflate, hard-score
   all 600 pairs, and emit the exact tuple. Otherwise emit an exact blocker and keep the bytes.
8. **CONTEST-AXIS events:** only after local closure and exact evaluator path/hash/custody
   resolution, queue separate contest-CPU and contest-CUDA exact replays; neither axis is inferred
   from the other.

Epoch counts, maximum dwell, and total timeout are capacity backstops selected by the governed
launcher after timing smoke. They do not choose curriculum transitions.

## Resumability, storage, and cleanup

- Complete state lives on the SSD root: generator/decoder, description variables, entropy model,
  optimizer, EMA shadow, event index, tau rung, Pose-gate state, RNG state, config, source hashes,
  and archive compiler state.
- Save atomically (`tmp` + `os.replace`) at every event boundary and periodically within any long
  event. Preserve every stage/event checkpoint with a distinct encoded filename.
- A resume must reload byte-close and continue from the recorded event; it may lose at most one
  intra-event interval.
- Run `tools/witness_memory_preflight.py` and the storage waterfall before timing smoke or launch.
  Governor REFUSE is authoritative.
- Success-only scratch is auto-cleaned. Material bytes are moved or deleted only after a
  machine-readable rebuild/cold-store manifest records path, bytes, SHA/tree hash, config/argv,
  source/runtime hashes, destination, and false-authority flags.

## Fail-closed readiness DAG

The operator GO remains non-executable until all are green:

1. typed `DirectCountedDescriptionJointSolveV1` and `PoseEligibilityOnTauFinalRungV1`, each with
   a real consumer and pure compile tests, including proof no backstop bypasses final-rung
   eligibility;
2. deterministic NumPy reference plus MLX/Torch portability surfaces: floating intermediates
   must meet measured parity `>=0.9997`, while final receiver RGB/raw output must be byte-identical;
3. archive grammar, parser section manifest, charged/free bijection, and quarantine gate;
4. the deterministic stratified n64 receiver/custody smoke defined above;
5. resume-from-every-event-boundary and EMA-shadow byte-close tests;
6. memory/storage/timing smoke with a prelaunch n600 decode estimate below 1,800 seconds, followed
   by a measured final n600 decode below 1,800 seconds for admission;
7. canonical evaluator availability, bytes/hash, dependency, and scorer-custody gate;
8. lane dispatch claim and governed launcher approval;
9. explicit operator GO.

Until then: `QUEUED_OPERATOR_GO_REQUIRED`, pointer unchanged.
