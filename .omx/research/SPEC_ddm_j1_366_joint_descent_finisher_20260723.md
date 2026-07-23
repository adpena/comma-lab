---
title: DDM J1 #366 grammar-parametrized joint-descent finisher SPEC
utc: 2026-07-23T00:32:10Z
tasks: [366, 383, 549, 578, 603, 613]
lane_id: ddm_j1_366_joint_descent_ticket
status: SEALED_PREPARED_BLOCKED_NOT_FIREABLE
research_only: true
execution_allowed: false
score_claim: false
verdict_scope: proposed v15 grammar-parametrized joint Seg/Pose descent vehicle and its launch-readiness apparatus
main_landing_review_required: true
---

# Outcome first

The #366 trigger has fired, and the next vehicle is now specified: optimize the compact v15
description itself through exact `R`, frozen SegNet, and frozen official-YUV6 PoseNet. Island
worldsheets, lane productions, the nested v6 temporal base, and a small shared solved-template
bank are the parameterization. They are not post-hoc corrections to a fixed picture.

The accompanying typed ticket is hash-sealed, but it is deliberately **not fireable**. The landed
receiver can parse and render the 133,941-byte v15 archive, while no existing governed training
consumer can load that archive as optimizer parameter state. Consequently there is also no real
J1 config for `witness_memory_preflight.py` to model. Claiming “fire after governor + memory” would
be false until MAIN lands and reviews the adapter, typed compiler/launcher route, and matching
memory model. This SPEC stops at that exact boundary; no launcher or training process ran.

Canonical pointer: `0.1910828242 [contest-CPU]` unchanged.

## 1. Settled premise — cite, do not reopen

Commit `968e499a99640f811fd13da8e30531b2cf127425` merged the v15 result and closes the
post-hoc realization search at formulation scope:

- v12 drained 100% of its obligation pool and plateaued at d_seg 0.034004.
- v14 proved the information/realization split: Movable mask debt 0.000282948812 became
  through-R Movable d_seg 0.291615222639; G4 transferred only 5.29%.
- v15's strict zero-collateral shared-template feasible set was empty: zero steps admitted;
  improving proposals harmed at least 13 off-target Movable cells or 23 off-target Lane cells.
- v13's priced trade was net-negative.
- R1 is the positive counterexample: when Pose was inside the descent, it crossed the photometric
  wall and reached byte-close d_pose 0.001610; post-hoc pose composition did not.

Therefore #366 is no longer fallback prose. It is the required change of formulation: descend in
the low-dimensional description coordinates while both frozen evaluator legs are in the objective.

## 2. Current vehicle and warm start

Warm start from the exact v15 composed receiver:

`ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes`

- archive: 133,941 bytes, SHA-256
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`;
- receipt: SHA-256
  `5ed6f830b3749a51e0d300a9104fda9a77e86bbeb3b81428a20e1ec0d3dcfcb8`;
- advisory receiver row: d_seg 0.027470296224, Movable 0.291615222639, Lane
  0.435195521828, d_pose 163.061327281443;
- counted template bank: six shared records / 86 bytes;
- decoder scorer dependency: none; GT argmax table: none.

This is **receiver-loadable**, not **optimizer-loadable**. `receive_carrier_compose_archive()` owns
the parse/render leg, but neither `direct_description_minimizer.py` nor the level-set trainer has a
v15-to-trainable-state adapter. The receipt is even more specific: it owns a 29,810-byte G1
worldsheet payload with 10 object slots, but has zero explicit worldsheet track/knot records and zero
lane-program/knot records. Thus MAIN must define a typed parameter-lift for the G1 payload and an
explicit zero-state or counted seed for lane productions; it may not pretend those optimizer
coordinates already exist. Stage 00 is a hard custody gate: the adapter must round-trip the v15
archive to byte-identical receiver output before one gradient step, and every newly born lane DOF
must become counted immediately.

## 3. Counted description coordinates

Only these low-dimensional, decoder-consumed degrees of freedom may train:

1. Movable island worldsheet tracks, knots, and sparse birth/death events.
2. Lane production programs: chart drift, knots, dash phase, width, and persistence.
3. The six shared row-band RGB templates and a bounded, explicitly counted solved-template DOF.
4. The nested v6 temporal Pose6 base and admitted compact xi-event refinements.

No decoded per-frame mask, RGB plane, scorer weight, GT table, or post-hoc pose payload is legal.
All video-derived state is counted in the final archive. Generic deterministic renderer code may
remain free only under the contest generator rule. Archive parse-back must prove every byte has one
owner and one receiver consumer.

## 4. Joint optimization law

At each admission point optimize the same contest action:

`S = 100*d_seg(R(A(z))) + sqrt(10*d_pose_yuv6(R(A(z)))) + 25*|A(z)|/37,545,489`.

The differentiable inner step uses the exact resize/uint8 straight-through contract and
differentiable YUV6. Authority remains chunked exact evaluator replay of the emitted receiver bytes.
The #76 margin loop ranks cells in frozen-scorer Fisher/top1-top2 margin, uses the corrected
inner-Jacobian (first order + realized secant), and stops reverse-waterfill when marginal
improvement falls below `25/37,545,489 = 6.65860993116e-7` score units per byte. Any residual basis
is curvelet/shearlet; no Fourier substitution is authorized.

Stability and finish controls are inherited, not re-invented:

- EMA decay 0.997; stage checkpoints save the EMA shadow.
- #378 amber is launch-blocking: grad clip 0.5, pose-gradient coefficient cap 25, gradient
  normalization, and per-group clipping.
- #383 `PoseFinishConditioningGate` uses the rolling sigma-min plateau as the primary engagement
  event. Sigma-star is advisory only. Before the real-signal replay exists, the valid terminal
  fallback is banked R1, d_pose 0.001610, with a loud disengaged alarm.
- #549 is evidence that the joint frozen-scorer target exists; it is not evidence that the present
  compact grammar can reach it.

## 5. Stage ladder and preregistered exits

The exact values are encoded in the sealed typed ticket. Their epistemic status matters:

| stage | trainable group | d_seg | Movable | Lane | provenance |
|---|---|---:|---:|---:|---|
| 00 adapter replay | none | 0.027470296224 | 0.291615222639 | 0.435195521828 | MEASURED current no-regression row |
| 01 worldsheets | island + templates | 0.020602722168 | 0.145807611320 | 0.435195521828 | DERIVED fraction ladder |
| 02 lane productions | lane + templates | 0.013735148112 | 0.145807611320 | 0.217597760914 | DERIVED fraction ladder |
| 03 pose finish | all + xi | 0.006867574056 | 0.072903805660 | 0.108798880457 | DERIVED fraction ladder; d_pose <=0.001610 MEASURED anchor |
| 04 box fork | all compact groups | 0.00116 | 0.01 WATCH | 0.01 WATCH | global gate SETTLED; per-stratum watch targets SPECULATIVE |

These are stop/fork thresholds, not promised efficacy. Missed stages preserve their last EMA
checkpoint and emit the exact active per-stratum, pose, rate, or realization blocker.

The box fork fires only for the same receiver-closed artifact at d_seg <=0.00116 and <=200,000
exact archive bytes with the counted pose stream present. Then, and only then, run exact
contest-CPU followed by contest-CUDA evaluation. Neither axis can be inferred from the other.

## 6. Resumability and storage contract

Every stage ends with an immutable, stage-encoded checkpoint written atomically. Periodic
intra-stage checkpoints bound loss. Each stores all grammar/template/xi parameters, optimizer and
scheduler state, EMA shadow, stage/step position, seed and RNG state, typed-config/archive/cache
hashes, and exact-evaluator telemetry. Prior stage checkpoints are never overwritten.

Bulk run output belongs first under `/Volumes/VertigoDataTier/pact`, then
`/Volumes/APDataStore/pact`. The governed consumer must preflight space and include a success-only
scratch cleanup/cold-store path with a machine-readable reproducibility manifest. If provenance is
incomplete, cleanup blocks and preserves bytes.

## 7. Governed launch contract

The sealed ticket names the future config `ddm_j1_366_joint_descent` and supplies dry-run and fire
argv templates. This name is intentionally not an accepted launcher choice today. MAIN must land:

1. a typed `DirectDescriptionJointDescentTypedConfigV1` compiler and exact compile-hash verifier;
2. the v15 receiver-to-optimizer adapter with byte-identical stage-00 test;
3. a governed launcher route (no raw flags) and canonical resume registration;
4. a real J1 memory model/config accepted by `witness_memory_preflight.py`;
5. a bounded real-n600 dry start that steps, checkpoints, and resumes without launching.

After that landing, the prelaunch checklist requires governor ADMIT, live system-aware memory
receipt, config freshness (rc=6 on stale schedule), same-outdir refusal, DSL/hash equality, startup
telemetry, and a separate operator GO. Until every row is green, `execution_allowed=false` is
binding.

## 8. Memory and wall-clock honesty

At 2026-07-23T00:31Z this host reported 128.0 GiB total and 96.2245 GiB available. The historical
R1 level-set config projected 67.6 GiB standalone. That number is a **surrogate for a different
consumer**, not J1 admission authority. No real J1 peak can be reported until the adapter exposes
its tensors and verdict cadence to the memory model.

R1's 108-epoch / 48-minute-per-epoch history gives an idealized 17.3-hour value if full n600
verdict cost scaled exactly by `eval-every=5`. Because training cost and this vehicle's adapter are
unmeasured, preregister 17–30 hours as a DERIVED planning band and require a bounded timing smoke
before fire. The baseline is MLX-GPU with `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (the measured ~17x
path) plus fused differentiable-R. If the real smoke exceeds 30 hours or intermediate tensors bind
the 116 GiB ceiling, the ticket conditionally proposes a fused grammar-raster + exact-R JVP/VJP
(SPECULATIVE 1.5–3x step gain) and a batched YUV6/Pose VJP (SPECULATIVE 1.2–2x affected-component
gain). Both require equal-config parity and timing receipts. Kernels change speed, never authority.

## 9. Triality and durable exit

- DSL: proposed typed config in
  `.omx/research/configs/ddm_j1_366_joint_descent_witness_program_20260723.json`, canonical-hashed;
  executable compiler leg is explicitly owed.
- DAG: `.omx/research/ddm_j1_366_joint_descent_366_DAG_FEED_20260723.md`.
- Equations: the action and marginal-byte law above; existing #76/#378/#383/#549 laws remain the
  authorities. No new empirical law is registered from a prep-only task.
- Durable artifacts: SPEC, ticket, memory receipt, checklist, DAG FEED, and adversarial review.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`;
`SPEC_v75_optimal_single_trunk_20260708.md`; `SPEC_v8_perclass_decomposition_20260708.md`; #366
historical SPEC; #603 PRIMARY SPEC; R1 launch-prep memo; #383 repaired-pose-gate memo; #378 amber
synthesis; #549 inverse-solve memo; v14/v15 configs, receipts, code, and Codex findings; curriculum
DSL and LawRefs; governed launcher; memory preflight; lane/task/progress/frontier state; operator
Fisher/reverse-waterfill directives dated 2026-07-19; 2026-07-23 operator compute mandate.

Craft standard: `docs/operating_manual_craft_handoff.md`. MAIN landing review is required.
