---
title: "Codex findings: DDM SCHED1 derived schedule synthesis"
date_utc: "2026-07-24T12:48:00Z"
lane_id: "lane_ddm_sched1_derived_schedule_synthesis_20260724"
research_only: true
score_claim: false
promotion_eligible: false
authority_axis: "source-custodied advisory"
verdict_scope: "schedule formulation and current compiler/consumer support"
pointer_before: "0.1910828242 [contest-CPU]"
pointer_after: "0.1910828242 [contest-CPU]"
pointer_delta: 0
main_review_required: true
---

# Verdict

`RESEAL_REQUIRED`

The current #366 J7 ticket is not the optimum derived schedule. It is a fixed
three-stage, `3 x 150`-step, clipped-Adam program with fixed d_seg targets,
fixed verdict cadence, a bare EMA decay, no typed optimizer assignment, no
metric-stage selector, no causal event-mark consumer, and no executable
TerminalSolve handoff. That structure contradicts the event-continuation law
and cannot be repaired by changing a few values.

The exact reseal input is
`.omx/research/configs/ddm_sched1_derived_schedule_reseal_input_20260724.json`,
13,906 bytes, SHA-256
`65f06475a949cbd1607f6f221d81b453ef89bf12c9d95a63dabfafee38dbe429`.
It is deliberately `execution_allowed=false`: the semantic and typed hashes are
withheld until the missing DDM compiler/consumer exists. No argv, invented
flag, launch, paid dispatch, scorer run, score, or pointer movement was
produced.

## The optimum-form schedule

The derived object is an event graph, not a fixed stage list:

1. **Resume-boundary recondition.** Bind the MAIN-reviewed WS3 receiver-closed
   start; explicitly load optimizer state or declare fresh moments; re-anchor
   schedule state before baseline-v0; protect full learning rate with
   `adam_v_variance_warmup_length_v1` and
   `rewarmup_beta2_memory_window_v1`. Exit only on the first exact n600
   component-safe residual admission.
2. **Costate-ranked joint continuation.** At every accepted receiver state,
   select the next eligible coordinate group by measured positive marginal
   score benefit per counted byte, not by the fixed order
   island -> lane -> all. Admit only exact receiver-through-R improvement;
   otherwise halve/shrink and roll back exactly.
3. **Pose-protected finish.** Engage the existing J7 exact-n600 d_seg plateau
   latch and activate the pose trust region. The latch is retained; the fixed
   “stage 3 starts after 300 steps” clock is removed.
4. **Terminal solve or governed stop.** If the NCDE basin detector fires,
   topology is stable, no transition remains, and MS2 has complete metric
   custody, attempt the typed quotient solve and accept or roll back through
   the same exact receiver gate. Today MS2 is blocked, so this branch stays
   default-off. Stop when the exact reverse-waterfill has no positive marginal
   above rate break-even.

Fixed counts and wall time remain safety/resource caps only after an explicit
budget is supplied. They never decide a handoff. Stage count becomes an output
of the event trace.

## Loss and optimizer structure

The exact contest functional remains the sole acceptance authority:

`100*d_seg_R + sqrt(10*d_pose_YUV6_R) + 25*archive_bytes/37545489`.

#430 derives a coherent metric cascade—birth, boundary formation,
winner-rival repair, finite applied-step finish—but its non-incumbent metrics
remain `DERIVED_UNMEASURED` and its n600 selector/consumer is missing.
Therefore the reseal preserves the exact functional for every acceptance and
keeps the proposal-metric alternatives default-off until the selector receipt
exists. It does not pretend that a support gap is a schedule.

For the current heterogeneous 368-coordinate DDM surface, clipped Adam remains
only a proposal generator; exact receiver-realized discrete search is the
actual admission optimizer. SOAP, SPD momentum, generic SignGD, and
Muon/Manifold-Muon are not imported. No actual DDM update-matrix custody exists,
and many coordinates are scalar/vector rather than eligible matrices. Any
alternative must first produce a matched per-group realized-update-RMS receipt
from copied state, reset, then face the exact receiver gate.

The current `beta2=0.999` is not established as the DDM optimum. Only the
conditional rewarmup law is derived. The reseal therefore leaves beta2, the
rewarmup profile shape/floor, and the EMA pinned quantity unresolved and
fail-closed. Beta2 is fixed within a continuation segment; it is not silently
annealed per step. EMA must resolve through `ema_decay_run_geometry_v1` after
the event-run cap and a pinned seed/warmup quantity exist; `0.997` is forbidden
as an unexamined fallback.

# Elementwise differential against the sealed J7 ticket

Every current schedule element below has exactly one allowed disposition.
“Preserve” means preserve only in the role stated; it does not preserve an
adjacent fixed-clock interpretation.

| Current element / covered paths | Disposition | Derived action | Named receipt or law |
|---|---|---|---|
| stage-end, exact rollback, atomic, EMA-shadow, and resume checkpoint contract | `SAME-and-derived` | Preserve; add event-boundary and every-accepted-state checkpoints | J7 ticket `checkpoint_contract`; SPEC v7.5 §8 |
| n600 exact chunked CPU advisory and deterministic telemetry | `SAME-and-derived` | Preserve authority boundary and batch32 | `ddm_j7_366_fire_readiness_receipt_20260724.json` |
| `derived_steps_per_n600_exposure=150`, `derived_total_steps=450`, `3 x maximum_steps=150` | `CONTRADICTS-derived-law` | Remove as primary clock; optional safety cap only after budget derivation | `curriculum_handoff_critical_nucleus_v1`; fixed-stage poison directives |
| measured `100.87..104.10 s/step` and `13.31..13.79 h` projection | `SAME-and-derived` | Retain as measured resource model, never as handoff evidence | J3/J7 ticket timing receipt |
| `checkpoint_interval_steps=37` | `INHERITED-PR95-unverified` | Resolve from explicit recovery-loss budget; null until then | no LawRef in current `value_provenance` |
| `learning_rate_quantum_fraction=0.25` | `INHERITED-PR95-unverified` | Do not treat as an optimum; remeasure by coordinate metric/update-RMS receipt | px1 `pact.optimizer_update_scale_receipt.v1` contract |
| `plateau_verdicts=2` | `CONTRADICTS-derived-law` | Replace fixed count with deterministic event predicates and fail-invalid detectors | #302/#315/#344 |
| complete `pose_finish_engage` latch | `SAME-and-derived` | Preserve exact n600 d_seg latch and checkpointed monotone state | J7 fire-readiness receipt |
| transition only when fixed d_seg/d_pose target is met | `CONTRADICTS-derived-law` | Replace targets with exact marginal/event transitions and governed stop | `witness_measured_reverse_waterfill_v1` |
| stage 1 fixed active groups/order | `CONTRADICTS-derived-law` | Treat C1 groups as initial proposal seeds; active group selected by measured marginal | C1/V19/J7 value provenance |
| stage 1 target `0.020602722168`, cap 150, verdict 50 | `CONTRADICTS-derived-law` | Remove all three actuators | no target/cadence LawRef |
| stage 2 fixed lane/shared order | `CONTRADICTS-derived-law` | Lane enters when its measured marginal wins, not after stage 1 | #302 event continuation; reverse-waterfill directive |
| stage 2 target `0.013735148112`, cap 150, verdict 50 | `CONTRADICTS-derived-law` | Remove all three actuators | no target/cadence LawRef |
| stage 3 fixed all-groups pose finish | `CONTRADICTS-derived-law` | Pose finish enters on the preserved latch; group set remains costate-ranked | J7 pose latch; #430 coherent cascade |
| stage 3 targets `d_seg=0.006867574056`, `d_pose=163.06116431842463`, cap 150, verdict 50 | `CONTRADICTS-derived-law` | Keep the measured pose value only as a protected reference, never a fixed stage target | J5/J7 receipt |
| `train_batch=4` and worst-geometry memory preflight | `SAME-and-derived` | Preserve until WS3/config changes invalidate the memory receipt | J7 worst-geometry receipt |
| `warm_start_pair=447`, `warm_start_steps=4` | `INHERITED-PR95-unverified` | Retain as historical probe provenance only; not a full-run clock | J5/J7 bounded-history receipt |
| `adam_beta2=0.999` | `INHERITED-PR95-unverified` | Require DDM-specific beta2 selection receipt | #405 `AdamBeta2`; px1 |
| `lr_rewarmup_c=2`, `lr_rewarmup_steps=2000` | `SAME-and-derived` | Re-resolve from selected beta2 at every explicit state boundary | `adam_v_variance_warmup_length_v1` |
| rewarmup `floor=0.1`, `shape=linear` | `INHERITED-PR95-unverified` | Leave unresolved; no silent profile transfer | `rewarmup_beta2_memory_window_v1` explicitly says profile is provisional |
| exact first admission, shrink, exact rollback, cumulative component gate | `SAME-and-derived` | Preserve as the receiver admission controller | J4/J5/J6A/J7 receipts |
| opening C1 groups, four candidate families, eight pair seeds | `SAME-and-derived` | Preserve as a seed menu, not an exclusive stage schedule | V19/C1/J7 value provenance |
| Q8 staging, denominator, geometric shrink multipliers | `SAME-and-derived` | Preserve current exact proposal ladder | J7 `proposal_ladder` and `camera_q8` provenance |
| `pure_priced_exact_n600` acceptance and residual-bucket gate | `SAME-and-derived` | Preserve | J5/J6A/J7 receipts |
| joint formula and receiver roundtrip | `SAME-and-derived` | Preserve without proxy promotion | evaluator-equivalent witness contract |
| `ema_decay=0.997` | `CONTRADICTS-derived-law` | Resolve by `ema_decay_run_geometry_v1`; no fallback | EMA executable LawRef |
| amber `grad_clip=0.5`, normalization, per-group clip, pose coeff cap 25 | `INHERITED-PR95-unverified` | Preserve only as current control until a matched stability receipt; not optimum schedule constants | current ticket; #405 default-off doctrine |
| V15 warm start with `optimizer_state_loadable=false` | `CONTRADICTS-derived-law` | Replace with MAIN-reviewed WS3 binding and explicit loaded-or-fresh state boundary | #517/#518 warm-start laws; missing WS3 receipt |
| effective 368-coordinate receiver surface | `SAME-and-derived` | Preserve until WS3/IS1 changes the receiver schema | J7 ticket/consumer |
| clipped Adam on all active coordinates, implicit in launcher | `INHERITED-PR95-unverified` | Make group assignment typed; retain only as current proposal control | launcher source; px1 |
| no metric-stage selector | `CONTRADICTS-derived-law` | Add typed default-off #430 selector and exact n600 gate | #430 build/findings |
| no NCDE/TerminalSolve handoff | `CONTRADICTS-derived-law` | Add typed conditional branch; keep off while MS2 custody is blocked | #344; `TerminalSolve`; MS2 findings |
| no DDM causal event-mark stream | `CONTRADICTS-derived-law` | Add resume-safe `pact.causal_manifest.v1` event marks | #474 event-mark spec |
| no hashed #405 default-off consumption receipt | `CONTRADICTS-derived-law` | Compile a hashed decision receipt; never auto-arm candidates | #405 comprehensive sweep |
| schema `DirectDescriptionJointDescentTypedConfigV1` fixed-stage parser | `CONTRADICTS-derived-law` | Add `DDMEventContinuationV1` and a DDM WitnessProgram target | #334/#339; current parser source |

# Required corpus sweep

| Surface | Re-derived disposition for #366 |
|---|---|
| #302 curriculum derivation | Fixed stage count/clock is cargo cult. The levelset-specific nucleus/Muon domains are not relabeled as DDM laws; their event-continuation structure is consumed. |
| #430 coherent whole | Metric cascade is derived, but non-incumbent choices and selector remain unmeasured. Encode as default-off typed choices, not a launch treatment. |
| #318/#320 DE mechanisms | Adaptive epsilon is `N/A-WHY`: DDM has no eikonal viscosity field. The reusable lesson is state-dependent stability, not its numeric epsilon schedule. |
| #315 event handoff | Consume event semantics. Do not emit its levelset trainer flags into the DDM launcher. |
| #344 NCDE | Shadow-only detector; unstable/low-r2 fits never fire. It may request a terminal-solve attempt but cannot actuate launch or promotion. |
| #341/#342 solve inventory and `TerminalSolve` | `TerminalSolve` is designed/not built and compiles to no argv. Current DDM MS2 verdict is `BLOCKED_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE`. |
| fixed-stage / constant poison memories | Consumed as structural vetoes. No PR95 schedule numbers or old-lineage code were used. |
| #518/#517/#270 warm starts | Preserve explicit state boundary, beta2 memory, resume-relative schedule positioning, and momentum-cold-start lessons. Do not import levelset Muon settings into DDM. |
| #469/#552/#556/#175/#222/#448/#443 | Existing geometry-specific candidates stay default-off or scoped; no generic duplicate optimizer is added. Beta2 conditional warmup is retained, beta2 optimum is not claimed. |
| EMA LawRef | Current bare `0.997` fails. `ema_decay_run_geometry_v1` is required and unresolved until event-run geometry plus a pinned quantity exist. |
| #312 | Loss/metric weights change only at event boundaries, never by per-step balancing. |
| #289 | Commit must use the canonical serializer with post-edit SHA; this unit does so at landing. |
| #475 | Grokking is a terminality/feature-poverty guard, not a witness schedule clock; no transfer. |
| px1 SOAP/Muon/Beyond | Adopt the update-RMS and actual-update polar-quality measurement contract; adopt no constants. |
| #474 causal manifest | Default-on read-only causal rows exist for the levelset trainer; DDM event marks are not built. The gap blocks verified event replay. |
| #405 default-off | No candidate is silently armed. The current warn-only consume gate is insufficient for #366 fire; hashed consumption is owed. |
| #334/#339 DSL | Schedule/Curriculum/WitnessProgram and typed support gaps are the correct paradigm, but the current target is the levelset trainer. A DDM target must land before hashes or argv exist. |

# Why typed compile refuses

`FullRunScheduleV1.from_semantic_program` requires a nonempty `stages` array,
and validates `maximum_steps` as a positive multiple of
`verdict_interval_steps`. `tools/launch_ddm_joint_descent.py` then loops those
stages and calls `clipped_adam_step`. It has no event graph, optimizer router,
metric selector, causal event-mark consumer, or terminal-solve hook.

The general `WitnessProgram` can express event curriculum intent and
`TrainerSupportGap`, but it compiles the levelset trainer. Its
`TerminalSolve.flags()` intentionally emits nothing and reports a support gap.
Therefore there is no honest semantic/typed hash for the proposed DDM schedule
yet. The reseal input withholds both hashes and emits an empty argv.

# Fire composition and exact blockers

Fire remains MAIN-owned:

`WS3 READY AND IS1 admissible final verdict AND SCHED1 SCHEDULE_VERIFIED_DERIVED`.

This isolated worktree contains WS1/WS2 artifacts but no WS3 final READY
receipt. It contains IS1 directives but no final IS1 verdict. SCHED1 itself is
`RESEAL_REQUIRED`. Hence `ready_to_fire=false` independently on all three
composition surfaces; this unit neither launches nor mutates the frontier.

The machine-readable reseal input names ten blockers. The minimum exact landing
delta is:

1. build `DDMEventContinuationV1`;
2. add a DDM `WitnessProgram.compile_*_with_constants` target;
3. bind MAIN-reviewed WS3 and IS1 receipts;
4. resolve beta2/profile and EMA LawRefs without fallback;
5. build the n600 metric selector;
6. add resume-stable DDM causal event marks;
7. compose the MS2 terminal solve only after its metric-custody blocker closes;
8. derive resource/checkpoint caps from explicit budgets;
9. compile real argv plus constant manifest and fail-closed semantic/typed hashes;
10. run a new independent three-pass review on that executable artifact.

# Triality and directive consumption

- **DSL:** proposed `DDMEventContinuationV1` must compose through
  Schedule/Curriculum/WitnessProgram; current support gap is explicit.
- **DAG:** #366 fire gate remains `ws3 x is1 x sched1`, MAIN-owned.
- **Equations:** consumes `curriculum_handoff_critical_nucleus_v1`,
  `rewarmup_beta2_memory_window_v1`,
  `adam_v_variance_warmup_length_v1`,
  `ema_decay_run_geometry_v1`, and
  `witness_measured_reverse_waterfill_v1`; no new unanchored equation was
  registered.
- **Directives:** all authority-file directives and the two relevant broadcast
  reverse-waterfill/Fisher directives were consumed. IS1 directives are
  recorded as pending inputs, not laundered into a final verdict.

# Source custody

Every normative input below was hashed at consumption; commit is the last
repository commit touching that path.

| Input | SHA-256 | Commit |
|---|---|---|
| current J7 ticket | `b3e5fe8adcc1ff6f4cc5fa3e4ac124e20cf6ed862810f0361b117dab5dd0e41f` | `26c2077892e028686b0486d64064c8b5fff7ea11` |
| DDM typed consumer | `e4dace7cbbe7d28d76102921ffc4f0e02d710486c64561d58c762b7568306352` | `125174a83275b35d1dfedeaef806afd1647e7504` |
| DDM launcher | `1239bd359f472605f0ebe1a0d1969d61ff984f086b97b2d3951803521195892a` | `cfb6f3270abc720bdee03fb00d29e02e6a2e60dd` |
| J7 findings | `70c815843659bf0ddd742da20ac0599355fb1e076d71311e3ac28a58ffe849d8` | `26c2077892e028686b0486d64064c8b5fff7ea11` |
| canonical DAG | `ab954f74e043a2c699199a93ec69bc1ff7baefd1a1b8f197de14c729dc2909e6` | `1bf072a3579933e5dcd5a09a8f3e75a8a169f4ef` |
| #302 symposium | `a095f09a33049b4632c4bec5572ac76f282480df22ea71a7a648a6b03d35637c` | `fab4f0145de96ccd05834f43b591260f378e040e` |
| #302 differential derivation | `6aa6d1948a5be898aaf646d2b5455c9376d933441fd11d83d73db2100951c6c8` | `e420d9a1b76680db8e90491a036e3ac33d44171d` |
| #315 event handoff | `014127646965977e1c2d38c02d2d2ac13c4708d9cf0c63c0c000bcdc68486bf9` | `bc2ece94f81b1f58cdf1bc7f9c95f45fda4da243` |
| #318/#320 adaptive epsilon | `aa1f89d93be547f4c5431f9b518ce615e767b1c452051c367933c93910091150` | `670ae1e4fc0cfa3ee3fca737902ef5578429da19` |
| #430 coherent schedule | `a059d5d047c9cd78b7fef96cc5bb236fab1d33781d98d9af8c37058e0152a77b` | `a728be3bc1db19749cf348b534867b58c0cff279` |
| #430 build spec | `ff8616d9005fbe7e3cdfa5bc0fba169f8d12906439cf57f0e372d109baa2b60e` | `d25d76662062c0962326ae352c133378ef7b60f8` |
| #430 findings | `f90bdddeb351cce2842d138bf365426ff98edcd35068b6d16f184114e7b2efda` | `d25d76662062c0962326ae352c133378ef7b60f8` |
| Schedule/Curriculum/WitnessProgram | `039ea875ad1f03c7cd4a7b41c3037cde35d7583659cfe0c9b2eb230a3023604b` | `46f7e84a8009b67fe1358d54894a5b805529ff33` |
| curriculum LawRefs | `4414188ac7f0eeab615ffadde0e0d968a235b82e6904b6dd97838f054acdcdad` | `563f9e70cc664916558e0b59489c863a7dc31dd5` |
| EMA LawRef | `470b44b8fff5c507878b2a2228987f472397a51f620082d55b515815d9014498` | `c7815b41ded2c86b7c99fad27c9d9dbba9b11506` |
| Adam variance warmup LawRef | `98fcbb25f3c23626b05a1233ea16a9a71984c513793b9948a9f5af78904f0e4f` | `4b53163c17845c60cf60082001d5221f57929966` |
| #344 NCDE | `01ceb6a898ff8575df31933b44da943dbed8b50069860be04ee5466dd2a1fe06` | `0793e6566291f99304c9c194f2d49b1992c645cd` |
| solve-don't-train inventory | `aff5e0165f028e6036bcb8daf0e294b77f28a2dadb949d31cd240b2bb177c978` | `5f05b07432cb1d9bb12e13334a441b9490081623` |
| DDM MS2 findings | `e0b5302e0ff72261d81b7795430716117f294fe8a392a2506e0cf0529924f410` | `1ae282ea6f43e40216bb9e2563b36e7464caa3cb` |
| #517/#518 warmup build | `9fcf19501bb20b67bb8cdfd83c866aee8d29435c3f893d75b8b8ede91167de42` | `ead2a1376042e380a3f06d205348163a8e38eabb` |
| px1 optimizer crosswalk | `6b4a0cd5e3d3a0ac1ebef3674c512484cb5967266b52894c82a5f312c29da918` | `5fd7b8f214a6e9fe1762718306e3166302213901` |
| Manifold-Muon | `bc3d923d7ab5238abd05983f593a7e4727f64b1d8ba4a05f0dc570c46c2bba2f` | `aefd1130e10f8f91bac66e9d3e56dd9de0a5a966` |
| SPD momentum | `bea62d192e0f8354043cdf64ec900b709a8c0873e3fa93ffd08bd2b4d3b0cd57` | `e711e4013c35425b4fdd603f7b07b938c5d25a2b` |
| matrix-calculus crosswalk | `7d919d48ec468152e998aa76da191a0b41971191880df2681138682dec4aa548` | `7cb9ffbef9c0d331cca470fe6203d0ee3b8672ad` |
| OptStep crosswalk | `12b1230b7889d1b48014b9c0d84c236deec33e1cfca1bd1437eb2a230b04a9e4` | `33395ddc0c2fec62fb03f9baf9cb52cd64e552f0` |
| #405 default-off sweep | `668595ea56ef75ba980a532a79b4f7a190dbe3d8a06725c4e9fc261fab4ff406` | `12d5d0379b9445b402d26c8058398fe7d7baf4f2` |
| #474 event-mark spec | `a0c15dca207f75191eb57dbec2f6c31279c62421ff200bebe06d0d177d5fa17d` | `94fa8a82c186077312186aa3a1a5765fd3d26131` |
| next-launch causal consumer | `4c92ceffaaf70066d224ca72094cd736ab5cefeb53a5800a46256b6b0cdfdc26` | `f49de340f5122e87a82f43a97a9eae3feb795ac5` |

# STORES CONSULTED

- delegated authority file, `CLAUDE.md`, `AGENTS.md`, operating manual, and `PROGRAM.md`
- canonical DAG, lane registry, subagent ownership/checkpoints, frontier pointer, per-arm and broadcast inboxes
- current #366 J1-J7 ticket, consumer, launcher, findings, fire-readiness, and three-review receipts
- #302/#430/#318/#320/#315/#344/#341/#342, poison-memory, warm-start, optimizer, EMA, causal-manifest, default-off, and DSL surfaces named above
- WS1/WS2 artifacts, WS3 absence, and IS1 directive-only state

Historical memory was used only to locate the curriculum audit and the
PR95-as-control rule; every factual conclusion in this memo was reverified
against the current repository.

MAIN landing review is required before this finding can affect the canonical
#366 fire gate.
