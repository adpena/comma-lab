---
title: DDM C1 composed receiver-closed candidate specification for Tasks 603 and 613
date_utc: 2026-07-23
lane_id: lane_ddm_c1_composed_candidate_spec_603_613_20260723
tasks: [578, 603, 613, 366]
research_only: true
execution_allowed: false
score_claim: false
d_seg_claim: false
d_pose_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: INFEASIBILITY_CERTIFIED_FOR_EXACT_COMPUTED_SET_FULL_COMPOSITION_COMPUTABLE_NOT_YET_COMPUTED
verdict_scope: "COMPOSITION x measured n600 receiver-closed prices; no family, launch, contest-axis, score, or promotion verdict"
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_landing_review_required: true
---

# Verdict first

The one-archive architecture is coherent, but it does **not** arithmetically reach the Task
#613 box under measured receiver-closed efficiencies. The measured v14 operating point has
3,240,528 Seg errors over 117,964,800 sites. `d_seg <= 0.00116` permits at most 136,839
integer errors, so the archive must remove **3,103,689 errors**. Perfectly eliminating every
currently measured Lane and Movable error removes only 726,416 errors, or 23.4049% of the
continuous in-box debt, and leaves **2,377,273 errors**. No measured receiver-closed component
in the composed inventory owns that residual all-role debt.

The candidate therefore remains a preregistered target architecture, not a fire ticket. The
live/successor measurements are coupled by

`E_v17 + E_v18b + E_j3 >= 3,103,689`, with `0 <= E_v17 <= 726,416`,

where each `E` is an integer count of errors removed by the **same final exact-R artifact**
relative to the 3,240,528-error control. The later operator supplement reports `rho<0` at all
measured v17 validity radii, so the current v17 formulation earns `E_v17=0`; the still-open
successor is the preregistered high-resolution-pre-uint8 versus post-quantization-int8 placement
race. Overlap is assigned once by sequential exact replay.
The stronger #366 pose-finish gate is `d_pose <= 0.00161` on that same artifact. Merely carrying
the existing 3,721-byte Pose6 stream does not satisfy the pose-finish gate: its current measured
`d_pose` is 163.061327281443.

## Candidate identity

`DDM-C1-COMPOSED-200K-v1` is one deterministic receiver with this fixed order:

`PREDICT -> PROJECT -> REALIZE -> FINISH -> CODE`

The base is the exact v15 archive lifted by J2, not an additive union of historical archives.
The J2 lift exposes 706 low-dimensional counted parameters while re-emitting the 133,941-byte
v15 archive and receiver camera bytes identically. A 270-byte counted Lane seed makes all
declared trainable groups receiver-owning at a 134,211-byte seeded control.

## Stage ownership and non-additive pools

| stage | mechanism selected for the candidate | named debt slice | measured price / evidence | admission rule |
|---|---|---|---|---|
| **PREDICT** | v13 birth/death worldsheet events over the G1 Movable description; Lane production seed; counted Pose6/xi temporal chart | Description and temporal support only. It owns no claimed exact-R error reduction until PROJECT replay. | G1 Movable knee 29,810 payload bytes and mask-space `d_seg=0.000282948812`; Lane knee 27,692 bytes. V13 moved Movable conditional `0.9895 -> 0.4813`, the largest measured descent in the arc. J2 exposes 163 tracks, 2,197 knots, 2,047 shape templates, 6 Lane seeds, and 706 counted parameters. | G1 knees are candidate-set bounds, not optima. The Lane knee competes with the current nested Lane stream; it is not added. Planar #601 `+180,280 B` and Screw6 #605 `+79 B` n16 are controls, not selected transports. |
| **PROJECT** | one camera-resolution receiver, uint8 projection, evaluator R, frozen SegNet and official YUV6 PoseNet | Accounting authority for every downstream slice; it reduces no debt by declaration. | V14 exact selected control: 133,247 B, `d_seg=0.027470296224`, `d_pose=163.061327281443`, 3,240,528 errors. Fixed paint realizes only 39.4381% of the Movable mask-to-receiver gap and leaves 60.5619%. | Every proposal must parse/re-encode identically and be replayed through the same exact-R master. Cell-space or mask-space improvements are not transferable claims. |
| **REALIZE** | solved shared templates plus contextual `2x2` and boundary-normal banks under epsilon-collateral `{0,16,32,64}`. Each correction races high-resolution FP pre-uint8 placement (sub-quantum dither/error-diffusion allowed) against exact post-quantization int8 lattice placement. | First ownership: the 726,416 Lane+Movable errors at the control. This is a ceiling, not a forecast. | V15 `1x1 x three row bands` is byte-positive but camera-output identical; its hard-zero-collateral feasible set is empty. V16 has a real `151 x 141` local map and sampled FD error `1.5648e-5`, but the one-shot validity radius is below the destructive uint8 quantum. The later operator supplement reports v17 `rho<0` at every measured validity radius, so it earns zero current credit. G4 horizon is the only measured narrow receiver-positive rule: +508 B, -6,320 errors, `d_seg=0.027416720920`. | Each correction record must name `application_stage`, exact added bytes, and realized flips at that stage. Pre-u8 and post-int8 variants compete; they do not stack by default. Accept only exact joint-objective improvement after uint8/R replay; relinearize after every accepted step. |
| **FINISH** | J2/J3 joint descent over the lifted worldsheet, Lane program, shared-template, and xi-event degrees of freedom; v18b generated columns may enter the same trust region | Residual all-role Seg debt after REALIZE, plus the pose tube. With maximal REALIZE credit the minimum Seg slice is 2,377,273 errors; pose target is `d_pose <= 0.00161`. | J2 pair-447 resume smoke moves `d_seg 0.022811889648 -> 0.022705078125` and pose MSE `38.906909943 -> 38.906646729` while proving crash-resume and exact checkpoint parse-back. No full-n600 finish rate is measured. V18 initial receipt is correctly blocked before pricing round 1 because no common exact-R master exists. | v18b and J3 compete on overlapping residual errors; their gains are not summed unless the later component is replayed on the earlier component's exact bytes. The coupled inequality, not memo attribution, is authority. |
| **CODE** | exact byte-home race per admitted stream: Aurenhammer LP representative; context arithmetic/Selfcomp block-FP; xi-keyed delta; queued 2:4 and MX-int4 entrants | Rate only. CODE owns zero Seg or Pose debt unless re-encoding changes exact-R output, in which case it returns to PROJECT. | Aurenhammer LP is 134 B versus tropical 137 B under the same coder. G4 derives 89,161 B / 18.166685% savings for its **future innovation stream** using free decoder-derived context. | Race alternatives on identical semantic content and final ZIP bytes. Free-context savings cannot be subtracted from the current 133,941-byte archive because the priced innovation stream is not in that archive. |

### Non-additive-pools law

1. G1 Lane 27,692 B competes with the 40,507-byte nested Lane home; it is not a new sidecar.
2. The 29,810-byte Movable worldsheet payload is already present as a 29,878-byte outer ZIP
   home in the v15 archive.
3. G4 contextual coding, xi-keyed deltas, Selfcomp, 2:4, and MX-int4 are alternative codings of
   a stream. Only the smallest exact receiver-preserving final ZIP home survives.
4. V17, v18b, and J3 can touch the same errors. Credit is the telescoping exact error count along
   one ordered archive chain, never the sum of independently measured deltas.
5. SegNet squeeze-excite makes local corrections frame-nonlocal: global-average pooling changes
   sigmoid channel gains and can rescale responses frame-wide. Same-frame proposals are therefore
   `COMPUTABLE_NOT_YET_COMPUTED` until composed and replayed together; joint replay is mandatory.
   Eval-BN, YUV6, R, and the rank-4 head
   are exact-linear stages; they do not make the upstream SE response additive.

## Waterfilled byte budget

The hard planning box is exactly 200,000 bytes; “KB” here does not mean 200 KiB. The score-byte
dual is

`lambda_B = 25 / 37,545,489 = 6.658589531221713e-7 score units per byte`.

For a measured receiver-closed stream-and-application curve `D_(i,a)(b)`, allocate until
`-dD_(i,a)/db <= lambda_B`, or until the stream/debt bound fires. Alternatives over the same
debt or the same correction at different application stages use the lower convex envelope. A
computable-not-yet-computed marginal earns no KKT allocation; the table's pending rows are
explicit engineering reservations for the next exact computation.

| byte home / reserve | bytes | status | accounting note |
|---|---:|---|---|
| v15 `predictor.zip` outer home | 100,099 | **MEASURED exact ZIP home** | Contains all inherited structure streams and the nested 3,721-byte xi/Pose6 stream. The pose line is contained, not additive. |
| G1 Movable worldsheet outer home | 29,878 | **MEASURED exact ZIP home** | Contains the measured 29,810-byte payload. |
| receiver realization profile | 85 | **MEASURED exact ZIP home** | Required camera/uint8 receiver contract. |
| solved-template outer home | 151 | **MEASURED exact ZIP home** | Six shared row-band templates; zero realized improvement at v15. |
| manifest | 3,345 | **MEASURED exact ZIP home** | v15 exact member home. |
| central directory and EOCD | 383 | **MEASURED exact ZIP home** | v15 exact container home. |
| **v15 exact control subtotal** | **133,941** | **MEASURED** | Exact sum of the six outer homes above. |
| Lane production seed | 270 | **MEASURED exact delta** | J2 receiver-owning seed; archive becomes 134,211 B. |
| contextual/bounded-collateral shared reserve | 25,789 | **DERIVED cap; COMPUTABLE_NOT_YET_COMPUTED spend** | Brings archive to Probe A's preregistered 160,000-byte ceiling. High-res pre-u8 and post-int8 correction records draw from this one reserve. Their separate bytes/realized-flip are computable on demand, not split by assumption. To consume the full Lane+Movable ceiling, the winning jointly replayed mix must average 28.1677 removed errors/B. |
| v18b first exact pricing rung reserve | 16,384 | **PREREGISTERED; COMPUTABLE_NOT_YET_COMPUTED** | First configured added-byte budget after a common exact-R master exists. |
| J3 xi/template/worldsheet finish reserve | 16,384 | **DERIVED; COMPUTABLE_NOT_YET_COMPUTED** | Payload growth allowance; J2's current 706-parameter lift itself is byte-identical, so the exact final growth remains to be computed. |
| final coder/container contingency | 7,232 | **DERIVED; COMPUTABLE_NOT_YET_COMPUTED** | May be reallocated only by exact marginal waterfill. |
| **hard total** | **200,000** | **DERIVED planning equality** | 65,789 B headroom over the seeded control. No feasibility claim. |

The earlier strict 154,524-byte stress box leaves only 20,313 B over the seeded control. It
cannot carry the full 25,789-byte v17 reservation without first proving at least 5,476 B of
receiver-preserving savings. G4's 89,161 B future-stream saving does not satisfy this condition.

### Omniscience-inversion computation map

The frozen evaluator space has no residual “unknown/estimate” epistemic category. Every pending
efficiency is **COMPUTABLE_NOT_YET_COMPUTED**; the only other statuses are **EXACT_COMPUTED** and
**INFEASIBILITY_CERTIFIED**.

| pending field | exact-computation surface |
|---|---|
| correction direction, support, and predicted flip | #391 exact adjoint in `src/tac/through_r/flip_inverse.py` |
| pre-u8 versus post-int8 lattice feasibility and exact joint Seg/Pose effect | #549 lineage via `tools/measure_realization_g2_lattice.py`, `tools/measure_joint_seg_pose_rate.py`, and the common receiver replay |
| full resize preimage/projector placement | #580 `tools/measure_resize_full_kernel.py` |
| exact archive bytes of each correction/coder alternative | `tools/measure_arith_selfcomp_rate_coders.py`, followed by the final deterministic ZIP composer |
| v18b generated-column rate/effect | `tools/probe_ddm_a1_column_generated_correction.py` after common-master closure |
| J3 full-n600 joint finish rate/effect | `tools/launch_ddm_joint_descent.py` after reviewed full-run mode and sealed ticket |

The current exact-computed component set is infeasibility-certified by the 3,103,689 versus
726,416 arithmetic. The wider multi-scale/v18b/J3 composition is computable-not-yet-computed,
not speculative.

## Debt-closure audit

### Exact arithmetic

| quantity | value | provenance |
|---|---:|---|
| sites | 117,964,800 | **MEASURED** v14 receipt |
| control errors | 3,240,528 | **MEASURED** v14 receipt |
| control `d_seg` | 0.027470296224 | **MEASURED** v14 receipt |
| continuous target allowance | 136,839.168 errors | **DERIVED** `sites * 0.00116` |
| maximum integer target errors | 136,839 | **DERIVED** floor of the allowance |
| integer errors that must be removed | **3,103,689** | **DERIVED** `3,240,528 - 136,839` |
| Lane control errors | 300,563 | **MEASURED** v14 receipt |
| Movable control errors | 425,853 | **MEASURED** v14 receipt |
| perfect Lane+Movable ceiling | 726,416 | **DERIVED** sum of measured role errors |
| role ceiling / continuous debt | 23.4049236% | **DERIVED** `726,416 / 3,103,688.832` |
| residual after perfect Lane+Movable repair | **2,377,273** | **DERIVED** `3,240,528 - 726,416 - 136,839` |

Road alone has 2,210,770 measured errors; Undrivable has 236,896; MyCar has 66,446. A
Lane+Movable-only candidate is therefore mathematically incapable of reaching the box, even at
perfect role repair. G3 independently shows broad pair debt (top-100 pairs carry only 18.70%),
which rules out a per-pair top-k closure and supports a shared all-role finish.

### Live-arm swing thresholds

| live arm | preregistered threshold on one exact artifact | consequence |
|---|---|---|
| **v17 successor: multi-scale epsilon-collateral contextual realization** | Current measured credit is **zero** because the reported v17 validity test has `rho<0` at all radii. The successor full-slice target is still **726,416** Lane+Movable errors at total archive `<=160,000 B`, Pose non-worse, exact joint objective negative. It must race high-res pre-u8 and post-int8 application and record exact bytes/realized-flip for each. The winning joint mix requires 28.1677 errors/B over the 25,789-byte reserve. | Until a successor row lands, the v18b/J3 residual is the full 3,103,689 errors. Any eventual successor credit transfers one-for-one, capped at 726,416. Its theoretical solo box threshold remains impossible under the role ceiling. |
| **v18b common-master column pricing** | After maximal v17 credit, its **solo** closing threshold is **2,377,273** additional errors at `<=40,000 B` shared residual headroom, Pose non-worse: 59.4318 errors/B if it consumes the whole residual reserve. Three complete pricing rounds and global exact replay are required. | Any smaller measured credit reduces the J3 threshold one-for-one; zero negative reduced-cost columns after all preregistered rounds closes only the tested column families. The current blocked receipt supplies zero credit. |
| **J3 full joint descent** | After maximal v17 and zero v18b credit, its **solo** closing threshold is **2,377,273** additional errors, final archive `<=200,000 B`, and official-YUV6 `d_pose <=0.00161`. More generally `E_j3 >= 3,103,689 - E_v17 - E_v18b`. | This is the decisive finish-rate measurement. Pair-447 descent establishes mechanism and resumability, not n600 efficacy. |

These thresholds are intentionally conditional. Assigning each arm an unconditional fractional
share would manufacture an efficiency measurement and violate the non-additive-pools law.

### Per-pair recursive solve-diff-repair contract

The executable unit is a pair recursion, not a single global correction pass:

`pair control -> solve -> exact-R diff -> G4 shared/local route -> repair -> exact-R replay -> ledger row`.

For pair `p`, derive its integer terminal threshold from its G3 atlas row and the current
waterfill allocation; there is no global per-pair constant. The exact thresholds are
`COMPUTABLE_NOT_YET_COMPUTED` until R2 prices the admitted curves and apportions the global
136,839-error allowance. Each recursion terminates in exactly one state:
`threshold-met`, `infeasible-certified`, or `budget-exhausted`.

G4 recurrence class decides whether a repair is shared or pair-local. A shared component is
installed and charged once under a stable `shared_component_id`; later pair rows reference it
with zero incremental shared payload bytes. Pair-local payload is charged to its owning pair.
Both routes still use sequential joint replay, so this byte ownership rule does not imply
additive SegNet effects.

Every R2/R3 run must preserve `<run_dir>/pair_convergence.jsonl`. Each append-only row binds
`pair_id`, G3 atlas receipt/hash, pre-state archive/runtime hashes, derived pair threshold,
waterfill dual and allocated bytes, candidate/application stage, G4 recurrence class and route,
shared-component id plus first-owner flag, exact incremental/final bytes, pre/post errors and
Pose, terminal state/reason, checkpoint id, and post-state hashes. R6 reads the complete ledger
and refuses missing pairs, nonterminal rows, duplicate shared charges, or a global terminal sum
above 136,839.

## Measurement chain to R6

No command below is authorized by this synthesis memo. R0-R4 are local/review gates; R5-R6
require the existing lane-claim, governed execution, custody, and operator authority.

| rung | gate and required receipt | existing tool surface |
|---|---|---|
| **R0 — spec/source closure** | Validate this JSON ledger, every cited SHA, lane id, `research_only=true`, and pointer immobility. Global lane validation's pre-existing 110 missing legacy paths must remain separately scoped. | `tools/lane_maturity.py`; Python's strict JSON loader for the ledger. |
| **R1 — deterministic receiver compile** | Re-emit the seeded v15/J2 control twice; require exact archive SHA, camera replay identity, all byte homes owned, and canonical parse/re-encode. | `tools/measure_ddm_v15_scorer_solved_templates.py --config ... --output-directory ...`; `tools/launch_ddm_joint_descent.py --ticket ... --out-dir ... --dry-run`. |
| **R2 — per-pair recursive solve chain** | For every pair, derive its threshold from the G3 atlas plus current waterfill, then iterate solve -> exact-R diff -> G4 shared/local route -> repair -> replay until `threshold-met`, `infeasible-certified`, or `budget-exhausted`. The multi-scale v17 successor, v18b common-master pricing, and J3 finish must preserve stage checkpoints and emit the append-only pair-convergence rows plus telescoping error/byte/pose deltas. Every correction names pre-u8 or post-int8 application; same-frame mixes are replayed jointly because SE makes them nonlocal. | Existing v16 control: `tools/measure_ddm_v16_coupled_joint_solve.py --config ... --output-directory ...`. Existing v18 surface: `tools/probe_ddm_a1_column_generated_correction.py --config ... --output-directory ...`. Existing J2 surface: `tools/launch_ddm_joint_descent.py`. The audit-proposed v17 runner, convergence-ledger emitter/validator, and J3 full-run mode are absent from this branch; R2 remains blocked until MAIN reviews those landed surfaces. |
| **R3 — full n600 advisory** | On the final exact bytes, require camera->uint8->R replay, global/per-role/per-pair errors, official YUV6 `d_pose`, batch custody, deterministic replay, exact archive bytes, score-unit value per byte, and a terminal convergence row for every pair. Verify shared-component ids are charged once. Authority remains `[macOS-CPU frozen-scorer advisory]`. | `tools/measure_ddm_v14_realization_fidelity.py --config ... --output-directory ...` and `tools/measure_ddm_v14_g4_receiver_projection.py --config ... --output-directory ...` are the existing n600 receiver/scorer references; the reviewed composed producer must use their common custody path. |
| **R4 — contest packet byte-close** | Attach a deterministic `inflate.sh` runtime, run it twice, require bit-identical full output, runtime-tree hash, archive SHA/size, no scorer weights or GT tables, <=1,800 s, and strict archive audit at 200,000 B. | `tools/audit_archive.py <archive> --strict --max-bytes 200000`. There is currently no DDM composed-runtime exporter in this branch; generic WITNESS byte-close tooling is not a valid substitute. This missing exporter is a named binding blocker. |
| **R5 — exact contest CPU** | Same R4 archive/runtime on Linux x86_64 CPU, durable work directory and JSON, 600 samples, exact component custody. No inference from macOS. | `experiments/contest_auth_eval.py --archive ... --inflate-sh ... --upstream-dir upstream --device cpu --work-dir ... --json-out ...`; then `scripts/adjudicate_contest_auth_eval.py` with its required custody inputs and `--required-device cpu --required-samples 600`. |
| **R6 — exact contest CUDA and paired adjudication** | Replay the identical archive SHA/runtime tree on contest CUDA. Require CPU/CUDA artifact identity, both component rows, strict adjudication, and the complete pair-convergence ledger: every pair terminal, shared bytes charged once, summed final pair errors `<=136,839`, and final hashes equal the adjudicated archive/runtime. | `experiments/contest_auth_eval.py` with the same verified flags and `--device cuda`; `scripts/adjudicate_contest_auth_eval.py --required-device cuda --required-samples 600`. The future composed-runtime landing must wire these tools to a strict pair-ledger validator; none exists on this branch. |

## Preregistered fork

1. **Composition feasible:** if one exact-R artifact satisfies
   `archive_bytes <= 200,000`, integer errors `<=136,839`, Pose6/xi is present, and
   official-YUV6 `d_pose <=0.00161`, hand this architecture and its exact hashes to the
   #366/J3 window. R4 must still close before R5/R6.
2. **Composition infeasible under measured efficiencies:** stop at the first binding slice:
   - current v17 is below target with `rho<0` at all measured radii -> move to the preregistered
     application-stage race: high-resolution FP pre-u8 dither/error-diffusion versus exact
     post-quantization int8 lattice, with joint same-frame SE replay;
   - v18b no common master or insufficient reduced-cost yield -> close the hybrid exact-R
     schema, then price generated all-role columns and coder entrants on equal bytes;
   - J3 insufficient all-role/pose descent -> reopen the FINISH parameterization, not a
     post-hoc pixel residual. The next rung is depth-stratified/object-local xi transport plus
     shared all-role worldsheet/curvelet columns, measured through exact R.
   - R4 missing -> build the deterministic DDM runtime exporter before any contest-axis eval.

## Constraint ledger

- No scorer weights, GT-argmax tables, decoded per-frame masks/RGB, or video-derived constants
  hidden in receiver code.
- Deterministic decode must finish within 30 minutes and be byte-identical across repeats.
- Every admitted video-derived degree of freedom is counted in the exact final ZIP.
- No mask-space/cell-space result becomes receiver authority without camera, uint8, R, and both
  frozen scorer legs.
- Stage-end and periodic checkpoints are distinct, atomic, preserved, and resumable.
- `v12` means only `FORMULATION:V12_FIXED_4096_ATOM_SEQUENTIAL_GREEDY_POSTSOLVE_POOL`.
- `v15` means only `INSTANCE:V15_1X1_THREE_BAND_ZERO_COLLATERAL_N64_REPLAY_N600`.
- `v16` means only `INSTANCE:V16_SINGLE_POINT_FULL_STEP_CONTINUOUS_LINEARIZATION`.
- Pointer remains `0.1910828242 [contest-CPU]` **UNMOVED**.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; full FEED-603 splay-v18b
window in `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`;
`.omx/research/ddm_a1_naive_verdict_audit_20260723_codex.md`; G1, G2, G3, G4, v12, v13,
v14, v14-G4, v15, v16, J2, and v18 receipts cited in the companion ledger; #601 planar and
#605 Screw6 receipts; #557 coder survey and #574 xi-delta artifacts; `.omx/state/lane_registry.json`;
`.omx/state/subagent_progress.jsonl`; per-arm and broadcast Codex inboxes. The
2026-07-23T04:07:14Z nonlinearity and 04:08:07Z multi-scale operator inbox supplements were
consumed after the first draft and are binding in the final non-additivity/application-stage
rules. The 04:11:17Z omniscience-inversion supplement supplies the final exact-computation
status taxonomy. The 04:16:28Z pair-recursion supplement supplies the pair-local threshold,
terminal-state, convergence-ledger, and shared-once routing contract.

## Triality

- **DSL/data leg:** `.omx/research/ddm_c1_composed_candidate_ledger_603_613_20260723.json`
- **DAG leg:** `.omx/research/ddm_c1_composed_candidate_spec_DAG_FEED_20260723.md`
- **equations leg:** `.omx/research/ddm_c1_composed_candidate_spec_canonical_equations_20260723.md`
- **lane id:** `lane_ddm_c1_composed_candidate_spec_603_613_20260723`
