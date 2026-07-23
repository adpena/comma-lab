# Codex findings — Quantizr re-mine and GEPA Optimize Anything Omni crosswalk

Date: 2026-07-23  
Lane: `lane_ddm_q1_quantizr_remine_gepa_omni_20260723`  
Delegation checkpoint:
`codex_delegate:ddm_q1_quantizr_remine_gepa_omni:20260723T173443Z`  
Authority: `research_only=true`; `$0`; `execution_allowed=false`; `actuation=NONE`  
Verdict: `ADOPT_FOUR_BOUNDED_FIRST_RUNGS_MAIN_LANDING_REVIEW_REQUIRED`  
Scope: public-source mechanism re-mine and apparatus crosswalk only; no training, eval,
provider dispatch, archive mutation, promotion, or score claim

## Outcome first

The prompt's “600 odd masks + warp” premise combines three different public mechanisms:

1. Quantizr PR #53 used both decoded masks and, according to the author, optical flow, but the
   exact model architecture was deliberately obfuscated. Its public runtime does not expose the
   flow direction, boundary rule, or separable flow cost.
2. Quantizr PR #55 stores only the second/odd mask and explicitly **drops** optical flow. It
   reconstructs frame 2 from the mask and reconstructs frame 1 by pose-FiLM conditioning.
3. Selfcomp PR #56 stores one grayscale field per pair and uses it for both frames, while separate
   per-frame six-parameter affine embeddings warp a shared oversized latent canvas.

Consequently, no exact causal `d_seg` cost of “odd-mask warp” can be derived from the public rows.
The whole-submission public rows are cross-architecture, cross-payload observations, not an A/B:
PR #53 reports `d_seg=0.00264182`, PR #55 `0.00061113`, and PR #56 `0.00115296`. These are quoted
source metadata, not Pact score claims. The correct verdict is
`NO_CAUSAL_WARP_COST_IDENTIFIABLE_FROM_PUBLIC_ROWS`; the family and the named boundary-policy
first rung remain open.

Four useful, bounded first rungs survive the correction:

1. test PR #56's oversized-canvas plus border-extension policy inside the existing #601/#605
   advection formulation;
2. make pose bytes causally active in the E1 successor through a typed, mutation-tested receiver
   edge, borrowing PR #55's late residual-FiLM separation as a design pattern;
3. test the PR #56 soft class-coordinate LUT as a proposal/paint coordinate, not as an RGB
   prototype table;
4. expose the existing Pact proposal engines through one custody-bearing adapter and run a
   historical `$0` matched-budget replay before considering GEPA Omni-style engine switching.

No public code was copied or incorporated by this arm.

## Epistemic and authority labels

- `MEASURED_PUBLIC`: literal public source or report field, pinned below.
- `MEASURED_LOCAL`: existing Pact receipt or executable source, pinned below.
- `DERIVED`: consequence re-derived from pinned mechanisms.
- `PROPOSED_FIRST_RUNG`: a measurement plan, not a result or launch authorization.
- Every negative below carries an explicit `verdict_scope`; no instance/formulation negative
  closes a family.

## Source custody

### Public Quantizr/selfcomp sources

| Source | Pin | Content custody | What it can establish |
|---|---|---|---|
| [PR #53 mask2mask](https://github.com/commaai/comma_video_compression_challenge/pull/53) | head `ef78345815e5a31548dea65f30cf2942b36d0ddf` | `submissions/mask2mask/inflate.py`: Git blob `637d13b15810bcb6114eb86a0ae31f365df2cb4c`, SHA-256 `df55103115e8facb2a38012467c5ff3d06cac607ccb9f1576389f0c63575f1d8` | Runtime consumes mask 1 and mask 2; author says successor #55 dropped optical flow. Exact flow implementation remains obfuscated. |
| [PR #55 quantizr](https://github.com/commaai/comma_video_compression_challenge/pull/55) | head `e0b643b0a7c21f62cc93b5d920bcf3fc0d5a33d9` | `compress.py` blob `689906eb8e835576df745ec4f940e142a35c2dd8`, SHA-256 `39dc3fe24e639cc9048b4950999193f81d54b2612d9b26852e9afc844dc3676b`; `inflate.py` blob `a123b9e534ec08252cbd6d52cf662152c4e9029f`, SHA-256 `b2777611a9870e3eac8eb5fa8b1d727a950e582ea97ba6067976f0dc74e5edeb` | Exact odd-mask extraction, AV1 flags, stored pose, residual-FiLM receiver, KL formula, and submitted combined curriculum source. |
| [PR #56 selfcomp](https://github.com/commaai/comma_video_compression_challenge/pull/56) | head `2b2d76de6f5aa34c76352c2cc02b03ed44a03a26` | `submissions/selfcomp/inflate.py`: Git blob `a815404e8b59b5c2fa7769c0275cbda77398bd56`, SHA-256 `e4255f980c07b3ea5796e701ba98e7aa66479cf2e515c763688bdaea2216676e` | Exact soft-class LUT and per-frame affine latent-canvas reconstruction. Compression/training code was not submitted. |

The repository mirror `reverse_engineering/quantizr_pr55/{compress.py,inflate.py}` has the same
Git blobs as PR #55. Its `ANATOMY.md` is an older interpretation, not source authority.

### Current Pact surfaces used for comparison

| Surface | SHA-256 | Authority used here |
|---|---|---|
| `.omx/research/advected_motion_base_20260721.json` (#601) | `f96d870301df9d81326f42db98f3e76689e7a707c19df5a37fabf8a342b1dea2` | `MEASURED_LOCAL`; n64 planar formulation negative, family open |
| `.omx/research/advected_screw6_chartlevel_20260721.json` (#605) | `98ccf6af4a3b482c71e2654353180aac32b99ad85be8647c36046f4591cd4f2a` | `MEASURED_LOCAL`; n16 one-depth chart formulation negative, family open |
| `.omx/research/ddm_e1_runtime_exporter_n600_20260723/ddm_e1_runtime_export_receipt.json` | `792b46a34746132f257c9835607de57628c63f5c69f0380a9e44a4c20b8d06b9` | `MEASURED_LOCAL`; exact archive member inventory and byte homes |
| `.omx/research/ddm_e1_runtime_exporter_n600_20260723/packet/inflate.py` | `453a9b5b6aaf133662f57e63105c3828d55ab7ef206e1722f8dc6e7c1a36b4e3` | `MEASURED_LOCAL`; receiver semantics |
| `.omx/research/ddm_dv2_sdwl1_n600_20260723/receipt.json` | `efc43fcda1f12f28df2b6059cd5e51e7ee2509a356d99b59e317b253927a709c` | `MEASURED_LOCAL`; exact declared fact-language payload only |
| `.omx/research/configs/ddm_j5_366_realized_acceptance_warmstart_20260723.json` | `9aa12699eb53e77351fd0b96eed16d527eaf4a443b177f07512edd5ed6f3d88d` | `MEASURED_LOCAL`; current typed #366 program contract |
| `src/tac/ddm_costate_organ.py` | `3dd8853e4be0dfd12dd44721cb2ef38320e0b0b682f9dbf99aae731163290c54` | current DDM scheduler, duties, custody, and resume surface |
| `src/tac/witness_control/regime_dispatch.py` | `1358a4c921a92067766b275f69bb56b85e9b760c779f2d7178625a06a6dc6fb1` | current past-only regime dispatcher and walk-forward gate |
| `src/tac/witness_control/gepa_reflection.py` | `9e8f86ec21eba9348495c4db02972c64f85af9d9ff723410f6889f1e2d342cad` | existing GEPA-style reflect/measure/Pareto/dispose cycle |
| `src/tac/witness_dsl/duty_queue_fire_tickets_20260719.py` | `65fac7fda0f9a42f13d4318707242f43164f797ce4bc5c91e1c72d2bf1bccc87` | fail-closed duty-ticket materializer |
| `tools/subagent_checkpoint.py` | `f32a5de476d30dadb4c539c11e24c29e3fc2970573483c73fc1428a47a2ab26f` | named-crux respawn and recovery context |
| `src/tac/probe_outcomes_ledger.py` | `d07b142f420b1ff35d9f140200e17cb2715ee264830f06f727a74803c8ea7111` | typed empirical outcome consumer |

## A. Quantizr re-mine ranked against today's surfaces

| Rank | Lesson | Their exact mechanism | Our named surface | Disposition and verdict scope | `$0` measurement plan | Borrowed accounting if code |
|---:|---|---|---|---|---|---|
| 1 | Pair-shared field plus explicit frame transform can separate stable semantics from motion. | **PR #56**, not #55: one decoded gray field becomes a five-channel soft-class field and is repeated for both frames. Frame indices `2i,2i+1` select separate bounded affine embeddings that warp a shared `1.25x` latent canvas with bilinear sampling, `padding_mode="border"`, and `align_corners=False`. It does **not** warp frame 1 from frame 2. | #601 planar advection and #605 full-screw chart formulation; successor depth-stratified advection | `ADOPT_BOUNDARY_POLICY_FIRST_RUNG`. Scope: boundary/placement policy only; it does not overturn either formulation negative or supply SE(3). | On the same already-named n16 source and exact receiver, A/B the existing warp boundary policy versus `1.25x` canvas + border extension. Keep transform direction explicit as target-frame inverse sampling; hold bytes, source, scorer batch, and chart coefficients fixed. Report exact `d_seg`, `d_pose`, exception bytes, and boundary-band deltas. | Current arm: `0` LOC, `0` bytes. Future: clean-room re-expression of the three public equations; if any PR source is copied, record exact lines and archive bytes before admission. |
| 2 | Stored pose must be a causal receiver input, and late residual conditioning limits semantic spill. | PR #55 stores a six-float pose row per pair. `pose_mlp: 6→48→48` drives gamma/beta in only frame 1's residual branch: `r=Norm(Conv(Conv(x)))`; `r'=r*(1+gamma)+beta`; output `SiLU(x+r')`. Frame 2 has a static head. The same odd/second mask feeds the shared trunk. | E1 successor pose stream and receiver; #417 counted-but-inert/both-directions doctrine | `ADOPT_CAUSAL_EDGE_PATTERN`; scope is receiver mechanics, not Quantizr weights or its distortion row. E1's exact archive has only `manifest.json`, `base/chart.ddb`, and `semantic/composed.dds`; no pose member or pose variable reaches `_render_batch`. Byte-home bijection is not causal pose consumption. | Add a typed pose member and explicit `pose6 → bounded frame-0 receiver coefficient` edge in the successor. Strict test: mutate one pose record, require scoped output bytes to change; restore it, require exact hash restoration; then measure exact joint receiver deltas and bytes. This arm does not implement or fire it. | Current arm: `0` LOC, `0` bytes. Prefer a clean-room residual-conditioning law. Any copied layer/source later must be itemized. |
| 3 | A scalar gray coordinate can be a corruption-tolerant soft semantic address. | PR #56 fixes targets `[0,255,64,192,128]`, `sigma=15`, computes `g_c(x)=exp(-(x-t_c)^2/(2 sigma^2))`, then applies a softmax over the five `g_c`. This is an **input class-membership LUT**, not an RGB output prototype palette. No dithering is present. Placement comes from the separate affine latent. | DDM paint/prototype-color proposal surface and E2 perturb→realize→score API | `ADOPT_AS_PROPOSAL_ONLY`; verdict scope: LUT coordinate/proposal formulation. It is not an admissible final color solve until the learned receiver, uint8/R chain, and scorer survival are measured. | On a fixed existing semantic field, compare hard class IDs versus the exact 256×5 LUT as proposal coordinates feeding the same receiver. Sweep only class-target ordering and sigma through a preregistered finite bank; exact admission remains joint receiver delta plus bytes. Add ordered-pair Fisher/margin audit; do not use visual smoothness. | Current arm: `0` LOC, `0` bytes. The equation and constants are public-source facts; implementation should be clean-room and cite PR #56. |
| 4 | Long-GOP screen-content choices deserve a semantic recurrence analogue, not a literal codec transplant. | PR #55 emits 600 gray `512x384` masks at 10 fps through libaom AV1: CRF 50, `cpu-used=0`, row-MT, GOP/keyint 1200, lag 48, ARNR 0, AQ 0, CDEF off, intra-block-copy on, OBMC off; OBU is then Brotli-11 compressed. | SDWL1/dv2 exact fact-language coder | `ALREADY_BETTER_OR_N_A_BY_TOOL`. Scope: exact semantic payload, not pixels. SDWL1 already uses whole-sequence causal state plus an inner arithmetic coder and outer zlib. CDEF/OBMC/AQ/CRF are lossy pixel tools and are N/A. The unresolved transferable primitive is IBC-like exact decoded-state copy. | Recode the frozen SDWL1 inventory with one exact copy-from-prior-subvector token; parse-back must remain exact. Compare complete payload bytes to the pinned selected SDWL1 receipt. Stop if no byte win; do not import a video codec or alter semantics. | Current arm: `0` LOC, `0` bytes. A future exact-copy token is an original SDWL1 extension; no Quantizr codec code is needed. |
| 5 | Staged semantic anchoring, pose fitting, then joint repair can reduce interference; soft-logit KL may improve proposal ranking. | PR #55 source defines five sequential runs across three freeze modes: anchor `400 @ 5e-4` with QAT at 200; anchor-boost `80 @ 1e-5`; finetune `320 @ 5e-5` with QAT at 120; joint `160 @ 1e-5`; micro-finetune `120 @ 5e-6`. `KL(logits,T=2)=T^2 KL(softmax(z_teacher/T) || softmax(z_student/T))`. The PR author states the combined script was LLM-composed to mimic several scripts and was not run end-to-end. | Typed #366 three-stage realized-acceptance descent | `OUR_CUSTODY_BETTER; ADOPT_KL_PROPOSAL_AB_ONLY`. #366 has 368 receiver DOF, exact Q8 proposals, exact n600 joint admission, EMA/checkpoints, group gates, rollback, and three preserved stage boundaries; it is not comparable to an 88K-parameter neural model. The public five-run schedule is not a calibration receipt. | Offline on already-produced proposal logits only, compare current proposal rank with `CE + T=2 KL`; evaluate both rankings using the same stored exact realized verdicts. A KL win changes proposal ordering only. It may never replace exact argmax/R/joint admission or authorize the run. | Current arm: `0` LOC, `0` bytes. Formula can be re-derived in clean-room code if MAIN later routes it. |
| 6 | Small depthwise-separable networks and FP4 codebooks can reduce a neural receiver's parameter payload. | PR #55 reports 88K parameters and about 64 KB compressed, with depthwise-separable convolutions and a nibble codebook. | #366 receiver-effective surface and analytic DDM grammar | `N_A_CURRENT_SUBSTRATE`. #366 exposes 368 analytic receiver DOF; neural parameter count and FP4 payload are not a like-for-like optimization target. | None now. Reopen only if an admitted DDM receiver introduces a counted neural weight tensor; then measure exact archive bytes and receiver deltas against the analytic control. | Current and planned: `0` external LOC/bytes. |
| 7 | Optical-flow conditioning was tried before pose-FiLM, but its public source cannot support exact transfer. | PR #53 runtime loads 1,200 masks and calls an obfuscated generator with mask 1 and mask 2. PR #55's author says it dropped optical flow. No public #53 compression source or legible flow receiver exposes direction, occlusion, padding, or cost. | #601/#605 advection lineage | `N_A_EXACT_MECHANISM`; verdict scope: public-source recoverability only. The advection family remains open under its already-named depth-stratified reformulation. | Do not reverse-attribute public whole-row differences to flow. The rank-1 #56 boundary A/B is the source-resolvable alternative. | `0` LOC/bytes; no de-obfuscation or code adoption. |

### Direction and boundary answer, explicitly

For PR #56, PyTorch `grid_sample` uses the affine grid as an output-to-source sampling map. Each
target frame index selects its own affine parameters; therefore the implementable direction is
“sample a shared oversized latent into target frame `2i` or `2i+1`,” not “forward-warp frame 2
into frame 1.” The oversized `1.25x` canvas plus `padding_mode="border"` extends boundary values
and avoids zero-fill seams. That boundary rule is the exact transferable first rung.

PR #53 cannot answer the same question from public source. PR #55 has no warp.

### E1 pose violation, exactly

E1's archive receipt names three members and no pose stream. Its runtime reconstructs two RGB
chart frames, then overlays the same composed semantic labels/palette on both. Pose may have
affected an upstream materialization, but the shipped render has no typed pose variable and no
receiver-use edge. Thus:

`pose_bytes_absent ∨ (pose_bytes_present ∧ render_independent_of_pose) ⇒ causal_pose_contract=false`.

The successor must prove both directions:

1. a counted pose byte changes an authorized receiver surface; and
2. no invisible or unused pose byte remains counted.

## B. GEPA Optimize Anything Omni crosswalk

Primary source:
[Optimize Anything with GEPA](https://gepa-ai.github.io/gepa/blog/2026/07/22/optimize-anything-omni/),
published 2026-07-22. The HTML retrieved on 2026-07-23 had SHA-256
`1757b6fafeef43e32640d800a2ed1a82279ca14697f78d17e9df1528074339d7`.

### What the new Omni surface actually adds

The post exposes a common `candidate → (score, info)` evaluator contract and interchangeable
engines:

- `gepa`: Pareto-selected parent plus one reflective LLM mutation;
- `autoresearch`: a longer-lived coding agent owns proposal, selection, and evaluation;
- `meta_harness`: the outer framework owns selection while a coding agent proposes;
- pipeline helpers for best-of, parallel, vote, sequential, and adaptive-sequential composition.

Its Omni recipe first spends a small matched budget across engines, selects the best result, then
seeds a fresh different optimizer from that winner. Its controlled Terrarium comparison keeps
task, model, evaluator server, and budget fixed and reports that no engine dominates every task.
Those findings motivate a Pact measurement; they do not transfer as evidence about this contest.

### Ranked crosswalk

| Rank | Omni lesson | Pact state | Disposition | Named consumer | `$0` first rung and guard |
|---:|---|---|---|---|---|
| 1 | One candidate/evaluator protocol makes genuinely different proposal engines composable. | `gepa_reflection.py`, `regime_dispatch.py`, the DDM costate organ, and duty tickets each use different candidate/report shapes. | `ADOPT_THIN_TYPED_ADAPTER`; do not replace the organs. | `tac.ddm_costate_organ`, `tac.witness_control.gepa_reflection`, `tac.witness_control.regime_dispatch` | Define a research-only envelope over historical artifacts: candidate ID/SHA, parent SHA, engine, source hashes, axis, exact cost, verdict scope, first rung, resume state, and authority. Adapter has `actuation=NONE`; no new optimizer call. |
| 2 | A fresh optimizer family can break a plateau that the incumbent keeps revisiting. | Pact respawns arms at named cruxes and records `respawn_context`, but the proposer family is manually chosen and not a typed plateau action. | `ADOPT_CONDITIONALLY`. Costate organ remains the arbiter; freshness is not authority. | `regime_dispatch.py` plateau classification, `ddm_costate_organ.py` duty order, `tools/subagent_checkpoint.py` recovery manifest | Replay historical completed proposal sequences. At a named plateau/crux, compare same-engine continuation with a different historical engine family under equal recorded proposal count. Require child lineage and checkpoint preservation. |
| 3 | Matched task/model/evaluator/budget comparison is needed because no optimizer dominates. | Pact has stronger receipts per experiment, but no single matched-budget comparison of reflective GEPA, regime dispatch, costate ranking, and named-crux respawn. | `ADOPT_MEASUREMENT`; current optimizer-superiority claims remain `NO_VERDICT_DATA`. | `tac.probe_outcomes_ledger.register_probe_outcome` | Historical `$0` replay only: fixed candidate pool, fixed authority filter, fixed proposal/evaluation count, fixed source hashes. Report selection regret and authoritative-candidate yield, not transferred blog scores. |
| 4 | Auxiliary state/info should carry execution findings to the next proposal. | Pact has richer findings memos, observatory rows, source hashes, and first-rung columns, but no uniform ASI schema across engines. | `ADOPT_TYPED_ASI_SUBSET`. | `gepa_reflection.py`, findings memo schema, duty-ticket provenance | Map existing fields only: exact failure classification, deciding signal, source hashes, verdict scope, unresolved first rung, and authority. Free-form prose cannot alter the score or bypass a gate. |
| 5 | Best-of/parallel/vote/sequential helpers make ensemble policy explicit. | The costate organ already schedules dependency frontier, freeing-before-spending, coarse-to-fine, and Gauss-Southwell validity; duty tickets fail closed. | `PARTIAL_ADOPT`. Best-of/parallel can be research-only scheduling policies; vote is not authority amplification. | DDM scheduler and duty queue | Dry-compile policies over historical candidate receipts. Every branch shares the same authority filter; voting cannot turn advisory evidence into contest evidence. |
| 6 | Reflective evolution plus Pareto disposal is useful in low-sample regimes. | Already implemented in `gepa_reflection.py`: grounded reflections, walk-forward measurement, complexity Pareto frontier, and adopt-only-if-better disposal. | `ALREADY_BETTER_CUSTODY; NOT_OMNI_COMPLETE`. | Existing GEPA cycle | Preserve it. Add an engine adapter only after the matched historical replay. Do not infer that the existing n≈1 result generalizes to DDM. |
| 7 | Optimizer output can be selected by scalar score plus auxiliary info. | Pact requires exact archive/runtime/source custody, axis separation, verdict scope, staleness hashes, resumability, and first-rung lineage. Generic Omni does not itself require these. | `PACT_STRICTLY_BETTER_AUTHORITY_FIREWALL`. | Candidate adapter admission gate | Refuse candidates missing any custody field. The evaluator's scalar is downstream of authority filtering and can never promote on its own. |

### Adversarial review in both directions

**What Omni correctly challenges in Pact**

1. Multiple named organs are not evidence that switching among them improves candidate selection.
   No matched-task, matched-evaluator, matched-budget receipt compares them.
2. Manual named-crux respawn can silently reuse the same proposal habits under a new arm name.
   Engine family and parent lineage need typed fields.
3. Fragmented candidate/report shapes make historical replay and portable parent seeding harder
   than Omni's common engine boundary.
4. “Already richer apparatus” is not a performance verdict. Until the replay exists, the honest
   status is `NO_VERDICT_DATA_ON_ENGINE_SUPERIORITY`.

**What Pact correctly challenges in generic Omni**

1. A scalar evaluator can compare stale, advisory, contest-CPU, contest-CUDA, or incomplete-byte
   evidence unless authority filtering precedes selection.
2. Auxiliary state is not custody unless it binds source hashes, exact artifact identity, axis,
   verdict scope, and validity horizon.
3. A fresh optimizer restart may discard resume state, parent provenance, or named-crux meaning.
4. Equal model-call dollars are not necessarily equal GPU cost, evaluator cost, operator attention,
   or irreversible risk.
5. Voting repeats an evaluator; it does not make that evaluator authoritative.
6. A failed formulation must not become a family-level negative merely because an engine marks it
   low-scoring.

### Safe synthesis

Use Omni as a **proposal-engine adapter**, not as a replacement controller:

`custodied sources → authority filter → costate/regime engine allocation →`
`duty-queue schedule → engine proposal → exact evaluator → typed ASI/findings →`
`probe-outcome ledger → named-crux respawn`.

The phase-1 engine comparison is a `$0` historical replay. A phase-2 fresh-engine call remains
unauthorized until MAIN reviews a typed adapter, a bounded budget, checkpoint preservation, and
the exact evaluator/axis contract.

## Triality and no-orphan routing

### Equations leg

1. Soft semantic coordinate:
   `p(c|x)=softmax_c(exp(-(x-t_c)^2/(2 sigma^2)))`.
2. Per-frame latent sampling:
   `L_i(u)=grid_sample(L_shared, A_i u; border, bilinear)`.
3. Residual pose conditioning:
   `h_1=SiLU(h + (R(h)*(1+gamma(p))+beta(p)))`; frame 2 omits the pose term.
4. Proposal-only logit matching:
   `L_KL(T)=T^2 KL(softmax(z_teacher/T) || softmax(z_student/T))`, `T=2`.
5. Custody-bearing optimizer selection:
   `argmin_e loss(e)` is defined only over candidates passing the common authority predicate.

### DSL leg

This arm specifies the minimum future adapter fields:

`candidate_sha`, `parent_sha`, `engine_id`, `engine_family`, `source_hashes`,
`axis_tag`, `score_claim`, `verdict_scope`, `validity_horizon`, `first_rung`,
`proposal_budget`, `evaluation_cost`, `resume_state`, `actuation`.

No new launcher flag, runtime schema, or executable config was added.

### DAG/feed leg

Named consumers:

1. PR #56 boundary first rung → #601/#605 depth-stratified advection successor.
2. PR #55 residual-FiLM causal edge → E1 successor/E2 pose-stream receiver.
3. PR #56 LUT proposal → E2 perturb→realize→score API and paint/prototype solver.
4. PR #55 KL proposal ranking → #366 proposal generator only.
5. AV1 IBC analogue → SDWL1 exact-copy-token probe.
6. Omni adapter → costate organ + regime dispatch + existing GEPA reflection.
7. Matched replay outcome → `tac.probe_outcomes_ledger`.
8. Fresh-engine named-crux lineage → duty queue + `subagent_checkpoint` recovery manifest.

The first-rung plans are routed here as durable findings, not executed. Any implementation must be
separately claimed and registered.

## Borrowed-substrate accounting

| Item | External code copied | External artifact bytes incorporated | Status |
|---|---:|---:|---|
| Quantizr PR #53 | 0 LOC | 0 B | read-only inspection |
| Quantizr PR #55 | 0 LOC | 0 B | repository mirror read-only inspection |
| Selfcomp PR #56 | 0 LOC | 0 B | read-only inspection over pinned public blob |
| GEPA Optimize Anything Omni | 0 LOC | 0 B | blog/API concept crosswalk only |
| Total | **0 LOC** | **0 B** | no borrowed substrate landed |

Technique, math, and geometry transfer is explicitly authorized. This arm does not adopt public
weights, archives, runtime code, codec payloads, or training code. Any later code adoption must
open a new itemized accounting row before merge.

## Stores consulted

- delegated authority file, verified at 6,129 bytes and SHA-256
  `89843ead322c7694a902590821619d9ee58df7c6ba825f5e65959f1f21853086`;
- `CLAUDE.md`, `AGENTS.md`, lane registry, subagent progress, per-arm inbox, and broadcast inbox;
- the pinned public PRs/blobs and the GEPA Omni post above;
- #601, #605, E1, SDWL1, #366, live DDM costate-organ, regime-dispatch, GEPA-reflection,
  duty-queue, checkpoint, and probe-ledger surfaces pinned above;
- latest DDM FEED entries for findings-first-rungs and the queued #582e/#582f directives.

No banned HNeRV/PR95/101/110/128 substrate source, archive, or payload was consumed. No live run,
provider, GPU, archive, or frontier pointer was touched.

## Honest blockers and landing boundary

1. Exact PR #53 flow geometry is `BLOCKED_PUBLIC_ARCHITECTURE_OBFUSCATED`; do not guess.
2. Warp-only `d_seg` is `NO_CAUSAL_AB_DATA`; public whole-submission rows are confounded.
3. Omni engine superiority in Pact is `NO_VERDICT_DATA`; the historical matched replay is owed.
4. Repository-wide lane validation still reports 110 pre-existing missing-evidence-path errors in
   other lanes. This new research lane adds no evidence reference and does not bypass the debt.
5. MAIN must review the base-to-branch diff and explicitly decide whether to merge. This arm does
   not merge itself, launch a first rung, or move the pointer.
