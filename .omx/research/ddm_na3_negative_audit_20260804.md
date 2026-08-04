---
schema: ddm_na3_negative_audit.v1
date_utc: 2026-08-04
arm: ddm_na3
lane_id: "lane_ddm_na3_negative_audit_20260804"
research_only: true
score_claim: false
promotion_eligible: false
axis: "[macOS-CPU advisory; scorer-free audit; zero scorer forwards]"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7541459 @ 358,084 B [macOS-CPU advisory]"
verdict_scope: "AUDIT over named 2026-07-31..2026-08-04 receipts plus task-ledger rows explicitly searched below"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_na3 - negative-findings scope audit

## Answer first

**#918 vs rt1:** not a contradiction. The valid part of "#918 RATE CODING IS CLOSED" is the
token entropy/coder stream closure on the explicit lattice. It does **not** close rt1/tz1's
lossy rate maps and adaptive quantization rungs, which have fresh measured byte receipts
(-24,605 B, -62,502 B, -106,099 B, -113,555 B) but still need the scorer-side d_seg/d_pose
verdict. Disposition: **RE-SCOPED**, not retracted. Token-coder transforms remain shut; lossy
map/rung candidates remain live and are already in `scorer_batch_20260804.md`.

**Cap-artifact census:** `cw1` found **321 cap-default sites**, **317 silent cap-default sites**,
over `experiments`, `tools`, and `src/tac` with tests excluded. Among load-bearing recent
negative/capped verdicts I found **four memo-level cap-censored or cap-retyped families**:
`et1/sq1` solved-paint eta floors, `sm1/#935` sq1 realizer terminal-iterate cap, `lr2` solved-paint
static-key ladder, and `os1/ss1` D2/rung-B stop-reason regrade. Only `#935` currently needs the
single n600 scorer-batch append from this arm; `lr2` first needs the already-named fresh-pair /
byte-close gate.

**Top 3 reopenings surviving this audit:**

1. `rt1/tz1` adaptive rate map: real byte wins, scorer-gated and already queued.
2. `#935` sq1 realizer uncap: 31/32 headline winners ended at the 25-step terminal iterate;
   appended to the scorer batch as a convergence-tested measurement spec.
3. `lr2` fixed-rule waterfill / feature-bearing solved-paint successor: not banked at n32, but the
   n8 cap artifact was real and the fresh-pair gate remains the right cheap falsifier before any
   byte-close claim.

No score moved. I ran no scorer, built no archive, and did not touch the contest pointer.

## Scope and stores consulted

Primary named receipts: `ddm_bo1_seg_base_objective_menu_order_20260803.md`,
`ddm_rt1_rate_axis_charter_receipt_20260804.md`, `ddm_tz1_token_sweep_rate_attack_20260804.md`,
`ddm_bz1_phase_field_byteclose_20260804.md`, `ddm_lr2_legal_realization_ladder_20260804.md`,
`ddm_et1_eta_on_the_priced_band_20260803.md`, `ddm_sq1_eta_seg_and_hinge_ab_20260803.md`,
`ddm_sm1_seg_search_transfer_20260803.md`, `ddm_ss1_selection_vs_search_20260803.md`,
`ddm_os1_optimization_sweep_termination_census_20260802.md`,
`ddm_fz4_20260804/fz4_map_repair_verdict.md`,
`ddm_qo1_repair_stream_optimal_form_20260804.md`, `ddm_fz1_frame0_repair_real_base_20260804.md`,
`ddm_ub1_untagged_verdict_scope_audit_20260801.md`,
`ddm_qj1_followon_backlog_join_20260804.md`, `ddm_fo1_orphaned_followon_detector_20260801.md`,
`ddm_p1a_followon_unknown_adjudication_20260801.md`,
`ddm_p2a_task_backlog_drain_20260801.md`, `ddm_hv2_two_week_harvest_20260803.md`,
`ddm_iv1_inventory_drain_20260803.md`, `gc16_full_stack_convocation_20260804.md`,
`ddm_cw1_silent_cap_defaults_baseline_20260804.json`, `.omx/state/main_hot_state.md`, and
`.omx/state/canonical_task_status.jsonl`.

Search discipline: bounded grep over `.omx/research`, `.omx/state`, `.omx/tmp/codex_runs`,
`docs`, `src`, and `tools` for the charter's named ids and phrases. Absence claims below are
scoped to those searched paths and exact phrases.

## Seed audit

| seed | audit result | disposition |
|---|---|---|
| `#918` "RATE CODING IS CLOSED" vs `rt1` | `bo1` confirms closure only for the explicit token coder axis. `rt1/tz1` measure different objects: lossy quantization/rung maps with byte wins and scorer risk bounds. | **RE-SCOPED.** Do not cite #918 against rt1/tz1. Keep token entropy/coder transforms closed. |
| `bz1` phase-field kill vs `lr2` ladder | `bz1` was mechanism-scoped: deterministic RGB block translation of a label-solved offset field did not byte-close. `lr2` did run and measured five legal realizers plus descendants. Offset-field carriers still lose; feature-bearing solved paint remains a gated successor, not a bz1 contradiction. | **FOLLOW-ON FIRED.** `lr2` narrows the family and leaves a fresh-pair/waterfill gate, not a generic reopen. |
| `et1/#932`, `#935`, `#938b` cap class | `et1` states every eta from the inherited paint solve is a floor; step budget raises eta and d_pose together. `sm1` source-verifies `sq1_stage_n32.json`: truth start won 0/32, while 31/32 winners are `dec@25`, the terminal 25-step iterate. `gc16` makes #935 a scorer-batch row. | **CENSORED, NOT WALL.** Append #935 convergence-tested measurement to scorer batch. |
| `fz4` map kill | `fz4` kills the tz1 `[16,12,8,4]` margin-coupled map plus current selective F0PR1 on the pu2 vehicle: seg leg alone fails before pose. The memo explicitly leaves gentler/pose-aware maps open. `qo1` is the folded home. | **FORMULATION ONLY.** Pose-aware/gentle maps are routed, not orphaned. |
| `#930` multi-start won 0/32 | Exact row found in `sm1` and canonical task status. The "truth" start won 0/32, so the transfer claim is implementation-refuted. But this same receipt exposes terminal-iterate cap censoring, 31/32 at `dec@25`. | **TRANSFER REFUTED; CAP REOPENED.** Treat as #935, not as multi-start family death. |
| `#846` two family over-scopes | `ub1` found two citation-transit failures: UB1-A turned a pose/witness/no-inverse formulation into a law against seg/TR1; UB1-B turned exact reversible L3 raster residual storage into "plane-storage family" death. | **RE-SCOPE REQUIRED.** No new experiment for the dead exact object; correct the noun and registry/roadmap citations. |
| `#879/#880/#886/#887` list heads | `qj1` joined the backlog: 47/47 ranked heads parsed, 45 queued with fire-order, 2 folded; 437 queued rows have owners, 0 unowned. `hv2/iv1` consumed #886/#887 by content and fired the highest-value heads. | **OWNED, NOT ORPHANED.** Continue from qj1/hv2/iv1 fire-orders; do not duplicate bare ids. |

## #918 vs rt1 detail

The clean adjudication is by object identity:

| object | evidence | verdict |
|---|---|---|
| Explicit token entropy/coder streams | `bo1` #918: Brotli q11 token lattice closure; tw1/gk2 corrections are price-key corrections, not a new coder family. `rt1/tz1` also find LZMA-filter and tight 3-bit packing losing on live streams. | **CLOSED within scope.** |
| Lossy token quantization/rung maps | `rt1/tz1`: L=14 saves 24,605 B; L=8 saves 106,099 B; adaptive `[16,12,8,4]` saves 113,555 B; derived-activity fallback saves 62,502 B. Every row is scorer-gated by pre-registered d_seg bounds. | **OPEN, scorer-gated.** |
| Endpoint/range refit | `rt1`: endpoint mass 33.2960%; range-refit blocked on continuous tokens/retrain. | **BLOCKED, not closed by #918.** |

Therefore the correct wording is: **token coder search is closed; rate-axis lossy maps are open
until the scorer batch proves whether their d_seg/d_pose damage stays under the byte win.**

## Cap-artifact census

`cw1` is the broad static denominator: **321 cap-default sites** and **317 silent cap-default sites**.
That is an apparatus census, not a claim that all 317 are live negative walls.

Load-bearing recent findings:

| finding | cap status | audit disposition |
|---|---|---|
| `et1/sq1` paint solve eta | 100% cap-pinned in `et1`; 10/25/50-step ladder still rising while d_pose also rises. | eta values are **floors**; budget is a joint seg/pose tradeoff, not a free gain. |
| `sm1/#935` sq1 realizer | 31/32 headline winners at the terminal 25-step iterate. | **append to scorer batch**: uncap and run to convergence criterion. |
| `lr2` solved-paint ladder | initial 30-step floors; uncapping moves n8 M32_U over zero, then n32 reverses to losing. | cap artifact real but no n32 bank; fresh-pair/waterfill gate before any score row. |
| `os1/ss1` D2 and rung-B | `os1` overread bound stops; `ss1` corrects D2 relin cap to 0/600 binding and separates placement vs more-search for rung-B. | retyped stop reasons; not a D2 cap wall. |

The cap lesson is not "raise every cap." It is: no negative becomes a wall until the stop reason is
typed, a convergence criterion exists, and the scorer/rate surface is remeasured after the budget
change.

## Backlog head adjudication

The live join receipts matter more than the bare ids:

| source | current state | next action |
|---|---|---|
| `#870` orphan class | `fo1`: live 14-day scope has 0 ORPHANED, 4 STAGED, 84 UNKNOWN, 10 EXECUTED. | Treat as retrieval/join debt, not a large live orphan population. |
| `#879` p1a unknowns | `p1a`: 86 UNKNOWN -> 25 not-follow-on, 21 already done, 40 real debt -> 29 open. `qj1`: top heads queued with owners. | Use qj1 owners/fire-orders. |
| `#880` p2a backlog | `p2a`: 114 open rows from harness transcript; false closure detector fixed. `iv1`: 16/16 T1 rows owned. | Continue row-by-row join/fold, not blanket reopen. |
| `#886/#887` | `hv2`: consumed by content; p1a 29 + p2a 18 reduced and routed. `iv1` fired head bundle. | Fold into hv2/iv1 content receipts. |
| p1a #28 paid Modal/T4 row | `iv1`: not fired because staging is not verified; no terminal artifact in searched `.py` population. | Do not dispatch until staging names the file. |

## Beyond-seed negative sweep

| receipt | negative wording audited | lens result |
|---|---|---|
| `ub1` UB1-A | post-hoc stored corrections "law" applied to seg/TR1 | **citation over-scope**: formulation pose/witness/no-inverse cannot kill seg/TR1 carrier. |
| `ub1` UB1-B | plane-storage family rate-dead | **noun over-scope**: exact reversible L3 raster residual storage is dead; rehomed/layer/five-type/generative/score-quotient reps are not closed by it. |
| `sm1` seg search | search defects on QA03 actuator are small | **scoped true**: real on this token actuator, but priced at ~0.06% of gap; says nothing about address/basis/carrier changes. |
| `ss1` selection vs search | D2 pfs1 cap diagnosis | **corrected**: relin cap never binds for D2 under correct params; placement/multistart is the useful axis. |
| `bz1/lr2` phase field | phase-field dead | **mechanism/family split**: offset field carriers lose; solved feature-bearing paint still requires its own gate. |
| `fz4` tz1 map | adaptive map loses | **formulation**: current map loses; gentle/pose-aware maps stay routed via qo1. |
| `qo1` lower-k repair | lower k cheap truncation negative | **formulation**: cropped existing k6 coeffs, not fresh lower-k solve. |
| `gc16` BR-D/floor text | live base beats flat-paint floor | **not contradiction**: flat-prototype paint floor is wrong object for trained textured renderer; carriers vs learned burn is a race. |

Rows that stood after attack:

- Exact reversible plane storage remains dead on its object: 409,526,925 B vs a 264,320 B box is
  not recoverable by entropy coding.
- Truth-content transport remains dead on its object: `sq1` truth paint eta_net -3.7640, 0/32.
- The fz4 tz1 map row is a real negative for that map: delta S +1.2148976, with pose damage
  dominating and seg failing break-even before pose.

## Scorer-batch action

I appended one new section to `.omx/research/scorer_batch_20260804.md`:

- `R4/#935` sq1 realizer uncap and convergence test, guarded by the pose-bank rule. This is a
  measurement spec, not a promoted candidate. It must not spend the single n600 slot until sg4
  releases it, and it must not promote from the batch note alone.

Existing batch rows already cover rt1 rate receipts and qo1 pair-bitpack F0PR1. I did not duplicate
those.

## Boundaries

- No scorer was run and no archive was built by this audit.
- No negative-existence claim is global; every absence above names its searched scope or source.
- The broad cap census is static and apparatus-level; the four load-bearing cap rows are the
  memo-level findings I classified in this charter, not an exhaustive proof that only four exist.
- The scorer batch is a queue artifact. The frontier only moves after byte-closed archives score
  through the evaluator.

own-vehicle frontier: `S = 0.7541459 @ 358,084 B [macOS-CPU advisory]`.
