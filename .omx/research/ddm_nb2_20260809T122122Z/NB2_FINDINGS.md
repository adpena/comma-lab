# ddm_nb2 — NB1 round 2: 138 source-body adjudications

**UTC:** 2026-08-09T12:21:22Z  
**Authority:** scorer-free source audit; no score, archive, eval, launch, dispatch, or promotion  
**Primary population:** exactly the 138 `PROBE:*` rows graded `UNREACHED` by NB1 because their source bodies had not been opened  
**Machine store:** `NB2_ROWS.jsonl`

## Verdict first

The stated population is dry: **138/138 adjudicated, 138/138 routed, 0/138 unreached, 0/138 unrouted**. A second, independent schema/identity self-audit added **0 new identities** and found **0 missing owners, dispositions, source-row locators, route stores, or body-recovery records**. This seals only the chartered 138-row population. It does not grade NB1's 24,627 AU1/VO2 candidate-generator rows.

The apparent population was mostly stale control-plane history, not 138 live negative experiments. Ninety rows are superseded by corrected evidence or by vehicle retirement, 41 fold as operational/design/already-scoped records, four have a named cure that was committed after the original row, and three remain worth a current-vehicle re-test. The later `backfill` event on many rows only added queryable reactivation criteria; it explicitly did not invalidate or re-evaluate the technical verdict (`retroactive_sweep...md:61-63`). Treating append-order backfill as fresh technical evidence would have reproduced the stale-headline genus.

## Denominators and outcomes

| surface | denominator | result |
|---|---:|---:|
| NB1 `UNREACHED` probe identities in scope | 138 | frozen from `ddm_nb1_20260809T030914Z/NB1_ROWS.jsonl` |
| locally available evidence targets opened byte-completely | 118 | 115 files + 3 complete directory trees |
| bodies recovered without a live local target | 20 | 16 full append-only ledger/event-history recoveries + 3 Git-commit recoveries + 1 remote-only locator recovered from the ledger |
| body adjudications | 138 | 100% |
| routed rows | 138 | 100% |
| unreached / unrouted | 0 / 0 | dry |

The 20 recovery rows do not pretend a missing file was opened. Their machine rows say exactly which recovery grade applies and retain the original `.omx/state/probe_outcomes.jsonl` row number and SHA. I did not find the six missing/no-longer-present referenced files at their recorded paths; ten other source rows never named a separate body path. Their original append-only rows and later correction/supersession events were sufficient to adjudicate the identity without inventing a body.

### Dispositions

| disposition | count | meaning here |
|---|---:|---|
| `SUPERSEDED` | 90 | corrected body or retired historical vehicle; lesson retained, vehicle not reopened |
| `FOLDED` | 41 | operational/design/planning record or an already-correctly-scoped negative; no new experiment |
| `ALREADY-DONE` | 4 | the exact named blocker was cured and committed |
| `QUEUED-WITH-FIRE-ORDER` | 3 | under-tested observation with a named current-vehicle measurement |
| `FIRED` / `DEFERRED` | 0 / 0 | this scorer-free arm neither owned a runnable measurement nor found a new measured blocker |

### Body grades

| grade | count |
|---|---:|
| `SUPERSEDED_RETIRED_OR_CORRECTED` | 89 |
| `NOT_A_LIVE_TECHNICAL_NEGATIVE` | 35 |
| `NEGATIVE_UPHELD_SCOPED` | 6 |
| `CURE_BUILT_AND_WIRED` | 4 |
| `REOPEN_UNDERTESTED` | 3 |
| `SUPERSEDED_BY_CORRECTED_BODY` | 1 |

The six upheld negatives are the n600 fixed-form/feature-source results from `grokking_ridge_round2`, `replace_round3`, `replace_round4`, `replace_round5`, `pre_se_multi_source`, and the corrected P0 K=2 costate receipt. They stay at their bodies' declared formulation or feature-source scope; none is promoted to a family verdict.

## Four-lens tally

Lens counts are **non-exclusive incidence counts**, so they do not sum to 138.

| lens | rows | finding |
|---|---:|---|
| L1 NAIVE | 6 | fake/placeholder execution path, explicitly pre-optimal renderer, or known implementation cure omitted |
| L2 TOY | 12 | a verdict outran its pair/sample population; includes four-pair latent projection, n24/n3 boundary work, three-pair curve/store work, and pair-0 YOPO |
| L3 GENERIC BASIS | 26 | dominant-class reducers, generic DWT/UNIWARD/HILL/HUGO/k-means/RFF/ridge/convex feature sources, or an unevaluated default standing where a derived/raced choice was owed |
| L4 OTHERWISE NOT OPTIMAL | 95 | vehicle-bound historical constants/objects now forbidden as transferable evidence; proxy/wrong-object forms are included only where the body actually used them |

The large L4 count is not inferred from a token census. The per-row substrate/body was opened or recovered, then checked against the binding no-old-lineage rule. The corpus-level NA2 estimate that roughly two-thirds of pre-July negative mass is retired is explicitly a mention heuristic and an upper bound (`ddm_na2_negative_audit_20260803.md:93-112`); it was recall evidence, not a substitute for these 138 adjudications.

## Ranked RE-OPEN table

The charter's measured gap decomposition is **seg 0.4015 S / pose 0.2776 S / rate 0.1126 S against PR130 floor 0.172141**. No body measured current-vehicle cost-to-falsify, so I do not fabricate a numeric stakes/cost quotient. Rank is by axis stake, directness, and reuse of an existing apparatus; every cost remains `UNMEASURED` until the owning run records it.

| order | row | why it survives | exact reactivation measurement | cost status / consumer |
|---:|---|---|---|---|
| 1 | `yousfi_detector_cost_blindspot_b_20260617` | Its `KILL` rests on n24 plus a three-pair/50-step in-cell treatment. A later body says the native-grid treatment is real through R but explicitly requires a full-population byte-closed A/B (`closure_reaudit...md:42-48`). Directly attacks the 0.4015 S seg gap. | Same-seed uniform vs detector-cost-weighted vs direct in-cell repair on one PR130-derived/current successor; stratified-random n>=120, prefer n600; actual R, parse-back, byte-close, one GT decoder. | `UNMEASURED`; task `nb2_retest_yousfi_native_grid_incell_pr130_20260809` |
| 2 | `ego_hood_per_frame_mask_region_corrected_reopen_20260617` | The original negative measured the all-frame static core, not the per-frame class-4 hood region. The corrected body calls the old region wrong and leaves survival/bytes open (`closure_reaudit...md:50-59`). | Per-frame hood-edge clamp on the current vehicle; stratified-random n>=120, prefer n600; byte-close/decode then actual-R survival against same-decoder untreated control. | `UNMEASURED`; reuses row-1 harness; task `nb2_retest_ego_hood_survival_pr130_20260809` |
| 3 | `yopo_first_layer_costate_pair0_early_boundary_late_20260713_v2` | The receipt itself says pair 0, K={1,2,4}, rank-or-kill ineligible, yet the ledger says `KILL`. This is an instance observation, not a population negative. It matters only if a live costate consumer wants the reuse. | Sealed current checkpoints; matched exact-vs-reused first-block costates at K={1,2,4}; stratified-random n>=32 for ranking and n>=120 before a negative; report descent/regret and runtime. | `UNMEASURED`; task `nb2_retest_yopo_firstblock_population_20260809` |

No old HNeRV/PR101/PR110 vehicle is authorized by these reopens. The 2026-07-23 ban makes those substrates lessons-only (`sub015_DAG...md:21859-21869`), while the current operator roadmap makes PR130 the new, explicitly attributed base (`main_hot_state.md:108-113`; `ddm_op1r.../CHARTER_optimal_path_forward.md:90-101`). That changed the plan from “rerun historical cures” to “transfer only the question onto a current base.”

## Reciprocal cure check

I searched the 138 probe IDs, their reactivation text, `.omx/research`, `.omx/state`, `src`, `experiments`, and Git history for the exact cure names and wire-in surfaces.

| original rows | cure disposition | evidence |
|---|---|---|
| `fp64_master_gradient_600_pair_independence_pending_20260518` | `ALREADY-DONE` | independence measurement/correction commit `6f17beaf8468bfd3ae7e5bcaf6aed82fc7f9f98b` |
| `vq_vae_k_sweep_dispatch_attempt...` + `harvest_e7_vq_k_sweep_1...` | `ALREADY-DONE` | A10G remediation and successful K=512 harvest commit `8134867d445a0a9c00ace62a9cbcc80c47a43ec4` |
| `pr110_opt7_l1...pending_trainer_auth_eval_wire_in...` | `ALREADY-DONE` | four-helper wire-in commit `86e3f4c382d99aff61e5114e2b6e3323f477c9eb` |

**Built-elsewhere-unwired cures found: 0 in the searched 138-row source/code/store scope.** The two Z6 identity-predictor rows are not counterexamples: their bodies say the exact MLX/Z6-v2 surfaces were not built, and those vehicles are now retired. The four rows above were built **and wired/executed**, so they are closed rather than orphaned.

## Axis lens

Reference receipt: `/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/result_summary.json`, SHA-256 `15f43860a2d0a32bd7191ee12f3f1f1308cf3345090a4ad1cf2f3bb67bc5aa2c`. It measures same-host DALI-vs-AV seg disagreement `1.7523023416288197e-04`, pose MSE `1.4061325055081397e-04`, and a **triangle-inequality upper bound**, not a score, of `0.0550214568 S`.

I did not find in these 138 bodies a negative whose treatment and control used different GT decoders. The one sub-bound exact-score margin that explicitly spans both contest axes, `pr101_op7_raw_delta...`, pairs baseline and candidate inside each axis with identical runtime-tree SHA; it does not subtract CPU from CUDA and is safe on this lens. **Unsafe-as-filed count: 0/138.** I did not duplicate `ddm_ax2`'s corpus sweep.

## Routing and no-signal-loss

All 138 rows have a named owner and exactly one charter disposition. The 135 closed rows route through the machine-readable `NB2_ROWS.jsonl` store and retain their `probe_outcomes_ledger` consumer ID. The three reopens route to named `canonical_task_status` task IDs with fire order 1-3. I did not append 138 correction events to the shared probe ledger because it was already modified outside this arm; the machine store preserves the full join without overwriting or co-committing unrelated state.

Routing denominator: **138 adjudicated / 138 routed / 0 unrouted**.

## Self-audit round

The second pass checked:

- exact set equality against NB1's 138 `UNREACHED` probe identities;
- 138 unique `probe_id` and `nb2_id` values;
- one allowed disposition, one non-empty owner, and one route store per row;
- source ledger row plus SHA for every row;
- explicit local-open or recovery grade for every body;
- fire order, trigger, owner, consumer store, and measurement for every queued row;
- commit SHA for every `ALREADY-DONE` row;
- no `UNREACHED`, `UNOWNED`, `MAIN to route`, scorer claim, or current-vehicle reuse on a closed row.

Result: **0 new rows, 0 identity differences, 0 schema violations, 0 unrouted rows**. The 138-row primary population is sealed at this round. The AU1/VO2 candidate-generator population remains exactly as NB1 typed it and is outside this seal.

## RECALL EVIDENCE

Sources searched before adjudication:

- full `.omx/research` content queries for each probe ID plus `prefix`, `cap`, `generic basis`, `metric`, `coder`, `wrong-object floor`, `built-elsewhere-unwired`, `stale headline`, `DALI`, `PyAV`, and `races-not-reputation`;
- canonical equations via `.venv/bin/python tools/list_canonical_equations.py --json` (**429** equations returned);
- `CANONICAL_RESEARCH_INDEX*`, all 11 `sub015_DAG_*` surfaces, design/SPEC documents, `.omx/state/canonical_task_status.jsonl`, and the complete event history for all 138 probe IDs;
- NB1's exact population and provenance store, not the 24,627 AU1/VO2 candidate bodies.

Beyond-charter findings that changed the plan:

1. The binding no-old-lineage FEED and the new PR130-base exception make old HNeRV/PR rows lessons-only, so historical instance evidence is `SUPERSEDED`, not a license to rebuild those vehicles.
2. The May 30 backfill landing added criteria without re-evaluating bodies, explaining why append-order “latest” rows can revive stale headlines.
3. The later closure re-audit already identifies the Yousfi three-pair and ego-region defects, so NB2 routes exact population tests rather than re-stamping their old bodies.
4. The live own-vehicle pointer moved after the common contract snapshot to `0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; no NB2 source audit changed it.

## Boundaries and mission status

Measured in this arm: source/body availability, byte counts and SHA-256 values, row/event joins, disposition/lens/route denominators, and dry self-audit checks. Not measured: any d_seg, d_pose, archive bytes, score delta, current-vehicle treatment effect, scorer runtime, or cost-to-falsify. No Metal/MPS/CUDA, scorer, eval, launch, dispatch, archive build, upstream write, PR130-intake write, or promotion occurred.

Pointer delta: **0**. This audit is means, not goal progress. Own-vehicle frontier remains **S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]** (`tq1c`; `main_hot_state.md:6-9`).
