# NB1 — optimal-form audit of the negative-result corpus

Date UTC: 2026-08-09  
Axis: `[scorer-free source/registry audit]`  
Score claim: `false`  
Promotion eligible: `false`

## Outcome first

NB1 materialized **24,786 typed records**:

- **24,627 machine-corpus candidate records**: 11,840 AU1 correction candidates, 8,157 AU1 headline/body candidates, and 4,630 VO2 instrument/source candidates. Every one is individually preserved as `UNREACHED`; none is promoted from keyword, numeric-window, or token metadata into a negative verdict.
- **159 canonical negative identities**: 145 latest logical probe negatives, 12 current blocked/cancelled task identities from the pre-routing snapshot, and 2 new reciprocal consumer defects.

Within the 159 canonical negative identities, the grade denominator is:

| grade | count | meaning here |
|---|---:|---|
| `PARADIGM_FALSIFIED` | 1 | the corrected n600 BEV v2 negative, scoped only to its exact G1-calibrated absolute chart |
| `IMPLEMENTATION_FALSIFIED_REOPEN` | 6 | four source-body probe rows plus two current built-cure/unwired consumer rows |
| `UNREACHED` | 152 | 138 probe bodies not opened in this bound, 12 operational task statuses that license no technical verdict, and 2 source-body rows where no treatment existed to grade |

Across the complete 24,786-row file, adding the 24,627 detector/instrument candidates yields **1 paradigm / 6 implementation / 24,779 unreached**.

This is a **partial audit with an exact remainder**, not a whole-corpus seal. The bounded body-read tranche reached 21/159 canonical identities: seven latest-negative probe identities, all 12 structured task-status negatives, and two reciprocal consumer findings. It did not body-adjudicate 138/145 current negative probes or any of the 24,627 AU1/VO2 candidate records.

No scorer forward, evaluator, launch, dispatch, archive build, promotion, Metal/MPS/CUDA operation, public-PR-intake edit, or upstream edit occurred. The exact contest pointer did not move.

## Optimal-form declaration and TOY bracket

Reference form is the charter's exhaustive, body-adjudicated, denominator-reported loop-until-dry audit. NB1 used one legal scope reduction: wall-clock bounding with an exact typed `UNREACHED` remainder.

AU1 and VO2 are an explicit **TOY-BRACKET only as candidate generators**. AU1 uses phrase/numeric windows and VO2's 4,512 new rows use a two-token source heuristic. Those mechanisms enumerate provenance edges; they do not grade L1–L4 and forfeit every family verdict. AU1 itself calls the outputs coarse candidate detectors, not truth rows (`AU1_RECEIPT.md:11,42-50`, commit `7b72d4edaf97a6702807bd2e956be1d75ee720b7`). VO2 calls its new rows deliberately overinclusive source candidates needing consumer confirmation (`ddm_vo2_20260806/RECEIPT.md:74-97`, commit `6eb9ab4a985fbe1f62b2fe3666f313007a62fb73`).

## Corpus denominators

| corpus | physical denominator | logical/candidate denominator | NB1 body-adjudicated | exact unreached |
|---|---:|---:|---:|---:|
| AU1 corrections | 11,840 rows | 11,840 unique `(source,line,phrase)` rows; 2,733 matched source memos out of 7,592 scanned | 0 | 11,840 candidate rows |
| AU1 headline vs body | 8,157 rows | 8,112 research-memo loci + 37 task-ledger rows + 8 git subjects; 1,866 unique source locators | 0 | 8,157 candidate rows |
| VO2 original registry | 4,630 rows | 4,630 unique `instrument_id`; 118 seeded instruments + 4,512 overinclusive source candidates | 0 | 4,630 candidate rows |
| probe outcomes | 662 append-only events | 370 latest logical probe IDs; 145 latest negatives = 116 DEFER + 18 KILL + 9 INDEPENDENT + 2 INFRASTRUCTURE_FAILURE | 7 | 138 negative identities |
| canonical task status, pre-routing snapshot | 522 append-only events | 190 latest task IDs; 11 blocked + 1 cancelled | 12, all as operational/custody `UNREACHED` | 0 structured status rows |
| research Markdown, pre-NB1 artifact snapshot | 8,088 files | 634 case-insensitive `verdict_scope` candidate files | selected bodies and mandatory priors only | unique source-negative denominator unknown |

AU1's two files are not additive negative populations. At pinned-body absolute-line resolution, 7,137 research-memo loci overlap; corrections contribute 11,840 loci and headline/body contributes 8,112, for **12,815 unique research-memo loci**, plus 45 ledger/git loci = **12,860 distinct evidence locators** before source-body adjudication. This still is not a count of distinct negative claims. Corrections use absolute file lines; research headline rows use body-relative lines and must be offset past the selected title. Current-tree regeneration drift affects 424 correction rows across four edited memos and 440 headline rows across three edited memos, but every row resolves at artifact commit `7b72d4e`.

VO2's R2 decomposition covers only 23/4,630 rows and is explicitly partial/unsealed (`RECEIPT.md:15-34`). NB1 therefore did not inherit its heuristic `NAIVE-NAMED` element tags as verdicts.

The charter's probe path spelling is stale: the live and TY1-read artifact is `.omx/state/probe_outcomes.jsonl`. Its current logical semantics are append-order last-row-wins per `probe_id`, not timestamp sorting (`src/tac/probe_outcomes_ledger.py:1133-1165`, commit `f39783f3eaea755b2357c6677e4ce7386607cdc9`). All 662 physical events remain preserved.

## Grade and lens tallies

Primary-lens denominator is the six `IMPLEMENTATION_FALSIFIED_REOPEN` rows:

| primary lens | rows | canonical rows |
|---|---:|---|
| L1 NAIVE | 1 | G14's unbounded stage-identifier handling stopped a lawful branch before the treatment ran |
| L2 TOY | 0 new | the load-bearing prefix/cap rows were already settled by NA1/NA2/NA3 and were deduped, not restamped |
| L3 GENERIC BASIS / METRIC / CODER / DEFAULT | 2 | G57 generic x264/equal allocation; M2 fixed global Morton/site ladder instead of the named adaptive Fisher/curvelet form |
| L4 OTHERWISE NOT OPTIMAL | 3 | BEV v1 invalid positive control; readiness expiry-helper bypass; costate coverage guard not propagated |

L4 is also a secondary lens on the L1 row and both L3 rows, so L4 incidence is 6/6 implementation reopens. The one paradigm negative carries no L1–L4 defect: BEV v2 corrected the positive control, ran n600, and falsified only the exact named chart.

### The seven body-read latest-negative probes

The deterministic tranche was the seven current negative identities with the greatest latest-event physical row numbers after append-order reduction.

| probe row | grade | scope and reason |
|---|---|---|
| `bev_staticity_v2_absolute_trajectory_n600_20260721` (`probe_outcomes:662`) | `PARADIGM_FALSIFIED` | D0 passes n64/n600, 0/600 singleton mismatches; Road/Lane remain non-static only for the exact G1-calibrated PoseNet chart (`bev_staticity_v2...md:1-67`, commit `eb04ed8a5a37bc0e99b75ea817591df65c1c0d7c`) |
| `bev_staticity_developability_c1_hood_gate_20260721` (`:661`) | `IMPLEMENTATION_FALSIFIED_REOPEN`, L4, `FIRED` | v1's invalid ego canonicalization failed its positive control; v2 fired the cure and produced the scoped chart result (`bev_staticity_developability...md:1-10,32-71,102-104`, commit `6d20159b7c99f25ab3d9f7593d9c5446311c4f74`) |
| `g103_g102_adversarial_review_20260727` (`:659`) | `UNREACHED` | direction admissible, but S01 executable compiler/public dispatch does not exist (`codex_findings_g102_adversarial...md:9-37,132-160`, commit `91b251e96ea3d57ace8962bd0681ed8fc30c3f95`) |
| `g57_direct_task_layered_x264rgb_46k_n600_20260726` (`:658`) | `IMPLEMENTATION_FALSIFIED_REOPEN`, L3+L4 | full n600 generic x264/equal-rate control measured `S=39.30593503092899 [macOS-CPU advisory]`; literal selected-preimage PROGRAM_RESIDUAL was not run (receipt commit `95b3143dd299ac7b7507c8c6166dac88eacfb011`) |
| `g14_full_n2_stage_id_blocker_20260726T143433Z` (`:652`) | `IMPLEMENTATION_FALSIFIED_REOPEN`, L1+L4 | 29 G8+A rows survived; a lawful long stage identity hit a harness refusal, not a representation verdict (receipt commit `50e42395d2d3e4ee4f4dcac1f2a3b4b1f014e9b7`) |
| `ddm_j7_ws1_launchability_and_pose_gate_20260724` (`:647`) | `UNREACHED` | 0/2 starts had archive/hash/bytes or compatible live optimizer-state custody; warm-start family never ran (receipt commit `26c2077892e028686b0486d64064c8b5fff7ea11`) |
| `ddm_m2_kinetic_laguerre_at_tolerance_n600_20260723` (`:646`) | `IMPLEMENTATION_FALSIFIED_REOPEN`, L3+L4 | all 72 registered cells failed Stage A, valid for the fixed ladder only; Fisher-ranked adaptive sites and curvelet/shearlet reformulations remain unmeasured (`codex_findings_ddm_m2...md:12-25,50-91,116-140`, commit `043b0000702aef1fe1f7c86b90060591fce3b75a`) |

Each row's complete metric tuple, cure, measurement, owner, disposition, evidence commit, and store pointer is in `NB1_ROWS.jsonl`.

## Reciprocal check: built cure, unwired consumer

Two current findings survived prior-audit dedupe:

1. **Readiness consumes expired blockers.** `tools/asymptotic_pursuit_candidate_readiness_assessment.py:390-432,1118-1121` filters `blocker_status=blocking` before reducing to latest and never applies expiry (commit `de75b8b137fcb89e7020b56bee4a7f28cf9f7fd5`). The canonical cure exists at `src/tac/probe_outcomes_ledger.py:1133-1198` (commit `f39783f...`). The private reader falsely blocks 6/12 canonical candidates on expired rows 310, 316, 318, 360, 500, and 512 while the canonical effective set has none for those candidates. Owner `ddm_nb1_readiness_successor`; task `nb1_probe_readiness_expiry_helper_wire_20260809`; fire order 2.
2. **Costate coverage qualification stops at the digest.** The activation ledger says its only writer is the retired launcher and 31 governed TR1 receipts produced zero rows (`src/tac/witness_dsl/activation_ledger.py:373-391`, commit `7f6301ed...`). `tools/costate_digest.py:992-1020` carries that vacuity, but `producer_bridge.py:285-335`, `shadow_controller.py:624-664`, and `ddm_costate_organ.py:547-552` rank the raw 198-row duty set without it. Owner `ddm_nb1_costate_coverage_successor`; task `nb1_costate_coverage_guard_propagation_20260809`; fire order 3.

Already-settled reciprocal rows were cited and not re-emitted as NB1 negatives: the strided sampler exists but the shared decoder remains prefix-only (NA2 `:435-441`); `head_relax_gain` is built but the no-distill interaction cell is empty (BA30 `:248-256`); directional basis #502 is built-never-raced (RV1 DAG FEED `:42-45`); BA31's build census is 197 rows including 8 built-elsewhere-unwired and 165 built-never-fired (`:286-300`); DN1 already routes `margin_targets` only inside an exercised margin A/B (`DN1_RECEIPT.md:58-71,104-115`); and the rank-4 hinge is already owned in the required-component store (`ddm_sb2_complete_the_stubs_20260731.md:163-180`, commit `50393d8308dc8cb7647d75b578889dd59e700e82`).

## Top three RE-OPEN measurements by stakes per cost

All three have paid cost `$0`; therefore a numerical stakes/paid-cost quotient is undefined. The ordering uses decision stake first and estimated wall-clock second. Every magnitude below is explicitly evidence-calibration, measured contrast, or projection—not a promised score move.

| rank | row and stakes | cost | named reactivation measurement | consumer / route |
|---:|---|---|---|---|
| 1 | **Four stored/post-hoc pose verdicts on prefixes.** Prefix difficulty is 2.54–4.21× population and can inflate the square-root pose contribution by up to about **+1.33 S**; this is evidence-calibration stake, not recoverable score. | `$0 CPU`; wall-clock unmeasured | Re-run the same four formulations on identical representative indices: matched-n whole-drive strided control plus seeded random/stratified `n>=120`; report pair IDs, population ratios, d_pose, and the original-prefix comparison. | pose arm; existing task `ddm_na2_strided_rerun_four_pose_family_verdicts_20260803` (`canonical_task_status:476`) |
| 2 | **Adam bias correction #824.** GC15's mechanism assigns a **plausible, unmeasured 0.011–0.047 S** range to the reset artifact; it is not banked. | `$0`, about two hours for the named arm; optimal-form replication may cost more | Matched same-checkpoint ON/OFF with identical schedule and reset magnitude, varying only `bias_correction`; run long enough to cover the registered reset transient, use at least two seeds, and route full d_seg/d_pose/bytes. | R1-B / burn owner; existing pending task `824` (`canonical_task_status:388`); GC15 commit `ca4850be524c4ca00625e1693b840da7c976368e` |
| 3 | **`head_relax_gain` without distillation.** The only existing contrast inside distillation is favourable by `d_seg=-5.73e-5`, i.e. **0.00573 seg-score units**, at 1.92× the stated noise; transfer to no-distill is unknown. | `$0`; one 40-epoch warm window | Matched no-distill A/B varying only `head_relax_gain`; verify init render identity, then route d_seg, d_pose, archive bytes, recomputed S, and the relax×distill interaction. | `dw1_head_relax_successor`; new task `nb1_head_relax_no_distill_interaction_cell_20260809`; BA30 `:238-256`, commit `05812c7f0645a0fdd84f53602b3309473c328e23` |

The broader recall removed QA03 from this table. BA31's 2026-07-31 memo called the split one measurement away, but the 2026-08-01 DAG FEED records the same-support/coder label-cost measurement and canonical equation, and states that the live in-place edit has no separable streams (`sub015_DAG...md:27069-27084,27124-27129`, commits `6f95b305685...` and `9f45920dca6...`). NB1's mistakenly registered task was immediately cancelled at task-ledger row 527; it is not open work.

## Mandatory prior-audit dedupe

Prior-local IDs are namespaced; none is treated as a global negative identity.

| prior | commit | what it settled; why it is not the NB1 denominator |
|---|---|---|
| TY1 | `3f0f3f514546acbacad32d104ac795155caac3f6` | 37/37 grouped implementations: 6 TOY, 15 NAIVE, 13 optimal-form, 3 not built (`ddm_ty1_20260806/RECEIPT.md:5-31`); explicitly leaves AU1/VO2 R2 (`NEXT_IF_RESUMED.md:3-6`) |
| NA1 | `0d5717c2daae04b4c783d292e4d649f287e4c9b4` | prefix, floor, exhaustion, and pose-veto lenses; union below 1% of 9,704 docs and explicit JSON/equation/worktree gaps (`ddm_na1...md:478-513`) |
| NA2 | `a47d38ceb4a9bbb980a4cd1fd3640a2aeb407920` | binary/basis/granularity/toy lenses; 4,906 counted-but-unread docs and unmeasured rate-axis prefix bias (`ddm_na2...md:560-573`) |
| NA3 | `ebfcf6b78ee4c4338a353599584d29038937b5e1` | 321 cap-default sites / 317 silent, but only four load-bearing cap families; explicitly nonexhaustive (`ddm_na3...md:27-33,93-105,155-162`) |
| BA29 | `a5695d1d9b0eae5769fa15a2b24e8fa314ae08b4` | 34 verdict-shaped statements, 31 placed and 3 unplaceable in its bounded window (`ddm_ba29...md:33-47,386-398`) |
| BA30 | `05812c7f0645a0fdd84f53602b3309473c328e23` | 44/44 placed across six response surfaces; 6 housekeeping and 8 unmeasured coordinates (`ddm_ba30...md:13-27`) |
| BA31 | `6fb0bced5c818a8ac9749d53bbf02dd69de52558` | 43/43 dated artifacts read; 34 is only a lexical match; separate 197-row build census (`ddm_ba31...md:286-300,509-526`) |
| EA1 | `392cf1d483a3eadf5c613f9e021a18d6ec9564a5` | eight post-RV1 memo groups N1–N8, no corpus denominator (`ddm_ea1...md:1-87`) |
| NG1 | `103aebfe6d33a8fca6bc314db0635ad739f763ed` | ten recent negative-signal rows, no global denominator (`ddm_ng1...md:16-20,49-62`) |
| RV1 DAG FEED | `36d14cf240d0006ada56250d96d2868dc02cc787` | 20 grouped negatives under R1–R8 plus 12 non-reactivations X1–X12; no later population check (`ddm_rv1...DAG_FEED:15-64`) |
| HG1 | `77cb4a9223f05f8faa3b05d2d3d9d4d73f616e9f` | two targeted geometry/instrument leads and local falsifiers; not a corpus denominator (`ddm_hg1...md:11-81,471-477`) |

## RECALL EVIDENCE

Queries and surfaces read before and during adjudication:

- Mandatory audit bodies and receipts, searched by local row IDs, source paths, and all seven selected `probe_id` values. No selected probe ID was found in the mandatory audits; those seven are new relative to that named scope.
- Full machine stores, parsed strictly line by line: AU1 correction/headline JSONL, original VO2 registry, all 662 probe events with documented latest-row semantics, and the 522-row pre-routing task snapshot.
- `.omx/research/` body recall with `rg -i 'prefix bias|generic basis|position.*label|margin_targets|expiry|ledger_coverage'`, plus `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*`.
- Canonical equations via `tools/list_canonical_equations.py --json`, targeted to `score_marginal_lagrange`, `margin_band_satisficing`, `ddm_dc1_correction_stream_label_cost`, and `menu_saturation`.
- Source-call searches for raw probe/task readers and costate duty-queue consumers; current code bodies were opened at the exact lines cited above.
- Live task/hot-state joins, including the harness/task-store distinction and later HV1/FEED dispositions before treating a row as ownerless.

Beyond the charter seeds, this recall found the two new consumer defects, DN1's already-routed `margin_targets` result, the required-component rank-4 row, and—most importantly—the later DC1 equation/DAG settlement that invalidated a new QA03 follow-on. That changed the plan: QA03 was cancelled rather than duplicated, and `head_relax_gain` became the third ranked open measurement.

## Routing denominator and append-only mutations

`NB1_ROWS.jsonl` routing denominator is **24,786 typed rows / 24,786 with owner + exactly one disposition + named store / 0 unrouted**.

Disposition values are limited to the charter vocabulary. Machine candidates and 138 unread probe bodies are queued to `nb1_machine_corpus_body_adjudication_r2_20260809`; detailed probes fold to their existing probe rows except the already-fired BEV cure; the 12 operational task rows fold to their existing task IDs; the two new reciprocal rows have new pending tasks.

The canonical task ledger moved append-only from the frozen audit input **522 events / 190 task IDs** to **528 events / 195 task IDs**:

- pending: machine-corpus R2, readiness expiry wiring, costate coverage propagation, and head-relax/no-distill;
- registered then cancelled in the same run: QA03 position/label split, after the later DAG/equation settlement was found.

No prior task status was overwritten. The cancellation is retained as evidence of the self-audit correction.

## Loop-until-dry record

| round | operation | new canonical rows | correction/exclusion |
|---|---|---:|---|
| R0 | strict machine census, identity validation, AU1 cross-file locator join | 0 negatives; 24,627 typed candidate edges | refused candidate→negative promotion |
| R1 | source-body read of seven latest negative probes + reciprocal caller/consumer sweep | 9 body rows: 4 implementation, 1 paradigm, 2 no-treatment unreached, 2 reciprocal implementation | excluded stale required-component RunConstantGates wiring and existing intentional all-event readers |
| R2 self-audit | mandatory-prior/DAG/equation/task rejoin over every R1 row and named follow-on | **0 new canonical rows** | cancelled duplicate QA03 task; kept rank-4 and `margin_targets` in prior stores; did not requeue task #936 from stale pending text |

R2 is dry only for the bounded body-read tranche. The global audit remains unsealed because 24,627 candidate rows and 138 current negative probe bodies are explicitly unreached.

## Honest limits

- There is no safe additive denominator of unique historical negative claims across all research memos. Prior audits use local/grouped IDs, citations drop scope, and AU1 detector rows overlap. NB1 therefore reports exact physical/candidate denominators and an exact current canonical negative denominator without pretending they are one population.
- `UNREACHED` does not mean clean, refuted, or absent. It means the body/identity join was not completed in the declared scope.
- Ranked stakes are not additive and are not score promises. The prefix row calibrates evidence; bias correction is a plausible range; head-relax transfers an observed within-distill contrast into a still-unmeasured interaction cell.
- No exact score was measured. No score component was recomputed by NB1. No pointer moved.

Own-vehicle frontier: **S = 0.7534578126155775 @ 357,837 B `[macOS-CPU advisory]`**, from `.omx/state/main_hot_state.md:5-20`; contest pointer 0.19108 remains borrowed/unmoved.
