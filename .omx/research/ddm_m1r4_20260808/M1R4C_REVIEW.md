VERDICT: FINDINGS_RESET_COUNTER

# ddm_m1r4c arithmetic, authority, and provenance review

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source/receipt inspection; scorer-free analytic review]`.
`score_claim=false`, `promotion_eligible=false`, `scorer_forwards_run=0`,
`metal_runs=0`, `launch_mutation=false`, `ticket_mutation=false`.

## Answer first

This is not a clean pass. I completed the element-level sweep and found twelve
distinct mechanisms. Two are launch-blocking:

1. the seal DAG does not enforce the review dependency when it evaluates an
   already-existing guard receipt; and
2. the ticket claims a live event-driven controller and atomic per-eval JSONL,
   but the pinned trainer has neither on its `mlx-train` path.

The corrected one-pixel arithmetic is numerically right, but the cure is not
structural: the derived values remain frozen in two structured fields and a
third evaluator string. The fp32 route also reuses the fp16 `16.0 GiB`
projection even though its keyed receipt measures a larger footprint. The
counter therefore remains `0/3`; do not fire.

## Frozen subject verification

The four binding content-hash checks passed before subject inspection:

| subject | charter prefix | independently computed SHA-256 | result |
|---|---|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `9c8373b5b352cacc` | `9c8373b5b352cacc2456a21eac0deb53e32f445eb942e4675043825a1d896500` | MATCH |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` | `60fc0501a65d8d09b9bacd57cafd414544eac340e4107fa52f0beccfa60bbee6` | MATCH |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `1ef18faf37e2f171` | `1ef18faf37e2f171d480b4e8073c453185f9ae00a1b3200b46d5bb258cd60895` | MATCH |
| `tools/ddm_seal_orchestrator.py` | `e592cb36fb00d502` | `e592cb36fb00d502693cf17ef43da0f01c7f7c7aecc7d59a3e25e6efeb36e2dc` | MATCH |

Commit `1381ac84cbb4119071709598e4c3b88beed7fe40` is an ancestor of the
review-time HEAD, and `git show 1381ac84cb:<path>` reproduces all four full
digests above. Live HEAD advanced through descendant receipt commits during the
review, exactly as the charter permits; it was never used as a freeze key. I
did not use sibling verdicts as evidence.

## Findings, most severe first

| id | severity | mechanism | smallest structural cure |
|---|---|---|---|
| M1R4C-F1 | LAUNCH-BLOCKING | `evaluate_gate` computes unmet dependencies but ignores them for guard gates. A previously passed burn-guard receipt can therefore satisfy the guard and make `FIRE` read `READY` while `review_passes` is still `0/3` (`ddm_seal_orchestrator.py:267-284`). | Make dependency satisfaction a prerequisite for every dependent gate, bind each guard verdict to the exact ticket SHA and review-counter snapshot, and have the in-process fire guard refuse an unsealed ticket rather than relying on operator choreography. |
| M1R4C-F2 | LAUNCH-BLOCKING | The ticket promises atomic per-eval JSONL and event evaluation at every eval row (`launch_ticket...json:760-786`). `run_mlx_train` only appends rows to an in-memory list, writes checkpoints every 250 steps, and writes the result at process end (`renderer.py:3079-3219,4025`). The only `_write_jsonl` calls belong to the offline facets analyzer (`:1584-1590,1850-1853`). A bounded corpus search found no consumer of this ticket's `stop_policy`. The actual live controls are `--steps 3250` and safe-run timeout. | Add a resume-safe atomic/append per-eval journal on the real training path and an executable controller that imports its inputs from the ticket, derives `TrajectoryStopConfig`, emits a typed stop reason, and gates resume/termination. Until then, remove the event-driven/telemetry claim and do not fire. |
| M1R4C-F3 | HIGH | The fp32 sigma argv carries `--projected-gib 16.0`, while its own keyed receipt measures MLX `13.707321 GiB` plus RSS `2.019913 GiB`. Under the ticket's stated sum-and-1.5 rule the projection is `23.590851 -> 24 GiB`; under the pinned renderer's max-and-1.5 rule it is `20.5609815 -> 21 GiB`. Neither produces 16. | Store one receipt-derived projection per argv key, including receipt SHA and rule ID, and generate each safe-run wrapper from that keyed object. Never copy the fp16 projection into the fp32 route. |
| M1R4C-F4 | HIGH | The scope cure is incomplete. The three named torch sites now derive `n` from `len(pair_ids)`, but the live CoreML success payload still says `n32-or-smaller` although `args.pairs` can select more than 32 (`renderer.py:2069-2105,2187-2199`). The MX1T markdown generator also hardcodes n32 (`:1593-1604,1683`) while its result carries runtime population. A targeted test still expects the old lie and now fails: expected `n32`, got `n3`. | Derive every emitted/report scope from the same runtime `pair_ids` object; if a route is intentionally capped, enforce the cap before work. Update tests to compute the expected scope from their fixture population. |
| M1R4C-F5 | HIGH | The ticket upgrades the underlying receipt axis from `[macOS-CPU advisory torch upstream SegNet]` to “d_seg authority” in both the root axis and nested d_seg measurement (`launch_ticket...json:508,702-708`). `score_claim=false` prevents a contest-score claim, but the authority label still drops the mandatory macOS advisory qualifier. | Import a typed axis/authority record verbatim from the measured receipt; never restate it in prose. Component truth may be described as frozen-scorer advisory, not contest authority. |
| M1R4C-F6 | HIGH | Mutable seal status is frozen as prose. `review_passes=[]` and the live orchestrator report `0/3`, but `main_fire_sequence` and `seal_gates_remaining` start at pass 2 and pass 3 (`launch_ticket...json:518-555,679-699`). A reader following the ticket omits fresh pass 1. | Delete stored “remaining” prose. Render the next steps from `build_gates/evaluate_gate` and the current review ledger at consumption time. |
| M1R4C-F7 | HIGH | Load-bearing evidence is path-bound, not content-bound. The guard compares normalized cache/init paths, and the ticket omits hashes for both mem-probe receipts, both verdict receipts, the 943,720,076-byte GT cache, and the PR130 init. RR16 measured a 4.3x d_seg swing from cache identity. All resolve now, but replacing bytes at the same path would preserve guard equality and invalidate memory/sigma/d_seg provenance. | Put SHA-256 in the ticket's input identity and keyed projection records; have the probe, guard, trainer, verdict, and harvester recompute and compare the same hashes. Bind harvest fields to receipt hashes. |
| M1R4C-F8 | MEDIUM | The sigma prose calls `2.5e-6` “objective units”; it is `5e-7 d_seg/eval * 5 rows = 2.5e-6 d_seg`, or `2.5e-4 S` after the `100*d_seg` objective. In addition, the determinism predicate is `len(distinct)==1 and len(ckpt_shas)==len(keys)`: the second conjunct is tautological and permits one non-null SHA plus missing SHAs to prove determinism (`ddm_seal_orchestrator.py:551-566`). | Use typed units and derive the S conversion. Require `all(ckpt_shas.values())`, the expected repeat count, and one distinct recomputed file SHA before setting checkpoint-derived sigma to zero. |
| M1R4C-F9 | MEDIUM | The n120 recipe imports `lr=2e-7`, CE `0`, softplus sentinel `-999`, and `bits=4` from PR130/n32 without current-n120 re-derivation. The ticket admits the batch-geometry transfer, and Plan15 explicitly required a reference-form LR rather than the `2e-7` probe convenience config (`ddm_plan15_20260808.md:34-42`). Under this charter's ladder these are BORROWED, not clean imports. | Give the stage recipe one typed owner and derive/select the n120 LR from a same-start, same-population bracket or a registered scaling law before interpreting a plateau. |
| M1R4C-F10 | MEDIUM | Operational literals have no pinned owner: d_seg timeouts `5400`, sigma timeout `3600`, d_seg projection `24`, verdict batch `32`, seed `20260808`, and the 5-step calibration horizon. The guard also independently hardcodes auto-microbatch 4 while the trainer reads `WC2_AUTO_MICROBATCH_ANCHOR["selected_default"]` (`mx1_fire_guard.py:121-137`; `renderer.py:86-100,2614-2658`). | Move launch policy and microbatch derivation into one imported typed registry/LawRef. The ticket should serialize the selected value plus source receipt SHA; the guard and trainer should call the same function. |
| M1R4C-F11 | LOW | Descriptive provenance has already drifted: the ticket cites PR130 `train.sh:113` for `--lr 2e-7`, but the current pinned external file has LR at line 112 and CE at 113. The doctrine says `28.7 s/step`; the keyed n120 receipt measures `29.32177193959554`, giving 982.21 rather than about 1,003.48 steps in 28,800 seconds. | Resolve flags by parsing a content-hashed argv/config artifact, not by a line number. Render wall-clock projections from the keyed receipt at read time. |
| M1R4C-F12 | LOW | The required common contract freezes a mutable pointer as “Live” (`_common_contract.md:52-56`), but the live board it orders reviewers to read holds a newer own-vehicle score and byte count (`main_hot_state.md:5-17`). | Import/render the common-contract frontier from the canonical hot-state pointer instead of copying score, bytes, axis, and gap prose into a second file. |

### Failure scenarios, one per finding

| id | concrete failure scenario |
|---|---|
| M1R4C-F1 | A stale-but-passing main guard receipt survives a review reset; the orchestrator marks that guard satisfied and exposes `FIRE` even though the review ledger is still `0/3`. |
| M1R4C-F2 | The burn runs until the fixed cap/timeout because no executable stop-policy consumer sees eval rows; a crash also loses the in-memory trajectory that the ticket calls atomic telemetry. |
| M1R4C-F3 | Safe-run admits the fp32 route using 16 GiB although the keyed receipt projects 21 or 24 GiB, so the supposedly governed calibration can overcommit memory or fail mid-stage. |
| M1R4C-F4 | A valid run over more or fewer than 32 pairs emits an n32 scope, causing later reviewers to bank or reject evidence against the wrong population. |
| M1R4C-F5 | A downstream ledger promotes a macOS-CPU component observation as cross-host/contest authority because the advisory qualifier was erased in the ticket. |
| M1R4C-F6 | An operator follows `main_fire_sequence` after a reset and starts at pass 2, silently skipping the required fresh first pass. |
| M1R4C-F7 | Cache, init, receipt, or verdict bytes change in place while their paths stay constant; the guard still passes although the measured memory and d_seg no longer describe the fired inputs. |
| M1R4C-F8 | A missing checkpoint SHA is treated as deterministic because the dict-length test is tautological, and the incorrectly labelled d_seg envelope is compared directly to S thresholds. |
| M1R4C-F9 | An n120 plateau is attributed to vehicle capacity even though the borrowed n32 learning rate or stage-tail recipe, not the vehicle, caused the trajectory. |
| M1R4C-F10 | The guard and trainer drift to different microbatch or resource-policy constants; preflight validates one execution shape and the trainer runs another. |
| M1R4C-F11 | A reviewer checks cited line 113, sees CE rather than LR, and falsely records the LR pin as verified; independently, the stale 28.7 s/step forecast schedules more work than the measured timeout window can execute. |
| M1R4C-F12 | A reviewer repeats the common contract's older score/byte pair as current and computes the wrong gap or pointer disposition despite having read the newer hot state. |

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | effect on this review |
|---|---|---|---|
| Governing/live authority | Full reads of `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, common contract, and `main_hot_state.md` | The common contract's claimed live frontier is stale; hot state is authoritative. M1 is unsealed and scorer-free. | Used hot-state pointer, ran no scorer/Metal, and made the stale common-contract literal C7 evidence. |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json`, exact filters for `trajectory_derived_stopping_law_v1` and `score_marginal_lagrange_multipliers_v1` | The stopping law says safety caps are reported, never convergence; the score law pins `lambda_seg=100` and denominator `37,545,489`. | Recomputed the event bar and S arithmetic independently; rejected a fixed-cap implementation masquerading as event control. |
| M1 antecedents | `rg` for `one_sample_flip`, `sigma`, `fire.guard`, `n120 receiver`, then reads of GC21, NG1, WC1/WC2/WC3, M1R2, M1R3, RR16, MX1G, and Plan15 | Plan15 requires per-eval JSONL and rejects the 2e-7 convenience config as reference form; WC2 retains the microbatch choice but points to an evicted mutable WC1 receipt; RR16 makes cache content identity load-bearing. | Added F2, F7, F9, and F10 rather than treating the ticket prose as proof. |
| Canonical index/DAG | `rg -i 'ddm_m1|mx1|one_sample_flip|event-driven receiver|sigma.*seal|fire.guard'` over `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` | No current-M1 entry was found in the canonical index; DAG hits were older unrelated M1 naming and general guard/event material. | Scoped negative: no newer indexed artifact superseded the pinned ticket or receipts in the searched index/DAG surfaces. |
| Task/P0 ledgers | Same terms plus `#984` over canonical task status, operator P0, and landing ledgers | Found the older built `DDMEventContinuationV1` engine record, but no ledger row binding it to this ticket/trainer. | Prompted the source-level consumer search; no M1 stop-policy consumer was found in searched `tools/src/experiments` scope. |
| Design/spec surfaces | `rg -i 'mx1|M1 n120|event-driven|receiver trajectory'` over SPECs, `ddm_plan15`, and `ddm_map1` | Plan15 line 40 names the per-eval JSONL as debt to be paid, not an optional description. | Turned the source mismatch into a launch-blocking NO-FAKE finding. |
| Current artifact tree | `shasum`, `stat`, `jq`, source `rg`, exact pair-id regeneration, and dry orchestrator status | All 25 existing run/receipt artifacts resolve; future main result/guard do not yet exist; current status is held at `0/3`. | Separated current resolvability from structural content-binding failures. |

## C1: full numeric-literal sweep

### Counting rule and denominator

I traversed every JSON scalar. The denominator is **238 numeric token
occurrences**: 35 JSON-number leaves plus 203 numeric tokens embedded in string
leaves. This deliberately counts repeated argv values separately. Numeric
substrings inside hashes/path identifiers were excluded by an identifier-boundary
rule; dates, task IDs, and line numbers appearing as prose claims were included.

The 128 command tokens are itemized first. The second table covers the remaining
110 tokens by exact JSON path family. Coverage is **238/238**; this was not a
sample.

### C1-A: executable command numbers (128/128)

| paths / semantic value | occurrences | class | derivation or receipt |
|---|---:|---|---|
| all safe-run `--rss-mb 90000` | 9 | IMPORTED | `ROW1_SAFE_RUN_RSS_MB=90_000`, pinned renderer `:76` |
| main `--timeout 28800` | 1 | IMPORTED | `ROW1_SAFE_RUN_TIMEOUT_S`, renderer `:77` |
| d_seg `5400` and sigma `3600` timeouts | 8 | ORPHAN-LITERAL | no registry, LawRef, or measured duration pin found |
| main plus five fp16 sigma `--projected-gib 16.0` | 6 | MEASURED-PINNED | fp16 mem receipt SHA `91ad0bee...`; ticket sum rule gives 16 |
| fp32 sigma `--projected-gib 16.0` | 1 | FROZEN-LITERAL | keyed fp32 receipt SHA `12efb06f...` derives 24 by ticket rule, not 16 |
| two d_seg verdict `--projected-gib 24.0` | 2 | ORPHAN-LITERAL | no verdict-memory receipt or named rule in ticket |
| all `--verdict-batch-size 32` | 11 | FROZEN-LITERAL | duplicated parser/guard defaults, no pinned measurement owner |
| all `--pairs 120` | 9 | IMPORTED | GC21/M1 population and runtime pair-id derivation |
| main `--steps 3250` | 1 | MEASURED-PINNED | WC1 derived recommendation, SHA `717ad481...`; safety cap only |
| six sigma `--steps 5` | 6 | ORPHAN-LITERAL | NG1 imports methodology, not this horizon |
| two `--mem-probe-steps 3` | 2 | ORPHAN-LITERAL | parser default copied into ticket, no derivation pin |
| all `--lr 2e-7` | 9 | BORROWED | PR130/n32 regime; no current n120 re-derivation |
| all `--ce-fraction 0.0` | 9 | BORROWED | PR130 stage-tail recipe |
| all `--softplus-fraction -999.0` | 9 | BORROWED | PR130 disable sentinel |
| all `--bits 4` | 9 | BORROWED | PR130 q4/init regime |
| all `--seed 20260808` | 9 | ORPHAN-LITERAL | recorded and reproducible, but no named seed-policy owner |
| all explicit `--microbatch-pairs 4` | 9 | MEASURED-PINNED | WC2 typed row SHA `b095561d...`; runtime derives 30 chunks at n120 |
| main `--checkpoint-every 250` | 1 | IMPORTED | GC21 predicate and P0 cadence |
| sigma checkpoint `5` plus probe checkpoint `3` | 8 | FROZEN-LITERAL | hand-copied from the corresponding horizon instead of derived |
| main `--eval-every 50` | 1 | IMPORTED | GC21 event predicate |
| sigma eval `5` plus probe eval `1` | 8 | FROZEN-LITERAL | hand-copied protocol cadence, no owner/import |
| **total** | **128** |  |  |

### C1-B: typed fields and prose claims (110/110)

| exact path family | tokens | class | disposition |
|---|---:|---|---|
| `fire_protocol.pre_fire_liveness_proof` | 2 | IMPORTED | RR8 refusal rule |
| `main_fire_sequence[*].expected` | 6 | FROZEN-LITERAL | stale 2/3 and 3/3 status, F6 |
| `n120_stratified_indices_source` | 2 | DERIVED-IN-PLACE | regenerated 120/120 IDs from seed; exact receipt match |
| `provenance.cf2_confound_rows` | 2 | IMPORTED | CF2 receipt |
| `provenance.gc21_convocation` | 2 | IMPORTED | pinned GC21 historical state |
| `provenance.init_checkpoint_source` | 3 | BORROWED | explicitly PR130 |
| `provenance.lr_provenance` | 5 | BORROWED | value verified, line pin stale, n120 transfer un-derived |
| `provenance.ng1_crosswalk` | 1 | IMPORTED | NG1 methodology only |
| `provenance.throughput_provenance` | 2 | MEASURED-PINNED | WC3 SHA `89810e8e...` |
| `resumability` | 1 | IMPORTED | source checkpoint cadence/atomic writer |
| `review_passes_required` | 1 | IMPORTED | GC21 three-pass law |
| `safe_run_projection.arithmetic` | 4 | FROZEN-LITERAL | repeats derived fields as prose |
| `safe_run_projection.axis` | 2 | MEASURED-PINNED | fp16 receipt timestamps |
| two measured peak number leaves | 2 | MEASURED-PINNED | exact fp16 receipt values |
| `measured_total_gib`, `projected_gib` | 2 | FROZEN-LITERAL | inputs are adjacent; consumer does not derive |
| `safe_run_projection.supersedes` | 2 | DERIVED-IN-PLACE | `66.268951/10.515302=6.302144...` |
| `seal_gates_remaining[*]` | 13 | FROZEN-LITERAL | mutable state; stale after reset, F6 |
| `sigma_calibration.dseg_unit_measurement` | 4 | MEASURED-PINNED / DERIVED-IN-PLACE | verdicts plus checkpoint identity; authority label separately fails C2 |
| `sigma_calibration.fp16_fp32_delta_measured` | 3 | MEASURED-PINNED / DERIVED-IN-PLACE | exact loss values and subtraction |
| `sigma_calibration.governance_note` | 4 | IMPORTED | historical governed-run identifiers/date |
| `sigma_calibration.protocol` | 5 | ORPHAN-LITERAL | repeat/horizon values are not supplied by NG1 |
| `sigma_calibration.sanity_sigma_measured` | 11 | MEASURED-PINNED / DERIVED-IN-PLACE | five losses plus mean/n/sigma/hash-count derivations |
| `sigma_calibration.seal_falsifiers` | 4 | FROZEN-LITERAL | duplicated bars; one wrong unit, F8 |
| `sigma_calibration.source` | 1 | IMPORTED | NG1 rank/methodology |
| `stop_policy.doctrine` | 6 | IMPORTED (4) / FROZEN-LITERAL (2) | the `1000`/`28.7` projection is stale |
| `stop_policy.invalidation` | 1 | IMPORTED | GC21 safety-cap law |
| `stop_policy.predicate` | 18 | IMPORTED (13) / FROZEN-LITERAL (5) | the five are duplicated evaluator/bar/one-flip values, F2 |
| `telemetry.cpu_torch_facets` | 1 | ORPHAN-LITERAL | future lane identifier, no executable binding here |
| **total** | **110** |  |  |

The unacceptable classifications are not twelve separate findings per repeated
occurrence; they collapse to the mechanisms F2-F3 and F6-F11 above.

## C2: authority labels

I audited eight authority-bearing claim families in the ticket/Main finding.

| claim family | evidence | verdict |
|---|---|---|
| ticket root MLX axis | MLX train telemetry only | CORRECT as research-signal |
| ticket “CPU-torch d_seg authority” (root and nested) | underlying verdict axis is `[macOS-CPU advisory torch upstream SegNet]` | INCORRECT/UNDERQUALIFIED, F5 |
| ticket memory `MEASURED` | fp16 receipt SHA `91ad0bee...`, sampled real load/train step | CORRECT for memory telemetry only |
| ticket WC3 throughput `MEASURED` | n32 five-step Metal wall-clock instrument, WC3 SHA `89810e8e...` | CORRECT but instrument-scoped; not a score |
| ticket LR `SOURCE-VERIFIED` | external `train.sh` SHA `1d014518...`; value is at line 112 | VALUE VERIFIED; cited line 113 is stale; current-n120 optimality is not measured |
| ticket n32 `MEASURED-descending` | MX1G advisory receipt SHA `d403283a...` | EVIDENCE EXISTS, but axis/population does not transfer optimality to n120 |
| Main finding header and n120 mislabel measurement | two verdicts have pair_count 120 and old `n32` scope | CORRECT `[macOS-CPU advisory]`, `score_claim=false` |
| Main finding “RE-DERIVED/cure” claim | arithmetic was recomputed, but output remains three frozen copies | NUMERIC HISTORY CORRECT; structural-cure implication is false under this charter's definition |

No contest-CPU/CUDA score is claimed by these artifacts. No
`upstream/evaluate.py` run occurred in this review.

## C3: scope-string census

The mechanical denominator was 13/13 `verdict_scope` assignments plus every
line matching `\bn[0-9]+` or `\bn=[0-9]+` in the four pinned artifacts: 19
ticket lines (many are paths), 28 renderer lines, zero guard lines, and one
orchestrator line. All 48 matching lines were read.

| surface | result |
|---|---|
| renderer `:1382`, `:1860`, `:2007` | CLEAN: all derive `f"n{len(pair_ids)}..."`; `pair_count` uses the same list. |
| five CoreML blocker scopes `:2082-2173` | CLEAN: typed ENVIRONMENT/FORMULATION/INSTANCE scopes; they do not assert n. |
| CoreML success `:2192` | FINDING: live scope says n32-or-smaller without an enforced cap. |
| four mem-probe blocker scopes `:3385-3391` | CLEAN: typed environment/instance scopes. |
| orchestrator sigma scope+n `:531-540` | CLEAN: both string and numeric n derive from `len(values)`. |
| orchestrator harvest horizon scope `:619-624` | CLEAN: calibration-vs-burn boundary is honest. |
| exact two `n32 arms` strings `:3438`, `:3742` | CONFIRMED as error/help/protocol prose for the v4 two-arm ticket author, not a verdict payload. |
| MX1T markdown `:1604`, `:1683` and recall prose `:1888`, `:1907` | FINDING: report population is hardcoded even though runtime result owns pair IDs. |
| ticket n120 source | CLEAN: receipt contains 120 unique IDs, regenerated exactly by `np.array_split(range(600),120)` with seed `20260808`; never prefix. |
| ticket sigma scope | CLEAN: six runs total, five fp16 repeats; the scoped sigma n is five. |
| existing d_seg verdict receipts | HISTORICAL MISLABEL: both say n32 but carry 120 IDs; Main finding correctly records this. Current source cure does not rewrite old evidence. |

The targeted test run independently exposed the unfinished cure:

```text
63 passed, 1 failed
test_run_torch_verdict_receipt_schema_with_checkpoint_pair_ids
expected: n32 arm-selection instrument
actual:   n3 arm-selection instrument
```

## C4: drift twins and rightful owners

| duplicated/drifting value | copies | rightful owner/import |
|---|---|---|
| one-pixel S and marginal bar | structured ticket fields plus evaluator string plus GC21 prose | N/H/W/eval inputs; one executable derivation consumed by monitor |
| review progress | `review_passes`, `main_fire_sequence`, `seal_gates_remaining`, orchestrator report | review ledger + `evaluate_gate` |
| per-eval telemetry | Plan15/ticket prose versus absent trainer implementation | trainer's durable journal schema and consumer receipt |
| fp16/fp32 safe-run projection | one singular ticket projection copied across dtype routes | keyed mem-probe receipt + projection LawRef |
| auto microbatch 4 | guard hardcode, renderer WC2 anchor, explicit ticket argv | shared microbatch-selection function/typed registry |
| sigma bars and units | GC21, ticket strings, orchestrator literals | typed sigma policy with unit enum |
| authority axis | verdict receipt, ticket root prose, nested ticket prose | receipt's typed axis record |
| three clean passes | ticket field, orchestrator default/detail, GC21 | ticket-required value imported by all rendered text |
| guard schema/tool | guard constants, ticket strings, orchestrator hardcoded path | guard module exports imported by author/orchestrator |
| live own-vehicle pointer | hot state, common contract, renderer report prose | hot-state canonical pointer API |

## C5: receipt and path resolvability

The whole-scalar path census found **41 distinct paths** in the ticket:
**39/39** paths expected to exist before the main fire resolve, while the two
future main outputs (`result.json` and `fire_guard_verdict.json`) are absent as
expected. The three additional document/source paths embedded in prose (GC21,
NG1, and `upstream/evaluate.py`) also resolve; the one brace template expands to
the five existing fp16 run directories. No unexpected path is missing.

Within that census, all **25/25 existing M1 run artifacts referenced directly
or transitively by the ticket** resolved and were hashed: six guard verdicts,
six training results, six step-5 checkpoints, two CPU verdicts, one sigma
harvest, two mem-probe receipts, and two mem-probe results. All five checkpoint
SHA-256 claims serialized in the ticket match the bytes on disk.

| evidence group | verified hashes / result |
|---|---|
| fp16/fp32 mem receipts | `91ad0bee7e16827205b5baff82de9087b261aec74df49f01f7e377cb59709ef9` / `12efb06fa41423f77e82beb5935375fb7eaf202b077264d2ee53fc66e5ccccd1` |
| fp16/fp32 mem results | `bd2a99f3d622b6e7af73303dbae16a19ad5cb6d910de3d2cd92f7e0a0b7fc94b` / `1c607e82c935d09af4c5b411af796a0701a1abe0fdba70ae4d1b897bb884c6fd` |
| five fp16 checkpoints | all `56047d059595b36887a77b2940ebfd15f607413ee82cbd09f2eb946e50eba55c`, 831,396 B each |
| fp32 checkpoint | `9f5ec7ef3ee5cb6b376e1cfbc201a9ffd950870f2969c42c3a08884d06236302`, 831,391 B |
| fp16/fp32 d_seg verdicts | `264d4073f32949b8a17dc35c90761789fbb16a733452dbf0f65b1b2eb981fc0f` / `128ebc55c975a440500877be0c7bd93b906f0bdcf9822d1c4d25e6401ad75dca` |
| sigma harvest | `bfdd921982eef458b90c53567bb60abffacff68a558f7a31cee838629663fe6d` |
| GT cache | `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4`, 943,720,076 B |
| PR130 init | `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf` |
| GC21 / NG1 / WC3 | `15f6d2febc23e7eb779ebaa93d902d7470aec612a9a4c6bba54cd9f6de1d06ee` / `26f76a4496ad55a2a15af69889f0a60d28a724c372c4f94ea59375b96ce28845` / `89810e8e9d27d46b8b1e99f18bbc78a784231ac601efaf0b964018d3e5dd207b` |
| evaluator | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| renderer / guard / orchestrator | match charter pins above |
| commit tokens `d14391b1d4`, `f9ab8fb399` | both resolve as Git commit objects |

No persisted artifact path uses the system temporary directory. The older v4
ticket authoring path uses only a repo-local transient directory and is absent
from this v5 ticket.

Resolvability is not identity: the ticket stores only the mutable paths for most
of the hashes above. Also, WC2's typed row pins the consumed WC1 receipt as
`4efecaf1...`, while the current mutable path hashes to `50ebae86...`; WC3
documents that round 2 evicted round-1 rows. The durable WC2 secondary receipt
still resolves, but the claimed primary content no longer does at that path.

## C6: independent arithmetic

### Event predicate

```text
pixel denominator = N*H*W
                  = 120*384*512
                  = 23,592,960 pixels

one_sample_flip_S = 100 / 23,592,960
                  = 5 / 1,179,648
                  = 4.238552517361111e-6 S

marginal_bar_S_per_step
                  = one_sample_flip_S / 50
                  = 1 / 11,796,480
                  = 8.477105034722223e-8 S/step (nearest binary float)
```

The two numeric ticket fields are correct. The defect is ownership: both remain
stored outputs beside their inputs, and the evaluator prose stores the bar and
3250 a third time.

```text
plateau envelope = 5e-7 d_seg/eval * 5 rows
                 = 2.5e-6 d_seg
                 = 2.5e-4 S under objective_S=100*d_seg
```

Thus ticket line 753 has the wrong unit.

### Microbatch

The explicit CLI value takes the first branch in both trainer and guard:

```text
effective microbatch = max(1, min(explicit=4, total_pairs=120)) = 4
chunk count          = ceil(120/4) = 30
```

The n120 fp16 and fp32 receipts both record source `explicit_cli`, microbatch 4,
and chunk count 30. Current fire/receipt equality is correct; F10 is the
duplicated default/owner, not this explicit derivation.

### Memory projections

```text
fp16 total = 8.493787 + 2.021515 = 10.515302 GiB
fp16 margin = 10.515302 * 1.5 = 15.772953 GiB -> 16 GiB
old/new ratio = 66.268951 / 10.515302 = 6.302144...

fp32 total = 13.707321 + 2.019913 = 15.727234 GiB
fp32 margin = 15.727234 * 1.5 = 23.590851 GiB -> 24 GiB
```

The ticket's fp16 arithmetic and 6.3x prose are correct. The fp32 argv's 16 is
not derived from its keyed receipt.

### Sigma seal

Five fp16 final losses are exactly
`0.0003770271432586014`, so mean is that value and sample sigma is zero. The
five checkpoint files are byte-identical at SHA `56047d05...`; distinct SHA
count is one. The fp32 loss is `0.0003566459927242249`:

```text
abs loss delta = 0.0003770271432586014
               - 0.0003566459927242249
               = 0.0000203811505343765
```

The ticket's binary-float spelling
`0.000020381150534376502` is consistent. `delta_in_sigma=null` is correct when
sigma is zero. Both CPU verdicts independently report
`d_seg=0.0010835435655381944`, so the d_seg delta is zero. This is a calibration-
horizon advisory result, not full-burn or score authority.

### Contest S arithmetic

`upstream/evaluate.py:63-65,90-92` defines the archive-byte rate and score. The
canonical denominator is 37,545,489. Using the current live pointer's
unrounded displayed components (`d_seg=0.004305419922`,
`d_pose=0.000716508925`, bytes `357837`):

```text
seg  = 100 * 0.004305419922                         = 0.4305419922
pose = sqrt(10 * 0.000716508925)                    = 0.084646850207...
rate = 25 * 357837 / 37,545,489                     = 0.238268970208...
S                                                    = 0.7534578126155775
```

This only rechecks the existing `[macOS-CPU advisory]` pointer. M1R4C produced
no archive and no new score.

## C7: self-check and apparatus findings

I re-derived rather than trusted the apparatus values:

1. All four subject digests match both the working-tree bytes and
   `git show 1381ac84cb:<path>`; that commit is an ancestor of review-time HEAD.
   The charter correctly says live-HEAD movement is not a finding.
2. The round's `0/3` state agrees across the empty ticket `review_passes` list
   and a read-only orchestrator status. Both named cure commits resolve as Git
   commit objects.
3. The charter's two named `n32 arms` strings are indeed historical help/prose,
   but I treated them as leads rather than an exhaustive count; the mechanical
   census found the separate live/report scope defects recorded in F4.
4. The common contract's “Live frontier” is older than the hot state it orders
   the reviewer to read. That duplicated mutable value is apparatus finding
   F12; hot state must own it.

The apparatus finding was not a reason to stop the completed content-hash
audit. Expected descendant HEAD movement did not change the reviewed bytes.

## Validation and boundaries

Read-only targeted tests:

```text
.venv/bin/python -m pytest -q \
  tools/tests/test_mx1_fire_guard.py \
  tools/tests/test_ddm_seal_orchestrator.py \
  experiments/tests/test_ddm_mx1_memory_probe.py

63 passed, 1 failed in 1.70s
```

The failure is the stale old-scope assertion documented in F4. A read-only
orchestrator status reported fp16/fp32 probes, sigma runs, d_seg verdicts, and
harvest satisfied; review counter `0/3`; main guard pending; FIRE held. No gate
was executed.

Measured/recomputed here: file hashes and sizes, receipt values, exact pair-id
regeneration, arithmetic, source/control-flow behavior, and test status. Not
measured here: no new d_seg/d_pose, no MLX/Metal training, no scorer forward, no
archive, no byte-close, no exact contest CPU/CUDA evaluation, no pointer move.

## Closing disposition

The frozen-literal genus is **not drained** in this artifact. I finished the
bounded sweep; I did not stop after the first finding. The result is twelve
mechanisms over a denominator of 238/238 numeric tokens, all 13/13
`verdict_scope` assignments, and every population-bearing regex hit in the four
pinned files. MAIN should cure the structural owners above and restart all
three independent passes from `0/3`.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
