VERDICT: FINDINGS_RESET_COUNTER

# ddm_m1r5c arithmetic, authority, and provenance review

Tags: `[no-triality] [p0-ledger-ok]`

Axis: `[source/receipt inspection; scorer-free analytic review]`.
`score_claim=false`, `promotion_eligible=false`, `real_scorer_forwards_run=0`,
`metal_jobs_run=0`, `launch_mutation=false`, `ticket_mutation=false`.

## Answer first

This is not a clean pass. The content freeze is valid, but the exhaustive sweep
found fifteen mechanisms. The most immediate is launch-blocking: the ticket's
human fire sequence and the seal orchestrator still route FIRE to
`argv_m1_n120_cap_saturated`, the raw `safe_run` child, instead of
`argv_m1_controller_fresh`. The raw child has the in-process eval stop, but when
`safe_run` itself reaches the wall-clock cap it kills that child and returns 124;
only the skipped controller survives that boundary and writes the typed
`QUEUE_RESUME` receipt.

The amendment did add real durable eval rows, a real staircase-aware consumer,
EMA custody, and a no-LR-jump resume. Those cures are substantive. They do not
make the current fire order, terminal-basis selection, resource arithmetic,
scope/authority labels, or provenance clean. The review counter stays `0/3` and
the burn must not fire from this ticket.

## Frozen subject verification

I computed the four hashes before inspecting the subject:

| subject | charter prefix | computed SHA-256 | result |
|---|---|---|---|
| `.omx/research/ddm_m1_20260808/launch_ticket_v5_event_driven.json` | `90cf28d390999ef9` | `90cf28d390999ef9cda47340d9ec01bc65a15fb9ab3f88c60625abc29b414ec9` | MATCH |
| `tools/mx1_fire_guard.py` | `60fc0501a65d8d09` | `60fc0501a65d8d09b9bacd57cafd414544eac340e4107fa52f0beccfa60bbee6` | MATCH |
| `experiments/ddm_mx1_pr130_semantic_renderer.py` | `8bad6a6b8be1b201` | `8bad6a6b8be1b20189791283f64638d01a1b406482a1a725bab712c77f27894c` | MATCH |
| `tools/ddm_seal_orchestrator.py` | `11c4368f009afc31` | `11c4368f009afc31aca67da37d7c9b2a9b109f3a00d20ca903800c26f3897a79` | MATCH |

`git show 393d67d016:<path>` reproduces all four values. The charter's binding
stop test therefore did not fire. Live HEAD movement was not used as evidence.

## Findings, most severe first

| id | severity | mechanism | smallest structural cure |
|---|---|---|---|
| M1R5C-F1 | LAUNCH-BLOCKING | The executable controller exists, but FIRE does not point to it. Ticket lines 616-653 guard and fire `argv_m1_n120_cap_saturated`; the ticket has no root `fire_argv_key`; and `build_gates` defaults to that same raw child (`ddm_seal_orchestrator.py:237-259`). `run_m1_controlled_train` is the only caller that survives a `safe_run` return code 124 and writes the wall-cap `QUEUE_RESUME` receipt (`renderer.py:4320-4401`). | Give the ticket one typed launch-route record distinguishing `guarded_child_key` from `controller_fire_key`. Make both the rendered fire sequence and orchestrator import it, guard the child, and FIRE the controller. Add a route-identity test in both directions. |
| M1R5C-F2 | HIGH | Terminal basis selection is asserted but not executable. The ticket says a selector will name the minimum live/EMA/K8 d_seg (`launch_ticket:895-919`), but `run_m1_schedule_selection` only compares step 0 with step 250 and selects the LR schedule (`renderer.py:4285-4317`). No terminal selector exists. The three terminal commands are also frozen to step 3250 even though legal outcomes include an earlier event stop and the 6500 extension. | Materialize terminal facet commands from the actual terminal receipt's live/EMA/tail paths, verify identical pair IDs and checkpoint step, then run a distinct terminal-basis selector that content-binds all three verdicts and names the minimum. Never synthesize terminal paths from a forecast step. |
| M1R5C-F3 | HIGH | The fp32 sigma route still carries `--projected-gib 16.0`. Its keyed receipt measures 13.707321 GiB MLX plus 2.019913 GiB RSS; under the ticket's own sum-and-1.5 rule this is 23.590851, hence 24 GiB. The singular fp16 projection was copied into a different dtype route. | Make safe-run projection a per-argv-key derivation from a content-hashed mem receipt and one projection LawRef. Generate the wrapper value; do not store an independently editable copy. |
| M1R5C-F4 | HIGH | Load-bearing identity remains path-only. The ticket omits content hashes for both mem receipts, both CPU verdicts, the GT cache, the PR130 init, the schedule inputs/receipt, and future journal/terminal evidence. The guard compares normalized paths (`mx1_fire_guard.py:391-404`), not the bytes. RR16 already measured a 4.3x d_seg swing from cache identity. | Define one content-identity object per input/output (path, bytes, SHA-256, producer/config hash) and require the probe, guard, trainer, selector, controller, and harvester to recompute/import it. Bind schedule and terminal selectors to ticket and input hashes. |
| M1R5C-F5 | HIGH | Scope derivation is incomplete. Four live renderer verdict sites correctly use `len(pair_ids)`, but CoreML success still emits `n32-or-smaller` while `args.pairs` can select more than 32 (`renderer.py:2265-2266,2354-2360`). The MX1T markdown generator also emits n32 regardless of the runtime result (`:1764,1846`). A targeted test still expects the old hardcoded n32 label and fails against the repaired n3 payload. | Derive every emitted population label from the same `pair_ids` object, or enforce a cap before work. Make tests compute expected scope from fixture population rather than restating n32. |
| M1R5C-F6 | HIGH | The ticket upgrades `[macOS-CPU advisory torch upstream SegNet]` component evidence to “d_seg authority” at lines 606 and 803. `score_claim=false` prevents a contest-score claim, but dropping the mandatory macOS advisory qualifier is still an authority escalation. | Import the typed axis verbatim from each verdict receipt. Describe it as frozen-scorer component evidence on macOS-CPU, never unqualified authority. |
| M1R5C-F7 | MEDIUM | The production recipe still transfers PR130/n32 values: LR `2e-7`, CE `0`, softplus sentinel `-999`, and bits `4` occur in ten run/probe configs each. The ticket admits the batch-geometry mismatch; Plan15 explicitly says the n120 reference form is not the `2e-7` probe convenience config. A step-250 accept/rollback gate limits adoption risk but still spends the calibration segment on the borrowed values; under this charter's ladder they remain BORROWED, not IMPORTED. | Derive the n120 recipe from a same-start, same-population bracket or registered scaling law before the governed segment. If a borrowed point is retained as an experiment, type it as a non-production candidate and do not call the resulting run reference form. |
| M1R5C-F8 | MEDIUM | The EMA decay's inner LawRef arithmetic is correct, but its load-bearing input is frozen. The measured K8 bank uses members at steps 1500..3250, a 1750-step endpoint span. The ticket asserts without a mapping law that K×cadence = 2000 is a “two-time-constant window,” stores `warmup_fraction=2000/3250`, then re-derives only `d=0.999` from that stored fraction (`launch_ticket:878-886`; `renderer.py:234-251`). | Register/import the mapping from actual member steps and averaging rule to an EMA target moment/window, then derive warmup fraction and decay from those inputs. Otherwise label 0.999 a candidate and let the real n120 terminal selector adjudicate it. |
| M1R5C-F9 | MEDIUM | Sigma determinism can pass with missing hashes. The harvester filters nulls into `distinct`, then tests `len(distinct)==1 and len(ckpt_shas)==len(fp16_keys)`; the second term is tautological because the loop inserted every key (`ddm_seal_orchestrator.py:580-595`). One real SHA plus four nulls therefore proves “bit-identical.” | Require the expected repeat count, `all(ckpt_shas.values())`, recomputed file hashes, and exactly one distinct value before setting checkpoint-derived sigma to zero. Add the one-present/rest-null positive control. |
| M1R5C-F10 | MEDIUM | Seal units and threshold ownership drift. `5e-7 d_seg/eval * 5 rows` is `2.5e-6 d_seg`, or `2.5e-4 S`, not “2.5e-6 objective units” (`launch_ticket:849-852`). The orchestrator separately hardcodes 2.0e-6 and 2.5e-6 (`:614-636`) instead of importing the ticket predicate; `plateau_eps_dseg_per_eval` is not consumed by the live stopping evaluator. | Store a typed threshold policy with a unit enum. Derive d_seg and S forms at use time, and make both harvester and stop evaluator import the same policy/LawRef. |
| M1R5C-F11 | MEDIUM | Mutable state is frozen as prose. `review_passes=[]` and live status are 0/3, while `main_fire_sequence` and `seal_gates_remaining` begin at pass 2 and pass 3 (`launch_ticket:616-633,778-799`). The Main finding header still says `status: OPEN` although its appended amendment says the scope cure was applied. | Remove copied “remaining”/status prose. Render next actions and document status from the review ledger plus current gate evaluation. |
| M1R5C-F12 | MEDIUM | Operational values lack one owner: 5400/3600 timeouts, 24-GiB CPU projections, verdict batch 32, seed 20260808, five-step sigma horizon, probe cadence, and the 6500 extension bound have no named derivation/receipt. The guard also hardcodes auto-microbatch 4 (`mx1_fire_guard.py:121-137`) while the trainer reads `WC2_AUTO_MICROBATCH_ANCHOR` (`renderer.py:95-109,2767-2816`). Current explicit CLI 4 happens to mask that twin. | Put launch policy and microbatch selection in one typed registry/LawRef, serialize source receipt SHA, and make ticket author, guard, trainer, and orchestrator call/import it. |
| M1R5C-F13 | LOW | LR provenance cites PR130 `train.sh:113`, but the content currently has LR on line 112 and CE on 113. The value is source-verified; the descriptive location is stale. | Bind to the content hash and parsed flag/value record, not a mutable line number. |
| M1R5C-F14 | LOW | The charter contradicts itself: its immutable-subject body correctly names commit `393d67d016` (line 47), while OPTIMAL FORM still pins subject tree `1381ac84cb` (line 150). `git show 1381ac84cb` reproduces three old subject hashes, not this round's ticket/renderer/orchestrator. | Generate one subject manifest from the four content hashes and import it into every charter section; do not repeat a tree literal. |
| M1R5C-F15 | LOW | The common contract copies a live frontier that has already drifted: 0.7539807296911207 @ 357,836 B (`_common_contract.md:52-56`) versus the canonical hot state 0.7534578126155775 @ 357,837 B (`main_hot_state.md:5-17`). | Render/import the pointer from the canonical hot-state API instead of keeping a second mutable score/byte literal. |

### Concrete failure scenarios

| id | failure scenario |
|---|---|
| F1 | MAIN follows the ticket or orchestrator and runs the raw safe-run argv. At 28,800 seconds, safe-run kills the child and returns 124; no surviving caller emits the promised wall-cap terminal/queue receipt. |
| F2 | A valid event stop at step 3000 or an extension ending at 6500 leaves the fixed step-3250 terminal commands absent or stale; even at 3250 no executable receipt chooses live versus EMA versus K8. |
| F3 | The fp32 calibration is admitted as a 16-GiB job although its own measured projection is 24 GiB, allowing resource overcommit. |
| F4 | Cache, init, receipt, or selector bytes change in place; every path equality still passes and later evidence is attributed to the old content. |
| F5 | A run with 40 CoreML pairs or a non-n32 MX1T fixture emits n32 scope and is banked against the wrong population. |
| F6 | A downstream ledger treats a macOS advisory component row as contest/cross-host authority because the qualifier was erased. |
| F7 | A plateau is attributed to vehicle capacity when the un-derived cross-regime LR/stage recipe caused it. |
| F8 | K/cadence or horizon changes while stored phi remains 0.61538; the LawRef faithfully certifies a decay derived from stale geometry. |
| F9 | Four repeat results omit checkpoint hashes and one has a hash; the harvester declares the five runs bit-identical and sets d_seg sigma to zero. |
| F10 | The predicate window/epsilon changes but the harvester's copied 2.5e-6 does not, or a d_seg envelope is compared directly to S. |
| F11 | An operator follows the ticket after reset and starts at pass 2, omitting the first required independent review. |
| F12 | Guard and trainer defaults drift to different microbatch/resource shapes; the measured envelope and executed job cease to be the same config. |
| F13 | A reviewer checks line 113, sees CE, and records a false LR provenance check. |
| F14 | A reviewer reconstructs the subject from 1381 and reviews the pre-amendment files. |
| F15 | A report recomputes the gap or disposition from the common contract's stale pointer. |

## RECALL EVIDENCE

| searched surface / query | found beyond the charter seeds | effect |
|---|---|---|
| Governing files: full `PROGRAM.md`, shared `CLAUDE.md`/`AGENTS.md`, operating manual, common contract, and live hot state | The common contract's copied pointer is stale; hot state is the authority. | Used the hot-state pointer and recorded apparatus F15. |
| Canonical equations registry: `trajectory_derived_stopping_law_v1`, `score_marginal_lagrange_multipliers_v1`, `ema_decay_run_geometry_v1` | The EMA law derives decay from a supplied warmup fraction; it does not derive that fraction from K8 member geometry. Safety caps report rather than convergence. | Recomputed the EMA inner law, separated it from the unowned K8-to-EMA mapping, and checked cap semantics. |
| Full M1 corpus search for `one_sample_flip`, `event-driven`, `K=8`, `schedule`, `2e-7`, and `n120` | GC21 owns the event constants; Plan15 explicitly rejects 2e-7 probe convenience as reference form; M1C1 requires a surviving controller and same-object basis comparison. | Produced F1, F2, F7, and F8 instead of accepting the amendment prose. |
| Canonical research index/DAG search for `ddm_m1`, `mx1`, `event-driven receiver`, and `sigma seal` | Did not find a newer indexed artifact that supersedes the pinned ticket in the searched index/DAG surfaces. | Kept the frozen ticket as the adjudicated object. |
| Live source/receipt tree: scoped `rg`, `jq`, `shasum`, Git object checks, read-only orchestrator status | Confirmed current 0/3 status, raw-child FIRE key, existing receipt/hash values, and 31 planned-but-not-yet-created paths. | Separated expected future outputs from stale/dynamically-wrong paths. |

## C1: full numeric-literal and population sweep

### Counting rule

I traversed every JSON scalar. The numeric denominator is **285/285 semantic
numeric token occurrences**: 46 JSON-number leaves plus 239 numeric tokens in
string leaves. Repeated argv values count separately. Numeric substrings inside
hashes and lexical identifiers/path names were excluded by an identifier-boundary
rule; population descriptors such as n120/n32 were then audited separately in
C3. Of the 239 string tokens, 162 are executable command values and 77 are
typed/prose claims. This is a complete grouped enumeration, not a sample.

### C1-A: executable command values (162/162)

| path/value family | occurrences | class | derivation or disposition |
|---|---:|---|---|
| all safe-run `--rss-mb 90000` | 15 | IMPORTED | `ROW1_SAFE_RUN_RSS_MB`, renderer line 82 |
| main fresh/resume `--timeout 28800` | 2 | IMPORTED | `ROW1_SAFE_RUN_TIMEOUT_S`, renderer line 83 |
| d_seg/CPU-facet `--timeout 5400` | 7 | ORPHAN-LITERAL | no keyed duration owner |
| sigma `--timeout 3600` | 6 | ORPHAN-LITERAL | no keyed duration owner |
| main + five fp16 sigma `--projected-gib 16` | 7 | MEASURED-PINNED | fp16 receipt SHA `91ad0bee...`; 10.515302×1.5 -> 16 |
| fp32 sigma `--projected-gib 16` | 1 | FROZEN-LITERAL | fp32 receipt derives 24, F3 |
| d_seg + five CPU-facet `--projected-gib 24` | 7 | ORPHAN-LITERAL | no CPU-verdict projection receipt |
| every `--verdict-batch-size 32` | 17 | FROZEN-LITERAL | repeated parser/ticket policy, no pinned owner |
| every `--pairs 120` | 10 | IMPORTED | GC21 population; runtime sample receipt is 120 unique, non-prefix IDs |
| fresh `--steps 3250` | 1 | IMPORTED | WC1/GC21 forecast safety bound, never convergence |
| resume `--steps 6500` | 1 | ORPHAN-LITERAL | extension bound has no derivation receipt |
| six sigma `--steps 5` | 6 | ORPHAN-LITERAL | NG1 imports method, not this horizon |
| two probe `--mem-probe-steps 3` | 2 | ORPHAN-LITERAL | no measurement/derivation owner |
| LR/CE/softplus/bits (`2e-7`, `0`, `-999`, `4`) | 40 | BORROWED | PR130/n32 recipe transferred to n120; F7 |
| every `--seed 20260808` | 10 | ORPHAN-LITERAL | recorded/reproducible, but no named seed-policy owner |
| every explicit `--microbatch-pairs 4` | 10 | MEASURED-PINNED | WC2 anchor; live n120 receipt derives 30 chunks |
| main `--checkpoint-every 250` | 2 | IMPORTED | GC21/P0 cadence |
| sigma/probe checkpoint cadence 5/3 | 8 | FROZEN-LITERAL | copied from each short horizon, not derived |
| main `--eval-every 50` | 2 | IMPORTED | GC21 event predicate |
| sigma/probe eval cadence 5/1 | 8 | FROZEN-LITERAL | copied protocol values, no typed owner |
| **total** | **162** |  |  |

### C1-B: numeric typed fields and prose claims (123/123)

| exact path family | tokens | class | derivation or disposition |
|---|---:|---|---|
| `review_passes_required` | 1 | IMPORTED | GC21 three-pass law |
| `safe_run_projection` numeric/prose | 12 | MEASURED-PINNED (peaks/date) / FROZEN-LITERAL (total, margin, ratio copies) | receipt SHA verified; outputs are not regenerated by the consumer |
| all `sigma_calibration` numeric/prose | 32 | MEASURED-PINNED / DERIVED-IN-PLACE / FROZEN-LITERAL | five losses/checkpoint hashes and two d_seg rows verified; wrong objective-unit copy is F10 |
| `stop_policy.executor.ema` | 12 | MEASURED-PINNED (K8 bank/delta) / FROZEN-LITERAL (window/phi mapping) / DERIVED-IN-PLACE (inner decay) | bank row resolves; runtime re-evaluates the EMA LawRef; F8 |
| `event_free_horizon_evals` | 1 | IMPORTED | M1C1 B3 five-interval gate |
| `safety_bound_steps_by_key` | 2 | IMPORTED (3250) / ORPHAN-LITERAL (6500) | forecast versus unowned extension |
| `executor.schedule` numeric/prose | 6 | BORROWED (base LR) / IMPORTED (250,3250) / DERIVED-IN-PLACE (2e-9 hold) | same-object gate exists; provenance class remains borrowed |
| `stop_policy.predicate` + invalidation numeric/prose | 18 | IMPORTED (GC21 constants) / DERIVED-IN-PLACE (one flip, marginal, objective) | runtime recomputes one flip; threshold unit copy fails F10 |
| `main_fire_sequence[*].expected` | 6 | FROZEN-LITERAL | copied mutable 2/3 and 3/3 state, F11 |
| `fire_protocol.pre_fire_liveness_proof` | 2 | IMPORTED | RR8 refusal rule |
| `n120_stratified_indices_source` | 2 | DERIVED-IN-PLACE | runtime selection; receipt has 120 unique non-prefix IDs |
| `provenance.*` | 15 | IMPORTED / MEASURED-PINNED / BORROWED | GC21/NG1/WC3 resolve; LR line stale; PR130 regime transfer explicit |
| `resumability` | 1 | IMPORTED | M1C1/P0 contract, but launch-route realization fails F1 |
| `seal_gates_remaining[*]` | 13 | FROZEN-LITERAL | copied mutable state, F11 |
| **total** | **123** |  |  |

Every unacceptable occurrence above is covered by F3, F7-F8, F10-F12, or the
route/selection findings; repeated copies were collapsed by mechanism rather
than inflated into separate findings.

## C2: authority labels

I audited the authority-bearing claim families in both required documents.

| claim family | evidence | conclusion |
|---|---|---|
| ticket MLX root axis | MLX train telemetry only | CORRECT as research-signal |
| ticket “CPU-torch d_seg authority” (root/nested) | underlying verdict axis `[macOS-CPU advisory torch upstream SegNet]` | OVERCLAIMED, F6 |
| ticket memory `MEASURED` | fp16 mem receipt SHA `91ad0bee...` | CORRECT for memory telemetry only |
| ticket throughput `MEASURED` | WC3 n32 receipt/memo SHA `89810e8e...` | CORRECT at n32 research-signal scope |
| ticket LR `SOURCE-VERIFIED` | external train.sh contains 2e-7 at line 112 | VALUE VERIFIED; line 113 stale; n120 optimality not measured |
| ticket EMA `DERIVED` | LawRef derives 0.999 from supplied phi/U; no K8-to-phi law | PARTIAL/OVERSTATED, F8 |
| ticket terminal commands called “exact n120 CPU facets” | future CPU-torch component verdicts, no contest archive | Acceptable only as exact component computation; adoption claim is not executable, F2 |
| Main finding header/measurement | two n120 advisory verdicts historically emitted n32 scope | CORRECT axis and measurement; `score_claim=false` |
| Main finding “numbers correct” | 120 per-pair entries and equal d_seg values in the receipts | CORRECT for those receipts |
| Main finding cache identity | path, explicitly not content hash | HONEST; it identifies F4 rather than claiming a hash |

No contest-CPU/CUDA score or archive measurement is claimed by these artifacts.

## C3: scope strings and populations

The mechanical census read **14/14 `verdict_scope` assignments** and all **184
candidate lines** matching `verdict_scope`, n-claims, population/pairs terms in
the four pinned files. The 184 include code variables and help text; none was
silently treated as a claim.

| surface | result |
|---|---|
| renderer lines 1548, 2024, 2171, 4305 | CLEAN: each derives n from `len(pair_ids)`; numeric count shares the same list |
| five CoreML blocker scopes and four mem-probe blocker scopes | CLEAN: typed ENVIRONMENT/FORMULATION/INSTANCE scopes; no population assertion |
| CoreML success line 2354 | FINDING: hardcoded n32-or-smaller without an enforced cap |
| MX1T generated prose lines 1764, 1846 | FINDING: n32 is emitted even if runtime checkpoint metadata has another pair population |
| exact historical “two n32 arms” strings at lines 3869 and 4204-4206 | VERIFIED as old v4 authoring/help protocol, not a verdict payload |
| ticket n120 sample | CLEAN: existing receipt has 120 unique IDs and is not `[0..119]`; runtime selection uses the recorded population/seed path |
| sigma `n=5` | CLEAN: five fp16 repeats and five matching checkpoint SHA claims |
| K8 bank `n32` | CLEAN for the historical bank: row has 32 pair IDs and n32 advisory scope |

Targeted tests independently expose the stale test-side copy:

```text
92 passed, 1 failed
expected: n32 arm-selection instrument
actual:   n3 arm-selection instrument
```

## C4: drift twins and rightful owners

| duplicated value/state | copies | rightful owner/import |
|---|---|---|
| fire route | ticket main sequence, controller routes, orchestrator default | typed launch-route record; guard child and FIRE controller |
| terminal step/basis | forecast 3250, fixed terminal argv paths, dynamic terminal receipt | terminal receipt + dynamic facet materializer + basis selector |
| memory projection | per-dtype receipts, singular ticket object, copied argv values | keyed receipt + projection LawRef |
| review progress/status | review list, main sequence, remaining prose, Main-finding header | review ledger + `evaluate_gate` |
| EMA geometry | K, checkpoint cadence, U/horizon, stored phi/decay | actual member-step record + EMA mapping LawRef |
| stop thresholds/units | ticket predicate/prose, orchestrator hardcodes | typed threshold policy with units |
| microbatch 4 | ticket, guard literal, trainer WC2 anchor | shared selection function/registry |
| authority axis | verdict receipt, ticket root prose, nested ticket prose | receipt typed axis record |
| input identity | paths in ticket/guard/receipts versus mutable bytes | shared content-identity manifest |
| subject tree | charter lines 47 and 150 | generated immutable subject manifest |
| live pointer | common contract and hot state | canonical hot-state pointer API |

## C5: receipt and path resolvability

The direct scalar census found **69 distinct path values**: **38 exist** and
**31 are absent planned outputs**. The three annotated prose references (GC21,
NG1, MX1T bank) resolve after parsing their path component. Planned absence is
not by itself a stale evidence claim; F2 is narrower: the fixed terminal paths
are not derivable for every legal event outcome, and no selector output is even
specified.

All five full SHA-256 claims in the ticket match their checkpoint bytes:

```text
run_1..run_5 mlx_stage_step000005.npz
56047d059595b36887a77b2940ebfd15f607413ee82cbd09f2eb946e50eba55c
```

Both short Git commits resolve. Additional independently computed identities:

| artifact | SHA-256 |
|---|---|
| fp16 mem receipt | `91ad0bee7e16827205b5baff82de9087b261aec74df49f01f7e377cb59709ef9` |
| fp32 mem receipt | `12efb06fa41423f77e82beb5935375fb7eaf202b077264d2ee53fc66e5ccccd1` |
| GT cache (943,720,076 B) | `286fe40a2a29aa6950684f43229fce3a4a284ac7ffc65040e7e18953b95787d4` |
| PR130 init | `1549607db224ea2c4681738dbcc80d2ba9dd453de72db1cf60309985d0602eaf` |
| MX1T bank | `b5c340f36a919cd30f5decca24ad16a84e1d179baf9d6a6745154b338d438489` |
| GC21 / NG1 / WC3 | `15f6d2fe...` / `26f76a44...` / `89810e8e...` |

The defect is that the ticket/guard do not store or enforce most of those
identities. No persisted ticket path uses `/tmp`; pytest scratch was transient
and is not cited as evidence.

## C6: independent arithmetic

### Event lattice and marginal bar

```text
pixels = N*H*W = 120*384*512 = 23,592,960
one_sample_flip_S = 100/23,592,960
                  = 4.238552517361111e-6 S
marginal_bar_S_per_step = one_sample_flip_S/50
                        = 8.477105034722223e-8 S/step
one_sample_flip_d_seg = one_sample_flip_S/100
                      = 4.238552517361111e-8 d_seg
```

The ticket values are correct, and `_load_m1_executor_policy` independently
recomputes the one-flip value (`renderer.py:201-220`). Five event-free eval
intervals at cadence 50 equal 250 steps; because the first row is the origin,
the earliest horizon-clear decision needs six unchanged rows. That agrees with
the code's step-distance implementation.

### Microbatch

```text
effective microbatch = max(1,min(explicit 4,total 120)) = 4
chunk count = ceil(120/4) = 30
```

The live fp16 receipt records `source=explicit_cli`, 4, and 30. Current equality
is correct; F12 concerns duplicated ownership/defaults.

### Memory

```text
fp16 total  = 8.493787 + 2.021515 = 10.515302 GiB
fp16 margin = 10.515302 * 1.5     = 15.772953 -> 16 GiB

fp32 total  = 13.707321 + 2.019913 = 15.727234 GiB
fp32 margin = 15.727234 * 1.5      = 23.590851 -> 24 GiB
```

The fp16 projection is numerically correct. The fp32 argv is not.

### EMA and schedule

```text
ticket convention: K*c = 8*250 = 2000
phi = 2000/3250 = 0.6153846153846154
d = 1 - 2/(phi*U) = 1 - 2/2000 = 0.999
actual K8 member endpoint span = (8-1)*250 = 1750 steps
terminal cosine LR = 2e-7 * 0.01 = 2e-9
```

The LawRef calculation and no-jump clamp are correct. The unproven step is the
mapping from the actual K8 member geometry to a 2000-update EMA warmup; that is
F8, not a floating-point error.

### Sigma and seal

The five fp16 losses are byte-for-byte equal at
`0.0003770271432586014`; sample sigma is zero. Their five checkpoint files have
the same full SHA. The fp32 loss is `0.0003566459927242249`, so:

```text
abs training-loss delta
= 0.0003770271432586014 - 0.0003566459927242249
= 0.0000203811505343765
```

Both advisory CPU verdicts report d_seg `0.0010835435655381944`, hence d_seg
delta zero. For the plateau envelope:

```text
5e-7 d_seg/eval * 5 rows = 2.5e-6 d_seg
100 * 2.5e-6             = 2.5e-4 S
```

Current status is 0/3; a read-only orchestrator status confirms the raw-child
FIRE key and holds FIRE behind the pending review gate.

### Current own-vehicle S

Using the hot-state displayed components only as a consistency check:

```text
seg  = 100 * 0.004305419922              = 0.4305419922
pose = sqrt(10 * 0.000716508925)         = 0.084646850207199
rate = 25 * 357837 / 37,545,489          = 0.238268970208378
S                                          = 0.753457812615578
```

This review produced no archive and no new score.

## C7: self-check and apparatus

The four content hashes are a valid immutable key, and commit 393d reproduces
them. The charter's second tree literal, 1381, does not; that is F14. The common
contract's copied “live” pointer also disagrees with the live board it requires
reviewers to read; that is F15. Neither apparatus defect changed the reviewed
bytes, so I completed the audit rather than stopping early.

## Validation and boundaries

Read-only targeted tests:

```text
.venv/bin/python -m pytest -q \
  src/tac/optimization/tests/test_trajectory_stopping.py \
  src/tac/pr130_lift/tests/test_mx1_pr130_lift.py \
  tools/tests/test_mx1_fire_guard.py \
  tools/tests/test_ddm_seal_orchestrator.py \
  experiments/tests/test_ddm_mx1_memory_probe.py

92 passed, 1 failed in 1.99s
```

The one failure is the stale n32 expectation in C3. The test process imported
the MLX package and printed a headless “No Metal device available” atexit
message; it launched no Metal job and performed no real scorer forward.

The required serializer was invoked with this receipt as its sole file,
post-edit SHA custody, `[no-triality] [p0-ledger-ok]`, review override for this
Markdown-only commit, and no co-author trailer. It failed before staging with
Git rc 128: `.git` could not create an object/index temporary file
(`Operation not permitted`) in the managed workspace-write sandbox. The shared
index remained untouched. No commit SHA exists; these reviewed receipt bytes
remain uncommitted on disk for MAIN/operator landing.

Measured/recomputed here: content hashes/sizes, path existence, Git object
resolution, receipt values, population counts, source control flow, threshold
arithmetic, and read-only gate status. Not measured here: no MLX training, no
Metal job, no real SegNet/PoseNet forward, no n600 job, no archive, no
byte-close, no exact contest CPU/CUDA evaluation, and no pointer move.

## Closing disposition

The frozen-literal genus is **not drained**. I completed the bounded sweep over
285/285 semantic numeric tokens, 14/14 verdict-scope assignments, all 184
population/scope candidate lines, all direct ticket paths, and every ticket SHA
claim. I did not stop after the first finding. MAIN should cure the structural
owners above and restart all three independent passes from `0/3`.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
