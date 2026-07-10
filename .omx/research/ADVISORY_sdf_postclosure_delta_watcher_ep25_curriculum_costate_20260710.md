# ADVISORY — SDF post-closure evidence delta: watcher #418, epoch-25 checkpoint, curriculum and costate — 2026-07-10

```yaml
schema: advisory_sdf_postclosure_evidence_delta_v1
observed_at_utc: 2026-07-10T20:58:39Z
lane_id: lane_advisory_codex_v752_v753_v8_fresh_eyes_20260710
lane_scope: research_only
parent_advisory: .omx/research/ADVISORY_sdf_witness_gate_evidence_closure_matrix_20260710.md
parent_advisory_sha256: acd002dda2e07666000ce56ac445855b823873e8cff076465ba2bc44cab4909b
delta_status: ADVISORY_EVIDENCE_DELTA_CLOSED
engineering_gates_passed: []
pointer_delta: 0
execution_authority: none
launches_by_this_unit: 0
evals_by_this_unit: 0
inflations_by_this_unit: 0
dispatches_by_this_unit: 0
harvests_by_this_unit: 0
signals_by_this_unit: 0
processes_stopped_by_this_unit: 0
owned_output: .omx/research/ADVISORY_sdf_postclosure_delta_watcher_ep25_curriculum_costate_20260710.md
```

## 0. Delta result

This is the first append-only evidence delta after the R0–R8 closure matrix. It records four real
changes without laundering any of them into an engineering gate pass:

1. **Public-frontier watcher #418 is operationally corrected.** The watcher now queries the actual
   contest repository and has a fresh PR128 row. It is not structurally sealed: the default endpoint
   lacks a regression test, upstream refresh remains opt-in, and the focused suite has one stale
   literal failure.
2. **PR128's title/body score mismatch is resolved.** The PR now consistently displays `0.187991`.
   It remains open and unratified, and its release tag still points at older source while serving the
   new archive asset.
3. **The live v7.5.2 run produced its first actual rolling checkpoint at epoch 25.** The checkpoint
   contains live weights, EMA, optimizer, RNG and controller state. This advances narrow intra-stage
   resumability evidence, but not preserved stage checkpoints, exact resume reproduction, R3 custody,
   or the six-leaf R8 observer.
4. **Curriculum pool #403 landed as recall/DSL apparatus.** It exposes 36 candidates and folds two
   candidates into DSL factories. It has zero measured candidates and introduces three P0
   candidate-on safety gaps: missing resume protection for hardness/head semantics and a silent
   additive-margin no-op composition.

The costate posterior's `.tier` versus `.status` contradiction remains live. The new run-artifact
contract landed at `df351d308`; it is useful telemetry apparatus but does not create run custody or
checkpoint authority.

Therefore:

```text
ADVISORY DELTA: CLOSED
ENGINEERING R0-R8 PASSES ADDED: NONE
SDF SCORE ROWS ADDED: NONE
VEHICLE PROMOTIONS ADDED: NONE
POINTER DELTA: ZERO
```

## 1. Authority snapshot and ownership boundary

| surface | value at snapshot | authority reading |
|---|---|---|
| branch | `main` | sole source of truth |
| `HEAD` | `df351d3083613057795818ee121c8dbf50e8e1a4` | includes committed curriculum pool #403 and run-artifact contract |
| `origin/main` | `3f349a316b6526fe63b268c9781410bf9382d65d` | local `main` was ahead by concurrent #419/#403/run-artifact landings |
| `CLAUDE.md` | SHA-256 `52405bac18c6227df1d99b597a2f55987614f035e501eb74a9d08be48e1dbdd7` | unchanged from full campaign preflight |
| `AGENTS.md` | SHA-256 `d2bdceb42d394d78bac4f9ddcaa9e0b3758d0be206fea2920784ccdc6f2ec495` | unchanged from full campaign preflight |
| canonical pointer file | SHA-256 `6111c56e68fc51c914bda6cad7b20b499087dc74e9fd41922e7f47fdf572bc90` | refreshed after watcher repair |
| CPU pointer | `0.19108282419209976 [contest-CPU]`, SHA `ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c`, 177,169 B | unchanged, PR110 click-polish lineage, not SDF |
| CUDA pointer | `0.20533002902019143 [contest-CUDA]`, SHA `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`, 186,876 B | different archive/lineage |
| live run | PIDs `88029`/`88030`, alive | observe only; no signal or mutation |
| live run directory | `experiments/results/levelset_v752_baseline_20260710T185913Z` | shared owner custody |

Shared dirt at the snapshot included the state/claim ledgers, concurrent `tools/dashboard_server.py`
work and `paper/__marimo__/`. This unit owns none of it and stages none of it.

The run-artifact slice changed during review, then landed at `df351d3083613057795818ee121c8dbf50e8e1a4`.
Its dedicated evidence below is bound to that commit; later bytes cannot silently upgrade the verdict.

## 2. Executive evidence-delta ledger

| delta | prior disposition | new narrow disposition | whole-gate effect |
|---|---|---|---|
| #418 watcher endpoint | wrong/stale repository | `OPERATIONAL_PASS / STRUCTURAL_TOO_WEAK` | no R0–R8 change |
| PR128 title/body score | `CONTRADICTED` | `PROVED_ALIGNED` at `0.187991` | still external/unratified; no promotion |
| PR128 tag/source/asset | mismatch | mismatch persists | no authority change |
| v7.5.2 rolling checkpoint | no current-run output checkpoint | `PROVED_PARTIAL` epoch-25 intra-stage checkpoint exists | R8.1 becomes `TOO_WEAK_CANDIDATE_ANCESTOR`; R8 whole gate remains open |
| preserved stage checkpoint | absent | still absent | no vehicle stage pass |
| exact resume reproduction | absent | still absent | resumability not fully proved |
| curriculum pool | absent | `PROVED_STATIC_APPARATUS` at commit `fcc2c49cd` | no measurement/promotion authority |
| Hardness/Head DSL mappings | absent | factories exist | candidate-on resume safety contradicted |
| costate posterior field boundary | contradicted | still contradicted | R8/controller remains refuse |
| run-artifact contract | absent | `COMMITTED_STATIC_TELEMETRY_CONTRACT / TOO_WEAK` | no custody/R0/R3 change |
| R0–R7 | open | unchanged | no pass |
| v7.5.3/v8 | design/build only | unchanged | no pass |

## 3. Public-frontier watcher #418

### 3.1 Operational repair

Commit `3fc58a0977a037bfcf2887bcd20e7c6a2f2f9614` changes
`src/tac/canonical_frontier_pointer.py` from `commaai/commavq` to
`commaai/comma_video_compression_challenge`. Commit `797cda465` records the DAG delta. The fresh
canonical snapshot was fetched at `2026-07-10T20:34:42.758917Z` and now contains the actual contest
PR family, including PR128.

The local citation mirrors also moved from stale `0.1919853363` to the current CPU pointer
`0.1910828242`. This is a genuine apparatus correction.

### 3.2 Why the repair is not structurally sealed

| predicate | status | exact evidence / blocker | reopen |
|---|---|---|---|
| production endpoint names correct repository | `PROVED` | `src/tac/canonical_frontier_pointer.py`, lines 477–495 | retain with regression |
| fresh correct-repo snapshot exists | `PROVED` | `.omx/state/canonical_frontier_pointer.json` fetched successfully at 20:34Z | automate freshness |
| default endpoint protected against recurrence | `TOO_WEAK` | #418 added no dedicated regression; existing fetch tests inject `test://` URLs | hermetic test must assert contest endpoint and reject `commavq` |
| upstream freshness automatic | `TOO_WEAK` | `tools/refresh_canonical_frontier.py` requires opt-in `--update-upstream`; local pointer refresh can leave upstream stale | separate upstream-age gate from local-pointer age |
| official evaluation ingested | `MISSING` | watcher stores PR number/title/state/time/URL, not official workflow/custody | bind official result, head, archive and workflow receipt |
| focused test suite clean | `CONTRADICTED` | bounded audit: `29 passed, 1 failed`; stale expected CPU literal `0.1920513169` in `src/tac/tests/test_canonical_frontier_pointer.py`, lines 291–305 | assert canonical pointer consistency rather than historical score literal |

The separate official surface remains `comma.ai/leaderboard`; GitHub PR metadata is public-reference
signal, not score authority.

## 4. PR128 current public authority delta

Fresh GitHub metadata at this snapshot reports:

| field | current public value |
|---|---|
| PR | `commaai/comma_video_compression_challenge#128` |
| state | `OPEN`, not draft, no reviews |
| title | `rhnerv_latent_polish (0.187991)` |
| head | `3eb39cac8261075888b1c562e9d9c2a7f1c7aebf` |
| claimed CPU components | `d_seg=0.00053309`, `d_pose=0.00002937` |
| claimed archive | 176,531 B, SHA-256 `cfd941de10e5c27a5c855f97b0c84e39f6171f23c53c150e4afd90915f41e395` |
| maintainer evidence | only the bot notice that a maintainer must trigger evaluation |

The earlier title/body mismatch is therefore closed. The stronger custody mismatch is not:

- release tag `rhnerv-latent-polish-20260709` still resolves to commit
  `ea478f64f230111e20f78f736673933c15b8ca49` (`0.188532` source);
- the release asset was updated at `2026-07-10T09:01:39Z` to the later `cfd941de...` archive; and
- there is no tag at the current `3eb39cac...` head.

The currently served asset may be internally valid; the public tag cannot identify the source that
rebuilds it. Required authority remains one immutable tuple:

```text
{PR head, source tree, tag, hydrated LFS inputs, packer, member hash,
 archive hash/bytes, report, official workflow, axis}
```

Until a maintainer-triggered official evaluation and coherent source/release tuple exist, PR128 is
`EXTERNAL_UNRATIFIED_HNERV_FAMILY_SIGNAL`. It does not move the local pointer and does not replace the
SDF program.

## 5. Epoch-25 checkpoint evidence

### 5.1 Exact retained bytes at snapshot

| file | bytes | SHA-256 | role |
|---|---:|---|---|
| `levelset_resume_state.npz` | 1,902,570 | `651b84e503323d96430694609292c611a86583537c3bf031d7b6c3bb0d366f3c` | rolling crash-resume state |
| `levelset_witness_ema_mlx.npz` | 482,472 | `076712589886fb7b038d4b45550f3191c719af5d8c07a6e3e4ab066c45461214` | rolling EMA deploy/byte-close checkpoint |

The log emits:

```json
{"stage":"checkpoint","kind":"intra_stage","epoch":25,
 "ema_latest":"levelset_witness_ema_mlx.npz",
 "resume_latest":"levelset_resume_state.npz","has_opt":true}
```

The resume archive loads with 202 keys: 20 live-parameter, 20 EMA, 40 optimizer, 80 fixed config,
six RNG, and additional recent-loss/event/controller state. Direct metadata includes:

```text
__resume_epoch=25
__resume_has_opt=1
__cfg_w_pose=1.0
__cfg_pose_finish_start_epoch=726
__posegate_fired_epoch=-1
__cfg_git_sha=6a34b66d6966546c4a3d677dc2f70879cd54a342
__cfg_git_dirty=1
```

The resume-registry manifest names Muon, lane-band, chroma-boundary, temporal-screw, pose gate, RNG,
tau-advance, event curriculum and birth-completion state. The trainer writes both rolling files via
temporary file plus `os.replace`.

### 5.2 Narrow passage and exact limits

| predicate | status | reason |
|---|---|---|
| actual periodic checkpoint exists | `PROVED` | files, hashes, epoch and optimizer-bearing log row exist |
| checkpoint structurally loads | `PROVED` | NPZ members and scalar metadata were read without mutation |
| live/EMA/optimizer/RNG/controller state present | `PROVED_PARTIAL` | named groups and resume manifest exist |
| atomic write implementation | `PROVED_SOURCE` | `_atomic_savez` uses same-filesystem temp plus `os.replace` |
| immutable preserved checkpoint | `MISSING` | filenames are rolling and will be overwritten at the next periodic save |
| stage-encoded checkpoint | `MISSING` | no curriculum transition has occurred; no `levelset_ckpt_*_ep*.npz` or stage resume exists |
| exact resume continuation | `MISSING` | no independent load-and-continue-to-matching-next-checkpoint receipt |
| complete custody | `MISSING` | `git_dirty=1` without a file-level dirty manifest; no DSL manifest or receiver fingerprint |
| R8 common ancestor | `TOO_WEAK` | pre-event candidate bytes exist, but are rolling, uncustodied and not bound to a six-leaf branch manifest |

This changes the old “no current-run checkpoint exists” statement. It does **not** satisfy R0, R3,
R8 experiment completion, or a vehicle stage exit.

### 5.3 Live pose-gate evidence

At epoch 22 the live run reported:

```text
classification=DEGENERATE_GUARD_TRIPPED
fired=false
should_ship_banked_r1=true
axis=[macOS-MLX advisory] NON-PROMOTABLE
```

The epoch-25 checkpoint confirms `__posegate_fired_epoch=-1`. This is a provisional recommendation,
not an executed banked-R1 graft. No base/graft/rollback archives or joint two-frame full-score receipt
exists. R2 remains contradicted/open.

## 6. Curriculum pool #403

Commit `fcc2c49cdd036738618b3bd38dc720ba0ebe7811` lands the candidate-pool module,
test, two DSL factories, costate-digest readout and DAG feed. At the snapshot its read-only summary is:

| pool state | count |
|---|---:|
| total | 36 |
| owed a fire | 28 |
| built-never-fired | 14 |
| needs-build | 12 |
| reformulation-queue | 2 |
| armed | 5 |
| retired-with-reason | 3 |
| measured | **0** |

This improves recall: curriculum candidates in loss, data order, initialization, preconditioning,
averaging, solve interleave and vehicle state can no longer remain only in chat. It is a SENSE/DSL
apparatus delta, not empirical evidence. No launch, trainer or harvest consumer currently records a
candidate firing/outcome into the pool; seeded statuses are not derived from a run manifest.

### 6.1 P0 candidate-on safety contradictions

| finding | status | consequence | required repair |
|---|---|---|---|
| `HardnessOversample` semantics absent from checkpoint provenance and F2 divergence guard | `CONTRADICTED` | on->off or changed oversample/weight/source/power/band resume can silently alter order and optimizer-step count | persist all five fields and fail closed on divergence |
| `HeadGeometry` head/additive-margin absent from checkpoint/resume guard | `CONTRADICTED` | ETF/frozen `out_sdf` can silently resume as trainable softmax | persist head semantics and validate restored topology/trainability |
| `HeadGeometry(head="additive-margin")` omits required positive margin-field-head weight | `CONTRADICTED` | the DSL factory can compile a documented but inert arm | composition validator must require the effective margin lever/weight |
| pool reader validates less than writer | `TOO_WEAK` | a manually appended `measured` row lacking `verdict_ref`/source/DSL disposition can be surfaced | apply the full record validator on read and quarantine invalid rows |

The live baseline carries none of the new hardness/head flags, so these defects do not invalidate the
epoch-25 checkpoint. They block any future candidate-on launch from claiming the binding resume
contract.

### 6.2 Static launch match does not close sealed custody

The live argv contains the nine levers currently expected by `witness_autoconfig.py`: tau, tail,
ladder, Polyak, area, birth-completion, dash-comb, temporal screw and pose gate. That lower-level
static composition match does not override the higher sealed-spec conflicts already recorded:
explicit `--epochs 3000`, launch-1 P0-force composition, no retained DSL program manifest, and no
baseline activation-ledger rows. The run remains advisory and observe-only.

## 7. Costate/posterior delta

The #403 digest line is SENSE-only. It reads the pool and preserves Operator-GO for live config and
stop actions. It does not repair the estimator-to-posterior boundary.

| predicate | status | exact defect | reopen |
|---|---|---|---|
| real `CostateEstimate` reaches posterior | `CONTRADICTED` | estimator exposes `.status`; `record_run_costates()` reads `.tier` and drops the object | real-object regression plus one completed byte-close posterior row |
| regression covers real boundary | `CONTRADICTED` | existing fake test object supplies `.tier`, masking the production mismatch | construct `CostateEstimate(status="MEASURED")` in test |
| `PARTIAL` prior semantics | `TOO_WEAK` | PARTIAL is recordable although its chain to S is unmeasured | specify/filter admissibility |
| posterior preserves evidence tier | `TOO_WEAK` | `all_posteriors()` discards tier; dashboard shows only count | preserve tier/provenance through read and display |
| current shadow/posterior evidence | `MISSING` | no `costate_shadow.jsonl`; no `.omx/state/costate_posterior.jsonl` | wait for a real valid shadow row after source repair |

The primary cross-run learning blocker remains the field mismatch. R8/controller advice remains
`REFUSE`.

## 8. Run-artifact contract landing

The following bytes landed together at commit `df351d3083613057795818ee121c8dbf50e8e1a4`:

| file | SHA-256 | state |
|---|---|---|
| `src/tac/witness_run_artifacts.py` | `99a3728c828037cbf738da6d5b5d0d64585bb63c2f02a789588a65197ae46b4f` | tracked at `df351d308` |
| `src/tac/tests/test_witness_run_artifacts_contract.py` | `1f80a41374c17ccac69ba4c4eb901ab94ef5941c1650963a6d257e5e4730b277` | tracked at `df351d308` |
| `tools/witness_checkin.py` | `54442785c3d87913849b41e95397bb1d8fd1c93d75e1b76db2c190574261e32f` | tracked at `df351d308` |
| `src/tac/checkpoint_retention.py` | `b809cee5568445e930a69b3ebc1bebc5e6fcc209e16d75fb0e4b1640641c235c` | tracked at `df351d308` |
| `src/tac/witness_control/shadow_controller.py` | `d4c9cf7995cd2377f5ff1aa50c65828afa51078f620449b6e033d4846ecf2f21` | tracked at `df351d308` |

Positive narrow changes:

- one canonical static-name module exists;
- checkin, checkpoint retention and shadow controller consume some names;
- the proposed test covers the pre-first-checkpoint false-RED case and excludes `observer.log`; and
- `EMA_BEST_NPZ` is declared.

The decisive patterns were re-read after landing: the broad `levelset_*` glob, arbitrary `*.log`
acceptance, absent dynamic-stage patterns and unproved `LIVE_NPZ` declaration all remained present.

It is still telemetry-only:

| defect | status | effect |
|---|---|---|
| `RUN_DIR_GLOB="levelset_*"` includes packets, validation/probe dirs and other non-runs | `CONTRADICTED` | fallback discovery can select unrelated historical state |
| every `*.log` except `observer.log` counts fresh | `CONTRADICTED` | unrelated log can false-GREEN a hung trainer |
| no durable-daemon binding or parseable trainer-row validation | `MISSING` | mtime is not process/telemetry identity |
| dynamic stage EMA/resume/Polyak patterns absent | `CONTRADICTED` | contract/test cannot prove mandatory preserved stage ladder |
| drift regex cannot see brace-form dynamic stage names | `CONTRADICTED` | proposed self-protection misses the most important artifacts |
| `LIVE_NPZ` declared without discovered producer | `TOO_WEAK` | contract claims an unproved output |
| no run/config/source/upstream/GT/checkpoint/archive/receiver manifest | `MISSING` | no custody or handoff authority |
| dummy resume mtime only; no registry/load validation | `MISSING` | no deterministic resume proof |
| migration claim says about 30 consumers; six consumers plus the test currently import the contract | `TOO_WEAK` | anti-hardcoding baseline tolerates most old couplings rather than completing migration |

No test was run by this advisory lane. The slice is now immutable source apparatus, but its targeted
semantic gaps remain. It cannot close R0, R3, stage preservation, v7.5.3/v8 handoff or promotion.

## 9. R0–R8 delta matrix

The parent matrix remains authoritative except for the narrow deltas below.

| gate | parent whole-gate status | post-delta whole-gate status | exact change |
|---|---|---|---|
| R0 custody/identity | `CONTRADICTED + MISSING + TOO_WEAK` | unchanged | training checkpoint is not a legal archive/custody receipt; run-artifact contract has no manifest |
| R1 receiver bijection | `TOO_WEAK + MISSING` | unchanged | no actual optional-family signed archive effect |
| R2 joint finisher | `CONTRADICTED + MISSING` | unchanged | `axis_s_component`, aggregate-filled Pose and frame1-only code measure remain in source |
| R3 n600 SDF center | `CONTRADICTED + MISSING + BLOCKED` | unchanged | epoch-25 checkpoint is not an n600 LVLS1 archive/evaluator receipt |
| R4 five-state cell | `NOT_AUTHORIZED + MISSING + BLOCKED` | unchanged | no five endpoints |
| R5 24-state RQTD | `NOT_AUTHORIZED + MISSING + BLOCKED` | unchanged | no state dictionary |
| R6 predictor/Hodge | `NOT_AUTHORIZED + MISSING + TOO_WEAK` | unchanged | no exact complex/held-out chords |
| R7 topology atoms | `MISSING + TOO_WEAK + BLOCKED` | unchanged | no typed receiver-realized atom |
| R8 Muon observer | `CONTRADICTED + MISSING + NOT_AUTHORIZED` | unchanged as a whole | R8.1 advances from absent to a rolling `TOO_WEAK_CANDIDATE_ANCESTOR`; no preservation, branches, collision policy, held-out fit, seed or posterior |

No engineering gate advances to pass.

## 10. Vehicle and curriculum implications

### 10.1 v7.5.2

Current literal state:

- process alive and training through epoch 25;
- first atomic rolling EMA/resume checkpoint exists;
- no stage-transition checkpoint yet;
- pose gate provisionally recommends banked R1 but has not fired/grafted it;
- current autoconfig lever set is present, but immutable DSL/config custody is absent; and
- sealed-spec argv contradictions remain unresolved.

The correct vehicle disposition remains `OBSERVE_ONLY / NONPROMOTABLE`. The checkpoint reduces crash
loss risk but does not justify a stop, resume experiment, branch, score claim or promotion from this
lane.

### 10.2 v7.5.3

#403 gives future curricula a better candidate inventory, but a candidate-on v7.5.3 run must refuse
until hardness/head semantics are resume-protected and the additive-margin prerequisite is enforced.
The existing appearance blockers remain: actual-archive A1/A2/A3 controls, exact-D through-R home
law, matched bytes, R0–R3 and full-score-safe finishing.

### 10.3 v8

No v8 entry predicate changes. v8 still waits for a registered v7.5 target-trajectory miss, P-C
geometry, risk closure, seal/n600 proof, class-isolation, actual increment-1b receiver consumption and
carrier byte-close/resume. Curriculum recall is method transfer only; no numerical effect transfers.

## 11. Exact next exit artifacts

### 11.1 Apparatus fixes exposed by this delta

| priority | exact artifact | exit criterion | owner disposition |
|---|---|---|---|
| A0 | watcher default-endpoint/freshness regression | contest API asserted; `commavq` rejected; local refresh cannot renew upstream age | #418 follow-up unassigned |
| A1 | watcher test literal repair | focused suite clean without hardcoding a historical frontier score | #418 follow-up unassigned |
| A2 | costate real-object boundary repair | `CostateEstimate(status="MEASURED")` appends, reads back with tier/provenance, one valid byte-close row exists | #247 owner unregistered |
| A3 | hardness resume closure | all five semantic fields persisted and divergence-refused; on->resume->next checkpoint reproducible | #403 follow-up |
| A4 | head resume/composition closure | head/additive margin persisted; ETF trainability topology restored; inert AM arm refused | #403/#218 follow-up |
| A5 | candidate-pool read integrity | read path applies full validator and quarantines incomplete `measured` rows | #403 follow-up |
| A6 | run-artifact contract hardening | precise run identity, trainer-owned log proof, dynamic stage patterns, registry/custody schema, focused tests pass | `df351d308` follow-up owner unassigned |
| A7 | checkpoint preservation/replay | stage-encoded copy retained; exact resume reaches matching next checkpoint under identical manifest | live-run owner, later authority |

### 11.2 Frontier dependency chain remains

The campaign's shortest score-authority path is unchanged:

```text
R1 actual-archive receiver proof
  -> R2 joint Seg/Pose finisher repair
  -> R3 retained contest-CPU n600 SDF center
  -> R4 five-state safety cell
  -> R5 24-state quotient dictionary
  -> R6 held-out predictor/Hodge
  -> R7 legal topology atoms
  -> R8 matched event/costate evidence
  -> exact v7.5.3 vehicle
  -> v8 only after a registered v7.5 miss
  -> same-byte CPU/CUDA promotion.
```

The apparatus fixes above prevent signal loss and false authority. They do not substitute for this
chain.

## 12. Negative and stopping posture

No broad negative is added.

| observation | narrowest status | forbidden inference |
|---|---|---|
| watcher test fails on stale literal | `APPARATUS_REGRESSION` | not a score or SDF negative |
| PR128 open/unratified/tag-incoherent | `PUBLIC_CUSTODY_BLOCKER` | not evidence its bytes are invalid or that SDF should stop |
| pose gate degenerates at epoch 22 | `INSTANCE_ADVISORY_SIGNAL` | not a fired fallback or vehicle negative |
| candidate pool has zero measured rows | `MEASUREMENT_BACKLOG` | not evidence candidates are ineffective |
| candidate-on resume gaps | `IMPLEMENTATION_BLOCKER` | not a curriculum-family negative |
| costate field mismatch | `IMPLEMENTATION_CONTRADICTION` | not a control-theory negative |
| run-artifact false-GREEN/broad glob | `APPARATUS_IMPLEMENTATION_BLOCKER` | not run failure |

The campaign stopping exit remains unmet. No certified lower bound, exhaustive legal-family proof or
decision-materiality closure exists. The live process remains observe-only and must not be stopped by
this advisory lane.

## 13. Literal launch/process dispositions

| action/surface | disposition by this unit |
|---|---|
| training launch | **NONE** |
| branch/resume | **NONE** |
| inflate/materialize | **NONE** |
| evaluator | **NONE** |
| dispatch/harvest | **NONE** |
| pointer refresh/move | **NONE; DELTA ZERO** |
| process signal/termination | **NONE** |
| PIDs 88029/88030 | **PRESERVED; OBSERVE ONLY** |
| PR128 asset mutation/submission action | **NONE** |
| shared source/state WIP | **PRESERVED; NOT STAGED** |
| owned mutation | **THIS NEW ADVISORY ONLY** |

## 14. Advisory closure criteria

This delta earns `ADVISORY_EVIDENCE_DELTA_CLOSED` only if mechanical validation confirms:

- every new positive is scoped to source, telemetry, checkpoint or public-reference authority;
- every changed parent disposition is explicit;
- all local paths exist at validation time, except deliberately external/missing artifacts;
- PR128 metadata is linked to the exact current head/tag/asset tuple;
- checkpoint hashes and structural counts match the frozen files;
- run-artifact hashes are bound to commit `df351d308`; later drift cannot silently upgrade the verdict;
- no placeholder marker remains;
- no source/state/shared file is staged; and
- the serializer commits only this new Markdown file.

This is not `ENGINEERING_GATE_PASS`.

## 15. Triality

### DSL leg

The delta records that `HardnessOversample` and `HeadGeometry` factories now exist while their
resume/precondition contracts do not. A DSL factory is not an effective arm unless its trainer effect,
resume state and composition preconditions all close.

### DAG leg

```text
#418 wrong repo
  -> correct-repo fresh snapshot
  -> endpoint/freshness regression still owed
  -> PR128 remains external until official custody

live v7.5.2
  -> epoch-25 rolling checkpoint
  -> preserved stage checkpoint owed
  -> exact resume reproduction owed
  -> R8 branches still owed

#403 pool
  -> candidate recall + DSL factories
  -> resume/precondition/read-integrity repairs
  -> matched n600 candidate-on/off only after those repairs

costate estimator.status
  -X-> posterior.tier reader
  -> cross-run posterior remains empty/refused
```

### Equation leg

No score equation changes. Exact authority remains

\[
\Delta S=100\Delta d_{seg}
+\sqrt{10d'_{pose}}-\sqrt{10d_{pose}}
+\frac{25\Delta B}{37{,}545{,}489}.
\]

The new checkpoint affects state persistence, not this objective. A future candidate-on resume must
satisfy the state identity law

\[
H(\text{checkpoint state},\text{DSL semantics},\text{receiver},\text{RNG},\text{next state})=0
\]

under a declared exact comparison, not mere file presence.

## 16. STORES CONSULTED and pointer honesty

### STORES CONSULTED

- full campaign preflight anchors and unchanged `CLAUDE.md`/`AGENTS.md` hashes;
- current top-10 Pact Claude memory, directives, lane registry, subagent ownership and active claims;
- `.omx/research/ADVISORY_sdf_witness_gate_evidence_closure_matrix_20260710.md`;
- commits `3fc58a097`, `797cda465`, `fcc2c49cd` and their source/tests/DAG surfaces;
- `.omx/state/canonical_frontier_pointer.json`, `reports/latest.md`, current-focus/next-experiment mirrors;
- fresh GitHub PR128 metadata, release object, tag ref and asset digest;
- live v7.5.2 process/log/checkpoint bytes and trainer checkpoint source;
- curriculum pool/DSL/test/digest sources and read-only pool summary;
- costate estimator/posterior/test/byte-close call path and dashboard authority surfaces;
- landed run-artifact/checkin/retention/shadow-controller slice and contract test; and
- four calibrated read-only audits: watcher, curriculum/live run, costate/R8, and run artifacts.

### HISTORICAL_PROVENANCE / pointer delta

Derived against local `main` at `df351d3083613057795818ee121c8dbf50e8e1a4`. Concurrent work may
advance `main`; the hash is an evidence anchor, not a fixed-HEAD claim. The run-artifact slice is
bound to its landing commit and content hashes.

The canonical CPU and CUDA pointers are unchanged. This unit caused exactly zero archive, run,
checkpoint, source, state, dispatch, evaluator, process or pointer mutation. Its sole owned output is
this advisory document.
