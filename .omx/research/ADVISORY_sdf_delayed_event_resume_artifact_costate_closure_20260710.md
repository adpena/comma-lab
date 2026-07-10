# ADVISORY — SDF delayed-event reset/resume, artifact semantics, and typed-costate closure — 2026-07-10

```yaml
schema: advisory_sdf_delayed_event_artifact_costate_closure_v1
observed_at_utc: 2026-07-10T21:31:31Z
final_observed_at_utc: 2026-07-10T21:39:09Z
lane_id: lane_advisory_codex_v752_v753_v8_fresh_eyes_20260710
lane_scope: research_only
parent_advisory: .omx/research/ADVISORY_sdf_ep31_moddim_hybrid_costate_delta_20260710.md
parent_advisory_sha256: 30608f4fd43e6d2f3a00bd18e16225a63cd687d6660c3f72b8dd592a82b170c8
delta_status: ADVISORY_EVIDENCE_DELTA_CLOSED
engineering_status: OPEN_BLOCKED
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
owned_output: .omx/research/ADVISORY_sdf_delayed_event_resume_artifact_costate_closure_20260710.md
```

## 0. Result

This append-only delta closes the advisory analysis of five defects exposed after the epoch-31
record. It does not close the engineering campaign.

1. The lane-band event fired early at epoch 31, but the source computes its AdamW treatment boundary
   from the fixed epoch-500 cap **before** the event gate updates. The event therefore changed the
   render target and cleared the spike history while skipping the configured optimizer-moment reset,
   eight-epoch LR rewarmup anchor, and event-boundary checkpoint. The same structural miss applies to
   early chroma and temporal-screw events.
2. The asynchronous event state is not transactionally resumable. Its payload and source epoch are
   separate unlocked fields; the pending message/snapshot is not preserved for this launch; and the
   last resume checkpoint predates the event. Event time can therefore depend on wall-clock arrival,
   and crash/resume equivalence is unproved.
3. The epoch-25 `EMA_BEST` artifact is not merely partial; its provenance metadata contradicts the
   run. It records `git_sha=unknown`, `upstream_sha=unknown`, and `git_dirty=0`, while the rolling
   checkpoint records the actual dirty source SHA, upstream SHA, and `git_dirty=1`.
4. Costate semantics are wider than the known `.status`/`.tier` persistence bug. A transition object
   labeled `dS` computes only `100*delta_d_seg`; trajectory `dS/depoch` can omit missing Pose/rate and
   remain `MEASURED`; and rollback selects by d_seg before comparing full score.
5. Commits `a0362725` and `feccfa39` genuinely migrated twenty-two consumers onto canonical
   run-artifact names; `3722879f` canonicalized the locked JSONL append helper; and `9c66ae72` added
   a valuable hardcode/duplication audit. These improve apparatus hygiene. They do not define
   correct artifact roles, schemas, provenance, event transactions, rollback semantics, or score
   authority. No engineering gate or R0–R8 gate moves.

```text
ADVISORY DELTA: CLOSED
ENGINEERING R0-R8 PASSES ADDED: NONE
NEW AUTHORITATIVE SDF SCORE ROWS: NONE
CURRENT LIVE RUN: LIVE / OBSERVE-ONLY / TREATMENT ATTRIBUTION OPEN
V7.5.3: DESIGN/BUILD-ONLY
V8: HOLD TRAINING EVENT
PR128: HNERV-FAMILY / EXTERNAL UNRATIFIED / NO DELTA
POINTER DELTA: ZERO
```

## 1. Authority snapshot and ownership boundary

### 1.1 Repository and canonical pointer

| surface | observed value | authority reading |
|---|---|---|
| branch | `main` | sole source of truth |
| `HEAD` | `feccfa39dcd59563c2e7bba6167cee21947634ea` | includes both filename migrations, JSONL helper extraction, and hygiene audit |
| first filename migration | `a0362725fd359fc5d4f709aa12e02263c46d6d90` | ten-consumer path hygiene; no claimed behavior change |
| JSONL helper extraction | `3722879fca72c9631e441975099c8de376d2da32` | one locked append implementation plus dedicated concurrency tests |
| second filename migration | `feccfa39dcd59563c2e7bba6167cee21947634ea` | twelve more consumers and an eight-file ratchet shrink; no claimed behavior change |
| `origin/main` | `feccfa39dcd59563c2e7bba6167cee21947634ea` | matched after concurrent push; no push by this unit |
| `CLAUDE.md` | SHA-256 `52405bac18c6227df1d99b597a2f55987614f035e501eb74a9d08be48e1dbdd7` | unchanged from full campaign preflight |
| `AGENTS.md` | SHA-256 `d2bdceb42d394d78bac4f9ddcaa9e0b3758d0be206fea2920784ccdc6f2ec495` | unchanged from full campaign preflight |
| parent advisory | SHA-256 `30608f4fd43e6d2f3a00bd18e16225a63cd687d6660c3f72b8dd592a82b170c8` | exact parent snapshot |
| hygiene audit | SHA-256 `2378098428976db1a1d7082c218ac8a23186ce02ae59669d55f84537f4efbd09` | committed read-only apparatus inventory |
| canonical pointer file | SHA-256 `6111c56e68fc51c914bda6cad7b20b499087dc74e9fd41922e7f47fdf572bc90` | unchanged |
| CPU pointer | `0.19108282419209976 [contest-CPU]`, archive SHA `ad02b012...d079c`, 177,169 B | PR110/click-polish lineage, not SDF |
| CUDA pointer | `0.20533002902019143 [contest-CUDA]`, archive SHA `9cb989ce...7cf4` | different archive/lineage |

The filename and JSONL-helper work that was dirty at 21:35:32Z landed concurrently as the two commits
above. Shared dirt at the final snapshot consists only of five `.omx/state/` ledgers and
`paper/__marimo__/`; it is concurrent partner work. This unit neither edits, stages, reverts,
absorbs, nor normalizes it. Its only owned mutation is this advisory.

### 1.2 Live process and artifact custody

| surface | value | disposition |
|---|---|---|
| launcher | PID `88029`, alive | do not signal |
| trainer | PID `88030`, alive | do not signal |
| run directory | `experiments/results/levelset_v752_baseline_20260710T185913Z` | shared owner custody |
| progress | loss telemetry through epoch 35 | advisory training telemetry only |
| latest full n600 telemetry | epoch 25 async CPU row, explicit axis absent | nonpromotable |
| latest pose gate | epoch 34 `DEGENERATE_GUARD_TRIPPED` | banked-R1 recommendation, not artifact selection |
| first live event | lane band at epoch 31, sensor epoch 25, lag 6 | real actuation; attribution open |
| latest rolling checkpoint | epoch 25 | predates event |
| latest retained deploy candidate | epoch-25 `EMA_BEST` | d_seg-only and provenance-contradicted |

There is still no complete SDF archive, raw output, LVLS1 center, immutable run manifest,
stage-encoded event checkpoint, contest-axis receipt, or matched no-event branch in the run directory.

The click-polish claim remains sister-owned. PR128 remains open, unreviewed, unmerged, and unchanged:
176,531 B archive SHA `cfd941de...e395`, current head `3eb39cac...`, stale tag target
`ea478f64...`. It remains an HNeRV-family payload-polish control, not a new family.

## 2. Executive evidence-delta ledger

| delta | positive evidence | contradiction / missing authority | whole-gate effect |
|---|---|---|---|
| epoch-31 lane event | source epoch, arrival lag and fire telemetry are explicit | intended optimizer treatment and boundary checkpoint skipped | R8 worsens/open |
| asynchronous sensor | immutable epoch-25 score snapshot existed in worker | queue/message/hash/arrival state not persisted | R0/R8 open |
| `EMA_BEST` bytes | exact deploy bytes retained and atomically written | false/unknown provenance; d_seg-only order; no full score | R0 contradicted |
| run-artifact migration | twenty-two consumers import one filename contract | roles/schemas/selection/resume semantics unchanged | apparatus-only; not an engineering/R pass |
| anti-hardcoding ratchet | baseline shrank 33 to 25; no new or stale entries | remaining raw-text exemptions permit live reintroduction and conflate docs/code | improved but structurally too weak |
| costate filename migration | digest reads canonical costate filename | costate equations and `.tier` bug unchanged | no controller pass |
| hygiene audit | high-EV duplication classes durably enumerated | `_proven_base()` and DSL baseline still diverge | config authority open |
| R1–R7 | no new endpoint evidence | all prior blockers persist | no pass |

## 3. Early event transition is half-wired

### 3.1 Declared and realized reset words

The launch requests:

```text
--stage-transition-rewarmup-epochs 8
--stage-transition-reset-moments
--lane-band-start-event lane_nucleus
--lane-band-start-epoch 500
```

The trainer computes `_stage_boundary_now` before event gates update. Its lane-band term becomes true
only when the fixed cap is reached. The event gate later fired at epoch 31, long before cap 500, so
its `just_fired` signal never entered the boundary handler.

The declared reset word is:

```text
R_intended = R_moments o R_LR_anchor o R_spike o R_band.
```

The realized reset word was only:

```text
R_realized = R_spike o R_band.
```

Runtime evidence agrees with source semantics. The log contains `start_event_fired` and
`lane_render_band_engage`, followed by ordinary loss rows. It contains no
`stage_transition_reset_moments`, LR-rewarm boundary record, or event checkpoint.

This is not a cosmetic omission. AdamW moments encode the old render-target landscape. The event
changes that landscape while carrying stale moments through it, despite a launch that explicitly
requested reset and rewarm treatment. Any later change cannot be attributed to “lane band under the
declared treatment.” Its correct verdict scope is:

```text
EARLY_EVENT_TREATMENT_CONFOUNDED
```

### 3.2 Family-wide spread

The same code pattern governs early annulus-triggered chroma and temporal-screw engagement. Their
boundary predicates also use fixed start epochs before event gates update. Until the event's actual
`just_fired` value enters one shared boundary transaction, all three early-event families can silently
skip optimizer treatment.

The fix cannot be a lane-only special case. One event transaction must consume every lever's typed
fire record and produce the reset word, checkpoint pair and resume record.

### 3.3 Checkpoint contract miss

The stage-checkpoint block recognizes segmentation-form events, tau-octave transitions, Muon, tail
and fixed stage boundaries. It does not recognize this early lever event. The latest rolling resume
file remains epoch 25 and contains:

```text
__lbg_fired_epoch = -1
__lbg_fired_by = ""
```

A crash after epoch 31 but before the next rolling save therefore restarts from state that predates
the message arrival and event. It has neither the completed epoch-25 message nor the epoch-31 latch.
The resumed fire epoch and optimizer path are not proved equal and can diverge.

This directly violates the “lose at most one intra-stage interval” spirit for the controller state:
the weights may resume, but the asynchronous decision that changed the treatment is not in the same
transaction.

## 4. Delayed asynchronous guard is not a Markov state

### 4.1 Proven storage/race gap

The async worker writes lane sensor payload and sensor epoch into two separate `_wire_sense` entries.
The training thread later reads them separately without one shared lock or immutable message object.
A payload/epoch mismatch is structurally possible between those operations.

`EventBackstopGate` persists fired epoch/by and optionally the sensor epoch. It does not persist:

- payload hash and complete payload;
- snapshot/checkpoint hash that generated it;
- schedule time and arrival time;
- uncertainty/noise floor;
- pending-message queue;
- worker/runtime identity; or
- whether an unseen result remains in flight.

The current launch does not enable the closed-loop sidecar that preserves pending snapshots.
`sensor_async_pending` is telemetry, not an actuation refusal rule.

### 4.2 Correct augmented state

Let the evaluator sense checkpoint state `x_tau` at epoch `tau`:

```text
y_tau = H_rho(x_tau),
a = tau + D_tau,
```

where `D_tau` is the asynchronous delay and `a` is the arrival/actuation epoch. The Markov state is
not just current weights and gate counters. It must be augmented:

```text
X_k = {x_k, optimizer_k, RNG_k, mode_k, Q_k, I_k, rho},
```

where `Q_k` is the pending-message queue and `I_k` is a set of immutable timestamped observations.

The epoch-31 event is a categorical **message-arrival/reset event**, not a transverse crossing of a
current-state guard `h(x_k)=0`. Ordinary current-state saltation is therefore the wrong derivative.
The jump derivative must include the stored sense-state/message edge and the discrete reset word.

This is consistent with primary control research that models asynchronous event-triggered samplers as
hybrid subsystems rather than assuming simultaneous measurement and control. See
[Williams, Chapman and Manzie](https://arxiv.org/abs/2211.13846) and the time-delay hybrid treatment
of [Zhang and Gharesifard](https://arxiv.org/abs/1912.02396). These papers motivate the state model;
they do not prove this implementation correct.

### 4.3 Separate the two estimands

One branch cannot identify both treatment and delay policy.

1. **Arrival-time treatment effect:** start immediately before arrival with the same persisted
   message; compare the complete event reset against a no-event control.
2. **Sense-time delay validity:** start from the sensing checkpoint; replay the observation and delay
   policy while holding the treatment rule fixed.

Without both, a result conflates the lane-band treatment, stale sensor, arrival epoch, optimizer
reset omission and intervening training trajectory.

### 4.4 Staleness certificate

For lane nucleus define guard slack at sensing time:

```text
m_tau = min(part_frac_tau - part_frac_min,
            within_flip_max - within_flip_tau).
```

A delayed result is safe to fire only if:

```text
LCB(m_tau) - L_hat * (a - tau) - epsilon_replay > 0,
```

where `L_hat` is a measured bound on guard-slack drift and `epsilon_replay` is the deterministic
replay floor. Otherwise the exact disposition is:

```text
STALE_GUARD_UNIDENTIFIABLE
```

No guessed Lipschitz bound is admissible. It must come from matched trajectories.

## 5. `EMA_BEST` is not a custody or full-score object

### 5.1 Exact metadata contradiction

| field | epoch-25 `EMA_BEST` | epoch-25 rolling resume |
|---|---|---|
| SHA-256 | `440b7e7b8fcd4003b0ae8f333da3932e172442068fc251d51f9dfa424dcd9bfe` | `651b84e503323d96430694609292c611a86583537c3bf031d7b6c3bb0d366f3c` |
| epoch | 25 | 25 |
| git SHA | `unknown` | `6a34b66d6966546c4a3d677dc2f70879cd54a342` |
| upstream SHA | `unknown` | `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41` |
| git dirty | `0` | `1` |
| optimizer/RNG/controller | absent | present |

The deploy builder defaults omitted provenance to unknown/not-dirty, and `_maybe_preserve_best()`
does not pass the run provenance supplied by the ordinary checkpoint path. `git_dirty=0` is therefore
not merely absent evidence; it contradicts the source state recorded by the same run.

### 5.2 Partial order mislabeled as best score

`levelset_best.json` carries only:

```text
{d_seg, epoch, path, ts}
```

The producer accepts strictly lower d_seg. It omits Pose, bytes, complete score, axis, receiver and
archive identity. The canonical artifact contract nevertheless comments `EMA_BEST_NPZ` as a
“best-scoring EMA checkpoint,” and migrated consumers now share that ambiguous role label.

The complete replacement order is:

```text
Delta S = 100*Delta d_seg
        + sqrt(10*d_pose_candidate) - sqrt(10*d_pose_parent)
        + 25*Delta bytes/37_545_489.
```

D_seg ordering is valid only after exact Pose and byte invariance. Incomplete async candidates must
remain `PENDING_UNORDERED`; completion order cannot initialize a full-score champion.

### 5.3 Required bank

Retain a Pareto bank over `(d_seg, d_pose, bytes)` and select by exact complete score only after all
components close. Every selected point requires both:

- deploy checkpoint with truthful source/receiver/config provenance; and
- complete resume checkpoint with live/EMA/optimizer/RNG/controller/pending-message state.

This is the minimum artifact pair for exact rollback and branch attribution.

## 6. Costate taxonomy and semantic contradictions

### 6.1 Source findings

| object | claimed meaning | actual computation | disposition |
|---|---|---|---|
| `transition_jump_costate()` | measured stage `Delta S` | `100*Delta d_seg` only | `MISLABELED_PARTIAL` |
| `chain_ds_depoch()` | measured trajectory `dS/depoch` | silently omits Pose/rate slopes when unavailable | `PARTIAL_MISLABELED_MEASURED` |
| `rollback_gain()` | recoverable full-score gain | chooses best row by d_seg, then compares full S | `ORDERING_MISMATCH` |
| `record_run_costates()` | persist identifiable estimates | reads `.tier`; production object exposes `.status` | `WRITE_PATH_BROKEN` |
| posterior | comparable cross-run costates | lacks chart/event/receiver/lag keys and evidence class | `NONCOMPARABLE_POOL` |

The transition function's docstring even says “plus pose channel where present,” but the code never
adds it. This is a direct source contradiction.

The focused suite is green—48 tests across costate, run-artifact contract and campaign surfaces—yet
does not cover real-object persistence, full transition score, early-event boundary treatment,
provenance truth or exact rollback. Green here is narrower than the exit.

### 6.2 Keep four mathematical objects distinct

1. **Terminal score covector:** exact local score partials at a complete endpoint.
2. **Trajectory slope:** empirical change per epoch inside one fixed mode and receiver chart.
3. **Hybrid jump sensitivity:** derivative or matched edge across one typed reset.
4. **Action advantage:** complete counterfactual score difference between event/no-event branches.

They have different domains, units and evidence requirements. One posterior must not pool them under
the word “costate.”

### 6.3 Hybrid information metric

A generalized task/rate metric must update at discrete events. Recent work on
[salted Fisher information](https://arxiv.org/abs/2603.29862) combines continuous information
accumulation with saltation updates. For Pact this is a proposal only: scheduled resets can use reset
VJPs, transverse guards can use validated saltation, and delayed message arrivals require the
augmented-state discrete jump above. Receiver critical faces still require exact endpoint edges.

Any generalized mode or costate measured before a render-target, topology, quantizer or receiver
event expires unless parallel transport or recomputation proves it remains in the same chart.

## 7. Canonical filename migration: positive but semantically incomplete

### 7.1 What commits `a0362725` and `feccfa39` prove

Twenty-two consumers now import names from `tac.witness_run_artifacts` rather than spelling them
locally. The first landing covered:

- dynamics analyzer and trace probes;
- campaign planner;
- showcase builder and costate digest;
- dash-comb, erasure, analytic-lane and heldout probes; and
- byte-close arm/checkpoint discovery.

The second landing covered twelve more pose, parity, determinism, dashboard, tau-crossover,
annulus, observer, exact-A/B and run-introspection consumers. It also reduced the ratchet baseline
from 33 files to 25. This reduces lexical drift and preserves behavior. It is legitimate
apparatus-only progress, not an engineering or R-gate pass.

### 7.2 What it does not prove

A filename constant does not define:

- producer, consumer and mutability ownership;
- required schema and provenance;
- deploy versus resume semantics;
- d_seg-best versus full-score-best order;
- event-transaction membership;
- receiver/config/axis identity;
- exact rollback destination; or
- retention/cold-store policy.

The contract declares `LIVE_NPZ` even though no corresponding trainer producer was found. It calls
`EMA_BEST_NPZ` best-scoring even though the producer orders only by d_seg.

### 7.3 Anti-hardcoding ratchet is not self-sealing

At the snapshot:

```text
migration baseline files = 25
current raw-text offenders = 25
stale baseline exemptions = 0
new offenders outside baseline = 0
```

The concurrent landing correctly removed eight exemptions. The test still checks only
`current - baseline`; each of the remaining 25 files is exempt because a docstring, comment, log
message, or other raw text still mentions a filename. Any such file can reintroduce a live literal
without failing. Raw-text scanning treats a docstring mention like a live path coupling, encouraging
permanent exemptions.

The semantic exit is an AST/live-expression scan plus an exact committed offender ledger. A removed
offender must cease being exempt in the same landing.

### 7.4 Run discovery remains over-broad

`RUN_DIR_GLOB="levelset_*"` matched 84 directories under `experiments/results`; only 39 had any
contract liveness path and 45 had none. The current `newest_run_dir()` correctly chose the live
v7.5.2 run, so there is no present misselection. A future packet/probe directory with a fresh named
artifact or arbitrary `.log` can still steal discovery.

`progress_log_paths()` accepts every `*.log` except names ending `observer.log`; it does not parse a
trainer PID binding or a valid `loss_terms`/verdict row. This remains a false-green surface.

### 7.5 Two consumer-semantic gaps

1. `costate_digest` imports `COSTATE_JSONL` but its no-sidecar fallback checks only hardcoded
   `run.log`. The live daemon writes `daemon.log`, so the new contract is not used to find fallback
   telemetry.
2. `campaign._resolve_best_ckpt()` searches preserved stage resume files and otherwise returns the
   rolling resume path. It ignores `levelset_best.json`; after the rolling checkpoint advances, a
   requested rollback to an earlier best can resolve to later state. `EMA_BEST` cannot repair this
   because it lacks optimizer/RNG/controller state and truthful provenance.

## 8. Launch-config duplication is now an explicit exit blocker

Commit `9c66ae72` durably audits a separate hygiene class. Its highest-EV finding is directly relevant
to vehicle launch authority:

- `witness_autoconfig._proven_base()` hand-maintains a 28-key shadow of
  `curriculum_dsl.BASELINE.base`;
- the two already diverge on `w_pose` and `verdict_pairs`; and
- no test proves an accepted mapping or overlap equality.

This violates single-source typed-config intent inside the launch actuator itself. Canonical filename
work does not address it. The correct repair needs design: one authority plus an explicit key-schema
translation, followed by an exact compiled-argv equality test and resume-manifest check.

The audit also records duplicated trainer paths, CLI-flag regexes, JSONL append helpers and canonical
store paths. Commit `3722879f` closes the specifically identified duplicated locked-JSONL helper
with one implementation and concurrency tests. The other hygiene classes remain open, and
`_proven_base()` versus `BASELINE` is still the load-bearing launch blocker.

## 9. Score geometry, allocation, and topology after events

### 9.1 Signed generalized allocation

The generalized pencil remains a proposal ranker:

```text
G_task v_i = lambda_i G_rate v_i.
```

It supplies sensitivity per local rate metric, not signed improvement. Pair it with a signed score
covector and exact `+v_i/-v_i` receiver probes. Under a high-rate quadratic approximation only, an
allocation has the form:

```text
b_i* = [0.5*log2((2*ln(2)*c_i*lambda_i)/(25/37_545_489))]_+,
```

where `c_i` is a measured quantization constant. This is falsified unless compiled held-out archives
beat coordinate and uniform allocations in exact full score.

A mode sensed at epoch `tau` must be transported or recomputed at arrival. Any intervening render,
topology or quantizer event expires it.

### 9.2 Group and topology integrability

For oriented class incidence matrix `B`, edge carriers must satisfy:

```text
psi = B^T u,
C psi = 0
```

for cycle matrix `C`. The rate-weighted projection is:

```text
P_grad = B^T (B W B^T)^+ B W.
```

Independent edge quantization can break cycle closure because generally:

```text
Q(P_grad psi) != P_grad(Q psi).
```

Projection/quantization order therefore requires an R4 five-state cell. Birth, merge or junction
events change `B` and invalidate old generalized modes, Hodge complexes and costates. Store
gauge-fixed node potentials or a spanning-tree basis unless exact parse-back proves cycle residual
zero.

## 10. R0–R8 evidence delta

| gate | new exact finding | status / reopen artifact |
|---|---|---|
| R0 | delayed event state not checkpointed; `EMA_BEST` provenance false/unknown | open/contradicted; transactional run manifest and triple replay required |
| R1 | path constants improve discovery only | open; actual archive signed consumption and typed roles still required |
| R2 | best/costate order is partial; finisher defects persist | `ACTUATION_REFUSE`; complete endpoint score and rollback required |
| R3 | no retained contest-CPU SDF center | missing; same center must bind modes, receiver, event state and score |
| R4 | reset-word and projection/quantization cells absent | missing/not authorized |
| R5 | no signed gauge-stressed generalized atlas | missing/not authorized |
| R6 | no delayed-state predictor versus no-jump/stale-sensor controls | missing/not authorized |
| R7 | no parse-back integrable topology atom | missing/blocking |
| R8 | early event skips declared treatment; queue/resume/costate not closed | contradicted/refuse |

No row is an engineering pass.

## 11. Exact exit predicates

### 11.1 `ASYNC_EVENT_TRANSACTION_CLOSED`

Pass only when every event family satisfies all of:

1. one immutable message contains payload, source epoch, source checkpoint hash, receiver hash,
   uncertainty, schedule/arrival times and message hash;
2. payload and metadata publish atomically under one lock/queue transaction;
3. pending and arrived messages persist in every rolling checkpoint;
4. actual `just_fired` drives the complete declared reset word;
5. immutable pre-arrival, pre-reset and post-reset resume/deploy checkpoints exist;
6. event checkpoint contains optimizer moments, LR-rewarm state, RNG, controller and queue;
7. crashes before sensing, while pending, after arrival and after reset reproduce the identical fire
   epoch, reset word, weights and next verdict;
8. stale-message inequality clears a measured replay floor;
9. matched event/no-event and delay-policy branches exist; and
10. complete full-score consequence is measured on retained endpoints.

### 11.2 `FULL_SCORE_BEST_BANK_CLOSED`

Pass only when:

1. candidates remain unordered until Seg, Pose, bytes, axis and receiver close;
2. selection uses the exact nonlinear score and a replay-derived admission floor;
3. async completion order cannot change the winner;
4. the bank retains the Pareto triple and per-class/per-pair components;
5. every point has truthful git/upstream/dirty/config/receiver/archive provenance;
6. deploy and complete resume checkpoints share one candidate ID;
7. rollback restores archive/raw/support/topology and optimizer/controller state; and
8. three fresh replays agree.

### 11.3 `COSTATE_TYPE_SYSTEM_CLOSED`

Pass only when:

1. terminal covector, trajectory slope, jump sensitivity and action advantage have distinct schemas;
2. any missing Pose/rate channel makes a full-score object `PARTIAL` or `UNIDENTIFIABLE`;
3. transition jumps compute exact nonlinear score or carry a narrower name;
4. rollback selects and compares under the same complete order;
5. persistence consumes the production evidence field and rejects invalid classes;
6. posterior keys include chart, event/reset word, receiver, lag, axis and units;
7. scheduled resets, transverse guards, delayed arrivals and receiver faces use their correct
   calculus; and
8. held-out sign calibration beats no-jump and stale-sensor ablations.

### 11.4 `RUN_ARTIFACT_SEMANTIC_CONTRACT_CLOSED`

Pass only when:

1. each artifact declares producer, consumers, role, schema, mutability and retention;
2. producer emissions and consumer reads are checked against the same contract;
3. deploy/resume/best/full-score roles cannot be confused by type or name;
4. provenance and receiver/config/axis fields are required and validated;
5. event-transaction artifacts are first-class contract members;
6. run discovery requires a typed manifest or bound trainer identity, not a broad name glob;
7. progress logs contain a valid trainer-bound row;
8. AST/live-expression hardcode offenders equal the committed migration ledger; and
9. migrated files lose their exemption in the same landing.

### 11.5 `CONFIG_SINGLE_SOURCE_CLOSED`

Pass only when:

1. exactly one typed object owns baseline values;
2. autoconfig derives from it through one explicit key translation;
3. overlapping keys are exact-equality tested;
4. compiled argv is compared to the intended DSL manifest;
5. resume metadata includes every semantic config field;
6. no launcher-side shadow/default can override it; and
7. the sealed vehicle config and realized argv are hash-bound.

None of these conjunctions currently passes.

## 12. Dependency-ordered advisory roadmap

1. **Protect the live run.** Do not signal or restart it. Preserve later artifacts only through the
   owning lane.
2. **Fix the event transaction before interpreting event effects.** Route actual `just_fired` through
   one reset/checkpoint transaction for lane, chroma and temporal-screw events.
3. **Make delayed state resumable.** Persist atomic messages and pending queues; run four crash-point
   equivalence tests.
4. **Replace d_seg `BEST` with typed banks.** Keep the legacy d_seg pointer honestly named if useful,
   but add a provenance-complete full-score/Pareto bank with paired resume state.
5. **Repair costate semantics.** Extinguish `.tier`/`.status`, fail closed on missing channels, and
   separate the four mathematical object types.
6. **Finish artifact migration semantically.** Roles and schemas first; shrink the exemption ledger;
   narrow run discovery and log liveness.
7. **Unify launch configuration.** Resolve `_proven_base()` versus `BASELINE` before another claimed
   canonical vehicle launch.
8. **Then close R0–R3.** Retained n600 archive, receiver controls, full-score finisher and contest-CPU
   center remain the prerequisites for atlas work.
9. **Only then run R4–R8.** Event reset cells, generalized modes, delayed predictor, topology atoms
   and six-leaf controller experiment.
10. **Keep descendants gated.** v7.5.3 exact-D/Pose-null and v8 integrability/receiver/rate exits
    remain downstream.

The existing century plan remains the long-horizon authority. This delta makes its immediate
controller and apparatus exits executable.

## 13. Literal launch and vehicle dispositions

| surface | literal disposition |
|---|---|
| current v7.5.2 process | `LIVE / OBSERVE-ONLY / DO NOT SIGNAL OR RESTART` |
| current lane-band treatment claim | `EARLY_EVENT_TREATMENT_CONFOUNDED` |
| current v7.5.2 promotion | `HOLD` |
| another v7.5.2 launch | `HOLD` until config/event/custody chain closes |
| v7.5.3 | `DESIGN/BUILD-ONLY`; no training EVENT |
| v8 | `HOLD TRAINING EVENT`; v7.5-first contract binding |
| `EMA_BEST` | `DSEG-SELECTED DEPLOY CANDIDATE / NOT FULL-SCORE / NOT R0 CUSTODY` |
| SDF finisher | `ACTUATION_REFUSE` |
| costate controller | `SENSE-ONLY / TYPE SYSTEM INVALID FOR ACTUATION` |
| filename migration | `APPARATUS-ONLY HYGIENE GAIN / NOT ENGINEERING OR R-GATE / SEMANTIC CONTRACT OPEN` |
| PR128 | `EXTERNAL_UNRATIFIED HNERV-FAMILY SIGNAL` |
| click-polish claim | sister-owned; untouched |
| pointer | unchanged |

## 14. Exact remaining blockers

1. no retained authoritative n600 SDF archive, typed run manifest or triple decode;
2. no actual-archive signed receiver controls;
3. no truthful joint Seg/Pose/byte endpoint finisher and rollback;
4. no contest-Linux CPU n600 SDF center;
5. no R4/R5 cells or signed generalized atlas;
6. no delayed-state R6 predictor or held-out no-jump/stale-sensor controls;
7. no reversible parse-back topology atom or integrability receipt;
8. early lane/chroma/temporal events bypass their declared optimizer treatment;
9. no transactional pending-message/checkpoint/resume state;
10. `EMA_BEST` carries false/unknown provenance and d_seg-only ordering;
11. transition and trajectory costates can be partial while labeled full-score measured;
12. production costate persistence still reads `.tier` instead of `.status`;
13. artifact constants define names but not semantic roles/schemas/provenance;
14. run discovery and log liveness remain over-broad;
15. rollback planning can fall back to a rolling checkpoint that no longer represents the best epoch;
16. the 25-file hardcode baseline can waive live-literal regressions inside an already exempt file;
17. `_proven_base()` and the typed DSL baseline are divergent shadow configs;
18. the live run remains attribution-confounded and axis-nonpromotable;
19. v7.5.3 exact-D, Pose-null and byte-close laws remain open;
20. v8 global-potential integrability, class isolation, receiver grammar and rate closure remain open;
21. PR128 remains unratified with stale tag/source custody; and
22. no same-byte CPU/CUDA SDF authority exists.

These are exact engineering blockers, not a negative verdict on the SDF witness family.

## 15. Triality and stores consulted

### DSL leg

One typed baseline must own config. Typed event declarations must compile both the guard and its full
reset/checkpoint transaction. Resume divergence guards must cover pending messages, reset word and
every candidate semantic field.

### DAG leg

```text
config single source
  -> typed event + atomic delayed message
  -> pre/post event checkpoint transaction
  -> full-score/Pareto bank + paired resume state
  -> typed costate/action schemas
  -> R0 custody
  -> R1 receiver consumption
  -> R2 exact endpoint acceptance
  -> R3 contest-CPU center
  -> R4/R5 event and mode atlas
  -> R6 delayed hybrid predictor
  -> R7 topology/integrability atoms
  -> R8 matched controller branches
  -> v7.5.3/v8 descendants.
```

### Equation leg

This delta sharpens:

1. intended versus realized reset words;
2. delayed-message augmented state;
3. staleness-margin certificate;
4. exact full-score candidate order;
5. high-rate generalized-mode allocation as a falsifiable approximation;
6. rate-weighted integrability projection; and
7. distinct schemas for terminal covectors, slopes, jumps and action advantages.

### Stores consulted

- unchanged full-preflight `CLAUDE.md` and `AGENTS.md` hashes;
- top-10 Pact Claude memory and current lane/subagent/directive/claim surfaces;
- canonical pointer and current `main`/`origin/main` state;
- restart handoff and complete advisory chain through the epoch-31 parent;
- live run process, log, checkpoint and NPZ metadata;
- trainer event, reset, checkpoint, async-worker and best-selection source;
- costate estimator and posterior source;
- run-artifact contract, both migration commits and focused tests;
- canonical locked-JSONL helper extraction and its concurrency tests;
- witness campaign rollback planner and costate digest;
- committed hardcode/duplication audit;
- fresh read-only PR128/release metadata; and
- the three primary hybrid/delayed-control sources linked above.

No external code was copied. Research informed this advisory only; any later OSS reuse still requires
file-level license and provenance review.

## 16. Pointer-delta honesty

```text
canonical CPU pointer before/after: 0.19108282419209976 [contest-CPU]
canonical CUDA pointer before/after: 0.20533002902019143 [contest-CUDA]
authoritative SDF score rows added:  0
R0-R8 engineering passes added:     0
launches/dispatches/evals/signals:   0 / 0 / 0 / 0
files owned by this unit:            this ADVISORY only
pointer delta:                       0
```

The advisory delta is closed. The engineering campaign remains open on the exact predicates above.
