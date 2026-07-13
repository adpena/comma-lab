# Causal/transition manifest build — 2026-07-13

**Lane:** `lane_causal_manifest_build_20260713`  
**Schema:** `pact.causal_manifest.v1`  
**Status:** `LOCAL BUILD`, `$0`, uncommitted for main review  
**Primary import surface:** `tac.causal_manifest`  
**Pointer delta:** `NONE`

## Outcome

`MEASURED` — One typed append-only schema now covers both required granularities:

1. immutable run-level treatment/custody manifests; and
2. ordered boundary observations, `(Z,A,R,Z')` transitions, exploration decisions, and explicit
   target-policy coverage receipts.

`MEASURED` — Producer wiring is additive and score-neutral:

- the level-set MLX trainer emits a run manifest at startup and boundary/transition rows at baseline,
  advisory-verdict, checkpoint, stage, and final boundaries; and
- the costate shadow writes a typed arm decision beside each persisted shadow report, with the
  exploration hook present but disabled pending operator GO.

`MEASURED` — Consumer surfaces exist in the same module: strict loader, FORE structural-support
checker, and HCM-L4 residual/closure skeleton. The HCM consumer never returns an unconfounded
certificate. The FORE consumer never converts a zero-propensity arm into numeric authority.

## Convergence consumed, not re-derived

The following settled inputs were treated as requirements:

- `.omx/research/fore_occupancy_ratio_dig_20260713.md` §5/§6: ordered
  state-action-reward-successor rows, full state/history custody, logged target-arm support, and
  initial/one-step coverage;
- `.omx/research/hcm_causal_attribution_dig_20260713.md` §4 and §7 item 1: typed treatment/run
  custody, pair outcomes, apparatus tags, leave-one-run-out residuals, negative-control moments,
  exact weighted-loss closure, and a frozen/no-update positive control; and
- `.omx/state/deferral_ledger.md` D40: behavior-policy exploration/randomization custody for organ
  schedule-arm OPE.

No theorem, score, action effect, or threshold was re-measured or re-derived.

## Public schema and named dependency for `launch_config_composer`

The sibling composer should depend only on the stable public surface:

```python
from tac.causal_manifest import (
    MANIFEST_FILENAME,
    SCHEMA_ID,
    RunTreatmentManifest,
    StagePlanEntry,
    canonical_sha256,
    freeze_fields,
)
```

The composer owns DSL config/tickets. This lane does not add a composer flag, ticket, or DSL file.
The manifest schema owns validation and persistence; callers must not append raw dictionaries to the
causal JSONL.

## Run-level field rationale

| field | status | why it is present |
|---|---|---|
| `schema_id`, `row_kind`, `row_id` | `DERIVED` | Version dispatch plus immutable append identity. Unknown versions/kinds fail closed. |
| `run_id` | `DERIVED` | Clustering/custody unit and join key for transitions, decisions, manifests, and coverage. |
| `treatment_vector`, `treatment_sha256` | `MEASURED` producer value | Exact typed argparse treatment. `out_dir` and `resume_from` are excluded because they are storage/continuation locators, not treatments. No invented flags. |
| `base_checkpoint` | `MEASURED` when present; explicit unavailable reason for fresh init | Separates a warm-start treatment from fresh initialization and carries the exact file SHA-256. |
| `seed` | `MEASURED` | Reconstructs training and behavior RNG roots. |
| `machine`, `backend`, `axis` | `MEASURED` | Blocks silent pooling across code/hardware/authority strata. Manifest rows are always non-promotable. |
| `data_order`, `data_order_sha256` | `MEASURED + DERIVED` | Captures the implemented `np.random.permutation(P)` plus optional PCG64 hardness-extra rule, pair counts, seeds, and hardness knobs. The algorithm description is derived directly from the producer code. |
| `stage_plan` | `MEASURED` | Records fixed/event caps and Muon boundary without assuming that an event fired. |
| `scorer_artifacts` | `MEASURED` or explicit unavailable | Carries the canonical upstream snapshot SHA-256; never writes a placeholder digest. |
| `cache_artifacts` | `MEASURED` | Carries both the exact active-target-array hash and, when used, the source GT-cache file hash. This distinguishes a subset/materialization from its source file. |
| `score_neutral`, `promotable` | `DERIVED invariant` | Hard-coded `true/false`; the logging row cannot acquire score authority. |

`ASSUMED` — Treating every argparse field other than the two path/continuation locators as part of the
treatment is deliberately conservative. It may over-separate observational-only configs, but it
cannot silently pool behavior-changing configs. Future relaxation requires a typed treatment-role
registry, not an ad hoc exclude list.

## Deliberate `Z` state summary

The state summary is intentionally a compact boundary sufficient-statistic candidate, not a claim
that the Markov condition has closed.

| `Z` field | status | purpose / cost decision |
|---|---|---|
| `boundary_id`, `sequence_index`, `boundary_kind` | `DERIVED` | Gives deterministic ordering. Epoch has a coarse slot and boundary kind/ordinal disambiguate baseline, verdict, checkpoint, stage, and final rows. |
| `epoch`, `stage` | `MEASURED` | Minimum training-clock/regime covariates required by both readers. |
| `policy_sha256` | `MEASURED` | Binds the state/action to the exact treatment or costate decision policy. |
| `data_order_cursor` | `DERIVED from measured counts` | Number of base-plus-extra pair visits implied through that epoch. Avoids serializing every permutation while retaining a cursor into the resume-owned RNG law. |
| `telemetry_history_sha256`, `telemetry_history_rows` | `MEASURED` | Commits to the ordered history without duplicating the full run log in every row. The underlying history remains in run artifacts; a changed prefix changes `Z`. |
| `checkpoint` and `resume_state_sha256` | `MEASURED` at checkpoint boundaries | Exact resume-bundle custody. Hashing occurs only after atomic checkpoint write. |
| `rng_state_sha256`, `controller_state_sha256` | `MEASURED if independently extracted`; currently `None` | The exact RNG/controller bytes already ride inside the resume bundle. Separate hashes remain absent rather than pretending extraction occurred. |
| `apparatus.weights_stepped`, `accepted_fraction`, `guard_path`, `measurement_mode` | `MEASURED` where producer exposes them | HCM covariates for frozen/skipped/guarded apparatus paths. |
| `apparatus.total_loss`, `loss_terms`, `negative_controls`, `positive_control` | `MEASURED only` | Supports equations (3)--(5). Missing decomposition/control rows return `INVALID_INPUT`; no term is reconstructed from memory. |
| `outcome` | `MEASURED` or explicit unobserved reason | Typed realized-through-R `d_seg`, `d_pose`, per-class `d_seg`, archive bytes, implied score, and axis. Observed rows must assert `through_r=true`. |
| `state_sha256` | `DERIVED` | Canonical hash over the entire state summary, verified on load. |

`ASSUMED` — Boundary-level state is the sane-cost first representation. It omits per-step optimizer
tensors and serial pair permutations because checkpoint/hash + data-order/RNG custody can recover
them without exploding JSONL volume. Whether this summary is Markov-sufficient is unproved and stays
a FORE admission question.

`ASSUMED` — Trainer boundary transitions use `pair_id=__aggregate_all_pairs__`. That is honest for a
whole-run checkpoint/verdict boundary but is not enough for HCM's pair plate. The HCM checker rejects
this sentinel as `missing_pair_outcome_custody`; no aggregate row is promoted to pair evidence.

## `A`, `R`, `Z'`, exploration, and coverage

- `ActionSummary` records action identity/type/arm/policy digest plus canonical typed parameters.
  The policy digest must equal `Z'`'s policy digest.
- `RewardObservation` names the estimand. Verdict boundaries use negative advisory implied score;
  checkpoint-only boundaries carry an explicit unobserved reason.
- `TransitionRow` requires strictly increasing ordered state, never a backward epoch. A late async
  verdict remains a boundary row without a fabricated transition.
- `ExplorationDecisionRow` records chosen arm, every considered alternative, normalized propensities,
  deterministic/randomized policy mode, policy digest, authorization hook, execution, actuation, and
  actual random seed/draw when randomized.
- `CoverageReceiptRow` is separate because logged propensities do not prove initial/one-step target
  coverage. It names the assessment method, working support, evidence, and verdict scope.

`DERIVED` — A randomized decision is invalid unless the hook is exactly
`externally_authorized` and the actual seed/draw are present. An advisory decision is invalid unless
`executed=false` and `actuation=NONE`. This makes the hook non-fireable by omission.

## Producer wiring map

### Trainer

File: `experiments/train_levelset_witness_realized_through_R_mlx.py`

- post-GT startup: hashes exact active targets, source cache, scorer snapshot, treatment, data-order
  law, base checkpoint, machine/backend/axis, and stage plan; then calls
  `CausalManifestWriter.ensure_run_manifest`;
- baseline: records realized advisory `Z_0` after the existing history append;
- async verdict: records after the verdict/history lock section, using the snapshot's liveness and
  realized outcome;
- sync verdict: records after the existing history append;
- `_do_checkpoint`: after the atomic deploy/resume writes, hashes the preserved resume bundle when
  present (otherwise rolling latest) and records checkpoint/stage state; and
- final checkpoint: declares `boundary_kind=final` and the resolved final stage.

There is no CLI flag. Logging is default ON because it is read-only score-neutral observability.
Failures produce a loud `causal_manifest_warning` row and leave training behavior unchanged; an
existing corrupt manifest is never rewritten or silently repaired.

The trainer region is distinct from the live Muon sibling: schema imports, post-GT startup custody,
post-history verdict hooks, and the checkpoint tail only. Optimizer construction/update/resume logic
was not changed by this lane.

### Costate organ

File: `src/tac/witness_control/shadow_controller.py`

`write_shadow_row` still writes the canonical costate shadow row, then calls the typed manifest
writer for the matching decision observation. The ordered recommendations plus refusals form the
alternative set. Current behavior is exactly deterministic:

```text
chosen propensity = 1
all other propensities = 0
exploration_hook = disabled_pending_operator_go
executed = false
actuation = NONE
```

This is D40's logging/randomization *hook*, not an operator authorization and not a causal-OPE cure by
itself.

## Consumer behavior

### Loader

`load_causal_manifest(path, strict=True)` validates every line, schema version, row kind, digest,
state hash, and immutable row id. `CausalManifestWriter` reloads the last ordered boundary from disk,
so the transition chain is resumable across process restarts.

### FORE support checker

`check_fore_support(rows_or_path, target_policy_sha256, target_arms)` returns
`ADMISSIBLE_STRUCTURAL_INPUT` only when manifests, fully observed transitions, matching target-policy
hash, explicit initial and one-step coverage receipt, and executed positive-propensity support for
every target arm exist. Otherwise it returns `NOT_IDENTIFIED` with exact blockers. It does not fit a
ratio, estimate value, or certify a contraction.

### HCM-L4 skeleton

`hcm_l4_residual_check(...)` consumes caller-supplied leave-one-run-out predictions keyed by immutable
transition row id. It validates treatment/source custody, pair identity, observed reward, at least two
whole runs, term closure, preregistered negative-control support, and the frozen/no-update positive
control. It computes whole-run negative-control moment standardization but deliberately does not
invent a p-value or bootstrap result.

The positive control must produce the known residual/mechanism break above the preregistered
threshold before the consumer can be trusted. Quiet output is `QUIET_NOT_CERTIFIED`, never
"unconfounded."

## D40 build-side closure note for main ledger review

Do **not** treat D40 identification as closed. Main may replace only D40's status cell after reviewing
this landing with the following append-preserving meaning:

```text
BUILD-SIDE CLOSED 2026-07-13 — pact.causal_manifest.v1 and the costate-shadow producer now log
chosen arm, alternatives, exact deterministic propensities, policy hash, execution/actuation, and a
validated externally-authorized randomization hook requiring actual seed/draw. IDENTIFICATION OPEN:
current rows remain deterministic 1/0, advisory, executed=false; a future operator-GO'd organ arm must
actually randomize/explore, execute, log positive target-arm support, and accrue multiple trajectories
before FORE/DR causal OPE is admissible.
```

Trigger/owner remain D40's existing trigger and `main`. This memo does not mutate the hot deferral
ledger.

## Tests and static verification

`MEASURED` locally:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_causal_manifest.py \
  src/tac/tests/test_witness_control_costate.py

67 passed

.venv/bin/ruff check \
  src/tac/causal_manifest.py \
  src/tac/tests/test_causal_manifest.py

All checks passed

.venv/bin/ruff check --select E9,F \
  src/tac/witness_control/shadow_controller.py \
  experiments/train_levelset_witness_realized_through_R_mlx.py

All checks passed

.venv/bin/python -m py_compile \
  src/tac/causal_manifest.py \
  src/tac/witness_control/shadow_controller.py \
  experiments/train_levelset_witness_realized_through_R_mlx.py

exit 0
```

The 67 tests cover frozen rows, canonical JSON, nonfinite rejection, digest typing, false-authority
outcomes, treatment drift, state hashes, forward-only transitions, strict loading, idempotence,
disk-resumed chains, late async rows, deterministic/randomized propensity rules, authorization and
actuation invariants, FORE support/refusal, HCM source/pair/closure/positive-control gates, graph
moment firing, trainer static wiring, and organ runtime emission.

`MEASURED` — the broad trainer/shadow files contain pre-existing lint findings outside this lane;
they were not mechanically rewritten because both are shared/hot surfaces. New schema/tests are fully
ruff-clean, and both producer files are syntax-clean plus focused undefined-name/fatal-error clean.

## What remains owed

1. `MEASURED gap` — Actual pair-level transition/outcome rows. Aggregate trainer boundaries are not
   HCM pair evidence.
2. `MEASURED gap` — Exact typed loss-term values/weights and preregistered negative controls at the
   same boundary, so equation (5) can close instead of returning `INVALID_INPUT`.
3. `MEASURED gap` — A frozen/no-update positive-control run that demonstrably triggers before HCM-L4
   is trusted.
4. `MEASURED gap` — Multiple independent runs and genuine whole-run cross-fitting/calibration. Rows
   from one trajectory are not independent runs.
5. `MEASURED gap` — A producer for `CoverageReceiptRow` backed by actual target initial and one-step
   support evidence.
6. `MEASURED gap` — Operator-GO'd organ exploration/randomization with actual execution, seed/draw,
   positive target-arm propensity, and multiple trajectories. Current deterministic alternatives
   remain unsupported.
7. `FORMALIZATION_PENDING` — Full HCM diagnostic vector (wild/permutation calibration, residual
   scale, lag-1, pair-rank, multiplicity) and FORE ratio/value fitting remain separate future builds.
8. `MAIN REVIEW` — Launch composer imports the named schema surface; main may update D40 and append
   the standalone DAG node to a shared DAG if collision-free.

## Triality and apparatus wire-in

- **DAG:** `.omx/research/causal_manifest_DAG_FEED_20260713.md`.
- **Equation:** no new canonical equation. HCM equations (2)--(5) remain parent design/check equations;
  the schema does not manufacture causal identification.
- **DSL:** `N/A-with-reason`. The manifest is apparatus, not a lever; a flag would violate default-on
  observability and create an orphanable off state. Future randomized policy behavior must be typed
  separately in the DSL under operator GO.
- **Sensitivity map:** no update until identified effects exist.
- **Pareto:** logging/hash/storage cost and coverage debt are explicit; no score movement claimed.
- **Bit allocator:** non-binding; JSONL never enters archive bytes.
- **Cathedral/autopilot:** fail-closed support/apparatus statuses are reusable read-only gates; no
  dispatch hook.
- **Continual learning:** typed causal rows are the durable signal; deterministic zero support remains
  a learned blocker rather than being smoothed away.
- **Probe disambiguator:** deterministic walk-forward versus authorized randomized schedule is decided
  by actual logged propensities and coverage receipts.

## Storage, execution, pointer, and stores consulted

`MEASURED` — The manifest is small run-local metadata. It creates no raw frames, tensor caches,
checkpoint copies, profiler traces, or temporary environments, so no bulk cleanup hook is applicable.
Checkpoint bytes are referenced and hashed in place after their existing atomic write. No evidence
path uses `/tmp`.

`MEASURED` — No training, scorer, evaluator, exact replay, archive mutation, GPU/provider dispatch,
or live-run mutation occurred. Current canonical `reports/latest.md` still reports
`[contest-CPU Linux x86_64] 0.1910828242` and `[contest-CUDA T4] 0.2053300290`; this build moved neither
axis.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5 §8 operating contract; v8 spec; top current project
memory entries; current lane/subagent registries; last-24-hour directives; FORE and HCM parent memos
and standalone DAG feeds; D40 in the deferral ledger; canonical `tac.jsonl_store`; trainer checkpoint,
verdict, history, and data-order implementations; costate shadow controller/tests; and
`reports/latest.md`.
