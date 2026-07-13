# D-A / D-B producers implementation spec — 2026-07-13

Checkpoint: `da_db_producers`  
Lane: `lane_da_db_producers_20260713`  
Authority: operator build request; local `$0`; no launch/eval/commit

## Scope and collision boundary

Implement only the additive trainer/DSL/resume/test surfaces needed for D-A, D-B, and the
queued #404 Q1-Q7 emissions. The shared trainer is owned by this lane for this round. Other agents
are live in scorer-surrogate, ladder-analysis, Metal-conv, compander, and #336 namespaces. Do not
revert or rewrite their work. The compander trainer/DSL/resume landing is complete and must remain
intact. Preserve the thread-standard, Muon, causal-manifest, and all existing checkpoint regions.

Do not launch training, run an evaluator, dispatch a provider, touch a run directory, commit, stage,
or mutate the canonical frontier. No paid or remote action. The final working tree stays uncommitted.

## Binding invariants

- Read `CLAUDE.md`, `AGENTS.md`, and `docs/operating_manual_craft_handoff.md` before editing.
- Configuration is typed-DSL-only. Pure read-only telemetry defaults ON. A path that can perturb the
  trajectory is a named DSL Lever, default OFF, and enters activation-ledger duty-to-measure.
- Every stateful callback/latch registers in the canonical resume registry with additive legacy
  behavior. A legacy sidecar has no new required key and restores to a deterministic fresh telemetry
  state; training weights/config/update semantics are unchanged.
- Use `time.perf_counter_ns()` or an injectable monotonic-nanosecond clock. Never use wall clock for
  duration. Never insert historical 95/5 values, estimates, zeros for unmeasured work, or subtract
  overlapping timers as if disjoint.
- Runtime observation failures are loud, score-neutral, non-promotable, and fail-open for training;
  the launch-ticket static producer gate remains fail-closed if the exact schema is absent.
- JSONL persistence uses `tac.jsonl_store.append_locked_jsonl` (fcntl `LOCK_EX`) and lives in the run
  directory, never `/tmp`. Existing stdout JSON stages remain additive/back-compatible.
- Preserve existing stage names. Uniform companion rows supplement them; readers of old rows must
  continue to work.

## D-A: `witness_component_wallclock.v1`

Add an importable hook module under `src/tac/witness_control/` and wire it into the real trainer step
loop. Produce `out_dir / witness_component_wallclock.jsonl` by canonical locked append. Emit one
per-epoch row containing these exact string-literal fields:

1. `teacher_forward_s`
2. `teacher_backward_s`
3. `witness_forward_s`
4. `witness_backward_s`
5. `realized_R_s`
6. `verdict_s`
7. `checkpoint_io_s`
8. `epoch_total_s`

The row also needs `schema: witness_component_wallclock.v1`, epoch, sample/call counts per component,
measurement scope, monotonic clock name, score-neutral/non-promotable labels, and a measured residual
or overlap description. Compose with the causal-manifest boundary pattern; do not create another
manifest vocabulary.

Measure real elapsed intervals only. Aggregate repeated observations by epoch. `epoch_total_s`,
actual verdict critical-path wait, and actual checkpoint I/O wrap their real trainer regions.
Subcomponent hooks must wrap the real witness/R/frozen-teacher operations in the update path or a
clearly labeled, same-function read-only decomposition probe. They must not change arithmetic,
optimizer/EMA state, RNG streams, data order, or scorer authority. If MLX laziness makes a timer only
graph-construction time, force honest synchronization for the observer or emit a loud measurement
error; do not relabel construction latency as component wall. If a teacher-vs-witness backward split
cannot be observed without changing the update path, keep the training path unchanged and implement
an isolated same-function VJP probe with its scope and sample count explicit. Never fabricate a
disjoint sum. Tests must prove no negative durations, no missing fields, exact aggregation, locked
append, failure isolation, and no historical constants.

The existing `profile_timing` row remains unchanged; the new row is its exact/additive sibling.

## D-B: `sps_gradient_role_conflict_engagement.v1`

Reuse the mathematics and thresholds from `tools/probe_sps_gradient_role_conflict.py` in the new
importable hook; do not import the CLI or duplicate inconsistent rules. Default sample count is the
same deterministic four strata (`K=4`; for n600 the canonical indices are 75,225,375,525). The
observer computes separate, update-free gradient passes, never calls `opt.update`/EMA, and runs under
the existing `MxRngGuard`/deterministic flattening discipline.

At the actual screw engagement callback emit `engagement: temporal_screw_engaged`; at the actual
phase-advection callback emit `engagement: phase_advection_engaged`. Screw ep450 is a fail-safe cap,
not a substitute for an earlier event fire. Phase ep726 is the current static terminal-band anchor.
Emit configured nominal boundary `+-N` cadence rows and always emit at an actual transition even if it
fires away from the nominal window. Default `N` is small (2 is acceptable) and configurable through
real parser fields held by the existing typed observer telemetry surface. The observer is default ON
when either target mechanism is configured, and inert/cost-free otherwise.

Each row includes schema, epoch, engagement, cadence/reason, actual-event flag, sampled pair indices,
per-role norms, prediction-vs-temporal cosine, coactive/negative scalar fractions, per-tensor
statistics, the probe's `-0.05` cosine and `0.10` material-negative-fraction thresholds, conflict
predicate, exact active temporal terms/weights, `score_neutral: true`, and non-promotable axis/scope.
Persist the emitted event/epoch latch through the canonical resume registry so a resume does not
silently duplicate or skip an engagement record. Legacy sidecars restore deterministically.

## Q1-Q7 additive batch

Land and test each numbered item; count them separately in the receipt.

1. Per-group clip activation: accumulate the real pre-clip norm returned at the existing group clip
   site; emit one `grad_clip_activation` row per epoch with global and per-group `n`, clipped fraction,
   mean/max norm. No extra gradient pass.
2. Term domination/inertness: extend domination coverage to `chroma_boundary`, `margin_saliency`,
   `temporal_screw`, `island_amplify`, and `persistence`. Add sustained `term_inert` alarms for an
   engaged lever whose post-weight share stays below `1e-6`. Keep alarm streaks per term, not one
   shared streak that lets terms mask each other. Pure telemetry defaults ON.
3. Add DSL factory `VerdictLiveGap(every=...)` holding `--verdict-live-gap-every`. Trainer default is
   `0` (OFF). Positive cadence performs the extra live-weight advisory inference and adds
   `d_seg_live`, `d_pose_live`, and explicit EMA-minus-live gaps to the corresponding verdict row for
   both sync and async paths. It never feeds training/controller decisions. It is registered in
   activation-ledger duty-to-measure automatically through the real Lever factory.
4. Emit explicit `tail_cycle_endpoint` at every tail boundary using the last completed measured
   verdict, with cycle/start/end/boundary reason and no invented endpoint when no verdict exists.
5. Emit would-fire/held rows for `powerlaw_meat` and `annulus_plateau` at verdict cadence, naming
   metric, threshold/dwell, sensor-data epoch, event-mode/held/fired state, and verdict scope.
6. Emit one `ladder_birth_complete` row per configured ladder class when its scheduled birth+hold+
   anneal completes. Include epoch, class id/name, final radius. Persist completion emission state in
   the canonical registry so resume is nonduplicating; old sidecars remain valid.
7. Add uniform companion rows `{stage: lever_engage, lever, status, epoch, via}` alongside existing
   arm/fire/complete stages. Status vocabulary is exactly `armed|fired|complete`. Do not delete or
   rename an existing stage. Cover at least screw, phase, chroma, lane/band, Muon, pose finish,
   margin-saliency, subpixel, satisficing, horizon, thin-lane, additive-margin, tail, and ladder.

## Tests and acceptance owned by implementation

Add at least 20 focused tests, preferably in one new test module plus narrow updates to existing DSL/
resume/ticket tests. Required coverage:

- pure timer aggregation, nested/overlap policy, clock injection, error row, fcntl append, all eight
  exact literals, schema and per-epoch row;
- SPS cosine/norm/fraction parity with the CLI math, zero-gradient behavior, conflict thresholds,
  deterministic strata, nominal window, actual-event override, no duplicate after resume;
- legacy resume with no new keys, new-state round-trip, direct-controller static completeness;
- Q1 aggregate math; Q2 per-term streak independence; Q3 factory/default/parser/manifest and sync+
  async source wiring; Q4/Q5 row contracts; Q6 one-shot/resume; Q7 vocabulary and companion presence;
- AST/static tests that `torch_thread_standard`, Muon switch, `pact.causal_manifest.v1`, checkpoint
  blocks, and compander wiring remain present;
- next-launch dependency detection sees both exact schema literals and therefore clears only D-A/D-B.

Run focused pytest with at least 20 tests passing, `ruff check` on every touched Python file,
`python -m py_compile` for the trainer and hook modules, and parser help/parse smoke. Do not run a
training job or scorer/evaluator. Stop after code/tests are green and report exact touched files and
commands; main owns ticket recompilation, receipts, DAG/pool rows, and the final memo.
