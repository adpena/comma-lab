# Dashboard LIVE tab — schema-driven v6 introspection (task #352)

[no-triality] — observability surface only; score-neutral; the frontier pointer
(contest-CPU 0.19110) is UNMOVED by anything here.

STORES CONSULTED: CLAUDE.md (§"'Off' is a tracked queue" default-off/observability-on;
§"Max observability"; §Triality "DSL HOLDS every designed lever" + `schedule_readback`
is the schedule SoT; §"VALUE-PROVENANCE LADDER"; NO-FAKE) · docs/operating_manual_craft_handoff.md
· MEMORY.md CURRENT-STATE (event-governed crucible_v6; #351 constants manifest; #247
costate SENSE; #329 mem_probe; confound-immune liveness signatures) · the live run
`experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z` (READ-ONLY:
launch.sh, constants_manifest.json, costate_shadow.jsonl, run.log) · sibling surfaces
tools/{dashboard_server,render_levelset_dashboard,dashboard_reload,dashboard_up}.py ·
src/tac/witness_dsl/schedule_readback.py · the trainer's own
`_softmax_temp_for_epoch`/`_hosc_beta_for_epoch`/`_lr_scheduled_for_epoch`.

## Why (the stale-lens problem)

The LIVE tab parsed the schedule through the LEGACY epoch-scripted `tau/l7` lens
(`_schedule_from_flags` / `buildScheduleRows`). crucible_v6 run-1 is EVENT-governed
(event-triggered curriculum with nucleus guard, τ event hand-off with a hard cap,
Muon finisher, LawRef-derived constants). Rendering an event-gated stage as a fixed
`ep [300, 726)` boundary is exactly the PR95-era assumption the operator just banned
from configs — the dashboard must not preserve it either.

## Introspection-layer schema (DESIGN LAW #1)

NEW `tools/witness_run_introspect.py` — ONE schema-driven layer, dependency-light
(stdlib + `tac.witness_dsl.schedule_readback`), fail-open, bounded-tail (incremental
over a multi-day run.log). `introspect_run(run_dir) -> dict` with every facet sourced
from a REAL artifact (NO-FAKE — absent artifact → `None` → panel absent, never
fabricated):

- **schedule** — DSL `read_schedule` stages, each tagged `klass ∈ {event, fixed}` with
  live arm state (event → pending/fired + hard-cap + DSL-derived trigger description;
  fixed → epoch literal). The classification is derived from the run's OWN config, so a
  future stage/event kind is additive here with no renderer rewrite.
- **constants** — `constants_manifest.json` (#351) → ranked table {value ·
  value-provenance ladder tier+label · equation_id · anchor sha256(12) · provenance}.
- **controller** — `costate_shadow.jsonl` (#247) last row → costate λ traces
  (value/band/status/method/units), classification, top recommendation, duty-to-measure
  queue (owed / never-fired / ranked), probe queue, per-axis EV producer signal, pointer.
- **liveness** — last `{"stage":"verdict"}` row's confound-immune signals (accepted-frac,
  weights_stepped, skip counts, frozen_epoch, ep_loss) + alarms (frozen / ep_loss_zero /
  low-accepted-frac). `frozen_epoch` is a BOOLEAN (a bare `False` is NOT frozen — guarded).
- **mem** — `{"stage":"mem_probe"}` (#329) rss/mlx rows → capped 64-pt series + peak +
  latest phase (a growing run emits hundreds; the tail-window is rendered, peak is tail-wide).
- **events** — genuine FIRED curriculum/optimizer transitions only (curriculum_transition_fired,
  muon_finisher_switch, moments-reset, rollback); setup-lever config rows are NOT events.
- **curves** — PLANNED τ/β/LR, FAITHFUL pure-python ports of the trainer's own anneal
  formulas (cosine_hold + hold-frac + geometric; hosc linear/cosine; LR warmup+cosine).
  τ/β HOLD past the Muon freeze (the finisher freezes the schedule at muon-start).

## Panels added (LIVE tab, conditional)

- **Controller panel** (`#costate`, upgraded `renderCostate`): costate λ table + DECIDE
  (duty queue / probe queue / axis EV / actuation-advisory) + confound-immune LIVENESS
  strip. Falls back to the legacy `META.costate` summary for pre-v6 runs.
- **Config panel schedule** (`renderConfig`): now schema-driven (`schClassifiedRows`) —
  event/fixed classification chips + live arm state (armed·pending / fired / active);
  legacy two-axis epoch renderer remains the fallback when no DSL read-back exists.
- **NEW telemetry panel** (`#telemetry`): planned τ/β/LR sparklines (inline SVG, no deps,
  Muon-freeze marker) · LawRef constants manifest table with provenance-ladder chips ·
  mem_probe RSS/MLX sparklines + peak · fired-event diamond markers.

Aesthetic extends the existing dark tabular-nums tokens (new `.kchip/.sdot/.lamrow/
.livestrip/.crv/.cst/.membars/.evchip` classes); semantic state colours (event=amber,
derived=violet, fixed=blue, fired=green, alarm=red) kept separate from the accent;
sparklines with emphasized endpoints; events as diamonds (distinct from epoch ticks).

## Conditional / degradation behavior

Every panel/section is presence-gated: absent artifact → section omitted, never a crash.
Verified: (A) pre-v6 dir (launch.sh only) → schedule+curves render, constants/controller/
mem absent; (B) introspect all-null → all new panels hide; (C) missing/None run dir →
`ok=False`, no crash; legacy `META.costate` still renders the controller panel.

## Wiring + performance

`LiveState.refresh()` computes `self.introspect` mtime-GATED (recompute only when the run
dir or its four source artifacts change mtime — never a full re-parse per 5 s tick);
shipped in `meta()` as `introspect` (+ `introspect_ok`). run.log is read via a bounded
256 KB tail. Import is fail-open (a broken tac install degrades to no new panels).

## Verification

- 27 unit tests (`tools/test_witness_run_introspect.py`) — classification (event/fixed),
  constants ranking+provenance, controller tail-parse (last-row-wins), liveness alarms +
  the frozen_epoch boolean guard, mem cap/series/peak, fired-vs-setup events, τ curve
  faithful to the trainer formula, τ hold + Muon freeze, LR warmup+honest note, β linear,
  graceful degradation (missing/None/pre-v6), bounded-tail. All pass.
- Regression: existing dashboard tests (render/traj/projection/server/schedule_readback) —
  63 pass. ruff F clean on all changed files.
- Node harness executed the three new render functions against the LIVE `/api/state`
  payload with a DOM shim — no runtime errors; produced λ table + SVG curves + LawRef
  table + schema-driven schedule with event chips.
- LIVE reload via `tools/dashboard_reload.py --port 8790` (SO_REUSEPORT zero-downtime);
  page HTTP 200; `/api/state.meta.introspect` healthy (schedule CE-fixed / tau-event-pending
  / Muon-fixed · 4 constants · 4 costates · τ/β/LR curves · mem 228 probes→64-pt series ·
  liveness ep tracked).

## Scope

LIVE tab + its data layer ONLY. ORACLE/WITNESS/RESIDUAL/SANDBOX untouched; every existing
LIVE feature (liveness cadence, checkpoint audit, staleness, access gate, run-info strip,
scorer breakdown, projection) preserved.
