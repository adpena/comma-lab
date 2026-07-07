# Hardcoded run-constants sweep — consumer surfaces (task #340, 2026-07-07)

Operator directive verbatim: **"Anything hardcoded related to a run should be DSL."**

Class: a run CONSUMER (dashboard, launcher wrapper, observer, viz/checkin tool) hardcodes a
parameter that duplicates a property of the run's own config. When the config evolves the
consumer silently drifts (the `--tau/--l7` dashboard mislabel incidents; the clip-constant
class fixed by `tac.clip_profile`). Canonical derivation sources:

* stage boundaries → `tac.witness_dsl.schedule_readback.read_schedule(run_dir)` (landed
  bd8def976: launch.sh through the REAL trainer argparse + fired-transition evidence;
  fail-open with visible "schedule: fallback")
* clip constants → `tac.clip_profile`
* everything else per-run → the run dir's `launch.sh` / resume sidecar `__cfg_*` keys
* frontier scores → `.omx/state/canonical_frontier_pointer.json` (already pointer-only)

Honest three-way classification used throughout:

* **CONSUMER HARDCODE (the bug class)** — a consumer duplicating a run property. Flagged.
* **TRAINER DEFAULT (fine)** — argparse defaults in `experiments/` trainers are the DSL's
  COMPILE TARGET (the DSL emits trainer argv); not consumer hardcodes. Out of scope.
* **PROVENANCE PIN (deliberate, fine)** — measurement/build tools pinning the exact config
  they measured (mod-dim=32, gt-cache npz defaults with explicit `--gt-cache` flags,
  874/1164 in byte-close/build tools that must reproduce exact bytes). Out of scope.

## Ranked inventory (blast-radius × silence)

| # | file:line | constant | duplicates | blast when config evolves | silence | derivation source | status |
|---|---|---|---|---|---|---|---|
| 1 | `tools/dashboard_reload.py:82-83` | `--tau 300 / --l7 600` argparse defaults, ALWAYS forwarded to dashboard_server | the run's stage boundaries | every hot reload of the LIVE dashboard re-poisons the DSL-derived stage map (explicit flags are OVERRIDES that defeat bd8def976's fix); live mod32cap run has NO l7 stage → mislabeled bands on the primary observability surface | total — labels look plausible | schedule_readback (server-side); flag = override only | **ROUTED** (default None; flags emitted only when explicitly given) |
| 2 | `tools/dashboard_supervisor.py:466-467` (+ env `DASH_TAU/DASH_L7` at ~303, server argv at ~323, self-relaunch argv at ~537) | `--tau 300 / --l7 600` forwarded through THREE surfaces | stage boundaries | the self-heal daemon RESPAWNS the server with the poison on every heal cycle — the fix in the server is undone autonomously, forever | total | schedule_readback; forward only explicit overrides | **ROUTED** (default None; all 3 forwarding surfaces conditional) |
| 3 | `tools/launch_witness_run.py:450` | printed operator hint `dashboard_reload.py --port N --tau 300 --l7 600` | stage boundaries | every launch with a down dashboard prints the exact poisoning command for the operator to copy-paste; hint constants were already stale (600 vs the renderer's 900 vs the live run's no-l7) | total (it's "help" text) | none needed — dashboard derives; hint drops the flags | **ROUTED** (flags removed from hint) |
| 4 | `tools/dashboard_up.py:129-130` | `--tau 300 / --l7 600` forwarded to render_levelset_dashboard | stage boundaries | legacy static-HTML path renders wrong stage bands | total | render_levelset_dashboard needs a derive-or-omit refactor first (its CLI takes required ints); then default None here | LIVE WARN (P1 ×2) |
| 5 | `tools/render_levelset_dashboard.py:1775,1779` (+ docstring `:14`) | `--tau 300 / --l7 900` defaults; docstring example `--tau 300 --l7 900` | stage boundaries | legacy renderer + every consumer that shells it inherits stale bands; note it DISAGREES with #1/#4 (900 vs 600) — two tools describing the same run differently, the class signature | total | schedule_readback on the resolved run dir (same pattern as dashboard_server:814-832) | LIVE WARN (P1 ×2 + P2 docstring) |
| 6 | `tools/render_witness_trajectory_dynamics.py:654-655` | `--tau 300 / --l7 900` "stage inference" defaults | stage boundaries | the trajectory-dynamics instrument attributes dynamics to the WRONG stage (its whole point is per-stage attribution) | total | schedule_readback(run_dir) | LIVE WARN (P1 ×2) |
| 7 | `tools/dashboard_supervisor.py:460` | `--run-dir default="experiments/results/levelset_openpilot_seeded_n200_DEPLOY"` | which run is live | supervisor brought up bare points the dashboard at an OLD run (server has auto_latest, but the pinned run-dir seeds globs/paths) | high — shows A run, just not the live one | newest `experiments/results/levelset_*` glob / DASH_AUTO_BASE_GLOB resolution | inventory (not routed — behavior change needs operator eyes) |
| 8 | `tools/dashboard_supervisor.py:470` | `--training-pid default=26124` | the training process identity | hardcoded PID of a long-dead process; liveness luckily falls back to `--training-sig` | high (masked by the sig fallback) | drop the magic default to 0 (sig-only) | inventory |
| 9 | `tools/render_comma_baseline_vs_ours_viz.py:61`, `tools/render_witness_morse_smale_viz.py:73` | `CAMERA_H, CAMERA_W = 874, 1164` | clip resolution | breaks silently on any non-0.mkv clip (the machinery is supposed to be clip-agnostic per the guiding principle) | quiet drift | `tac.clip_profile` | LIVE WARN (P3 ×2) |
| 10 | `tools/dashboard_server.py:112-113`, `tools/dashboard_supervisor.py:468-469` | `goal_dseg=0.00092 / goal_dseg_15=0.00032` | derived break-even d_seg targets | goal lines drift from the equations leg; per memory these targets carry ERROR BARS (the 0.018 pose term was the BORROWED ancestor 3.4e-5) | quiet | canonical_equations / frontier pointer read-back (future) | inventory (semi-deliberate; env/flag-overridable) |

### Known-accepted (checked, NOT the bug class)

* `experiments/train_levelset_witness_realized_through_R_mlx.py` argparse defaults — TRAINER
  DEFAULTS (DSL compile target).
* `tools/dashboard_trajectory_model.py:487-488` `schedule.get("tau_start") or 300` — the
  ACCEPTED derive-first-with-fallback pattern (derives from the run's own schedule; the
  literal is a last-resort fallback in a function documented as such). P4 excludes it by design.
* gt-cache argparse defaults (`gt_n600.npz` etc. across ~15 tools) — provenance-pin-ish;
  every tool exposes `--gt-cache`; the path itself is the canonical shared-cache location.
  Watch item: if the cache layout moves, this becomes a class; candidate for a
  `tac.gt_cache.default_path(n)` helper later.
* EMA `0.997` in tools — the CLAUDE.md canonical constant (non-negotiable), or explicit
  historical provenance pins (`build_stage8_muonjump_checkpoint.py` reads the value FROM the
  source manifest — correct pattern).
* mod-dim `32` defaults in measurement tools — provenance pins for the measured config.
* `rss_mb=2500 / projected_gb=1.5` in dashboard launchers — resource caps for the TOOL
  process, not run properties (the trainer-side memory preflight derives from the real config).
* `legal_frame_feasibility_smoke.py --tau (float 5e-4)` — a tolerance, not the stage epoch;
  P1 requires `type=int` so it is excluded (verified).
* 874/1164 in build/byte-close/bench tools (~100 files) — deliberate: those tools must
  reproduce the exact measured bytes; the canonical home is `tac.clip_profile` and NEW
  display tools are gated (P3), but retro-editing measurement tools would risk byte-identity.

## Routings landed (fallback-and-visible-marker semantics)

All three follow the dashboard_server pattern: **derive by default (flag/env default None →
server runs the DSL schedule read-back), explicit value = OVERRIDE only, read-back failure →
visible "schedule: fallback" marker** (the fail-open lives in dashboard_server:814-832 /
schedule_readback.read_schedule, already tested for the no-launch.sh case by
test_schedule_readback.py).

1. `tools/dashboard_reload.py` — `--tau/--l7` default None; `dash_cmd` emits the flags only
   when explicitly given.
2. `tools/dashboard_supervisor.py` — `--tau/--l7` default None; conditional at all THREE
   forwarding surfaces (`_server_env` DASH_TAU/DASH_L7, `_spawn_server` argv, self-relaunch argv).
3. `tools/launch_witness_run.py` — the printed dashboard hint no longer carries `--tau/--l7`.

## WARN-gate

`src/tac/run_constant_gates.py::check_no_hardcoded_run_constants_in_consumers`
(**strict=False**, warn-only; heuristic scan, false-positive risk real). Patterns P1
(stage-flag int argparse default) / P2 (literal `--tau <int>`/`--l7 <int>` in strings) /
P3 (874/1164 in display tools only; live 0 → regression guard... plus 2 genuine viz hits) /
P4 (`tau_start|l7_start|muon_start = <int>` assignment). Waiver `# RUN_CONSTANT_OK:<rationale>`
(placeholder rejected). 20 tests in `src/tac/tests/test_run_constant_gates.py`.

**Live count at landing: 9** — rows 4/5/6/9 above (all genuine class members; the queue).

**Deferred with named blocker:** wiring into `preflight_all()` — the review-counter sibling
held ~200 uncommitted lines in `src/tac/preflight.py` at landing time; staging that file
would have absorbed the sibling's in-flight hunks (the absorbed-hunks class, recurrence 3+).
Standalone entry: `.venv/bin/python -m tac.run_constant_gates [--strict]`. Wire-in is a
one-line callsite once the sibling lands.

## Pointer

Exact frontier pointer **0.19110 UNMOVED** — this landing is apparatus/means (observability
correctness), not a score row.
