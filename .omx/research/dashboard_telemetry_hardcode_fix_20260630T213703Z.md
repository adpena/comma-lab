# Dashboard time-telemetry: hardcode → DATA-DERIVED (cadence/ETA fix + sophisticated trajectory model)

UTC: 2026-06-30T21:37:03Z · authority: `[macOS-MLX] advisory · NON-PROMOTABLE` · pointer 0.19110 UNMOVED.
Dashboard-only change. The live n600 training (pid 38641) + cloudflared tunnel were untouched; the
dashboard was hot-reloaded zero-downtime (SO_REUSEPORT) with healthz 200 verified before+after.

## The bug (operator 2026-06-30: "hardcoded garbage")

The dashboard's cadence/liveness telemetry false-flagged the live n600 run as stale/slow and showed
wrong ETAs because the cadence used a **HARDCODED prior** `cadence_prior_min = 18.0` (min) that did not
match the real ~43-min n600 inter-verdict cadence (104.5 s/epoch × eval-every-25 ≈ 2613 s). Direct
violation of the "Telemetry accuracy vital" non-negotiable (every readout accurate OR understand+FLAG
when it isn't; never hardcoded/mis-thresholded). Measured live gaps: ep0→25 = 2613 s, ep25→50 = 2344 s
(median 2478.5 s ≈ 41 min). Async verdict itself takes ~200–240 s (`verdict_async_done` secs in the log).

## What was removed (hardcoded constants → data-derived)

| Removed/retired constant | Was | Now |
|---|---|---|
| `_CADENCE_PRIOR_MIN = 18.0` (min) | a fabricated cadence presented as if measured | **gone** — inert shim (0.0); cadence is the run's MEASURED gap; "calibrating" until the 1st gap |
| `_STALE_FLOOR_MIN = 10.0` (min) | hardcoded minute floor in `max(floor, K×cadence)` | **0.0** by default (no floor); opt-in only via `--stale-floor-min` |
| `_CADENCE_K = 2.5` | stale multiple | `2.0` (operator's 1.5–2.0 range); dimensionless POLICY knob, not a time |
| `_MIN_CADENCE_SAMPLES = 3` | needed 3 arrivals (2 gaps) before "measured" → 18m prior for the whole 2-verdict window | `_MIN_CADENCE_GAPS = 1` — **>=1 measured gap (>=2 verdicts) => measured** |
| `dashboard_server._cad_args = SimpleNamespace(... cadence_prior_min=18.0, stale_floor_min=10.0, cadence_k=2.5)` | the hardcoded seeds | `stale_floor_min=0.0, cadence_k=rld._CADENCE_K`; `eval_every`/`preferred_cadence_s` filled PER REFRESH from the run's own schedule + measured current-stage rate |

## New data-derived cadence/liveness logic (`tools/render_levelset_dashboard.py`)

1. **Cadence = MEASURED inter-verdict gap.** `_measure_cadence` returns `median(positive gaps)` as soon
   as there is >=1 gap (sourced from the verdict `ts`, or self-observed arrivals for ts-less logs). With
   the trainer's per-verdict `ts` this is the **2nd verdict** — the live n600 reads its real ~41-min
   cadence immediately, never the 18-min prior.
2. **Bootstrap estimate** before any gap: if an independent seconds/epoch is available
   (`eval_every × spe`), cadence is a clearly-LABELED `"estimate"`; otherwise `"calibrating"` (no number).
   Never a hardcoded constant.
3. **Next-verdict ETA = last_verdict + measured cadence** (`max(0, cadence − verdict_age)`) — an honest
   countdown that can legitimately be ~33–40 min for n600.
4. **Stale = K × MEASURED cadence + async grace, DOUBLE-GATED + async-aware.** `_async_grace_s` parses the
   MAX `verdict_async_done` secs from the log (≈240 s). STALE fires only when cadence is KNOWN AND the
   verdict is overdue past `K×cadence + grace` AND **the log file is also quiet** past `cadence + grace`
   (genuinely hung). So a healthy run mid-cadence — or mid in-flight async eval (log still being written
   by checkpoints) — is NEVER flagged. While calibrating (no gap) we never time-stale; a dead process in
   that window surfaces via the SEPARATE `meta.training_alive` signal, not a fabricated minute bound.

## Upgraded trajectory math (`tools/dashboard_trajectory_model.py`, pure numpy, testable)

The prior projection was a naive linear extrapolation. Replaced with the lab's actual deep-math
(grounded in `.omx/research/post_muon_application_plan_optimal_form_20260630T1710Z.md`: Agmon–Benger–
Ordentlich–Tishby ISIT 2021 arXiv:2103.02646 critical-slowing near a rate-distortion topological
transition; Rose 1998 deterministic annealing):

- **CRITICAL-SLOWING POWER LAW** `d_seg(t) = c + a·(t−t0)^(−α)` — `fit_critical_slowing` fits (c,a,α,t0)
  via a two-stage grid (nonlinear α,t0; closed-form linear LS for c,a) with local refinement to lift the
  (α,t0,c) degeneracy. Surfaces the projected asymptote `d_seg_∞`, exponent α, R², and a **confidence
  flag** (high/medium/low by n + R²). <5 points → `ok=False` ("calibrating", never confident-wrong).
- **GOAL-ETA WITH CONFIDENCE BANDS** via the model; honest `"asymptote_above"` ("won't reach at current
  trajectory") when the projected asymptote is above the target; band from the residual std.
- **STAGE-AWARE WALL-CLOCK**: `per_stage_seconds_per_epoch` measures seconds/epoch PER stage (Muon is
  slower — Newton–Schulz); stages not yet entered get a clearly-FLAGGED estimate (Muon ≈ 1.6× the slowest
  measured AdamW-stage rate, never silently CE's rate). `completion_eta` sums the right per-stage rate over
  remaining epochs (flags the estimated portion). `current_stage_cadence` = current stage rate × eval-every,
  recomputed each tick so a stale-stage rate is never carried across tau@300 / l7@600 / muon@726.
- **implied_S projection** = `100·d_seg_∞ + √(10·DEPLOY_SIDECAR_D_POSE) + 25·bytes/37.5M` with the d_seg
  band propagated and bytes modeled from the live blob trend. Uses the SOLVED stored-pose sidecar
  (telemetry accuracy), not the monitoring pose.

All inputs are the run's OWN verdict trajectory + `meta.schedule`; nothing is n600-special-cased (works
for n200 / n600 / residual-INR). Surfaced in `/api/state.projection`; the JS only RENDERS the
server-computed numbers + their flags (the fit math is NOT in JS).

## Verification (live, post-reload)

- `/api/state` liveness: `kind=live`, `cadence_source=measured`, `cadence_s≈2478.5` (the REAL ~41-min
  n600 cadence, not 18 m), honest next-verdict ETA, NOT falsely stale.
- `/api/state.projection`: `ok=True`, `stage=CE`, stage-aware `next_verdict_cadence_source=measured`,
  `dseg_model.ok=False` at 3 verdicts (correctly "calibrating" — fit needs ≥5; not confident-wrong).
- `/healthz` 200 before AND after the zero-downtime reload.

## Tests (99 green)

- `experiments/tests/test_dashboard_telemetry_hardcode_fix.py` (12): cadence-from-1-gap; n200 vs n600 each
  measure their own (no shared 18m); calibrating pre-data (no number, no false stale); stale = K×measured;
  async-in-flight NOT stale; double-gated hung = stale; async grace measured from log; ETA = last+cadence;
  bootstrap estimate labeled; retired constants inert; stage-aware preferred-cadence override.
- `experiments/tests/test_dashboard_trajectory_model.py` (21): power-law asymptote/α recovery (noiseless +
  noisy); few-points→calibrating; low-confidence flag; goal-ETA band monotonicity; "won't reach" honesty;
  per-stage spe (CE measured / Muon flagged estimate); two-stage measured-separately; completion-ETA
  Σ-over-stages + flagged estimate; current-stage cadence recompute at boundary; implied_S band; end-to-end
  build_projection; never-raises-on-garbage.
- `experiments/tests/test_render_levelset_dashboard_self_follow.py` (52): updated the 5 tests that encoded
  the OLD hardcoded contract (prior/floor/2.5/min_samples) to the new data-derived contract; +2 new
  (overdue-but-log-fresh NOT stale; calibrating never time-stales).

## Wire-in / observability

The fix is the canonical liveness path imported by BOTH dashboards (`render_levelset_dashboard` +
`dashboard_server`), so the meta-refresh and ASGI dashboards never disagree. New `projection` key in the
WS snapshot/update + `/api/state`. Generalizes to any future run with zero hand-config.

Cross-refs: CLAUDE.md "Telemetry accuracy vital" + "Max observability"; memory
`[[telemetry-accuracy-vital-or-know-when-not-20260627]]`;
`.omx/research/post_muon_application_plan_optimal_form_20260630T1710Z.md` (critical-slowing grounding).
