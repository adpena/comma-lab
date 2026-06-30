# Dashboard live-run observability + no-silent-failure launch hardening — landing

**UTC:** 2026-06-30T20:18:23Z · **Axis:** infrastructure / observability (advisory; NON-PROMOTABLE).
**Pointer:** 0.19110 UNMOVED — this is MEANS (tooling), not a score move. NO exact row touched.

## Source
Operator 2026-06-30: (1) "Ensure all errors provide detailed debugging information no silent
failures"; (2) "full observability" + "missing stages and details on the setup and config and
schedule and curriculum"; (3) telemetry-accuracy on the pose number; (4) "That needs to happen
automatically in the future." (Items 2–4 relayed via coordinator; folded in because they match the
operator's documented "telemetry accuracy vital" + "build the automated value-generator" principles.)

Live constraints honored throughout: the n600 training daemon
(`levelset_n600_v2_attrclean_20260630T194549Z`, pid 38641) was NEVER signalled; the :8790 dashboard
behind the cloudflared named tunnel stayed `healthz 200` (pre=200, mid=200, post=200 across a
SO_REUSEPORT zero-downtime reload).

## What landed

### Deliverable 1 — dashboard surfaces the LIVE run immediately (auto, every tick)
Root cause: `LiveState.refresh()` resolved the watched log via
`rld._resolve_watched_log` = newest-mtime log that ALREADY has a `verdict` line. A freshly-launched
run emits its config stages (`gt`/`front_end`/`structured_init`) for seconds-to-minutes BEFORE its
first verdict, so it was INVISIBLE (dashboard latched on the prior arm) until ~ep25 (~45 min).

Fix (`tools/render_levelset_dashboard.py` + `tools/dashboard_server.py`):
- New `_is_run_log` (run-identity = early `{"stage":"gt"}` line, or any verdict) + `_resolve_run_log`
  (newest-mtime run log, verdict-bearing OR warming up) — sibling of `_has_verdict`/`_resolve_watched_log`.
- `refresh()` now resolves BOTH; when the newest RUN log is strictly newer than the newest
  VERDICT-bearing log and has no verdict of its own → `warming_up=True`: meta + liveness follow it,
  its own (empty) trajectory renders as "warming up", never a foreign trajectory. Otherwise the
  prior behavior (verdict run + full resume-ancestry trajectory) is UNCHANGED.
- This re-resolves EVERY refresh tick → **all FUTURE launches auto-appear with NO manual repoint/reload.**
  The ONLY thing that ever needs a manual zero-downtime reload (`tools/dashboard_reload.py`) is new
  dashboard CODE, not a new run. `auto_latest` is preserved (no hard `--run-dir` pin).

### Full observability — CONFIG / SETUP / SCHEDULE / CURRICULUM panel
- `render_levelset_dashboard.parse_run_config(run_dir)` parses the run's OWN artifacts —
  `launch.sh` primary (full flag set), `run.log` stage-lines fallback — into
  `{source, flags, groups, schedule}`. Generalizable to ANY future run, zero hand-config.
- `meta()` exposes `config`, `schedule`, `warming_up`, `deploy_sidecar_d_pose`, and resolves the
  curriculum boundaries (`tau`/`l7`/`muon_start`) from the run's flags (not the dashboard defaults).
- HTML/JS: a collapsible `setup · config · schedule · curriculum` panel (config groups +
  curriculum stage epoch-ranges); the d_seg chart now draws ALL 4 stage bands at their
  flag-derived ranges including the FUTURE Muon band (xmax extended to `schedule.epochs`).
- LIVE verified: CE [0,300) · tau [300,600) · l7 [600,726) · Muon [726,1000); epochs 1000, eval 25;
  all 6 config groups populated; `config.source=launch.sh`.

### Telemetry accuracy — implied_S uses the DEPLOY stored-pose sidecar
The witness trains d_seg only (`--w-pose 0`); the verdict's `d_pose` (~163) is MONITORING-ONLY, so the
verdict's own `implied_S` (~68.99) is dominated by an untrained pose term (`sqrt(10·d_pose)` ≈ 40).
- `_slim` now recomputes the DISPLAYED `implied_S` with `DEPLOY_SIDECAR_D_POSE = 3.4e-5` (the solved
  Quantizr stored-pose sidecar): `100·d_seg + sqrt(10·3.4e-5) + 25·bytes/37_545_489`. At ep0 this is
  28.60 (tracks d_seg), not the misleading 68.99. The raw value is preserved as
  `implied_S_monitoring`; the d_pose chart is relabeled "MONITORING ONLY … deploy pose = stored
  sidecar"; the implied_S chart says "d_seg + DEPLOY sidecar pose + rate". Honest + adjustable
  (named constant). NO training change.

### Deliverable 2 — NO silent failures in the launch path
`tools/spawn_durable_daemon.py`: after `Popen`, `_verify_child_survived` bound-waits (`--verify-s`,
default 3.0s) and confirms the child survived exec. A child that exited nonzero (or rc=0 with a
failure marker in the log) → EXIT 4 with DETAILED debug (status, cmd[0], full cmd + len, cwd, log
path, failure-marker lines, 20-line log tail, collapse-to-argv[0] HINT) + the registry row marked
`stopped`. The optimistic "detached session" success line is now only printed when VERIFIED alive.
The direct-Popen path is also wrapped (exec-fail → detailed debug, no traceback). `--verify-s 0`
disables (back-compat).
`tools/safe_run.py`: spawn-failure messages now carry `len(cmd)`, `cmd[:3]`, `cwd`, and an explicit
collapse-to-one-arg HINT (the exact unquoted-shell-expansion class that masked the original failure);
"no command given" prints the received argv.

### Canonical automated launcher — `tools/launch_witness_run.py`
ONE command for future witness launches (no more hand-assembly + manual verify): (a) derive config
(`tac.witness_autoconfig`) + FLAG-VALIDATE every emitted flag vs the trainer's real argparse
(never-invent — refuses before writing); (b) write the command to `<out_dir>/launch.sh` (a SCRIPT →
daemon cmd is `bash launch.sh`, 2 clean tokens, the word-split bug structurally impossible);
(c) launch durably via `spawn_durable_daemon` (auto-verifies liveness); (d) verify the
`custom_grouped_backward active=true` perf line (loud warn on the ~17x-slow footgun); (e) confirm the
dashboard is up (auto-tracks once up). `--dry-run` emits+validates+writes only (CPU-only, safe).
Determinism/resumability preserved (`--seed`/`--ckpt-every`/`--stage-checkpoints` in the emitted cmd).

## Tests (78 green incl. regression)
- `src/tac/tests/test_no_silent_failure_launch_hardening.py` (15): `_verify_child_survived` running/
  nonzero/clean+marker/clean cases; `_scan_log_for_failures`; `_decode_exit`; `_do_start` direct-exec-
  fail → 4+detail; safe_run-wrapped dead child → 4+detail+registry stopped; live child → 0 VERIFIED;
  `--verify-s 0` back-compat; safe_run `_spawn_debug` collapse hint / not-found / no-command / valid.
- `src/tac/tests/test_dashboard_liveobs.py` (18): `_is_run_log`/`_resolve_run_log` warming-over-
  verdict; launch.sh flag + schedule parsing; `parse_run_config` launch.sh + run.log fallback;
  `_implied_s_deploy`/`_slim` recompute + monitoring preserve; `refresh()` warming-up (0 verdicts) +
  normal (schedule + deploy implied_S).
- `src/tac/tests/test_launch_witness_run.py` (11): flag validation (55/55) + invented-flag refuse;
  launch.sh structure + round-trip through the schedule parser; perf-env active/inactive/not-seen;
  dashboard-down; `--dry-run` no-spawn; refuse-before-write.
- Existing `test_dashboard_server.py` made hermetic (`auto_latest=False`) so it no longer races a live
  run's log mtime; `test_spawn_durable_daemon_{memguard,lifecycle}` still green.

## Live verification
`dashboard_reload.py --port 8790 --tau 300 --l7 600` → `reload_done ok=true` pre/mid/post healthz=200.
`/api/state`: run_dir=…levelset_n600_v2_attrclean…, n_pairs=600, training_alive=true, muon_start=726,
schedule stages CE/tau/l7/Muon, config.source=launch.sh, trajectory implied_S=28.60 (deploy) /
implied_S_monitoring=68.99 / d_pose=163.3 (monitoring). n600 trainer pid 38641 ALIVE, untouched.

## 6 wire-in hooks (per Subagent coherence)
research_only=true (advisory tooling). sensitivity-map N/A · Pareto N/A · bit-allocator N/A · cathedral
autopilot N/A (observability/launch infra, not a score-deployable substrate) · continual-learning:
the launcher + auto-track dashboard ARE the durable system-intelligence wire-in (every future launch
inherits the verified path) · probe-disambiguator N/A. Means, not ends — pointer 0.19110 unmoved.
