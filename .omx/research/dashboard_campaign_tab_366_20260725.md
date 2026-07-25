# Dashboard CAMPAIGN tab — live #366 joint-descent view (2026-07-25)

**Operator directive:** "found a fable subagent to update the live dashboard so I
can watch real time" (the #366 joint-descent CAMPAIGN).

## What landed

- **`src/tac/ddm_campaign_run_reader.py`** — canonical READ-ONLY contract for
  `ddm_366_campaign_*` run dirs (sister of `tac.witness_run_artifacts` for witness
  runs). Auto-discovers the LATEST run dir by freshest-signal mtime (latest-run law,
  `dashboard_refresh_shows_latest_run_zero_manual_20260711`); structural marker =
  `launch_manifest.json` argv invoking `launch_ddm_joint_descent.py --full-run`
  (a run PROPERTY, never a name; excludes the 39 ws*/smoke/preflight probe dirs
  that share the run-identity schema). Incremental mtime-gated parse of
  `telemetry/step*.json` (`ddm_joint_descent_full_run_step.v1`, ADVISORY_BATCH_LOCAL),
  `verdicts/*_n600.json` (`ddm_joint_descent_chunked_stage_verdict.v1`, EXACT n600
  advisory axis), geometry events, `checkpoints/*.npz` count (= accepted steps),
  `full_run_receipt.json` (schedule + stage targets), ticket-schedule fallback for
  LIVE runs (receipt-less) via launch_manifest `--ticket`. Launcher liveness via
  run.pid + psutil cmdline scan. Snapshot schema
  `ddm_campaign_dashboard_snapshot.v1`, `score_claim=false` always.
- **`tools/dashboard_server.py`** — new `/api/campaign` route (gated like
  `/api/state`; server-cached ≥4 s, thread-locked) + a **CAMPAIGN tab** in the LIVE
  instrument: exact-n600 d_seg/d_pose verdict traces with the sealed stage-target
  lines (0.020603 / 0.013735 / 0.006868; d_pose 163.061) and ema(●)/live(○)
  parameter_shadow marks; batch-local per-step initial→final d_seg descent strip
  (labelled ADVISORY_BATCH_LOCAL, never conflated with n600); gradient_norm;
  seconds/step vs the sealed 312 s/step budget line; pose-finish engage-gate panel
  (classification, exact-verdict history, derived earliest-engagement window from
  settle_window+hysteresis); per-class d_seg bars (canonical comma10k order) from
  the latest exact verdict; status strip (run dir, pid ALIVE/DEAD, telemetry/
  verdict/log freshness, accepted-checkpoint count, receipt verdict + pointer).
  Client polls `/api/campaign` every poll interval while the tab is active.
  All charts honest-axis (y floor at 0, targets included in range); tabular-nums.
- **Tests:** `src/tac/tests/test_ddm_campaign_run_reader.py` (12) over REAL fixture
  rows copied from `ddm_366_campaign_v5_cured_20260725T062259Z`
  (`src/tac/tests/fixtures/ddm_campaign_run/`, JSON-only; gitignored artifact
  classes synthesized per-test).

## How to watch

Open the dashboard (`tools/dashboard_ctl.py ensure-up`, port 8790 →
`http://127.0.0.1:8790/#live/campaign`, or the tunnel at comma-lab.adpena.com with
the access key) → LIVE meta-tab → **CAMPAIGN** tab. New attempts (fresh
`ddm_366_campaign_*` dirs) are auto-followed — zero manual repoint.

## Coordination

Sister codex arm `ddm_ct1_campaign_telemetry_encode` owns the costate-digest rows
(`tools/costate_digest.py` untouched here); this landing owns the dashboard web
surface only.
