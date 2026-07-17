# Launcher-chain "death" postmortem — run 20260716T211713Z (c2_drystart_v2_cadence25)

**Date:** 2026-07-17 (~01:15Z) · **Author:** durability P0 subagent (p0_launcher_chain_durability_20260717)
**Axis:** [apparatus/forensics — non-promotable]. Pointer 0.19108 UNMOVED; everything here is MEANS.

## VERDICT (headline): THE CHAIN NEVER DIED. The "death at ~23:47Z" was a PHANTOM — a mis-measurement produced by three stacked observability failures, not a kill.

Evidence grade per claim below: **MEASURED** = re-derivable from an artifact/command named here; **INFERRED** = consistent reconstruction.

## The actual timeline (MEASURED)

| UTC (07-16/17) | Event | Evidence |
|---|---|---|
| 21:17:12.7 | outer safe_run pid=pgid **59278** spawned via spawn_durable_daemon (`--timeout 21600`, detached: pid==pgid, ppid==1) | `.omx/state/durable_daemons.json` row `c2_drystart_v2_cadence25`; `ps -o pid,pgid,ppid -p 59278` |
| 21:17:13–17 | launcher gates ran; daemon log `.omx/tmp/c2_drystart_v2.log` last flushed byte at 21:17:17 (5.4K) | file mtime + content ends at the `custom_grouped_backward` row |
| 21:17:23 | PASS 1 (dry_start) inner safe_run child group 60051 started, timeout 9000s (=1800+3×2400) | registry row `saferun_..._pid60011`, `started_utc` |
| ~21:59–23:45 | pass-1 trainer (pid 60052) warm-start epochs 651→654 at ~1612 s/ep (accum ~69s + **span_epoch_tail ~1543s = the mod-dim ablation observer at ckpt-every-1 cadence** — the §C confound) | `dry_start/witness_component_wallclock.jsonl` ep651–653; `dry_start/run.log` |
| 23:47:23.9 | pass-1 inner safe_run **CLEAN timeout**: `SAFE_RUN {"status":"timeout","exit":124,"peak_rss_mib":84009,"elapsed_s":9000.5}`; registry row marked `stopped/safe_run_exit_timeout` | first line of `dry_start/run.log`; registry row |
| 23:47:24 | launcher SURVIVED, persisted `dry_start/run.log` (mtime 23:47:24) | file mtime |
| 23:47:25.5 | **PASS 2 launched** (dry_start_resume; inner safe_run pid 67363, child group 67379, timeout 9000s → expiry ~02:17:25Z) | registry row `saferun_..._pid67363`; `dry_start_resume/launch.sh` mtime 23:47:25 |
| 23:47:31 | pass-2 trainer (pid 67380) began boot (`causal_manifest.jsonl` created) | file mtime |
| 01:03Z (07-17) | **ALL FOUR PROCESSES ALIVE**: 59278 (Ss, 3h45m), 67363, 67379, **67380 R-state, 100% CPU, ~44 GiB RSS**; pass-2 already wrote `levelset_resume_state.npz` + EMA in `dry_start_resume/` (resume round-trip on track) | `ps -o pid,pgid,ppid,stat,etime,rss -p 59278,67363,67379,67380`; `ls dry_start_resume/` |

Expected completion: pass-2 inner timeout ~02:17:25Z → launcher parses, writes `dry_start_report.json` (OLD code shape — the chain runs the pre-fix launcher loaded at 21:17), daemon log flushes on exit.

## Why it LOOKED dead (the three stacked failures — each MEASURED tonight)

1. **Block-buffered daemon log.** The launcher's stdout is redirected to the daemon log → 8KiB block buffer. Everything after the last pre-pass-1 flushed line (5.4K at 21:17:17) — including the `# dry-start PASS 1/PASS 2` narration — sat in the launcher's buffer for 4+ hours while it blocked in `subprocess.run`. A frozen log ≠ a dead chain. (Memory `launcher_buffered_log_not_hung_orphan_spawn_respawn_id_collision_20260715` warned EXACTLY this; the recurrence shows a memory without a tool does not stop the misdiagnosis.)
2. **Mis-fired process check.** A `ps ax | grep -E "launch_witness_run|safe_run|train_levelset"` pipeline returned empty tonight (reproduced by this agent, rc=1) while a direct `ps -p 59278,67363,67379,67380` showed all four alive — the grep pipeline (rtk-proxied) is not a trustworthy liveness instrument. The main agent's "both vanished" observation is INFERRED to be the same instrument class failing.
3. **Registry says "running" unconditionally.** `durable_daemons.json` reconciles exits only on explicit `--reconcile` — its status column carries no liveness signal in either direction (documented in sdd's own docstring L27-30).

Plus a priming effect (INFERRED): pass-1's trainer genuinely disappearing at 23:47:23 (the *intended* safe_run timeout kill) made "everything died at 23:47" plausible; pass-2's ~40-min silent boot (captured stdout, no disk writes between causal_manifest at 23:47:31 and the first checkpoint ~00:5x) left nothing to contradict it.

## The suspects, adjudicated

- **Harness 5-min sweep / SIGURG-144 class** (`daemon_5min_harness_long_call_sweep_kill`, `sigurg_144_...`): NOT the cause tonight (nothing was killed). The class itself remains real (ledger anchors stand).
- **Sandbox-teardown non-durability** (sdd docstring, EMPIRICAL ANCHOR 2026-07-07): NOT the cause tonight; the chain was properly detached (pid==pgid, ppid==1) and survived. Signature table in the docstring remains the reference for a REAL silent death.
- **Memory governor / jetsam / reboot / sleep**: all MEASURED absent (governor+blackbox logs empty on 07-16 for this window; `kern.boottime` = Jul 2; no pmset sleep events in the window).

## The REAL bug that pass-1 measured (the §C confound — MEASURED, unchanged from the prompt)

`--ckpt-every 1` (bench crash-resume lever) × default-ON `--mod-dim-ablation` (fires at checkpoint cadence, trainer L13379; gate `_mdd_abl_on` L8698/L9036) = ~1543 s/epoch observer tail → bench "measured" ~1612 s/ep for a config whose real amortized pace at `--ckpt-every 25` is `~69.4 + 1543/25 ≈ 131 s/ep ≈ 2.2 min/ep`. Fixed in commit `0860542e42` (bench passes inject `--no-mod-dim-ablation`; receipt decomposes typical/extra/amortized with provenance labels).

## ⚠ Standing hazard until superseded

When the live v2 chain completes (~02:17Z) it will write a **GREEN receipt whose sec_per_ep fields are CONTAMINATED** (pass-1 gross ~2250 s/ep, observer ON) for typed_config_hash `2d486e3bff…`. It validly proves boot/resume/peak-RSS (peak 84009 MiB MEASURED); its **sec/ep must NOT drive any wall-clock decision** — the fixed delta re-bench receipt (with `bench_marginal_decomposition`) supersedes it for pace.

## Actions landed (commit `0860542e42`)

B1 failure receipts + per-pass progress (real-SIGTERM kill-smoke PASSED) · B2 flush-safe logging (SIGKILL flush-survival smoke PASSED) · B3 durable-spawn marker + harness-child WARN · B4 `tools/witness_chain_watchdog.py` composite liveness (pid tree × run-dir mtimes × receipt; live-fired — reported this very chain ALIVE while its log looked dead) · §C bench-validity fix + decomposition receipt.

**The class-fix lesson:** liveness verdicts must come from a TOOL that composes kernel-truth signals (`witness_chain_watchdog.py`), never from log tails, grep pipelines, or registry status — and a chain must leave receipts (green, failed, or progress) at every exit so silence is itself a signal.

## ADDENDUM (01:30Z) — instrument #2 root-caused: rtk-proxied shell pipelines are LOSSY

Three concrete same-session instances (all MEASURED tonight): (a) `ps ax -o ...,command | grep -E
"launch_witness_run|safe_run|train_levelset"` returned rc=1 while 4 matching processes ran; (b)
`ps ax -o rss= | awk '{s+=$1}'` summed 0.5 GiB while one live trainer alone held 44 GiB; (c) a
`sort -rn | head` top-RSS list showed only low-pid system daemons. Mechanism: the rtk command proxy
(user-global CLAUDE.md hook) applies ~80-char line truncation and ~200-row result caps to proxied
pipelines — the ps `command` column is cut BEFORE the tool names (grep can never match) and row caps
sample the low-pid daemons. Direct `ps -p <pids>` and python-`subprocess` ps (what
`tools/witness_chain_watchdog.py` uses) were consistently correct. OPERATIONAL RULE (binding): process
liveness/RSS verdicts NEVER come from shell pipelines in an rtk session — run the watchdog.

Also corrected: the memory-governor REFUSE at 01:0xZ was RIGHT, not an accounting artifact — "used
62-66 GiB" included the live pass-2 trainer's 44 GiB and "active-growth 27.6 GiB" was that trainer's
registered growth-to-peak (71.54 − 43.96). The governor was protecting the live chain from a competing
launch. (The balloon experiment was unnecessary-but-harmless; the 20-GiB cache reclaim it produced did
not change the correct refusal.)
