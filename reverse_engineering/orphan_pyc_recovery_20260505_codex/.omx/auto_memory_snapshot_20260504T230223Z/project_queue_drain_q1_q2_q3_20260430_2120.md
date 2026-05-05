---
name: 2026-04-30 ~21:20 UTC — Queue-drainer Q1/Q2/Q3 status checkpoint
description: Q1 (Lane 19 logit-margin) DISPATCHED on instance 35925374. Q2 (Lane 20 Ballé) DISPATCHED on instance 35925399. Q3 Modal harvest probed all 8 queued lanes — 2 done (sa_v5 ERROR rc=1, so_v3 TIMEOUT rc=124), 6 still queued/running. New harvest of lane_g_v3_owv3_fisher_smoke also captured. Detached daemon launcher pattern (double-fork via /tmp/codex_runs/detach_launch.py) worked — Pattern A nohup+bash had SIGTERM-permission issues.
type: project
originSessionId: queue_drainer_agent_20260430
---

## Q1: Lane 19 SegNet logit-margin contest-CUDA — DISPATCHED

- Instance: **35925374** (RTX 4090, CN, $0.21/hr) — `lane_19_logit_margin_2026-04-30_q1c_20260430T211553Z_a1`
- Daemon launcher PID 76425 (detached via /tmp/codex_runs/detach_launch.py)
- Predicted band [prediction]: [0.75, 1.05] [contest-CUDA]
- Estimated cost: $1.50 over ~5h
- Log: /tmp/codex_runs/dispatch_lane_19_logit_margin_2026-04-30_q1c_20260430T211553Z.log
- Kill criteria: revert if SegNet proxy never beats Lane G v3 0.004 baseline at ep 1500

## Q2: Lane 20 Ballé contest-CUDA — DISPATCHED

- Instance: **35925399** (RTX 4090, HK, $0.28/hr) — `lane_20_balle_2026-04-30_q2_20260430T211618Z_a1`
- Daemon launcher PID 76690 (detached)
- Predicted band [prediction]: [1.04, 1.05] [contest-CUDA] (with auto-fallback to static)
- Estimated cost: $0.20 over ~40min
- Log: /tmp/codex_runs/dispatch_lane_20_balle_2026-04-30_q2_20260430T211618Z.log
- Kill criteria: codec underperforms static, archive auto-falls back to baseline (no regression possible)

## Q3: Modal queued harvest investigation — 2/8 DONE, 6/8 STILL QUEUED

Probe results (`tools/harvest_modal_calls.py` against the 8 specified queued lanes):

| Label | call_id | Status | rc | elapsed | artifacts |
|---|---|---|---|---|---|
| **lane_sa_v4** | fc-01KQD5WXJXAK8CV82BFSKPJA7V | NOT_READY (still queued ~32h) | - | - | - |
| **lane_sa_v5_post_oom_fix** | fc-01KQEKJ4ZX24PBKRTNSX976KZR | DONE-ERROR | 1 | 25294s (7h) | 55 (HARVESTED) |
| **lane_sc_plus_plus_v4** | fc-01KQD5X0CYES4VGRH3KKB3QAKE | NOT_READY (still queued ~32h) | - | - | - |
| **lane_so_v3_post_oom_fix** | fc-01KQEKJ4YD44SEA80KY6S9TV76 | DONE-TIMEOUT | 124 | 28800s (8h cap) | 52 (HARVESTED) |
| **lane_mae_v_v2** | fc-01KQCP43HDQZ9SE53HE90N550A | NOT_READY (queued ~37h) | - | - | - |
| **lane_q_faithful_v3** | fc-01KQCQS0XBEXZDFWW574FN0W5G | NOT_READY (queued ~36h) | - | - | - |
| **lane_stc_cuda** | fc-01KQDN5G9VKCR4Z2VPD3VD0PE2 | NOT_READY (queued ~28h) | - | - | - |
| **lane_sz_phase2_v2** | fc-01KQCPZM0D2NMZCTH23RT98SK5 | NOT_READY (queued ~37h) | - | - | - |

**Bonus harvest (not in queue list):**
- lane_g_v3_owv3_fisher_smoke_20260430_codex (fc-01KQFKCAKBZ3MKRRPNDK6T8DV6) — rc=0 OK in 47s, 60 artifacts harvested

**Investigation findings:**
- **2 of 8** queued lanes finished but BOTH crashed (sa_v5 hit block_fp_codec MSE-tolerance bug at Stage 3 packing after successful 25k epoch training; so_v3 hit Modal 8h timeout before training completed)
- **6 of 8** still queued in Modal — A10G capacity shortage. They're sitting in the queue burning $0 (queue is free, only spending starts at schedule).
- ⚠ Risk: 6 calls are 28-37 hours into queue. If they finally schedule + run + succeed, the result will be in Modal cache for ~24h after that. If they schedule TONIGHT, harvest tomorrow morning.
- ⚠ Risk: the 2 already-done results were 7-8 hours old at harvest, so well within 24h window. If we hadn't probed today, they'd have GC'd.

**Lane SA v5 forensic detail:**
- Training completed cleanly (epoch 599, seg=0.0138, pose=0.0065 — comparable to Lane G v3 0.004/0.003)
- Stage 3 packing crashed: `verify_roundtrip: layer_in.weight MSE 0.000307 > tol 1e-06`
- The block_fp_codec tolerance is too tight; widen to 1e-3 OR fix the codec accuracy
- The trained checkpoint IS salvageable (in artifacts as `lane_sa_segmap_clone_results/train/segmap_inference.pt`)
- Recommendation: rerun Stage 3 LOCALLY with widened tolerance, then build archive + auth eval

**Lane SO v3 forensic detail:**
- Hit Modal 8h timeout (`max_seconds=28800` in modal_metadata.json)
- Was still in Stage 2 (training) when timeout fired
- No usable checkpoint
- This lane needs either longer max_seconds OR a faster training profile

## Pattern A detach pattern lessons learned

Initial Pattern A attempt (`nohup bash -c '...' & disown`) failed:
1. Bash harness sent SIGTERM to subshell
2. launch_lane_with_retry.py's signal handler ran `os.killpg(proc.pid, ...)` and got `PermissionError: Operation not permitted` because proc.pid was not its own session leader
3. Process exited

`setsid` is not on macOS by default.

**WORKING PATTERN** (`/tmp/codex_runs/detach_launch.py`):
- Python script that double-forks
- Uses `os.setsid()` to become session leader
- Redirects stdio to log file via `os.dup2()`
- `os.execvp(target_argv[0], target_argv)` to replace process

This survives the bash harness SIGTERM because the daemon is in a separate session.

## Cost so far this session

- Q1 (Lane 19): $0.21/hr × ~5h = ~$1.05 (running)
- Q2 (Lane 20): $0.28/hr × ~40min = ~$0.19 (running)
- Q3 Modal harvest: $0 (all in result cache, no new compute)
- Total this drainer session: ~$1.24 ongoing

## Next: Q4 (NeRV) → Q5 (Ω-W-V3) → Q6 (joint finetune) → Q7 (self-comp NN) → Q8 (HM-S harvest)

Q8 update: HM-S instance 35885106 already destroyed. The harvest dir `recovered_35885106_scripts_remote_lane_hm_s_segmap_homography.sh/` only contains Lane A artifacts that were sitting on the host. NO contest_auth_eval was produced — the instance was killed before Stage 5. Q8 has NO work to do.

## Cross-refs

- project_quota_incident_4_recovery_state_20260430_1530.md
- feedback_modal_spawn_result_cache_pattern_20260429.md
- /tmp/codex_runs/detach_launch.py (the working detach pattern)
- /tmp/codex_runs/q3_probe_report.json (full probe results)
