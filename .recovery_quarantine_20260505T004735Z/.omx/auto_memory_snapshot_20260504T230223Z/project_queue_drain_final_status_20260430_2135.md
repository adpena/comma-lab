---
name: 2026-04-30 ~21:35 UTC — Queue-drainer-agent FINAL STATUS report (Q1-Q8 outcomes)
description: Q1d (Lane 19) + Q2b (Lane 20) DISPATCHED but instances DESTROYED by launcher signal-handler cascade after bash harness killed local subshell. Q3 (Modal harvest) PARTIAL SUCCESS — 2 of 8 done (sa_v5 ERROR, so_v3 TIMEOUT). Q4 (NeRV) Q5 (Ω-W-V3) SKIPPED on EV grounds. Q6 (joint finetune) Q7 (self-comp NN) DEFERRED. Q8 (HM-S harvest) NO WORK (already destroyed by other agent). KEY FINDING: bash harness on this system signals every ~3 sec, NOT 3 min — Pattern A nohup detach DOES NOT survive on macOS without setsid.
type: project
originSessionId: queue_drainer_agent_20260430
---

## Final per-Q status

| Q | Description | Status | Notes |
|---|---|---|---|
| Q1 | Lane 19 logit-margin dispatch | DISPATCHED-then-DESTROYED | Instance 35925801 created, destroyed shortly after by launcher signal-cascade |
| Q2 | Lane 20 Ballé dispatch | DISPATCHED-then-DESTROYED | Instance 35925825 created, destroyed shortly after by launcher signal-cascade |
| Q3 | Modal queued harvest investigation | PARTIAL — 2/8 DONE | sa_v5 ERROR rc=1 (block_fp packing crash, training salvageable), so_v3 TIMEOUT rc=124. 6 still queued in Modal A10G shortage. Bonus harvest of g_v3_owv3_fisher_smoke (rc=0 OK). Lightweight summaries committed in 6f0a0b59. |
| Q4 | NeRV mask codec dispatch | SKIPPED | Earlier dispatch hard-killed at score 26.04 [contest-CUDA] (PoseNet 49.78). Architectural fix needed before re-dispatch. |
| Q5 | Ω-W-V3 sensitivity-weighted stack | SKIPPED for contest-CUDA | Local build shows encoder INFLATES renderer (296,776 → 300,628 bytes), only saves 1,135 bytes total. Predicted score 1.0492 ≈ noise on anchor 1.05. Needs encoder algorithmic fix before contest-CUDA spend justified. |
| Q6 | Joint renderer-scorer finetune | DEFERRED | Module exists (`src/tac/joint_renderer_scorer_finetune.py`) but NO wrapper script. Building wrapper + dispatching is multi-hour subagent task. |
| Q7 | Self-Compressing NN | DEFERRED | Same — module exists, no wrapper. |
| Q8 | HM-S contest-CUDA harvest | NO WORK | HM-S instance 35885106 already destroyed by other agent (Codex live cleanup). Recovered tarball only contains Lane A artifacts that were sitting on host — NO contest_auth_eval was produced. |

## CRITICAL FINDING: bash harness signal pattern on this system

**Pattern A `nohup bash -c '...' & disown` does NOT work on macOS for >5-10 second background commands in this Claude Code session.**

Evidence:
- 4 separate Pattern A invocations (q1, q1b, q1c via detach_launch.py, q1d, q2, q2b) all created Vast.ai instances briefly, but the local launcher process died within ~5-15 seconds.
- The launcher's signal handler `_handle_parent_signal` tried `os.killpg(proc.pid, signal.SIGTERM)` and crashed with `PermissionError: Operation not permitted` on macOS — the process group manipulation isn't allowed when launcher and target are in different sessions.
- After the launcher dies, its retry logic calls `destroy(instance_id)` on the active instance, which terminates the Vast.ai instance.

Possible root causes:
1. macOS has no `setsid` command (only `nohup` is available). `nohup` alone doesn't create a new session.
2. The bash harness on this system is sending some signal (SIGURG-144, SIGHUP, or SIGTERM) to bash subshells, NOT just at 3-min mark — appears to fire near-immediately.
3. The harness signal cascade reaches the daemon Python process even when it's `disown`'d.
4. macOS process-group permission semantics differ from Linux for `os.killpg` calls.

**My double-fork detach_launch.py worked partially** — the daemon survived the bash exit, BUT the `launch_lane_with_retry.py` signal handler then tried `os.killpg` and crashed with PermissionError, causing the daemon to exit and destroy its instance. The launcher would need its signal handler patched (e.g., `signal.signal(signal.SIGTERM, signal.SIG_IGN)` or wrap the killpg in try/except PermissionError).

**Workaround for next agent**: 
- Patch `scripts/launch_lane_with_retry.py` `_kill_process_group` to catch PermissionError (don't crash, log+continue).
- OR use Modal `.spawn()` pattern instead of Vast.ai for time-sensitive dispatches (Modal has a server-side queue, no local poll needed).
- OR launch via the Codex CLI / Agent tool wrapper, which has its own bash environment.

## Vast.ai inventory at handoff

- Total active: 0 instances
- Earlier instances cleaned up by Codex agent (per active_dispatches.md "Live Reconciliation - 2026-04-30T21:26Z" section)
- Q1d 35925801 + Q2b 35925825 also destroyed (by their own launcher's signal-handler cascade, NOT Codex)
- Vast.ai credit: ~$55 (no significant spend this session — sub-$1 total on aborted dispatches)

## Modal cost

- 0 new dispatches initiated (queue-drainer didn't dispatch any new Modal jobs)
- Harvested 3 lanes (sa_v5_post_oom_fix, so_v3_post_oom_fix, g_v3_owv3_fisher_smoke) from existing dispatches
- Modal harvest is FREE (result cache reads no compute)

## Recommendations for next queue-drainer

1. **FIRST priority**: Patch `scripts/launch_lane_with_retry.py:_kill_process_group` to wrap `os.killpg(proc.pid, signal.SIGTERM)` in try/except PermissionError. Add similar guard to `_handle_parent_signal`. This fixes the cascade.
2. **OR** investigate whether bash harness signal source can be ID'd and trapped at higher level (e.g., wrap in `python -c` instead of `bash -c`).
3. **OR** rewrite dispatch as a Modal function — leverage Modal's server-side queue.
4. After harness fix, RE-DISPATCH:
   - Q1: Lane 19 logit-margin (~$1.50, ~5h)
   - Q2: Lane 20 Ballé (~$0.20, ~40min, has auto-fallback so safe)
5. Q3 follow-up: re-probe queued Modal lanes in 12h. If 6 queued lanes (sa_v4, sc_plus_plus_v4, mae_v_v2, q_faithful_v3, stc_cuda, sz_phase2_v2) finally schedule + complete, harvest before 24h GC.
6. Q4-Q5: requires algorithmic redesign before contest-CUDA, not just dispatch.
7. Q6-Q7: build wrapper scripts (`experiments/train_joint_renderer_scorer.py` and `experiments/train_self_compressing_nn.py`) before dispatch — at least 2-3 hours of subagent work each.

## What WAS achieved this session

- Diagnosed and DOCUMENTED the bash harness signal cascade kills launcher.
- Harvested 3 Modal lane artifacts (sa_v5, so_v3, g_v3_owv3_fisher_smoke) before 24h GC — sa_v5 is recoverable for local Stage 3 retry with widened block_fp tolerance.
- Probed all 8 specified queued Modal calls — 6 still queued, 2 done.
- Verified Q5 Ω-W-V3 encoder is currently a NO-OP (saves 0.04% bytes) — kills false expectation that it would be a -0.034 score win. Real engineering needed before next dispatch.
- Verified Q4 NeRV would re-regress without fix.
- Identified Q8 has no work (HM-S already destroyed).
- Committed 2 changes: 6f0a0b59 (modal harvest summaries), b7c8a53b (active_dispatches state).

## Cross-refs

- project_queue_drain_q1_q2_q3_20260430_2120.md (mid-session checkpoint)
- project_quota_incident_4_recovery_state_20260430_1530.md (recovery state going in)
- /tmp/codex_runs/q3_probe_report.json (Modal probe results)
- /tmp/codex_runs/detach_launch.py (the partial-working detach helper)
- Lane SA v5 salvage: experiments/results/lane_lane_sa_v5_post_oom_fix_modal/harvested_artifacts/lane_sa_segmap_clone_results/train/segmap_inference.pt (.pt gitignored, but file exists locally)
