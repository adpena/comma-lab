---
name: Q-FAITHFUL has NO recoverable trained checkpoint — all 4 prior attempts failed differently; only path forward is H100 SXM retrain
description: 2026-05-01 ~23:35 UTC. Investigated user's question "did Q-FAITHFUL ever produce a trained checkpoint we can recover from instead of retraining". Verdict: NO. v1 + v2 Modal runs CRASHED (missing --tag, --auth-eval-on-best incompatibility); v3 dispatched but never harvested (24h Modal TTL); RTX 4090 ep 810 P2 was preempted (instance destroyed, disk gone). Files in lane_q_faithful_*_modal/submissions/robust_current/ are scaffold from OTHER lanes (FP4A magic, has motion module, base_ch=36 — NOT JointFrameGenerator). H100 SXM retrain is the only recovery path.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The 4 prior Q-FAITHFUL attempts (forensic)

| Attempt | When | GPU | Status | Why no checkpoint |
|---|---|---|---|---|
| Modal v1 | 2026-04-29T13:31Z | T4 | CRASHED | `train_renderer.py: error: the following arguments are required: --tag` → fix landed commit 85a60fd6 |
| Modal v2 | 2026-04-29T13:35Z | T4 | CRASHED | `[train] CONFIG ERROR: --auth-eval-on-best is enabled but variant='quantizr_faithful' is not FP4A-exportable. The post-training auth eval would crash after hours of GPU spend.` (codex R5-2 Finding #1) → fix landed commit 8c1069b4 |
| Modal v3 | 2026-04-29T08:45Z | T4 | UNHARVESTED | Dispatched via spawn(), 24h FunctionCall TTL passed; result lost per `feedback_modal_spawn_result_cache_pattern_20260429.md` |
| Vast 35959478 RTX 4090 | 2026-05-01T18:29Z | RTX 4090 | PREEMPTED | First healthy run with all fixes (commits 85a60fd6+8c1069b4+c504a330+538bbaf8). Trained to ep 810/3000 P2, loss 15.33. Vast preempted at ~23:18 UTC, instance + disk destroyed |

## What's actually in the lane_q_faithful_*_modal/ dirs (NOT Q-FAITHFUL trained weights)

The dirs contain files copied from prior lane scaffolds during the Modal harvester's "pull anything that looks like a result" pass:

```
lane_q_faithful_modal/submissions/robust_current/renderer_fp4_ep3560.bin
  → Magic: b'FP4A' (Lane A/G v3 format, NOT QFAI)
  → Config: pair_mode='asymmetric', HAS motion_hidden=32 module, base_ch=36
  → DEFINITIVELY NOT Q-FAITHFUL JointFrameGenerator (which has c1=56 c2=64, NO motion)
```

```
lane_q_faithful_modal/submissions/robust_current/auth_eval_renderer_fp4.json
  → score: 6.94, device: mps (advisory only per CLAUDE.md)
  → checkpoint config: motion_hidden=32, max_flow_px=20.0, flow_only=false
  → This is a Lane V/V2/K era warp-based asymmetric artifact, NOT Q-FAITHFUL
```

```
lane_q_faithful_modal/submissions/robust_current/contest_eval_result.json
  → score: 2.25, device: mps, archive_bytes=671782
  → Same MPS-noise advisory; contest-CUDA never run on these weights
```

ALL files in v1 + v2 modal dirs have `Apr 30 11:25` timestamp (same minute) — a single batch-copy harvest that grabbed scaffolds, not a real training output.

## The Modal-empty-harvest trap pattern

When `modal_train_lane.py spawn()` dispatches a function that CRASHES early in Stage 0/1, the harvester later grabs whatever files exist in the expected output dir at harvest time. If those files are leftover scaffolds from previous tenants (the Modal volume mount has prior-lane state), the harvested-artifacts dir LOOKS populated but contains the wrong content. The modal_metadata.json was set BEFORE the crash, so it points to a call_id that never produced real outputs.

Detection: check (a) the magic bytes of the binary files, (b) the timestamps (single-minute batch suggests scaffold, not training-spread), (c) the Modal log for crash signatures BEFORE training started.

## Why the user's intuition was correct ("broken and being redesigned")

The current `scripts/remote_lane_q_faithful_jointgen.sh` HAS been redesigned across these commits:
- `c05b2d37` (Apr 28): 1:1 architecture replica + 12 tests landed
- `551002e8` (Apr 28): profile + train_renderer dispatch + QFAI inflate arm
- `d8ade90a` (Apr 28): deploy script landed
- `85a60fd6` (post v1 crash): pass --tag (required by train_renderer.py argparse)
- `8c1069b4` (post v2 crash): --no-auth-eval-on-best (variant not FP4A-exportable)
- `538bbaf8` (Apr 29): Sentinels for v3-v5 redispatches
- `c504a330` (Apr 30): canonical git fetch+reset pattern (replaces fragile git pull --ff-only)
- `931b11c5` (Apr 28): Quantizr R2 + Contrarian R2 VETO: auto-disable --half-frame on use_zoom_flow=False
- `8c1069b4`, `4f4a0e4e` (Apr 30): Modal harvest + the full-rebuild

The CURRENT script is the post-redesign, post-fix version. The RTX 4090 run that was preempted at ep 810 was actually proving the redesign works (loss healthy, P2 transition visible).

## Recovery verdict

**NO TRAINED CHECKPOINT EXISTS to reuse.** The H100 SXM emergency redeploy (subagent ac130930b4717ee87, in flight) is the ONLY path forward. The current script is the FIXED post-redesign version. Predicted ETA on H100: 2-4h vs the 17-19h that was being eaten on the preempted 4090.

## Lessons compounded

1. **`spawn()` + crash = empty harvest masquerading as success.** Always verify post-harvest that the magic bytes / config match the EXPECTED architecture for the lane, not just "files exist".
2. **Vast 4090 preemption is FREQUENT enough that it's the dominant risk for any training > 4h.** Per the fast-chip mandate, H100/A100 SXM should be the floor for time-critical work.
3. **Lane scripts that crash in Stage 0/1 should EMIT A SENTINEL FILE** so the harvester can distinguish "real output" from "scaffold detritus". The 538bbaf8 commit (`Sentinels for v3-v5 redispatches`) was an attempt at this but didn't cover Lane Q-FAITHFUL specifically.

## Cross-refs

- `feedback_q_faithful_4090_preemption_redeploy_h100_20260501.md` (the preemption + redeploy memo)
- `feedback_modal_spawn_result_cache_pattern_20260429.md` (the empty-harvest pattern)
- `project_lane_q_faithful_design_20260428.md` (the 1:1 Quantizr replica design — STILL VALID, just never produced output)
- `project_lane_q_faithful_retrain_dispatch_20260501.md` (the prior 4090 dispatch report)
- `feedback_fast_chip_directive_no_waiting_20260501.md` (the H100/A100 mandate)
- Commit history: c05b2d37 / 551002e8 / d8ade90a / 85a60fd6 / 8c1069b4 / 538bbaf8 / c504a330 / 931b11c5 — the full redesign + fix arc
