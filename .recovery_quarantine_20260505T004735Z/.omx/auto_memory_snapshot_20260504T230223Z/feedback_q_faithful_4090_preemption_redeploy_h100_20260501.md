---
name: Q-FAITHFUL training preempted on RTX 4090 35959478 — emergency redeploy to H100 SXM 34383502 (5h training lost)
description: 2026-05-01 ~23:20 UTC. Q-FAITHFUL retrain on Vast.ai RTX 4090 35959478 was Vast-preempted at ep 810/3000 P2 (loss 15.33). All 5h of training progress LOST. SSH connection refused, `vastai show instances` returns EMPTY (zero active instances). Immediately redeploying to H100 SXM 34383502 ($2.47/hr, 81GB VRAM, CUDA 13.0, NV driver 580.126.09, Texas) via subagent ac130930b4717ee87. Predicted new ETA 2-4h on H100 (vs 17-19h on RTX 4090). Net wallclock savings: 13-15h at +$5-10 cost vs sunk $5 on dead 4090.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## What happened

1. **2026-05-01 18:29 UTC**: Q-FAITHFUL retrain dispatched on warm RTX 4090 35959478 ($0.26/hr) by subagent a1f688e0ea962bea2. Subagent chose Option A (warm 4090) over Option B (A100 SXM4) for "risk-reduction" — bootstrap dependencies pre-installed.
2. **18:53 UTC**: ep 70/3000 P1, loss 0.0232, GPU 95-99% util, ETA 13.6h. Healthy.
3. **23:09 UTC** (last heartbeat): ep 810/3000 P2, loss 15.33 (P2 introduces pose+seg scorer terms — expected jump), pose 18.86, seg 0.022, ETA stretched to 17-19h due to slower P2 step time.
4. **23:18 UTC**: SSH connection refused.
5. **23:20 UTC**: `vastai show instances` returns EMPTY. `vastai show instance 35959478` errors `start_date is None` (instance destroyed). **Vast preempted us mid-training.**

## Why this is the WRONG call from the start

User mandate (per `feedback_fast_chip_directive_no_waiting_20260501.md`): "make sure we are using fast chips because we don't have time to waste waiting for results". H100 SXM is the explicit new default for time-critical dispatches.

Subagent a1f688e0ea962bea2's "risk-reduce on warm 4090" was a violation of this mandate. Even without preemption, 17-19h on 4090 vs 2-4h on H100 SXM was the wrong call. With preemption, we lost 5h + still have to redeploy.

## Lessons

1. **Never prioritize "warm box" over "fastest chip" for time-critical training.** A 4090 that's "ready to go in 30s" is worse than an H100 that takes 5 min to bootstrap if total wall-clock is 17h vs 4h.
2. **Vast preemption is FREQUENT** — task list shows DISPATCH 35958897, 35956905, etc. all destroyed by preemption. Assume 4090 instances get preempted ~once per 5h on average. Anything > 4h on 4090 is HIGH-risk.
3. **H100/A100 SXM have lower preemption rates** in observed history — they're more expensive so users hold them more committedly.
4. **The `feedback_fast_chip_directive_no_waiting_20260501.md` rule needs to be HARDER enforced** in dispatch subagent prompts. The current language allows "risk-reduce" as an out. Future: make the H100 search MANDATORY for any training > 1h.

## What's preserved across the preemption

- All committed code (QZS3 packer at cdf099c4, leaderboard intel at 57f59754) — git, durable
- Lane registry entries — durable
- Reverse-engineering memory files — durable
- The `quantizr_faithful_renderer` codebase + `LANE_Q_FAITHFUL_88K` profile + `scripts/remote_lane_q_faithful_jointgen.sh` — durable, READY for re-launch on H100

## What's lost

- 5h of training compute on the now-dead 4090 (~$1.30 sunk cost)
- The ep 810/3000 P2 partial checkpoint (would have been a useful resume point if we'd SCP'd it during training)
- The "Phase 2 ep 2100 lock-in" Quantizr-recipe convergence point (would have been observed at ~ep 2100)

## Redeploy plan (in flight via subagent ac130930b4717ee87)

- Provision H100 SXM 34383502 via vastai create
- Bootstrap (uv + ffmpeg-BtbN + unzip + cu124 torch)
- Verify torch.cuda.is_available() == True before launching
- Pattern A detached nohup launch of `scripts/remote_lane_q_faithful_jointgen.sh`
- Heartbeat + monitoring
- Auto Stage 5 contest-CUDA auth eval at end (per CLAUDE.md non-negotiable)
- Predicted ETA 2-4h on H100 SXM
- Predicted cost: $5-10
- Net win vs prior 4090: -13 to -15h wall clock at +$4 marginal cost

## Cross-refs

- `feedback_fast_chip_directive_no_waiting_20260501.md` (the fast-chip mandate)
- `project_lane_q_faithful_retrain_dispatch_20260501.md` (the prior dispatch report — wrong choice documented)
- `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md` (canonical bootstrap pattern)
- `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md` (driver 580 IS cu13-compatible; cu124 fallback always safe)
- Task list entries: #341 (prior preemption pattern), #344 (the 0.9974 deploy champion still safe), #346 (Council 22/22 verdict, still active)

## Action item for future dispatches

ALL training dispatches > 1h MUST start with chip-supply probe in this priority order:
1. H100 SXM (any) — preferred default
2. H200 — even better if available
3. A100 SXM4 80GB — fallback
4. A100 SXM4 40GB — second fallback
5. RTX 5090 — third fallback
6. RTX 4090 — ONLY if all above unavailable AND training < 1h

Subagent dispatch prompts should HARD-CODE this priority. Wrap the probe in a single command like `scripts/probe_fastest_chip.py` if needed.
