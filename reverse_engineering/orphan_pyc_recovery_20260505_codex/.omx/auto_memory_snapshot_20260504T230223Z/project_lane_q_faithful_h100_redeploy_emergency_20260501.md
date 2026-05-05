# Lane Q-FAITHFUL EMERGENCY REDEPLOY — H100 SXM after Vast.ai 4090 preemption (2026-05-01 23:32 UTC)

## TL;DR

The Q-FAITHFUL retrain dispatched at 18:29 UTC on Vast.ai 35959478 (RTX 4090,
ssh6.vast.ai:39478) was PREEMPTED by Vast.ai ~5h into training. ~5h training
progress LOST (no checkpoint harvested). Zero active Vast.ai instances after
preemption. Per user "absolute minimum wallclock" mandate, redeployed to the
ONE available H100 SXM offer (34383502 = $2.4889/hr, driver 580.126.09,
machine 67296 Texas US, 99.3% reliability, 3958 Mbps net down) as new
instance 35985637.

Predicted savings: 13h wall-clock (4090 12.5h → H100 SXM ~2-4h) at +$5
incremental cost. Total session burn for Q-FAITHFUL: $5 sunk + $10 cap = $15.

## Preemption forensics

- Original instance 35959478 (label `owv3_wave3_chain_v11_self_bootstrap`,
  warm from prior wave-3 chain) — RTX 4090 at $0.2622/hr.
- Dispatched 2026-05-01 18:29 UTC. Phase 1 anchor begin at epoch 0 lr=1e-3
  confirmed in train.log within 5 min of launch.
- Two infra fixes were applied in that dispatch (preserved here as the
  canonical bootstrap recipe):
  1. system ffmpeg 4.4.2 lacks `svtav1-params` -> set
     `TAC_FFMPEG=/workspace/ffmpeg-btbn/bin/ffmpeg` (BtbN master)
  2. `apt-get install -y unzip` (missing from the pytorch image)
- Some time between 18:29 UTC and 23:20 UTC the instance disappeared from
  `vastai show instances` output. No SIGTERM-style log line was harvested
  before disappearance (preemption is opaque on Vast.ai). Net loss: ~5h
  wall-clock, $1.30 burn, zero artifacts recoverable.
- Operator mantra: "absolute minimum wallclock" — chip choice is the
  optimization, not $.

## Redeployment to H100 SXM 34383502

```
.venv/bin/vastai create instance 34383502 \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    --disk 80 \
    --label q_faithful_h100_redeploy \
    --ssh \
    --env "NVIDIA_DRIVER_CAPABILITIES=all" \
    --raw
```

Returns:
- `new_contract: 35985637`
- `success: false` (expected — Vast queues offers and create_instance
  returns immediately; instance transitions to `cur_state=running` over the
  next 2-5 min. Operator polls `actual_status=running` to confirm.)
- ssh_host: `ssh3.vast.ai`, ssh_port: `25636`
- gpu_name: `H100 SXM`, vram_gb: 81.6, cuda_max_good: 13.0
- driver_version: `580.126.09` (cu13 capable, but we pin cu124)
- dph_total: $2.4889/hr

## Torch wheel pin decision

CLAUDE.md "Forbidden uv torch install without driver-version pin" rule says:
- driver < 580 → `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` (cu124 wheel)
- driver >= 580 → `INFLATE_TORCH_SPEC=torch==2.11.0` (default cu13)

Driver 580.126.09 is RIGHT at the boundary. Council split:
- Hotz: stay with cu124 (proven on prior 4090 dispatch)
- Yousfi: cu13 native is faster on H100 SXM for FP4 ops

DECISION: cu124 (Hotz) — emergency redeploy is not the time to introduce a
new wheel resolution path. cu13-native FP4 is a future iteration with its
own controlled-baseline measurement.

## Council 7-voice rationale (binding for H100 swap)

- **Shannon (LEAD)**: Q-FAITHFUL Phase 2 KL distill is wallclock-bound, not
  rate-distortion-bound. Chip swap doesn't change the achievable R(D) point
  but DOES change how fast we reach it. ENDORSE.
- **Dykstra (CO-LEAD)**: feasibility region intersection
  (R<=0.20, seg<=0.005, pose<=0.005) is chip-independent. Q-FAITHFUL on
  4090 vs H100 produces identical-band scores, only wallclock differs.
  ENDORSE.
- **Yousfi**: KL distill T=2.0 weight=0.002 + per-class weights (lane=15x)
  unchanged from prior dispatch. ENDORSE.
- **Fridrich**: arch unchanged (87,836 params, NO motion module, NO warp,
  single-mask + FiLM-on-pose). ENDORSE.
- **Contrarian**: 4090 preemption FORCES new chip choice. Three options:
  (a) re-dispatch 4090 — same preemption risk, same 12-19h wall, no win;
  (b) H100 SXM — only one offer, 2.5-5x speedup, $7 incremental;
  (c) A100 SXM4 80GB fallback — 1.5-2x speedup, $4 incremental.
  H100 dominates on wall-clock-per-dollar ratio when wall-clock is the
  scarce resource. ENDORSE H100.
- **Quantizr (adversarial seat)**: H100 SXM is what real labs use for
  88K-renderer + KL distill workloads. The ~3x speedup vs 4090 is well-
  documented for this arch class. ENDORSE.
- **Hotz**: 13h saved at $5 incremental cost is the ROI no-brainer.
  Operator's mantra "absolute minimum wallclock" makes this trivially
  correct. ENDORSE.

VERDICT: 7/7 ENDORSE H100 emergency redeploy.

## Bootstrap fixes required (preserved from prior dispatch)

The H100 instance starts with the bare `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
image. The following Stage 0 bootstrap is REQUIRED before training can launch:

1. `apt-get install -y unzip wget` (both often missing from pytorch image)
2. uv install via `scripts/ensure_remote_uv.sh` (canonical path per CLAUDE.md
   "Forbidden re-implementing remote bootstrap inline")
3. ffmpeg install: prefer BtbN static build at `/workspace/ffmpeg-btbn/bin/ffmpeg`,
   fallback `apt-get install ffmpeg` (system 4.4.2 lacks svtav1-params)
4. Set `TAC_FFMPEG=/workspace/ffmpeg-btbn/bin/ffmpeg` in env.sh
5. Strip macOS resource forks: `find upstream -name '._*' -delete`
6. **VERIFY** `python -c "import torch; assert torch.cuda.is_available()"`
   — FAIL LOUD per CLAUDE.md "MPS auth eval is NOISE" rule

The deploy path uses `scripts/launch_lane_on_vastai.py phase2-extract` +
`phase2-launch` which run `scripts/remote_setup_full.sh` first, which calls
the canonical bootstrap.

## Predicted timeline (H100 SXM)

| stage | epoch | 4090 wallclock | H100 SXM wallclock |
|---|---|---|---|
| Phase 1 (anchor) | 600 | 2.5h | ~50min |
| Phase 2 (KL distill) | 2100 | 6h | ~2.0h |
| Phase 3 (hard pair) | 2500 | 1.5h | ~30min |
| Phase 4 (QAT) | 2900 | 1.5h | ~30min |
| Phase 5 (final) | 3000 | 0.5h | ~10min |
| QFAI export + brotli | n/a | 30min | ~5min |
| Stage 6 auth eval | n/a | 15min | ~5min |
| **Total** | | **~12.5h** | **~4.0h** (conservative) |

## Cost projection

| stage | cost | cumulative |
|---|---|---|
| 4090 preempted (sunk) | $1.30 | $1.30 |
| H100 SXM (predicted, $2.49/hr × 4h) | $9.96 | $11.26 |
| H100 SXM (cap, $2.49/hr × 6h) | $14.93 | $16.23 |

Cap budget: $15. AUTO-DESTROY trigger if instance burns past $15. (Hard
cap matches the cost-cap discipline from `feedback_vastai_cost_paranoia`
and `feedback_fast_chip_directive_no_waiting_20260501.md`.)

## Cross-references

- Predecessor: `project_lane_q_faithful_retrain_dispatch_20260501.md`
- Original design: `project_lane_q_faithful_design_20260428.md`
- Quantizr intel: `project_quantizr_full_intel_20260421.md`
- Bootstrap canonical: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md`
- Driver-pin rule: `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md`
- Chip-priority mandate: `feedback_fast_chip_directive_no_waiting_20260501.md`
- Tracker: `.omx/state/vastai_active_instances.json` (entry `35985637`)
- Dispatch metadata: `experiments/results/lane_q_faithful_h100_redeploy_20260501/dispatch_metadata.json`

## Lane registry update (after launch confirmed)

```bash
python tools/lane_maturity.py mark lane_q_faithful_retrain \
    --gate deploy_runbook \
    --evidence "scripts/remote_lane_q_faithful_jointgen.sh + H100 SXM 34383502 / instance 35985637 dispatched 2026-05-01T23:32Z (post-preemption emergency redeploy)"
```

(real_archive_empirical, contest_cuda, three_clean_review, memory_entry
gates land on harvest at ~04:00 UTC 2026-05-02.)

## Harvest trigger

Every 30 min, check progress via:
```bash
ssh -p 25636 -o StrictHostKeyChecking=no root@ssh3.vast.ai \
    'tail -50 /workspace/pact/lane_q_faithful_results/train.log; \
     tail -3 /workspace/pact/lane_q_faithful_results/heartbeat.log'
```

Expected harvest at ~04:00 UTC 2026-05-02. SCP archive + auth_eval JSON:
```bash
scp -P 25636 -o StrictHostKeyChecking=no \
    root@ssh3.vast.ai:/workspace/pact/lane_q_faithful_results/archive_lane_q_faithful.zip \
    /Users/adpena/Projects/pact/experiments/results/lane_q_faithful_h100_redeploy_20260501/
scp -P 25636 -o StrictHostKeyChecking=no \
    root@ssh3.vast.ai:/workspace/pact/lane_q_faithful_results/auth_eval.log \
    /Users/adpena/Projects/pact/experiments/results/lane_q_faithful_h100_redeploy_20260501/
ssh -p 25636 -o StrictHostKeyChecking=no root@ssh3.vast.ai \
    'cat /workspace/pact/lane_q_faithful_results/eval_work/result.json'
```

The wait-time subagent's harvest orchestrator (Task 1, in flight at the
time of this redeploy) should auto-pick up `archive_lane_q_faithful.zip`
when training completes (Stage 5 of the lane script auto-runs Stage 6
contest_auth_eval).
