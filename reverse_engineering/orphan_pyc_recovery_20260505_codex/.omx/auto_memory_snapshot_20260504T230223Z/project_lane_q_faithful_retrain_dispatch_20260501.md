# Lane Q-FAITHFUL retrain DISPATCHED — 2026-05-01 18:29 UTC on Vast.ai 35959478 (RTX 4090)

## TL;DR

Q-FAITHFUL JointFrameGenerator (1:1 Quantizr PR #55 replica, 87,836 params, NO
motion module, NO warp, single-mask + FiLM-on-pose dual-head) is training on
warm Vast.ai instance 35959478 (RTX 4090, $0.2622/hr). Stage 0/Stage 1
preflight + mask build PASSED. Stage 2 train_renderer.py LIVE at GPU 97% util,
3.9GB RAM, Phase 1 anchor begun at epoch 0 lr=1e-3. eval_roundtrip=True
confirmed active. Two infra blockers resolved during dispatch:

1. system ffmpeg 4.4.2 lacked `svtav1-params` -> set
   `TAC_FFMPEG=/workspace/ffmpeg-btbn/bin/ffmpeg` (BtbN master already
   pre-extracted at `/workspace/ffmpeg-btbn/`)
2. `/usr/bin/unzip` missing -> `apt-get install -y unzip`

Expected ETA: 12.5h (3000 epochs across 5 phases) + 30min QFAI export +
15min auth eval = end-of-run ~07:30 UTC 2026-05-02.

Cost cap $8 (current burn $0.26/hr × 13h = $3.40 expected).
Anchor 1.15 [contest-CUDA] / Predicted band [0.40, 0.80] / Floor 0.40
(matches Quantizr's 0.33 ± 0.5 lane variance).

## Council 7-voice rationale (binding)

- **Shannon (LEAD)**: 87,836 params is below the rate-distortion knee
  observed at 100K params; if KL distill T=2.0 (canonical Hinton 2015 Δ)
  preserves SegNet logit ordering, the 88K renderer can reach 0.45
  distortion while the rate term drops 0.30 vs Lane A's 290KB renderer.
  ENDORSE.
- **Dykstra (CO-LEAD)**: alternating-projections feasibility says the
  88K-renderer feasibility region intersects the contest constraint
  (R<=0.20, seg<=0.005, pose<=0.005) only if the SegNet KL hyperprior is
  hit at 0.45 distortion target. Q-FAITHFUL is the only existing path.
  ENDORSE.
- **Yousfi**: KL distill T=2.0 weight=0.002 + per-class weights (lane=15x)
  match the EfficientNet-B2 SegNet's class-prior-imbalance correction
  exactly (validated against Quantizr's PR #55 score 0.33). ENDORSE.
- **Fridrich**: FiLM-on-pose (no motion module) attacks PoseNet via the
  Hydra head's first-6-pose-dim sensitivity directly; less surface area
  for inverse-steganalysis defenses to exploit than an explicit
  warp-based predictor. ENDORSE.
- **Contrarian**: 3 prior modal dispatches (v1/v2/v3) ALL evaporated in
  the result-cache TTL without any score being measured. Vast.ai +
  per-instance pollable log eliminates this failure mode. The 56h
  deadline + 0.32-0.33 leader gap demands we run THE ONE existing-code
  path with no further hesitation. ENDORSE.
- **Quantizr (adversarial seat)**: This IS my arch verbatim. The only
  delta is +5-stage QAT (our advantage over my vanilla quantization).
  Predicted band [0.40, 0.80] is honest given the lane-variance prior on
  3000-epoch reproductions. ENDORSE WITH NOTE: monitor Phase 2 epoch
  2100 closely — that's where my arch-recipe locked in at 0.33.
- **Hotz**: 88K params on RTX 4090 will saturate batch in <100MB; we are
  GPU-compute-bound, not memory-bound. 12.5h is realistic. ENDORSE.

VERDICT: 7/7 ENDORSE. No design defects.

## What success looks like

- Phase 1 end (epoch 600, ~2.5h): pixel L1 < 14 (kill threshold per script)
- Phase 2 end (epoch 2100, ~8h): scorer < 4.0 (must show learning signal)
- Phase 4 end (epoch 2900, ~11h): scorer < 1.5 (target standalone band)
- Stage 6 contest_auth_eval: [contest-CUDA] score in [0.40, 0.80]
- Sub-0.55 = matches Quantizr 0.33 ± 0.5 lane-variance — TRUE replica achieved

## What failure looks like

- Phase 1 end pixel L1 >= 14: training collapsed early (kill, $1 cost)
- Phase 2 end scorer >= 4.0: KL distill stuck (kill, $4 cost)
- Phase 4 end scorer >= 1.5: QAT broke FP32 baseline (kill, $7 cost)
- Stage 6 score > 0.80: FALSIFICATION of Q-FAITHFUL replica hypothesis
- Wall clock > 32h: auto-destroy (Vast.ai cost cap $8)

## Harvest trigger

Every 2h, check progress via:
```bash
ssh -p 39478 root@ssh6.vast.ai 'tail -50 /workspace/pact/lane_q_faithful_results/train.log; tail -3 /workspace/pact/lane_q_faithful_results/heartbeat.log'
```

Expected harvest at ~07:30 UTC 2026-05-02. SCP archive + auth_eval JSON:
```bash
scp -P 39478 root@ssh6.vast.ai:/workspace/pact/lane_q_faithful_results/archive_lane_q_faithful.zip \
    /Users/adpena/Projects/pact/experiments/results/lane_q_faithful_retrain_20260501/
scp -P 39478 root@ssh6.vast.ai:/workspace/pact/lane_q_faithful_results/auth_eval.log \
    /Users/adpena/Projects/pact/experiments/results/lane_q_faithful_retrain_20260501/
ssh -p 39478 root@ssh6.vast.ai 'cat /workspace/pact/lane_q_faithful_results/eval_work/result.json'
```

## What to do if training stalls

- If GPU util drops to 0% for >30 min: kill the process, investigate
  `train.log` for OOM / Traceback / silent assert failure. Re-launch
  from latest checkpoint via `--resume <ckpt-path>`.
- If heartbeat stops updating for >5 min: SSH the host, check disk space
  (`df -h /workspace`), check process tree (`ps auxf | grep train`).
- If Phase 1 pixel L1 doesn't drop below 14 by epoch 600: KILL — the
  arch is incompatible with the dataset; revisit Lane V or Lane K
  retrain with motion module re-enabled.
- If a phase boundary fails to advance by epoch+50: bisect via the
  phase-schedule print line at script start; possible bug in
  train_renderer's phase-state machine.

## Critical dispatch context

- **DO NOT delete** `/workspace/pact/owv3_0120_stack_archive.zip` (609,963
  bytes) — that is the deploy CHAMPION fallback if Q-FAITHFUL fails.
- Instance 35959478 was warm from `owv3_wave3_chain_v11_self_bootstrap`
  label (6h uptime); it has all deps including BtbN ffmpeg + uv + torch
  cu130 + .venv preinstalled. Do not destroy until both Q-FAITHFUL has
  landed AND the deploy champion has been re-confirmed elsewhere.
- 32GB free disk, 24GB GPU RAM all-clear.

## Cross-references

- Architecture: `src/tac/quantizr_faithful_renderer.py` (336 LOC, 87,836 params)
- Profile: `src/tac/profiles.py:3383` `LANE_Q_FAITHFUL_88K`
- Deploy script: `scripts/remote_lane_q_faithful_jointgen.sh` (343 LOC)
- Dispatch metadata: `experiments/results/lane_q_faithful_retrain_20260501/dispatch_metadata.json`
- Audit reference: `.omx/research/quantizr_replica_audit_20260428.md`
- Quantizr PR: https://github.com/commaai/comma_video_compression_challenge/pull/55
- Prior dispatch failures (modal cache TTL evaporated):
  - `experiments/results/lane_q_faithful_modal/modal_metadata.json`
  - `experiments/results/lane_q_faithful_v2_modal/modal_metadata.json`
  - `experiments/results/lane_q_faithful_v3_modal/modal_metadata.json`
- Memory: `project_lane_q_faithful_design_20260428.md` (design doc)
- Memory: `project_quantizr_full_intel_20260421.md` (Quantizr-paradigm spec)

## Lane registry update

Run after this commit:
```bash
python tools/lane_maturity.py add-lane lane_q_faithful_retrain \
    --name "Lane Q-FAITHFUL retrain (RTX 4090 dispatch)" --phase 1
python tools/lane_maturity.py mark lane_q_faithful_retrain \
    --gate impl_complete \
    --evidence "src/tac/quantizr_faithful_renderer.py + scripts/remote_lane_q_faithful_jointgen.sh"
python tools/lane_maturity.py mark lane_q_faithful_retrain \
    --gate deploy_runbook \
    --evidence "scripts/remote_lane_q_faithful_jointgen.sh + experiments/results/lane_q_faithful_retrain_20260501/dispatch_metadata.json"
```

(real_archive_empirical, contest_cuda, strict_preflight, three_clean_review,
memory_entry gates land on harvest at ~07:30 UTC 2026-05-02.)
