# HPRC PR95-Style Pose-Guard Curriculum And Config Audit

## Scope

Codex landing for the HPRC train/export/archive control arm after the full-600
native-rate smoke showed rate collapse was no longer the binding blocker but
PoseNet distortion was catastrophic.

This pass turns the next HPRC rung from posthoc collapse into train-time
protected optimization:

- 8-stage `hprc_pr95_pose_guard_rate_v1` curriculum, PR95-style in shape:
  anchor, residual warm start, protected pose/score fit, gentle rate probe,
  rate ramp, protected compaction, byte polish, protected repair polish.
- `score_protection_recon` train-time pathway: the P18/P19 residual protection
  tensor now upweights reconstruction and residual-token repair gradients, not
  just post-training rate collapse.
- Protection-surface shape preflight on both direct trainer and queue builder,
  so a P18/P19 surface from the wrong residual grid or pair count fails before
  work launches.
- Queue-owned smoke using canonical queue SQLite state.

## Empirical Smoke

Run root:
`/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_pr95_pose_guard_queue_smoke_20260601T122750Z`

Queue state:
`/Users/adpena/Projects/pact/.omx/state/experiment_queue_hprc_compact_receiver_campaign_hprc_pr95_pose_guard_queue_smoke_20260601T122750Z.sqlite`

Worker result:

- steps started: 3
- successes: 3
- failures: 0
- skipped: 0
- stages: 8
- training backend: MLX
- portable runtime: numpy
- receiver proof: intentionally deferred at 32-pair smoke
- training archive bytes: 64,946
- lossless rate-collapsed archive bytes: 58,219
- rate-collapsed archive SHA-256:
  `28c0255c25b45ad9c506738a7784787c49b9239e65cb091b25fd607612e1893f`

This is not score authority. It proves the protected curriculum and
rate-collapse chain are executable through the queue.

## Config And Arbitrariness Findings

1. The first direct smoke failed because the reused protection tensor was
   `(64,16,16,3)` while the trainer default residual grid was `24x32`. This was
   a real config-arbitrariness bug class: a surface could be semantically right
   but geometrically wrong. It is now preflighted before queue write and before
   training.
2. Canonical queue state matters. The runner refused SSD-local SQLite and
   required `.omx/state/experiment_queue_<queue_id>.sqlite`, preventing orphan
   execution state.
3. The current production HPRC knobs still need learned/queued sweeps, not hand
   tuning: residual grid, basis count, protected reconstruction weights,
   prox/l1 schedule, high-res pose sidecar width, and final MLX gate threshold.
4. The present 8-stage schedule is an executable scaffold, not the final
   curriculum. PR95-class comparison means longer runs with staged telemetry,
   checkpoint resume, MLX advisory scoring, CPU replay gates, and exact auth
   only after local wins.

## Next Engineering Action

Scale this from a 32-pair smoke to a queue-owned 32/128/600 campaign with the
same protected curriculum, then add an explicit pose-pathway arm:

- low-res interior HPRC base,
- protected high-res pose residual/sidecar or crisper boundary pathway,
- native rate proxy active from mid-curriculum onward,
- MLX full-video gate before local CPU replay,
- CPU/CUDA exact only for byte-closed local winners.

## Authority

All rows remain false-authority. MLX is acquisition/advisory; exact CPU/CUDA
auth remains the only promotion authority.
