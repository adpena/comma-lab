# HPRC Full600 Distortion Failure And Waterfill Pivot

Timestamp: 2026-06-01T11:18:27Z
Author: Codex
Status: LANDED_FINDING_PLUS_QUEUE_WIRING

## Finding

The full600 HPRC native-rate campaign proved the current compact receiver can
collapse rate, but not yet preserve the scorer-relevant signal:

- Result root:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_native_rate_b7106_full600_20260601T105158Z`
- Local replay summary:
  `local_cpu_replay/local_submission_replay_summary.json`
- Archive bytes: 57,493
- Rate term: 0.00153129
- Local score estimate: 43.226745304360634 `[macOS-CPU advisory]`
- SegNet distortion: 0.06693314
- PoseNet distortion: 133.18959045
- Exact CPU gate: blocked with `local_score_not_below_auth_target`

The replay cleanup path succeeded: inflated scratch and extracted archive
scratch were deleted after success, while the small replay summary, stdout,
stderr, gate, and archive artifacts remained durable on the SSD tier.

## Root Cause

The run did not represent a fair P18/P19 waterfill rate-collapse test. The
training queue consumed native P18/P19 artifacts for the train-time residual
protection surface, but the downstream `transcode_hprc_rate_collapse` step only
used those surfaces when the separate `--hprc-rate-collapse-*` flags were also
provided. The archived rate-collapse report confirms:

```json
{"residual_importance_enabled": false}
```

That means the full600 result is a demotion of uniform/posthoc HPRC residual
collapse, not a demotion of scorer-informed P18/P19 waterfill.

## Landing

The queue builder now treats native P18/P19 surfaces as canonical scorer
surfaces for both phases:

- train-time residual protection surface;
- post-training receiver-closed rate-collapse materializer.

When native P18/P19 artifacts are present and explicit HPRC rate-collapse
artifacts are omitted, the builder automatically forwards the native artifacts
into `tools/transcode_hprc_compact_receiver_rate_collapse.py`, enables lossy
residual collapse, uses the default low-spec `dz0_qd10`, protects high
importance cells with `dz0_qd1`, and confines coarsening to `eligible_low`.

This closes the silent-uniform-collapse failure class that produced the
distortion blow-up above.

## Canonical Implication

HPRC stands for hierarchical predictive residual coding/codec in this lane:
the receiver should transmit a small deterministic hierarchy plus residual
tokens, not dense video. The current rate term proves that small archives are
possible. The binding problem is preserving PoseNet pair geometry and SegNet
class boundaries while spending nearly no bits on class interiors.

The next HPRC score-lowering test should therefore be:

1. MLX-trained, numpy-portable HPRC compact receiver.
2. Native rate-aware residual pressure from full-video P18/P19 surfaces.
3. Rate-collapse materializer reusing the same P18/P19 surfaces by default.
4. Local replay only after the rate gate says the archive can plausibly clear
   the frontier with distortion reserve.
5. Exact CPU/CUDA auth only for byte-closed local winners.

## Next Candidate Shape

The next candidate should use a multi-resolution scorer budget:

- high-resolution or less-coarsened tokens for P19 pose-sensitive pair geometry;
- crisp SegNet boundary preservation;
- aggressive coarsening/fill for low-gradient class interiors;
- measured palette/codebook fills, including openpilot-style class colors only
  as a tested candidate family, not assumed authority.

The deterministic allocator target is still the full contest action:

```text
S = 100*d_seg(full_video) + sqrt(10*d_pose(full_video)) + 25*bytes/N
```

The approximation boundary is operational, not mathematical: chunked MLX/VJP is
allowed only as exact deterministic reduction over the full video before each
accepted update.
