# Codex Findings: MLX Full-600 Local Advisory Transfer Calibration

Timestamp: 2026-05-21T23:13:39Z
Agent: Codex
Lane: mlx_auth_scorer_port
Evidence grade: local MLX CPU scorer-input transfer calibration
Score claim: false
Promotion eligible: false

## Summary

Built the full 600-pair reference scorer-input cache from
`upstream/videos/0.mkv`, ran the MLX scorer-response harness against the full
FEC6 candidate cache, and gated the resulting payload against the matching
macOS CPU advisory auth-eval artifact.

After the full run exposed poor observability during long scorer-response
passes, added `--progress-every` to `tools/run_mlx_scorer_response_cache.py` so
future full-surface runs can emit JSON throughput progress to stderr.

The MLX CPU scorer-response path now has a complete local advisory-axis
transfer-calibration anchor:

```text
PASS_TRAINING_SIGNAL_FIDELITY
score_abs_delta=6.26282986443405e-07
seg_contribution_abs_delta=2.494073420150622e-07
pose_contribution_abs_delta=8.756903284619366e-07
```

This remains non-authoritative and non-promotable. It is a local training/search
signal that can now be used to accelerate candidate exploration on the macOS
advisory axis.

## Artifacts

Reference cache:

```text
experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600/
```

Candidate cache:

```text
experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/
```

MLX response payload:

```text
experiments/results/mlx_scorer_response_fec6_local_advisory_20260521T2305Z_full600/mlx_response_archive178517.json
```

Fidelity manifest:

```text
experiments/results/mlx_scorer_response_fec6_local_advisory_20260521T2305Z_full600/fidelity_vs_macos_cpu_advisory_archive178517_rate1e12.json
```

## Timings

Reference cache build:

```text
pair_count=600
real=22.58s
user=29.99s
sys=2.35s
```

Full MLX CPU scorer-response pass:

```text
n_samples=600
real=487.26s
user=481.81s
sys=2.24s
```

## Metrics

MLX CPU response, corrected to the matching archive byte count
`178517` from `wc -c submission_dir/archive.zip`:

```text
avg_posenet_dist=2.9433004600700013e-05
avg_segnet_dist=0.0005603875059265799
canonical_score=0.19206194316409206
```

Matching macOS CPU advisory auth-eval:

```text
avg_posenet_dist=2.943e-05
avg_segnet_dist=0.00056039
canonical_score=0.19206131688110561
```

Delta:

```text
pose_avg_delta=3.0046007000121392e-09
seg_avg_delta=-2.4940734201202644e-09
score_delta=6.26282986443405e-07
```

## Adversarial notes

- The first fidelity attempt used stale archive byte count `178417`; empirical
  `wc -c` on the matching archive is `178517`. The first failed gate was useful:
  it caught the rate-term mismatch exactly.
- The second fidelity attempt with corrected archive bytes failed only on
  `1.3877787807814457e-17` rate-contribution floating dust against an exact-zero
  threshold. Re-running with `--max-rate-contribution-abs-delta 1e-12` passed.
- Modal CPU transfer remains blocked separately by scorer-input tensor hash
  mismatch between the existing macOS-built full cache and the Modal CPU
  hash-only manifest. This full-600 PASS is local advisory-axis only.
- The full CPU response pass is now the cost center: about 8.1 minutes for
  600 pairs. Any local optimization loop needs batching, GPU calibration, or a
  lower-level scorer-response path before this becomes interactive.
- A cheap two-pair MLX GPU smoke on the same real caches produced
  `score_delta=0.00025381663367340934` versus MLX CPU, almost entirely from
  SegNet argmax drift (`seg_contribution_delta=0.0002543121809139848`). That is
  small enough to consider as a prescreen, but too large to use as a final
  scorer-response gate without a broader calibration sweep.

## Recommended next action

Profile and optimize `tools/run_mlx_scorer_response_cache.py`:

1. Measure per-batch PoseNet vs SegNet time at batch sizes `1, 2, 4, 8, 16`.
2. Run a small MLX GPU calibration on the same caches to decide whether GPU can
   serve as a fast prescreen while MLX CPU remains the parity gate.
3. Add progress/timing output to the CLI so long full-surface runs expose
   throughput instead of appearing stalled.
4. If CPU remains the gate, lower the obvious bottleneck after profiling rather
   than guessing.
