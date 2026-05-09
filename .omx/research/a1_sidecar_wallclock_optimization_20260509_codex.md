# A1 sidecar wall-clock optimization and no-regression guard (2026-05-09)

## Scope

Target: `tools/build_a1_per_pair_latent_correction_sidecar.py`

Goal: improve local wall-clock for A1 per-pair latent sidecar search without
changing score-custody status, archive claims, or default greedy semantics.

## Durable change

The tool now decodes only the leading ground-truth pairs required by the
requested search indices. A smoke search over pairs `0..9` decodes 20 frames
instead of all 1200 frames.

## Rejected default

An experimental `--candidate-batch-size 128` decoder-forward batch was tested on
the 10-pair smoke path:

```text
command:
  /usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
    --smoke --candidate-batch-size 128 \
    --output-dir experiments/results/a1_sidecar_codex_batched_smoke_20260509T204500Z

observed:
  gt_frames shape=(20, 874, 1164, 3)
  proxy-mse search: 10 pairs in 183.1s
  real 183.84
  user 473.44
  sys 36.37
  new archive SHA-256 6cca6972e3d768789f332c5bfa1c465d45a7f2787860b1aa157021faa9de6583
  new archive bytes 178316
```

This regressed wall-clock versus the prior scalar-path smoke and therefore is
not the default. The option remains available for explicit benchmarking, but
`--candidate-batch-size` now defaults to `1` and preserves the scalar loop.

## Evidence policy

The 128-batch archive is not dispatch-ready and is not a score claim. It is a
wall-clock negative and a no-regression signal for future sidecar optimization.

## Scalar-default proof

After restoring the scalar default, a one-pair smoke used only the required
ground-truth prefix:

```text
command:
  /usr/bin/time -p .venv/bin/python tools/build_a1_per_pair_latent_correction_sidecar.py \
    --n-pairs 1 \
    --output-dir experiments/results/a1_sidecar_codex_scalar_1pair_20260509T211500Z

observed:
  gt_frames shape=(2, 874, 1164, 3)
  proxy-mse search: 1 pairs in 16.7s
  real 17.36
  new archive SHA-256 d979632acb2602202d1e65d1d13c73276bc97c9c5fc889f3713ef1842e8fb89e
  new archive bytes 178316
```

This is a functionality and wall-clock-shape proof only. It is not an exact
score result and not dispatch-ready.

Next high-EV speed work should attack true CPU throughput: pair-level
parallelism, model/runtime reuse across workers, or native deterministic packet
compiler paths. Do not promote larger decoder-forward chunks without a measured
wall-clock win on the same smoke contract.

## DX follow-up from this pass

Parallel `review_tracker.py mark-file` invocations hit DuckDB process-lock
contention during this work. The tracker now retries briefly on lock contention
instead of failing immediately.

```text
proof:
  two concurrent mark-file subprocesses both returned 0
  source_index.py: 31 entities marked reviewed
  build_a1_per_pair_latent_correction_sidecar.py: 13 entities marked reviewed
```
