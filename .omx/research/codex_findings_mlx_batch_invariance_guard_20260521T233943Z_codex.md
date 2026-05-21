# Codex Findings: MLX scorer batch-invariance guard

## Summary

Added a reusable batch-invariance audit for local MLX scorer adapters. The audit
compares one batched scorer response against concatenated singleton responses
on the same scorer-input cache window. This directly guards a new conformance
failure mode: deterministic MLX GPU batch-shape drift.

This is local MLX evidence only. It does not create score authority.

## Code changes

- `src/tac/local_acceleration/mlx_batch_invariance.py`
  - New manifest builder for batched-vs-singleton scorer output comparison.
  - Checks PoseNet output max-abs delta, SegNet logit max-abs delta, and SegNet
    argmax pixel differences.
  - Emits false-authority fields (`score_claim=false`, `promotion_eligible=false`,
    `ready_for_exact_eval_dispatch=false`).
- `tools/audit_mlx_scorer_batch_invariance.py`
  - CLI wrapper over the reusable module.
- `src/tac/local_acceleration/mlx_scorer_response.py`
  - Requires explicit `allow_gpu_research_signal=True` for `device_type="gpu"`.
- `tools/run_mlx_scorer_response_cache.py` and
  `tools/profile_mlx_scorer_response_cache.py`
  - Add `--allow-gpu-research-signal` for GPU scorer-response/profile paths.
- `.gitignore`
  - Ignores local MLX scorer-input caches and response/profile roots by default.
- `src/tac/tests/test_mlx_batch_invariance.py`
  - Covers pass/fail manifests, singleton concatenation order, and CLI rejection
    of non-audit singleton batch size.

## Empirical audit on FEC6 reference cache

Cache:

`experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600`

Pair window:

`start_pair=16`, `batch_pairs=2`

CPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_batch_invariance.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --repo-root . \
  --device cpu \
  --start-pair 16 \
  --batch-pairs 2 \
  --run-id mlx_batch_invariance_reference_pairs16_2_cpu_20260521 \
  --output experiments/results/mlx_scorer_batch_invariance_reference_20260521T2358Z_pairs16_2_cpu/audit.json
```

Observed CPU verdict:

- `passed`: `true`
- `verdict`: `PASS_MLX_BATCH_INVARIANCE`
- PoseNet output max abs: `7.62939453125e-6`
- SegNet logit max abs: `2.384185791015625e-5`
- SegNet argmax diff pixels: `0`

GPU command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_batch_invariance.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --repo-root . \
  --device gpu \
  --start-pair 16 \
  --batch-pairs 2 \
  --run-id mlx_batch_invariance_reference_pairs16_2_gpu_20260521 \
  --output experiments/results/mlx_scorer_batch_invariance_reference_20260521T2358Z_pairs16_2_gpu/audit.json
```

Observed GPU verdict:

- `passed`: `false`
- `verdict`: `FAIL_MLX_BATCH_INVARIANCE`
- PoseNet output max abs: `0.0707550048828125`
- SegNet logit max abs: `0.0483393669128418`
- SegNet argmax diff pixels: `4`

## Interpretation

MLX CPU is batch-stable on this scorer-input window under the configured
tolerances. MLX GPU is repeat-stable for a fixed batch shape, but batch shape
changes the scorer outputs enough to move PoseNet output and SegNet argmax
decisions. Therefore MLX GPU must not be used for scorer-response training or
candidate ranking at `batch_pairs=2` without a passing batch-invariance audit.

Operational rule until further calibration:

- CPU remains the local scorer-response reference.
- GPU may be used for fast prescreening only after the specific device and
  batch shape pass `tools/audit_mlx_scorer_batch_invariance.py`.
- If GPU fails, use singleton GPU rows for exploration and CPU spot checks for
  selection; do not route rank/kill or paid dispatch from failing GPU batch
  shapes.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_batch_invariance.py \
  tools/audit_mlx_scorer_batch_invariance.py \
  src/tac/tests/test_mlx_batch_invariance.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_preprocess.py -q
```

Results:

- `ruff`: pass
- focused MLX scorer/cache/fidelity suite: 46 passed

## Authority status

Passing this guard means only that one local MLX device and batch shape is
internally batch-invariant on a chosen cache window. It is not an auth-eval
score, not a contest-axis result, and not sufficient for candidate promotion.
