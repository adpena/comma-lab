# D4 GTScorerCache Pose Shape Fix - 2026-05-15

## Scope

Fix the D4 Wyner-Ziv frame-0 smoke blocker where the canonical F3
`GTScorerCache` rejected the real public PoseNet output shape.

This is a hardening/throughput unlock, not a score claim.

## Failure

D4 Modal smoke failed in:

- `experiments/results/lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T232700Z__smoke__100ep_modal/harvested_artifacts/modal_lane_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T232700Z__smoke__100ep.log`

Failure class:

```text
GTScorerCacheError: gt_pose must be 3D (N, 2, 12); got shape (200, 12)
```

Root cause:

- `GTScorerCache` assumed the cached PoseNet target tensor was always
  shaped `(N, 2, 12)`.
- The real public scorer/trainer path can emit flat PoseNet output
  `(N, 12)`.
- Downstream score helpers consume `gt_pose[..., :6]`, so both `(N, 12)` and
  `(N, 2, 12)` are valid and preserve the direct scorer-path semantics.

## Patch

Files changed:

- `src/tac/training_optimization/scorer_cache.py`
- `src/tac/tests/test_training_optimization_scorer_cache.py`

Behavior:

- Accept `gt_pose.dim() in (2, 3)`.
- Keep `gt_seg` strict at `(N, K, H, W)`.
- Keep matching pair-count validation.
- Add regression coverage for flat `(N, 12)` PoseNet cache lookup.

## Verification

Unit and integration tests:

```bash
.venv/bin/ruff check src/tac/training_optimization/scorer_cache.py src/tac/tests/test_training_optimization_scorer_cache.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_training_optimization_scorer_cache.py -q
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_f3_priority_gtscorercache_wire_in.py src/tac/tests/test_f3_backport_wave_v2_trainers_wired.py -q
```

Results:

- `ruff`: passed
- `test_training_optimization_scorer_cache.py`: `32 passed`
- F3 wire-in tests: `98 passed`

Direct real-scorer probe:

```text
{'pairs_shape': (2, 2, 3, 384, 512), 'gt_pose_shape': (2, 12), 'gt_seg_shape': (2, 5, 384, 512), 'n_pairs': 2}
```

This confirms the public scorer path emits `(N, 12)` for this D4 cache build
and the canonical cache now accepts it.

## Remaining Blocker

A local CPU trainer smoke with `--full-cpu` still fails before training because
the shared substrate `device_or_die` path refuses non-smoke CPU execution even
when the trainer parser exposes `--full-cpu`. That is a separate CLI contract
bug. It does not invalidate this cache-shape fix.

## Next Action

Refire the D4 Modal smoke after the patch is pushed, using the canonical lane
claim and provider recovery path. The expected outcome is either a real D4
first-anchor training result or the next infrastructure/model blocker with
cache shape removed from the failure surface.
