# Codex Findings: RGB-To-YUV6 Partner Gate Repair

- UTC: 2026-05-30T23:52:14Z
- Commit target: main
- Scope: partner commit `aa914c16f` (`Migrate 3 rgb_to_yuv6 sister impls to canonical helper`).

## Finding

The partner rgb-to-yuv6 migration passed its dedicated behavioral parity suite,
but the pushed Python surface was not `ruff` clean. The failures were local to
the committed files: import ordering, `__all__` sorting, stale typing imports,
style-only nested conditionals, an import-after-`pytest.importorskip` test
pattern, and one closure-capture warning in the PoseNet embedding hook.

## Fix

The repair keeps the migration semantics intact:

- canonical helper wrappers remain byte-stable;
- the composition operator's float64 fork keeps its documented precision
  rationale;
- the test keeps `pytest.importorskip("torch")` and marks the subsequent
  canonical imports with the explicit E402 exception;
- the hook closure now binds the batch-local feature list in the callback
  signature, removing the loop-variable capture hazard.

## Verification

- `.venv/bin/python -m ruff check src/tac/composition/yuv6_chroma_subsampled_perturbation_operator/operator.py src/tac/constrained_gen.py src/tac/local_acceleration/pr95_hnerv_mlx_training.py src/tac/saliency.py src/tac/tests/test_rgb_to_yuv6_canonical_extraction_migration.py`
- `.venv/bin/python -m pytest src/tac/tests/test_rgb_to_yuv6_canonical_extraction_migration.py -q`
- Result: 21 passed.
