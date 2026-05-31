# Codex findings: Z7 MLX archive-bound contract bridge

UTC: 2026-05-31T01:55:28Z

## Verdict

Z7 canonical SSD MLX export no longer stops at a byte-closed archive. The export
boundary now emits a shared archive-bound candidate adapter package with:

- `archive.zip` file custody;
- generated `submission/inflate.sh` receiver proof;
- runtime tree SHA-256;
- deterministic replay bundle;
- receiver-proof gate;
- exact-axis blocker;
- posterior-update hook;
- false-authority score/readiness fields.

The receiver proof exercises the generated contest runtime against the emitted
`0.bin`, hashes the raw receiver output, and records whether the raw output was
retained. By default it removes the bulky raw file after hashing, preserving the
proof without accumulating large replay outputs in long MLX runs.

## Authority Boundary

The package can mark `archive_bound_candidate_ready_for_exact_handoff=true`
only after byte custody plus receiver runtime proof pass. It still forces
`ready_for_exact_eval_dispatch=false`, `score_claim=false`, and exact-axis
blockers until contest CPU/CUDA custody and lane preclaim exist.

MLX remains `[macOS-MLX research-signal]`; this bridge makes MLX-trained Z7
packets executable by the contract-first queue, not score-authoritative.

## Entropy Position

`z7_mamba2_mlx_ssd_predictive_coding_archive` now classifies as
`before_entropy_coder` with neural predictive-coding tags. The classifier is
scoped to neural/predictive archive transform tokens so generic MLX diagnostic
rows do not inherit this entropy-stage label accidentally.

## Validation

- `ruff` on touched files: pass
- `pytest src/tac/tests/test_archive_bound_candidate_adapter_spine.py -q`: 7 passed
- `pytest src/tac/tests/test_z7_mamba2_mlx_module_smoke.py::test_z7_mamba2_mlx_canonical_ssd_backend_uses_helper_and_exports_bridge -q`: 1 passed
- `pytest src/tac/tests/test_z7_mamba2_mlx_module_smoke.py src/tac/tests/test_z7_mamba2_mlx_backend_lineage.py src/tac/tests/test_z7_mamba2_canonical_helper_rewire.py -q`: 24 passed
- `git diff --check` on touched code/test files: pass

## Remaining Exact Blocker

The produced package is exact-ready input, not exact authority. Promotion still
requires contest CPU/CUDA replay, lane preclaim, and exact-axis harvest.
