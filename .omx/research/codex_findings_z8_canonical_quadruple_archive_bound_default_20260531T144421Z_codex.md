# Codex Findings: Z8 Canonical Quadruple Archive-Bound Default

UTC: 2026-05-31T14:44:21Z

## Verdict

The provenance-clean predictive stack priority now has one fewer advisory-only escape hatch: the Z8 canonical quadruple path emits an archive-bound candidate export by default after training.

## What changed

- Added `load_real_video_pair_targets_numpy(...)` as the canonical numpy real-video pair loader for Z8. It returns adjacent frame-0 and frame-1 streams from one `tac.data.decode_video` call.
- Kept `load_real_video_targets_numpy(...)` as a backward-compatible frame-0 wrapper for the existing training loop.
- Wired `experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py --canonical-quadruple-binding` to call `export_z8hpc1_archive_from_canonical_quadruple(...)` by default after M9 training.
- The emitted training artifact now records deterministic replay argv and `archive_bound_export` metadata: archive path, SHA-256, bytes, adapter package path, receiver proof path, and false-authority fields.
- If archive-bound export fails, the CLI writes `archive_bound_export_blocker.json` and exits fail-closed instead of leaving a JSON-only training artifact that could be mistaken for a candidate.

## Authority boundary

This does not grant score authority. Z8HPC1 currently proves Mallat wavelet archive-byte pixel consumption; Mamba, Dreamer, and Wyner-Ziv sections remain archive-custody-only until distinguishing byte-mutation receiver proofs show they affect pixels. MLX and macOS CPU rows stay advisory until exact CPU/CUDA authority signs an archive/runtime packet.

## Verification

- `.venv/bin/ruff check experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py`
- `.venv/bin/python -m pytest src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_archive_candidate_bridge.py -q`
- `git diff --check -- experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py`
