# Codex Findings: MLX Master-Gradient Heuristic Guard

UTC: 2026-05-27T14:15:00Z

## Summary

The untracked MLX master-gradient extractor was preserved and hardened as a
fast local heuristic surface, not promoted to canonical master-gradient
authority.

Two subagent audits exposed the important tension:

- The queue stack can consume `.omx/state/master_gradient_anchors.jsonl` as a
  planning-only prior.
- The current MLX extractor is not yet a canonical per-byte gradient: it uses
  tensor-level finite differences, attributes uniformly over decompressed
  mantissa spans, and does not prove full source-runtime frame parity.

The landing therefore keeps signal without smuggling authority.

## Implemented Contract

- `src/tac/master_gradient_mlx_extractor.py` now exposes a typed MLX tensor-FD
  gradient heuristic with explicit blockers:
  `source_runtime_full_frame_parity_missing`,
  `canonical_archive_byte_domain_mapping_missing`, and
  `per_weight_or_per_byte_projector_missing`.
- `tools/extract_master_gradient_mlx.py` writes `.npy` custody, sidecar
  metadata, and `.omx/state/mlx_research_signal_manifest.jsonl` rows as
  `[macOS-MLX research-signal]`.
- Canonical `master_gradient_anchors.jsonl` writes are opt-in and fail-closed:
  `--write-anchor` refuses the current heuristic result until the canonical
  eligibility blockers are gone.
- Pair-batch execution was added so full-video per-pair extraction can scale
  without decoding/scoring all pairs in one giant MLX batch.

## Authority Boundary

The emitted MLX rows are useful for local probe ranking, diagnostics, and
follow-up tooling. They are not score claims, not promotion/rank-kill evidence,
not exact-dispatch readiness, and not canonical master-gradient anchors yet.

The next valid authority upgrade requires:

1. source-runtime full-frame parity for the decoded frames being scored;
2. a canonical byte-domain split with scored archive bytes and gradient-subject
   bytes separated;
3. a per-weight or per-byte projector that does not collapse tensor structure
   into uniform attribution;
4. regression tests showing that the anchor row remains planning-only even
   after entering `master_gradient_anchors.jsonl`.

## Verification

- `ruff check` passed for the MLX extractor, CLI, and tests.
- `py_compile` passed for the MLX extractor and CLI.
- `pytest src/tac/tests/test_master_gradient_mlx_extractor.py -q` passed.
