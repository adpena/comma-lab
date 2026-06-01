# HPRC sparse protected residual v3 hardening

## Verdict

The v3 sparse protected-residual grammar is useful as a receiver-compatible
tool for ultra-sparse protected geometry tokens, but it must not be selected
blindly for normal compressed HPRC archives.

The full600 sparse-0.8% protected-pathway artifact showed the entropy-position
failure precisely:

- source `0.bin`: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_sparse008_protected_pathway_600_20260601T133014Z/hprc_compact_receiver_archive_export/0.bin`
- protected tensor shape: `(1200, 48, 64, 3)`
- protected nonzeros: `66086`
- protected nonzero fraction: `0.005975658275462963`
- dense v2 payload bytes: `13824028`
- dense v2 brotli-wrapped section bytes: `1013822`
- sparse v3 raw payload bytes: `3095262`
- sparse v3 brotli-wrapped section bytes: `1038803`

Sparse coordinates reduce raw section bytes, but after the actual rate-collapse
entropy position the coordinate stream is `24981` bytes worse than dense zeros.
That is the same structural lesson as the Z8 coefficient work: the right
representation depends on where the transform lands relative to entropy
concentration.

## Landing

Implemented:

- v3 sparse protected residual grammar in `learned_receiver.py`
- strict decode validation for duplicate/non-monotone sparse indices
- positive finite scale validation for v1/v2/v3 residual paths
- context-aware storage selection: dense v2 for compressed non-ultra-sparse
  protected paths, sparse v3 only below the conservative ultra-sparse fraction
  threshold or raw mode
- rate-collapse callers now pass the downstream entropy-position context into
  the residual packer
- tests proving sparse decode, dense decode, selector behavior, malformed
  sparse refusal, nonfinite scale refusal, and protected-path render impact

Validation:

- `uv run ruff check src/tac/substrates/hprc/learned_receiver.py src/tac/substrates/hprc/rate_collapse.py src/tac/substrates/hprc/training_adapter.py src/tac/substrates/hprc/tests/test_learned_receiver.py src/tac/substrates/hprc/tests/test_rate_collapse.py`
- `PYTHONPATH=. uv run pytest src/tac/substrates/hprc/tests/test_learned_receiver.py src/tac/substrates/hprc/tests/test_rate_collapse.py src/tac/substrates/hprc/tests/test_training_adapter.py -q`
- result: `50 passed`

## Next action

Do not spend more effort replaying oversized HPRC archives whose protected path
is just a posthoc dense RGB sidecar. The next HPRC score-moving step is native
rate-aware training plus pair-scoped protected geometry tokens: use P18/P19
full-video surfaces to decide which pairs/regions need high-res geometry, train
with that pathway present, then run the existing receiver-proof rate-collapse
and local replay gates.

