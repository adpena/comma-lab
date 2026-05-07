# NWCS Beta Encoding Loop Greenup - 2026-05-07

## Scope

Worker NWCS-beta tightened the single-tensor NWCS encoding loop only. No GPU
eval, remote dispatch, lane claim, score claim, or promotion claim was made.

## Concrete Result

- `encode_with_variable_codebook(...)` keeps its existing bytes-only API.
- `encode_with_variable_codebook_manifest(...)` now returns the same deterministic
  bytes plus a strict stream manifest for build/readiness tooling.
- `inspect_variable_codebook_stream(...)` parses NWCS1 tensor streams before they
  enter a renderer container and fails closed on malformed shape, bucket,
  code-index, block-count, trailing-byte, or byte-accounting inconsistencies.
- Bucket assignment now uses stable rank quantiles with original block index as
  the tie-breaker, avoiding platform-dependent quantile interpolation.
- fp16 scale/tail serialization and decode now use explicit little-endian byte
  order.
- Non-empty streams with zero total sensitivity now reject as non-promotable
  no-signal inputs instead of silently encoding a no-op sensitivity map.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_neural_weight_codec_sensitivity.py src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py`
  passed: `22 passed in 1.37s`.
- `.venv/bin/ruff check src/tac/neural_weight_codec_sensitivity.py src/tac/tests/test_neural_weight_codec_sensitivity.py src/tac/tests/test_neural_weight_codec_sensitivity_renderer_format.py`
  passed.

## Remaining Blockers

- This is build-only readiness hardening. A dispatch-ready NWCS artifact still
  requires validated component/corpus sensitivity custody, archive custody,
  exact CUDA auth eval, adjudication, and terminal dispatch-claim linkage.
