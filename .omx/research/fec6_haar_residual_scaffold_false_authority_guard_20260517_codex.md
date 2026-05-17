# FEC6 Haar Residual Scaffold False-Authority Guard - 2026-05-17

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this
follow-up.

## Bug Class

The `fec6+Haar residual` WIP had two linked false-authority risks:

1. The design memo described the lane as dispatch-ready even though the
   generated `inflate.py` intentionally raises `NotImplementedError`.
2. The design claimed a 3-5 KB Brotli-compressed residual stream while the
   current builder emits raw int8 Haar bands with no entropy coder.

This could have produced a bad provider run that looked like a method negative
or score evidence when the measured object was only an incomplete runtime and
an uncompressed payload scaffold.

## Fix

The builder now emits scaffold-only custody in `build_manifest.json`:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- `runtime_scaffold_only=true`
- `runtime_consumption_proof=false`
- `byte_consumption_proof=false`
- `payload_compression=none_raw_int8_phase1_scaffold`
- `dispatch_blockers=[inflate_py_phase1_scaffold_raises_NotImplementedError, fec6_base_inflate_path_not_wired, no_runtime_consumption_proof, haar_payload_not_entropy_compressed]`

The design memo now states the current lane is scaffold-only, that Brotli/ANS/
range/LZMA coding must be measured before byte-budget claims, and that exact
dispatch is blocked until runtime consumption plus full-frame inflate success
are proven.

## Test Coverage

`src/tac/tests/test_fec6_plus_haar_residual.py` now builds a tiny deterministic
packet and asserts the manifest remains dispatch-ineligible while the generated
runtime still contains `NotImplementedError` and the payload is declared raw
uncompressed scaffold bytes.

## Routing Consequence

`fec6+Haar residual` remains viable as a Rule #6/FEC6 RGB-layer residual
candidate, but the next score-moving object is Phase 2 runtime and entropy
closure:

1. wire generated `inflate.py` to canonical fec6 base inflate;
2. apply decoded Haar residuals to rendered RGB frames;
3. prove Haar bytes are consumed by mutation/no-op tests;
4. measure entropy coding for the emitted band streams;
5. only then produce a paired CPU/CUDA exact-eval dispatch packet.

Until then, this lane is not rank/kill/promote eligible and should not consume
provider budget.

## Verification

Run before commit:

```bash
.venv/bin/python -m pytest src/tac/tests/test_fec6_plus_haar_residual.py
.venv/bin/python -m ruff check src/tac/fec6_haar_residual.py src/tac/tests/test_fec6_plus_haar_residual.py tools/build_fec6_plus_haar_residual_packet.py
.venv/bin/python -m py_compile src/tac/fec6_haar_residual.py src/tac/tests/test_fec6_plus_haar_residual.py tools/build_fec6_plus_haar_residual_packet.py
git diff --check -- src/tac/fec6_haar_residual.py src/tac/tests/test_fec6_plus_haar_residual.py tools/build_fec6_plus_haar_residual_packet.py .omx/research/fec6_plus_haar_residual_design_20260517.md .omx/research/fec6_haar_residual_scaffold_false_authority_guard_20260517_codex.md
```
