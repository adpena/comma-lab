# L5 v2 PacketIR Derived-Prefix Adaptive Recode - 2026-05-16

## Purpose

Turn the L5 v2 PacketIR adaptive-context near miss into a real byte-positive
planning candidate without granting score authority.

The previous adaptive prototype stored the first `context_order` bytes of the
target section. For PR106 PacketIR sections, those bytes are section magic:

- `decoder_packed_brotli` starts with `HDM9`
- `latents_and_sidecar_brotli` starts with `HLM3`

For `context_order=2`, the runtime can derive `HD` or `HL` from the section
grammar instead of storing it per archive. This removes 2 stored bytes from the
adaptive prototype.

## Result

Regenerated `.omx/research/l5_v2_packetir_section_entropy_matrix_20260516_codex.json`
with derived-prefix adaptive rows:

- derived_prefix_adaptive_prototype_row_count: `4`
- rate_positive_derived_prefix_adaptive_prototype_row_count: `2`
- best_rate_positive_derived_prefix_adaptive_prototype:
  - section: `latents_and_sidecar_brotli`
  - context_order: `2`
  - source_section_bytes: `15774`
  - integrated_section_bytes: `15773`
  - delta_bytes_vs_source_section: `-1`
  - prefix_source: `derived_from_section_magic`
  - prefix_bytes: `0`
  - decoder_seed_prefix_bytes: `2`
  - lossless_roundtrip_proven: `true`
  - no_op_detector_passed: `true`

The matrix remains non-promotional:

- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- blockers:
  - `prototype_runtime_decoder_not_integrated`
  - `full_frame_same_runtime_parity_missing`
  - `exact_cuda_auth_eval_missing`
  - `contest_auth_eval_adjudication_missing`

## Verification

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_context_recode.py \
  tools/build_l5_v2_packetir_section_entropy_matrix.py \
  src/tac/optimization/l5_staircase_v2.py \
  tools/operator_briefing.py \
  src/tac/tests/test_pr106_context_recode_l5_matrix.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py

PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_pr106_context_recode_l5_matrix.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_operator_briefing.py \
  -q

PYTHONPATH=src:. .venv/bin/python tools/operator_briefing.py --json
```

Result: `80 passed`.

## Next Gate

The next concrete L5 v2 PacketIR action is runtime binding: integrate the
derived-prefix adaptive decoder into a candidate PacketIR runtime, prove
full-frame same-runtime parity, then run paired exact CPU/CUDA eval. Until then
the row is planning evidence only.

## Runtime-Binding Primitive Follow-Up

Added `decode_adaptive_context_recode_section(...)` in
`src/tac/packet_compiler/pr106_context_recode.py`.

The helper decodes both:

- stored-prefix adaptive rows; and
- derived-prefix adaptive rows where the decoder seed prefix is recovered from
  PacketIR section grammar (`HDM9` / `HLM3`) instead of stored per archive.

Focused verification:

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_context_recode.py \
  src/tac/tests/test_pr106_context_recode_l5_matrix.py

PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_pr106_context_recode_l5_matrix.py \
  -q
```

Result: `4 passed`.

Status remains non-promotional. This is a reusable decoder primitive, not yet a
submission runtime patch, full-frame parity proof, or exact-eval artifact.
