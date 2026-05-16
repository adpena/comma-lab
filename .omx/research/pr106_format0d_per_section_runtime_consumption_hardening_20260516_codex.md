# PR106 Format0D Per-Section Runtime Consumption Hardening - 2026-05-16

## Summary

Hardened PR106 Format0D runtime-consumption proof so base and extra streams are
probed independently. A single global semantic-digest mutation can no longer
claim both `base_format0c_sidecar_payload` and
`extra_pr101_ranked_no_op_payload` were consumed.

## Failure Class

- `runtime_base_format0c_sidecar_payload_consumption_not_proven`
- `runtime_extra_pr101_ranked_no_op_payload_consumption_not_proven`

Before this patch, the proof mutated the extra PR101 stream and reused the
global digest-change claim for both Format0D correction streams. A runtime that
ignored the base Format0C stream but consumed the extra stream could therefore
look fully consumed. The proof now emits `runtime_section_consumption_probes`
and derives each `runtime_consumed_score_affecting_sections` value from its own
section-local mutation.

## Closure Hardening

Format0D exact closure now also binds runtime section identity by offset, not
only SHA-256 and length. This prevents a shifted or ambiguously re-associated
section identity from satisfying the closure check.

## Code Surfaces

- `src/tac/packet_compiler/pr106_sidecar_packet.py`
- `src/tac/packet_compiler/pr106_runtime_consumption.py`
- `src/tac/packetir_exact_closure.py`
- `src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py`
- `src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
- `src/tac/tests/test_packetir_exact_closure.py`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_packetir_exact_closure.py -q
# 94 passed
```

## Interpretation

This is a false-authority fix for PacketIR custody. It does not change the
measured PR106 Format0D score and does not make PR106 promotable. It prevents
future Format0D candidates from clearing runtime-consumption closure unless
each score-affecting sidecar stream is independently decoded and applied.
