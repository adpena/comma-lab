# PR106 PacketIR No-Op Materialization Dispatch Blocker - 2026-05-16

## Summary

Landed a fail-closed blocker for PR106 latent score-table materializations that
preserve every score-affecting PacketIR payload byte and do not reduce charged
archive bytes. Such artifacts are valid controls/proxy materializations, but
they should not consume exact-eval dispatch slots or be interpreted as score
lowering candidates.

## Failure Class

- `packet_ir_materialization_no_score_affecting_payload_change_without_rate_gain`

This catches the no-op/control case:

- source and candidate `emitted_payload_sha256` are identical;
- candidate archive bytes are equal to or larger than the source archive;
- the artifact is still materialized and preserved, but exact-eval dispatch
  blockers include the no-op class.

Rate-only recodes remain testable: if the score-affecting payload is unchanged
but the candidate archive is byte-smaller, the no-op blocker does not fire.

## Code Surfaces

- `tools/materialize_pr106_latent_score_table_candidate.py`
- `src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py`

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py -q
# 15 passed

.venv/bin/python -m pytest \
  src/tac/tests/test_packetir_exact_closure.py \
  src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py -q
# 42 passed
```

## Interpretation

This is not a PR106/PacketIR negative. It is dispatch hygiene: PacketIR score
table materialization remains useful, but no-op packet controls must be visibly
classified before exact-eval spend. To reactivate a candidate, produce either a
score-affecting PacketIR payload change or a byte-smaller rate-only archive,
then run the existing runtime consumption, full-frame parity, and paired
contest CPU/CUDA closure gates.
