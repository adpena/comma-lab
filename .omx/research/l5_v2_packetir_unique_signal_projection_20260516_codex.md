# L5-v2 PacketIR Unique Signal Projection - 2026-05-16

## Context

The L5-v2 PR106 stack-cell surface previously compressed PacketIR candidates
down to generic `candidate_id`, `format_id`, archive path/SHA, and axis scores.
That was too lossy for the UNIQUE-AND-COMPLETE-PER-METHOD directive: exact
radix, prefix, rank-elided, latent-score-table, and other PacketIR families can
have different runtime, axis-gap, and sidecar semantics even when they share a
surface-level score band.

## Fix

`src/tac/optimization/l5_staircase_v2.py` now carries richer PacketIR source
signal into L5-v2 stack-cell candidates:

- source notes and `sidecar_kind`;
- source artifact warnings such as source `score_claim=true`;
- runtime-consumption path/SHA, runtime directory, source/content tree hashes,
  current Modal-uploaded runtime content hash, current-match flag, derivation
  source, and backfill-required flag;
- axis component distances for CPU and CUDA;
- CPU-minus-CUDA score, SegNet, and PoseNet deltas;
- artifact SHA and source authority flags per exact axis.

All projected stack rows remain planning-only:

- `score_claim=false`;
- `promotion_eligible=false`;
- `ready_for_exact_eval_dispatch=false`;
- blockers still require composite archive materialization, composite paired
  exact eval, and sideinfo consumption proof.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  tools/cathedral_autopilot.py \
  src/tac/tests/test_cathedral_autopilot.py

PYTHONPATH=src:. .venv/bin/pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_cathedral_autopilot.py -q
```

Observed: `86 passed`, ruff clean.

## Next

When the PR106 runtime-consumption proofs are regenerated against the current
runtime and paired rows reopen, this richer projection lets L5-v2/Cathedral rank
stack candidates with their actual method semantics rather than treating all
PacketIR formats as interchangeable byte/score rows.
