# L5 v2 PR106 PacketIR stack-evidence wire-in

Date: 2026-05-16
Owner: Codex
Scope: L5 v2 staircase / PR106 PacketIR / stack-of-stacks planning

## Purpose

PR106/R2 PacketIR candidates are useful byte-closed evidence for
stack-of-stacks planning, but they are not Time-Traveler L5 v2 score evidence.
This landing wires the committed PR106 candidate matrix into the L5 v2
readiness payload with explicit non-promotional semantics.

## Code Surfaces

- `src/tac/optimization/l5_staircase_v2.py`
  - added `l5_v2_packetir_stack_evidence_payload()`
  - added the payload to `l5_v2_dispatch_readiness()`
- `src/tac/tests/test_l5_staircase_v2.py`
  - verifies axis-labelled paired-row extraction
  - verifies fail-closed behavior when the matrix artifact is absent

## Evidence Boundaries

The L5 v2 payload reads:

- `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.json`
- expected SHA-256:
  `03889d2af21468a752fb031375b040cce00fa78a934e1224c217e1c6f64bdd23`

Current extracted paired candidates:

- `format_0x0c_exact_radix`
- `format_0x0d_latent_score_table`
- `prefix_top_16_pr101grammar`

Every extracted row carries both `contest_cpu` and `contest_cuda` evidence as
separate axes. The payload keeps:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Verification

Commands run from repo root:

```bash
.venv/bin/ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_pr106_packetir_candidate_matrix.py -q
.venv/bin/python - <<'PY'
from tac.optimization.l5_staircase_v2 import l5_v2_packetir_stack_evidence_payload
p = l5_v2_packetir_stack_evidence_payload()
print(p["schema"])
print("paired_candidate_count", p["paired_candidate_count"])
print("blockers", p["blockers"])
print("candidates", ",".join(row["candidate_id"] for row in p["paired_candidates"]))
PY
```

Observed results:

- `All checks passed!`
- `49 passed`
- payload schema `l5_v2_packetir_stack_evidence_v1`
- paired candidate count `3`
- blockers `[]`

## Follow-Up

The next stack-of-stacks step is to convert this planning payload into a
composition-cell selector that can propose byte-closed pairings while still
requiring fresh paired exact evidence for any composite packet.
