# PR106 PacketIR Paired Dispatch Template Canonicalization - 2026-05-16

## Context

After the L5-v2 PacketIR evidence reactivation, the PR106 candidate matrix still
hand-built paired Modal command strings locally. That was correct today but left
a duplicate source of truth next to `tac.deploy.modal.paired_dispatch`.

## Fix

- `src/tac/packet_compiler/pr106_candidate_matrix.py` now generates target
  command templates from
  `tac.deploy.modal.paired_dispatch.paired_auth_eval_dispatch_command_template`.
- Shell rendering uses `shlex.join` with the existing `PYTHONPATH` prefix.
- A regression test compares a representative Matrix target command exactly
  against the canonical helper output.
- The PR106 PacketIR matrix JSON artifact was regenerated and the L5-v2 pinned
  matrix SHA was updated.

## Evidence

- `.venv/bin/python -m ruff check src/tac/packet_compiler/pr106_candidate_matrix.py src/tac/tests/test_pr106_packetir_candidate_matrix.py`
  - PASS
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_pr106_packetir_candidate_matrix.py -q`
  - PASS: 9 passed

## Status

This removes a local duplicate paired-dispatch command builder from the PR106
PacketIR Matrix path. Score, promotion, rank, and exact-dispatch readiness flags
remain false.
