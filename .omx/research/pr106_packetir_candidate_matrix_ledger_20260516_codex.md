# PR106 PacketIR candidate matrix ledger

Date: 2026-05-16
Owner: Codex
Scope: PR106/R2 PacketIR identity, runtime-consumption, and exact-axis evidence join

## Purpose

This landing turns the scattered PR106/R2 candidate evidence into one
regenerable, non-promotional audit surface. It is intended to prevent repeated
manual joins across PacketIR identity proofs, runtime-consumption manifests, and
Modal auth-eval directories.

The matrix is explicitly not a promotion surface:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `contest_cpu` and `contest_cuda` axes are kept separate and are never
  converted into each other.

## Artifacts

- JSON: `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.json`
  - SHA-256: `03889d2af21468a752fb031375b040cce00fa78a934e1224c217e1c6f64bdd23`
- Markdown: `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.md`
  - SHA-256: `9fb06495194f1db9fd6aad5c2a059b00eb88f2ece0f8c0ead503654dc621ffd8`

## Current Matrix Summary

- Schema: `pr106_packetir_candidate_matrix_v1`
- Candidate count: 16
- Status counts:
  - `paired_exact_measured`: 3
  - `single_axis_exact_measured_needs_pair`: 9
  - `runtime_consumed_needs_paired_exact_eval`: 4

The paired rows are candidates with both valid `contest_cpu` and
`contest_cuda` exact evidence under the same archive SHA. Single-axis rows are
not equivalent to paired evidence and must not be promoted until the missing
axis is measured or the lane is explicitly closed as axis-specific.

## Implementation Surfaces

- `src/tac/packet_compiler/pr106_candidate_matrix.py`
- `tools/build_pr106_packetir_candidate_matrix.py`
- `src/tac/tests/test_pr106_packetir_candidate_matrix.py`
- `src/tac/packet_compiler/__init__.py`

The reusable API is exported from `tac.packet_compiler` so follow-on tools can
consume the matrix directly instead of reimplementing the join.

## Verification

Commands run from repo root:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pr106_packetir_candidate_matrix.py -q
.venv/bin/ruff check src/tac/packet_compiler/pr106_candidate_matrix.py src/tac/packet_compiler/__init__.py tools/build_pr106_packetir_candidate_matrix.py src/tac/tests/test_pr106_packetir_candidate_matrix.py
.venv/bin/python tools/build_pr106_packetir_candidate_matrix.py
```

Observed results:

- `6 passed`
- `All checks passed!`
- generator emitted the artifact paths and SHA-256 values listed above.

## Follow-Up

1. Exact-pair the two runtime-consumed semantic-prefix candidates if they remain
   high-EV after the next PR106 review pass.
2. Decide whether the nine CUDA-only rows need CPU pairs or should be closed as
   CUDA-axis-only diagnostics.
3. Feed the paired rows into L5-v2/stack-of-stacks planning only as
   axis-labelled evidence, never as a global frontier claim.
