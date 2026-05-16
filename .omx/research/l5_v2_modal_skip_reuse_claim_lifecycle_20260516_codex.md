# L5-v2 Modal Skip/Reuse Claim Lifecycle Hardening - 2026-05-16

## Context

The paired Modal auth-eval dispatcher can skip an axis when
`--skip-axis-if-promotable-anchor-exists` finds a runtime-bound promotable anchor
for the same archive SHA. Before this hardening, the skip path wrote an
`anchor_repointer_*.json` manifest but did not append a terminal row to
`.omx/state/active_lane_dispatch_claims.md` for the reused axis.

That created an orphan-signal class: an operator-visible paired dispatch could
finish with no provider spend and no terminal claim evidence for the skipped
axis, even though the axis was intentionally closed by anchor reuse.

## Fix

- `tools/dispatch_modal_paired_auth_eval.py` now records a terminal
  `completed_reused_existing_anchor` claim row for every skipped axis before
  writing the repointer manifest.
- The repointer manifest now records the terminal claim lane id, synthetic
  instance/job id, status, and claims ledger path.
- `src/tac/deploy/claims.py` now allows callers to pass an explicit
  `--claims-path` through the shared claim command helper. This keeps tests and
  forensic work from mutating the live repository ledger while preserving the
  canonical claim tool path.

## Evidence

- `.venv/bin/python -m ruff check src/tac/deploy/claims.py tools/dispatch_modal_paired_auth_eval.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py`
  - PASS
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_dispatch_modal_paired_auth_eval.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_staircase_v2.py -q`
  - PASS: 73 passed
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_dispatch_modal_paired_auth_eval.py src/tac/tests/test_deploy_claims.py src/tac/tests/test_deploy_claims_active_row.py src/tac/tests/test_claim_lane_dispatch.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`
  - PASS: 123 passed
- `tools/operator_briefing.py --json --top 10`
  - L5-v2 PacketIR matrix artifact SHA matches the expected pinned SHA.
  - PacketIR dispatch targets are not suppressed.
  - 13/13 next exact-eval target templates route through
    `tools/dispatch_modal_paired_auth_eval.py`.
  - No direct `experiments/modal_auth_eval.py` or
    `experiments/modal_auth_eval_cpu.py` template leaked from the L5-v2 target
    surface.

## Status

This closes the known paired Modal skip/reuse orphan-claim bug class for the
L5-v2 PacketIR dispatch surface. Score claims remain disabled until paired
CPU/CUDA exact-eval artifacts return with complete custody.
