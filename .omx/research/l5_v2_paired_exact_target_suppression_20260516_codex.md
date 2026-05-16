# L5 v2 paired exact-target command suppression

- date: 2026-05-16
- scope: Cathedral L5 v2 validation queue / paired Modal auth-eval custody
- code:
  - `src/tac/deploy/modal/paired_dispatch_contract.py`
  - `tools/cathedral_autopilot.py`
  - `tools/all_lanes_preflight.py`
- tests:
  - `src/tac/tests/test_modal_paired_dispatch_contract.py`
  - `src/tac/tests/test_cathedral_autopilot.py`
  - `src/tac/tests/test_all_lanes_operator_briefing_gate.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Finding

Cathedral's L5 v2 validation queue copied PacketIR exact-target command
templates after matrix loading, while the paired-dispatch/single-axis leak
checks lived only in `tools/all_lanes_preflight.py`. That left a false-authority
path where an unsafe `experiments/modal_auth_eval.py` command could appear in a
Cathedral target row before the operator briefing preflight rejected it.

## Change

Paired auth-eval command safety is now a reusable Modal deploy-layer contract:
`tac.deploy.modal.paired_dispatch_contract.paired_auth_eval_dispatch_command_blockers`.
Cathedral applies the same contract while loading L5 v2 PacketIR targets and
suppresses executable `command_template` / `execute_command_template` fields
unless the command uses `tools/dispatch_modal_paired_auth_eval.py` with paired
runtime custody flags. `tools/all_lanes_preflight.py` now uses the same helper,
removing a duplicated definition.

## Verification

- `.venv/bin/python -m ruff check src/tac/deploy/modal/paired_dispatch_contract.py tools/cathedral_autopilot.py tools/all_lanes_preflight.py src/tac/tests/test_modal_paired_dispatch_contract.py src/tac/tests/test_cathedral_autopilot.py src/tac/tests/test_all_lanes_operator_briefing_gate.py`
- `.venv/bin/python -m pytest src/tac/tests/test_modal_paired_dispatch_contract.py src/tac/tests/test_cathedral_autopilot.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`
