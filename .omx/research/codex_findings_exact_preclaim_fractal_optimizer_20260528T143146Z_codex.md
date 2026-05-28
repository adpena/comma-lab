# Codex Findings: Exact Preclaim + Fractal Optimizer Closure

Date: 2026-05-28T14:31:46Z
Agent: codex

## Landed Slice

- Added a queue-owned exact-dispatch provider preclaim gate for Lightning execute rows. The preclaim step writes a redacted `exact_dispatch_provider_preclaim_check.v1` artifact and exits before `claim_lane_dispatch.py` can mutate the lane ledger when `LIGHTNING_STUDIO`, `LIGHTNING_TEAMSPACE`, owner identity, or `LIGHTNING_SSH_TARGET` are missing.
- Wired execute-mode materializer exact-eval queues as `provider_preclaim_check -> claim_lane_dispatch -> dispatch_exact_eval`; dry-run queues remain unchanged.
- Extended repair-family stack search with `repair_family_fractal_marginal_surface.v1`, ranking level/stage marginal cells by improvement per byte, stack penalty, negative demotion, and byte-credit pressure.
- Added a ranked `repair_family_stack_acquisition_frontier_path.v1` frontier from the hypergraph tensor so the autonomous loop can inspect more than one viable stack path rather than only the primary path.

## Authority Boundary

All new artifacts remain false-authority:

- no score claim;
- no promotion eligibility;
- no rank-or-kill authority;
- no exact dispatch readiness;
- MLX rows remain advisory only;
- exact CPU/CUDA handoff still requires byte-closed archive/runtime custody plus existing exact-ready dispatch gates.

## Verification

- `ruff check` on touched files: passed.
- `pytest src/tac/tests/test_exact_dispatch_provider_preclaim_tool.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_repair_family_materializers.py src/tac/tests/test_repair_campaign_materialization_queue.py -q`: 104 passed.
- Missing-Lightning smoke: preclaim exited 2 and wrote only presence/absence env status, no values.
- `git diff --check`: passed.
- Review tracker: three Codex recursive clean passes marked; policy checks report 0 violations for all touched files.

## Recursive Senior Engineer Review

Pass 1, dispatch custody: clean. Execute queues cannot claim the lane before local provider route prerequisites are proven in a file-backed artifact.

Pass 2, optimizer math/integration: clean. Hypergraph stack search now emits both the primary path and a ranked frontier plus per-level marginal surfaces, preserving entropy ordering, byte-credit feasibility, negative posterior demotion, and false-authority semantics.

Pass 3, signal preservation: clean. Tests cover the preclaim ordering, redaction, false-authority payloads, stack frontier, and fractal marginal surface. No unrelated formatter churn is included.

