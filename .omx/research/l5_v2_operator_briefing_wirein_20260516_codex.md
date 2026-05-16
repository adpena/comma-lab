# L5-v2 operator briefing wire-in - 2026-05-16

## Scope

This landing wires the L5-v2 / PR106 PacketIR readiness surface into
`tools/operator_briefing.py`, so normal operator flow sees the fail-closed
frontier state without grepping `.omx/research` artifacts. It is not a score
claim and does not dispatch.

## Operator Surface

- Human briefing adds:
  `Phase 9 - L5-v2 / PR106 PacketIR frontier readiness`.
- JSON briefing adds:
  `l5_v2_frontier_readiness`.

The new surface records:

- TT5L side-info proof presence.
- L5-v2 gate-probe and score/rank dispatch readiness booleans.
- PR106 PacketIR matrix path, candidate count, and status counts.
- Runtime-bound paired PacketIR candidate count.
- Stack-cell candidate count.
- `next_exact_eval_target_count`.
- The full fail-fast PR106 exact-eval target list plus a small human-readable
  sample.
- Canonical paired-dispatch command templates that route through
  `tools/dispatch_modal_paired_auth_eval.py` with
  `--expected-runtime-tree-sha256 auto` and
  `--skip-axis-if-promotable-anchor-exists`.
- Nested L5-v2 and PacketIR blockers, including
  `l5_v2_packetir_no_runtime_bound_paired_exact_candidates`.

All rows explicitly keep:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Current Readout

- TT5L side-info proof present: true.
- PacketIR candidates: 16.
- PacketIR status counts:
  - `paired_exact_blocked`: 3
  - `runtime_consumed_needs_paired_exact_eval`: 4
  - `single_axis_exact_measured_needs_pair`: 9
- Runtime-bound paired PacketIR candidates: 0.
- L5-v2 stack-cell candidates: 0.
- Next exact-eval targets: 13 paired target units.
- All next-target templates use the paired Modal dispatcher. They do not carry
  `<AXIS_SPECIFIC_MODAL_UPLOADED_RUNTIME_TREE_SHA256>` placeholders or direct
  `experiments/modal_auth_eval*.py` single-axis entrypoints.
- `tools/all_lanes_preflight.py` validates the full target list, not only the
  sample shown in the human briefing.

## Verification

```text
.venv/bin/ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py
All checks passed.

.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q
17 passed in 62.58s

PYTHONPATH=src:. .venv/bin/python -m pytest src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
17 passed in 0.17s

PYTHONPATH=src:. .venv/bin/python -m pytest \
  src/tac/tests/test_pr106_packetir_candidate_matrix.py \
  src/tac/tests/test_operator_briefing.py \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py -q
41 passed in 67.55s

.venv/bin/python tools/operator_briefing.py --json --top 1
l5_v2_frontier_readiness.schema = pact.l5_v2_frontier_readiness.v1
l5_v2_frontier_readiness.score_claim = false
l5_v2_frontier_readiness.ready_for_exact_eval_dispatch = false
l5_v2_frontier_readiness.next_exact_eval_target_count = 13
```

## Note

The direct all-lanes operator-briefing dispatch gate still reports current
worktree dispatch-claim state (`dispatch_claim_summary:active_count:3`). That
is pre-existing active-claim hygiene, not introduced by this wire-in. The
focused operator-briefing regression tests pass and the L5-v2 surface itself is
non-promotional.
