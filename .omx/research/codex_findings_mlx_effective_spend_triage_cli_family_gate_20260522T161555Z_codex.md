# Codex Findings - MLX Effective Spend-Triage CLI Family Gate Coverage

Date: 2026-05-22T16:15:55Z

## Summary

Inspected the existing MLX effective spend-triage selector and dynamic learned
sweep planner after the recent family-gating commits.

The implementation already enforces family-gated candidate selection in
`tac.optimization.mlx_effective_spend_triage_selection`: when `--family` is
omitted, selection defaults to the effective gate's
`spend_triage_allowed_families`; when a requested family is not in that gate,
selection fails closed before writing a manifest.

The missing coverage was at the operator CLI boundary:
`tools/select_mlx_effective_spend_triage_candidates.py` had a happy-path CLI
test but did not directly prove those family-gated behaviors.

## Patch

Added focused tests in
`src/tac/tests/test_mlx_effective_spend_triage_selection.py`:

- CLI default selection uses only the effective gate's allowed family list.
- CLI explicitly refuses a requested family without the effective family gate
  and does not emit a selection manifest.

## Authority

No score, promotion, rank, kill, or dispatch authority changed.

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This remains candidate-generation-only local MLX spend-triage machinery.
