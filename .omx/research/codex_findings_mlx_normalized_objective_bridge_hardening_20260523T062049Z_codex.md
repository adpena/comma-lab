# Codex Findings: MLX Normalized Objective Bridge Hardening

Date: 2026-05-23T06:20:49Z
Agent: Codex
Lane: `lane_codex_mlx_normalized_objective_extinction_20260523`

## Summary

The harvested audits found that older MLX/DQS1 consumers had treated raw
singleton-window gains as if they were full-video score movement. Most selector,
portfolio, quality-speed, and next-probe surfaces were already normalized, but
the decoder-q selective window bridge still emitted ambiguous raw-window aliases
and did not require projected full-video improvement before bridge planning.

Landed fixes:

- Decoder-q selective window bridge rows now require positive normalized
  full-video gain and negative projected full-video delta before planning.
- Work units no longer emit ambiguous `observed_mlx_gain` or
  `byte_budget_margin_vs_break_even` aliases; they keep explicit raw-window and
  normalized-full-video fields only.
- Coalesced runs no longer emit ambiguous `local_mlx_gain_sum_non_authoritative`
  with raw-window units; raw and normalized sums remain separately named.
- Scorer-response dataset markdown and next-probe acceptance text now report the
  planning scope and normalized MLX byte-budget margin surface explicitly.

## Evidence

- `.venv/bin/python -m pytest -q src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_feedback.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_quality_speed_delta.py`
  returned `139 passed in 2.23s`.
- `.venv/bin/python -m ruff check src/tac/optimization/decoder_q_selective_window_bridge.py src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/tests/test_scorer_response_dataset.py`
  returned `All checks passed!`.
- `git diff --check` passed.

## Remaining Work

- Continue searching for ambiguous MLX/window/raw aliases in downstream
  artifacts and retire or scope-label them when they are not full-video values.
- Land the resource-bounded experiment queue worker after objective semantics
  are fully normalized, so parallel local MLX/CPU execution scales the right
  acquisition signal.

## Authority

`score_claim=false`; `promotion_eligible=false`;
`ready_for_exact_eval_dispatch=false`; `rank_or_kill_eligible=false`.
