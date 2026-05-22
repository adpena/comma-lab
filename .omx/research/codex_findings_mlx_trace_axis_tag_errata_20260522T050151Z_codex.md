# Codex Findings Errata: MLX Trace Axis Tag

timestamp_utc: 2026-05-22T05:01:51Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: CORRECTED_BY_ERRATA

## Scope

Errata for
`codex_findings_mlx_segnet_trace_compare_authority_gate_20260522T050055Z_codex.md`.

## Correction

That memo used stale shorthand for the MLX evidence tag:

- stale: `evidence_tag="[macOS-MLX]"`
- stale: `score_axis="[macOS-MLX]"`

The canonical current contract is:

- `evidence_grade="macOS-MLX-research-signal"`
- `evidence_tag="[macOS-MLX research-signal]"`
- `score_axis="[macOS-MLX research-signal]"`

The code contract in `src/tac/local_acceleration/__init__.py` and
`src/tac/local_acceleration/mlx_segnet_trace_compare.py` enforces the full
research-signal tag. The prior memo's shorthand is documentation drift only,
not the runtime contract.

## Authority

- score_claim: `false`
- score_claim_valid: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- promotable: `false`
