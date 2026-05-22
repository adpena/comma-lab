# Codex Findings: MLX Decoder-Q Response Surface

## Scope

Convert the matched decoder-q-vs-FEC6 MLX family delta into a planner-visible
window response surface. This turns the global mixed-sign decoder-q result into
explicit preserve/suppress labels for downstream byte-neutral candidate search.

All artifacts are local `[macOS-MLX research-signal]` only and carry
`score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false`, and
`promotable=false`.

## Landed reusable surfaces

- `src/tac/optimization/decoder_q_response_surface.py`
- `tools/plan_decoder_q_response_surface.py`
- `tools/plan_ll_scorer_response_next.py --decoder-q-response-surface`

The existing family-delta helper now also includes `pose_term`, `seg_term`, and
`scorer_term` deltas by default, so response surfaces can classify axis
dominance on score terms instead of raw distortion units.

## Artifacts

- Family delta:
  `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_decoderq_minus_fec6_family_delta.json`
- Response surface:
  `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_decoderq_response_surface_plan.json`
- Planner with response surface:
  `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/ll_next_probe_plan_same_axis_600rows_oof_validated_decoderq_surface.json`

## Empirical result

| Metric | Value |
|---|---:|
| Matched windows | 300 |
| Preserve candidate effect | 76 |
| Suppress or invert candidate effect | 224 |
| Neutral | 0 |
| Preserve gain sum | 0.028526397509450963 |
| Suppress harm sum | 0.1535767647957981 |
| Net score delta sum | +0.12505036728634714 |
| Axis dominance | seg: 216, pose: 84 |

Top preserve windows are SegNet-dominated improvements at pairs 109, 68, 98,
59, 257, 134, 229, and 151. Top suppress/invert windows are SegNet-dominated
regressions at pairs 61, 44, 125, 76, 233, 1, 250, and 132.

## Planner effect

`ll_decoder_q_window_signed_response_surface` is now the priority-1 next probe
when the response surface is attached. The prior MLX parity-gated response
harvest remains priority 2, with the strict singleton full-300-pair parity gate
passing.

Only one prohibition remains in that plan:
`do_not_widen_coordinate_sparse_residual_sidecar`.

## Finding

The next decoder-q optimizer should not rank candidates by a global score delta
alone. The correct local surface is signed and window-specific: preserve the 76
improving windows and penalize, invert, or mask out the 224 regressing windows.
The dominant effect is SegNet-term movement, not rate or PoseNet.

## Next action

Wire this response surface into decoder-q candidate generation as an objective:
maximize preserve-window gain while minimizing suppress-window harm under
fixed-length archive constraints. Candidate generation should emit matched
family-delta predictions before any official inflate or exact CUDA spend.
