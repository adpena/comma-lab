# Codex Findings: MLX Decoder-Q Family Delta

## Scope

Build a matched per-window delta surface for decoder-q candidate
`d1f1e56e042692f2` against the FEC6 MLX response family.

This is local `[macOS-MLX research-signal]` only. It is not a score claim and is
not eligible for promotion, rank/kill, or exact-eval dispatch selection without
CUDA confirmation.

## Landed reusable surfaces

- `src/tac/optimization/scorer_response_family_delta.py`
- `tools/compare_scorer_response_families.py`

The helper matches two scorer-response families on `source_start_pair` and
emits candidate-minus-reference deltas for score, scorer term, PoseNet, and
SegNet fields while preserving false authority.

## Artifact

- `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_decoderq_minus_fec6_family_delta.json`
- `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_decoderq_minus_fec6_family_delta.md`

## Empirical result

| Metric | Value |
|---|---:|
| Matched windows | 300 |
| Candidate better / worse / tie | 76 / 224 / 0 |
| Mean decoder-q minus FEC6 score delta | +0.00041683455762115714 |
| Best window delta | -0.0015431094684751623 at pair 109 |
| Worst window delta | +0.0025598805968375105 at pair 61 |

The effect is not uniform. The same tensor-domain decoder-q nudge improves 76
windows and regresses 224 windows. The largest movements are SegNet-dominated
quantized steps, e.g. top improvements show `seg_delta=-1.52587890625e-05`
and top regressions show `seg_delta=+2.5431334506720304e-05`.

## Finding

Global candidate response is too lossy for this lane. The next decoder-q
optimizer should train or rank on the matched per-window family delta surface:
window sign, magnitude, and SegNet/PoseNet axis split. This exposes the
Photoshop-knob style boundary the operator asked for: some windows want the
q-adjustment, others want the opposite or no adjustment.

## Next action

Route `scorer_response_family_delta.v1` into the decoder-q waterbucket planner
as a non-authoritative training/rerank label, then test whether candidate
generation can preserve the 76 improving windows while suppressing the 224
regressing windows before any exact-eval spend.
