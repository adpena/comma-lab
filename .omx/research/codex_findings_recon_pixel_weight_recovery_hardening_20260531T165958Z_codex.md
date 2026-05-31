# Codex Findings: recon_pixel_weight recovery hardening

- Timestamp UTC: 2026-05-31T16:59:58Z
- Scope: `src/tac/substrates/_shared/mlx_score_aware/{bundle.py,loss.py}`,
  `src/tac/substrates/_shared/mlx_score_aware/tests/test_recon_pixel_weight_channel.py`,
  `tools/register_recon_pixel_weight_boundary_refinement.py`
- Authority: `[macOS-MLX research-signal]` / `[macOS-CPU advisory]` only; no
  contest CPU/CUDA score claim, promotion, or rank/kill authority.

## Findings

Curie reviewed the recovered recon-channel work and found one hard bug plus two
anchor-honesty issues:

1. All-zero `recon_pixel_weight` maps were accepted. Under mean normalization
   this produced zero reconstruction loss instead of scale-preserving
   re-weighting.
2. Dense no-transfer anchors encoded the failed saliency-vs-uniform transfer as
   residual `0.0`, which hid the sign/boundary miss from canonical posterior
   consumers.
3. The registrar claimed correction-concentration percentages not present in
   the A/B artifact.

## Fix

- `recon_pixel_weight` now refuses non-positive total mass after finite and
  non-negative validation.
- The registrar now records the transferred #1587 claim as a directional
  positive-margin hypothesis. A non-positive dense empirical margin becomes
  `residual=1.0` with `residual_type=sign_boundary_miss`.
- Unsupported correction-concentration prose was removed from the registrar and
  notes.
- Regression coverage now checks zero-mass rejection, direct stop-gradient into
  the weight map, and sign/boundary residual encoding.

## Verification

- `ruff check src/tac/substrates/_shared/mlx_score_aware/bundle.py src/tac/substrates/_shared/mlx_score_aware/loss.py src/tac/substrates/_shared/mlx_score_aware/tests/test_recon_pixel_weight_channel.py tools/register_recon_pixel_weight_boundary_refinement.py src/tac/tests/test_register_recon_pixel_weight_boundary_refinement.py`
- `pytest -q src/tac/tests/test_register_recon_pixel_weight_boundary_refinement.py src/tac/substrates/_shared/mlx_score_aware/tests/test_recon_pixel_weight_channel.py -p no:cacheprovider`

Both passed locally before commit.
