# Compact Runner MLX Profile Passthrough - Codex Findings 2026-06-01

## Verdict

Landed the compact-base runner integration that lets HPRC
`hprc_mlx_component_neutralization_profile.v1` section-value evidence drive the
bounded runner directly from every PR95/HNeRV, Stage-8, PACT-NeRV-VQ, and
PACT-NeRV selector-v4 execution/adaptation path.

## What Changed

- Added repeatable `--mlx-profile` support to
  `tools/run_compact_renderer_mlx_spine_runner.py`.
- Threaded profile paths through every `build_spine_bounded_runner_plan(...)`
  call emitted by the compact runner.
- Preserved false-authority semantics: MLX profiles route local section
  value-per-byte decisions only; exact CPU/CUDA still owns score authority and
  promotion.
- Fixed stale blocker behavior so `full_video_mlx_scorer_replay_not_attached`
  is not re-added by the compact wrapper after a profile is explicitly supplied
  and consumed by the bounded runner.
- Added selector-v4 regression coverage proving a profile-backed
  `decoder_qw` section becomes `measured_mlx_advisory` and routes through the
  bounded plan without promotion authority.

## Verification

- `.venv/bin/ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py -q`

Both passed.

## Remaining Score-Lowering Work

This closes the wiring gap for consuming section-value evidence, but it does
not itself generate new full-video MLX profiles. The next score-moving action is
to run full-coverage compact-base exports under the hard byte ceilings, attach
real MLX component-neutralization profiles with section/pair/region value
measurements, and let the bounded runner admit only bytes whose measured
non-rate value beats the contest byte price.
