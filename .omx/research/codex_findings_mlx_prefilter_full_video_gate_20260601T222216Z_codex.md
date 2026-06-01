# Codex Findings: MLX Prefilter Full-Video Gate

UTC: 2026-06-01T22:22:16Z

## Finding

The HiNeRV local CPU replay gate treated any `--mlx-profile` path as a
full-video MLX prefilter. That was too permissive: the real 128-pair HiNeRV
direct MLX prefilter profile explicitly says
`scope_status.full_video = sampled_prefix_requires_full_video_rerun`, yet the
runner would have considered `has_full_video_mlx_prefilter = true` for a
600-pair archive merely because the path existed.

That is a false-authority risk. Sampled MLX evidence is useful acquisition and
demotion signal, but it must not unlock local CPU replay or clear
`full_video_mlx_scorer_replay_not_attached`.

## Fix

Added the reusable coverage classifier:

- `tac.substrates.hprc.mlx_prefilter_coverage`

The classifier loads each profile, records bytes/SHA/provenance, reads both
top-level and nested pair counts, and only grants
`has_full_video_mlx_prefilter = true` when:

- schema is `hprc_mlx_component_neutralization_profile.v1`
- `scope_status.full_video` is `executed`
- declared pair/sample coverage is at least the canonical contest pair count

The HiNeRV runner now writes this coverage report into
`mlx_prefilter_coverage` and uses it for default local CPU replay gating. The
HPRC bounded runner now also treats sampled profiles as missing full-video MLX
authority while still preserving sampled section-value rows and routing them
back to full-video work orders.

The real 128-pair HiNeRV profile from
`codex_hi_nerv_128pair_direct_mlx_prefilter_20260601T2200Z` was reclassified
with the new helper and saved as:

- `.omx/research/hinerv_128pair_mlx_prefilter_coverage_20260601T222216Z_codex.json`

It correctly records `has_full_video_mlx_prefilter = false` with blockers
`full_video_mlx_scorer_replay_not_attached`,
`sampled_mlx_prefilter_requires_full_video_rerun`,
`mlx_profile_not_full_video_executed`, and
`mlx_profile_pair_count_below_full_video`.

## Regression Coverage

Added/updated tests proving:

- nested `mlx_response_summary.max_pairs` counts as pair coverage
- sampled 128-pair profiles do not unlock default local CPU replay
- full-video `executed` profiles do unlock default replay for full coverage
- sampled section-value profiles keep their work orders and now carry a plan
  blocker for missing full-video MLX authority

## Verification

Passed:

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m ruff check \
  tools/run_compact_renderer_mlx_spine_runner.py \
  src/tac/substrates/hprc/mlx_prefilter_coverage.py \
  src/tac/substrates/hprc/spine_bounded_runner.py \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py \
  src/tac/substrates/hprc/tests/test_mlx_prefilter_coverage.py \
  src/tac/substrates/hprc/tests/test_spine_bounded_runner.py

PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  -m pytest --import-mode=importlib -q \
  src/tac/substrates/hprc/tests/test_mlx_prefilter_coverage.py \
  src/tac/substrates/hprc/tests/test_spine_bounded_runner.py \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Result: 30 passed.

## Next Action

Run the corrected full-video HiNeRV/SNeRV/PR95 candidate flow only after a real
600-pair MLX prefilter profile exists. The 128-pair HiNeRV profile remains a
durable demotion/acquisition signal, not a CPU replay trigger.
