# Selector-V4 MLX Section-Value Profiler - Codex Findings 2026-06-01

## Verdict

Landed a PSV4 section-value profiler that materializes baseline plus
neutralized selector-v4 archive variants and emits the shared
`hprc_mlx_component_neutralization_profile.v1` surface consumed by the compact
bounded runner.

## What Changed

- Added `tools/profile_pact_nerv_selector_v4_mlx_section_value.py`.
- Reused the existing selector-v3 cache materialization, MLX scorer replay,
  window splitting, section delta math, false-authority flags, and HPRC profile
  schema by parameterizing the shared helper paths instead of copying the whole
  profiler.
- Added deterministic PSV4 ZIP member replacement for profiled archive variants.
- Added tests proving:
  - decoder, latent, and selector neutralizations stay parseable;
  - receiver-state and absent residual bytes stay fail-closed;
  - the emitted report uses PSV4 schema/layout keys;
  - the compatibility profile is exactly the HPRC component profile expected by
    the bounded runner.

## Verification

- `.venv/bin/ruff check tools/profile_pact_nerv_selector_v3_mlx_section_value.py tools/profile_pact_nerv_selector_v4_mlx_section_value.py src/tac/tests/test_profile_pact_nerv_selector_v4_mlx_section_value.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_profile_pact_nerv_selector_v4_mlx_section_value.py src/tac/substrates/pact_nerv_selector_v4/tests/test_section_value.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/tests/test_profile_compact_renderer_mlx_section_value.py -q`
- CLI help smoke for both selector-v3 and selector-v4 profilers.

All passed.

## Live Smoke Blocker

Attempted a one-pair live smoke on the prior selector-v4 archive:

`/Volumes/VertigoDataTier/pact/compact_selector_v4_1pair_1epoch_codex_20260601T1652/pact_nerv_selector_v4_mlx_training/archive.zip`

The profiler correctly materialized the baseline and neutralized archives, but
MLX cache materialization refused the partial archive because `inflate.sh`
emitted `0.raw=128B` while the contest raw contract expects
`3,662,409,600B` for the full 1200-frame video. This is a custody blocker, not
a score result: selector-v4 section-value replay needs a full-coverage archive
or an explicitly research-only partial-cache path that cannot feed promotion.

Smoke artifact root:

`/Volumes/VertigoDataTier/pact/selector_v4_section_value_smoke_20260601T171620Z`

## Next Score-Lowering Action

Run a full-coverage selector-v4 or PR95-style compact base under the hard byte
ceilings, feed its archive/projection into this profiler, then pass the emitted
`hprc_mlx_component_neutralization_profile.json` back into
`tools/run_compact_renderer_mlx_spine_runner.py --mlx-profile ...` so section
bytes are admitted or cut by measured full-video value-per-byte.
