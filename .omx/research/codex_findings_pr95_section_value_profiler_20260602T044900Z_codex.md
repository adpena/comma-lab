# PR95 Section-Value Profiler - Codex Findings 2026-06-02

## Verdict

PR95/HNeRV now has a real section-value profiler for the common HPRC packet
spine. It materializes byte-closed neutralized archives for `decoder_qw` and
`latents_rc`, runs them through the PR95 `inflate.sh` runtime, builds MLX scorer
caches, emits shared `hprc_mlx_component_neutralization_profile.v1` rows, and
stays false-authority.

The bounded runner also now knows the PR95 profiler and emits an executable
full-video section-value work order when the PR95 source projection carries the
runtime submission directory.

## Live Smoke Evidence

Smoke output:

`/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pr95_hnerv_live_2pair_smoke_20260602T0420Z`

Profile:

`/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pr95_hnerv_live_2pair_smoke_20260602T0420Z/pr95_hnerv_mlx_section_value_profile.json`

The smoke used the live 600-pair PR95 archive but scored only two pairs after
inflate. PR95 `inflate.sh` still renders the full raw output before the cache
tool samples, so this is a real runtime-bound smoke rather than a cheap parser
only test.

Sampled advisory deltas:

- `decoder_qw`: archive `178357 -> 16450` bytes, removed `161907` bytes,
  sampled `delta_nonrate = 94.89359603298607`, sampled objective delta
  `94.78578880746291`, verdict `protect_candidate_value_exceeds_rate_price`.
- `latents_rc`: archive `178357 -> 162507` bytes, removed `15850` bytes,
  sampled `delta_nonrate = 9.073133824789359`, sampled objective delta
  `9.062579960382372`, verdict `protect_candidate_value_exceeds_rate_price`.

This sampled result says both sections carry score signal and should not be
blind-cut. It is not a full-video section-value verdict and not score authority.
The runner correctly blocks it with `sampled_mlx_prefilter_requires_full_video_rerun`.

Disk hygiene check: no raw files or files over 500 MB remained under the smoke
output after completion; the retained SSD artifact was about `28M`.

## Runtime-Bound Work Order

Runtime-bound projection:

`/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pr95_hnerv_live_runtime_bound_projection_20260602T0445Z/pr95_hnerv_spine/pr95_hnerv_runtime_bound_representation_spine_manifest.json`

Runtime-bound bounded plan:

`/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pr95_hnerv_live_runtime_bound_projection_20260602T0445Z/hprc_spine_bounded_runner_plan.json`

The generated full-video profile work order is queued, not blocked, and carries:

- `tools/profile_pr95_hnerv_mlx_section_value.py`
- `--archive <live PR95 archive>`
- `--submission-dir /Users/adpena/Projects/pact/experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon`
- `--sections decoder_qw latents_rc`
- `--max-pairs 600`
- SSD output under `/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/...`

This is the next real score-lowering measurement: full-video PR95 section value
so decoder/latent bytes can be protected, QAT-shaped, recoded, or attacked only
where `delta_nonrate + rate_cost` proves it.

## Verification

- `ruff check tools/profile_pr95_hnerv_mlx_section_value.py tools/build_hprc_representation_spine_projection.py tools/run_compact_renderer_mlx_spine_runner.py src/tac/substrates/hprc/representation_spine.py src/tac/substrates/hprc/spine_bounded_runner.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py src/tac/substrates/hprc/tests/test_representation_spine.py src/tac/tests/test_profile_pr95_hnerv_mlx_section_value.py`
- `pytest src/tac/tests/test_profile_pr95_hnerv_mlx_section_value.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py src/tac/substrates/hprc/tests/test_representation_spine.py -q`
- Result: `31 passed`.

