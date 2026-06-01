# PACT/VQ int2 full-video section-value harvest - Codex findings

UTC: 2026-06-01T22:24:26Z

## Scope

Harvested the completed 600-pair PACT-NeRV-VQ compact carrier export and the
four-codec decoder sweep. The best rate-only codec variant was
`int2_scale_bundled`, then priced with full-video MLX scorer replay over
section-neutralized archive variants.

All rows are `[macOS-MLX research-signal]` and false-authority. They are not
contest CPU/CUDA score claims.

## Artifact Custody

- Baseline trained export:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_2000ep_codex_20260601T194633Z/pact_nerv_vq_mlx_training/archive.zip`
- Baseline export bytes: 192,810
- Baseline receiver proof: passed
- Codec sweep report:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_2000ep_codec_sweep_codex_20260601T215400Z/compact_decoder_codec_sweep_report.json`
- Best codec variant: `int2_scale_bundled`
- Best codec archive bytes: 54,930
- Best codec SHA-256:
  `bebcadb846182fbc666b3151202fe46e8e15a8a991c6bbbb7085cc65dacc66d7`
- Full-video section-value profile:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_2000ep_int2_section_value_codex_20260601T220100Z/pact_nerv_vq_mlx_section_value_profile.json`
- Bounded-runner plan with profile consumed:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_2000ep_int2_section_value_codex_20260601T220100Z/hprc_spine_bounded_runner_plan_with_int2_section_value.json`

## Full-Video MLX Replay Verdict

Baseline `int2_scale_bundled` full-video MLX advisory:

- `avg_segnet_dist`: 0.5258133446673552
- `avg_posenet_dist`: 141.26604855855305
- `score_rate_contribution`: 0.03653767833467291
- `canonical_score` advisory: 90.20324809883085

This is not exact-gate plausible for frontier movement. Do not spend contest
CPU/CUDA exact budget on this `int2_scale_bundled` artifact unless a later
trained artifact materially improves the full-video MLX replay.

Section-neutralization signals:

- `decoder_qw`: protect. Removing 16,441 archive bytes worsened total MLX
  advisory by +0.4242007215117525 despite lower rate; nonrate value per removed
  KiB was +0.027102467195758716.
- `selectors_rc`: protect strongly. Removing 682 bytes worsened total MLX
  advisory by +4.081158769025464; nonrate value per removed KiB was
  +6.128404096873082.
- `codebooks_q`: cut/demote for this artifact. Removing 14,852 bytes improved
  total MLX advisory by -1.5538006766047516. This is a substrate/training signal,
  not a promotion signal, because the resulting score remains around 88.65.
- `residual_rc`: absent; keep demoted unless a measured residual candidate
  satisfies `delta_nonrate + rate_cost < 0`.

## Engineering Fixes Triggered

The first full-video profile attempt exposed stale path coupling:

- `profile_pact_nerv_*_mlx_section_value.py` conflated code root, upstream
  scorer/video root, and reference-cache root.
- `build_mlx_scorer_response_payload(_batch)` inferred `repo_root/upstream`.
- `run_mlx_scorer_response_cache.py` lacked repo bootstrap, so SSD worktrees
  could import stale local code.

The landed fix makes `--upstream-dir` explicit through section-value profilers,
bounded-runner work orders, compact runner plan writers, and MLX scorer-response
CLIs. Reference cache defaults are now repo-root relative rather than
tool-root absolute.

## Remaining Blockers

- `mlx_local_response_is_advisory_not_score_authority`
- `class_region_boundary_scopes_require_logits_or_boundary_cache_extension`
- `contest_cpu_cuda_exact_eval_not_executed`

Next action: launch the next long PACT/VQ run with `--coder-aware-qat` only
after current heavy MLX/inflate jobs are clear, then rerun codec sweep plus
full-video section-value replay. Exact-gate only if full-video local evidence is
plausibly frontier-adjacent.
