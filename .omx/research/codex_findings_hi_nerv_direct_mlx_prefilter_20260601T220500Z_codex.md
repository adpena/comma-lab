# HiNeRV direct MLX prefilter landed 2026-06-01

Codex finding: the HiNeRV 128-pair PR95-curriculum smoke is rate-small but still
distortion-dead under the MLX scorer prefilter. It must not enter local CPU
replay or exact auth until training changes the fit axis materially.

Artifacts:

- Candidate runner report:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_128pair_real_teacher_pr95curriculum_20260601T215234Z/compact_renderer_mlx_spine_runner_report.json`
- Direct candidate cache:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_128pair_direct_mlx_prefilter_20260601T2200Z/candidate_cache_report.json`
- Reference cache:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_128pair_direct_mlx_prefilter_20260601T2200Z/reference_cache/manifest.json`
- MLX response:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_128pair_direct_mlx_prefilter_20260601T2200Z/mlx_scorer_response.json`
- Absorbed bounded-runner report:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_128pair_profile_absorb_20260601T2200Z/compact_renderer_mlx_spine_runner_report.json`

Result, false-authority MLX gpu research-signal:

- n_samples: 128
- archive bytes: 46,556
- score: 90.77733730741079
- avg_segnet_dist: 0.5059117504861206
- avg_posenet_dist: 161.24370777606964
- rate contribution: 0.03099972942155581
- direct receiver raw sha256:
  `f3e1659c71fc7e4e7281cf5bd1b20136fb45374fdc5bdaf0cb042582a6e2cd40`
- upstream modules.py sha256:
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`

Code hardening landed in this slice:

- `tools/materialize_mlx_scorer_cache_from_submission.py` now supports
  `--receiver-direct-cache` for deterministic receiver-rendered scorer caches.
  Existing `--hprc-direct-cache` remains compatible.
- HiNeRV HIV1 archives now direct-render into scorer-input caches without
  writing multi-GB raw scratch, with
  `hi_nerv_direct_receiver_render_cache_identity_audit.json`.
- `tac.substrates._shared.inflate_runtime.rgb_pair_to_uint8_frames` centralizes
  clamp, bicubic resize, round, and uint8 lowering so direct-cache and
  `inflate.py` share the same raw-byte semantics.
- `tac.substrates.hi_nerv.inflate.build_model_from_archive` centralizes HIV1
  archive-to-model reconstruction for shell inflate and direct-cache.
- MLX scorer-response tools now accept `--upstream-dir`; response payloads
  record scorer upstream snapshot custody. This fixes SSD worktree layouts
  where source lives on `/Volumes/VertigoDataTier` and upstream scorer weights
  remain in `/Users/adpena/Projects/pact/upstream`.

Verdict:

Do not spend CPU replay or exact auth on this 128-pair HiNeRV candidate. The
rate thesis is alive, but distortion is not close. Next score-lowering work is
native rate-aware and scorer-aware training, not replaying this archive.

Next required queue step:

Use the now-working direct-cache/MLX-response path as the gate for 32/128/600
HiNeRV and SNeRV campaigns. Promote only candidates with materially lower MLX
distortion into local CPU replay. Full-video 600-pair response remains required
before exact auth; sampled 128-pair profiles are demotion/prior signal only.
