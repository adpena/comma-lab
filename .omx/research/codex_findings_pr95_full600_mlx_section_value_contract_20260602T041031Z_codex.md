# PR95 Full-600 MLX Replay Section-Value Contract - Codex Findings 2026-06-02

## Verdict

The PR95 compact-carrier pivot now has a full-coverage MLX advisory replay and
bounded-runner attachment, but it is not exact authority and it still lacks
per-section neutralization or added-section value evidence. The score-lowering
signal is concrete: the next spend target is not more generic planning, it is
section replay for `decoder_qw` and `latents_rc`, followed by long score-aware
training or faithful Stage-8 continuation if the bytes remain valuable.

## Live Artifact Evidence

- Runner report:
  `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/pr95_hnerv_stage8_faithful_scoreaware_600pair_1ep_smoke_20260602T0405Z/compact_renderer_mlx_spine_runner_report.json`
- Archive:
  `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/pr95_hnerv_stage8_faithful_scoreaware_600pair_1ep_smoke_20260602T0405Z/pr95_hnerv_mlx_training/pr95_public_archive.zip`
- Archive bytes/SHA:
  `178357` bytes,
  `09eacafaba78d98bca869a0f224d0edfa971af5d7c9293b8e7ce72b7f58d36a0`
- Receiver proof:
  `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/pr95_hnerv_stage8_faithful_scoreaware_600pair_1ep_smoke_20260602T0405Z/receiver_proof/pr95_hnerv_receiver_proof.json`
- Corrected MLX section profile:
  `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/pr95_hnerv_stage8_faithful_scoreaware_600pair_1ep_smoke_20260602T0405Z/full_video_mlx_value_profile_20260602T0412Z/pr95_hnerv_mlx_section_value_profile_corrected_20260602T0434Z.json`
- Corrected bounded-runner plan:
  `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/pr95_hnerv_stage8_faithful_scoreaware_600pair_1ep_smoke_20260602T0405Z/hprc_spine_bounded_runner_plan_with_full600_profile_corrected_20260602T0434Z.json`

The MLX advisory components for the full 600-pair replay are:

- `avg_segnet_dist = 0.0006151750350545626`
- `avg_posenet_dist = 3.4939636860826796e-05`
- `nonrate_score = 0.08020965074990809`
- `rate_term = 0.11876060530201112`
- `canonical_score = 0.1989702560519192`

This is close enough to be useful for routing, but it remains
`[macOS-MLX research-signal]`. It is not a contest CPU/CUDA score, not
promotion authority, and not a rank/kill signal.

## Contract Fix Landed

The previous profiler/runner wording conflated two different missing facts:

1. full-video baseline MLX replay; and
2. per-section neutralization or candidate-addition replay.

After the full-600 replay exists, the correct remaining blocker is
`section_neutralization_or_ablation_replay_missing`, not
`full_video_mlx_section_value_replay_missing`.

The profiler now also distinguishes candidate-spend sections from packet
metadata:

- `decoder_qw`: `162289` bytes, rate cost `0.10806158364324407`,
  missing section neutralization/addition replay.
- `latents_rc`: `15868` bytes, rate cost `0.010565849868142616`,
  missing section neutralization/addition replay.
- `rdo_plan`: `495` bytes, metadata/no runtime spend replay required.
- `receiver_state`: `80` bytes, metadata/no runtime spend replay required.

The regenerated bounded-runner plan has blockers:

- `contest_cpu_cuda_exact_eval_not_executed`
- `some_sections_missing_value_per_byte_measurement`

and no longer asks for baseline full-video MLX replay.

## Score-Lowering Implication

The one-epoch PR95 score-aware control export is not a family-demotion artifact.
It confirms custody and routing, but the artifact still records:

- `pr95_mlx_scoreaware_teacher_distillation_is_advisory_not_exact_contest_loss`
- `pr95_stage_hparams_and_cosine_schedules_not_all_source_matched`
- `pr95_qat_c1a_and_resume_semantics_not_ported_to_mlx`
- `pr95_export_forward_parity_not_established`
- `pr95_hnerv_stage8_muon_continuation_not_wired`

So the immediate frontier work is:

1. run decoder/latent section neutralization or candidate-addition replay on the
   full-video MLX cache;
2. use the measured value-per-byte rows to decide whether decoder bytes should
   be protected, recoded, QAT-shaped, or compressed by scale/bitplane;
3. continue many-epoch score-aware PR95/HiNeRV/SNeRV training only after the
   packet spine and receiver proof stay byte-closed;
4. exact-gate only if local evidence plausibly beats or approaches the current
   frontier and all exact blockers are removed.

## Verification

- `ruff check tools/profile_compact_renderer_mlx_section_value.py src/tac/substrates/hprc/spine_bounded_runner.py src/tac/tests/test_profile_compact_renderer_mlx_section_value.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py`
- `pytest src/tac/tests/test_profile_compact_renderer_mlx_section_value.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q`
- Result: `17 passed`.

