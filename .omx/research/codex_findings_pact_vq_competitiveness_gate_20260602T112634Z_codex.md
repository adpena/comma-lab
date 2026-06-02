# Codex Findings: PACT-VQ Competitiveness Gate

Created: 2026-06-02T11:26:34Z

Axis: `[macOS-MLX research-signal]`, false-authority. This is not a contest CPU/CUDA score, promotion claim, rank claim, or kill claim.

## Artifacts

- Gate report: `.omx/research/pact_vq_competitiveness_gate_20260602T112634Z.json`
- Gate CLI: `tools/gate_pact_nerv_vq_competitiveness.py`
- Gate module: `src/tac/substrates/pact_nerv_vq/competitiveness_gate.py`
- Focused tests: `src/tac/substrates/pact_nerv_vq/tests/test_competitiveness_gate.py`
- Codec sweep: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/decoder_codec_sweep_20260602T0300Z/compact_decoder_codec_sweep_report.json`
- Source full-video replay: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/full_video_section_value_profile_20260602T0242Z/pact_nerv_vq_mlx_section_value_profile.json`
- Best-codec replay: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/int2_mixed_full_video_baseline_replay_20260602T0318Z/pact_nerv_vq_mlx_section_value_profile.json`

## Result

Verdict: `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION`

The best current codec remains `int2_mixed`, and its receiver proof passed. It is a real rate-axis improvement, but the full-video replay says the improvement is rate-only:

| row | archive bytes | advisory score | non-rate score | rate score | d_seg | d_pose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source replay | 39,874 | 90.66354296056916 | 90.63699250067236 | 0.02655045989679346 | 0.5048259229958058 | 161.237585550944 |
| int2 replay | 37,580 | 90.66201548013069 | 90.63699250067236 | 0.0250229794583312 | 0.5048259229958058 | 161.237585550944 |

Delta: `-2,294` replay archive bytes, `-0.0015274804384688423` advisory score, `0.0` non-rate score. The current PACT-VQ artifact is therefore not exact-spendable by itself, but it is not demoted or discarded: receiver-proven saved bytes remain a reusable rate primitive for later full-stack ordering and composition.

The family should not be abandoned, and neither should the byte-saving codec primitive. The conditional route is narrower: PACT-VQ must become a scorer-faithful retraining lane where decoder/codebook bytes carry real SegNet/PoseNet value, or those bytes must be removed/collapsed. The current rate envelope is attractive; the current fit is not. Order matters: the rate primitive can be useful after decoder/codebook fit improves, after section ordering changes, or when bundled with another carrier whose distortion term is already under control.

## System Intelligence

The gate is now executable and tested, so future compact PACT-VQ artifacts can be classified without redoing this reasoning by hand:

```bash
.venv/bin/python tools/gate_pact_nerv_vq_competitiveness.py \
  --codec-sweep-report /path/to/compact_decoder_codec_sweep_report.json \
  --source-replay-profile /path/to/source/pact_nerv_vq_mlx_section_value_profile.json \
  --best-codec-replay-profile /path/to/best/pact_nerv_vq_mlx_section_value_profile.json \
  --output-json .omx/research/pact_vq_competitiveness_gate_<stamp>.json
```

The gate stays false-authority even when a local row is good. A good local row becomes `LOCAL_CANDIDATE_FOR_EXACT_SPEND_TRIAGE`, not a promotion or score claim. A byte-saving but distortion-failed row becomes `PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION`, not a demotion.

## Blockers

- `mlx_local_replay_is_false_authority`
- `contest_cpu_cuda_exact_eval_not_executed`
- `pact_vq_distortion_not_competitive_at_current_fit`
- `best_codec_improves_rate_only_distortion_unchanged`
- `codec_sweep_report_needs_replay_attachment`

## Next Work

1. Attach replay-aware adjudication to the compact codec sweep so stale `full_video_mlx_scorer_replay_not_attached` blockers can be superseded by a matching replay gate.
2. Preserve `int2_mixed` as the default current PACT-VQ codec portfolio winner for rate and make it available to full-stack composition, but do not exact-eval this rate-only artifact before a fit gate.
3. Continue PACT-VQ through scorer-faithful retraining, structural decoder/codebook byte collapse, or carrier-order/bundling experiments that materially reduce `d_seg` and `d_pose` at the same small byte scale.
4. Promote only after receiver proof plus full-video local distortion gate, followed by claimed paired contest CPU/CUDA exact replay.
