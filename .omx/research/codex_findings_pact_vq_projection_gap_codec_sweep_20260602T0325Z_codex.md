# Codex Findings: PACT/VQ Projection-Gap Codec Sweep

Generated: 2026-06-02T03:25Z
Axis: `[macOS-MLX research-signal]` and receiver-proof custody only. No contest CPU/CUDA score authority.

## Artifacts

- Source archive: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/runner_output/pact_nerv_vq_mlx_training/archive.zip`
- Source SHA-256: `553814dd43eecb67aec15d45a490f010752a56279522be47155774927c7a0db9`
- Source bytes: `39,935`
- Receiver proof: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/runner_output/pact_nerv_vq_mlx_training/receiver_proof/pact_nerv_vq_mlx_receiver_proof.json`
- Full section-value profile: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/full_video_section_value_profile_20260602T0242Z/pact_nerv_vq_mlx_section_value_profile.json`
- Codec sweep: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/decoder_codec_sweep_20260602T0300Z/compact_decoder_codec_sweep_report.json`
- Best int2 replay: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/int2_mixed_full_video_baseline_replay_20260602T0318Z/pact_nerv_vq_mlx_section_value_profile.json`

## Findings

The 2,000-epoch `capacity_l16_e32_k64_ch48` PACT/VQ repair exported a byte-closed receiver-proven archive, but it remains distortion-uncompetitive:

- Baseline full-video MLX advisory score: `90.66354296056916`
- SegNet: `0.5048259229958058`
- PoseNet: `161.237585550944`
- Rate contribution: `0.02655045989679346`

Section-value replay showed decoder/codebook bytes are currently negative structural signal, while selectors still carry value:

- `decoder_qw`: remove `14,523` archive bytes, non-rate delta `-0.03513397161614762`, total delta `-0.04480424119233817`; cut/repair signal.
- `codebooks_q`: remove `2,089` archive bytes, non-rate delta `-0.003319033532307003`, total delta `-0.004710012885382753`; cut/repair signal.
- `selectors_rc`: remove `442` archive bytes, non-rate delta `+0.0009570955982383111`, total delta `+0.0006627859409604753`; protect.
- `residual_rc`: absent; demoted.

The decoder codec portfolio sweep is real and receiver-proven. Best variant:

- Codec: `int2_mixed`
- Archive: `/Volumes/VertigoDataTier/pact/hprc_projection_gap_repairs/pact_nerv_vq_pact_nerv_vq_pvq_projection_gap_aa80443421cfbcb2/capacity_l16_e32_k64_ch48/decoder_codec_sweep_20260602T0300Z/int2_mixed/archive.zip`
- SHA-256: `8b319bfc8786346694e7d532c298187f1de4ac0f518c4c26a3ddad9c88c6ac2d`
- Bytes: `37,641`
- Receiver proof passed.
- Full-video MLX replay preserved SegNet/PoseNet exactly versus the source candidate and improved advisory score only by the rate delta: `90.66354296056916 -> 90.66201548013069`.

## Verdict

This is a valid rate-axis byte win and a good integration proof for the codec portfolio, but it is not exact-gate plausible because the compact carrier is still severely underfit. The next score-lowering work should not spend exact CPU/CUDA on this artifact. It should use the signal to repair the training/export objective: decoder/codebook bytes are being paid without carrying enough score value, selectors are the only protected section, and the default post-export codec should consider `int2_mixed` before `int2_scale_bundled` for this family.

Required next actions:

1. Feed codec-sweep reports into bounded-runner acquisition so archive-bound codec winners can trigger baseline replay automatically.
2. Move PACT/VQ capacity repair away from blindly larger decoder/codebook grids and toward scorer-faithful training that makes decoder/codebook bytes valuable.
3. Keep selectors protected unless a later full-video replay proves their value below rate price.
4. Exact gate only after a receiver-proven compact candidate has local full-video evidence plausibly near frontier.
