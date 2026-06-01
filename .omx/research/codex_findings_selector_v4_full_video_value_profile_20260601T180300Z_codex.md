# Codex findings - selector-v4 full-video value profile

- **Date:** 2026-06-01T18:03:00Z
- **Axis:** `[macOS-MLX research-signal]`, false authority; no score claim.
- **Lane:** `pact_nerv_selector_v4` compact base / HPRC spine section pricing.

## What landed

A full-coverage `pact_nerv_selector_v4` compact archive was regenerated after the raw receiver-custody fix and proved through generated `inflate.sh` as a real contest `.raw` file.

Artifacts:

- Archive: `/Volumes/VertigoDataTier/pact/compact_selector_v4_full600_1epoch_rawfixed_codex_20260601T173709Z/pact_nerv_selector_v4_mlx_training/archive.zip`
  - bytes: `27522`
  - sha256: `de7a7677805e6edb07cd02f7d1b43e8acc1c2bc6fed5af9a4d80a4dbc3159f9b`
- Receiver proof: `/Volumes/VertigoDataTier/pact/compact_selector_v4_full600_1epoch_rawfixed_codex_20260601T173709Z/pact_nerv_selector_v4_mlx_training/receiver_proof/pact_nerv_selector_v4_mlx_receiver_proof.json`
  - bytes: `3301`
  - sha256: `922836d915531b91b0ecb330a39be6ddff7affa77e1a32b83956c99a423c838c`
  - proof result: `runtime_consumption_proof_passed=true`, `receiver_contract_satisfied=true`, `receiver_output_kind=file`, `receiver_output_bytes=3662409600`
- Full-video section-value profile: `/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pact_nerv_pact_nerv_selector_v4_psv4_ccb349766791f52a/hprc_mlx_component_neutralization_profile.json`
  - bytes: `24662`
  - sha256: `48c92aa96acfa80e5e1c70fc716a5c75f722d7e3227bf35f61b6b3f011f8b097`
- Bounded-runner plan with profile feedback: `/Volumes/VertigoDataTier/pact/compact_selector_v4_full600_1epoch_rawfixed_codex_20260601T173709Z/hprc_spine_bounded_runner_plan_with_full_video_profile.json`
  - bytes: `21829`
  - sha256: `9696c75e12b9f098b54e26ed350732803eed66de3e4c750e21f585dca2763284`

## Measured section verdict

Full-video profile rows:

- `decoder_qw`: removing `6155` archive bytes worsened non-rate by `+0.2962359723`; total MLX advisory delta `+0.2921376105`. **Protect.**
- `latents_rc`: removing `9551` archive bytes had measured non-rate delta `0.0`; total MLX advisory delta `-0.0063596189`. **Cut or redesign; current 1-epoch row is not using these bytes.**
- `selectors_rc`: removing `30` archive bytes had measured non-rate delta `0.0`; total MLX advisory delta `-0.0000199758`. **Cut.**

The bounded runner now emits explicit section spend recommendations from measured full-video evidence:

- `decoder_qw` -> `protect_section_bytes_measured_value_exceeds_rate_price`
- `latents_rc` -> `cut_section_bytes_measured_removal_improves_objective`
- `selectors_rc` -> `cut_section_bytes_measured_removal_improves_objective`

## Interpretation

This does not prove the compact lane is score-competitive. It proves the current full-coverage 1-epoch selector-v4 packet is receiver-custody-valid, under hard byte ceilings, and suitable for disciplined long-training decisions. The immediate signal is that decoder weights carry all measured value in the current short run; latents/selectors are dead bytes at this training point. For score lowering, long training must either make latents/selectors earn their bytes under the score-exact oracle or remove/recode them automatically.

## Next action

Run long selector-v4/HNeRV/RNeRV compact training under the score-exact oracle with section pricing in-loop. Promote only archive-bound candidates whose full-video section profile keeps `decoder_qw` protected while making latent/selector bytes score-positive, or whose cut-latent/cut-selector packet remains receiver-proven and improves local replay enough to justify exact CPU/CUDA gating.
