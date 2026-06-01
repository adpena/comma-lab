# Codex Findings: PACT-NeRV Int8 Raw Adapter And Section Value

Generated: 2026-06-01T11:47:12Z
Author: Codex

## Summary

Executed the 98292-byte PACT-NeRV int8 selector row through the official raw
output adapter path, ran full local CPU replay, then ran full-video MLX section
pricing for decoder, latents, selectors, and residual admission.

The row is byte-excellent but distortion-bad. It should not receive exact spend
as-is. The useful signal is section-level: decoder and latent bytes are
protected; selectors are dead spend; residual tokens are absent and remain
demoted unless a future candidate proves `delta_nonrate + rate_cost < 0`.

## Raw Adapter Custody

Root:

- `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_decoder_raw_adapter_20260601Tlocal`

Archive:

- `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_decoder_raw_adapter_20260601Tlocal/archive.runtime_raw.zip`
- bytes: 98939
- sha256: `de4a236ba8a60caad07a9e66c0b7a44dd0a38de61216262b686e6efe469c3fb0`

Receiver proof:

- `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_decoder_raw_adapter_20260601Tlocal/receiver_proof_raw/hprc_spine_receiver_execution_report.json`
- result: passed
- receiver output: 3662409600 raw bytes

Local replay:

- `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_decoder_raw_adapter_20260601Tlocal/local_cpu_replay/local_submission_replay_summary.json`
- axis: `[macOS-CPU advisory]`
- evaluation passed: true
- local score estimate: 90.20437441851149
- SegNet distortion: 0.5048244
- PoseNet distortion: 157.26026917
- rate: 0.00263518
- score authority: false
- exact dispatch ready: false

Verdict: the archive is far below the hard byte ceilings, but the compact base
does not preserve enough scorer signal. It is a compact-base fidelity failure,
not a container/rate failure.

## Full-Video MLX Section Pricing

Profile:

- `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_section_value_20260601Tlocal/pact_nerv_selector_v3_mlx_section_value_profile.json`
- compatibility profile:
  `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_section_value_20260601Tlocal/hprc_mlx_component_neutralization_profile.json`
- scope: 600 pairs full video
- device: MLX GPU research signal
- authority: false

Rows:

| section | variant | archive bytes | bytes removed | delta non-rate | delta rate | delta total | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| decoder_qw | neutralize_decoder_qw | 38688 | 60251 | +0.1792958238 | -0.0401186678 | +0.1391771560 | protect |
| latents_rc | neutralize_latents_rc | 70890 | 28049 | +0.0460476872 | -0.0186766778 | +0.0273710094 | protect |
| selectors_rc | neutralize_selectors_rc | 98929 | 10 | +0.0 | -0.0000066586 | -0.0000066586 | cut/recode |
| residual_rc | residual_absent_no_admission | 98939 | 0 | +0.0 | +0.0 | +0.0 | demote absent residual |

The bounded runner now consumes this profile:

- `/Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_decoder_raw_adapter_20260601Tlocal/hprc_spine_bounded_runner_raw_with_section_value_v2.json`

Runner result:

- remaining blocker: `contest_cpu_cuda_exact_eval_not_executed`
- no missing section-value blocker remains
- decoder and latents: `admit_section_bytes_for_receiver_proof`
- selectors: `protect_or_shrink_by_smaller_recode_only`
- residual token row: `demote_residual_token_variant`
- receiver-state metadata: `metadata_contract_no_mlx_replay_required`

## Code Landed

Implementation commit already pushed:

- `8b96ba57b Add PSV3 MLX section value profiler`

It added:

- `src/tac/substrates/pact_nerv_selector_v3/section_value.py`
- `tools/profile_pact_nerv_selector_v3_mlx_section_value.py`
- `src/tac/substrates/pact_nerv_selector_v3/tests/test_section_value.py`
- projection/family-bound section evidence matching in
  `src/tac/substrates/hprc/spine_bounded_runner.py`

This memo accompanies the follow-up metadata fix that treats receiver-state
bytes as decode contract metadata rather than a falsely missing section replay.

## Verification

- `ruff check` passed on touched files.
- `pytest src/tac/substrates/pact_nerv_selector_v3/tests/test_section_value.py src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q` passed.
- `pytest src/tac/substrates/hprc/tests/test_spine_bounded_runner.py -q` passed after the receiver-state metadata fix.

## Next Action

Do not exact-dispatch this PACT int8 base by itself. Its current score failure
is too large. The next score-lowering work is to train/sweep a more faithful
compact base under the same raw-output packet spine, while keeping the measured
section rule:

- protect decoder bytes only when their full-video non-rate value beats rate;
- protect latent bytes only when their full-video non-rate value beats rate;
- cut or recode selectors unless a future runtime proves they drive pixels;
- admit residual tokens only when `delta_nonrate + rate_cost < 0`.
