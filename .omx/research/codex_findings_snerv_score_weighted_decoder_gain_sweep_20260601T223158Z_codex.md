# Codex Findings: SNeRV Score-Weighted Decoder Gain Sweep

UTC: 2026-06-01T22:31:58Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Lane: `lane_snerv_score_weighted_hf_decoder_fit_smoke_20260601`

## Why This Sweep Exists

The first score-weighted HF decoder smoke used `--hf-decoder-saliency-gain 8.0`
and worsened PoseNet sharply versus the least-squares waterfill baseline. This
sweep tests whether that was merely an over-aggressive gain setting.

Fixed config:

- `--n-pairs 1`
- `--levels 4`
- `--bits-per-coeff 5.0`
- `--step-map-coder-mode waterfill`
- `--step-map-waterfill-bits-per-coeff 6.0`
- `--hf-decoder-fit-mode score_weighted`

The sweep rows are advisory-only and false-authority by construction.

## Control And Rows

The least-squares waterfill control remains the best row:

- Path: `.omx/research/snerv_inverse_steg_advisory_waterfill_20260601T221200Z.json`
- SHA-256: `bca8424b18f5153e0a7c3b0746938219846ac21f740fcdd853c7dec14e6b5d32`
- Archive bytes: `33754`
- `d_seg_linf`: `0.02264404296875`
- `d_pose_linf`: `2.1390697956085205`
- `score_linf`: `6.911887587116307`

Generated gain rows:

| Gain | Artifact | SHA-256 | Archive bytes | d_seg_linf | d_pose_linf | score_linf |
|---:|---|---|---:|---:|---:|---:|
| 0.0 | `.omx/research/snerv_score_weighted_decoder_fit_gain0p0_advisory_20260601T223158Z.json` | `4273f61b734f9dce413191db60aad48d347472d3630b3848a511406d6fa08c32` | `33824` | `0.02264404296875` | `2.1390697956085205` | `6.911934382873807` |
| 0.25 | `.omx/research/snerv_score_weighted_decoder_fit_gain0p25_advisory_20260601T223158Z.json` | `15d77a719b901f1a3e081366432cc2920d5629158e19a9cbc33eee3289c985e0` | `33825` | `0.022267659505208332` | `5.7368927001953125` | `9.823517111445413` |
| 0.5 | `.omx/research/snerv_score_weighted_decoder_fit_gain0p5_advisory_20260601T223158Z.json` | `29ee263c1880c352eab423d38f2d627bf3dea2d19d4a583821c085605ac77ee5` | `33824` | `0.022298177083333332` | `5.7418036460876465` | `9.82980920230965` |
| 1.0 | `.omx/research/snerv_score_weighted_decoder_fit_gain1p0_advisory_20260601T223158Z.json` | `3d152a111e12d1d74ebc7543b8dcbef1ea8292edcd4f70a3d6527b8109cac4c6` | `33824` | `0.022292296091715492` | `5.744192600250244` | `9.830876055205663` |
| 2.0 | `.omx/research/snerv_score_weighted_decoder_fit_gain2p0_advisory_20260601T223158Z.json` | `40463720c61f7fed81e1f7eaa5397fb75caca66c2a49e53b4f9bd23f72505fd0` | `33824` | `0.022292613983154297` | `5.745372772216797` | `9.831654838175454` |
| 8.0 | `.omx/research/snerv_score_weighted_decoder_fit_waterfill_advisory_20260601T222541Z.json` | `b1e0bed066f20a5aa923dabccfe0bdbbcedc5cd4c71060b9bed9078d31494c46` | `33824` | `0.02230326272547245` | `5.746245861053467` | `9.833247919737238` |

Combined sweep artifact:

- Path: `.omx/research/snerv_score_weighted_decoder_fit_gain_sweep_20260601T223158Z.json`
- SHA-256: `87210984063ab158484049a41d86bd2a5f363e6f62b1e8c5f86dc0ad53380477`
- Best pose row: `least_squares_baseline_existing`
- Best score row: `least_squares_baseline_existing`
- Any positive score-weighted gain improves score versus baseline: `false`
- Any positive score-weighted gain improves pose versus baseline: `false`

Adjudication artifact:

- Path: `.omx/research/snerv_score_weighted_decoder_fit_gain_sweep_adjudication_20260601T223158Z.json`
- SHA-256: `967eaae4acd445a9221354dc93235b790c33702f4014abd38c253a1241168fe6`
- Rows: `7`
- Classification counts: `{"rate_below_frontier_pose_or_seg_destroyed": 7}`
- Ready for exact eval dispatch: `false`
- Promotion eligible: `false`
- Frontier score claim: `false`

## Verdict

NO-GO for scalar DWT-saliency gain tuning of the current linear HF decoder.

`gain=0` validates the control path by matching the least-squares distortion.
Every positive gain tested slightly reduces SegNet distortion but catastrophically
worsens PoseNet distortion. The bad `gain=8` result was not just too much gain;
the local direction itself is wrong for PoseNet.

Do not exact-dispatch this sweep. Do not promote it. Do not treat it as a score
claim, rank claim, or kill claim.

## Next Code Move

Keep the receiver-visible waterfill packet work. Move score-aware decoder
optimization out of scalar DWT-band weighting and into one of:

1. reconstructed-frame/scorer-loop decoder-weight training;
2. learned/nonlinear HF decoder QAT with PoseNet and SegNet loss in-loop;
3. per-component protective weighting that treats PoseNet as the hard constraint
   instead of letting SegNet saliency dominate the fit.

The success metric must be lower replayed `d_pose_linf` and `d_seg_linf` at
similar archive bytes, not lower weighted HF residual.

## Environment Note

A first rerun attempt with system `python3` failed because that interpreter did
not have `pywt`. The actual sweep used `.venv/bin/python`, which has
PyWavelets `1.8.0`. No dependency or global environment change was made.
