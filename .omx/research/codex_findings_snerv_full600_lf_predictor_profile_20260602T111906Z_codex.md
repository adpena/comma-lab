# Codex Findings: SNeRV Full-600 LF Predictor Profile

Created: 2026-06-02T11:19:06Z

Axis: `[macOS-CPU advisory]`, false-authority. This is not a score claim, promotion claim, rank claim, or exact CPU/CUDA result.

## What Ran

Decoded the full-600 SNeRV SNAR1 packet from:

`/Volumes/VertigoDataTier/pact/snerv_lf_delta_sharedshape_bits2p0_l2_affine_lastframe_package_full600_metadataonly_20260602T032545Z/candidate.snar`

Then measured receiver-decodable lossless LF residual grammars over all `3600` LF planes of shape `[110,146]`.

## Result

The current LF payload is `9,996,235` bytes. Simple deterministic LF predictors do not collapse it:

| candidate | compressed bytes | delta vs current LF payload |
| --- | ---: | ---: |
| raster_delta | 9,995,880 | -355 |
| gradient2d | 12,235,500 | +2,235,265 |
| prev_frame_same_channel | 11,148,564 | +1,158,329 |
| prev_pair_same_slot | 11,917,216 | +1,926,981 |
| linear_prev_frame_same_channel | 16,385,868 | +6,386,233 |
| prev_frame_plus_raster_delta | 11,701,208 | +1,700,973 |
| prev_pair_slot_plus_raster_delta | 12,456,576 | +2,456,341 |

## Verdict

This closes the obvious lossless deterministic LF-predictor branch negatively for the current full-600 packet. The existing raster-delta family is already near the measured simple-codec floor. The score-lowering work is not another handcrafted LF delta, previous-frame predictor, or 2D gradient predictor.

The next SNeRV work must change the representation before coding: learned/scorer-preserving LF generation, low-resolution/SR carrier, or score-aware decoder fit that reduces the entropy of the stored LF residuals. This is the same representation-size compatibility issue the PACT-VQ lane may be facing: tiny decoder/state bytes are not enough if the representation must carry too much explicit scorer-relevant signal.

## Authority Boundary

Remaining blockers:

- `paired_contest_cpu_cuda_auth_eval_missing`
- `contest_cpu_or_cuda_exact_axis_payload_required`
- `lane_dispatch_claim_required_before_exact_eval`

Do not promote, rank, kill, or submit from this result.
