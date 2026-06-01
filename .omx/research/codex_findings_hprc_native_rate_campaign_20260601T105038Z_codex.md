# HPRC native rate-aware campaign smoke landed

## Status

- Schema: `codex_findings_hprc_native_rate_campaign.v1`
- Generated: `2026-06-01T10:50:38Z`
- Axis: `[macOS-CPU advisory]`
- Authority: false. No score claim, no promotion claim, no exact-dispatch readiness.
- Commit: `3fffa5230 Add native rate-aware HPRC training surface`

## What Changed

Native HPRC training now consumes a P18/P19-derived residual-token protection
surface before archive export. Protection semantics are explicit:
`1=protect_from_rate_pressure`, `0=safest_to_shrink`. The queue builder can
compile the protection surface as a pre-training queue step, then run local
training, rate-collapse transcode, and follow-up gates through
`experiment_queue.v1`.

## Executable Evidence

Tiny queue smoke:

- Queue root: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_native_rate_queue_smoke/hprc_native_rate_smoke_20260601T104144Z`
- Training root: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_native_rate_smoke_20260601T104144Z`
- Worker result: 3 started, 3 succeeded, 0 failed.
- Surface: 4 frames, 8x8x3 residual grid, `rate_pressure_mean=0.318750`.
- Archive: 47,890 bytes, receiver proof requested and produced, false-authority fields false.

Real P18/P19 bounded campaign:

- Queue root: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_native_rate_campaign/hprc_native_rate_b7106_real_p18p19_20260601T104929Z`
- Training root: `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_native_rate_b7106_real_p18p19_20260601T104929Z`
- P19 input: `/Volumes/VertigoDataTier/pact/scorer_region_cascade_b7106_campaign_20260531T214623Z/nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch/p19_posenet_null_pairs.json`
- P18 input: `/Volumes/VertigoDataTier/pact/scorer_region_cascade_b7106_campaign_20260531T214623Z/nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch/p18_segnet_region_waterfill.json`
- Worker result: 8 started, 8 succeeded, 0 failed.

Campaign rows:

| pairs | P19 in range | rate pressure mean | native residual nonzero frac | training bytes | best collapsed bytes | saved bytes | best variant |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 4 | 0.105420 | 0.963033 | 66,386 | 40,537 | 25,849 | `residual_tokens_dz1_qd12` |
| 128 | 19 | 0.125186 | 0.910299 | 147,534 | 47,246 | 100,288 | `residual_tokens_dz1_qd12` |

## Verdict

The loop is executable and non-orphaned: P18/P19 priors now affect HPRC
train-time rate pressure, then flow through queue-owned rate collapse and
follow-up gates. This is not a frontier claim. Both bounded rows stop at
`gate_archive_rate_before_local_replay`; exact CPU/CUDA remains blocked until a
600-pair/full-video candidate clears local rate/replay gates.

The next score-moving step is to promote this from prefix projection to a
full-video 600-pair native-rate campaign, with real full-video P18/P19 surface
scope, archive-rate gate, receiver proof, local replay, and exact-auth refusal
unless local replay wins.
