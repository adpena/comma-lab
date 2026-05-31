# Codex Findings: Grouped Scorer-Region Cascade Queue

Generated UTC: 2026-05-31T03:04:00Z

## Scope

Expanded the P18/P19/P11 receiver-closed distortion-budget lane beyond a single
12-pair leaf into a queue-owned grouped campaign over PoseNet-null fraction,
SegNet region count, receiver patch pair count, RGB/YUV-proxy deltas, selector
codec family, and repack order.

## Landed Engineering

- Added `scorer_region_selector_cascade_campaign_queue` for deterministic grouped
  campaign enumeration, `experiment_queue.v1` materialization, and partial/full
  campaign harvests.
- Fixed selected-survivor custody: receiver patch materialization and shell
  inflate output-change proof now consume `selected_local_survivor_archive` from
  the chain report instead of a fixed P15 path.
- Added shared scorer-region chain-report archive resolver in
  `tac.optimization.scorer_region_waterfill`.
- Extended the exact-ready bridge to consume local CPU advisory, local CPU eureka,
  MLX response, and scorer-response dataset artifacts.
- Fixed campaign harvest ranking so MLX partial acquisition cannot outrank a
  failed full local CPU gate.

## Empirical Result

Canonical queue:
`.omx/research/scorer_region_selector_cascade_campaign_20260531T023734Z/queue.json`

First completed variant:
`nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch`

Evidence:

- Receiver patch changed real shell-inflated output on `0.mkv`: 139421 differing
  raw bytes, shape preserving, no proof blockers.
- Local CPU advisory score: `0.1920003362662307`.
- Current CPU frontier pointer used by eureka gate: `0.19198533626623068`.
- Local CPU delta versus frontier: `+0.000015000000000015001`, so this variant is
  not exact-auth-worthy.
- MLX partial acquisition over 12 pairs reported `0.18445031691375097`, but the
  full local CPU gate falsified it as a promotion signal.
- Exact-ready bridge now records `local_cpu_eureka_trigger_false` and
  `local_cpu_score_not_below_auth_frontier`.
- Retention pass deleted the 3.4G raw/cache payloads for the completed variant;
  retained local component directory is about 320K.

## Queue State

After the first full loop and bridge refresh:

- `succeeded`: 16
- `queued`: 177
- `ready_for_exact_eval_dispatch`: false
- `score_claim`: false

Partial campaign report:
`.omx/research/scorer_region_selector_cascade_campaign_20260531T023734Z/partial_campaign_report.json`

## Verdict

The grouped cascade infrastructure is now real and receiver-closed, but the
first low-amplitude RGB frame-1 waterfill variant is CPU-negative. The immediate
next search should keep the queue machinery and change the operator set:
stronger/different deltas, YUV-native receiver patches, grouped region coverage,
and variants selected by CPU-calibrated MLX acquisition rather than raw MLX
partial score.

## Next 12-Week Tranche

1. Drain the remaining grouped campaign variants under queue control, but gate
   exact auth strictly on local CPU eureka and preserve MLX/CPU disagreement as
   calibration data.
2. Add YUV-native receiver patching, grouped frame/pair/region operation sets,
   and per-region selector codecs so the P18/P19/P11 cascade searches structure
   instead of isolated deltas.
3. Train the acquisition layer to predict full CPU outcome from MLX partial rows,
   proof metadata, pair/region features, and prior bridge blockers.
4. Promote only CPU-positive, receiver-proven, byte-closed candidates to exact
   CPU auth, then CUDA anchors.
5. Keep PR95/HNeRV MLX as the control arm: bounded longer training, export
   parity, archive smoke, and the same final-rate/distortion-budget cascade.
