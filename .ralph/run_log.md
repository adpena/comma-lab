# run log

## 2026-04-10T21:30:00-05:00 - promoted floor synchronized

- authoritative promoted floor: **1.33**
- variant: `dilated_h64`
- platform: `modal_a10g`
- evidence: `reports/raw/2026-04-10-dilated-h64-authoritative/robust_current-dilated-h64-authoritative-cpu-report.txt`
- mirrors are now expected to be derived from canonical promoted_result.json

## 2026-04-11T23:46:58 — mask_renderer on modal_a10g ep70

- proxy=None
- Notes: L1 loss 0.031, Phase 1 pretrain, 31s/ep

## 2026-04-11T23:47:00 — dp_sims on modal_a10g ep80

- proxy=None
- Notes: L1 loss 0.018 LEADING, Phase 1, 17.7s/ep

## 2026-04-11T23:47:00 — wavelet on modal_a10g ep110

- proxy=8.2
- Notes: Phase 2 scorer training, 12.9s/ep

## 2026-04-11T23:47:00 — dilated_h64 on modal_a10g ep103

- proxy=1.407
- Notes: Steadily improving from 1.476

## 2026-04-12T00:25:49 — dp_sims on modal_a10g ep100

- proxy=4.54
- PoseNet: 0.75700000
- SegNet: 0.05100000
- Notes: First Phase 2 epoch. FP4 saved 2.3MB. Score will improve rapidly.

## 2026-04-12T00:25:50 — dilated_h64 on modal_a10g ep78

- proxy=1.423
- PoseNet: 0.06700000
- SegNet: 0.03400000
- Notes: Steady improvement from 1.476 start

## 2026-04-12T00:29:21 — dp_sims on modal_a10g ep100

- proxy=4.54
- PoseNet: 1.47200000
- SegNet: 0.00700000
- Rate: 2.262
- Notes: Only 1 Phase 2 epoch. Died at scorer start. SegNet excellent (0.007). Must resume P2 on Lightning.

## 2026-04-12T00:29:21 — dilated_h64 on modal_a10g ep84

- proxy=1.413
- PoseNet: 0.07200000
- SegNet: 0.00600000
- Rate: 0.046
- Notes: 84/2500 epochs. INT8 45KB saved. On track for sub-1.3.

## 2026-04-12T00:33:59 — dp_sims on modal_a10g ep109

- proxy=2.93
- PoseNet: 0.53200000
- SegNet: 0.00600000
- Rate: 2.262
- Notes: 9 Phase 2 epochs. SegNet 0.006 matches best! PoseNet needs training. Trajectory: sub-1.5 by ep 150.

## 2026-04-12T00:33:59 — dilated_h64 on modal_a10g ep99

- proxy=1.4
- PoseNet: 0.07000000
- SegNet: 0.00600000
- Rate: 0.046
- Notes: 99/2500 epochs. On track. Needs 800+ more.

## 2026-04-12T01:17:03 — dp_sims on modal_a10g ep189

- proxy=2.5
- PoseNet: 0.48200000
- SegNet: 0.00300000
- Rate: 2.262
- Notes: FINAL Modal run. SegNet 0.003 TIES Quantizr! PoseNet 480x gap + rate 5.7x gap = entire remaining problem.
