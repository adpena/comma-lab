---
name: Encoder Sweep Results (2026-04-10)
description: Tested 6 encoder variants — all within ~2% of current 877KB. Encoder is near-optimal. Gains are in post-filter.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Tested against current: SVT-AV1 latest, CRF 34, preset 0, fg=22, keyint=180, 8-bit, 524x394.

| Variant | Size | vs Current |
|---------|------|-----------|
| Current | 876,737 | baseline |
| keyint=-1 (infinite GOP) | 867,736 | -1.0% |
| 10-bit yuv420p10le | 868,893 | -0.9% |
| fg=30 + denoise=0 | 897,765 | +2.4% |
| 582x436 (50% downscale) | 1,028,783 | +17.3% |
| Full stack (10b+fg30+d0+k-1) | 861,566 | -1.7% |

**Conclusion:** Encoder is near-optimal. Maximum combined savings: 1.7% (0.01 score).
Not worth retraining the post-filter for these gains.

**Future extreme optimization:** If pushing for absolute maximum:
- Try SVT-AV1 v2.3.0 (older version, PR#51 claims it outperforms latest on this content)
- Try building SVT-AV1 from source with specific tuning flags
- Try fractional CRF (33.5) — small CRF changes can shift rate/distortion tradeoff
- Try 10-bit + keyint=-1 combined + retrain post-filter on new distribution
- The distortion (not just file size) should be measured — encode may help SegNet even if size is similar

**Inspired by:** PR#51 (ArmanJR) who scored 1.94 with encoder-only approach using
SVT-AV1 v2.3.0, 10-bit, infinite GOP, fg=30, denoise=0.

**Why:** Encoder gains are <2% on rate (0.01 score). Post-filter gains are 10-30%
on distortion (0.1-0.3 score). Resource allocation should favor training.
