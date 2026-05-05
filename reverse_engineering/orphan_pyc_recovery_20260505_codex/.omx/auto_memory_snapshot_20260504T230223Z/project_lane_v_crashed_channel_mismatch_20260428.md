---
name: Lane V CRASHED — channel mismatch (1-ch input vs 3-ch weight)
description: 2026-04-28 Lane V (Quantizr replica 88K + half-frame from epoch 0) crashed at ~7.6h elapsed with a torch RuntimeError: input[1, 1, 384, 512] expected 3 channels but got 1. NEVER reached auth eval. ~$1.90 wasted on 7.6h × $0.25/hr. Bug is in Lane V's training data pipeline — likely half-frame warp returning 1-channel masks instead of 3.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The crash

```
File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(input, self.weight, self.bias)
RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[1, 1, 384, 512] to have 3 channels, but got 1 channels instead
```

Conv layer expected 3-channel input (RGB-like) but got 1-channel (monochrome mask). This is a Lane V architecture or pipeline bug.

## What Lane V was supposed to do

- **Architecture**: Quantizr replica 88K params (small DSConv U-Net)
- **Training**: half-frame masks JOINTLY from epoch 0 (vs Lane D's bolt-on retrofit)
- **Schedule**: 600ep P1 + 1500ep P2 + 400ep P3 + 400ep P4 + 100ep P5 = 3000 epochs
- **Predicted wall clock**: ~12h on 4090 ($3.00 at $0.25/hr)
- **Predicted band**: [0.50, 1.10] — biggest swing in the portfolio per skunkworks

## What was wasted

- 7.6h × $0.25/hr = ~$1.90 burned
- 3 codex review rounds + green pass before deploy → all happy with the code
- Lane V was THE moonshot lane for crossing into Quantizr's territory

## Root cause hypothesis (need session 2 investigation)

Likely candidates for the channel-mismatch:
1. **Half-frame warp expand**: when an even-frame mask is "warped" from its odd-frame neighbor, the warp output might be 1-channel (just the mask) instead of being broadcast to RGB-equivalent 3 channels
2. **Profile mismatch**: `quantizr_replica_88k_halfframe` profile might have wrong input_channels for the entry conv (declared 3 but actually feeding mask as 1)
3. **Data pipeline bug**: AsymmetricPairGenerator / data loader might return mask without channel duplication

## Where to look (next session)

- `src/tac/profiles.py` — quantizr_replica_88k_halfframe profile, check input_channels declaration
- `src/tac/renderer.py` — entry conv expects (B, 3, H, W); check if mask gets repeated to 3-channel
- `src/tac/data.py` half-frame pair generator — does it return 1-channel or 3-channel mask?
- `scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh` — flag wiring

## Composability impact

Lane V was the architectural anchor for Stack C (Aggressive moonshot, predicted [0.40, 0.70]):
- Lane V renderer + Lane W per-pair SC + Lane LM-V2 zero-cost poses

Without Lane V working, Stack C is dead. Need a Lane V V2 that fixes the channel issue.

## Cross-references
- `project_quantizr_full_intel_20260421` — Lane V's source (Quantizr 88K + half-frame + KL distill)
- `project_lane_g_v3_stacking_skunkworks_20260428` — Stack C composition that needed Lane V
- `feedback_half_frame_breaks_posenet` — earlier failed half-frame attempt (Lane D)
- `feedback_per_instance_verify_pattern_20260428` — verify script CAUGHT this crash via heartbeat staleness + traceback grep
- `project_lane_g_v3_landed_1_05_20260428` — current frontier (Lane G v3, dilated-h64 baseline)
