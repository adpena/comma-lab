---
name: Dilated Architecture Breakthrough
description: Dilated h=64 scored 1.33 authoritative — 5.6x PoseNet improvement over standard, initially appeared to fail at h=32
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Dilated convolutions initially appeared to fail at h=32 (improved local score but didn't transfer to int8).
At h=64 with 905 epochs on Modal A10G, they scored **1.33 authoritative** (down from 1.51 standard).

The key: PoseNet improved 5.6x (0.01229 -> 0.00218). The expanded receptive field
captures spatial correlations PoseNet's early convolutions attend to.

**Why:** Architecture changes need sufficient capacity and training time to overcome
quantization sensitivity. This is the most important finding since the hardening round.

**How to apply:**
- Dilated is now the default architecture for all training lanes
- The standard architecture should be considered superseded for new experiments
- h=96 dilated (currently training on Modal) could push even lower
- SegNet (0.00610) is now the dominant score component — next improvement must come from there
