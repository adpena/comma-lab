---
name: Competitive landscape as of 2026-04-24 — szabolcs-cs at 0.36, Quantizr done
description: New competitor PR#56 at 0.36. Quantizr says done. He says sub-0.30 via conv dim sweep. Yousfi noted boundary artifacts. Self-compression at 1 bit/weight is new technique.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Leaderboard (2026-04-24)

1. Quantizr — 0.33 (PR#55, says he's done)
2. szabolcs-cs — 0.36 (PR#56, NOT YET CI-evaluated)
3. mask2mask — 0.60 (PR#53, Quantizr's earlier entry)
4. neural_inflate — 1.89 (PR#49)

## Key Intelligence

**Quantizr (done):**
- "sub 0.30 is possible just by sweeping through different conv dims"
- "further improvement could possibly be made with some neural mask encoding"
- "this is as much as I'm gonna work on this"
- eval_roundtrip: he shared it as "a helpful hint to others"

**szabolcs-cs (PR#56, new #2):**
- Self-compression at ~1.017 bits/weight (NOT FP4, novel)
- Better PoseNet than Quantizr (0.000397 vs 0.000513)
- Worse SegNet (0.001153 vs 0.000613) — "underfit, no architecture search"
- Smaller archive (279KB vs 300KB)
- "Same trick as Quantizr" for SegNet (independent discovery)

**Yousfi (contest organizer):**
- Noted boundary artifacts from conv padding on Quantizr's submission
- Teased "Leave some time for the v2, v3"
- Clarified in PR#54: scorer weights must be in archive

**How to apply:**
- Conv dim sweep is the path to sub-0.30 (Quantizr said so himself)
- Our wilde profile (base=16, mid=24) is one point in that sweep
- Consider wider sweep: base in [12,16,20,24], mid in [20,24,28,32]
- Boundary artifacts: consider reflection padding instead of zero padding
- Self-compression (1 bit/weight) is worth researching
- Quantizr is done — we have time to iterate
