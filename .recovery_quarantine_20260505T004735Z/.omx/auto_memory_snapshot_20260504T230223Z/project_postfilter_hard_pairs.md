---
name: Hard-pair postfilter is the biggest lever — 0.874 point net benefit
description: 46KB postfilter trained on hard pairs reduces PoseNet 1.114→0.210. Net +0.874 after 0.030 rate cost. Uses existing curriculum training (Phase 3 hard_frame_ratio). Difficulty.pt computed.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Hard-Pair Postfilter Strategy (2026-04-23)

Difficulty map shows top 20% pairs have pose_d 227x worse than bottom 80%.
A postfilter focused on these pairs is the single biggest lever.

**Curriculum training (Approach 4):**
1. Phase 1: all pairs, learn general corrections (2000 epochs)
2. Phase 2: top 30% hardest, specialize (2000 epochs) 
3. Phase 3: top 10% hardest, extreme focus (1000 epochs)

**Archive: 46KB (int8). Rate cost: 0.030. PoseNet savings: 0.904. Net: +0.874.**

**Pipeline:** renderer(masks, pose=TTO) → postfilter → .raw output

**Infrastructure:** Trainer.fit() with proven_baseline profile + hard_frame_ratio.
difficulty.pt already computed and saved.

**How to apply:** After float training completes:
1. Export ASYM renderer
2. Generate all 600 pairs' output with CRF50 masks + matched poses
3. Train postfilter using `python -m tac lossy --profile proven_baseline`
4. Export int8 (46KB)
5. Wire into inflate_renderer.py (load postfilter, apply after renderer)
