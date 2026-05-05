---
name: Python library idea for task-aware codec
description: User suggested generalizing our techniques into a pip-installable Python library for task-aware video compression post-filters
type: project
---

The user suggested (2026-04-09) that since comma.ai would likely use these techniques in production, we should package our approach as a reusable Python library.

**Proposed name**: `task-aware-codec` or `scorer-guided-postfilter`

**Core technique**: train a tiny CNN post-filter against a frozen scorer via backprop, using:
- Saliency-weighted loss (α=20 PoseNet gradient saliency)
- QAT + EMA weight averaging (decay=0.997)
- Best-checkpoint selection with int8-quantized evaluation
- Width scaling (h=8 → h=96) with LeCun's log-linear curve
- Dilated convolutions for RF matching (fastvit-matched 15×15)

**Generalizable to any task**: replace PoseNet/SegNet with any frozen perception network. The library learns corrections specific to what the scorer cares about.

**Key abstractions**: Scorer, PostFilter, Trainer (QAT+EMA loop), Evaluator (faithful proxy), Quantizer (int8 save/load)

**Why this matters**: task-aware video codecs are an active research area. Our approach of backpropagating through a frozen scorer is novel and empirically validated (2.01 → 1.727 in one session, 0.22 below leaderboard #1).

**Timeline**: after competition submission. Could be a strong open-source release alongside the PR.
