---
name: SegNet vs PoseNet Priority — CORRECTED After Wrong Checkpoint Discovery
description: Step curve was run on wrong checkpoint. Qualitative findings (phase transition) likely hold but absolute ratios need re-validation.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## CORRECTION (2026-04-15 late session)

The "SegNet 77:1 paradigm shift" was measured on the WRONG checkpoint — a 5-epoch smoke-test
model, not the auth=0.87 renderer. The absolute numbers (PoseNet=165, SegNet=0.50) are GARBAGE.

### What was WRONG:
- Step curve ran against experiments/results/fridrich_renderer/renderer_best.pt (5-epoch smoke model)
- Pair difficulty map also used wrong model
- All Vast.ai experiments used same wrong path (hardcoded in client.py)
- The "77:1 leverage ratio" was computed from garbage data

### What is CORRECT (re-run with right checkpoint):
- Pair difficulty map (correct): PoseNet mean=0.01726, SegNet mean=0.00215
- These match auth eval (PoseNet=0.031, SegNet=0.00217)
- At the AUTH level: SegNet contributes 0.217 (25%), PoseNet contributes 0.557 (64%)
- **PoseNet still dominates the auth score**, not SegNet

### What STILL NEEDS re-validation:
- Step curve (phase transition at 100 steps) — qualitatively likely correct, needs re-run
- Cosine vs constant LR comparison — same qualitative finding likely holds
- The "SegNet is frozen at 0.50" finding was from wrong model — need to see what happens with right model

### Lesson:
ALWAYS verify checkpoint identity before running experiments. A 2-pair sanity check
(PoseNet should be < 1.0 for trained model) would catch wrong-checkpoint errors in <1s.
