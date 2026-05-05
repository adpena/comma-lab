---
name: Regenerate Don't Download — Always Use Latest Code
description: Instead of downloading old experiment results, regenerate with current (improved) code. Every regeneration is an opportunity to improve.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
When experiment results are lost or need updating, REGENERATE with the latest code rather than downloading old results from Modal/previous runs.

**Why:** Each code improvement (hinge loss, two-phase TTO, simulate_resize fix, etc.) makes regenerated results BETTER than the originals. Downloading old v5a/v5b frames from Modal would use the OLD code without hinge loss, without two-phase TTO, without the simulate_resize fix. Regeneration gives us strictly better data.

**How to apply:**
- Don't waste time downloading old TTO frames from Modal
- Instead, spin up a Vast.ai 4090 and regenerate with ALL improvements applied
- The new TTO recipe (hinge loss + two-phase + embedding loss + constant LR + 100-150 steps) should produce BETTER frames than the old 500-step runs
- Every regeneration is an experiment — compare to old results to validate improvements
- Record both old and new results for the paper's ablation study
- Exception: if you need the EXACT old frames for comparison/debugging, download those specifically

**Cost consideration:** A full 1200-frame TTO on 4090 takes ~10 min at 150 steps ($0.04). Regeneration is nearly free compared to the DX cost of managing old artifacts.
