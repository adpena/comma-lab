---
name: Competitor Alert — mask2mask PR#53 claims 0.60
description: Quantizr submitted mask2mask scoring 0.60 on Apr 11. If verified, we drop to #2. Neural model, not codec tuning.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## PR#53: mask2mask by Quantizr (2026-04-11)
- Claimed score: **0.60** (vs our 1.33)
- SegNet: 0.00264 (54% better than baseline)
- PoseNet: 0.00066 (3.3x better than our auth 0.00218)
- Rate: 0.01029 (archive 386KB vs our 903KB)
- Requires GPU for inflation
- Author says: "not a ffmpeg settings tweak", "<0.50 easily possible"
- Architecture deliberately obfuscated

## Status: OPEN PR (not yet eval-verified by organizers)
The contest uses automated eval via GitHub Actions. Until the eval workflow
runs and confirms, this is self-reported. But the numbers look plausible.

## Implications
- If verified: we are #2, trailing by 0.73 points
- The approach is likely a full neural codec, not a post-filter
- 386KB archive suggests learned video representation + decoder model
- GPU-required inflation = heavier model than our 45KB CPU filter
- The author claims even better is possible

## Our response options
1. Accept #2 and focus on best writeup ($1000 prize)
2. Investigate neural codec approaches (massive pivot, limited time)
3. Double down on our approach and aim for best score within post-filter paradigm
4. Analyze the submission once merged to learn from it

**Why:** Existential competitive threat. Changes prize strategy.
**How to apply:** Convene emergency council. Reassess whether to optimize score or writeup.
