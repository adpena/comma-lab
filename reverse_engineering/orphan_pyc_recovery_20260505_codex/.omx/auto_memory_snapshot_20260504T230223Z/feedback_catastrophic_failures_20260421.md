---
name: CATASTROPHIC FAILURES — 4 measurement bugs invalidated ALL prior scores
description: Mask resolution (48x64 vs 384x512), wrong archive, overlapping pairs, eval_roundtrip=False. Score went from "0.87" to 103.27 to 2.01 when all bugs fixed. INEXCUSABLE.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
On 2026-04-21, we discovered that EVERY score reported across multiple sessions was
wrong due to 4 compounding measurement bugs:

1. **masks.mkv at 48x64** — renderer trained on 384x512, fed 48x64 masks, output
   upscaled 18x. PoseNet 94.63 (catastrophic) vs 0.013 with correct masks.
   Score: 103.27 → 2.01 from this single fix.

2. **Wrong archive for rate** — auth evals used 119-180KB renderer-only archive
   instead of full submission archive. Rate off by 0.108 points.

3. **1199 overlapping pairs** — auth_eval.py used range(N-1) instead of upstream's
   non-overlapping range(0, N-1, 2). All eval_checkpoint() scores wrong.

4. **eval_roundtrip=False** — all TTO runs optimized against wrong proxy.
   noise_std=0 meant the Hotz STE fix was dead code.

**Why:** Every component quietly produced wrong output with no downstream validation.
Warnings instead of errors. Assumptions instead of checks. Component-level testing
instead of full e2e pipeline testing.

**How to apply:**
- NEVER trust a score that hasn't been produced by the FULL pipeline:
  inflate_renderer.py → upstream evaluate.py
- NEVER skip the e2e test after ANY change to masks, archive, renderer, or scoring
- EVERY auth eval must start with archive validation via submission_archive.py
- EVERY new scoring code must be diffed against upstream evaluate.py line by line
- Warnings are bugs. Silent fallbacks are bugs. Only hard errors prevent measurement disasters.
