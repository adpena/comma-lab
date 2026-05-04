---
name: Test Before Commit — Stop Shipping Bugs
description: Every agent implementation has had bugs. Review-after is not enough. Test end-to-end before committing.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
CRITICAL LESSON (2026-04-12): Every implementation shipped by agents had bugs found by subsequent reviews. The review passes themselves introduced new bugs. "Theoretically sound" exploits failed experimentally (AllNorm invariance, BT.601 fix).

**Pattern:** Build fast → review after → find bugs → fix → introduce new bugs → review again → find more bugs. This cycle doesn't converge.

**What went wrong:**
- preprocess_input bypassed 4 times across 3 review passes
- AllNorm invariance assumed without reading source code → 2.15 regression
- Undefined variables in "fixed" code (Armijo lambda_seg)
- 179 modules built in one session, each with latent bugs
- DX pipeline never tested end-to-end before deployment

**How to apply:**
- Run smoke tests BEFORE committing, not after
- Test the full pipeline end-to-end before declaring anything "done"
- Read the actual source code (scorer modules.py) before making assumptions
- Fewer modules built carefully > many modules built quickly
- Every "theoretically free" exploit must be A/B tested before enabling
- The Contrarian's question comes FIRST, not last
