---
name: Quantizr Analysis — 0.60 with 386KB archive
description: Quantizr bundles neural renderer in 386KB archive. Says <0.50 easily possible. Weights inside zip per rules.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Quantizr PR#53 — Score 0.60

- Archive: 386,192 bytes (386KB) — video + neural renderer weights ALL inside
- PoseNet: 0.00066 (exceptional)
- SegNet: 0.00264
- Rate: 0.01029 (25× = 0.257)
- GPU required (T4), also works CPU <30min
- Architecture "minimally obfuscated"
- Quote: "I think <0.50 is pretty easily possible as a slightly different architecture gets a score at least 10% better"

## Implications for Us

- Weights MUST be inside archive.zip (Quantizr does this)
- Our real baseline: ~1.36 (with 46KB postfilter in 910KB archive)
- Quantizr's 386KB archive is SMALLER than ours despite including neural weights
- Their model is more efficient: better quality at lower rate
- GPU lane is the path to competing: our constrained gen approach could beat 0.60 if it works

**Why:** Confirms the rules, sets the target, shows what's possible.
