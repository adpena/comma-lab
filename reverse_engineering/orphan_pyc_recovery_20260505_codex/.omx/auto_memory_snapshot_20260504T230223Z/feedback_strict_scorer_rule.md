---
name: Strict Scorer Rule — Canonical Answer
description: CANONICAL DECISION — scorer weights at inflate time trigger archive inclusion rule. No inflate-time TTO with scorers. Period.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The user has made a binding decision: adopt the STRICT reading of the scorer weights rule.

**Rule:** "External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. This applies to the PoseNet and SegNet."

**Canonical interpretation:** If our inflate script loads PoseNet or SegNet for ANY purpose (TTO optimization, mask extraction, embedding computation, gradient descent), those weights must be in archive.zip. Since including them (~73MB) destroys the rate term, this means:

- **NO TTO at inflate time** — TTO requires loading scorers for gradient computation
- **NO mask extraction at inflate time** — requires loading SegNet (already fixed via mask pre-extraction)
- **NO scorer-based optimization at inflate time** — period

**Why:** The user personally agrees with the strict reading. Risk/reward is asymmetric: DQ on final judging loses everything. tensor_inversion precedent (scored but flagged) is not reliable.

**How to apply:**
- All inflate-time code must work WITHOUT loading PoseNet or SegNet
- TTO is a COMPRESS-TIME tool only (unlimited compute, results used as training data)
- The adaptive TTO feature (INFLATE_TTO) must be labeled "non-compliant" and off by default
- Never claim a contest-compliant score that depends on inflate-time scorer access
