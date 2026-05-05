---
name: MEASUREMENT BEFORE CELEBRATION — Verify Archive Size and Auth Pipeline
description: Critical failure — celebrated auth 0.36 that was actually ~0.41 due to wrong archive in eval. Never celebrate a score without verifying the measurement apparatus.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
We celebrated auth 0.36 (claimed to be 0.03 from Quantizr) but the actual score
was ~0.40-0.41 because every auth eval used a renderer-only archive (119KB) instead
of the full submission archive (183KB+). The rate component was wrong by 0.04-0.05.

**Root cause:** The Modal auth eval creates an archive from renderer.bin ONLY. The
upstream evaluate.py reads archive.zip size for rate calculation. Our auth evals used
a smaller archive than the actual submission would have.

**The lesson:**
1. NEVER celebrate a score without verifying the entire measurement pipeline
2. The auth eval must use the EXACT archive that will be submitted
3. Before ANY auth eval: verify archive.zip contains ALL submission artifacts
4. Before reporting ANY score: print the archive size and verify it matches expected
5. Auto-auth-eval in training must use the REAL archive, not a temporary one
6. Every auth eval report must include the archive size used

**What this missed:**
- Rate was 0.122 (183KB archive) but eval used 0.079 (119KB archive)
- Every "auth" score in this session was optimistic by 0.04-0.05
- The proxy-auth drift also went undetected for 2560 epochs
- Both failures are MEASUREMENT failures, not code failures
