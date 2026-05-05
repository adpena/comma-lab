---
name: Score Uncertainty — 1.33 Cannot Be Reproduced Locally
description: Local eval consistently gives ~1.96 with correct checkpoint. 1.33 cannot be reproduced. Need auth PR for real score.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Finding (2026-04-12, confirmed after extensive investigation)

The 1.33 score CANNOT be reproduced locally despite matching:
- Same checkpoint (md5 62e4b2dc, dilated_h64)
- Same archive (0.mkv from Apr 9)
- Same venv (upstream, torch 2.10.0, PyAV 17.0.0)
- Same macOS (26.4, same as Apr 10)
- Same evaluate.py (--device cpu)
- Same batch_size, seed, etc.

Local eval consistently gives ~1.96 (PoseNet 0.057 vs the 1.33's PoseNet 0.002).

### Theories investigated and eliminated:
1. Wrong checkpoint — ELIMINATED (correct md5, score changed by only 0.03)
2. DALI vs PyAV — ELIMINATED (both use PyAV on --device cpu)
3. Torch version — ELIMINATED (both venvs produce identical PoseNet outputs)
4. macOS update — ELIMINATED (same version)

### Remaining theories:
1. The 1.33 eval used stale pre-inflated frames from a previous run
2. The 1.33 report was copied from proxy training output
3. Something in the eval environment was different that we can't identify

### True baseline: UNKNOWN
Our local score is ~1.96. The real auth score requires a PR submission.
Compliant score (with postfilter bundled): ~1.98.

**Action:** Submit a PR to get the real auth score before any more optimization.
