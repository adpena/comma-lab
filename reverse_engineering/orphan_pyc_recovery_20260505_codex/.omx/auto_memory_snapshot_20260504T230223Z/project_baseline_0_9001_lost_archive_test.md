---
name: 0.9001 baseline LOST — committed "archive_correct" is actually the 53.61 (48x64-mask) one
description: Verified contest_auth_eval on Vast.ai 4090 (2026-04-27) returned 53.60 on submissions/baseline_dilated_h64_0_90/archive_baseline_0_9001.zip. This matches historical "48x64 masks upsampled" entry of 53.61 exactly. The actual 0.9001 archive uses full-res 384x512 masks and is NOT in the repo.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27 on Vast.ai RTX 4090, instance 35665497, $0.16 total cost.**

The eval tool worked correctly — `experiments/contest_auth_eval.py` shipped a clean RESULT_JSON, score formula recomputation matched the report (28.31 + 25.06 + 0.23 = 53.59), all R3-R5 council validations passed without raising. The bug is in the SAVED ARCHIVE, not the eval tool.

**Result on archive_baseline_0_9001.zip:**
- Final score: **53.60**
- PoseNet dist: 62.78  (5,800x worse than expected 0.0107)
- SegNet dist: 0.283  (118x worse than expected 0.0024)
- Rate (unscaled): 0.00921 (matches expected)
- Archive: 345,802 bytes (matches expected ~338KB)

**Cross-reference to historical record:**
`project_true_score_20260421.md` entry: "48x64 masks (old, upsampled): seg=28.3, pose=25.0, rate=0.23, total=53.61".

Our 53.60 reproduces that historical 53.61 entry to within 0.01. The eval pipeline is REPRODUCIBLY CORRECT. The masks.mkv inside our committed "0.9001 baseline" is 48x64 resolution (1/8 of native 384x512), NOT full-resolution.

**What this means:**
- The "0.9001 baseline" archive I committed in `submissions/baseline_dilated_h64_0_90/` is mislabeled. It's the 53.6-scoring archive.
- The real 0.9001 archive (which used full-res 384x512 masks per the run_log entry "Pinned dilated h64 + CRF=50 + matched poses") was never saved into the repo and the original Vast.ai instance is destroyed.
- The "0.9001" run_log entry is now likely uncorroborated by reproducible artifacts.

**How to apply:**
1. NEVER claim a baseline score from run_log alone — verify by re-running auth eval on the archive that produced it. If the archive is missing, the score is unreproducible and should not be quoted as a baseline.
2. To recover a real "0.9001 baseline," need to either:
   (a) Find the original full-res 384x512 masks.mkv that pairs with this renderer.bin + poses, OR
   (b) Re-encode masks at 384x512 from the GT video using SegNet, build a fresh archive, and treat the new measured score (whatever it is) as the new verified baseline.
3. Stop trusting historical run_log scores until each is matched to a saved + reproducible archive.
4. **Eval tool itself is verified.** `experiments/contest_auth_eval.py` reproduced a HISTORICAL number (53.61) from saved artifacts to within 0.01. That's strong evidence the tool is contest-faithful.

**Cost lesson:** $0.16 of CUDA spend produced a definitive answer ("the eval tool works; the archive is wrong"). Cheap. Future verification runs should be the FIRST step on any "is this baseline real" question, not the last.
