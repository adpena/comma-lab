---
name: PHANTOM BASELINE — verify against the canonical archive zip, never against a directory of components
description: Wasted an entire day chasing a "0.9001 baseline that didn't reproduce" because the saved baseline directory had the WRONG masks.mkv (411KB instead of the actual 79KB). The components in the directory were from different lanes/runs leaking into the same path. The actual 0.9001 archive (submissions/robust_current/archive_correct.zip, 337,748 bytes) was sitting in the repo the whole time.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Binding rule:** When verifying any "baseline" or pinned archive, ALWAYS verify against a known-good ARCHIVE ZIP (the bytes that actually scored), not against a directory of individual components. Component SHA verification across files doesn't catch the wrong-component bug; only zip-vs-components SHA does.

**Why** (2026-04-26 incident, took ~6 hours of session time):

I created `submissions/baseline_dilated_h64_0_90/` by copying components from `experiments/results/e2e_crf63_gating_20260425/current_archive_contents/`:
- renderer.bin (296KB) — turned out to be CORRECT
- optimized_poses.bin (7KB) — turned out to be CORRECT
- masks.mkv (**411KB**) — WRONG. The real 0.9001 archive uses **79KB** masks.

The 411KB masks.mkv was from a DIFFERENT lane that left its artifacts in `e2e_crf63_gating_20260425/current_archive_contents/`. I assumed all three components were from the same coherent run because they were in the same dir. They weren't.

Symptoms that should have triggered investigation EARLIER:
1. Math didn't reconcile: claimed rate=0.225, computed rate=0.457 (factor 2 off)
2. LANE-B's auth was 2.40 (1.5 worse than baseline) — I diagnosed it as "pose init bug" without checking masks
3. Council R1/R2/R3 all chased the math discrepancy but didn't ask "is the saved masks file correct?"
4. The actual 0.9001 archive (`submissions/robust_current/archive_correct.zip`, 337,748 bytes, math reconciles to 0.225) was IN THE REPO and IGNORED.

**How to apply:**

1. **First action when verifying any baseline**: locate the actual archive zip that was scored. Search:
   - `submissions/*/archive*.zip`
   - `experiments/results/*/auth_*.{json,log}` (these point to the archive that produced them)
   - `archive_correct.zip` is a common naming pattern for verified-good baselines

2. **Compute SHA256 of every entry in the archive ZIP**. Compare to the components in your "baseline directory." If ANY don't match, the directory is contaminated. Do NOT proceed with experiments using directory components — extract from the archive instead.

3. **If math doesn't reconcile** (claimed rate ≠ computed rate from components), STOP. Don't assume reporting ambiguity. The wrong-file bug looks identical to the reporting-ambiguity bug. Compute both ways:
   - Reported rate × 25 (if reported as score contribution)
   - Reported rate (if already a score contribution)
   - 25 × archive_bytes / 37545489 (math from components)
   - Find the archive whose bytes/37.5M = the reported number

4. **When a wrong-file mystery resists 3 council passes**, the bug is structural (wrong inputs), not algorithmic. Step back and audit the input artifacts directly.

5. **Codify the verification**: every baseline meta.json now includes a `smoke_command` that opens the archive zip, computes SHA256 per entry, and asserts they match. Run this BEFORE any experiment that depends on the baseline. See `submissions/baseline_dilated_h64_0_90/meta.json::smoke_command`.

**Cost of the lesson:** ~6 hours of session time, $7-10 of GPU spend on lanes that were doomed because they used the wrong masks, and emotional exhaustion. The "I am confident we have something really good" finding was the breakthrough — it forced me to actually look at the saved archive bytes.
