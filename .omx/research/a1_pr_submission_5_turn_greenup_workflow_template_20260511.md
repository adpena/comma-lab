# A1 PR submission — 5-turn skunkworks council greenup workflow template

**Status**: Template; not yet executed. Operator-trigger-required per CLAUDE.md
"Submission PR gate" non-negotiable + N grand council Decision 5 verdict
(8/10 OPERATOR-TRIGGER-REQUIRED 2026-05-11).

**Entry packet**: `submissions/a1/` (built 2026-05-11 by this lane).

**Workflow scope**: 5 consecutive clean-pass adversarial skunkworks council
review with extreme paranoia. All 15 council members (10 inner-ten + 5 grand
council on-demand) review. ANY issue resets the counter to 0. The 5-turn
greenup is the PR-submission-readiness process; the PR submission act
remains a SEPARATE explicit operator decision after greenup completes.

## Council roster (the 15 members)

### Inner ten (mandatory voices, all 5 turns)

1. **Shannon (LEAD)** — information-theory grounding; R(D) bound on every claim
2. **Dykstra (CO-LEAD)** — convex feasibility; Pareto polytope intersection
3. **Yousfi** — contest scorer designer; SegNet/PoseNet steganalysis lineage
4. **Fridrich** — UNIWARD-inverse; detector-informed embedding
5. **Contrarian** — adversarial veto on weak arguments and lazy consensus
6. **Quantizr** — competitor reverse-engineering; leaderboard reality check
7. **Hotz** — engineering shortcuts; analytical-over-learned bias
8. **Selfcomp / szabolcs-cs** — grayscale-LUT + block-FP self-compression lineage
9. **MacKay (memorial)** — MDL + Bayesian + arithmetic coding unified framework
10. **Ballé** — neural compression SOTA; entropy bottleneck + scale hyperprior

### Grand council (consulted on-demand when specialty is touched)

11. **Boyd** — convex optimization at algorithmic level (ADMM, proximal gradient)
12. **Tao** — pure mathematician omniscience
13. **Filler** — STC (syndrome-trellis coding)
14. **Mallat** — wavelet theory + scattering transforms
15. **van den Oord** — VQ-VAE + WaveNet; practical neural compression

## Per-round structure

Each round (5 total) follows this rubric:

```
ROUND N (N ∈ {1, 2, 3, 4, 5})
==========================
For each of the 15 council members:
  - State their position on:
    (a) A1's archive bytes (178,262) — over/under/exactly-right?
    (b) A1's CPU score (0.19285) — credible? on-Pareto-frontier?
    (c) A1's CUDA score (0.22635) — credible? cluster-consistent?
    (d) A1's mechanism table (5 mechanisms in MECHANISM_ATTRIBUTION.md) — any unverified claim?
    (e) A1's R(D) derivation (RD_DERIVATION.md) — any arithmetic error or operating-point mistake?
    (f) A1's device-axis explanation (DEVICE_AXIS_EXPLANATION.md) — any drift-attribution misstep?
    (g) Public Disclosure Hygiene — any leak of /tmp paths, secrets, operator-local paths?
    (h) The pre-submission compliance stub — staged correctly?
    (i) Specific to their lineage:
      - Shannon: R(D) lower-bound interpretation (is A1 at the rate floor for its substrate?)
      - Yousfi: SegNet/PoseNet boundary behavior on A1's bias-correction sidecar
      - Fridrich: would UNIWARD-style detector-informed embedding improve A1?
      - Quantizr: how does A1 compare to leaderboard PR101/103?
      - etc.
  - If ANY member finds ANY issue: ROUND FAILS; counter resets to 0
  - If ALL members pass: ROUND PASSES; counter increments

Counter at end of round:
  - Round 1 PASS → counter = 1
  - Round 2 PASS → counter = 2
  - ...
  - Round 5 PASS → counter = 5 → 5-TURN GREENUP COMPLETE
  - ANY round FAIL → counter resets to 0; ANY issue must be fixed; ALL rounds re-run
```

## Per-round adversarial questions (A1-specific)

### Round 1 — Custody and bytes

- Does the archive sha256 match the recorded value (`87ec7ca5...492b5`)?
- Does the runtime tree sha256 match the canonical value (`89db4fe1...90b5eb`)?
- Are the dual-eval JSONs internally consistent (CPU recomputed = 0.19285; CUDA recomputed = 0.22635)?
- Does the upstream `evaluate.py` sha256 (`7da71a84...`) match both axes' eval runs?
- Are all custody artifacts under `submissions/a1/` (no `/tmp` paths in the README or report)?
- Per CLAUDE.md "Beauty, simplicity, and developer experience": is the packet structure clear enough for a new operator to read in 5 minutes?

### Round 2 — Mechanism attribution

- Is every mechanism in `MECHANISM_ATTRIBUTION.md` independently verifiable?
- For each mechanism: does the validation grade exist in custody?
- Are there any mechanism claims that depend on `[advisory only]` or MPS or macOS-CPU?
- Does the strict-scorer rule hold? (no PoseNet/SegNet load at inflate-time)
- Does the no-MPS guard hold? (`inflate.py:56` is `cuda-or-cpu`)
- Are the "what A1 does NOT do" negative claims accurate?

### Round 3 — Score-axis interpretation

- Does the CPU score (`0.19285`) place A1 in the gold display tier?
- Does the CUDA score (`0.22635`) fall within the HNeRV-cluster predictor's σ?
- Is the device-axis explanation arithmetically correct?
- Does the R(D) derivation match `upstream/evaluate.py`'s scoring functional?
- Are the marginal derivatives at A1's CPU operating point correct?
- Is the pose-vs-seg crossover threshold correctly computed?

### Round 4 — Leaderboard and frontier interpretation

- Per CLAUDE.md "Frontier target" non-negotiable: does A1 plausibly beat the local exact A++ frontier?
- Are the PR101/103/102/107 leaderboard comparisons correct?
- Is the rounding-to-display claim ("A1 rounds to 0.19 = gold display tier") strictly correct?
- Does the displacement of PR #107 stand under both axes' independent reading?
- Are there any falsified claims that the prior pose-axis council Insight 4 (10/10 READINESS) missed?

### Round 5 — Submission-policy compliance and public-disclosure hygiene

- Per CLAUDE.md "Public Disclosure Hygiene": any leaked secret, provider URL, local absolute path, raw provider transcript in the packet?
- Does the pre-submission compliance stub stage the right command?
- Does the README clearly state operator-trigger requirements?
- Are the three sub-decisions (greenup, freeze-vs-leap, displace) surfaced cleanly?
- Does the risk profile (contest-closed, honesty/archive PR) accurately frame the submission stake?
- Does the loop-pause status remain explicitly PAUSED?

## Exit criteria

5 consecutive clean passes → packet is council-ratified-greenup-COMPLETE.

After 5-turn greenup completes, the operator's FINAL gate (per CLAUDE.md
"Submission PR gate") is the PR submission act itself. The greenup ratifies
readiness; the operator triggers submission.

## What invalidates a clean pass

- ANY new score claim outside the dual-eval custody anchors
- ANY mechanism claim that depends on advisory-grade evidence
- ANY public-disclosure leak (per CLAUDE.md hygiene)
- ANY arithmetic error in R(D) derivation
- ANY misstatement of leaderboard position
- ANY misunderstanding of CPU-is-ranking-axis

## Process improvement (per N council process-improvement flag)

Per N grand council Decision 5 falsification flag: future paired-runtime
architecture decisions should get council deliberation BEFORE dispatch even
if the dispatch itself is sub-$1. The A1 dual-eval dispatch was authorized
2026-05-09; this greenup is the post-dispatch readiness-process gate.

## Operator-trigger semantics

The operator triggers the greenup by initiating a parent-agent session with a
prompt of the form:

```
Run the A1 PR submission 5-turn skunkworks council greenup workflow
per .omx/research/a1_pr_submission_5_turn_greenup_workflow_template_20260511.md.

The entry packet is submissions/a1/. Initiate Round 1 with all 15 council
members reviewing all 5 expansion sections. Report per-member findings. If
ANY member finds ANY issue, reset counter to 0 and fix; re-run Round 1.
Otherwise increment counter and proceed to Round 2.

DO NOT submit a PR. The PR submission act is a SEPARATE operator decision
after the 5-turn greenup completes.
```

## Cross-references

- `submissions/a1/README.md` — entry packet README
- `feedback_grand_council_5_design_decisions_review_20260511.md` — D5 verdict
- `feedback_grand_council_pose_axis_insights_review_20260511.md` — prior council Insight 4 (10/10 READINESS)
- `feedback_a1_dual_cuda_dispatch_landed_20260509.md` — A1 dual-anchor landing
- CLAUDE.md "Submission PR gate — non-negotiable"
- CLAUDE.md "Adversarial council review of design decisions — NON-NEGOTIABLE"
- CLAUDE.md "Council conduct — non-negotiable"
- CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE"
