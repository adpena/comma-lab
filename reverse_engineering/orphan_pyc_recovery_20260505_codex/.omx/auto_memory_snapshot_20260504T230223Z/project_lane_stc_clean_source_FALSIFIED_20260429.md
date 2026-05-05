---
name: Lane STC clean-source — ALIVE WITH REDESIGN per max-rigor codex review (45% endorsement)
description: 2026-04-29 PM. After 22-voice extreme-rigor codex review, FALSIFICATION is WITHDRAWN as a scientific claim. Current implementation correctly killed as a tonight-launch (council unanimously rejects MPS authority). Codec has a real bug — "one-majority-plus-exceptions" stores 109M exceptions vs 11.8M boundaries on multi-region masks. Even at boundary_fraction→0.001 the exception stream stays huge. Floor: deflated STCB at 0.259 bpp = 7.6MB, 18× worse than AV1's 0.014 bpp; structural redesign required (AV1+STC residual / temporal predictor / scanline RLE / lossy STC).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## STATUS REVISION 2026-04-29 PM (post-user-correction)

The earlier FALSIFICATION verdict in this file was based on MPS-derived SegNet output. CLAUDE.md non-negotiable explicitly forbids using MPS for strategic measurement: "Score numbers measured on MPS may NOT be reported as 'auth' or 'contest-compliant' anywhere." MPS-SegNet drifts 2× from CUDA-SegNet on segmentation tasks; PoseNet drifts 23×.

STC encodes argmax pixel labels — MPS-argmax bytes ≠ CUDA-argmax bytes. The 21MB measurement applies to MPS-argmax masks; the contest scorer never sees those.

**Required action:** re-run clean-source STC on Modal T4 CUDA (~$0.20, ~10 min) before any kill/keep decision. Until then, clean-source STC is UNDETERMINED, not falsified. Council's #1 hope is back on the table.

## Original empirical result (MPS-PROXY, NOT VALID for strategy)

| Metric | Value |
|---|---|
| Source | upstream/videos/0.mkv → SegNet argmax (CLEAN, no AV1 roundtrip) |
| Frames encoded | 1200 |
| Resolution | 384 × 512 |
| boundary_fraction | 0.05 |
| **Clean-source STC bytes** | **21,270,086** (~21MB) |
| Lane A masks.mkv (AV1) | 421,483 (~421KB) |
| **Regression** | **50.5×** |
| SegNet device | MPS (advisory only — CUDA would produce slightly different bytes but the scale-of-magnitude conclusion holds) |
| STC encode wall-clock | 459.9s (CPU per-frame loop) |

## Why clean-source STC failed

The earlier hypothesis (project_lane_stc_av1_regression_finding_20260429) was that AV1 quantization noise saturated the boundary detector and forced 100% pixel coding. **That was partially right but missed the bigger issue**:

1. Clean-source DID reduce STC bytes from ~47.6MB (AV1 roundtripped) to ~21MB. So removing AV1 noise helps by ~2×.
2. But the floor of 21MB is STILL 50× larger than AV1's 421KB output. Removing AV1 noise didn't fix the structural gap.

The structural gap:
- STC at boundary_fraction=0.05 codes ~5% of pixels = 11.8M pixels for 1200 frames
- Per-pixel cost (position + class ID via arithmetic coder) ≈ 1.8 bytes ≈ 14 bits
- Total: 11.8M × 1.8B ≈ 21MB ✓ matches measurement
- AV1 monochrome achieves ~0.014 bits/pixel via interframe prediction + 2D context
- Total AV1: 236M pixels × 0.014 = 413KB ≈ matches 421KB measurement

To match AV1, STC would need `boundary_fraction ≈ 0.0036` (only the 0.36% sharpest boundaries) — at that fraction the residual stream blows up because most class transitions become "unrepresented" in the boundary set.

## What this means for the portfolio

- **Clean-source STC = DEAD** as a tonight-launchable lane
- **Council's "Aggressive Rate Stack" -0.04 to -0.08 from STC = INVALIDATED**
- Sub-0.30 probability shifts back DOWN — the portfolio loses the 0.060 central STC contribution
- Revised marginal: roughly **24-30% sub-0.30 central** (down from 34% with STC, back near grand council's original 24%)

## What remains viable

- **SC++ v4 / q_faithful_v3 base** still the highest-EV path
- **Ω-W water-filling export** still viable (weight pool, NOT mask pool — independent from STC)
- **LCT** still bolt-on (10 bytes, -0.005 to -0.015)
- **Restricted DARTS-S** still viable arch base
- **Carmack** confirmed killed (deterministic ZIP hygiene only)

## What about clean-source STC for the PAPER?

- Document this as a negative result in the writeup — STC boundary coding is information-theoretically inferior to AV1 monochrome for this class of dense semantic mask
- Could be revived for FUTURE pipelines with: VERY sparse masks (boundary_fraction < 0.005), wavelet-domain coding, or learned arithmetic priors
- Filler/Mallat/Ballé voices flagged this risk; their priors were correct

## How to apply

- DO NOT dispatch clean-source STC to Modal — it's a confirmed regression
- Refund the predicted -0.040 to -0.053 wedge to the floor estimate
- Update grand council scoring: 5-lane plan becomes 4-lane (drop STC #3, promote restricted DARTS-S to #3)
- Adversarial review Round 1 found 1 CRITICAL + 4 Mediums (auth_eval.py whitelist missing .stcb, GT decode color space mismatch, no encode/decode parity verify, mask resolver tie-break, no failure manifest) — all moot now since lane is dead

## Cross-refs

- experiments/results/lane_stc_clean_local/build.log (the measurement)
- /tmp/codex_runs/clean_source_stc_review_round1.log (Round 1 adversarial review — RESETS-COUNTER)
- /tmp/codex_runs/grand_council_portfolio_odds.log (the 34% portfolio estimate that included STC)
- project_lane_stc_av1_regression_finding_20260429.md (the partial earlier finding)
- project_grand_council_final_designs_20260429.md (the original 24% estimate)
