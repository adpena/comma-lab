---
name: Grand Council 22-voice FINAL designs — floor 0.270, sub-0.30 prob 24%, kill list + 5 stacks
description: 2026-04-29 PM. 22-voice grand council post-STC-AV1-regression deliberation. Replaces senior-eng top-1 (STC) with Quantizr-family base + export-time discipline. Top-5 lanes ordered by EV with concrete byte budgets + Modal vs Vast.ai allocation. Section 3 single highest-EV experiment: clean-source STC archive build + auth eval on Modal T4 CUDA.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Revised theoretical floor

- **0.270 central**, 0.255-0.295 feasible band (was 0.245 senior eng).
- Reason: the prior 0.245 counted STC as immediately-stackable -0.03 rate move. AV1 regression removes that from current Lane A/G anchors. Clean-source STC can recover part of it but carries integration + source-distribution risk.
- Hard sub-0.30 math: at 250KB archive, rate term = `25*250000/37545489 = 0.1665`, so non-rate must be ≤ ~0.1335. Quantizr is roughly there; Selfcomp is not without KL/architecture/diet.
- **Sub-0.30 probability: 24% central, 15-35% confidence band.** Rises >30% if SC++/q_faithful lands ≤0.35 before diet; falls <15% if both land >0.42.

## REVISED via portfolio recheck (same day, codex Pattern A)

User pushback: "we have a full team working on this and it's too early to be so conservative and pessimistic"

Council re-ran portfolio OR-math:

| Base lane | P(score ≤ 0.45) | P(score ≤ 0.40) |
|---|---:|---:|
| SC++ v4 | 0.38 | 0.24 |
| q_faithful_v3 | 0.34 | 0.21 |
| DARTS-S a3 | 0.19 | 0.08 |
| sz_phase2_v2 | 0.18 | 0.08 |
| SA v4 | 0.14 | 0.05 |
| mae_v_v2 | 0.13 | 0.05 |
| lane_w_v2 | 0.11 | 0.04 |

- Raw OR P(any ≤ 0.45) = 81.9%; P(any ≤ 0.40) = 56.0%
- Hassabis correlation haircut: 65-70% / 40-45%
- Stack gain central 0.075-0.085 (Ω-W weight pool + STC mask pool NON-overlapping)
- Marginal sub-0.30 = 0.21·0.80 + 0.21·0.55 + 0.25·0.18 + 0.33·0.03 = **0.339**

**REVISED SUB-0.30 PROBABILITY: 34% central, band 23-46%.**

## CORRECTION 2026-04-29 PM (post-MPS-error)

The earlier note in this file ("STC FALSIFICATION refunds 0.060") was based on an MPS-derived measurement that CLAUDE.md explicitly forbids using for strategic decisions. That refund is WITHDRAWN. The 34% portfolio probability stands, conditional on a CUDA validation of clean-source STC (Modal T4, ~$0.20). Until that CUDA measurement lands, clean-source STC is UNDETERMINED, not falsified, and the council's #1 hope is back on the table.

Important corrections:
- Ω-W (weight pool) and STC (mask pool) are **NOT overlapping** — full additive stack gain
- Telemetry gap: SA/SC++/q_faithful/W rely on auth_eval.log not RESULT_JSON; Ω-W parser looks for `"score"` while canonical writes `final_score`. Clean-source STC writes byte manifest but no final JSON. **Fix before ranking.**
- Resurrect: restricted DARTS-S only as independent arch base (do NOT resurrect old KL/ADMM/wavelet)

## Top-5 lanes by EV with byte budgets + dispatch targets

| Rank | Lane | Byte budget | Dispatch target | Stackability |
|---|---|---|---|---|
| 1 | **Quantizr-family base** (SC++ v4 / q_faithful_v3 winner) | 250KB total: 105KB grayscale/mask, 85KB SegMap weights post-block-FP, 16KB pose, 4KB codebooks/LCT, 40KB reserve | Modal A10G/T4, auth on Modal T4 CUDA | Mutually exclusive base |
| 2 | **Ω-W water-filling export** | Reduce weight payload 85-95KB → 60-75KB; frees 15-30KB | Encoder-only Modal T4/A10G, 3 budgets | Stackable with #1 only if SegMap checkpoint exists |
| 3 | **Clean-source STC** | mask payload ≤150KB inside 250KB class-mask stack | Modal T4 CUDA encoder-only; NEVER on AV1-decoded masks | Mutually exclusive with analog grayscale unless redesigned as residual |
| 4 | **LCT / analog target tuning** | 10 bytes payload, ≤512B with metadata; Δ -0.005 to -0.015 | Inside existing Modal A10G training OR encoder-only sweep | Stackable with SC++ + DARTS-S; irrelevant to raw AV1 class masks |
| 5 | **Restricted DARTS-S** | Same 250KB ledger as #1 with arch replacing weights | Vast 4090 only, 3 configs maximum | Mutually exclusive with #1 as final base; stackable with #2/#4/diet |

**Carmack container**: keep deterministic minimal ZIP as free hygiene; **kill as a lane**. Measured 0-500B = 0-0.00033 score, no strategic value.

## Section 3: Single highest-EV experiment (next 4 hours)

**LAUNCH clean-source STC archive build + auth eval on Modal T4 CUDA.**

Predicted Δ: baseline Lane G is 694,074B archive bytes. Each 45KB saving = -0.030 score; 100KB = -0.0666; 170KB = -0.113.
- Distortion may move either way because exact clean masks replace AV1-decoded masks → only contest-CUDA score counts.

**Abort rules:**
- masks.stcb roundtrip fails
- Archive >650KB
- MPS/CPU used for strategic measurement
- masks.stcb not recognized by inflate

**Commit rules:**
- Score ≤1.00 OR
- Archive saves ≥100KB with non-rate regression <0.02

## Section 4 — KILL LIST

**Cancel / keep killed:**
- SO Hessian fallback
- STC-on-AV1 anchors (NOT clean-source — that lives)
- Carmack standalone (deterministic-ZIP hygiene only)
- UNIWARD standalone
- GP polynomial
- Joint ADMM
- Old KL loss mode (not Quantizr-style corrected)
- Adaptive rebalance
- PoseNet clamps
- Wavelet as primary (deferred paper lane)
- C3
- Expanded DARTS (only restricted 3-config)

**Do not cancel:**
- q_faithful_v3
- SC++ v4
- DARTS-S restricted

**Keep as control:** SA v4 only until SC++ has a viable checkpoint, then stop.

**Let reach next auth endpoint, then stop unless sub-0.45 signal:** sz_phase2_v2, mae_v_v2, lane_w_v2.

## Section 5 — Composition stacks

| Stack | Components | Predicted score (central) | Notes |
|---|---|---|---|
| **Likely** | SC++/q_faithful base + Ω-W + LCT + deterministic ZIP | 0.318 (band 0.30-0.34) | Base dominates; Ω-W is separate weight-rate pool; LCT small distortion gain |
| **Aggressive Rate** | Quantizr-class base + Ω-W + clean-source STC (where representation-compatible) + LCT | 0.300 (band 0.285-0.315) | Discount mask-rate additivity by ~50% — STC and grayscale/latent masks overlap |
| **Moonshot** | DARTS-S improved base + Ω-W + LCT + archive diet | 0.286 (band 0.265-0.305) | Wins only if arch search improves non-rate without growing weights beyond 250KB ledger |

## Per-voice key prescriptions (compressed)

- **Shannon LEAD**: 15KB must buy ~0.01 score or it is waste
- **Dykstra CO-LEAD**: do not stack three mask-rate attacks naively; treat every byte pool as constrained simplex
- **Yousfi**: kill AV1-reencode STC permanently; clean-source STC if pure byte parsing
- **Fridrich**: original STC valid only before lossy quantization; prioritize grayscale/analog latent masks
- **Contrarian**: no new long lane without CUDA auth endpoint, byte cap, abort rule
- **Quantizr**: keep q_faithful_v3 alive through auth
- **Hotz**: cancel anything that cannot produce submission-shaped zip tonight
- **Selfcomp**: SC++ is still main path; add LCT if wired, use Ω-W on weights
- **MacKay**: archive diet must report entropy + side info + actual zip bytes
- **Ballé**: Ω-W needs SegMap checkpoint, not bare renderer.bin
- **Boyd**: no ADMM; grid + projection; 3 budgets per exporter
- **Tao**: define each lane by source distribution; no transfer of byte estimates across them
- **Filler**: cap masks.stcb; if boundary bitmap + residuals doesn't beat AV1 by ≥45KB, stop
- **van den Oord**: LCT bolt-on for grayscale-LUT/SC++ only; no standalone GPU lane
- **Carmack**: ZIP overhead ~328 bytes on Lane A; 50KB dream dead
- **Hassabis**: diversify only across independent failure modes
- **Hinton**: SC++ may use corrected train-time KL if eval_roundtrip + CUDA auth
- **Karpathy**: every running lane must surface JSON score + archive bytes
- **Schmidhuber**: no C3, no giant search, no ADMM
- **Jack-from-skunkworks**: first archive <300KB with non-rate <0.16 = base for all final stacks
- **Chair**: next 96 hours are not for new theory; they are for landing exact archives

## How to apply

- Section 3 launch is the sole 4-hour priority: clean-source STC archive build + Modal T4 auth.
- All deferrals/kills above are binding for the next 96 hours.
- Stack predictions are central + band — measure each landing immediately, do not infer.
- Floor 0.270 + sub-0.30 prob 24% means: do not bet the lab on a single sub-0.30 outcome; ship the best landing we have at deadline (May 3).

## Cross-refs

- /tmp/codex_runs/grand_council_final_designs.log (full 477KB transcript)
- project_lane_stc_av1_regression_finding_20260429.md (the empirical setback that triggered this council)
- project_senior_engineer_review_floor_revised_245_20260429.md (the prior floor — superseded)
- project_codex_theoretical_floor_brutal_20260429.md (codex 0.22-0.30 brutal-floor analysis)
- experiments/build_clean_source_stc_archive.py (Section 3 launchable)
- scripts/remote_lane_stc_clean_source.sh (Section 3 dispatch script)
