# Lever G — Engineered deterministic correction $0 smoke + verdict (2026-06-10)

**Task #52, lever G** (the zero-byte distortion lever, GOAL_standing_v3). Authority of every
number below: **`[local CPU-torch advisory]`** — exact upstream `DistortionNet` on CPU, GT decode via
`frame_utils.yuv420_to_rgb` (the carrier harness enforces it). NOT contest-CPU/CUDA; a candidate-
generator advisory. NO MPS. ZERO archive bytes (lever G is a deterministic decode-time rule, not a
data sidecar).

## What lever G is (and what it is NOT)
Lever G = a FIXED, zero-archive-byte, deterministic transform applied to the carrier's decoded
frame1 at inflate time (NO scorer loaded — strict-scorer rule) that reduces the carrier's exact
`d_seg` by nudging SegNet(comp-frame1) argmax toward SegNet(GT-frame1) argmax on net. Canonical proof
of the CLASS: PR95-family L28 (subtract 1.0 from specific RGB channels; 0 bytes; ~−0.0001..−0.0005).
**The existing `engineered_corrections{,_v2}.py` are per-pixel int8 DATA sidecars that cost archive
bytes — they are NOT lever G** (lever G is a deterministic function, code constants only). So this unit
adds a NEW zero-byte deterministic-rule surface if a rule wins; it does not duplicate the sidecar.

## Base + harness (apples-to-apples)
- **Base carrier**: the current contest-CPU frontier decode = `fp11_source_brotli_recode` (the
  recoded-R3 hold `0.19109982` is a *lossless byte-recode* of this same carrier, so the DECODED
  frames are byte-identical — lever G's per-pixel effect is identical on either; the rule's value is
  ratified on the frontier carrier later). archive_sha (frontier pointer) =
  `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`.
- **Harness**: `experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis/render_and_score_lib.py`
  — reproduces inflate.py per-pair decode (incl DQS1 + channel postproc), GT via `yuv420_to_rgb`,
  exact `DistortionNet.compute_distortion` on CPU. Verified: baseline mean `d_seg`≈5.6e-4,
  `d_pose`≈2.3e-5 (matches the standing residual).

## DIAGNOSTIC (measured 2026-06-10, 24 random pairs, `diag_disagreement.py`)
- mean d_seg = 5.648e-4; avg **111 disagree pixels/frame** over 196608 (model grid 384×512).
- **Disagreement is bidirectional and ~symmetric**: comp0→gt1 = 27.5% vs comp1→gt0 = 26.2%;
  comp2→gt0 9.7% vs comp0→gt2 6.8%; comp3→gt2 6.7% vs comp2→gt3 6.1%. A GLOBAL class-bias rule that
  helps one direction HURTS its mirror → ~cancels. **This is the crux that makes a fixed rule hard.**
- **Margin separates near-perfectly**: disagree pixels have comp top1−top2 median 0.156 (91.8% < 0.5);
  agree pixels median 10.124 (99.7% > 0.5). The residual lives at razor-thin SegNet boundaries.
- **Inflate constraint**: the margin field needs SegNet, which is FORBIDDEN at inflate. So the rule
  must be a pure function of decoded RGB pixels that *implicitly* helps the boundary without seeing it.

## PRE-REGISTRATION (written BEFORE measuring the rules)
**Rule family X** (each a deterministic function, ZERO archive bytes, NO scorer at inflate, NO per-pixel
data; constants are the only fitted quantity, selected by offline search over the family — honestly a
"fixed rule selected by offline search over rule family X", NO-FAKE class 6):
  1. **G1 — global per-channel additive offset** `frame1[c] += b_c` (the PR95-L28 class; 3 constants).
  2. **G2 — global per-channel affine** `frame1[c] = a_c*frame1[c] + b_c` (6 constants; contrast/bias).
  3. **G3 — fixed 3×3 separable smoothing / unsharp on frame1 RGB** (1–2 constants: blend α, kernel).
     Rationale: boundary argmax noise may be high-freq; a fixed low-pass could denoise comp toward GT.
  4. **G4 — fixed boundary morphology in LUMA**: a fixed small-radius median/morph on the luma channel
     (the SegNet input is RGB but luma carries most structure); pixel-space only.

**Ranking signal**: aggregate EXACT `d_seg` (and `d_pose` guard) over a TRAIN frame split; the winning
constant tuple is the argmin of mean d_seg subject to `Δd_pose ≤ +1e-6` (corrections must not harm
pose; if a rule helps seg but harms pose it is restricted/rejected — seg-safe / pose-null cone).

**PREDICTION** (pre-registered): given the bidirectional symmetry, I predict **NO global fixed rule
cuts d_seg ≥ 10% relative at zero bytes** on a held-out split. The honest expectation is that the
symmetric boundary flips cancel under any global pixel-space op; the most likely positive is a small
(<5%) effect from a fixed low-pass that denoises boundary speckle, and it may not generalize.

**KILL CRITERION**: if the best fixed rule cuts < 2% relative d_seg on the HELD-OUT split (or harms
pose net-score), **DEFER lever G** with the finding + reactivation criteria. PROCEED (→ ratify on the
frontier carrier, exact contest-CPU) only if a single rule generalizes to ≥ a few-% d_seg cut at
Δd_pose ≤ 0 net-score on held-out.

## RESULT (measured 2026-06-10, `lever_g_measure.py`; `lever_g_result.json`)
**VERDICT: DEFER (global-fixed-rule subclass CLOSED).** The pre-registered prediction is CONFIRMED —
even stronger than predicted: NO rule beats the identity.

| split | baseline d_seg | baseline d_pose |
|---|---|---|
| train (pairs 0–13) | 5.282e-4 | 2.05e-5 |
| held-out (pairs 300–313) | 5.272e-4 | 4.50e-5 |

| rule | best constant | held-out d_seg | rel cut | pose-safe |
|---|---|---|---|---|
| G1 per-channel offset | b = [0,0,0] | 5.272e-4 | **+0.00%** | yes |
| G3 low-pass blend | α = 0.0 | 5.272e-4 | **+0.00%** | yes |

The coordinate-descent + α-search BOTH selected the **identity** (every nonzero offset/blend either
raised train d_seg or violated the Δd_pose ≤ +1e-6 guard). KILL criterion (<2% held-out cut) → DEFER.

**Why (crux confirmed):** (1) the diagnostic's bidirectional-symmetric boundary flips cancel under any
GLOBAL pixel-space op (a rule helping comp→gt1 hurts the mirror comp1→gt0); (2) the frontier carrier
ALREADY applies the PR95-L28 channel postproc (`render_and_score_lib.py:144-147` — frame0 R/B, frame1 G
−1.0), so it is already AT the global-fixed-rule local optimum. There is no *additional* zero-byte
global correction to harvest. ZERO archive bytes spent; pointer UNMOVED.

**Reactivation (DEFER, not KILL — the zero-byte-correction paradigm is intact per Catalog #307):** a
winning rule must be SPATIALLY CONDITIONED on the boundary — but the boundary needs SegNet (forbidden at
inflate). The only inflate-legal conditioning signals are the decoder's OWN output structure (its
edges/gradients, regenerable without the scorer). Reactivation = a rule keyed on the renderer's
self-computed edge map (not the scorer's margin), OR accept this closes lever G and route firepower to
lever C (the smaller amortizer) where the residual actually lives.
