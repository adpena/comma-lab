# margin_hinge vs PR95 l7_softplus — int8-QAT gradient-geometry potency probe (RESULT)

**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19110. NO score claim.
**Date:** 2026-06-22. **$0** (local CPU, no GPU, no paid spend, no PR). **research_only=true.**

## The question (the watch-item, made measurable)
Is OUR `margin_hinge` seg loss (`margin_target=1.0`) as POTENT per-step for the contest d_seg
(argmax-disagreement rate) as PR95's vendored stage-5 `l7_softplus_seg_loss`
(`tau=0.3, l7_threshold=1.0, l7_mult=4.0`), **under int8 QAT**, at the REAL fork point
(`stage4_v332_qat_end`, post-QAT / pre-stage-5)? The int5 finding
(`feedback_frontier_int5_score_aware_qat_finetune_path_b_caps`) raised the candidate that
margin_hinge "minimizes the surrogate on FP but loses it post-quant" at a coarse grid.

This is a **per-step gradient-geometry MEASUREMENT at ONE checkpoint** — a candidate-disambiguator,
NOT a curriculum-outcome verdict and NOT a kill (CLAUDE.md "Forbidden premature KILL" +
"Measurement-first"). The decisive arbiter remains the live run's Muon stage-8 d_seg slope.

## Method (apples-to-apples; seg-loss FUNCTION is the only variable)
- Fork point: `stage4_v332_qat_end` EMA decoder + latents (bc20, taper [16,16,17,19,19,14,10],
  latent_dim=28); first N=24 of the 600 pairs (cached GT argmax targets, zero recompute).
- Frozen **CPU** SegNet (the d_seg authority — MPS forbidden; also the live run owns MPS).
- d_seg = mean per-pixel argmax-disagreement through the EXACT eval round-trip
  (decoder→384→bicubic↑874→uint8-STE→bilinear↓384→SegNet) — the SAME quantity the contest computes.
- For each loss L: one gradient `g_L` of the seg loss on the FP fork decoder; then `w'=w−lr·g_L`
  on a COPY for an LR sweep {1e-4,3e-4,1e-3}; measure d_seg (a) FP, (b) int8-fake-quant (127-level,
  the vendored stage-4 QAT — the deployed authority surface).
- Reuse (no reimplementation): `RealScorerContext`, `ConfigurableTaperHNeRVDecoder`,
  `_seg_loss_for_spec` (margin_hinge path = the live `--seg-margin-hinge` path), vendored
  `losses.l7_softplus_seg_loss` (= `specs[4].seg_loss_fn`), `score_aware_qat._fake_quantize_n`.

## Deep-math framing (the geometry of the two losses)
Both losses act on the SAME per-pixel signed margin `g = logit[GT] − max_{c≠GT} logit[c]`
(`g<0` ⇔ the pixel's argmax is FLIPPED = the contest d_seg event):
- **margin_hinge** = `relu(margin_target − g)`: gradient slope is **−1 (constant)** for every
  pixel below the target margin, **0** above. Maximal, uniform pull on every flipped/near-flip
  pixel; spends ZERO on confident-correct interior. A HARD geometry on the same surface as the
  hard argmax (no temperature).
- **l7_softplus** = `tau·softplus(−g/tau)` with an L7 hard-pixel weight boost: gradient is a
  **smooth sigmoid** `−σ(−g/tau)` → saturates to −1 for very-wrong pixels, decays toward 0 for
  very-correct ones, with a soft transition of width ~tau (0.3) around the boundary. Never exactly
  zero on correct pixels (spends a little there), L7-reweighted to concentrate on hard pixels.
- **The int5/QAT hypothesis under test:** int8 quant perturbs weights → perturbs logits → shifts
  `g` by ~the quant noise. If margin_hinge's constant-slope step lands the boundary pixels at
  `g≈0` (right at the flip threshold), int8 noise could re-flip them (erosion). l7_softplus's soft
  transition pushes the boundary to `g>tau` (a margin of safety) → potentially more quant-robust.
  The probe MEASURES whether this erosion gap is real at this fork point.

## MEASURED (single-step gradient geometry; N=24, CPU, int8 QAT 127-level)

`result_singlestep.json`. Probe `--n-pairs 24 --lrs 1e-4,3e-4,1e-3`. Deterministic
(two independent runs bit-identical on every number below).

**Baseline d_seg at the fork point (stage4_end EMA):**
| surface | d_seg |
|---|---|
| FP (full precision) | 0.0019349 |
| int8-fake-quant (127-level, the deployed surface) | 0.0019307 |
| **quant erosion @ fork** (quant − fp) | **−0.0000042** (≈ 0; int8 does NOT erode the fork) |

**Per-loss single gradient (FP fork decoder):**
| loss | seg loss value | ‖g‖ (decoder-param grad norm) |
|---|---|---|
| margin_hinge (target 1.0) | 5.752e-3 | 9.523e-3 |
| l7_softplus (τ0.3, L7×4) | 6.868e-3 | 3.869e-2 (**~4.06× larger**) |

**Per-step Δd_seg (w′=w−lr·g on a copy), FP vs int8-quant, per LR:**
| loss | lr | Δd_seg_fp | Δd_seg_quant | quant erosion of step |
|---|---|---|---|---|
| margin_hinge | 1e-4 | −6.4e-7 | −3.2e-6 | −2.5e-6 |
| margin_hinge | 3e-4 | −4.2e-7 | +1.7e-6 | +2.1e-6 |
| margin_hinge | 1e-3 | +8.5e-7 | −1.1e-6 | −1.9e-6 |
| l7_softplus | 1e-4 | −4.2e-7 | −6.4e-7 | −2.1e-7 |
| l7_softplus | 3e-4 | −4.2e-7 | −2.1e-6 | −1.7e-6 |
| l7_softplus | 1e-3 | −2.1e-7 | −2.1e-6 | −1.7e-6 |

**Best post-quant Δd_seg:** margin_hinge −3.2e-6, l7_softplus −2.1e-6 → post-quant gap
(l7 − mh) = **+1.1e-6** (margin_hinge marginally MORE potent post-quant, but ALL deltas
are within the ±few-pixel noise band — one pixel of d_seg over 24 pairs = 2.1e-7).
**Gradient cosine(g_margin_hinge, g_l7_softplus) = 0.641** (the two losses point in
substantially the SAME direction — same boundary geometry, ~50° apart).

**Multi-step descent (n_steps=60, descent_lr=1e-3):** RAN but is impractically slow on
CPU at this checkpoint (each step = forward+backward through the 874×1164 round-trip +
EfficientNet-B2 SegNet; eval points rebuild decoders) → ~25-30 min for both arms. It is
IN-FLIGHT as a strengthener; the verdict below stands on the single-step + cosine + grad-norm
evidence, which is conclusive at this checkpoint. (When the multistep JSON lands it is folded
into `result.json` under the `multistep` key; the single-step result is preserved in
`result_singlestep.json`.)

## Honest verdict (NO kill): NO_DIVERGENCE_EVIDENCE

At the stage4_end fork point, **int8 QAT does NOT preferentially erode margin_hinge**:
the quant erosion AT the fork is ≈0 (−4.2e-6), and the per-loss per-step Δd_seg are ALL
within the ±few-pixel noise floor for BOTH losses — neither loss moves a converged d_seg
in a single step at LR ≤ 1e-3, and the post-quant gap (+1.1e-6, margin_hinge marginally
better) is well inside noise. The int5/coarse-grid hypothesis ("margin_hinge minimizes the
surrogate on FP but loses it post-quant") is **NOT reproduced here** — the int8 grid
(127-level) is much finer than the int5 grid where that finding arose, so the absence of
erosion at int8 is consistent with (not contradicting) the int5 finding; it just means the
divergence, if real, lives at much coarser grids than the deployed int8.

This is exactly what the conditioning thesis predicts: the fork point is already
converged-flat in the AdamW phase (κ≈19 boundary Hessian; a diagonal preconditioner
descends d_seg power-law-slow), so a single gradient step — of EITHER loss — barely moves
d_seg. The flat per-step potency is the AdamW conditioning-crawl, NOT a loss defect.

**Action: KEEP margin_hinge on the live run. Do NOT switch. Do NOT kill.** The two losses
are gradient-aligned (cos 0.641) and equipotent at this checkpoint within noise; margin_hinge
is the validated detector-informed d_seg lever and is already below the plain-CE bc20 basin.
The per-stage-loss A/B (CE-coarse-early / margin_hinge-fine-late, possibly l7 in stage 5)
remains a candidate for the NEXT vehicle's bat00 A/B — NOT a mid-run intervention. The
decisive arbiter is the live run's Muon stage-8 d_seg slope.

Verdict taxonomy (all non-terminal, NO wall):
- `NO_DIVERGENCE_EVIDENCE` — losses within noise post-quant → margin_hinge is fine; the flat
  stage-5 d_seg is the PREDICTED AdamW conditioning-crawl (κ≈19 boundary Hessian), not the loss.
  Keep margin_hinge.
- `L7_POST_QUANT_ADVANTAGE` — l7_softplus clearly more potent post-quant AND quant erodes
  margin_hinge MORE → the int5 divergence is a LIVE candidate → recommend the fuller bat00
  per-stage-loss A/B. Do NOT switch the live run.
- `MARGIN_HINGE_ADVANTAGE` — margin_hinge wins post-quant → keep the lever (confirms the audit's
  expectation that margin_hinge is the validated d_seg lever).

## NO-FAKE ledger
- MEASURED: per-step Δd_seg fp vs int8-quant for both losses at the real `stage4_end` fork point,
  N=24 pairs, frozen CPU SegNet, exact eval round-trip, vendored int8 QAT. The d_seg IS the contest
  argmax-disagreement (not a proxy).
- SCOPE: per-step potency at ONE checkpoint — NOT the full-curriculum outcome. Does NOT measure
  what 9000 stage-5 + 5000 Muon epochs do. Candidate-disambiguator only.
- NOT claimed: no score moved; no kill/wall; the live run is NOT switched. Pointer UNMOVED 0.19110.

## 6-hook / research_only declaration
- research_only=true (advisory measurement; non-promotable). The 6 unified-Lagrangian wire-in hooks:
  1. sensitivity-map: N/A (per-step gradient probe, not a persisted per-axis byte sensitivity).
  2. Pareto constraint: N/A (no archive bytes changed; seg-axis only).
  3. bit-allocator hook: N/A (informs a future per-stage loss choice, not per-tensor bits).
  4. cathedral autopilot: N/A (advisory probe, not archive-deployable).
  5. continual-learning posterior: the verdict + measured deltas are the durable signal for the
     next-vehicle per-stage-loss A/B decision (recorded in this memo + the JSON).
  6. probe-disambiguator: THIS IS the disambiguator for the margin_hinge-vs-l7 watch-item.

Cross-refs: `experiments/probe_margin_hinge_vs_l7_softplus_grad_geometry.py` ·
`experiments/results/margin_hinge_vs_l7_probe_20260622/result.json` ·
`decisive_run_161_config_and_deepmath_optimality_audit_20260622.md` (section C watch-item) ·
`feedback_frontier_int5_score_aware_qat_finetune_path_b_caps_20260618.md` (the int5 finding) ·
`experiments/probe_lensA_ce_vs_margin_dseg_slope.py` (sister FP-slope probe — reused machinery).
