# THREAD-B SHARED NEW MATH — differentiable d_seg surrogate (MEASURED) + QA-entropy design (2026-06-11)

**Node:** the cross-node dependency the DAG (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` §2
THREAD-B "SHARED NEW MATH") names for B1 (Cool-Chic), B6 (IB carrier) + A2 (postfilter).
**Authority:** every figure `[macOS-CPU advisory]` (torch 2.11 CPU, the EXACT authority decode path; NO
MPS; GT decode ONLY via `frame_utils.yuv420_to_rgb`). Frontier UNMOVED 0.191 — this is the enabling math,
not a pointer move. `score_roadmap_update_eligible=False`; `mechanism_update_eligible=True`.

---

## 1. The problem (the genuinely-new math)

The contest SegNet term is a NON-differentiable set functional (`upstream/modules.py:111-113`, verified):
`d_seg = mean( argmax(SegNet(rendered_last_frame)) != argmax(SegNet(gt_last_frame)) )` over a 384×512
field. Any smaller-basis retrain or learned postfilter that wants to descend the EXACT d_seg needs a
differentiable surrogate that PROVABLY correlates with this argmax-flip rate.

## 2. Reuse targets (search-and-familiarize-first; nothing rebuilt)

- `src/tac/score_aware_loop/live_segnet_loss.py` — the surrogates ALREADY exist (PR95 1:1 port): `ce_seg_loss`
  (L81), `tau_softplus_seg_loss` (L92), `smooth_disagreement_seg_loss` (L100), `l7_softplus_seg_loss` (L114),
  the shared `_target_minus_runnerup_margin` core (L42), and the non-proxy observable `exact_d_seg_from_logits`
  (L155). The surrogate work was NOT to invent a loss — it was to MEASURE which one tracks exact d_seg.
- `src/tac/score_aware_loop/targets.py:30` `load_frozen_distortion_net` (frozen SegNet+PoseNet, YUV6-patched).
- `src/tac/analysis/segnet_boundary_marginals.py:98` `logit_margin` + `summarize_boundary_features` (the
  boundary-band risk axis).
- `src/tac/entropy_bottleneck.py:19` `EntropyBottleneck` (Ballé factorized CDF, uniform-noise-train /
  round-eval STE, `bits_per_element`) + `src/tac/learnable_entropy_model.py:457` `LearnableEntropyModel`
  (Ballé 2018 hyperprior, int8 quantization, brotli codec) — the QA-entropy foundation, both already built.
- Prior descent proof: `.omx/research/inert_loop_fix_20260610T193900Z.md` (the surrogate DESCENDS exact d_seg
  0.508→0.081 on real EfficientNet-B2 over a training trajectory). This memo adds the ORTHOGONAL,
  population-level correlation the DAG asked for.

## 3. The chosen surrogate + WHY (the math)

Per-pixel margin `m_i = z_target(i) − max_{c≠target} z_c(i)` (the shared core). `m_i < 0` IS an argmax flip
(a d_seg hit). The contest's exact indicator is `1[m_i < 0]`. The surrogate menu squashes `−m_i/τ`:

- **`smooth_disagreement` = `mean σ(−m_i/τ)`** — the SOFT 0/1 disagreement indicator. **At τ→0 it converges
  to the exact argmax-flip RATE itself** (`σ(−m/τ) → 1[m<0]`), so minimizing it directly minimizes d_seg.
  This is the principled pick: it is not *correlated with* d_seg, it is a smoothing OF d_seg.
- `tau_softplus`, `l7_softplus` — convex relaxations (softplus is the integral of σ); gradient peaks near the
  m≈0 decision boundary, which is exactly where the optimal-teacher memo
  (`feedback_optimal_teacher_and_sensitivity_tools_landed_20260531.md`) showed ~99% of d_seg gradient lives
  (confident interior pixels can never flip → zero d_seg sensitivity). `l7` adds a hard-pixel boost.
- `ce_seg_loss` — standard CE; correlated but lower slope (it spends gradient on confident pixels too).

**Recommendation for the smaller-basis nodes:** train with **`smooth_disagreement_seg_loss` (τ≈0.3, anneal
toward ~0.1)** as the primary seg term — it is the tightest, lowest-bias smoothing of the exact functional;
keep `ce_seg_loss` as the warmup/stabilizer (its broader support conditions early training). This matches
PR95's stage progression (CE→softplus→smooth-disagreement→L7).

## 4. The MEASURED surrogate↔exact-d_seg correlation (the decisive deliverable)

Method: decode REAL GT frames from `upstream/videos/0.mkv`; build a population by degrading each at 12 graded
strengths × 3 perturbations (additive luminance noise, box blur, uint8 requant) in the NATIVE 0–255 scorer
input space; run BOTH the exact scorer (true argmax-flip d_seg vs the clean-GT argmax) and each surrogate
(margin loss on the degraded SegNet logits vs the same GT argmax targets). Train = first 4 frames, held-out =
last 4 frames. Module: `src/tac/score_aware_loop/surrogate_correlation.py`. Artifact:
`.omx/research/surrogate_dseg_correlation_20260611T145051Z.json` (108 points/group).

| surrogate | Pearson r | Spearman ρ (full) | **Spearman ρ (suprafloor, the meaningful regime)** | OLS slope | held-out ρ (suprafloor) |
|---|---|---|---|---|---|
| ce_seg_loss | 0.956 | 0.681 | **0.982** | 0.333 | **0.992** |
| tau_softplus | 0.985 | 0.778 | **0.992** | 1.186 | **0.992** |
| **smooth_disagreement** | 0.986 | 0.743 | **0.991** | 1.001 | **0.994** |
| l7_softplus | 0.991 | 0.800 | **0.991** | 0.316 | **0.984** |

**Verdict: every surrogate is a strongly-positive, near-monotone tracker of the EXACT d_seg in the regime
that matters.** Pearson r ≈ 0.96–0.99 (magnitude). Suprafloor Spearman ρ ≈ 0.98–0.99 on BOTH train and
held-out (the rank ordering is essentially exact once d_seg > 5e-3). `smooth_disagreement` has the cleanest
profile (slope ≈ 1.0 — it IS d_seg, smoothed). Gradient flow + descent are already proven in
`inert_loop_fix`; this confirms the population-level monotonicity a training signal needs.

## 5. Question-all-interpretations (the recursive greenup; 3 passes)

- **Is the full-cloud ρ≈0.74 a weakness?** NO — diagnosed as an artifact. The full cloud is dominated by
  near-zero-d_seg points (small perturbations all "pass": exact d_seg ~ argmax SAMPLING NOISE, ~1e-4, where
  surrogate ordering is meaningless because every render is already good). Filtering to the SUPRA-floor
  regime (exact d_seg > 5e-3 ≈ the contest operating-point) lifts ρ to 0.98–0.99. The dampener is the floor,
  not the surrogate. (`is_strong_positive` uses the suprafloor ρ when ≥8 suprafloor points exist.)
- **Held-out artifact?** NO — the suprafloor ρ is 0.984–0.994 on the held-out (last 4) frames, equal to
  train. Not overfit to the test frames.
- **LIVE render vs EMA shadow (the shadow-lag lesson `capstone_ema_shadow_lag_*`)?** N/A here — this measures
  the surrogate-vs-exact functional relationship on FIXED frames (no EMA in the loop). The lesson DOES bind
  the consuming trainer: it must measure exact d_seg on the LIVE render with EMA-decay WARMUP
  `min(decay,(1+t)/(10+t))`, not the frozen-near-init shadow. Flagged for B1/B6/A2.

## 6. The usable surrogate spec the smaller-basis nodes import

```python
from tac.score_aware_loop.live_segnet_loss import (
    smooth_disagreement_seg_loss,  # primary seg term: σ(−margin/τ) → exact d_seg at τ→0
    ce_seg_loss,                   # warmup/stabilizer
    exact_d_seg_from_logits,       # the non-proxy validation observable
)
from tac.score_aware_loop.targets import load_frozen_distortion_net, build_gt_targets
# loss = 100 * smooth_disagreement_seg_loss(live_segnet_logits, gt_argmax) + 1 * pose_loss(...)
# gradient flows render -> frozen SegNet -> render params (NO learnable student head — that was the
# inert-loop elephant, ledger #75/#76). Validate with exact_d_seg_from_logits on the LIVE (EMA-warmup) render.
```
Contract: SegNet logits `(B, 5, h, w)`, GT argmax targets `(B, h, w)` int64 (the d_seg reference). The
`ScoreAwareTrainer` in `trainer.py` already wires this whole loop; a Cool-Chic/IB/KAN carrier plugs in as the
only variable.

## 7. The QA-entropy design (lighter — for B1 Cool-Chic latent grids)

The B1 historical DEFER blocker was "export-design": the multiresolution latent grid must survive int8/FP4A
quantization. The Ballé quantization-aware pattern (ALREADY in `entropy_bottleneck.py`) is the answer — train
the rate term so the post-quantization bits are what the loss minimizes:

1. **Train-time soft quantization:** `y_hat = y + U(−Δ/2, Δ/2)` (additive uniform noise, the differentiable
   stand-in for rounding to grid Δ); `EntropyBottleneck.forward` already does this with Δ=1. For an int8 grid
   pick Δ = (max−min)/255 per-channel; for FP4A use the non-uniform FP4 code points (reuse `fp4_quantize.py`)
   and add noise in the *log-domain* level spacing.
2. **Rate term:** `bits = −log2( CDF(y_hat+Δ/2) − CDF(y_hat−Δ/2) )` with a learnable factorized (or
   ARM/hyperprior) CDF — `EntropyBottleneck.rate_loss()` returns exactly this. Add it to the Lagrangian:
   `L = 100·d_seg_surrogate + 1·pose + λ·bits_per_element`.
3. **Eval-time:** `y_hat = round(y/Δ)·Δ` (the STE round; `EntropyBottleneck` switches on `self.training`),
   then arithmetic-code under the SAME CDF (`LearnableEntropyModel`/`mask_entropy_coder.py` already implement
   the int8 + brotli/range path). Train-eval consistency is the whole point: minimizing `bits` minimizes the
   ACTUAL archive bytes the quantized grid costs, so there is no FP4A surprise at export.
4. **Cool-Chic specifics:** the ARM (autoregressive entropy model over the latent grid) conditions the CDF on
   decoded-causal neighbors — port the 5.0 ARM to predict (loc, scale) and feed `EntropyBottleneck`'s
   `_channel_params`. The multiresolution grids each get their own λ (coarse grids tolerate coarser Δ).

This is design-only (B1 lane is GPU-gated); the kernels it composes all exist and are int8/FP4A-tested.

## 8. The single biggest risk

**The surrogate-exact gap at the argmax boundary under quantization.** The correlation is measured on
continuous degradations; the consuming carrier renders QUANTIZED frames (int8/FP4A). At a pixel whose margin
`m_i ≈ 0`, an int8 rounding step can flip the argmax in a way the smooth surrogate (evaluated on the
pre-round logits) does not see — the surrogate says "fine" while the exact d_seg ticks up. Mitigation:
(a) train with the eval-roundtrip in the inner loop (the `ScoreAwareTrainer` already does bicubic-up → STE
round → bilinear-down, so the surrogate sees the post-round logits); (b) anneal τ toward ~0.1 so the surrogate
sharpens toward the true indicator near the boundary; (c) weight the surrogate by the boundary-band map
(`segnet_boundary_marginals.logit_margin` → `exp(−margin/τ)`) so capacity concentrates exactly on the
flip-prone pixels (the optimal-teacher boundary-TCKD direction). Secondary risk: τ-bias — at finite τ the
surrogate's minimizer is slightly off the exact-d_seg minimizer (the smoothing bias); the τ-anneal closes it.

## 9. 6-hook wire-in

#1 sensitivity-map ACTIVE (the surrogate IS the score-aware gradient source every smaller-basis node's
sensitivity depends on; the boundary-band weight is the per-pixel allocation prior). #2 Pareto N/A (no
bytes). #3 bit-allocator ACTIVE-via-QA-entropy-design (the rate term + per-grid λ is a bit-allocator hook).
#4 cathedral-autopilot N/A (research surface, not archive-deployable directly). #5 continual-learning ACTIVE
(the suprafloor-ρ≈0.99 verdict reseeds the B1/B6/A2 priors: the surrogate is VALIDATED, train against it).
#6 probe-disambiguator THIS IS the disambiguator (resolves the DAG's open "does the surrogate track exact
d_seg across a population, not just one trajectory" question — YES, ρ≈0.99 suprafloor, held-out-confirmed).

## 10. Cross-refs

`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (THREAD-B SHARED NEW MATH node) ·
`inert_loop_fix_20260610T193900Z.md` (the descent proof this complements) ·
`pr95_elephant_audit_20260610T185556Z.md` (the inert-loop bug class this validates against) ·
`feedback_optimal_teacher_and_sensitivity_tools_landed_20260531.md` (boundary-band / DKD theory) ·
`capstone_ema_shadow_lag_reverses_seg_wall_verdict_20260611T070000Z.md` (the LIVE-vs-shadow caveat for
consumers) · `hnerv_muon/src/losses.py` (the PR95 surrogates ported in `live_segnet_loss.py`).
