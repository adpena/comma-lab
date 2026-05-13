# Ledger 04 — DARPA Program Lineage (OFFSET / EXPLAIN / Mosaic / ASSIST)

**Date:** 2026-05-13
**Lane:** `lane_expert_team_aerospace_stealth_analytic_alien_tech_20260513`
**Evidence grade:** `[stealth-engineering-analog]` and `[mathematical-derivation]`.

---

## 0. Persona discipline

DARPA (Defense Advanced Research Projects Agency, 1958-) funded the transformative
research programs that produced the F-117, stealth in general (XST 1976), the Internet
(ARPANET 1969), GPS (NAVSTAR), and adversarial AI (XAI 2017). Recent programs we draw on:

- **OFFSET** (Offensive Swarm-Enabled Tactics, 2017-2021): 250-UAV swarms for urban
  reconnaissance. Connection: many specialized agents → mosaic of capability.
- **Mosaic Warfare** (DARPA strategic concept, 2017-): low-cost replaceable platforms
  fielded as a kill-web instead of monolithic high-cost platforms. Connection: compositional
  encoder design.
- **XAI / EXPLAIN** (Explainable AI, 2017-2021): force ML models to expose reasoning.
  Connection: scorer-blindspot identification.
- **ASSIST** (Adaptive Sensing for Smart Surveillance, 2018-): adaptive sensor allocation
  across regions of interest. Connection: bit-allocator design.

---

## 1. OFFSET swarm tactics → multi-encoder mosaic

### 1.1 OFFSET principle

The OFFSET program demonstrated that 250 small, specialized UAVs can collectively achieve
mission outcomes that a single large UAV cannot — at lower cost-per-mission and lower
risk. Each UAV is task-specific (one carries jammer, one carries seeker, one carries
relay). The SWARM is the capability; no individual UAV is critical.

### 1.2 Contest analog

Ledger 02's B2 mosaic encoder swarm is the OFFSET analog applied to compression. Each
specialized renderer handles a behavior-class of pairs; the COMPOSITION is the encoder.

### 1.3 Derived technique D1 — Heterogeneous renderer mosaic (refinement of B2)

**Refinement over B2.** Beyond behavior-class partitioning (easy/medium/hard/catastrophic
pairs), introduce ROLE specialization:
- Role A: SegNet-specialist renderer (optimizes seg-distortion at fixed pose+rate)
- Role B: PoseNet-specialist renderer (optimizes pose-distortion at fixed seg+rate)
- Role C: Rate-specialist renderer (minimizes bytes; sacrifices a small amount of seg/pose)
- Role D: Boundary-specialist renderer (handles argmax-flip pixels)

Each role-renderer is ~5-10 KB. Inflate-time dispatch reads per-pair role labels (~2 bits
each = 150 B) and applies the role-specific decoder.

Mathematical optimization: per-pair, choose the role-renderer that minimizes
`100·d_seg + sqrt(10·d_pose) + 25·B_p / 37,545,489`.

**Predicted Δscore:** -0.005 to -0.015 (refinement of B2 with role specialization).
**Build cost:** 8-12 days (3-4 role renderers + dispatch logic + per-pair labels).
**Risk:** dispatch overhead bites if role-prediction is wrong > 5% of time.

---

## 2. DARPA XAI / EXPLAIN → scorer-blindspot identification

### 2.1 XAI principle

DARPA's XAI program (2017-2021) required ML models to expose reasoning via Local
Interpretable Model-Agnostic Explanations (LIME), SHAP attribution, integrated gradients,
or similar feature-attribution. A retrospective by Gunning et al. (2021) showed that
explanations reveal SPURIOUS features the model relies on — features the analyst can then
exploit OR defend against.

### 2.2 Contest analog

SegNet's argmax depends on which 5-class logit is highest. We can ATTRIBUTION-decompose:
which input pixels does each argmax decision rest on? If the attribution is concentrated
in a small subset of pixels (e.g. boundary pixels), the rest is BLINDSPOT free bytes.

LIME/SHAP/IG attribution per-pixel on SegNet over the 1200 last-frames gives an exact map
of where SegNet is sensitive — and where it isn't.

### 2.3 Derived technique D2 — Attribution-driven bit allocation

**Mathematical formulation.** For each pixel `p` in each last-frame `f`:
1. Compute SegNet attribution `a_seg(p, f)` via integrated gradients (Sundararajan 2017)
   or expected gradients (Erion 2021).
2. For each pair `(f_0, f_1)` and each pixel of `f_0`, compute PoseNet attribution
   `a_pose(p, f)`.
3. Total per-pixel sensitivity: `a(p, f) = α·a_seg(p, f) + β·a_pose(p, f)`.
4. Allocate decoder bit-budget proportional to `a(p, f)`: low-sensitivity pixels get FEWER
   bits, high-sensitivity pixels get MORE bits.

**Predicted Δscore:** -0.004 to -0.015 (refines per-pixel bit allocation).
**Build cost:** 4-6 days (attribution compute is GPU-hungry but one-shot; bit-allocator
mod is shallow).
**Risk:** attribution maps depend on the renderer trained against them; circular
dependence. Use a stop-gradient version: compute attribution on a FIXED HNeRV checkpoint,
then re-train.

---

## 3. DARPA Mosaic Warfare → multi-encoder kill-web

### 3.1 Principle

Mosaic Warfare (DARPA Strategic Technology Office, 2017-2019) reframes military force from
"single high-cost platform" to "kill-web of low-cost mosaic tiles." Each tile is small,
replaceable, specialized. The composition is the capability.

### 3.2 Contest analog

The "kitchen sink anti-pattern" in CLAUDE.md is the OPPOSITE of Mosaic Warfare: PR105's
1776 LOC, 21 files monolith LOST to rem2's 241-LOC mosaic. The contest empirically
validates that small composable encoders > large monolithic encoders.

### 3.3 Derived technique D3 — Substrate kill-web architecture

**Operational rule.** Every substrate that lands at Lane Maturity Level 2+ must be:
- ≤ 350 LOC per substrate (CLAUDE.md non-negotiable)
- Composable with at least 2 other substrates via the `tac.composition.registry` primitive
- Independently replaceable (the inflate.py budget allows for substrate-swap)

D3 is meta-architectural. It doesn't add a substrate; it constrains how substrates compose.

---

## 4. DARPA ASSIST → adaptive sensor allocation

### 4.1 Principle

ASSIST (Adaptive Sensing for Smart Surveillance, 2018-2021) demonstrated that adaptively
allocating sensor resources to regions of interest based on real-time activity beats
uniform high-resolution sensing across the entire field of view. Key insight: **most of
the world is boring; spend resolution where activity is**.

### 4.2 Contest analog

The 1200 frames are NOT uniform in scorer-relevance. Some pairs are extreme (sharp turns,
braking events); most are routine (highway cruise). Spending uniform decoder fidelity
across all 1200 frames is suboptimal.

D1 (heterogeneous renderer mosaic) handles role-specialization; D4 below handles
TEMPORAL non-uniformity.

### 4.3 Derived technique D4 — Activity-driven temporal bit allocation

**Mathematical formulation.** Pre-compute per-pair "activity score" `act_p`:
`act_p = ||pose_delta_p||_2 + α·d_seg_residual_p`. Higher activity → larger byte budget.
The total budget is fixed; bits are redistributed across pairs.

For 70-80% "easy" pairs at activity quantile 0-0.7, allocate ~50-60% of budget.
For 10-20% "medium" pairs, allocate ~25-30%.
For 5-10% "hard" pairs, allocate ~10-20%.

Compare to PR101's uniform allocation: PR101 spends equal bytes on a 0.5s highway cruise
and a sharp braking event. D4 reallocates 30-50% of bytes from easy pairs to hard pairs.

**Predicted Δscore:** -0.003 to -0.010 (depends on activity-bin partition quality).
**Build cost:** 3-5 days (activity-score profile + per-pair budget allocator).

---

## 5. DARPA Explainable AI / adversarial defense → scorer-robustness probing

### 5.1 Adversarial principle

DARPA's adversarial AI research (Gunning 2021 retrospective) demonstrated that ML models
have ADVERSARIAL EXAMPLES — minimum-perturbation inputs that flip the model's output.
For the contest scorer, adversarial examples are pixel-level minimum-perturbations that
flip SegNet's argmax or shift PoseNet's pose output.

### 5.2 Contest analog

The DUAL question: instead of adversarially ATTACKING the scorer (which would invalidate
our submission), use the adversarial-example INVERSE — minimum-perturbation that PRESERVES
the scorer's output. The space of perturbations preserving the scorer's output is the
SCORER-EQUIVALENCE-CLASS (Grand Council §4) — exactly the orbit we want to encode.

The DARPA "expected gradients" tool (Erion 2021) is a closed-form way to characterize
this orbit per-pixel.

### 5.3 Derived technique D5 — Equivalence-class-aware perturbation tracking

**Formulation.** During training, maintain per-pixel "stable-orbit-radius" map:
`r(p) = max ε s.t. SegNet_argmax(f + ε·δ) = SegNet_argmax(f) for all unit δ`. Use this
map to:
1. Penalize encoder bits spent within the orbit (orbit-bytes are wasted).
2. Reward encoder bits spent at orbit-boundary (those are scorer-relevant).

This is the SABOR (A1) refinement: instead of treating "stable interior" as a binary
property, use the orbit RADIUS to determine per-pixel encoder budget.

**Predicted Δscore:** -0.005 to -0.015 (refines A1 with continuous orbit radius).
**Build cost:** 5-7 days (orbit-radius compute + encoder loss-function mod).

---

## 6. Council adversarial review

- **van den Oord (VQ-VAE creator; DeepMind).** "D1 heterogeneous renderer mosaic is exactly
  the mixture-of-experts (MoE) architecture. MoE has 30 years of literature; the
  contest-specific challenge is dispatch-overhead budget. With 100 LOC inflate budget,
  you can fit MoE-4 but not MoE-16." → **AGREE.** D1 is mixture-of-experts under
  Mosaic-Warfare framing. **Verdict: ENDORSE D1 with explicit MoE-4 bound.**

- **MacKay (memorial seat).** "D2 attribution-driven bit allocation is Bayesian-MDL
  applied per-pixel: bits should follow the posterior over scorer-sensitivity. The
  literature is well-established (Han et al. 2015 deep compression; Liu et al. 2017
  channel pruning). The novelty is per-pixel granularity." → **ENDORSE D2**.

- **Hassabis.** "D5 equivalence-class-aware perturbation tracking is the AlphaFold meta-
  pattern: don't predict the structure point; predict the ORBIT and pick the
  cheapest-to-encode element. This is structurally correct." → **STRONG ENDORSE D5**.

- **Karpathy.** "DARPA program names are vibe. The actual MATH is: A1 + D2 + D4 stack to
  cover (pixel-level + temporal) bit allocation. That's a 3-layer hierarchy; train it
  end-to-end." → **AGREE.** D2 + D4 + D5 are the three Levels of attribution-driven
  bit allocation. They stack composably.

- **Contrarian.** "MoE adds dispatch logic; the contest's inflate.py budget is
  binding-tight at 100 LOC. Don't propose architectures that don't fit." → **VALID
  CHALLENGE.** Revise D1 to MoE-4 maximum, with shared backbone (per Carmack's review in
  ledger 02). 4 heads × 20 LOC + shared backbone 20 LOC = 100 LOC, exactly at budget.

---

## 7. Reactivation criteria

- D1 heterogeneous mosaic: if MoE-4 dispatch overhead > 5 KB, reactivate at MoE-2.
- D2 attribution bit allocation: if attribution-map varies > 30% across HNeRV checkpoints,
  reactivate with consensus-attribution from N checkpoints.
- D3 substrate kill-web: this is operational discipline; reactivation N/A.
- D4 activity-driven temporal: if activity-bin partition is unreliable on held-out pairs,
  reactivate with smoother activity-score.
- D5 equivalence-class-aware perturbation: if orbit-radius compute is > $20 GPU per
  training run, reactivate with sampled-orbit-radius (10 pixels per frame, not all).

---

## 8. Citations (DARPA program lineage)

- DARPA OFFSET program: <https://www.darpa.mil/research/programs/offensive-swarm-enabled-tactics>
- DARPA OFFSET final flight tests (2021): <https://www.darpa.mil/news/2021/offset-swarms-take-flight>
- "OFFensive Swarm-Enabled Tactics (OFFSET)" DTIC report: <https://apps.dtic.mil/sti/pdfs/AD1125864.pdf>
- DARPA Mosaic Warfare overview — War on the Rocks 2019: <https://warontherocks.com/2019/12/mosaic-warfare-small-and-scalable-are-beautiful/>
- DARPA XAI program homepage: <https://www.darpa.mil/research/programs/explainable-artificial-intelligence>
- Gunning, D. et al. "DARPA's explainable AI (XAI) program: A retrospective" — Applied AI Letters 2021: <https://onlinelibrary.wiley.com/doi/full/10.1002/ail2.61>
- Sundararajan, M. et al. "Axiomatic Attribution for Deep Networks" — ICML 2017 (Integrated Gradients): <https://arxiv.org/abs/1703.01365>
- Erion, G. et al. "Expected Gradients" — Nature Machine Intelligence 2021: <https://www.nature.com/articles/s42256-021-00343-w>
- DARPA ASSIST program: <https://www.darpa.mil/research/programs/adaptive-sensing-for-smart-surveillance>

---

## 9. Wire-in (Catalog #125)

1. Sensitivity-map: D2 per-pixel attribution map IS the sensitivity-map primary signal.
2. Pareto: D1 MoE-4 dispatch shifts the Pareto frontier inward by ~5-10% feasible bytes.
3. Bit-allocator: D2 + D4 + D5 directly drive the bit-allocator.
4. Cathedral autopilot: D3 substrate kill-web architecture IS the autopilot's composition layer.
5. Continual-learning: D5 orbit-radius map feeds posterior on scorer-sensitivity priors.
6. Probe-disambiguator: D1 MoE-4 vs single-renderer is the canonical disambiguator (probe via 10-pair sample).

---

**End ledger 04.**
