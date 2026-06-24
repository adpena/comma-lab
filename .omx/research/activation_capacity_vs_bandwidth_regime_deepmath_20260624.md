# Activation choice is REGIME-DEPENDENT: bandwidth-limited (n100) vs capacity-limited (n600/contest) — the deep-math grounding (2026-06-24)

**Source:** operator 2026-06-24 — *"All must be deep math and algebra and topology and physics and geometry
and calculus and more grounded."* Applied to the live n600 FINER-confirm CAUTION (FINER reversing to 1.03×
SIREN @ ep352 vs its stable 0.85× lead at n100). This memo derives, from first principles, WHICH hypothesis
(H1 late-emergence vs H2 capacity-fragile) the theory predicts — turning "wait and see" into a falsifiable
prediction + a sharpened plan. Authority `[analysis]`; all d_seg via byte-closed CPU-authority; pointer
UNMOVED 0.19110; NO score claim.

## 0. The question, made precise
The decoder has fixed P params (bc20 ≈ 83K) at b bits/param. It must represent n per-frame boundary
configurations (the lane-island argmax edges move per frame). FINER won d_seg at n=100 by −18.7% (lower
FLOOR, both plateaued). At n=600 the advantage is weaker and reversing. WHY — and what does it imply?

## 1. Two regimes (the algebra of the binding constraint)
Per-frame information budget = **P·b / n bits/frame** (fixed P shared across n frames).
- **n=100:** budget = P·b/100 — generous. The model is over-parameterized for the per-frame target; the
  binding constraint is the **inductive bias / representable frequency band** (which edges CAN the activation
  express). → the activation's spectral reach sets the floor.
- **n=600:** budget = P·b/600 — 6× tighter. The model approaches the under-parameterized regime; the binding
  constraint shifts to **representational efficiency** (how many params does one edge COST). → params-per-edge
  sets the floor; spectral reach is moot if you can't afford the bits to use it.

The crossover is where `P·b/n` falls below the bits needed to represent the n-th boundary configuration at the
activation's efficiency. n=100→600 plausibly crosses it (the empirical reversal is the signature).

## 2. NTK / spectral lens — why FINER wins ONLY in the bandwidth-limited regime
A coordinate-MLP with activation σ has an NTK whose eigenspectrum sets the learnable frequencies (Tancik
2020, Yüce 2022 structured-dictionary). SIREN (ω fixed) is band-limited at ω; **FINER's** variable-frequency
`sin((|x|+1)x)` has instantaneous freq ω_eff = ω(2|x|+1) → a **broader reachable band** (higher freqs where
|x| large). In the bandwidth-limited regime (n100), the broader band lets FINER place the high-freq edge SIREN
smooths → lower floor (the measured −18.7%). In the capacity-limited regime (n600), BOTH are sines of the same
parametric family → **same params-per-edge efficiency** → their floors converge; FINER's extra band is
unusable under-budget. → **FINER ≈ SIREN at n600 is the PREDICTION** (H2), not a surprise.

## 3. MDL / approximation-theory lens — why the EDGE criterion favors steps under capacity pressure
The d_seg metric is **pointwise argmax-at-the-edge** — an **L∞ criterion near the boundary** (a pixel flips
iff the rendered logit crosses the facet at THAT pixel). Approximation theory:
- A **step / piecewise-constant** representation encodes one edge in **O(1) params** (location + height),
  with **zero overshoot** (exact in L∞ at the edge).
- A **Fourier/sine** representation of a step needs **O(1/ε) harmonics** to suppress Gibbs overshoot to ε,
  and **never** removes the L∞ overshoot (Gibbs constant ≈ 9% persists for any finite N). The overshoot is
  exactly what flips shallow-margin pixels (the texture-survival wall #149).
→ Under capacity pressure (n600), **params-per-edge is the binding cost**, and **step bases are O(1/ε) more
efficient than sines for edges** — AND strictly better in the L∞-at-edge norm that d_seg actually is. So the
capacity-limited winner is predicted **step-native (hosc/step_basis), not bandwidth (FINER).**

## 4. Geometry / topology lens — the basis must match the target's structure
The argmax target is a **piecewise-constant function on a stratified domain** (regions ∪ codim-1 boundaries).
The natural function-space basis for piecewise-constant functions is **indicator/step functions**, not
sinusoids. Representing a discontinuous (codim-1) jump with a smooth (C^∞) sine basis is a category mismatch
that the Gibbs phenomenon quantifies. hosc (`tanh(β sin)` → square wave) and step_basis (Σ soft-Heaviside)
are the topology-matched basis; their efficiency advantage is structural, hence **regime-robust** (it does
not depend on having spare capacity — it IS the capacity efficiency).

## 5. Physics lens — free energy under a bit budget
View training as minimizing a free energy F = distortion + (1/β_T)·description-length at fixed budget. In the
generous-budget (n100) phase the entropy term is slack → the model explores the full hypothesis space and the
inductive bias (bandwidth) dominates the minimum. In the tight-budget (n600) phase the description-length
term binds → the minimum is dominated by the MOST COMPRESSIBLE representation of the target = the
fewest-params-per-edge basis = steps. Same conclusion: the regime flip moves the optimum from bandwidth to
efficiency.

## 6. Empirical cross-check (existence-proof discipline — the data is consistent, not yet conclusive)
- **Across n:** n100 FINER 0.813× (win) vs n600 ep352 1.03× (reversed). Two points consistent with
  advantage-decays-with-n (capacity). NOT yet conclusive (n600 not converged).
- **Within the n600 run:** ratio 0.95 (ep50) → 0.95 (ep150) → 1.03 (ep352). Early training = under-capacity
  = bandwidth matters = FINER ahead; later = capacity filling = converge = FINER erodes. The within-run
  trend independently supports the regime mechanism.
- **Falsification of THIS memo:** if the n600 run CONVERGES with FINER clearly < SIREN (≤0.90×), H1 wins and
  the capacity framing is wrong (FINER's win IS a lower floor that just emerges late). Verdict at ~ep1500.

## 7. The sharpened plan (what the grounding CHANGES)
1. **PREDICTION (falsifiable):** the n600 FINER confirm converges to FINER ≈ SIREN (within ~±5%), NOT a
   −18.7% floor win. (Let it run; do not over-claim FINER either way.)
2. **The activation screen MUST run at n600** (the capacity-limited CONTEST regime), not only n100 — because
   the n100 ranking is bandwidth-regime and does NOT transfer. The n100 screen is a cheap pre-filter; the
   n600 screen is the load-bearing one.
3. **PREDICTED n600 ranking:** step-native (hosc, step_basis) > learnable (fkan, can learn steps) > FINER ≈
   SIREN ≈ finer_gauss (all sine-family, capacity-equivalent) > gauss/sinc (low-pass). The step-native
   advantage GROWS under capacity pressure (opposite of FINER's).
4. **Generator design consequence:** at the contest's capacity-limited operating point, the architecture
   lever is the **parameter-efficiency-of-edges** (step basis), and capacity-routing to the lane/horizon
   bands matters MORE (every param counts under budget). The geometry-prior + step-native activation
   compose.
5. **The deeper unification:** sub-0.15 at the capacity-limited contest regime is an **MDL problem** — the
   winning vehicle is the one that represents the argmax-edge manifold in the FEWEST bits (params×bits). That
   is the same statement as "the 8-dim nonlinear island manifold + a basis matched to it (steps)" — the
   convergence and this regime analysis are the same MDL principle from two directions.

## 8. NO-FAKE ledger
- DERIVED: the regime split (P·b/n budget), NTK bandwidth (FINER wins bandwidth-limited only), MDL
  params-per-edge (steps O(1) vs sine O(1/ε), L∞-at-edge), topology basis-match, free-energy framing.
- MEASURED: n100 FINER 0.813× (win); n600 ep352 1.03× (reversing); within-run 0.95→1.03. Consistent with the
  capacity prediction; NOT conclusive until n600 convergence (~ep1500).
- PREDICTION (falsifiable at ~ep1500): n600 FINER ≈ SIREN; step-native wins the n600 screen.
- NOT claimed: no score moved; pointer UNMOVED 0.19110. This grounds the plan; it does not move a row.

## 6-hook wire-in
#1 sensitivity-map: capacity-per-frame is a new prior axis (budget P·b/n). #2 Pareto: the regime crossover
bounds the achievable (bandwidth vs efficiency frontier). #3 bit-allocator: under capacity pressure,
params-per-edge IS the allocation objective (MDL). #4 cathedral: N/A. #5 continual-learning: this memo + the
n600 confirm + the n600 activation screen. #6 probe-disambiguator: the n600 screen IS the bandwidth-vs-
efficiency disambiguator; this memo predicts its outcome.
