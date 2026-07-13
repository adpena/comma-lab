# DIRECTIVE (operator-routed, 2026-07-13) → weyl_symmetry_unification: compute the ACTUAL Noether currents

Operator supplied the full field-theoretic Noether machinery (slide): for r infinitesimal symmetries
δx_ν = ε_r X_rν and δη_ρ = ε_r Ψ_rρ of the action, the r conserved currents are
  j_r^ν = [ ∂L/∂(∂_ν η_ρ) · ∂_σ η_ρ − L δ^ν_σ ] X_rσ − ∂L/∂(∂_ν η_ρ) · Ψ_rρ,   ∂^ν j_r^ν = 0,
with the translation specialization X_μν = δ_μν, Ψ = 0 giving the stress-energy tensor T^μν.

SHARPEN YOUR TASK 3 from "which subgroup?" to the CURRENT COMPUTATION: the witness IS a field theory — the
level-set field φ (η_ρ) over (x, t) with OUR S_τ energy density as L (the unified variational level-set flow,
canonical in tac.canonical_equations + the GR unified action L10). For EACH one-parameter factor of the invariance
group you enumerate (task 1), plug its (X, Ψ) into the boxed formula and compute j_r^ν EXPLICITLY:
- Phase/zero-mode translations (L87's conserved charge) → which (X,Ψ)? Derive the charge Q = ∫ j^0 and check it
  reproduces the measured per-boundary phase constants.
- Spacetime translations → T^μν: the temporal component is the ego-motion ξ transport (the L83/L87 covariance law);
  state the correspondence exactly.
- RATE LAW COROLLARY (the payoff): every conserved charge is FREE TEMPORAL RATE — conserved = derivable from
  initial data, zero bytes/frame. Enumerate the charges → the derived per-clip byte saving vs the current CGauge
  accounting. A charge that is only APPROXIMATELY conserved (symmetry broken by events) pays only its BREAKING —
  that is exactly the v8 event-coding law; connect them.
Label every step DERIVED vs INFERRED; where the discrete/stratified structure breaks Noether's hypotheses
(argmax is not smooth — the currents live on the smooth τ>0 family and limit τ→0), say so precisely.
