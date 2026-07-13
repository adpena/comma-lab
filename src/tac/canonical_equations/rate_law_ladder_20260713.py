"""Canonical registration: THE RATE-LAW LADDER (2026-07-13) — the four-rung derived rate theory.

Four operator-sourced digs landed in one day and COMPOSE into the program's derived rate theory.
Each law is DERIVED (grounded in the named memo, adversarially reviewed by its agent); none is a
score claim; the pointer moves only through byte-closed exact rows. This module is the equations
leg of the triality for the ladder; the DAG leg is FEED-math-ladder-20260713; the DSL leg is N/A
(laws, not levers — the levers they spawn register separately).

RUNG 1 — WEYL GROUPOID (#464, weyl_symmetry_group_unification_20260713.md):
  The scoring functional's invariance structure is NOT a global group. Exact completion =
  the score-fiber permutation group  G_S ≅ ∏_s Sym(S⁻¹(s));  constructive realization = a
  STATE-DEPENDENT STRATIFIED GROUPOID  [K_B ∨ K_R ∨ K_Y ∨ F_P ∨ C_A ∨ F_phot] ⋊ H_cov.
  Noether: the conserved charge is the CONJUGATE MOMENTUM of the phase zero-mode (not the
  zero-mode itself); broken symmetries pay only their breaking (= the v8 event law).

RUNG 2 — SETOID RATE LAW (#466, infdesc_foundations_dig_20260713.md):
  x ≈_{U,D} y  ⟺  x,y ∈ D ∧ U(x)=U(y);   D/≈ ≅ U(D);   R^sem = H([W]) = H(U(W)).
  Groupoid orbits satisfy E_G ⊆ E_{U,D}; equality = FIBER-COMPLETENESS. An incomplete
  invariance atlas pays exactly the conditional entropy  GAP = H(q_G(W) | U(W)) ≥ 0.
  The quotient theorem alone proves NO byte saving — a receiver-computable section is required.

RUNG 3 — BURNSIDE / SECTION LAW (#467, garrett_algebra_dig_20260713.md):
  |im S| ≤ 2^64  ⇒  the ideal quotient-label costs ≤ 64 bits per frozen scoring axis, so the
  semantic floor H(U(W)) is negligible and ESSENTIALLY ALL PAYLOAD IS THE SECTION COST — the
  witness paradigm (#155) DERIVED from orbit-counting. Twist term (section-invariant):
  R_twist^ideal = H(Θ | q_H, 𝒜, public)  (Schreier class; H² obstruction awaits typing H_cov).

RUNG 4 — MARKED CONDITIONAL CHAIN RULE, temporal (#468, condprob_homotopy_lie_dig_20260713.md):
  H(X, W′ | C) = H(X|C) + H(E|X,C) + H(Φ|E,X,C) + H(Δᴱ|Φ,E,X,C)
  (ξ-transport cost → MARKED events → receiver phase → event residuals). Corrections banked:
  Weyl strata refine to σ=(κ,ω,a,r) — homotopy type κ is a PROJECTION (groupoid strictly finer);
  L85 flicker spikes occupy the receiver-phase/lattice channel, not necessarily π₀/π₁; flip
  conditional-independence is universally FALSE, empirical I(F;C|M,ξ) UNKNOWN (probe queued).

COMPOSED OPERATING STATEMENT (the ladder's single sentence):
  Rate work = SECTION ENGINEERING: capture invariances (shrink the fiber-completeness gap
  H(q_G|U)) + build receiver-computable sections + pay only marked-event/phase/twist terms in
  time. Algebra never replaces fiber-completeness or descent.
"""

from __future__ import annotations

EQUATION_ID = "rate_law_ladder_v1"
_UTC = "2026-07-13"
_AXIS = "[DERIVED theory; grounded per-rung in the named memos; score_claim=false]"

# Rung anchors (memo paths are the empirical grounding surfaces)
RUNG1_MEMO = ".omx/research/weyl_symmetry_group_unification_20260713.md"
RUNG2_MEMO = ".omx/research/infdesc_foundations_dig_20260713.md"
RUNG3_MEMO = ".omx/research/garrett_algebra_dig_20260713.md"
RUNG4_MEMO = ".omx/research/condprob_homotopy_lie_dig_20260713.md"

# Rung-3 constants (DERIVED)
SCORE_IMAGE_LOG2_BOUND_BITS = 64          # |im S| <= 2^64 per frozen scoring axis (float64 image)
QUOTIENT_LABEL_IDEAL_BITS_MAX = 64        # => semantic floor H(U(W)) <= 64 bits/axis (ideal)

# Named gap/twist terms (symbols; measured values are OWED — see the costate pool rows)
GAP_TERM = "H(q_G(W) | U(W))"             # fiber-completeness gap (rung 2) — un-captured invariance rate
TWIST_TERM = "H(Theta | q_H, A, public)"  # section-invariant Schreier/cocycle rate (rung 3)
TEMPORAL_CHAIN = "H(X|C) + H(E|X,C) + H(Phi|E,X,C) + H(Delta^E|Phi,E,X,C)"  # rung 4

# Honest status ledger for the ladder's open measurables (each has a costate pool row)
OWED_MEASURABLES = (
    "fiber_completeness_gap_n600",         # measure H(q_G|U) empirically via conditional codelength
    "flip_conditional_mi_I_F_C_given_M_xi",  # nested n600 cross-fitted conditional-codelength probe
    "h2_obstruction_after_typing_Hcov",    # type the extension, then derive split-vs-twist
    "event_marks_telemetry",               # rung-4: events need MARKS not counts (v8 law upgrade)
)


def composed_statement() -> str:
    """The ladder's single operating sentence (for digest/dashboard consumers)."""
    return ("Rate work = section engineering: shrink the fiber-completeness gap H(q_G|U), build "
            "receiver-computable sections, pay only marked-event/phase/twist terms in time; the "
            "semantic quotient label is <= 64 ideal bits per frozen axis (Burnside).")
