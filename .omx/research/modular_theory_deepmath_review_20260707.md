# Modular theory — targeted deep-math pass (Door A: Tomita–Takesaki · Door B: modular representation theory) — 2026-07-07

`review_status: recovery-committed` — found untracked 2026-07-08 by the hardening sweep
(author agent apparently credit-died before committing); content complete + internally
consistent on sanity read, but the verdicts are UNREVIEWED by fresh eyes beyond that read
(verdict-review-status discipline, memory L81). Attribution: deep-math research subagent, 2026-07-07.

**Agent:** deep-math research subagent (report-only; no launches, no trainer/tool edits; $0 advisory).
Task: operator directive *"in addition to group theory we should explore MODULAR THEORY
implications"* — sibling of `.omx/research/group_theory_deepmath_review_20260707.md`, same
label discipline, same honesty bar: does modular theory yield a CONCRETE lever (loss term,
controller update, measurable quantity, rate trick, threshold), or is it a beautiful
re-description of what we already have?

**Label discipline:** MEASURED (ours) · THEIRS (paper) · DERIVED · INFERRED · SPECULATIVE ·
RHYMES-ONLY. **Authority:** $0, MEANS. Pointer contest-CPU **0.19110 UNMOVED**.

**Proactive-recall result (what we ALREADY have in this space):** grep of `.omx/research/`,
`src/tac/`, `docs/` for `modular|tomita|takesaki|KMS|araki|brauer|decomposition matrix|defect
group` returned **EMPTY** — this is the first modular-theory pass, no prior to contradict. The
adjacent surfaces it must sit consistently beside:
- **Costate λ = ∂S/∂x** (`costate_lambda_marginal_ds_v1`): the triality's 4th object, an exact
  finite-dim Pontryagin shadow price `(100, 5/√(10 d_pose), 25/37545489)`. NOT a log-density.
- **Frozen softmax = categorical exponential family**; Fisher `F = diag(p) − ppᵀ`; measured
  `0.978 = tr F read through the margin`; anisotropy 9.56:1 (infogeo lens Ch.2).
- **τ = ε = ħ** (`tau_eps_hbar_one_dequantization_two_scales_v1`): the softmax free energy
  `F_τ = ⟨distortion⟩ + τ·H(softmax(φ/τ))` is a **Gibbs free energy at temperature τ**;
  softmax `p ∝ exp(−E/τ)` is a **Gibbs state at β=1/τ**. CE→τ curriculum = mirror-descent
  Γ-convergence homotopy (Raskutti–Mukherjee MD ≡ natural gradient, PROVEN).
- **FEED-08k code coder** (group review §E, −3,108 B): a **char-0, Maschke-semisimple Z₂
  (frame-parity) isotype split** — the ORDINARY representation, not a modular one.
- **STC / syndrome-trellis coding** (Filler grand-council seat): char-2 LINEAR CODES (parity
  checks over GF(2)) — coding theory, already in lineage.

---

## Door A — Tomita–Takesaki modular theory. VERDICT: DECISIVE-NEGATIVE (real chain, degenerate for us)

**THEIRS (skeleton):** vN algebra M, cyclic-separating Ω, modular operator Δ = S*S (S: aΩ↦a*Ω),
**modular Hamiltonian K = −log Δ**, modular flow σ_t(a) = Δ^{it} a Δ^{-it}, **KMS condition**
(state ω is thermal at β w.r.t. σ_t), **Araki relative entropy** S(ω‖φ) = −⟨Ω|log Δ_{φ|ω}|Ω⟩ via
the relative modular operator. The chain the operator floated — *relative-entropy → Fisher →
modular Hamiltonian → KMS-temperature* — is REAL abstract mathematics (Kubo–Mori: the quantum
Fisher IS the Hessian of relative entropy and IS a modular-operator integral ∫ρ^s A ρ^{1−s} ds).

**The decisive finite-dim caveat (DERIVED, theorem — this is the whole verdict):** our objects are
**classical / commutative**. A per-pixel 5-class softmax is a probability VECTOR (diagonal state),
i.e. the state lives on a **commutative** von Neumann algebra `ℓ^∞({Road,Lane,Undriv,Movable,
MyCar})`. **Every state on an abelian algebra is tracial** (ω(ab)=ω(ba) since ab=ba) ⇒ **Δ = 1,
modular flow σ_t ≡ identity, K ≡ 0.** Tomita–Takesaki says *literally nothing* for classical
probability. The non-trivial content (non-inner modular automorphisms, Type III factors, Connes
cocycle, thermal time) needs infinite-dim / continuum QFT we do not have. Even the richest thing
available to us — a Type I_n matrix algebra with non-tracial ρ — gives σ_t(a)=ρ^{it}aρ^{-it} =
Heisenberg evolution = **pure spectral calculus of ρ**, nothing eigendecomposition doesn't already
give. So each link of the seductive chain DEGENERATES the moment the algebra is commutative, and
hands back exactly the classical object we already hold. Point-by-point:

1. **Is our costate λ = ∂S/∂config a modular Hamiltonian?** **NO (false friend).** K = −log ρ is
   the log-density of a STATE, generating a flow that PRESERVES the state (equilibrium). λ = ∂S/∂x
   is the gradient of an EXTERNAL cost (d_seg+d_pose+bytes), generating gradient descent that
   CHANGES the state toward lower cost — opposite role, different type. The only rhyme is
   "log-derivative-shaped," and λ isn't even a log (only λ_pose = 5/√(10 d_pose) is nonlinear, from
   the score's √, not a log-partition). The costate is a **Pontryagin costate** — which we already
   have, correctly named. Relabelling it "modular Hamiltonian" over-claims.

2. **Does KMS-at-β map onto τ-annealing and PREDICT a schedule?** **Correspondence GENUINE but
   already ours; schedule NO.** softmax at τ IS a Gibbs/KMS state at β=1/τ — TRUE, and already the
   `τ=ε=ħ` free-energy view. But KMS is a **static equilibrium condition at each fixed β**; it says
   nothing about how to VARY β over training time. The cooling LAW (how fast to anneal τ) is set by
   our Γ-convergence / critical-slowing / homogenization analysis (`deepmath_lens_dynamics_
   transition_easing`), not by KMS. KMS adds the *name* "KMS state," not a new schedule, threshold,
   or controller update. (Thermal-time hypothesis — modular flow generating a canonical time — is
   the one place it *could* add dynamics, but for a classical Gibbs categorical state that flow is
   trivial: RHYMES-ONLY.)

3. **Araki relative entropy / relative modular operator as a NEW distortion for d_seg?** **NO.** In
   finite commutative dim, S(ω‖φ) = −⟨Ω|log Δ_{φ|ω}|Ω⟩ reduces **exactly** to classical KL
   `Σ ω_i log(ω_i/φ_i)` — which we already use (CE surrogate + Fisher = its 2nd-order expansion).
   Moreover d_seg is an **argmax-Hamming** disagreement, not a relative entropy — Araki is not even
   the right functional form for the distortion term (it's the right form for the CE TRAINING
   surrogate, which is already Fisher/KL). No new distortion measure.

**The single "if anything" candidate (recorded, graded not-decisive):** relative-entropy
**MONOTONICITY under CPTP maps** (Araki/Uhlmann/Lindblad DPI). Our R operator (render→uint8→
resize→scorer) is a deterministic Markov map, so D(p‖q) ≥ D(R#p‖R#q). This is a REAL, correct
CEILING on achievable task-space separation through R — but it is the **classical** data-processing
inequality (needs no modular theory), and it certifies a bound we already respect operationally
("through-R authority"). It bounds how much R destroys; it does not open new score. Could be
registered as a ceiling equation if we ever want the DPI stated formally, but it is not a lever.

**Cheapest $0 test:** none required — the negative is a THEOREM (abelian ⇒ trivial modular flow).
A fig-leaf confirmation (compute σ_t on a per-pixel 5-class state, watch it return identity) would
cost $0 and teach nothing.

---

## Door B — Modular REPRESENTATION theory (char-p, blocks, Brauer, decomposition matrices). VERDICT: DECISIVE-NEGATIVE (inapplicable — wrong characteristic)

**THEIRS:** for a group algebra `kG` over a field of characteristic p | |G|, Maschke FAILS →
non-semisimple, indecomposable ≠ irreducible, **blocks / defect groups / Brauer characters /
decomposition matrices** (ordinary irreps ↦ modular irreps mod p). The natural char-p companion to
the FEED-08k group-decomposition coder.

**Adjudication (DERIVED):** the entire subject requires working in **characteristic p dividing the
group order**. We never do.
- **Our symbols are char 0.** Weights are fp32 quantized to int8; the successful FEED-08k coder is
  a **Z₂ parity split** — |Z₂|=2, so modular rep theory bites only in **char 2**, but our int8
  symbols are ordinary integers/reals, not GF(2) elements. The parity-deinterleave that won −3,108 B
  is precisely the **ordinary (char-0) irrep decomposition of Z₂** (trivial ⊕ sign, two 1-dim
  irreps) where **Maschke HOLDS** and the split is clean/semisimple — *that is why it worked*. There
  is no non-semisimple structure to exploit; modular rep theory would only matter if the char-0
  decomposition DIFFERED from a mod-p one, which requires char p | |G| we do not have.
- **Brotli/arithmetic coders use integer arithmetic mod 256, but that is a numerical
  implementation detail, not a kG-module over GF(p).** No group symmetry of the payload is being
  reduced mod a prime dividing a group order.
- **The only rate lever in this neighborhood is weight-symmetry orbit coding, and it is a MEASURED
  NO** (group review §C: 387 B / 0.47% theoretical cap, −8 B realized through brotli, controls
  +32/+72/+251 B). Even the char-0 orbit slack is negligible and unrecoverable; the char-p
  refinement has nothing to act on.
- **The genuinely useful finite-field object we DO use is coding-theoretic, not
  representation-theoretic:** STC / syndrome-trellis coding = **linear codes over GF(2)** (Filler
  seat), for mask payload. That is coding theory (parity-check matrices), a different subject from
  blocks/Brauer characters of `kG`. It is already in lineage; modular rep theory proper adds
  nothing to it.

**Cheapest $0 test:** none — inapplicable by characteristic (theorem-level), and the sole adjacent
rate lever (weight-symmetry orbits) is already measured NO.

---

## Bottom line (one paragraph, honest)

**Modular theory is NOT a real new axis for this project — it is an elegant lens that does not move
the exact score, in both readings.** Door A (Tomita–Takesaki): the chain relative-entropy→Fisher→
modular-Hamiltonian→KMS is real *general* mathematics, but our states are classical/commutative, so
every link DEGENERATES to an object we already hold under a clearer name — modular flow is trivial
(abelian ⇒ tracial), Araki relative entropy = the classical KL we already use, KMS temperature =
the `τ=ε=ħ` Gibbs free-energy view, and the costate λ is a Pontryagin shadow price, NOT a modular
Hamiltonian (false friend). No new schedule, distortion, threshold, or controller update; the one
correct byproduct (relative-entropy DPI through R) is the classical data-processing ceiling we
already respect. Door B (modular representation theory) is *inapplicable* — our weights are char 0,
the winning FEED-08k Z₂ split is the ordinary Maschke-semisimple decomposition, we never code over
GF(p) with p | |G|, and the only adjacent rate lever (weight-symmetry orbits) is already MEASURED
NO. Nothing to build; nothing to test that isn't a theorem. Recorded so no future pass re-derives
it. **Consistent with `group_theory_deepmath_review_20260707.md`:** the group-theory value was the
*orbit-coding / free-action-counted-coordinates* MDL framing and the FEED-08k coder — both live in
**char-0 ordinary** representation theory + rule-118 economics; modular (operator-algebra OR char-p)
theory is the extension into regimes (continuum / char p) that our finite classical setup does not
occupy.

Sources (verified): Tomita–Takesaki / Bratteli–Robinson (abelian ⇒ trivial modular group,
standard) · Araki relative entropy = classical KL in commutative case (Ohya–Petz *Quantum Entropy*)
· Kubo–Mori Fisher ↔ modular operator (Petz) · Raskutti–Mukherjee arXiv:1310.7780 (MD ≡ natural
gradient, ours) · in-tree: `costate_lambda_marginal_ds_20260705.py`, infogeo lens Ch.2,
`group_theory_deepmath_review_20260707.md` §C/§E.
