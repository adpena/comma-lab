# Crosswalk — arXiv 2607.21366 "HOPE: Hilbert Operator for Progressive Encoding" vs the live campaign (operator drop 2026-07-26)

`research_only=true` · `score_claim=false` · `promotion_eligible=false` · pointer
`0.1910828242 [contest-CPU]` **UNMOVED** · effective competitive frontier `0.172`
(PR130, official leaderboard, per routing card §6). Every claim below is tagged
**PAPER-MEASURED** (their numbers) / **PAPER-DERIVED** (their math, transcribed) /
**REPO-MEASURED** (our named receipts, recalled not re-derived) / **DERIVED**
(follows from their math + our measured state) / **CONJECTURE** / **OPEN-QUESTION**.
Nothing here is contest evidence; the pointer moves only via `upstream/evaluate.py`
exact rows.

## Paper identification + custody

- **Title:** *Hilbert Operator for Progressive Encoding (HOPE): A Mathematical
  Framework for Deconstructing Learned Representations in Deep Networks*
- **Authors:** Hossein Mobahi (Google DeepMind), Peter L. Bartlett (Google
  DeepMind + UC Berkeley). arXiv:2607.21366v1 [cs.LG], submitted 2026-07-23.
  Preprint; no venue claimed. No code release referenced anywhere in the paper.
- **PDF SHA-256:** `8106a86c0c72aee05b0fbcc5e13095edc03fe725538b1bfa5c50c18ea806067b`
  (fetched via arxiv.org/pdf/2607.21366, 2026-07-26).
- **Read boundary:** full main body (Sections 1–12, Eqs. 1–29, Tables 1–2,
  Figures 1–4) + Appendix ToC + Appendix A (Hilbert space construction,
  Eqs. 30–35), B.1–B.5 (conv adaptation, ΔP footprint, cross-action overlap,
  decoupled cache, BN numerical stability, Eqs. 36–42), C.1 opening (axioms +
  Lemma C.1), E in full (kernel formulation, Eqs. 73–85), F.1–F.2 (block
  eviction derivation, Eqs. 86–97). Not read line-by-line: C.2–C.3 proof
  details, D, G, H, I. The abstract was NOT the mechanism source.

## Faithful technical summary

**The object.** A neuron `i` (incoming weights `w_raw,i`, BN affine `(γ_i, β_i)`,
BN running stats `(μ_i, σ_i²)`, PH-1 activation Ψ, outgoing weights `w_out,i`) is
first rewritten in *effective* parameters that absorb BN
(`w_eff = γ w_raw/√(σ²+ε)`, `b = β − γμ/√(σ²+ε)`, Eq. 1), then lifted to a
continuous function `f_i(x) = Ψ(y_i)·w_out,i` with `y_i = w_effᵀx + b` (Eq. 2),
and embedded as a **rank-1 Hilbert-Schmidt operator** `f_i = g_i ⊗ w_out,i` in the
tensor space `H ≅ L₂(X, P_X; ℝ) ⊗ ℝᶜ` (Eq. 31). The inner product factors
(Eq. 32–33):

    ⟨f_i, f_j⟩_H = K(i,j) · ⟨w_out,i, w_out,j⟩,   K(i,j) ≜ E_{x~P_X}[Ψ(y_i)Ψ(y_j)]

and **capacity** is the HS norm `‖f_i‖_H = ‖w_out,i‖₂·√K(i,i)` (Eq. 75). This
quotient kills both BN normalization invariance and PH-1 rescaling invariance
(λ on input weights / 1/λ on output weights cancels in the norm) — "raw
magnitudes are optimization artifacts" is made structural.

**The measure.** P_X is a maximum-entropy Gaussian surrogate constrained by the
layer's own BN statistics (`E[w_rawᵀx] = μ`, `Var(w_rawᵀx) = σ²`), justified by
CLT + Diaconis-Freedman: every neuron sees its input only through 1-D
projections, so a Gaussian surrogate matched on those projections is
asymptotically faithful for the 2-D slices the kernels actually integrate over
(the "post-ReLU support paradox" box). This makes the whole framework
**data-free for BN networks** (one-time small calibration pass otherwise).
Under P_X the pre-activation is exactly `y_i ~ N(β_i, γ_i²)` (Eq. 78).

**Closed-form kernels (ReLU).** Self-kernel (Eq. 3/79):
`K(i,i) = (γ²+β²)Φ(β/|γ|) + β|γ|φ(β/|γ|)` (φ,Φ = normal PDF/CDF). Cross-kernel:
exact = truncated-bivariate-normal moments (Eq. 83); practical zero-bias
approximation = arc-cosine order-1 kernel
`K(i,j) ≈ (1/π)(√(1−ρ̂²) + (π−arccos ρ̂)ρ̂)·√(K(i,i)K(j,j))` (Eq. 5/85), with the
pairwise warped correlation `ρ̂_ij = 2κ/(1+√(1+4κ²))` obtained analytically from
a **local 2×2 max-ent surrogate** via Woodbury (Eqs. 4, 80–81) — no covariance
inversion, no forward pass.

**The cost functional.** Axioms (Magnitude Neutrality; Connectivity Preservation
`J→∞` as layer capacity→0; Infinitesimal Capacity Dependence) force layer
capacity to be **L1-additive** `E(Φ)=Σ‖f_k‖_H` (Lemma C.1, from partition
invariance) and the transition cost to be a path integral `J = ∫ −c(Φ)Ė/E dt`.
A closed-form **upper bound** replaces the intractable path integral: straight
line in H^N for the numerator (Euclidean projection distance D), terminal
capacity for the denominator, `c(Φ)=N` to normalize by average feature capacity:

    J_prune = N‖f_i‖ / (E_a − ‖f_i‖)                                   (Eq. 6)
    J_merge = N·√(‖f_i−f_p‖² + ‖f_j−f_p‖²) / (E_a − ‖f_i‖ − ‖f_j‖ + ‖f_p‖)

**Merging = constrained rank-1 projection.** The pair `[f_i, f_j]` spans a rank-2
subspace; a physical parent must have tied outputs `[f_p, f_p]` (α=β=1
constraint makes Eckart-Young inapplicable). The optimal parent is closed-form
(Eqs. 12–15): direction `û` = principal eigenvector of `AᵀA` restricted to the
2-D child subspace (A ≜ w_out,i(w̃_in,i)ᵀ + w_out,j(w̃_in,j)ᵀ), a sign fix via the
exact objective, output direction v* by Cauchy-Schwarz alignment, scale
`s* = (a+bE_rem)/(2E_rem+b)` — note the optimal MAGNITUDE depends on the layer's
remaining capacity E_rem, not just the pair. Physical parameters (raw weights +
BN stats of the parent) are recovered exactly (Eqs. 16–18, App. D), with
variance-clamp edge cases handled (App. B.5).

**Macro block eviction.** For residual blocks, granular pruning cannot remove
W₃ (skip-dimension lock); a depleted pathway degenerates to an uncalibrated
bias injection `Y = X + b` (App. F.1 — catastrophic downstream ReLU clipping).
Block eviction projects F(X)→0 (pure identity), priced under the SAME axioms by
coupling the layer with the skip's "parallel survival capacity"
`E_identity = Σ_k √(γ_k²+β_k²)` (Eqs. 19–20, 96), linearized (ln(1+x)≤x) so
macro and micro actions compete in one currency.

**The encoding loop.** Rate-distortion trajectory planning: knapsack
`min Σa_k J_k s.t. Σa_k ΔP_k ≥ P₀−P_budget` (Eq. 21), relaxed to continuous →
Dantzig greedy on **distortion rate** DR = J_k/ΔP_k^init (Eqs. 22–23), receding
horizon (execute one action, re-evaluate). Two engineering points they prove
matter: (a) use the STATIC initial footprint ΔP^init, not the live shrinking
footprint — live-ΔP inflates neighbors' DR as layers compress and traps the
optimizer in fragmented states (Dantzig item-independence violation, Eq. 38;
uniform-scaling argument restores fair ordering, Eq. 39); (b) an O(1)
**decoupled cache**: the expensive per-pair quantities (u*, v*, a, b) depend
only on that layer's weights/BN — they cache two scalars per pair, and only s*
and J are recomputed from the live E_rem, giving O(1) cost queries and O(N²)
scans with JIT rank-2 SVD only for the winning pair.

**Applications (proof-of-concept, their own framing).**
- ResNet-50/ImageNet structured compression: accuracy-vs-density curve
  dominates L1-input, L1-joint, and BN-scale pruning baselines
  [PAPER-MEASURED as a plot; no tabulated numbers — effect sizes at any
  operating point are not quotable].
- **DEFT** (Dispersed Elastic Fine-Tuning): use the progressive-encoding prune
  costs J_prune^(i) to partition a trained net into frozen high-capacity core
  vs plastic low-capacity slack (percentile threshold, Eqs. 24–27); MERGE
  redundant core copies into rank-1 parents to mint fresh plastic capacity;
  sever slack→core connections at init (structural mask, Eq. 28); scale
  gradients by elasticity `g = E_out ⊙ ∇L` (Eq. 29). CIFAR-100→SVHN transfer:
  H-Score 65.82±3.96 vs Head-Only 45.79, Full-FT 13.88, EWC 12.54, PEFT 10.18
  [PAPER-MEASURED]; target acc 89.79 (vs 94.09 full-FT) with 52.14 source
  retention (vs 7.52).

## The frame that matters for us (one paragraph)

HOPE is not a codec paper — it is a **projection calculus**: put trained
weights in a function space whose metric is the expectation over the input
measure, quotient out every scale symmetry, then price every discrete
architectural move (prune / merge / evict) with a closed-form, O(1),
capacity-normalized distortion, and descend a rate-distortion knapsack
greedily. That is, almost verbatim, the shape of our open problem: we have
distortion SOLVED in scorer-output coordinates and are blocked at
**finite-price MATERIALIZATION in actuation coordinates** (routing card §5:
0/162 actionable same-object prices, 0/37 materialized MS4D buckets), with the
measured DIRECTION crux that all single-coordinate edges worsen joint S and
only composed moves open (card §§3–4). HOPE's answer to "why can't you price
actions?" is: you priced them in the wrong space, against the wrong measure,
without quotienting gauge — price them as operators against the input measure
and the prices come out closed-form. One structural advantage we hold that the
paper does not: **our input measure is not a surrogate** — the scorers' P_X is
the literal 600-pair set in custody, and the witness's P_X is a deterministic
coordinate grid — so HOPE's Gaussian max-ent scaffolding (their weakest
assumption) is unnecessary here; the operator calculus applies with EXACT
empirical kernels.

## Crosswalk table

| # | HOPE mechanism (label) | Our surface / crux | Verdict + consumer |
|---|---|---|---|
| 1 | Capacity-normalized pool cost `J_prune = N‖f‖/(E_a−‖f‖)` — cost of draining a pool diverges as the pool empties (PAPER-DERIVED, axiomatic) | `opportunity_pools_non_additive` LAW (REPO-MEASURED 07-18: same-pool levers COMPETE, never sum) + KKT waterfill (`src/tac/boundary_math/boundary_routing.py`) | **DERIVED convergence, upgradeable to apparatus.** HOPE gives our empirically-measured non-additivity a closed ANALYTIC form: pool competition = shrinking-denominator capacity. Candidate canonical-equation registration (`hope_pool_competition_capacity_denominator_v1`-shaped) as a PRIOR for waterfill step-pricing — measured prices remain authority. |
| 2 | Static-ΔP^init vs live-ΔP DR inflation (Eq. 38–39): pricing against a live shrinking budget breaks item independence, inflates neighbors' cost-rates, traps greedy in fragmented states (PAPER-DERIVED) | Our economics law: all distortion↔byte prices are **UPPER BOUNDS through the proposal-search channel** (REPO-MEASURED 07-24) + rate-law ladder PAYLOAD=SECTION-COST | **Named sibling confound.** Adopt as a checklist row for any ms2r/ms4d-class waterfill: price actions against a FROZEN footprint schedule; re-derive only after execution (receding horizon). CAVEAT (DERIVED): their uniform-scaling fairness argument assumes footprints scale uniformly — with context-adaptive entropy coders ours do NOT (removing a coefficient changes neighbors' coded size), so the fix here is measured-coder-bytes-per-section, not their α-scaling. |
| 3 | Merge as constrained rank-1 HS projection with closed-form parent (Eqs. 12–18) — a principled 2→1 **composed move** with exact physical-parameter recovery (PAPER-DERIVED) | DIRECTION crux (REPO-MEASURED, card §§3–5: 16/16 single-coordinate nulls `[0]`; ONLY joint/composed moves open, e.g. PC1 −2.761 S) | **The paper's deepest structural echo.** HOPE independently establishes that in a quotiented function space the productive move class is COMPOSED (2→1 projections), and singles (prune) are usually dominated when correlation exists. Consumer: descent-line finisher + witness-rate — a merge-move family over witness trunk neurons / carrier coefficients is a lawful generator of composed candidates where our sealed single-ray alphabet is exhausted (rg4: 25/25 typed exclusions, alphabet EXHAUSTED). CONJECTURE on effect size. |
| 4 | Data-free per-neuron capacity of a FROZEN BN network from its own checkpoint stats (Eqs. 1, 75, 79) (PAPER-DERIVED) | The scorers: SegNet efficientnet-b2 (BN-rich, frozen, in custody) + PoseNet FastViT-T12; SegNet head already factored EXACT rank-4 linear (REPO-MEASURED `segnet_head_rank4_linear_flipdist_v1`) | **Instrument for the FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK family** — one of the exactly-3 NEW coordinate families rg4's measured-hard wall demands (card §4: 9 of 25 blocks). HOPE-style per-channel capacity fields computed from SegNet's own BN buffers (or exactly over the n600 measure) give a principled, $0, closed-form site-local weighting for that codebook: which scorer channels carry capacity ⇒ which input-space sites can move argmax per stratum. Extends our margin-field Fisher surrogate (Pearson 0.978, REPO-MEASURED) one layer INTO the scorer, per-channel. DERIVED fit; unbuilt. |
| 5 | DEFT core/slack partition + structural mask + elasticity-scaled gradients (Eqs. 24–29) (PAPER-MEASURED: H-Score 65.82 vs 45.79 next-best) | The measured joint-spill regression: PC1's 4-step live continuation REGRESSED +0.127593 (REPO-MEASURED, card §3 — "live co-optimization from W_joint+PC1 is adverse"); the #383 terminal joint pose-finish gate | **Named candidate mechanism for the continuation failure.** If the regression is pose-descent corrupting seg-critical capacity (joint spill), DEFT's recipe — freeze high-capacity seg-core, route pose gradient into low-capacity slack, sever slack→core — is a $0-derivable gradient mask for the finisher. CONJECTURE (mechanism-fit unverified); falsifier: recompute the 4-step continuation with an elasticity mask; if it still regresses, the spill is not capacity-routed. |
| 6 | Scale-symmetry quotient: capacity ⊥ raw magnitude; dead neurons detected analytically (K(i,i)→0) (PAPER-DERIVED) | Null-subspace law (REPO-MEASURED 07-17: gauge = 52% of head-norm RATE-NEUTRAL; ker(A)≈52% scorer-invisible) | **Convergent — and a cheap pre-byte-close pass.** Same theorem from the other side: what the receiver/scorer cannot see must not be paid for. HOPE's per-neuron version prunes/merges witness gauge mass at ZERO distortion BEFORE quantization, shrinking the object the coder sees. Consumer: rate line, witness byte-close preflight. DERIVED. |
| 7 | Progressive encoding = anytime R-D curve over MODEL states; train big, project down along measured J (Sec. 9–11.1) (PAPER-DERIVED + plot) | TRAIN-LEAST / surgical Kolmogorov-projection doctrine (REPO binding 07-16) + the bc20-vs-bc36 capacity-rate trilemma (CLAUDE.md, measured) | **Doctrine-confirming with an algorithm attached.** The trilemma's resolution ("capacity WITHOUT scaling") gets a concrete ladder: train at d_seg-adequate capacity, HOPE-project down the R-D curve to the byte box, re-pricing each step through the frozen scorer (OUR oracle replaces their J as accept authority — their J is self-distortion, a surrogate in our terms). DERIVED composition; each rung must land as a measured n600 row. |
| 8 | O(1) decoupled cache: cache pair-scalars (a,b) that depend only on frozen weights; recompute only capacity-dependent s*, J from live E_rem (App. B.4) (PAPER-DERIVED) | ms2r-class waterfill preflights; the 0/162 finite-price wall (REPO-MEASURED card §5) | **Engineering pattern worth copying regardless of the rest.** Separate frozen-geometry scalars from live-budget scalars so every candidate price is an O(1) closed-form query, not a measurement. Our EV2 7-home stream-level price table (eureka memo A1) is the same shape; HOPE shows how to keep it EXACT under a shrinking budget. DERIVED. |
| 9 | Block eviction under one currency with `E_identity = Σ√(γ²+β²)` parallel survival capacity (Eqs. 19–20) (PAPER-DERIVED) | Macro-vs-micro action unification in our rate ladder (drop whole carrier/section vs quantize one coefficient) | **Meta-lesson adopted, formula not.** Our archive has no skip connections; the transferable content is the axiom that macro deletions must compete in the SAME normalized currency as micro moves, with the surviving parallel description playing E_identity's role (e.g. the generic inflate.py generator is the "skip" that survives when a learned section is evicted). CONJECTURE-grade analogy; useful as waterfill design language only. |
| 10 | Gaussianity of pre-activations via CLT + Diaconis-Freedman 2-D slice argument (Sec. 4, App. E.1) (PAPER-DERIVED) | Our Fisher/margin surrogate discipline (frozen-scorer Fisher metric; margin↔curvature Pearson 0.978 REPO-MEASURED) | **A justification we can borrow, and mostly don't need.** For the scorers, exact n600 expectations are computable — no surrogate required (our advantage over the paper's own setting). Where a cheap analytic pass is wanted (e.g. sweeping hypothetical carriers without decoding), the D-F argument bounds when Gaussian-surrogate kernels are faithful. DERIVED. |

## Tensions / negatives (signal, stated plainly)

1. **Their distortion is not our distortion.** J prices SELF-distortion of the
   compressed network in its own function space. Our authority is
   S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489 through frozen third-party
   scorers on exact archive bytes. Any HOPE-derived ordering is a
   surrogate/prior (NO-FAKE class 8); the accept authority stays the n600
   frozen-scorer verdict (pc2-style accept-loop), and the exact pointer moves
   only through `upstream/evaluate.py`. This is a role assignment, not a
   rejection — HOPE proposes, our oracle disposes.
2. **PH-1 restriction does not cover our witness.** The closed-form kernels
   assume positively-homogeneous activations (ReLU family). The witness runs
   sin/step_basis/annealed-hosc (verified live in
   `experiments/train_levelset_witness_realized_through_R_mlx.py`) — none
   PH-1, so Eqs. 3–5 do NOT transfer as-is. However (DERIVED, cheap owed
   derivation if adopted): under a Gaussian pre-activation, the step/indicator
   activation's kernel is the arc-cosine ORDER-0 kernel (bivariate-normal
   orthant probability — closed form), and E[sin(y_i)sin(y_j)] is closed-form
   via characteristic functions. The kernel TABLE extends to our basis; and
   with the exact grid measure, empirical kernels need no closed form at all.
3. **Data-free-ness is their headline, and moot for us.** Their max-ent
   Gaussian exists because they lack data. We hold the exact input measures
   (600 pairs; deterministic coordinate grid). Adopting HOPE here means
   adopting the OPERATOR CALCULUS with exact kernels, not the surrogate — a
   strictly stronger position than the paper's.
4. **Effect sizes are not quotable.** The compression result is a plot
   (no numbers); DEFT's numbers are CIFAR-100→SVHN scale. Nothing here
   licenses a predicted-ΔS band for us (Catalog #296/#324 discipline: any band
   would need our own Dykstra-feasibility/measured anchor).
5. **Param-count rate ≠ our rate.** Their ΔP is parameter count under
   fixed-precision; our counted rate is entropy-coded section bytes
   (PAYLOAD=SECTION-COST ladder). The DR denominator must be measured coder
   bytes per action; their uniform-scaling fairness argument (Eq. 39) can fail
   under context-adaptive coding (see row 2). Anyone importing Eq. 23 verbatim
   imports a confound.
6. **No code.** No repository is referenced; all algorithms would be
   reimplemented from Eqs. 1–23 + App. B/E (they are complete enough to
   reimplement — the paper is unusually self-contained).
7. **Banned-lineage check:** clean. Nothing here touches HNeRV/PR95/110/128
   vehicles or the borrowed-incumbent bank; HOPE is method-math, consumed as
   lessons/apparatus only, on OUR vehicles.

## Ranked takeaways

1. **[DERIVED, apparatus-grade] Price actions as operators against the exact
   input measure, with gauge quotiented — the materialization wall may be a
   coordinates problem, not a physics problem.** The describe line's 0/162
   finite-price wall (card §5) demands prices in actuation coordinates; HOPE
   demonstrates a complete pricing stack (capacity norm + pool denominator +
   decoupled cache + DR greedy) in which every price is a closed-form query.
   Consumer: ms2r/ms4d waterfill re-pose (with measured-coder-byte
   denominators per tension 5); the EV2 7-home stream-level table is the
   nearest existing surface.
2. **[DERIVED fit, unbuilt] HOPE-style per-channel capacity of SegNet from its
   own frozen BN buffers = a principled generator for the
   FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK coordinate family** — one of
   the exactly-3 new families rg4's measured-hard wall (25/25 typed
   exclusions) says the top24 blocks demand. $0, analytic, no forward passes;
   composes with the exact rank-4 head factorization we already hold. This is
   the single most concrete new-build this paper suggests.
3. **[CONJECTURE, cheap falsifier] DEFT elasticity masks against the measured
   PC1 continuation regression (+0.1276).** Partition the vehicle into
   seg-core/pose-slack by analytic capacity, sever slack→core, rerun the
   4-step continuation. If regression persists, joint spill is not
   capacity-routed — either way the DIRECTION crux gains a measured bit.
4. **[DERIVED convergence] Two of our hard-won empirical laws fall out of
   HOPE's axioms as theorems:** non-additive pool competition (= the
   1/(E_a−‖f‖) denominator) and rate-neutral gauge mass (= the scale-symmetry
   quotient; our ker(A)≈52%). Register the analytic forms as canonical-equation
   PRIORS; keep measured prices as authority. Independent-derivation
   convergence is evidence the underlying geometry is real.
5. **[DERIVED doctrine + algorithm] Train-big-then-project gets an explicit
   ladder:** HOPE's progressive encoding is the algorithmic form of the
   TRAIN-LEAST/Kolmogorov-projection doctrine — an anytime R-D curve over
   model states with composed (merge) moves, priced per step, greedy under a
   frozen footprint schedule, with OUR frozen-scorer oracle substituted as the
   accept authority. Long-horizon consumer: witness/carrier byte-close
   pipeline whenever a trained vehicle next approaches the box.

## Honest bottom line

No purchase on the next exact row directly — HOPE moves no bytes and scores
nothing of ours. Its value is that a strong independent group, axiomatizing
"deconstruct a trained network under a frozen budget," arrived at the SAME
structural conclusions our receipts forced on us this month (quotient gauge
before pricing; pools compete by shrinking denominator; singles stall,
composed moves open; price in function space against the input measure), and
ships closed-form machinery for the one thing we are measurably missing:
cheap, lawful, per-action prices in actuation coordinates. Rows 1–3 name the
consumers; each lands only as a measured n600/exact row per standing law.

## STORES CONSULTED

`.omx/research/council_coherent_optimal_path_routing_20260725.md` (§1–§5: the
two-line map, D1–D4, j12 §3 measured nulls + PC1 −2.761/+0.1276, rg4 §4
25-block wall + 3 named coordinate families, §5 materialization 0/162) ·
`.omx/research/fable_eureka_hunt_tier_breakthrough_20260725.md` (§0–§5; A1
7-home price table, A3 pose descent door) · CLAUDE.md (THE GOAL · witness
capstone §§ · rule-118 boundary · NO-FAKE classes · trilemma · Pose-solved
clarification) · MEMORY.md index + memories recalled: `opportunity_pools_non_
additive_rate_distortion_reachable_20260718` · `null_subspace_rate_measure_
20260717` · `distortion_byte_economics_are_upper_bounds_20260724` ·
`train_least_surgical_kolmogorov_projection_realization_doctrine_20260716` ·
`meet_it_where_it_is_carry_thing_itself_smallest_basis_n600_20260721` ·
`no_old_lineage_ban_hnerv_pr_substrates_20260723` · `borrowed_incumbent_rate_
polish_permanently_dead_20260725` · canonical equation `segnet_head_rank4_
linear_flipdist_v1` (recalled) · house style from `.omx/research/arxiv_2607_
16035_crosswalk_20260720T182136Z.md` · live-repo verifications: witness
activations (hosc/step in `experiments/train_levelset_witness_realized_
through_R_mlx.py`), `src/tac/boundary_math/` surface listing · paper PDF
(sha above), read boundary as stated.
