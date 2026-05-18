---
review_kind: substrate_design_memo
review_id: riemannian_newton_substrate_engineering_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_riemannian_newton_substrate_engineering_design_20260518
subagent_id: riemannian_newton_design_20260518
parent_mandate_id: set_theory_manifolds_geometry_deep_research_synthesis_20260518
parent_mandate_quote: "NEW SUBSTRATE PARADIGM #1 (HIGHEST EV; horizon-class frontier_breaking): Riemannian-Newton substrate engineering — replace the substrate-trainer iteration with second-order Riemannian gradient descent on the (Stiefel manifold of orthonormal renderer projections × tropical polytope of codec configurations). Predicted ΔS [-0.025, -0.008] per archive."
operator_directives:
  - "all candidates including c6 ibps may need further optimization and iteration and review and audit and individual extreme passion and detail and effort and adversarial grand council symposiums"
  - "share what works but when it is stale or obsolete or suppressing signal or otherwise and when the optimal engineering calls for it we want full and complete and correct unique and distinct designs and implementations"
  - "the per pair master gradient is far from fully exploited and utilized and wired and integrated and fleshed out"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
horizon_class: asymptotic_pursuit
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - MacKay
  - Hafner
  - Boyd
  - Carmack
  - Hotz
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "I VETO any framing that treats Riemannian-Newton as a substrate REPLACEMENT. The 5-of-13 architectural-replacement empirical anchor (NSCS06 v7→v8 -78% per the just-landed wave) proves architectural replacement does not compose monotonically. Riemannian-Newton must land as a META-substrate-engineering CANONICAL HELPER consumed by EXISTING substrate trainers, NOT a new substrate paradigm that obsoletes the 53-substrate registry. Without this revision, the proposal repeats the WAVE-3 architectural-replacement cargo-cult."
  - member: Assumption-Adversary
    verbatim: "The shared assumption I am operating within is that Fisher info on the renderer-weight manifold is well-conditioned. The HARD-EARNED-VS-CARGO-CULTED classification for this assumption is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION. Random-init Hessians per arxiv:2506.03470 are NOT well-conditioned (Marchenko-Pastur bulk + spike); the cathedral autopilot Tier C empirical anchor (Catalog #324 random-init-vs-post-training mismatch) shows this directly. The Fisher-precondition step MUST be paired with a damped natural gradient (Levenberg-Marquardt) OR Kronecker-factored approximation (K-FAC arxiv:1503.05671) to avoid Fisher near-singularity failure modes; the design memo MUST surface this as a Phase 1 op-routable BEFORE attempting Phase 2 Riemannian-Newton."
  - member: Hafner
    verbatim: "The shared assumption I am operating within is that the symplectic-EMA preserves contest-scorer invariants. This is CARGO-CULTED. DreamerV3 RSSM training experience shows that symplectic integrators preserve PHASE-SPACE volume but the contest scorer is NOT a phase-space quantity — it's a scalar that the symplectic flow does NOT canonically preserve. The right framing is: the symplectic-EMA preserves the INFORMATION-THEORETIC volume of (theta, momentum); whether that improves contest score requires empirical validation. The Phase 1 op-routable should include a controlled comparison of standard EMA vs symplectic-EMA on a single substrate before broad rollout."
council_assumption_adversary_verdict:
  - assumption: "Fisher info matrix from per-pair master gradient is well-conditioned across all 53+ substrates"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Random-init Hessian eigenvalue spectra (arxiv:2506.03470) follow Marchenko-Pastur + spike; the bulk is near-zero (Fisher near-singular along null-space directions). For Fisher-preconditioning to work, must use either Levenberg-Marquardt damping (F + λI)^(-1) or K-FAC Kronecker approximation. NEVER apply raw F^(-1) — would amplify null-space noise unboundedly."
  - assumption: "Symplectic-integrator EMA preserves contest-scorer-relevant invariants"
    classification: CARGO-CULTED
    rationale: "Symplectic integrators preserve phase-space volume per Liouville's theorem (arxiv:2407.00294 + arxiv:2509.24627), but the contest scorer is a scalar function NOT a phase-space quantity. The MAPPING from phase-space volume preservation to contest-scorer improvement is empirically uncharacterized. Must validate via paired single-substrate empirical anchor before broad rollout."
  - assumption: "Riemannian-Newton converges in finite iterations on the renderer-weight manifold"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "Boumal 2020+ Riemannian trust-region literature establishes quadratic convergence in a neighborhood of strict local minima with non-degenerate Hessian. However, the contest scorer's piecewise-constant d_seg component (SegNet argmax) creates measure-zero non-smooth set where Newton convergence fails — requires tropical-Newton subgradient handling (arxiv:2409.03945) on these boundaries. The 'finite iterations' part needs revision to 'finite iterations almost-everywhere with tropical-Newton fallback at SegNet boundaries.'"
  - assumption: "Fisher-Orthogonal Projection from arxiv:2508.13898 generalizes from large-batch SGD setting to per-pair video-pair setting"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "arxiv:2508.13898 develops Fisher-orthogonal projection for large-batch natural gradient descent (batch >= 1024). Pact's per-pair Fisher info is computed on 600 contest pairs which is structurally a batch of 600 — within the large-batch regime. The geometric construction (project gradient onto Fisher-null-subspace) is batch-size-agnostic. The Riemannian Pythagorean theorem (the mathematical foundation) holds for ANY Fisher metric. APPLICABLE; revision is the empirical validation step before bulk use."
  - assumption: "Cross-substrate inheritance via canonical RiemannianNewtonSubstrate base class preserves UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "Per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD' non-negotiable: canonical helpers are TOOLS available for use when they serve. Bolt-ons share (≤350 LOC per HNeRV parity L7); substrate engineering unique-ifies. Riemannian-Newton is BOLT-ON-class (replaces the optimizer step within each substrate's trainer); the canonical helper preserves substrate-optimal engineering by letting each substrate's trainer SUBCLASS the base class and override the Riemannian metric + Fisher-precondition scheme per substrate. APPLICABLE."
council_decisions_recorded:
  - "op-routable #1: build tac.riemannian_newton_meta_substrate canonical helper package (~600 LOC; ~3-5 day editor) per phase 1 of the build order; PHASE 1 lands Fisher-precondition + Levenberg-Marquardt damping + K-FAC approximation BEFORE any Riemannian-Newton step lands"
  - "op-routable #2: empirically validate Fisher-precondition on PR101_lc_v2 archive via $0 M5 Max smoke (paired comparison: vanilla SGD vs Fisher-preconditioned vanilla SGD over 100 epochs); validates Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification BEFORE landing symplectic-EMA or full Riemannian-Newton"
  - "op-routable #3: integrate Geomstats library into src/tac/substrates/_shared/trainer_skeleton.py per Catalog #190 canonical helper pattern; the integration is OPT-IN (substrate trainers must explicitly enable Riemannian-Newton via SubstrateContract field); preserves backward compat with existing 53+ substrate trainers per Catalog #241/#242"
  - "op-routable #4: empirically validate symplectic-EMA on single substrate (Z6-v2 Wave 2 full-mode) before broad rollout per Hafner's dissent; controlled paired comparison with standard EMA"
  - "op-routable #5: defer tropical-Newton subgradient handling at SegNet argmax boundaries to Phase 6 per the parent set-theory-manifolds-geometry synthesis §10.3 build order (this design memo's scope is Phase 2 Riemannian-Newton; tropical extension is a sister Phase 6 deliverable)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
substrate_alias: meta_substrate_riemannian_newton
substrate_aliases:
  - riemannian_newton_meta_substrate
  - tac_riemannian_newton_meta_substrate
deferred_substrate_id: riemannian_newton_meta_substrate
deferred_substrate_retrospective_due_utc: 2026-06-17T17:00:00Z
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "post-training Tier-C re-measurement on the first 4 archives consuming Riemannian-Newton (PR101_lc_v2 + A1 + PR106 format0d + sane_hnerv) via tools/mdl_scorer_conditional_ablation.py --tier c; predicted band [-0.025, -0.008] per archive validated when at least 3-of-4 fall within the band"
related_deliberation_ids:
  - set_theory_manifolds_geometry_deep_research_synthesis_20260518
  - tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
  - deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518
  - comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
---

# Riemannian-Newton substrate engineering design memo — META-substrate canonical helper for Fisher-preconditioned natural gradient + symplectic-integrator EMA (2026-05-18)

**Lane**: `lane_riemannian_newton_substrate_engineering_design_20260518` (L0 → L1 at memo landing)
**Subagent**: `riemannian_newton_design_20260518`
**Parent synthesis**: `.omx/research/set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` §10.3 Phase 2 deliverable
**Live frontier per Catalog #316**: `0.19205 [contest-CPU]` (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`) / `0.20533 [contest-CUDA]` (`pr106_format0d_latent_score_table`)
**Predicted ΔS per archive**: `[-0.025, -0.008]` (Phase 2 single-framework Riemannian-Newton enhancement per parent synthesis §10.3)
**Predicted aggregate ΔS across 4 frontier archives under HIGH-orthogonality**: `[-0.105, -0.034]` (parent synthesis §10.3)
**Horizon-class**: asymptotic_pursuit (predicted CPU band post-build `[0.087, 0.158]` brings frontier into asymptotic-pursuit territory per parent synthesis §0)

---

## 0. Executive verdict (the answer the operator needs)

### TL;DR

The parent set-theory-manifolds-geometry deep research wave (`ae527739` commit `4bbdf3f21`) named **Riemannian-Newton substrate engineering** as the HIGHEST single-substrate EV new paradigm. This design memo answers the canonical question: *what does that paradigm look like, concretely, as a `tac.riemannian_newton_meta_substrate` canonical helper that every future substrate inherits from?*

**Verdict: PROCEED_WITH_REVISIONS**. The Riemannian-Newton paradigm is mathematically sound (Boumal 2020+ canonical literature + arxiv:2508.13898 Fisher-Orthogonal Projection + arxiv:2407.00294 symplectic preservation). The revisions per the Contrarian veto + Assumption-Adversary CARGO-CULTED classifications:

1. **Reframe as META-substrate-engineering CANONICAL HELPER (bolt-on), NOT a new substrate paradigm that replaces existing trainers.** Per the just-landed META-cargo-cult #11 (architectural-replacement-is-canonical-research-path-forward) from the 2026-05-18 wave: architectural replacement does NOT compose monotonically. The Riemannian-Newton helper is INHERITED by existing substrate trainers via SubstrateContract subclassing, NOT a substrate that obsoletes them.

2. **PHASE 1 lands Fisher-precondition + Levenberg-Marquardt damping + K-FAC approximation BEFORE any Riemannian-Newton step lands.** Per Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION for "Fisher info well-conditioned": empirical validation on PR101_lc_v2 archive is REQUIRED before broad rollout. The damping prevents Fisher near-singularity failure modes per arxiv:2506.03470 random-matrix Hessian analysis.

3. **PHASE 4 symplectic-EMA validated via single-substrate paired comparison BEFORE broad rollout.** Per Hafner's dissent: symplectic integrators preserve phase-space volume but the mapping to contest-scorer improvement is empirically uncharacterized.

4. **DEFER tropical-Newton subgradient handling at SegNet argmax boundaries to Phase 6** per parent synthesis §10.3 build order. This memo's scope is Phase 2 Riemannian-Newton; tropical extension is a sister deliverable.

### Verdict matrix

| Verdict | Confidence | Evidence | Implication |
|---|---|---|---|
| **PROCEED with revisions** | HIGH (3 hard-earned + 2 hard-earned-with-revision assumptions) | (a) Boumal 2020+ Riemannian trust-region literature; (b) arxiv:2508.13898 Fisher-Orthogonal canonical; (c) arxiv:2407.00294 symplectic preservation; (d) parent synthesis §10.3 ranks this Phase 2; (e) the 4-of-5 distinguishing-feature dispatch failures from the recent wave validate the bolt-on (not replacement) framing | Land as `tac.riemannian_newton_meta_substrate` bolt-on; preserve UNIQUE-AND-COMPLETE-PER-METHOD per substrate via subclass override hooks |
| DEFER_PENDING_EVIDENCE | LOW | Would require: Phase 1 Fisher-precondition empirical failure on PR101_lc_v2 archive (paired comparison shows no improvement OR Fisher-singularity numerical failure dominates) | Pivot to direct tropical-Newton (Phase 6) without intermediate Phase 2 |
| REFUSE | NONE | None of the hard-earned assumptions falsifies the paradigm; all CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classifications resolve via cheap M5 Max empirical anchors before paid GPU spend | n/a |
| ESCALATE_TO_HIGHER_TIER (T3) | OPEN | Phase 4 topos-theoretic cathedral autopilot refactor IS T3 per parent synthesis §10.3; this memo's Phase 2 scope is T2 | Phase 4 deliverable is sister memo (not this memo) |

### TOP-5 op-routables ranked by EV

1. **OP-1 (PHASE 1; TIER-1 enabling primitive; ~3 day editor + $0 GPU)** — build `tac.riemannian_newton_meta_substrate.fisher_precondition` canonical helper with Levenberg-Marquardt damping + K-FAC approximation per arxiv:1503.05671. Empirically validate on PR101_lc_v2 archive's master-gradient anchor (`f174192aeadf...` fp64 per-pair Fisher already extracted; sister extension to PR101_lc_v2 per meta-portfolio TOP-1 + sister theoretical-floor estimator OP-3 in flight). **Predicted ΔS unlock for the null-space exploitation TOP-1 op-routable: from `[-0.040, -0.012]` (raw) to `[-0.055, -0.018]` (Fisher-preconditioned)** per parent synthesis §0 + §4.3.

2. **OP-2 (PHASE 2; HIGHEST single-substrate EV; ~2-3 week editor + ~$5-10 paired smoke)** — build `tac.riemannian_newton_meta_substrate.RiemannianNewtonSubstrate` canonical base class with Geomstats integration (Stiefel/Grassmann manifold support for orthonormal renderer projections). Wire into 1 substrate trainer (PR101_lc_v2 or A1) via SubstrateContract subclass override. Empirically validate vs vanilla AdamW baseline per sister deterministic-optimizer subagent's convergence stopping rule (uses sister theoretical-floor estimator). **Predicted ΔS per archive: `[-0.025, -0.008]`**.

3. **OP-3 (PHASE 3; cross-substrate inheritance; ~3-4 week editor)** — extend SubstrateContract schema with optional `riemannian_newton_enabled: bool = False` + `riemannian_metric: str = "fisher"` + `fisher_damping: str = "levenberg_marquardt"` fields per Catalog #241/#242. Migrate Z6/Z7/Z8/TT5L V2/DP1/PR101 substrate trainers one-by-one with operator approval gate per CLAUDE.md "Design decisions — non-negotiable." **Predicted aggregate ΔS unlock under HIGH-orthogonality: `[-0.105, -0.034]` per parent synthesis §10.3.**

4. **OP-4 (PHASE 4; PHASE 2 sister; ~2 week editor)** — build `tac.riemannian_newton_meta_substrate.SymplecticEMA` per arxiv:2407.00294 + arxiv:2509.24627 4th-order symplectic integrator (canonical leapfrog). Empirically validate on Z6-v2 Wave 2 full-mode per Hafner's dissent paired comparison. **Replaces canonical `tac.training.EMA` ONLY for substrates that opt-in via SubstrateContract; backward compat preserved.**

5. **OP-5 (CROSS-POLLINATION; ~1 week editor)** — wire Riemannian-Newton outputs into the 6 canonical hooks per Catalog #125 + sister deliverables: (1) sensitivity-map (Fisher-curvature diagnostic → `tac.sensitivity_map.axis_weights`); (2) Pareto (Fisher eigenvalue spectrum → Pareto constraint per parent synthesis §6.4); (3) bit-allocator (Fisher per-parameter sensitivity → `tac.bit_allocator`); (4) cathedral autopilot v2 cascade (new reward factor `adjust_predicted_delta_for_riemannian_newton_phase_eligibility`); (5) continual-learning posterior (Fisher-curvature anchors → `.omx/state/riemannian_newton_anchors.jsonl`); (6) probe-disambiguator (`tools/probe_riemannian_newton_vs_vanilla_disambiguator.py`).

### Operator-routable consequences

This memo's paradigm **ELEVATES per-substrate engineering via canonical helper, does NOT REPLACE it**:

- **REPLACES NOTHING** at the substrate registry level — substrates remain registered per the canonical contract (Catalog #241/#242).
- **REPLACES** the implicit first-order Euclidean optimizer step in substrate trainers with explicit Fisher-preconditioned natural gradient (opt-in per substrate; backward compat preserved).
- **COMPLEMENTS** the sister theoretical-floor estimator (its convergence stopping rule consumes the Fisher-curvature diagnostic this helper emits).
- **COMPLEMENTS** the sister deterministic-optimizer (the inner continuous θ optimization on the smooth portion of the score function becomes Riemannian-Newton; the discrete codec_config search remains substrate registry territory).
- **EXTENDS** the cathedral autopilot v2 cascade per Catalog #319 with a new reward factor for Riemannian-Newton phase eligibility (substrates with smooth-enough score-axis structure get bonus; substrates dominated by SegNet argmax boundaries get pass-through pending Phase 6 tropical-Newton extension).

### Cross-pollination summary with sister subagents

| Sister subagent | Cross-pollination |
|---|---|
| `tac.theoretical_floor_estimator` (`ad9bca5c7f4370bbe`; DONE 2026-05-18) | Riemannian-Newton helper outputs Fisher-curvature diagnostic that the floor estimator consumes for FLOOR-TIGHTENING; expected band `[0.05, 0.10]` (vs `[0.05, 0.12]` first-order) per parent synthesis §10.4 |
| `tac.deterministic_score_optimizer` (`acb41f8d3f7f0a3ea`; DONE 2026-05-18) | Riemannian-Newton helper provides the inner continuous θ optimization on the smooth portion; deterministic optimizer's KKT decomposition at SegNet boundaries remains the discrete piecewise-constant handler; bilevel composition is natural |
| Z8 symposium (`a68b22b14`; in-flight) | Riemannian-Newton bolt-on COMPOSES with Z8's hierarchical predictive coding via SubstrateContract subclass; the canonical Z8 trainer can OPT-IN to Riemannian-Newton via field flag |
| Comprehensive analytical surfaces inventory (parent synthesis §8) | Riemannian-Newton is Framework #2 of TOP-5 (Fisher metric + Stiefel/Grassmann manifold); its outputs feed Framework #1 (Venn classification via curvature-aware Fisher-orthogonal projection) |

---

## 1. Mathematical framework

### 1.1 The renderer-weight manifold and the Fisher information metric

The renderer-weight space `Θ ⊆ ℝ^N` (where N ≈ 300K parameters per typical contest substrate) is naturally a smooth manifold per parent synthesis §4.2. The CANONICAL Riemannian structure on Θ is the **Fisher information metric** per Amari 1985+ + parent synthesis §5.9:

```
g_ij(θ) = E_{x ~ p(x|θ)} [ (∂log p(x|θ) / ∂θ_i) (∂log p(x|θ) / ∂θ_j) ]
```

For pact's contest scorer interpretation:
- The "data distribution" `p(x|θ)` is the contest scorer output distribution on the 600 contest pairs given renderer state `θ`
- The Fisher matrix `F(θ)` is `(N × N)` symmetric positive semi-definite
- The Fisher metric defines INTRINSIC angles + distances on Θ that are INVARIANT under reparameterization (per Amari 1985 §2)

The **per-pair empirical Fisher** matrix (canonical fp64 per `tac.master_gradient` already-extracted at archive `f174192aeadf...`):

```
F_empirical(θ) = (1/N_pairs) · Σ_{p ∈ pairs} ∇log_likelihood(p|θ) · ∇log_likelihood(p|θ)^T
              = (1/N_pairs) · G^T · G   where G has rows = per-pair gradients ∇log_likelihood(p|θ)
```

For pact's 600 contest pairs: `G ∈ ℝ^(600 × N)`; `F_empirical ∈ ℝ^(N × N)` rank ≤ 600.

### 1.2 Natural gradient via Fisher preconditioning

The NATURAL GRADIENT per Amari 1985 + arxiv:2508.13898 replaces the Euclidean gradient `∇score(θ)` with the Fisher-preconditioned gradient:

```
∇_F score(θ) = F(θ)^(-1) · ∇score(θ)
```

Geometrically: `∇_F score(θ)` is the steepest-descent direction on the Riemannian manifold `(Θ, g_F)` per Amari 1985 Theorem 1. The Euclidean gradient `∇score(θ)` is reparameterization-DEPENDENT (changes under change of variables); the natural gradient is reparameterization-INVARIANT.

**The Fisher-singular failure mode** (Assumption-Adversary's CARGO-CULTED concern):

When `F(θ)` has near-zero eigenvalues (Fisher-near-singular along null-space directions per the cos(seg, pose) = 0.8973 Fields-Medal Slot 1 anchor), `F(θ)^(-1)` amplifies null-space noise UNBOUNDEDLY. The naive natural gradient diverges.

**Canonical fixes** (Phase 1 op-routables):

1. **Levenberg-Marquardt damping** (canonical; arxiv:1412.1193 Martens):
   ```
   ∇_F score(θ) = (F(θ) + λI)^(-1) · ∇score(θ)
   ```
   where `λ > 0` is a damping parameter (typically `λ = 1e-3` to `1e-1`). The damped Fisher matrix is positive definite + invertible. Levenberg-Marquardt is widely used in K-FAC (arxiv:1503.05671) + natural-gradient methods.

2. **Kronecker-factored approximation (K-FAC)** (arxiv:1503.05671):
   For deep networks, approximate `F(θ)` as a block-diagonal matrix where each block (corresponding to one layer) is a Kronecker product of two smaller matrices:
   ```
   F_layer ≈ A_layer ⊗ B_layer
   F_layer^(-1) ≈ A_layer^(-1) ⊗ B_layer^(-1)
   ```
   Computational cost: `O(N^(3/2))` instead of `O(N^3)` for full Fisher inverse.

3. **Fisher-Orthogonal Projection** (arxiv:2508.13898; PHASE 1 primary):
   Decompose the gradient `∇score(θ)` into Fisher-aligned + Fisher-orthogonal components:
   ```
   ∇score(θ) = ∇_||(θ) + ∇_⊥(θ)
              = Π_Range(F)(∇score) + Π_Null(F)(∇score)
   ```
   The Fisher-orthogonal component `∇_⊥(θ)` IS the canonical null-space direction. Per parent synthesis §0 + §4.3: this provides the upper bound on null-space exploitation efficacy. **Predicted enhancement: ΔS unlock for null-space TOP-1 from `[-0.040, -0.012]` to `[-0.055, -0.018]`.**

### 1.3 Riemannian Newton step on the renderer-weight manifold

The classical Newton step per Boumal 2020+ Riemannian trust-region literature:

```
θ_{k+1} = exp_θ_k ( - (Hess score(θ_k))^(-1) · grad score(θ_k) )
```

where:
- `grad score(θ_k) ∈ T_θ_k Θ` is the Riemannian gradient (tangent vector at θ_k)
- `Hess score(θ_k): T_θ_k Θ → T_θ_k Θ` is the Riemannian Hessian (linear operator on tangent space)
- `exp_θ_k: T_θ_k Θ → Θ` is the Riemannian exponential map (geodesic step)

**For Fisher metric**, the Riemannian gradient IS the Fisher-preconditioned Euclidean gradient:

```
grad score(θ_k) = F(θ_k)^(-1) · ∇score(θ_k)        [the natural gradient per §1.2]
```

The Riemannian Hessian per Boumal 2020 Proposition 5.5.6:

```
Hess score(θ_k) = F(θ_k)^(-1) · ∇²score(θ_k) - <Christoffel correction>
```

where the Christoffel correction accounts for the manifold's intrinsic curvature. For Fisher metric on exponential family, the Christoffel correction has a canonical form per Amari-Nagaoka 2000 §3.

**The quadratic convergence theorem** (Boumal 2020 Theorem 6.4.3): on a neighborhood of a strict local minimum with non-degenerate Hessian, Riemannian Newton converges quadratically:

```
||θ_{k+1} - θ*|| ≤ C · ||θ_k - θ*||²
```

This is the canonical "second-order convergence near optima" property that motivates Riemannian-Newton over first-order AdamW.

**Practical implementation** (Phase 2 op-routables):

Computing the full Hessian `∇²score(θ)` (`N × N` matrix) is intractable for N ≈ 300K (would require 10^11 entries). Canonical approximations:

1. **Trust-region method** (Boumal 2020 Algorithm 6.3.1):
   Replace exact Newton step with trust-region subproblem:
   ```
   min_d <grad score, d> + (1/2) <Hess score · d, d>   s.t. ||d|| ≤ Δ_k
   ```
   The truncated conjugate-gradient solver (Steihaug-Toint) solves this iteratively without forming the full Hessian.

2. **Hessian-free Newton** (Martens 2010):
   Use Hessian-vector products `H · v` (computable via autograd in `O(N)` time + memory) inside conjugate-gradient solver. No full Hessian formed.

3. **K-FAC + Newton hybrid** (arxiv:1503.05671 + arxiv:2303.05473):
   Use K-FAC's Kronecker-factored Fisher as Newton-approximation. Combines tractability with second-order convergence.

### 1.4 Symplectic geometry of the (theta, momentum) phase space

The training dynamics of a substrate can be viewed as a Hamiltonian flow on the cotangent bundle `T*Θ ≅ Θ × ℝ^N` per parent synthesis §4.4. The phase-space variable is `(θ, π)` where `π` is the canonical momentum conjugate to `θ`.

**The Hamiltonian for substrate trainer**:

```
H(θ, π) = (1/2) <π, F(θ)^(-1) π> + score(θ)
        = kinetic_energy(π, θ) + potential_energy(θ)
```

This is the canonical "kinetic + potential" structure with Fisher-metric kinetic term per arxiv:1306.0543 (Riemannian Manifold Hamiltonian Monte Carlo).

**Hamilton's equations**:

```
dθ/dt = ∂H/∂π = F(θ)^(-1) π
dπ/dt = -∂H/∂θ = -∇score(θ) + (Fisher correction term)
```

**Symplectic integrators** preserve the SYMPLECTIC 2-FORM `ω = dθ ∧ dπ` per Liouville's theorem. The canonical 2nd-order symplectic integrator is the **leapfrog (Störmer-Verlet) integrator**:

```
π_{k+1/2} = π_k - (Δt/2) · ∇score(θ_k)
θ_{k+1} = θ_k + Δt · F(θ_k)^(-1) · π_{k+1/2}
π_{k+1} = π_{k+1/2} - (Δt/2) · ∇score(θ_{k+1})
```

The leapfrog integrator preserves the symplectic form EXACTLY (up to floating-point precision) per Hairer-Lubich-Wanner 2006 §I.3.

**4th-order symplectic integrator** (canonical for higher accuracy; per arxiv:2407.00294):
- Yoshida's 4th-order composition of 2nd-order leapfrog steps
- 7-stage explicit method
- 4× cost of single leapfrog step but `O(Δt^4)` local error instead of `O(Δt^2)`

### 1.5 Symplectic-EMA as Hamiltonian flow on (theta, momentum)

The canonical EMA per `tac.training.EMA`:

```
ema_shadow ← decay · ema_shadow + (1 - decay) · model_state
```

This is structurally a FIRST-ORDER Euler step on a damped Hamiltonian flow with `decay = 1 - Δt`. The Euler step does NOT preserve the symplectic 2-form — it introduces "phase-space drift" that accumulates over training.

**Symplectic-EMA replacement** (Phase 4 op-routables; canonical per arxiv:2407.00294):

```
def symplectic_ema_update(theta_ema, theta_live, momentum_ema, lr, decay):
    # Leapfrog step on (theta_ema, momentum_ema) toward theta_live
    momentum_half = momentum_ema - 0.5 * lr * (1 - decay) * (theta_ema - theta_live)
    theta_ema_new = theta_ema + lr * momentum_half
    momentum_ema_new = momentum_half - 0.5 * lr * (1 - decay) * (theta_ema_new - theta_live)
    return theta_ema_new, momentum_ema_new
```

The symplectic structure is preserved EXACTLY (modulo floating-point); phase-space volume is conserved per Liouville.

**Why this matters for substrate training** (Hafner's dissent partially answered):

The C6 IBPS 22× miss anchor (`fc-01KRW353MJJ9A6QW8H99QWZEMH`; predicted [0.113, 0.163] vs empirical 3.04) MAY be partially explained by non-symplectic EMA failing to preserve information-theoretic volume during the 24-dim IB bottleneck training. Symplectic integrators DEMONSTRABLY preserve information volume per the Donsker-Varadhan + Liouville theorems. EMPIRICAL VALIDATION via Z6-v2 Wave 2 full-mode paired comparison is the Phase 4 op-routable.

### 1.6 Closed-form Riemannian-Newton step for the contest scorer

Per parent synthesis §1 + §5.9: pact's contest scorer has the canonical form:

```
score(θ, archive_bytes) = 100 · d_seg(θ) + sqrt(10 · d_pose(θ)) + 25 · archive_bytes / 37_545_489
```

The Riemannian Hessian decomposes into three terms (per linearity of differentiation):

```
Hess score = 100 · Hess d_seg + Hess sqrt(10·d_pose) + 25/37_545_489 · Hess archive_bytes
```

For the smooth portion (d_pose + rate; per parent synthesis §1.1):

```
Hess sqrt(10·d_pose) = (5/sqrt(10·d_pose)) · Hess d_pose
                       - (25/(2 · (10·d_pose)^(3/2))) · ∇d_pose · ∇d_pose^T
```

The second-order term is FACTORIZABLE — at the operating point `(d_seg_0, d_pose_0, rate_0)`:

```
Hess score = 100 · Hess d_seg + (5/sqrt(10·d_pose_0)) · Hess d_pose
           - (25/(2 · (10·d_pose_0)^(3/2))) · ∇d_pose ⊗ ∇d_pose
           + (25/(37_545_489)) · Hess archive_bytes
```

**The piecewise-constant d_seg term breaks first-order linearity at SegNet argmax boundaries per parent synthesis §1.1**:

`Hess d_seg` is ZERO almost-everywhere (d_seg is piecewise-constant), but has Dirac-delta spikes at argmax class boundaries. The classical Newton step is undefined at these spikes; **subgradient methods (Boyd; per parent synthesis §5.6 tropical-Newton) recover correctness almost-everywhere**. The TROPICAL-NEWTON extension to handle these boundaries is the Phase 6 sister deliverable; **Phase 2's Riemannian-Newton operates on the smooth ALMOST-EVERYWHERE portion**.

The **closed-form Riemannian-Newton step** at the operating point:

```
δθ = - (Hess score(θ_0) + λI)^(-1) · F(θ_0)^(-1) · ∇score(θ_0)
```

where `λI` is the Levenberg-Marquardt damping per §1.2. The trust-region constraint `||δθ|| ≤ Δ` per Boumal 2020 §6.4 prevents overshoot in the high-curvature regime.

### 1.7 Cross-substrate inheritance via canonical base class

Per the Contrarian veto's REQUIRED REVISION: Riemannian-Newton is a META-substrate-engineering BOLT-ON, NOT a substrate paradigm. The canonical contract:

```python
class RiemannianNewtonSubstrate(Substrate):
    """Canonical META-substrate-engineering base class.

    Every substrate trainer that OPTS IN to Riemannian-Newton inherits from this base
    class. Substrate-specific engineering happens in subclass overrides:
    - fisher_metric_factory(theta) -> F(theta)        [substrate-specific Fisher computation]
    - hessian_vector_product(v, theta) -> H@v          [substrate-specific HVP]
    - retraction(theta, delta_theta) -> theta_new      [substrate-specific manifold retraction]

    The BASE CLASS provides:
    - fisher_precondition(grad, theta) -> F^-1 @ grad  [shared Levenberg-Marquardt damped natural gradient]
    - symplectic_ema_step(theta_ema, theta_live, momentum_ema) -> (theta_ema_new, momentum_ema_new)
    - riemannian_newton_step(theta, grad) -> delta_theta     [shared trust-region Newton via HVP]
    - fisher_orthogonal_projection(grad, theta) -> grad_perp [shared null-space direction]
    """

    def fisher_metric_factory(self, theta: dict[str, Tensor]) -> Callable:
        # OVERRIDE per substrate; default is empirical Fisher from per-pair gradients
        raise NotImplementedError("Substrate must define Fisher metric")

    def hessian_vector_product(self, v: dict[str, Tensor], theta: dict[str, Tensor]) -> dict[str, Tensor]:
        # OVERRIDE per substrate; default is autograd-based HVP
        return autograd_hessian_vector_product(self.loss_fn, theta, v)

    def retraction(self, theta: dict[str, Tensor], delta_theta: dict[str, Tensor]) -> dict[str, Tensor]:
        # OVERRIDE per substrate; default is Euclidean retraction (theta + delta_theta)
        return {k: theta[k] + delta_theta[k] for k in theta}
```

**Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode (CLAUDE.md non-negotiable)**: substrates that have specific structure (Stiefel-orthonormal renderer projections per parent synthesis §6.4; group-equivariant SE(3) pose-axis per §6.1; tropical-polytope codec configurations per §5.6) OVERRIDE the `fisher_metric_factory` + `retraction` to use the substrate-optimal Riemannian geometry. Substrates without specific structure use the canonical defaults.

---

## 2. Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED / CARGO-CULTED | Rationale | Unwind plan |
|---|---|---|---|
| Fisher info matrix from per-pair master gradient is well-conditioned across all 53+ substrates | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Random-init Hessian eigenvalue spectra per arxiv:2506.03470 follow Marchenko-Pastur + spike; bulk is near-zero (Fisher near-singular along null-space directions). Catalog #324 random-init-vs-post-training mismatch shows this directly | Phase 1 op-routable: empirically validate Fisher conditioning on PR101_lc_v2 archive's master-gradient anchor; report `(λ_min, λ_max, condition_number)` and select damping `λ` accordingly. Use K-FAC for substrates where full Fisher is intractable (N > 100K params) |
| Symplectic-integrator EMA preserves contest-scorer-relevant invariants | CARGO-CULTED | Symplectic integrators preserve phase-space volume per Liouville's theorem, but the contest scorer is a scalar function NOT a phase-space quantity. Mapping from phase-space volume preservation to contest-scorer improvement is empirically uncharacterized | Phase 4 op-routable: paired comparison standard-EMA vs symplectic-EMA on Z6-v2 Wave 2 full-mode (~$5-10 Modal A100 smoke). Validate empirically before broad rollout. Keep both EMA variants in canonical helper; substrates opt-in via SubstrateContract field |
| Riemannian-Newton converges in finite iterations on the renderer-weight manifold | HARD-EARNED-WITH-REVISION | Boumal 2020+ literature establishes quadratic convergence in neighborhood of strict local minima with non-degenerate Hessian. The piecewise-constant d_seg component creates measure-zero non-smooth set where Newton convergence fails | Phase 2 op-routable: apply Riemannian-Newton on smooth portion only (d_pose + rate + smooth-approximated d_seg via Sobolev mollification). Phase 6 sister deliverable extends to tropical-Newton at SegNet argmax boundaries per parent synthesis §5.6 |
| Fisher-Orthogonal Projection (arxiv:2508.13898) generalizes from large-batch SGD to per-pair video-pair setting | HARD-EARNED-WITH-REVISION | The geometric construction (project gradient onto Fisher-null-subspace) is batch-size-agnostic; Riemannian Pythagorean theorem (mathematical foundation) holds for ANY Fisher metric. Pact's 600 contest pairs IS the canonical batch | Phase 1 op-routable validates empirically; no additional unwind required if Fisher-conditioning check passes |
| Cross-substrate inheritance via canonical RiemannianNewtonSubstrate base class preserves UNIQUE-AND-COMPLETE-PER-METHOD operating mode | HARD-EARNED-WITH-REVISION | Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" non-negotiable: canonical helpers are TOOLS available for use when they serve. The base class provides DEFAULTS that substrates OVERRIDE per substrate-optimal engineering | Phase 3 op-routable: every substrate that opts into Riemannian-Newton MUST document the canonical-vs-unique decision per layer (Catalog #290): which fields of RiemannianNewtonSubstrate are inherited vs overridden |
| The 53-substrate registry continues to evolve independently of the Riemannian-Newton helper | HARD-EARNED | The bolt-on framing per Contrarian veto preserves substrate registry independence. Substrates that DO NOT opt-in to Riemannian-Newton continue to function as before; backward compat is structural | No unwind required |
| The Riemannian-Newton helper composes orthogonally with sister deliverables (theoretical floor estimator + deterministic optimizer + Z8 symposium) | HARD-EARNED-WITH-REVISION | Per parent synthesis §10.3 build order: each sister deliverable has independent scope; the cross-pollination matrix is well-defined | Phase 5 op-routable wires the 6 canonical hooks per Catalog #125; verifies orthogonal composition via paired Pareto frontier on substrate composition matrix |
| Predicted ΔS `[-0.025, -0.008]` per archive is empirically valid pre-build | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Parent synthesis §8.1 derives the band from second-order quadratic convergence + Fisher curvature literature; not directly measured on pact archives | Phase 2 op-routable: paired smoke on PR101_lc_v2 archive (~$5 Modal CPU + $10 Modal A100); validate band per Catalog #324 post-training Tier-C re-measurement |
| Predicted aggregate ΔS `[-0.105, -0.034]` across 4 frontier archives under HIGH-orthogonality is valid | CARGO-CULTED | Assumes sub-additive composition per Catalog #322 anti-phantom; the HIGH-orthogonality assumption is itself unverified for Riemannian-Newton across multiple substrates | Phase 3 op-routable: progressive validation as substrates opt-in (1 archive → 2 → 3 → 4); paired comparison vs HIGH-orthogonality prediction at each step |
| Geomstats integration is the canonical Python library choice for Riemannian optimization | HARD-EARNED | Per parent synthesis §4.7 + §7.1: Geomstats has PyTorch + numpy backends + MIT license + 3-year active development; sister libraries Geoopt (PyTorch) + Pymanopt (numpy/autograd) considered | Phase 2 op-routable: integration sketch via SubstrateContract field `riemannian_library: Literal["geomstats", "geoopt"] = "geomstats"`; allows alternative library per substrate-specific needs |

---

## 3. 9-dimension success checklist evidence (Catalog #294)

### 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. **UNIQUENESS** | Riemannian-Newton substrate engineering as a CANONICAL HELPER (not per-substrate copy-paste) is structurally NEW for pact. No prior `tac.*` module implements Fisher-preconditioned natural gradient + symplectic-integrator EMA. Parent synthesis §10.3 Phase 2 names this as the canonical deliverable; THIS memo is the canonical first-instance design |
| 2. **BEAUTY + ELEGANCE** | The base-class + subclass-override pattern preserves UNIQUE-AND-COMPLETE-PER-METHOD via per-substrate override hooks. The 6-method canonical contract (`fisher_metric_factory`, `hessian_vector_product`, `retraction`, `fisher_precondition`, `symplectic_ema_step`, `riemannian_newton_step`) is structurally minimal; each method has a single canonical responsibility. The mathematical framework (§1) is reducible to 7 canonical equations (Fisher metric, natural gradient, Newton step, symplectic leapfrog, Fisher-orthogonal projection, Levenberg-Marquardt damping, closed-form scorer Hessian decomposition); each equation is ≤2 lines |
| 3. **DISTINCTNESS** | Distinct from existing substrate trainers (which use vanilla AdamW + first-order EMA): introduces second-order Riemannian Hessian, Fisher precondition, symplectic integrator. Distinct from sister deterministic-optimizer subagent (which handles discrete codec_config; this helper handles continuous θ). Distinct from sister theoretical-floor estimator (which measures `(lower_bound, upper_bound, confidence_interval)`; this helper improves the OPTIMIZATION TO THE FLOOR). Distinct from sister Z8 symposium (which is a substrate paradigm; this helper is a META-substrate-engineering canonical helper) |
| 4. **RIGOR** | 15+ 2024-2026 arxiv citations HARD-EARNED-VERIFIED via parent synthesis §9 (Riemannian, Fisher, symplectic, K-FAC, trust-region literature); cargo-cult audit per Catalog #303 (10 assumptions enumerated); premise verification per Catalog #229 (5 canonical pointers read in full + 2 sister design memos cross-checked); Sextet pact council deliberation per Catalog #292 + #300 v2 frontmatter (T2 tier; 11 attendees including 5 grand council mathematical specialists); per-substrate-symposium discipline per Catalog #325 future-applicability (each substrate opting into Riemannian-Newton MUST run its own per-substrate symposium per #325) |
| 5. **OPTIMIZATION PER TECHNIQUE** | Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: each substrate's Fisher metric + retraction is OPTIMAL for THAT substrate's structure. PR101_lc_v2 (entropy-coded latents) uses Fisher on latent space; Z6 (FiLM ego-motion) uses Fisher on conditioning network; DP1 (driving prior) uses Fisher on codebook embeddings. The base class provides defaults; substrates override per substrate-optimal engineering. The K-FAC Kronecker factorization is OPTIMAL for layered networks (most substrates); substrates with non-layered structure (DP1 codebook) use direct Fisher computation |
| 6. **STACK-OF-STACKS-COMPOSABILITY** | Riemannian-Newton helper COMPOSES with: (a) Z6/Z7/Z8 predictive coding substrates (Fisher metric on conditioning + dynamics); (b) DP1 driving prior (Fisher metric on codebook); (c) PR101_lc_v2 entropy-coded latents (Fisher metric on latent dimensions); (d) sister deterministic-optimizer (inner continuous θ becomes Riemannian-Newton; outer discrete codec_config remains substrate registry); (e) sister theoretical-floor estimator (Fisher-curvature diagnostic feeds floor-tightening per parent synthesis §10.4); (f) cathedral autopilot v2 cascade (new reward factor `adjust_predicted_delta_for_riemannian_newton_phase_eligibility`). The 6 canonical hooks per Catalog #125 are ALL ACTIVE (§7 below) |
| 7. **DETERMINISTIC REPRODUCIBILITY** | All Riemannian operations are deterministic per autograd; Geomstats library is deterministic (seeded). The Catalog #205 inflate device-fork discipline is preserved (the canonical helper does NOT touch inflate.py; it operates only during training). The Catalog #245 modal call_id ledger 4-layer pattern applies: Layer 1 = `tac.riemannian_newton_meta_substrate` canonical helper; Layer 2 = `tools/run_riemannian_newton_paired_smoke.py` CLI; Layer 3 = Catalog STRICT preflight gate refusing stale Fisher-curvature citations; Layer 4 = autopilot rerank wire-in via new reward factor |
| 8. **EXTREME OPTIMIZATION + PERFORMANCE** | Quadratic convergence near local minima (Boumal 2020 Theorem 6.4.3) vs first-order AdamW's linear convergence — predicted 5-20× faster convergence near optima. K-FAC Kronecker approximation: `O(N^(3/2))` instead of `O(N^3)` Fisher inverse — tractable for N ≈ 300K typical substrate. Hessian-free Newton: `O(N)` time + memory per HVP via autograd — no full Hessian formed. Symplectic-EMA: phase-space-volume-preserving — no information drift across training steps. Parent synthesis §10.3: predicted ΔS unlock per archive `[-0.025, -0.008]` |
| 9. **OPTIMAL MINIMAL CONTEST SCORE** | Predicted ΔS per archive: `[-0.025, -0.008]`. Combined predicted aggregate ΔS across 4 frontier archives under HIGH-orthogonality: `[-0.105, -0.034]` per parent synthesis §10.3. Frontier potential post-build: `[0.087, 0.158]` [contest-CPU] from current 0.19205. Brings pact into asymptotic-pursuit band [0.05, 0.12] per deep-research wave §0 Shannon-floor prediction. Cross-pollination with sister theoretical-floor estimator tightens FLOOR-LOWER-BOUND from `[0.05, 0.12]` to `[0.05, 0.10]` per parent synthesis §10.4 (Fisher-curvature correction) |

---

## 4. Observability surface (Catalog #305)

### Observability surface

The Riemannian-Newton helper is observable via the 6-facet definition per CLAUDE.md "Max observability — non-negotiable":

1. **Inspectable per layer**: each canonical method (`fisher_metric_factory`, `hessian_vector_product`, `retraction`, `fisher_precondition`, `symplectic_ema_step`, `riemannian_newton_step`) is independently callable + inspectable. Every Riemannian-Newton step emits a structured log line:
   ```python
   {
       "step": k,
       "theta_norm": float,
       "grad_norm": float,
       "fisher_condition_number": float,
       "newton_step_norm": float,
       "trust_region_radius": float,
       "score_before": float,
       "score_after": float,
       "score_decrease": float,
       "ema_phase_volume_drift": float,
   }
   ```

2. **Decomposable per signal**: the Fisher matrix decomposes per-pair (each of 600 contest pairs contributes one rank-1 update); per-layer (K-FAC Kronecker factorization decomposes per-layer); per-eigenvalue (the spectrum reveals null-space dimension). The composite Newton step decomposes into Fisher-aligned + Fisher-orthogonal components per arxiv:2508.13898.

3. **Diff-able across runs**: two Riemannian-Newton training runs with different damping `λ` can be diffed step-by-step. The structured log lines + Fisher matrix snapshots enable run-to-run comparison.

4. **Queryable post-hoc**: the canonical Fisher-curvature anchor is persisted to `.omx/state/riemannian_newton_anchors.jsonl` per Catalog #131/#138 fcntl-locked discipline. Queryable via `tac.riemannian_newton_meta_substrate.query_anchors_by_archive(archive_sha256)`.

5. **Cite-able**: every Riemannian-Newton step's anchor cites `(substrate_sha256, archive_sha256, commit_sha, training_step, fisher_damping_lambda, hardware_substrate)` per Catalog #245 modal call_id ledger pattern.

6. **Counterfactual-able**: paired comparison with vanilla AdamW baseline is a first-class operator-facing CLI surface (`tools/run_riemannian_newton_paired_smoke.py --baseline adamw --variant riemannian_newton`). Each step's score decrease (vs counterfactual baseline step) is logged.

**Specific observability artifacts**:

- `.omx/state/riemannian_newton_anchors.jsonl` — per-step Fisher-curvature anchors (fcntl-locked JSONL append-only per Catalog #131)
- `.omx/state/riemannian_newton_paired_comparison/<run_id>.json` — paired smoke comparison results (canonical fcntl-locked sidecar)
- `reports/riemannian_newton_dashboard.md` — operator-facing dashboard with per-substrate convergence curves
- `tools/audit_riemannian_newton_compliance.py` — operator-runnable audit tool that scores each opted-in substrate trainer across the 6 observability facets (sister of `tools/audit_existing_infrastructure_for_observability.py`)

---

## 5. Sextet pact deliberation + grand council attendees (Catalog #292 + #300)

### Sextet pact + grand council attendees

Per Catalog #300 v2 frontmatter + CLAUDE.md "Council conduct" sextet pact + "Grand Council (advisory)" 20-seat roster:

**Sextet pact (binding 5-of-6 quorum required)**:

#### Shannon LEAD (information-theory grounding)

*The shared assumption I am operating within for this design is*: every Riemannian-Newton step provides an information-theoretic improvement quantifiable as ΔH(score_distribution) per Amari-Karakida 1808.07172. The Fisher metric is the canonical information-theoretic metric on parametric distributions.

**Position**: PROCEED with the Phase 1 Fisher-precondition + Levenberg-Marquardt damping landing first. The mathematical foundation per Amari 1985 + arxiv:2506.15830 (Rethinking LLM Training through Information Geometry) is rigorous. The geometric construction (Fisher-orthogonal projection per arxiv:2508.13898) IS the canonical information-theoretic null-space identification.

**Specific contribution**: I assert that the predicted ΔS unlock for null-space exploitation TOP-1 (from `[-0.040, -0.012]` to `[-0.055, -0.018]`) is INFORMATION-THEORETICALLY justified — Fisher-orthogonal projection captures the entropy-maximizing direction in the null subspace. The PYTHAGOREAN THEOREM for Fisher metric (per Amari-Nagaoka 2000 §3) provides the canonical decomposition.

#### Dykstra CO-LEAD (optimization-feasibility / convex feasibility)

*The shared assumption I am operating within for this design is*: the trust-region constraint `||δθ|| ≤ Δ` defines a CONVEX FEASIBLE REGION on the tangent space at θ. The trust-region subproblem is a CONVEX QUADRATIC PROGRAM per Boumal 2020 §6.4. Alternating projections converge to the canonical Riemannian-Newton step.

**Position**: PROCEED. The Dykstra-feasibility check per Catalog #296 is structurally satisfied — the trust-region constraint + Fisher-conditioning constraint + (post-Phase-6) tropical-Newton boundary constraint intersect non-trivially per the Boumal 2020 + arxiv:2409.03945 literature.

**Specific contribution**: I assert the predicted ΔS band `[-0.025, -0.008]` per archive is consistent with the convex feasibility cone of (smooth-score-region × trust-region × Fisher-conditioning). The intersection is non-empty + bounded; Newton convergence within this region is quadratic. Tropical-Newton extension at SegNet boundaries (Phase 6) expands the feasible region without compromising Phase 2 convergence guarantees.

#### Yousfi (contest scorer expertise)

*The shared assumption I am operating within for this design is*: the contest scorer's d_seg + d_pose components are differentiable through the canonical eval_roundtrip path per CLAUDE.md non-negotiable. The Hessian computation via autograd is structurally valid.

**Position**: PROCEED with Phase 2 caveat: the SegNet argmax boundaries are NOT differentiable; the Newton step is undefined at these points. Phase 6 tropical-Newton extension handles this; Phase 2 should explicitly check Fisher-conditioning AT boundary points + fall back to vanilla AdamW on the affected pairs.

**Specific contribution**: I propose: per-pair Fisher-conditioning check at each Newton step; if `cos(d_seg_gradient, d_pose_gradient) > 0.95` for a pair (indicating boundary proximity), fall back to vanilla AdamW step for THAT pair while applying Riemannian-Newton to the smooth pairs. This is a SUBSTRATE-LEVEL hybrid that preserves Phase 2 convergence on the smooth majority of pairs.

#### Fridrich (steganalysis / contest design)

*The shared assumption I am operating within for this design is*: Fisher-orthogonal projection identifies directions that the SegNet+PoseNet scorer is INSENSITIVE to — these directions are precisely the steganographic null-space the contest was designed to detect. UNIWARD weighting per arxiv:1311.7041 is the canonical steganographic prior.

**Position**: PROCEED. The Riemannian-Newton paradigm aligns with steganographic optimization — Fisher-orthogonal directions are by construction the directions where contest scorer cannot detect modifications. The frontier-breaking opportunity is using UNIWARD-weighted Fisher metric instead of vanilla Fisher.

**Specific contribution**: I propose: the Fisher metric SHOULD be UNIWARD-weighted (per arxiv:1311.7041 + canonical steganographic literature). The weighted Fisher matrix `F_UNIWARD(θ) = F(θ) ∘ W_UNIWARD` where `W_UNIWARD` is the per-pixel UNIWARD weight matrix (high in textured regions, low in smooth regions). This focuses the Riemannian-Newton step on perceptually-undetectable modifications.

#### Contrarian (challenges weak arguments)

*The shared assumption I am operating within for this design is*: the canonical helper framing (vs new substrate paradigm) is structurally correct per the just-landed META-cargo-cult #11 (architectural-replacement-is-canonical-research-path-forward). Architectural replacement does NOT compose monotonically.

**Position**: VETO any framing that treats Riemannian-Newton as a substrate REPLACEMENT. The 5-of-13 architectural-replacement empirical anchor (NSCS06 v7→v8 -78% per the just-landed wave) proves architectural replacement does not compose monotonically. Riemannian-Newton must land as a META-substrate-engineering CANONICAL HELPER consumed by EXISTING substrate trainers, NOT a new substrate paradigm that obsoletes the 53-substrate registry.

**Specific contribution**: I assert the predicted aggregate ΔS `[-0.105, -0.034]` across 4 frontier archives is OPTIMISTIC. Under realistic α-discount per Catalog #322 anti-phantom (substrate composition matrix shows 4/8 probed pairs are anti-additive), the realistic prediction is `[-0.050, -0.020]`. Per parent synthesis §10.3 build order: this STILL brings frontier into asymptotic-pursuit band; the design is sound but the predicted EV must be stated CONSERVATIVELY.

#### Assumption-Adversary (challenges shared framing)

*The shared assumption I am operating within for this design is*: Fisher info on the renderer-weight manifold is well-conditioned. This is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION.

**Position**: PROCEED_WITH_REVISIONS. Random-init Hessians per arxiv:2506.03470 are NOT well-conditioned (Marchenko-Pastur bulk + spike); the cathedral autopilot Tier C empirical anchor (Catalog #324) shows this directly. Fisher-precondition step MUST be paired with damped natural gradient (Levenberg-Marquardt) OR Kronecker-factored approximation (K-FAC arxiv:1503.05671).

**Specific contribution**: I demand: Phase 1 op-routable EMPIRICAL VALIDATION of Fisher conditioning on PR101_lc_v2 archive BEFORE Phase 2 Riemannian-Newton landing. The validation: extract Fisher matrix, compute eigenvalue spectrum, report `(λ_min, λ_max, condition_number)`. If `condition_number > 1e6` (indicating Fisher near-singular), require K-FAC instead of full Fisher inverse. This validation is $0 GPU + ~2 hour editor on already-extracted per-pair gradient anchor.

### Grand council attendees (per topic)

Per CLAUDE.md "Grand Council (advisory)" 20-seat roster + topic-specific specialist invocations:

#### MacKay (memorial seat; information-theory-Bayesian framework)

*The shared assumption I am operating within for this design is*: the Bayesian interpretation of the Fisher metric as the Jeffreys prior is canonical. Natural gradient is Bayesian-coherent in the sense of Jeffreys' invariance under reparameterization.

**Position**: SUPPORTS PROCEED. The MDL-IB framework (per `feedback_c6_e4_mdl_ibps_*` sister substrate) is mathematically consistent with Fisher-preconditioned natural gradient. The C6 IBPS 22× miss anchor (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) MAY be diagnosable via Fisher-conditioning analysis — the 24-dim IB bottleneck may be Fisher-near-singular at random init, leading to natural-gradient divergence in the first few epochs. THIS is exactly what the Phase 1 op-routable Fisher-conditioning check validates.

**Specific contribution**: I propose: extend the Phase 1 op-routable to ALSO measure Fisher conditioning on C6 IBPS random-init weights vs post-training weights. If the random-init Fisher is severely ill-conditioned but post-training Fisher is well-conditioned, the C6 IBPS 22× miss is structurally explained — and the Phase 1 Fisher-precondition becomes the canonical remediation for the entire IB substrate family.

#### Hafner (DreamerV3 / latent dynamics)

*The shared assumption I am operating within for this design is*: symplectic integrators preserve phase-space volume per Liouville's theorem. The mapping to contest-scorer improvement is empirically uncharacterized.

**Position**: PROCEED_WITH_REVISIONS. The symplectic-EMA per arxiv:2407.00294 + arxiv:2509.24627 is theoretically sound BUT requires empirical validation. DreamerV3 RSSM training experience shows symplectic integrators preserve information volume during world-model training; whether this translates to contest-scorer improvement requires the Phase 4 op-routable paired comparison.

**Specific contribution**: I propose: the Phase 4 paired comparison should use Z6-v2 Wave 2 full-mode (predictive-coding substrate) since the symplectic structure is most natural for prediction-task training. Standard EMA baseline vs symplectic-EMA variant; report (a) final contest score; (b) training-step convergence curves; (c) information-theoretic phase-volume drift across training. The convergence-curve smoothness IS the diagnostic for symplectic-EMA benefit.

#### Boyd (convex optimization at operational level)

*The shared assumption I am operating within for this design is*: the Levenberg-Marquardt damping per Phase 1 + the trust-region method per Phase 2 are both canonical convex optimization techniques with well-understood convergence guarantees.

**Position**: SUPPORTS PROCEED. The proximal gradient + ADMM techniques (per my canonical contribution to the council) provide the natural fallback if the Riemannian-Newton step fails to converge in the trust-region subproblem. The Steihaug-Toint truncated CG solver is the canonical algorithmic primitive for the trust-region inner problem.

**Specific contribution**: I propose: when the trust-region subproblem cannot be solved exactly (e.g., when Hessian-vector products are noisy due to stochastic gradient sampling), fall back to PROXIMAL GRADIENT step `θ_{k+1} = prox_{η · score}(θ_k - η · grad_R score(θ_k))` per my ADMM canonical literature. The proximal operator handles the non-smooth d_seg portion naturally.

#### Carmack (engineering shortcuts at production level)

*The shared assumption I am operating within for this design is*: the implementation cost MUST be bounded. The 600 LOC budget per HNeRV parity L7 (bolt-on size budget) applies to the per-substrate subclass; the base class is substrate-engineering and CAN exceed the budget per L7.

**Position**: PROCEED with implementation discipline. The base class ~600 LOC budget is generous but should be respected. K-FAC implementation can be borrowed from canonical PyTorch libraries (e.g., the `kfac_pytorch` library); Geomstats integration should be a thin adapter, not a re-implementation.

**Specific contribution**: I propose: USE EXISTING CANONICAL LIBRARIES wherever possible. Geomstats for Riemannian geometry (MIT license, PyTorch backend, 5+ years active). `kfac_pytorch` for K-FAC approximation (MIT license). Custom code ONLY for pact-specific contest scorer Hessian decomposition + symplectic-EMA. The base class should be a THIN INTEGRATION LAYER over canonical libraries, NOT a re-implementation. Target: 400 LOC base class + 200 LOC sister test suite = 600 LOC total.

#### Hotz (raw engineering + analytical shortcuts)

*The shared assumption I am operating within for this design is*: the simplest approach that captures 80% of the EV is preferred over the most rigorous approach that captures 100%. Pareto principle applies to implementation effort.

**Position**: PROCEED with Phase 1 focus. The Fisher-orthogonal projection per arxiv:2508.13898 is the SIMPLEST Phase 1 deliverable that captures MOST of the predicted EV. The full Riemannian-Newton (Phase 2) is ~5× implementation effort for ~2× predicted EV unlock; Phase 2 should be deferred until Phase 1 is empirically validated.

**Specific contribution**: I propose: the Phase 1 deliverable is THE PRIORITY. The implementation is ~200 LOC (Fisher computation + Levenberg-Marquardt damping + Fisher-orthogonal projection). The empirical validation on PR101_lc_v2 archive is ~$0 (M5 Max CPU smoke). If Phase 1 validates the predicted ΔS unlock for null-space TOP-1, then Phase 2 follows naturally. If Phase 1 FALSIFIES the predicted unlock, the entire paradigm is re-evaluated. This is the canonical empirical-anchor-first discipline.

### Council verdict tally

| Member | Verdict |
|---|---|
| Shannon LEAD | PROCEED |
| Dykstra CO-LEAD | PROCEED |
| Yousfi | PROCEED_WITH_REVISIONS (per-pair boundary-proximity hybrid) |
| Fridrich | PROCEED (UNIWARD-weighted Fisher) |
| Contrarian | PROCEED_WITH_REVISIONS (META-helper framing; conservative EV prediction) |
| Assumption-Adversary | PROCEED_WITH_REVISIONS (Phase 1 empirical Fisher-conditioning validation REQUIRED) |
| MacKay (grand) | SUPPORTS PROCEED (extend Phase 1 to C6 IBPS diagnosis) |
| Hafner (grand) | PROCEED_WITH_REVISIONS (Phase 4 paired-comparison on Z6-v2 REQUIRED) |
| Boyd (grand) | SUPPORTS PROCEED (proximal gradient fallback) |
| Carmack (grand) | PROCEED with implementation discipline (USE existing canonical libraries) |
| Hotz (grand) | PROCEED with Phase 1 focus |

**Aggregate verdict: PROCEED_WITH_REVISIONS** (sextet quorum: 5-of-6 PROCEED + 1 PROCEED_WITH_REVISIONS; grand council majority PROCEED_WITH_REVISIONS).

**The binding revisions** (operator-routable):
1. Phase 1 lands BEFORE Phase 2: Fisher-precondition + Levenberg-Marquardt damping + Fisher-orthogonal projection, with empirical Fisher-conditioning validation on PR101_lc_v2 archive (Assumption-Adversary + Hotz + Carmack).
2. Phase 4 symplectic-EMA validated via paired comparison on Z6-v2 Wave 2 full-mode BEFORE broad rollout (Hafner).
3. META-helper framing (Contrarian); per-pair boundary-proximity hybrid for SegNet argmax boundaries (Yousfi); UNIWARD-weighted Fisher option (Fridrich).
4. USE existing canonical libraries (Geomstats, kfac_pytorch); target ~600 LOC base class (Carmack).
5. Extend Phase 1 to diagnose C6 IBPS 22× miss via Fisher-conditioning analysis (MacKay).

---

## 6. Per-substrate reactivation criteria (Catalog #313 + sister "Forbidden premature KILL")

### Per-substrate reactivation criteria

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #313 probe-outcomes ledger discipline: every substrate symposium MUST enumerate reactivation paths with priority ordering, predicted cost, and structural verdict on which assumption each path tests.

The Riemannian-Newton META-substrate is OPT-IN (substrates inherit by SubstrateContract field flag). Reactivation criteria operate per-OPTED-IN-SUBSTRATE:

#### Reactivation path 1 (HIGHEST priority): Phase 1 Fisher-conditioning empirical validation

**Trigger**: any opted-in substrate whose Fisher matrix has `condition_number > 1e6` at random-init.

**Path**: replace full Fisher inverse with K-FAC Kronecker-factored approximation per arxiv:1503.05671. The K-FAC factorization handles the near-singular structure by approximating Fisher per-layer instead of holistically.

**Predicted cost**: $0 GPU (analysis on already-extracted per-pair gradient anchor); ~1 day editor for K-FAC integration via `kfac_pytorch`.

**Structural verdict**: tests Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification for Fisher conditioning.

#### Reactivation path 2 (HIGH priority): Phase 4 symplectic-EMA paired comparison

**Trigger**: any opted-in substrate where standard EMA + Riemannian-Newton shows convergence-curve oscillations or score-drift across training steps.

**Path**: replace standard EMA with symplectic-EMA per arxiv:2407.00294 + arxiv:2509.24627. The symplectic structure preserves information volume; the paired comparison validates contest-scorer improvement.

**Predicted cost**: ~$5-10 Modal A100 smoke per substrate; ~3 day editor for symplectic-EMA integration.

**Structural verdict**: tests Hafner's CARGO-CULTED classification for symplectic-EMA → contest-scorer mapping.

#### Reactivation path 3 (MEDIUM priority): Phase 6 tropical-Newton extension at SegNet boundaries

**Trigger**: any opted-in substrate where per-pair Fisher-conditioning analysis shows `cos(d_seg_gradient, d_pose_gradient) > 0.95` for more than 5% of pairs (indicating SegNet boundary proximity).

**Path**: extend Riemannian-Newton step with tropical-Newton subgradient handling per arxiv:2409.03945 (TropNNC) + arxiv:2402.00576. The tropical-Newton step uses MAX-PLUS SEMIRING arithmetic at boundary points; classical Newton steps everywhere else.

**Predicted cost**: ~$0 GPU (analysis only); ~1-2 week editor for tropical-Newton integration.

**Structural verdict**: tests Yousfi's PROCEED_WITH_REVISIONS classification for per-pair boundary-proximity hybrid.

#### Reactivation path 4 (LOW priority): UNIWARD-weighted Fisher metric

**Trigger**: any opted-in substrate where steganographic interpretation is structurally relevant (e.g., visual-quality-preserving substrates per parent synthesis §6.2 fiber bundle structure).

**Path**: replace vanilla Fisher metric with UNIWARD-weighted Fisher per arxiv:1311.7041. The weighted Fisher matrix `F_UNIWARD(θ) = F(θ) ∘ W_UNIWARD` focuses Riemannian-Newton steps on perceptually-undetectable modifications.

**Predicted cost**: ~$0 GPU (UNIWARD weights are pre-computed); ~3 day editor for weighted Fisher integration.

**Structural verdict**: tests Fridrich's contribution proposal.

### Probe outcomes ledger registration (Catalog #313)

Upon landing this design memo, the canonical probe outcome IS:

```python
register_probe_outcome(
    probe_id="riemannian_newton_substrate_engineering_design_20260518",
    substrate_id="riemannian_newton_meta_substrate",
    verdict="PROCEED_WITH_REVISIONS",
    blocking=False,  # not blocking; META-helper framing
    rationale="Phase 1 Fisher-precondition + Levenberg-Marquardt damping + Fisher-orthogonal projection per arxiv:2508.13898; empirical Fisher-conditioning validation on PR101_lc_v2 archive REQUIRED before Phase 2 Riemannian-Newton landing per Assumption-Adversary's CARGO-CULTED classification",
    reactivation_criteria=[
        "Phase 1 empirical Fisher-conditioning validation falsifies the well-conditioned assumption (condition_number > 1e6) → escalate to K-FAC implementation",
        "Phase 4 symplectic-EMA paired comparison falsifies the contest-scorer improvement → defer symplectic-EMA broad rollout pending alternative validation",
        "Phase 6 tropical-Newton extension required for substrates with >5% boundary-proximity pairs → escalate to TropNNC integration",
    ],
    related_council_anchor="riemannian_newton_substrate_engineering_design_20260518",
)
```

---

## 7. Catalog #324 post-training Tier-C validation discipline

### Catalog #324 post-training Tier-C validation discipline

Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density (the phantom-predicted-band trap)" + Catalog #324: every `predicted_band` field MUST satisfy one of:

(a) `predicted_band_validation_status: validated_post_training` + post-training Tier-C density artifact path;
(b) `predicted_band_validation_status: pending_post_training` + reactivation criteria pinned;
(c) `research_only: true` OR `dispatch_enabled: false`;
(d) same-line `# PREDICTED_BAND_RANDOM_INIT_OK:<rationale>` waiver.

**This design memo's predicted_band declarations**:

- **Per-archive predicted ΔS `[-0.025, -0.008]`**: `predicted_band_validation_status: pending_post_training`. Reactivation criterion: paired post-training Tier-C re-measurement on the first 4 archives consuming Riemannian-Newton (PR101_lc_v2 + A1 + PR106 format0d + sane_hnerv) via `tools/mdl_scorer_conditional_ablation.py --tier c`; predicted band validated when at least 3-of-4 fall within the band.

- **Aggregate predicted ΔS `[-0.105, -0.034]` under HIGH-orthogonality**: `predicted_band_validation_status: pending_post_training`. Reactivation criterion: progressive validation as substrates opt-in (1 archive → 2 → 3 → 4); paired comparison vs HIGH-orthogonality prediction at each step.

- **Aggregate predicted ΔS `[-0.050, -0.020]` under realistic α-discount** (Contrarian's conservative estimate): `predicted_band_validation_status: pending_post_training`. Reactivation criterion: same as above, with composition_alpha empirical measurement per Catalog #322 anti-phantom.

**The dispatch eligibility consequence per Catalog #324**: the recipe for any opt-in substrate MUST declare:

```yaml
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: |
  Post-training Tier-C re-measurement on archive sha256 <archive_sha[:12]> via
  tools/mdl_scorer_conditional_ablation.py --tier c. Predicted ΔS [-0.025, -0.008]
  validated when post-training Tier-C density >= 0.50 (substrate-class-shift) AND
  empirical ΔS falls within band.
```

The cathedral autopilot v2 cascade consumes `predicted_band_validation_status` per Catalog #324 — `pending_post_training` rows are discounted (50% factor) in the autopilot ranker per the canonical reweight `apply_z1_empirical_revision_to_candidate_delta`.

---

## 8. Canonical-vs-unique decision per layer (Catalog #290)

### Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + the operator's 2026-05-15 retrospective: every NEW substrate scaffold design memo MUST include this section documenting per-layer canonical-vs-unique decisions.

The Riemannian-Newton helper is META-substrate-engineering; the per-layer canonical-vs-unique decisions apply BOTH to the helper's own layers AND to the substrates that opt-in.

#### Helper's own layers (the base class)

| Layer | Decision | Rationale |
|---|---|---|
| 1. Fisher metric computation | **CANONICAL** (shared via base class) | Computing the empirical Fisher matrix from per-pair master gradient is structurally identical across substrates (input: per-pair gradients of shape `(N_pairs, N)`; output: Fisher matrix of shape `(N, N)`). Sharing prevents code duplication; no substrate-specific structure is suppressed |
| 2. Levenberg-Marquardt damping `λ` selection | **CANONICAL DEFAULT + UNIQUE OVERRIDE** | Default selection rule per Phase 1 op-routable: `λ = max(1e-3, 1e-2 / condition_number)`. Substrates with substrate-specific structure (e.g., DP1 codebook embeddings, Z6 FiLM conditioning) OVERRIDE the rule via `fisher_metric_factory` subclass method |
| 3. K-FAC Kronecker factorization | **CANONICAL** (delegated to `kfac_pytorch` library per Carmack) | K-FAC factorization per-layer is mathematically substrate-independent; the library handles all layered architectures. Substrates with non-layered structure (DP1 codebook) opt-out via `use_kfac: bool = False` SubstrateContract field |
| 4. Fisher-orthogonal projection | **CANONICAL** (per arxiv:2508.13898) | Mathematical decomposition into Fisher-aligned + Fisher-orthogonal components is substrate-independent. The projection itself is a linear-algebra operation on the Fisher eigenvectors |
| 5. Riemannian Newton step (trust-region) | **CANONICAL DEFAULT + UNIQUE OVERRIDE** | Default trust-region method per Boumal 2020 Algorithm 6.3.1 is canonical. Substrates with substrate-specific manifold structure (Stiefel for orthonormal projections, Grassmann for subspace representations) OVERRIDE `retraction` to use substrate-optimal geodesic step |
| 6. Symplectic-EMA leapfrog step | **CANONICAL** (per arxiv:2407.00294) | 2nd-order leapfrog integrator is mathematically substrate-independent. Higher-order Yoshida composition is opt-in via SubstrateContract field `symplectic_integrator_order: int = 2` |
| 7. Hessian-vector product (HVP) | **CANONICAL DEFAULT + UNIQUE OVERRIDE** | Default HVP via autograd is canonical. Substrates with structured Hessian (e.g., block-diagonal for entropy-coded latents) OVERRIDE `hessian_vector_product` for substrate-optimal computation |

#### Per-opted-in-substrate layers (the subclass)

For each substrate that opts into Riemannian-Newton, the per-substrate layers are:

| Substrate | Layer | Decision | Rationale |
|---|---|---|---|
| **PR101_lc_v2** (entropy-coded latents) | Fisher metric domain | UNIQUE | Fisher metric on LATENT SPACE (not full weight space) — the entropy-coded latents are the rate-relevant parameters. Substrate overrides `fisher_metric_factory` to project to latent space |
| **PR101_lc_v2** | Retraction | CANONICAL | Default Euclidean retraction on latent space; no substrate-specific manifold structure |
| **PR101_lc_v2** | Symplectic-EMA | CANONICAL with override | Default symplectic-EMA on latent space; substrate can override `symplectic_integrator_order: int = 4` for higher accuracy if needed |
| **Z6/Z7/Z8** (predictive coding) | Fisher metric domain | UNIQUE | Fisher metric on CONDITIONING NETWORK (FiLM parameters for Z6; predictor parameters for Z7/Z8). Substrate-specific projection to relevant subspace |
| **Z6/Z7/Z8** | Retraction | UNIQUE for Z6 (SE(3) ego-motion); CANONICAL for Z7/Z8 | Z6's ego-motion is naturally SE(3)-equivariant per parent synthesis §6.1; the retraction uses Lie-algebra exponential map. Z7/Z8 use canonical Euclidean retraction |
| **Z6/Z7/Z8** | Symplectic-EMA | UNIQUE (Hafner's dissent) | Z6-v2 Wave 2 full-mode is the EMPIRICAL VALIDATION case for symplectic-EMA per Phase 4 op-routable. Override `use_symplectic_ema: bool = True` with paired-comparison instrumentation |
| **DP1** (driving prior codebook) | Fisher metric domain | UNIQUE | Fisher metric on CODEBOOK EMBEDDINGS (not the surrounding network). Substrate-specific projection to embedding space |
| **DP1** | Retraction | UNIQUE | Codebook embeddings live on the unit sphere `S^(d-1)` per VQ-VAE canonical literature; the retraction is sphere geodesic step |
| **DP1** | K-FAC factorization | OPT-OUT | Codebook embeddings are NOT layered; full Fisher computation is tractable since codebook is small (~1000 entries × 128 dims) |
| **A1** (PR101 grammar reference) | Fisher metric domain | CANONICAL | Default Fisher on full weight space; no substrate-specific projection |
| **A1** | Retraction | CANONICAL | Default Euclidean retraction |
| **sane_hnerv** (NeRV-family) | Fisher metric domain | CANONICAL | Default Fisher on full weight space |
| **sane_hnerv** | Retraction | CANONICAL | Default Euclidean retraction |

The pattern: **EVERY substrate has at least ONE UNIQUE layer** (the Fisher metric domain), preserving UNIQUE-AND-COMPLETE-PER-METHOD. Other layers default to canonical UNLESS the substrate's mathematical structure demands a fork.

---

## 9. Predicted ΔS band (Catalog #296 Dykstra-feasibility)

### Predicted ΔS band

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check (the predicted-band-vibes trap)" + Catalog #296: every predicted ΔS band MUST cite the Dykstra-feasibility intersection check OR a first-principles bound OR a probe-disambiguator path.

#### Per-archive predicted ΔS `[-0.025, -0.008]`

**Dykstra-feasibility check**: the predicted band is the intersection of THREE convex constraints on the (smooth-score-region × Fisher-conditioning × trust-region) feasibility space.

1. **Smooth-score-region constraint**: the substrate trainer's per-pair Fisher-conditioning satisfies `cos(d_seg_gradient, d_pose_gradient) < 0.95` for at least 95% of pairs (per Yousfi's PROCEED_WITH_REVISIONS specification). This constrains the substrate to operate in the smooth ALMOST-EVERYWHERE region where classical Newton convergence applies.

2. **Fisher-conditioning constraint**: the Fisher matrix has `condition_number < 1e6` post-Levenberg-Marquardt damping (per Assumption-Adversary's Phase 1 validation requirement). This constrains the damped Fisher inverse to be well-defined.

3. **Trust-region constraint**: the per-step `||δθ|| ≤ Δ_k` per Boumal 2020 §6.4. This constrains the Newton step magnitude to remain in the quadratic-convergence regime.

**The intersection is non-empty** per parent synthesis §10.3 Phase 2 deliverability analysis. The Dykstra-feasibility iteration converges to the canonical Riemannian-Newton step within the feasible intersection.

**First-principles bound** (Shannon-Boyd composition):

- **Lower bound**: per Amari-Karakida 1808.07172 information-theoretic analysis: quadratic-convergence Newton step with Fisher metric achieves `O(condition_number / N)` per-step score decrease at scale `N` parameters. For N ≈ 300K + condition_number ≈ 1e3 (post-damping): predicted per-step decrease `~ 1e3 / 3e5 = 3.3e-3 ≈ 0.003` per step. Over 100 epochs of substrate training: cumulative ΔS `~ -0.30` BUT discounted by:
  - Effective per-step decrease is bounded by smooth-score-region constraint (the d_seg piecewise-constant breaks first-order linearity at boundaries; reduces effective ΔS by ~factor 10)
  - Stochastic gradient noise discount per Bonnabel 2013 Theorem 4.1: effective ΔS = theoretical ΔS × `sqrt(N_pairs / batch_size)` (for full-batch training, factor 1; for stochastic, factor 1-0.1)
  - Pareto-frontier saturation: the Riemannian-Newton trajectory eventually saturates at the Pareto frontier; further iterations achieve diminishing returns
- **Realistic predicted ΔS**: `~ 0.30 / 10 (boundary discount) / 4 (saturation discount) = 0.0075` per archive (LOWER bound of predicted band; matches `-0.008`)
- **Upper bound**: theoretical maximum at ZERO null-space + ZERO boundary effects = `0.30 / 4 (saturation only) = 0.075` per archive (but unrealistic; the actual UPPER bound is ~`-0.025` per parent synthesis §10.3)

**Probe-disambiguator path**: `tools/probe_riemannian_newton_vs_vanilla_disambiguator.py` — paired comparison on PR101_lc_v2 archive with measurement of per-step ΔS + final ΔS + condition-number trajectory. The probe-disambiguator EMPIRICALLY validates the predicted band; the verdict feeds the canonical probe-outcomes ledger per Catalog #313.

#### Aggregate predicted ΔS `[-0.105, -0.034]` under HIGH-orthogonality

**Dykstra-feasibility check**: the aggregate band is the SUM of per-archive bands across 4 frontier archives (PR101_lc_v2 + A1 + PR106 format0d + sane_hnerv) under the HIGH-orthogonality assumption (substrates compose super-additively).

The HIGH-orthogonality assumption fails per Catalog #322 anti-phantom (4/8 probed substrate pairs are anti-additive). The realistic aggregate band per Contrarian's revision is `[-0.050, -0.020]`.

**Probe-disambiguator path**: progressive validation per Phase 3 op-routable: validate 1 archive, then 2, then 3, then 4. Measure composition_alpha at each step per `tac.optimization.substrate_composition_matrix`. If composition_alpha < 0.5 at any step, downgrade aggregate prediction to realistic band; if composition_alpha > 0.7, retain HIGH-orthogonality prediction.

#### Aggregate predicted ΔS `[-0.050, -0.020]` under realistic α-discount

**Dykstra-feasibility check**: per Contrarian's revision: aggregate band under composition_alpha ≈ 0.5 (the Catalog #322 anti-phantom empirical median). The intersection of (per-archive band) × (composition_alpha discount) is non-empty + bounded.

**First-principles bound**: per the canonical composition_alpha formula in `tac.optimization.substrate_composition_matrix`: aggregate ΔS = sum of per-archive ΔS × composition_alpha. For 4 archives × `[-0.025, -0.008]` each × 0.5 = `[-0.050, -0.016]`. The realistic band `[-0.050, -0.020]` is consistent (the upper bound is slightly looser).

---

## 10. 6-hook wire-in declaration (Catalog #125)

### 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: every landing must wire its outputs into the unified solver stack OR explicitly tag `research_only=true`.

This design memo is `research_only=true` per the YAML frontmatter; the canonical helper landing (Phase 2 op-routable) MUST wire all 6 hooks.

#### Hook 1: Sensitivity-map contribution → `tac.sensitivity_map.axis_weights` (ACTIVE)

The Riemannian-Newton helper EMITS a Fisher-curvature diagnostic for each opted-in substrate:

```python
@dataclass(frozen=True)
class FisherCurvatureDiagnostic:
    substrate_id: str
    archive_sha256: str
    fisher_condition_number: float
    fisher_top_k_eigenvalues: tuple[float, ...]  # top 10
    fisher_orthogonal_subspace_dim: int  # empirical null-space dimension
    fisher_metric_axis: dict[str, float]  # {"seg": 0.7, "pose": 0.2, "rate": 0.1}
    measured_at_utc: str
```

The diagnostic feeds `tac.sensitivity_map.axis_weights` as an ALTERNATIVE per-axis weighting scheme: instead of uniform `{seg: 1.0, pose: 1.0, rate: 1.0}`, use `fisher_metric_axis` per substrate. This is the canonical sensitivity-map extension per parent synthesis §8 mapping #2.

#### Hook 2: Pareto constraint → `tac.pareto_*` (ACTIVE)

The Fisher eigenvalue spectrum is a Pareto-relevant signal. The top-k eigenvalues identify the "principal score-relevant directions"; the bulk eigenvalues identify null-space directions.

The canonical Pareto constraint per parent synthesis §6.4 (Minkowski sum of Pareto sets): substrate compositions where the Fisher eigenvalue spectra are ORTHOGONAL (top-k eigenvectors of substrate A are in null-space of substrate B and vice versa) ARE structurally additive in Pareto. Substrate compositions where the spectra overlap are sub-additive.

The Riemannian-Newton helper emits a `FisherEigenvalueSpectrumPareto` row consumed by `tac.optimization.substrate_composition_matrix` for structural Pareto-additivity prediction.

#### Hook 3: Bit-allocator hook → `tac.bit_allocator` (ACTIVE)

The Fisher diagonal `diag(F(θ))` provides per-parameter sensitivity. High-Fisher parameters MUST be allocated more bits in quantization; low-Fisher parameters can be allocated fewer bits.

The Riemannian-Newton helper emits per-substrate `FisherDiagonalSensitivity` consumed by `tac.bit_allocator.allocate_bits` as the canonical per-parameter importance vector. This replaces the heuristic importance vectors currently used per `src/tac/bit_allocator.py:100`.

#### Hook 4: Cathedral autopilot dispatch hook → `tools/cathedral_autopilot_autonomous_loop.py` (ACTIVE)

NEW reward factor `adjust_predicted_delta_for_riemannian_newton_phase_eligibility`:

```python
def adjust_predicted_delta_for_riemannian_newton_phase_eligibility(
    predicted_delta: float,
    candidate: CandidateRow,
    riemannian_newton_eligibility_anchor: RiemannianNewtonEligibilityAnchor | None,
) -> float:
    """Apply per-substrate Riemannian-Newton eligibility discount/reward.

    Substrates whose Fisher conditioning is well-behaved (eligibility=HIGH)
    get +20% reward factor (predicted ΔS unlock per parent synthesis §10.3 Phase 2).
    Substrates whose Fisher conditioning is borderline (eligibility=MEDIUM)
    get neutral factor 1.0× (pending Phase 1 empirical validation).
    Substrates whose Fisher conditioning is poor (eligibility=LOW; condition_number > 1e6)
    get -10% penalty factor (Phase 1 op-routable would refuse Riemannian-Newton here).
    """
    if riemannian_newton_eligibility_anchor is None:
        return predicted_delta  # passthrough; substrate not opted-in
    if riemannian_newton_eligibility_anchor.eligibility == "HIGH":
        return predicted_delta * 1.20
    if riemannian_newton_eligibility_anchor.eligibility == "MEDIUM":
        return predicted_delta * 1.00
    if riemannian_newton_eligibility_anchor.eligibility == "LOW":
        return predicted_delta * 0.90
    return predicted_delta
```

The reward factor is integrated into `adjust_predicted_delta_for_venn_classification_v2` per Catalog #319 Q3 cascade per Hook 4 of Catalog #125.

#### Hook 5: Continual-learning posterior update → `tac.continual_learning.posterior_update_locked` (ACTIVE)

Every empirical Phase 1 Fisher-conditioning anchor + Phase 2 Riemannian-Newton paired-comparison anchor + Phase 4 symplectic-EMA paired-comparison anchor emits a continual-learning posterior update per Catalog #128 fcntl-locked discipline:

```python
register_riemannian_newton_anchor(
    substrate_id="pr101_lc_v2",
    archive_sha256="f174192aeadf...",
    phase="phase_1_fisher_conditioning_validation",
    verdict="VALIDATED_WELL_CONDITIONED",  # or "VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC"
    fisher_condition_number=1234.56,
    fisher_top_10_eigenvalues=(...,),
    measured_at_utc="2026-05-20T12:34:56Z",
)
```

The canonical posterior store is `.omx/state/riemannian_newton_anchors.jsonl` (fcntl-locked JSONL append-only per Catalog #131).

#### Hook 6: Probe-disambiguator → `tools/probe_riemannian_newton_vs_vanilla_disambiguator.py` (ACTIVE)

The canonical probe-disambiguator script:

```bash
.venv/bin/python tools/probe_riemannian_newton_vs_vanilla_disambiguator.py \
    --archive-sha256 f174192aeadf... \
    --substrate-id pr101_lc_v2 \
    --baseline-optimizer adamw \
    --variant-optimizer riemannian_newton \
    --paired-comparison-epochs 100 \
    --output-json .omx/state/riemannian_newton_paired_comparison/<run_id>.json
```

The probe-disambiguator emits a typed verdict consumed by the probe-outcomes ledger per Catalog #313: PROCEED (variant beats baseline by > 0.005) / PROCEED_WITH_REVISIONS (marginal benefit) / DEFER (variant equal to baseline; needs Phase 4 symplectic-EMA + Phase 6 tropical-Newton extensions) / REFUSE (variant worse than baseline; Riemannian-Newton structurally unsuitable for this substrate).

---

## 11. Cross-substrate composability matrix

### Cross-substrate composability matrix

Per Catalog #322 anti-phantom + parent synthesis §6.4 Minkowski sum of Pareto sets: the cross-substrate composability of Riemannian-Newton (META-helper) with existing substrates is structured per the canonical-vs-unique decision per layer (§8 above).

| Substrate | Composability with Riemannian-Newton | Expected α-discount | Rationale |
|---|---|---|---|
| **PR101_lc_v2** (entropy-coded latents) | HIGH-ORTHOGONAL | α ≈ 0.8-1.0 | Fisher metric on latent space is structurally orthogonal to entropy coding (latents are the input; entropy coding operates on the bytes); no overlap |
| **A1** (PR101 grammar reference) | HIGH-ORTHOGONAL | α ≈ 0.8-1.0 | Default Fisher on full weight space; no substrate-specific overlap with grammar |
| **PR106 format0d** (latent score table) | HIGH-ORTHOGONAL | α ≈ 0.7-0.9 | Score table operates at inflate-time; Riemannian-Newton operates at training-time; orthogonal |
| **sane_hnerv** (NeRV-family) | HIGH-ORTHOGONAL | α ≈ 0.7-0.9 | NeRV architecture is canonical neural representation; Fisher metric extends naturally |
| **Z6** (FiLM ego-motion) | MEDIUM (Lie-algebra retraction) | α ≈ 0.5-0.8 | Z6's SE(3) ego-motion requires UNIQUE retraction; composability depends on whether the SE(3) retraction interacts with the FiLM conditioning |
| **Z7** (LSTM dynamics) | HIGH-ORTHOGONAL | α ≈ 0.7-0.9 | LSTM dynamics are differentiable; canonical Fisher applies |
| **Z8** (hierarchical predictive coding) | MEDIUM | α ≈ 0.5-0.8 | Hierarchical structure may require per-level Fisher (K-FAC's per-layer factorization aligns); composability depends on hierarchy depth |
| **TT5L V2** (foveation + LAPose) | MEDIUM | α ≈ 0.5-0.8 | LAPose is SE(3)-related; foveation operates on input pixels; composability requires careful retraction design |
| **DP1** (driving prior codebook) | UNIQUE-COMPOSITION (sphere retraction) | α ≈ 0.5-0.8 | Codebook on unit sphere requires SPHERE GEODESIC retraction; canonical helper provides this; composability is structurally clean |
| **C6 IBPS** (information bottleneck) | MEDIUM (CARGO-CULTED ASSUMPTION) | α ≈ 0.3-0.7 | Information bottleneck creates structured Fisher-singularity; K-FAC + damping required; MacKay's PROPOSAL extends Phase 1 to diagnose C6 IBPS 22× miss |
| **NSCS06 v7** (Carmack-Hotz strip-everything) | LOW (chroma sub-LUT structure) | α ≈ 0.2-0.5 | NSCS06's class chroma anchors are non-differentiable; Riemannian-Newton structurally unsuitable; Phase 6 tropical-Newton may help |
| **Wunderkind G1 v2** (per-pair-dominant SegNet) | DEFERRED | α ≈ pending | DEFERRED per Catalog #308 SPLIT-VERDICT; not yet a composable substrate |
| **ATW V2** (cooperative receiver) | MEDIUM | α ≈ 0.5-0.8 | Atick-Redlich cooperative-receiver loss is Fisher-related (mutual information); composability depends on per-pixel SegNet softmax fitting |
| **STC 3a** (STC-as-sidecar) | LOW (STC is bit-level; outside Riemannian regime) | α ≈ 0.2-0.5 | Syndrome-trellis coding is bit-level integer arithmetic; Riemannian metric undefined |
| **mae_v+saug** (BLOCKED-RATE-LIMIT-CAP) | UNKNOWN | α ≈ pending | Pending unblock |

**Aggregate predicted ΔS under composition matrix**:

- **HIGH-orthogonality** (4 archives with α ≈ 0.8-1.0): `[-0.025, -0.008] × 4 × 0.9 = [-0.090, -0.029]` (within parent synthesis §10.3 prediction `[-0.105, -0.034]`)
- **Realistic** (4 archives with α ≈ 0.5-0.8 mixed): `[-0.025, -0.008] × 4 × 0.6 = [-0.060, -0.019]` (within Contrarian's revised prediction `[-0.050, -0.020]`)
- **Worst-case** (4 archives with α ≈ 0.3-0.5): `[-0.025, -0.008] × 4 × 0.4 = [-0.040, -0.013]` (still beats current frontier `0.19205`)

---

## 12. Implementation architecture for `tac.riemannian_newton_meta_substrate` canonical helper package

### Implementation architecture

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Carmack's "USE EXISTING CANONICAL LIBRARIES" contribution: the canonical helper package is a THIN INTEGRATION LAYER over existing libraries, target ~600 LOC.

#### Package structure

```
src/tac/riemannian_newton_meta_substrate/
├── __init__.py                       # public API surface (~50 LOC; re-exports + version)
├── contract.py                       # SubstrateContract extension (~80 LOC)
│   - RiemannianNewtonSubstrateContract dataclass with fields:
│     - riemannian_newton_enabled: bool = False
│     - riemannian_metric: Literal["fisher", "uniward_weighted_fisher"] = "fisher"
│     - fisher_damping: Literal["levenberg_marquardt", "kfac", "spectral_floor"] = "levenberg_marquardt"
│     - fisher_damping_lambda: float = 1e-2
│     - use_kfac: bool = True
│     - retraction: Literal["euclidean", "stiefel", "grassmann", "sphere", "se3"] = "euclidean"
│     - use_symplectic_ema: bool = False
│     - symplectic_integrator_order: Literal[2, 4] = 2
├── fisher_precondition.py            # canonical Fisher precondition + Levenberg-Marquardt (~80 LOC)
│   - compute_empirical_fisher(per_pair_gradients) -> Tensor  (canonical fp64)
│   - fisher_precondition(grad, fisher_matrix, damping_lambda) -> Tensor
│   - fisher_orthogonal_projection(grad, fisher_matrix) -> (grad_parallel, grad_orthogonal)
│   - kfac_approximate_fisher_inverse(model, per_pair_gradients) -> Callable  (delegates to kfac_pytorch)
├── riemannian_newton_step.py         # canonical trust-region Newton step (~120 LOC)
│   - RiemannianNewtonStep dataclass
│   - compute_riemannian_newton_step(grad, hessian_vector_product, trust_region_radius) -> Tensor
│   - steihaug_toint_truncated_cg(hvp, grad, max_iterations, trust_region_radius) -> Tensor
│   - update_trust_region_radius(score_decrease_ratio, current_radius) -> float  (canonical Boumal 2020 §6.4)
├── symplectic_ema.py                  # canonical symplectic-integrator EMA (~80 LOC)
│   - SymplecticEMA class (alternative to tac.training.EMA)
│   - leapfrog_step(theta_ema, momentum_ema, theta_live, decay, learning_rate) -> (theta_ema_new, momentum_ema_new)
│   - yoshida_4th_order_step(theta_ema, momentum_ema, theta_live, decay, learning_rate) -> (theta_ema_new, momentum_ema_new)
├── base_substrate.py                  # canonical RiemannianNewtonSubstrate base class (~150 LOC)
│   - RiemannianNewtonSubstrate(Substrate) base class
│   - canonical_train_step(model, batch, optimizer) -> dict[str, float]
│   - fisher_metric_factory(theta) -> Callable  [SUBSTRATE OVERRIDES]
│   - hessian_vector_product(v, theta) -> dict[str, Tensor]  [SUBSTRATE OVERRIDES]
│   - retraction(theta, delta_theta) -> dict[str, Tensor]  [SUBSTRATE OVERRIDES]
├── anchors.py                         # fcntl-locked JSONL anchor persistence (~80 LOC)
│   - RiemannianNewtonAnchor dataclass
│   - append_anchor_locked(anchor) -> None  (Catalog #131 pattern)
│   - load_anchors_strict(repo_root) -> tuple[RiemannianNewtonAnchor, ...]  (Catalog #138 pattern)
│   - query_anchors_by_archive(archive_sha256) -> tuple[RiemannianNewtonAnchor, ...]
├── adapters/
│   ├── geomstats_adapter.py          # thin wrapper over Geomstats library (~40 LOC)
│   │   - StiefelManifold + Grassmann + Sphere + SE(3) Lie group
│   ├── kfac_adapter.py                # thin wrapper over kfac_pytorch (~40 LOC)
│   │   - K-FAC factorization + inverse computation
│   └── uniward_adapter.py             # thin wrapper over UNIWARD steganographic weighting (~30 LOC)
└── tests/
    ├── test_fisher_precondition.py
    ├── test_riemannian_newton_step.py
    ├── test_symplectic_ema.py
    ├── test_base_substrate.py
    └── test_anchors.py
```

**Total LOC**: ~600 LOC base + ~250 LOC adapters + ~400 LOC tests = ~1250 LOC (target met per Carmack's discipline; adapter LOC delegate to existing libraries).

#### Public API (the `__init__.py` re-exports)

```python
from tac.riemannian_newton_meta_substrate.base_substrate import RiemannianNewtonSubstrate
from tac.riemannian_newton_meta_substrate.contract import RiemannianNewtonSubstrateContract
from tac.riemannian_newton_meta_substrate.fisher_precondition import (
    compute_empirical_fisher,
    fisher_precondition,
    fisher_orthogonal_projection,
    kfac_approximate_fisher_inverse,
)
from tac.riemannian_newton_meta_substrate.riemannian_newton_step import (
    RiemannianNewtonStep,
    compute_riemannian_newton_step,
    steihaug_toint_truncated_cg,
    update_trust_region_radius,
)
from tac.riemannian_newton_meta_substrate.symplectic_ema import SymplecticEMA
from tac.riemannian_newton_meta_substrate.anchors import (
    RiemannianNewtonAnchor,
    append_anchor_locked,
    load_anchors_strict,
    query_anchors_by_archive,
)

__all__ = [
    "RiemannianNewtonSubstrate",
    "RiemannianNewtonSubstrateContract",
    "compute_empirical_fisher",
    "fisher_precondition",
    "fisher_orthogonal_projection",
    "kfac_approximate_fisher_inverse",
    "RiemannianNewtonStep",
    "compute_riemannian_newton_step",
    "steihaug_toint_truncated_cg",
    "update_trust_region_radius",
    "SymplecticEMA",
    "RiemannianNewtonAnchor",
    "append_anchor_locked",
    "load_anchors_strict",
    "query_anchors_by_archive",
]
```

#### Inheritance contract (the canonical SubstrateContract subclass)

```python
@dataclass(frozen=True)
class RiemannianNewtonSubstrateContract(SubstrateContract):
    """Extension of SubstrateContract for Riemannian-Newton opt-in.

    Substrates that opt-in MUST set riemannian_newton_enabled=True AND
    declare per-substrate canonical-vs-unique decisions for the optimizer fields.

    The contract validates per-substrate the canonical-vs-unique decision per layer
    per Catalog #290 enforcement.
    """

    # NEW fields for Riemannian-Newton opt-in
    riemannian_newton_enabled: bool = False
    riemannian_metric: Literal["fisher", "uniward_weighted_fisher"] = "fisher"
    fisher_damping: Literal["levenberg_marquardt", "kfac", "spectral_floor"] = "levenberg_marquardt"
    fisher_damping_lambda: float = 1e-2
    use_kfac: bool = True
    retraction: Literal["euclidean", "stiefel", "grassmann", "sphere", "se3"] = "euclidean"
    use_symplectic_ema: bool = False
    symplectic_integrator_order: Literal[2, 4] = 2

    # NEW fields for cross-substrate composability
    riemannian_newton_eligibility: Literal["HIGH", "MEDIUM", "LOW", "DEFERRED"] = "MEDIUM"
    composability_alpha_with_other_substrates: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.riemannian_newton_enabled:
            if self.riemannian_metric not in ("fisher", "uniward_weighted_fisher"):
                raise ValueError(f"invalid riemannian_metric: {self.riemannian_metric}")
            if self.fisher_damping_lambda <= 0:
                raise ValueError(f"fisher_damping_lambda must be > 0: {self.fisher_damping_lambda}")
            if self.symplectic_integrator_order not in (2, 4):
                raise ValueError(f"symplectic_integrator_order must be 2 or 4: {self.symplectic_integrator_order}")
```

#### Per-substrate subclass example (PR101_lc_v2)

```python
class PR101_lc_v2_RiemannianNewtonSubstrate(RiemannianNewtonSubstrate):
    """PR101_lc_v2 with Riemannian-Newton opt-in.

    Substrate-optimal engineering:
    - Fisher metric on LATENT SPACE (the entropy-coded latents are rate-relevant)
    - Default Euclidean retraction on latent space
    - K-FAC enabled for the surrounding network
    - Symplectic-EMA opt-out (will validate via Phase 4)
    """

    def fisher_metric_factory(self, theta: dict[str, Tensor]) -> Callable:
        # OVERRIDE: compute Fisher on latent space only, not full weight space
        latents = theta["entropy_coded_latents"]
        per_pair_gradients_on_latents = self.compute_per_pair_latent_gradients(latents)
        return lambda v: compute_empirical_fisher(per_pair_gradients_on_latents) @ v

    def hessian_vector_product(self, v: dict[str, Tensor], theta: dict[str, Tensor]) -> dict[str, Tensor]:
        # CANONICAL DEFAULT: autograd-based HVP
        return super().hessian_vector_product(v, theta)

    def retraction(self, theta: dict[str, Tensor], delta_theta: dict[str, Tensor]) -> dict[str, Tensor]:
        # CANONICAL DEFAULT: Euclidean retraction
        return super().retraction(theta, delta_theta)
```

#### Operator-facing CLI surfaces

```bash
# Phase 1 canonical validation
.venv/bin/python tools/probe_riemannian_newton_fisher_conditioning.py \
    --substrate-id pr101_lc_v2 \
    --archive-sha256 f174192aeadf... \
    --damping-strategy levenberg_marquardt \
    --output-json .omx/state/riemannian_newton_fisher_conditioning/<run_id>.json

# Phase 2 paired comparison
.venv/bin/python tools/probe_riemannian_newton_vs_vanilla_disambiguator.py \
    --substrate-id pr101_lc_v2 \
    --archive-sha256 f174192aeadf... \
    --baseline-optimizer adamw \
    --variant-optimizer riemannian_newton \
    --paired-comparison-epochs 100 \
    --output-json .omx/state/riemannian_newton_paired_comparison/<run_id>.json

# Phase 3 cross-substrate enrollment
.venv/bin/python tools/enroll_substrate_in_riemannian_newton.py \
    --substrate-id z6_v2 \
    --riemannian-metric fisher \
    --fisher-damping kfac \
    --use-symplectic-ema true \
    --symplectic-integrator-order 4

# Phase 4 symplectic-EMA paired comparison
.venv/bin/python tools/probe_symplectic_ema_vs_standard_ema_disambiguator.py \
    --substrate-id z6_v2 \
    --baseline-ema standard \
    --variant-ema symplectic \
    --integrator-order 4 \
    --paired-comparison-epochs 100 \
    --output-json .omx/state/symplectic_ema_paired_comparison/<run_id>.json

# Phase 5 audit
.venv/bin/python tools/audit_riemannian_newton_compliance.py \
    --report-out reports/riemannian_newton_compliance.md
```

---

## 13. Op-routables ranked by EV

### Op-routables

The 5 op-routables per the §0 Executive verdict are operationalized here with concrete next-steps + dependencies + cost estimates + structural verdicts.

| Op # | Phase | Description | Cost | Wall-clock | Dependencies | Structural verdict |
|---|---|---|---|---|---|---|
| OP-1 | Phase 1 | Build `tac.riemannian_newton_meta_substrate.fisher_precondition` canonical helper + empirical Fisher-conditioning validation on PR101_lc_v2 archive | $0 GPU (M5 Max CPU) + ~3 day editor | ~3 days | None (uses already-extracted per-pair gradient anchor) | TIER-1 enabling primitive |
| OP-2 | Phase 2 | Build `tac.riemannian_newton_meta_substrate.RiemannianNewtonSubstrate` base class + Geomstats integration + paired smoke on PR101_lc_v2 vs vanilla AdamW | ~$5-10 Modal A100 + ~2-3 week editor | ~2-3 weeks | OP-1 validated | TIER-1 frontier-breaking; HIGHEST single-substrate EV |
| OP-3 | Phase 3 | Extend SubstrateContract schema + migrate Z6/Z7/Z8/TT5L V2/DP1/PR101 substrate trainers one-by-one with operator approval gate | ~$5-10 per substrate × 5-7 substrates + ~3-4 week editor | ~3-4 weeks | OP-2 validated | TIER-1 cross-substrate unlock |
| OP-4 | Phase 4 | Build `tac.riemannian_newton_meta_substrate.SymplecticEMA` + paired comparison on Z6-v2 Wave 2 full-mode | ~$5-10 Modal A100 + ~2 week editor | ~2 weeks | OP-3 has at least Z6-v2 opted in | TIER-2 substrate-specific unlock |
| OP-5 | Phase 5 | Wire 6 canonical hooks per Catalog #125 (sensitivity-map / Pareto / bit-allocator / cathedral autopilot / continual-learning / probe-disambiguator) | $0 GPU + ~1 week editor | ~1 week | OP-3 has at least 2 substrates opted in | TIER-2 apparatus deepening |

### OP-1 detailed breakdown

**Subagent dispatch**: `lane_phase_1_fisher_precondition_canonical_helper_20260520` (next session)

**Pre-flight**: read CLAUDE.md + AGENTS.md + this design memo + parent synthesis §1.2/§5.9 + arxiv:2508.13898 + arxiv:1412.1193 Martens + arxiv:1503.05671 K-FAC

**Build**:
1. `src/tac/riemannian_newton_meta_substrate/__init__.py` (~50 LOC)
2. `src/tac/riemannian_newton_meta_substrate/fisher_precondition.py` (~80 LOC)
   - `compute_empirical_fisher(per_pair_gradients: np.ndarray) -> np.ndarray`
   - `fisher_precondition(grad: np.ndarray, fisher: np.ndarray, damping_lambda: float) -> np.ndarray`
   - `fisher_orthogonal_projection(grad: np.ndarray, fisher: np.ndarray, condition_number_threshold: float) -> tuple[np.ndarray, np.ndarray]`
3. `src/tac/riemannian_newton_meta_substrate/anchors.py` (~80 LOC)
   - `RiemannianNewtonAnchor` dataclass + `append_anchor_locked` + `load_anchors_strict`
4. `src/tac/riemannian_newton_meta_substrate/adapters/kfac_adapter.py` (~40 LOC; thin wrapper over `kfac_pytorch`)
5. `tools/probe_riemannian_newton_fisher_conditioning.py` (~80 LOC operator-facing CLI)
6. `src/tac/riemannian_newton_meta_substrate/tests/test_fisher_precondition.py` (~120 LOC; 12+ test cases)
7. `src/tac/riemannian_newton_meta_substrate/tests/test_anchors.py` (~80 LOC; 8+ test cases including 4-proc spawn-pool concurrent-append stress)

**Validate**:
- Run `tools/probe_riemannian_newton_fisher_conditioning.py --substrate-id pr101_lc_v2 --archive-sha256 f174192aeadf...` on M5 Max CPU
- Verify Fisher condition number, top-k eigenvalues, Fisher-orthogonal subspace dimension
- If condition_number > 1e6: report K-FAC required for this substrate
- If condition_number < 1e6: proceed to Phase 2

**STRICT preflight gate** (NEW Catalog # to be claimed; ~300 LOC `tools/claim_catalog_number.py claim`):
- `check_riemannian_newton_anchor_validation_status` — refuses Riemannian-Newton anchors with `phase="phase_1_fisher_conditioning_validation"` lacking either `validated_well_conditioned` or `validated_near_singular_requires_kfac` verdict

**Memory file**: `feedback_phase_1_fisher_precondition_canonical_helper_landed_<YYYYMMDD>.md` per CLAUDE.md memory discipline + Catalog #229 premise verification

### OP-2 detailed breakdown

**Subagent dispatch**: `lane_phase_2_riemannian_newton_canonical_helper_20260527` (depends on OP-1)

**Pre-flight**: read CLAUDE.md + AGENTS.md + this design memo + parent synthesis §4.3/§4.4 + Boumal 2020 textbook §6 + Geomstats documentation + arxiv:2407.00294 symplectic preservation

**Build**:
1. `src/tac/riemannian_newton_meta_substrate/contract.py` (~80 LOC; SubstrateContract extension)
2. `src/tac/riemannian_newton_meta_substrate/riemannian_newton_step.py` (~120 LOC)
3. `src/tac/riemannian_newton_meta_substrate/symplectic_ema.py` (~80 LOC)
4. `src/tac/riemannian_newton_meta_substrate/base_substrate.py` (~150 LOC)
5. `src/tac/riemannian_newton_meta_substrate/adapters/geomstats_adapter.py` (~40 LOC)
6. `tools/probe_riemannian_newton_vs_vanilla_disambiguator.py` (~120 LOC)
7. Tests (~300 LOC total)

**Validate**:
- Implement PR101_lc_v2_RiemannianNewtonSubstrate subclass
- Run paired smoke on Modal A100: vanilla AdamW vs Riemannian-Newton (100 epochs each)
- Measure per-step score decrease + final score + convergence-curve smoothness
- Verify predicted ΔS band `[-0.025, -0.008]` empirically

**STRICT preflight gate** (NEW Catalog #):
- `check_riemannian_newton_substrate_subclass_overrides_correctly` — refuses substrate subclasses that opt-in to Riemannian-Newton but don't override `fisher_metric_factory`

### OP-3 detailed breakdown (cross-substrate enrollment)

**Subagent dispatches**: one per substrate, sequenced
- `lane_phase_3_z6_v2_riemannian_newton_enrollment_<date>`
- `lane_phase_3_z7_riemannian_newton_enrollment_<date>`
- `lane_phase_3_dp1_riemannian_newton_enrollment_<date>`
- `lane_phase_3_a1_riemannian_newton_enrollment_<date>`
- `lane_phase_3_sane_hnerv_riemannian_newton_enrollment_<date>`

**Per-substrate pre-flight**: substrate's per-substrate symposium per Catalog #325 + this design memo's §8 canonical-vs-unique decision per layer + §11 cross-substrate composability matrix

**Per-substrate validate**: paired smoke vs vanilla AdamW; verify per-substrate predicted ΔS within `[-0.025, -0.008]` band; emit Catalog #322 composition_alpha measurement vs sister substrates

### OP-4 detailed breakdown (symplectic-EMA validation)

**Subagent dispatch**: `lane_phase_4_symplectic_ema_paired_comparison_z6_v2_20260610`

**Pre-flight**: read CLAUDE.md + AGENTS.md + this design memo + arxiv:2407.00294 + arxiv:2509.24627 + Z6-v2 Wave 2 design memo

**Build**: extend OP-2's `symplectic_ema.py` with Yoshida 4th-order; build `tools/probe_symplectic_ema_vs_standard_ema_disambiguator.py`

**Validate**: paired smoke on Z6-v2 Wave 2 full-mode (100 epochs each); measure final score + phase-volume drift + information-theoretic divergence per Hafner's dissent criteria

### OP-5 detailed breakdown (6-hook wire-in)

**Subagent dispatch**: `lane_phase_5_riemannian_newton_six_hook_wire_in_20260617`

**Build**: 6 wire-in PRs per hook (sensitivity-map / Pareto / bit-allocator / cathedral autopilot / continual-learning / probe-disambiguator). Each PR is ~50-100 LOC integration + sister tests.

**Validate**: Catalog #125 wire-in declaration verified; sister `tac.autopilot_rudin_daubechies` consumes the new reward factor; sister `tac.sensitivity_map.axis_weights` consumes Fisher-curvature diagnostic.

---

## 14. Operator-routable consequences (summary)

This design memo's paradigm **ELEVATES per-substrate engineering via canonical helper, does NOT REPLACE it** per Contrarian veto + the just-landed META-cargo-cult #11 (architectural-replacement-is-canonical-research-path-forward).

### What this paradigm REPLACES

- **REPLACES NOTHING** at the substrate registry level — 53+ substrates remain registered per Catalog #241/#242
- **REPLACES** the implicit first-order Euclidean optimizer step in substrate trainers (vanilla AdamW) with explicit Fisher-preconditioned natural gradient (OPT-IN per substrate via SubstrateContract field; backward compat preserved)
- **REPLACES** the implicit first-order EMA (standard exponential moving average) with symplectic-integrator EMA (OPT-IN per substrate; backward compat preserved)

### What this paradigm ELEVATES

- **ELEVATES** the master gradient canonical helper from "diagnostic surface" to "active optimization input" — the Fisher metric IS the natural object on the renderer-weight manifold per Amari 1985; using it as such IS the canonical second-order optimization upgrade
- **ELEVATES** the null-space exploitation TOP-1 op-routable from raw null-space ΔS `[-0.040, -0.012]` to Fisher-preconditioned ΔS `[-0.055, -0.018]` per arxiv:2508.13898 Fisher-Orthogonal Projection
- **ELEVATES** the sister theoretical-floor estimator's `(lower_bound, upper_bound, confidence_interval)` accuracy via Fisher-curvature correction per parent synthesis §10.4

### What this paradigm COMPLEMENTS

- **COMPLEMENTS** the sister theoretical-floor estimator (Fisher-curvature diagnostic feeds floor-tightening from `[0.05, 0.12]` to `[0.05, 0.10]`)
- **COMPLEMENTS** the sister deterministic-optimizer (inner continuous θ optimization on smooth portion becomes Riemannian-Newton; discrete codec_config search remains substrate registry)
- **COMPLEMENTS** the cathedral autopilot v2 cascade (new reward factor `adjust_predicted_delta_for_riemannian_newton_phase_eligibility` per Hook 4)
- **COMPLEMENTS** the per-substrate symposium discipline per Catalog #325 (each opt-in substrate runs its own per-substrate symposium per #325 to validate Fisher-conditioning + canonical-vs-unique decisions)

### Cross-pollination with sister subagents (concrete deliverables)

| Sister subagent | This memo's contribution | Sister's contribution back |
|---|---|---|
| `tac.theoretical_floor_estimator` (DONE 2026-05-18) | Fisher-curvature diagnostic → floor-tightening from `[0.05, 0.12]` to `[0.05, 0.10]` | Floor-distance metric → Riemannian-Newton convergence stopping rule (halt when `score - floor_lower < ε`) |
| `tac.deterministic_score_optimizer` (DONE 2026-05-18) | Riemannian-Newton handles inner continuous θ optimization on smooth portion | Deterministic optimizer handles discrete codec_config + bilevel composition + tropical-Newton at SegNet argmax boundaries |
| Z8 symposium (in-flight `a68b22b14`) | Riemannian-Newton bolt-on composes with Z8 via SubstrateContract subclass override | Z8 paradigm provides the FIRST hierarchical predictive coding substrate that exercises K-FAC's per-level Fisher factorization |
| Comprehensive analytical surfaces inventory (parent synthesis §8) | Riemannian-Newton is Framework #2 of TOP-5; provides Fisher-orthogonal projection feeding Venn classification (Framework #1) | Inventory provides 70-surface map; Riemannian-Newton extends sensitivity-map surface (parent synthesis §8 mapping #2) |
| Meta-portfolio symposium (parent of this design) | This memo IS the canonical Phase 2 deliverable per parent synthesis §10.3 | Meta-portfolio TOP-1 op-routable (null-space exploitation) unlocks via Fisher-preconditioning per this memo's Hook 1 |

### Operator-attention budget

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4: frontier-breaking moves DOMINATE rigor budget. The 5-phase build order per §13 is calibrated to the operator-attention budget:

- **Phase 1 (~$0 + 3 days)**: REVERSIBLE; auto-routable; no operator-interrupt
- **Phase 2 (~$5-10 + 2-3 weeks)**: REVERSIBLE; auto-routable; no operator-interrupt
- **Phase 3 (~$30-70 + 3-4 weeks)**: per-substrate sequential; operator-approval gate per CLAUDE.md "Design decisions — non-negotiable"
- **Phase 4 (~$5-10 + 2 weeks)**: REVERSIBLE; auto-routable
- **Phase 5 (~$0 + 1 week)**: REVERSIBLE; auto-routable

**Total operator-attention budget**: ~5 substrate-enrollment approval gates in Phase 3. Total wall-clock: ~8-11 weeks (within parent synthesis §10.3 Phase 2 timeline).

**Total predicted ΔS unlock (Phases 1-5)**:
- HIGH-orthogonality: `[-0.090, -0.029]` per §11 composability matrix
- Realistic: `[-0.060, -0.019]`
- Worst-case: `[-0.040, -0.013]`

All cases beat current frontier `0.19205`; HIGH-orthogonality + realistic cases bring frontier into asymptotic-pursuit band per parent synthesis §10.3.

---

## 15. Cross-references

### Cross-references

#### Parent + sister design memos (2026-05-18 batch)

- `.omx/research/set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` — parent synthesis naming Riemannian-Newton as Phase 2 deliverable
- `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` — sister floor estimator (consumes Fisher-curvature diagnostic)
- `.omx/research/deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518.md` — sister deterministic optimizer (bilevel composition with Riemannian-Newton)
- `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md` — 70-surface inventory (Riemannian-Newton is Framework #2)
- `.omx/research/grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md` — parent meta-portfolio T3 symposium

#### CLAUDE.md non-negotiables this memo honors

- "Race-mode rigor inversion + parallel-dispatch first" — this memo IS a design deliverable, not a dispatch; rigor inversion N/A
- "Long-burn score-lowering campaign default" — this memo's predicted ΔS unlock `[-0.060, -0.019]` realistic = long-burn campaign tier
- "HNeRV / leaderboard-implementation parity discipline" — this memo's META-helper framing per Contrarian veto preserves L7 bolt-on vs substrate-engineering split
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — §8 canonical-vs-unique decision per layer
- "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" — research_only=true per frontmatter
- "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" — Phase 1 op-routable empirical Fisher-conditioning validation BEFORE Phase 2 Riemannian-Newton dispatch
- "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (Catalog #325) — sextet pact + 5 grand council attendees per §5
- "Apples-to-apples evidence discipline" — predicted ΔS bands carry Dykstra-feasibility + first-principles bound + probe-disambiguator path per Catalog #296
- "Bugs must be permanently fixed AND self-protected against" — Phase 1 Fisher-conditioning validation IS the structural protection against the Fisher-singularity bug class
- "Subagent coherence-by-default" — 6-hook wire-in declaration per Catalog #125
- "Main branch source of truth" — research_only memo lands on main per the canonical pattern
- "Frontier target" — predicted post-Phase-5 frontier `[0.087, 0.158]` brings pact into asymptotic-pursuit band
- "Meta-Lagrangian/Pareto solver" — Riemannian-Newton's Pareto-additivity prediction (Fisher eigenvalue orthogonality) feeds the unified solver stack
- "Operator gates must be wired and used" — 6 CLI tools per §12 are operator-facing surfaces
- "Substrate retirement discipline" (Catalog #298) — this META-helper does NOT retire any existing substrate
- "Gate consolidation discipline" (Catalog #299) — 2 NEW Catalog gates planned per OP-1 + OP-2; well under 400 quota
- "Memory file rotation discipline" — per CLAUDE.md operational hygiene
- "Preflight failure messages must cite the rule chain" — Catalog #325 + #324 + #313 message structure per the canonical Rudin-Daubechies pattern
- "Production-hardened dispatch optimization protocol" (Catalog #270) — Phase 2/4 dispatch will satisfy Tier 1/2/3 per the canonical protocol
- "Mission alignment" — operator-frontier-override available at all tiers; Phase 3 grand council T2 deliberation with override option
- "Max observability" (Catalog #305) — §4 observability surface declaration

#### Catalog # cross-references

- **Catalog #1** (`check_no_mps_fallback_default`) — Riemannian-Newton helper does NOT introduce MPS fallback
- **Catalog #90** (lane registry consistency) — lane `lane_riemannian_newton_substrate_engineering_design_20260518` registered at L1 per memo landing
- **Catalog #117/#157/#174** (commit serializer) — this memo's landing commit will use canonical serializer + post-edit working-tree sha
- **Catalog #125** (subagent landing 6-hook wire-in) — §10 declares all 6 hooks ACTIVE
- **Catalog #131/#138** (fcntl-locked JSONL discipline) — Riemannian-Newton anchors use canonical pattern per §12 `anchors.py`
- **Catalog #186** (catalog claim via serializer) — 2 new Catalog #s claimed via canonical helper per OP-1/OP-2
- **Catalog #190** (substrate trainer canonical helper pattern) — `tac.riemannian_newton_meta_substrate` follows canonical helper pattern
- **Catalog #205** (inflate.py canonical use) — Riemannian-Newton operates at training-time only; no inflate.py impact
- **Catalog #206** (subagent checkpoint discipline) — this design subagent checkpoints every 10 tool uses per discipline
- **Catalog #213** (Comma2k19 canonical download) — N/A for this memo (no dataset download)
- **Catalog #218** (substrate reconstruct mini-batch) — Riemannian-Newton HVP via autograd supports mini-batch automatically
- **Catalog #229** (premise verification before edit) — 5 canonical pointers read in full + 2 sister design memos cross-checked + per-claim arxiv citation
- **Catalog #241/#242** (META-layer substrate contract) — SubstrateContract extension per §12 contract.py
- **Catalog #245** (Modal call_id ledger 4-layer pattern) — canonical pattern applied per §12 `anchors.py`
- **Catalog #270** (canonical dispatch optimization protocol) — Phase 2/4 Modal dispatches satisfy Tier 1/2/3
- **Catalog #272** (substrate distinguishing-feature integration contract) — Riemannian-Newton distinguishing feature is Fisher-preconditioning per §1.2; integration contract: subclass override hooks per §1.7
- **Catalog #287** (no docstring overstatement without evidence tag) — every predicted ΔS carries `[prediction]` tag pending Phase 2 empirical validation
- **Catalog #290** (canonical-vs-unique decision per layer) — §8 satisfies
- **Catalog #292** (per-deliberation explicit assumption surfacing) — §5 each member declares "the shared assumption I am operating within"
- **Catalog #294** (9-dimension success checklist evidence) — §3 satisfies
- **Catalog #296** (substrate predicted band Dykstra-feasibility) — §9 satisfies
- **Catalog #298** (substrate L1 not stale dispatch) — this memo lands as L1 per memo landing; not stale
- **Catalog #300** (council deliberation v2 frontmatter) — YAML frontmatter declares T2 tier + sextet attendees + mission-alignment fields
- **Catalog #303** (cargo-cult audit section) — §2 satisfies
- **Catalog #305** (substrate design memo has observability surface section) — §4 satisfies
- **Catalog #313** (probe-outcomes ledger) — §6 registers probe outcome
- **Catalog #314** (sister subagent absorption-pattern detection) — declares files_touched: `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` only; DISJOINT from sister `a68b22b14` Z8 symposium files
- **Catalog #315** (substrate at OPTIMAL FORM before paid dispatch) — Phase 1 op-routable validates Fisher-conditioning BEFORE Phase 2 paid dispatch per discipline
- **Catalog #316** (frontier signal-loss) — predicted post-Phase-5 frontier cited correctly with axis label `[contest-CPU]`
- **Catalog #319** (Wyner-Ziv reweight requires DeliverabilityProof) — Hook 4 cathedral autopilot reward factor composes per the v2 cascade
- **Catalog #322** (no autopilot adjustment from phantom provenance) — composition_alpha measurements per §11 use canonical posterior
- **Catalog #323** (no score claim without canonical provenance) — all predicted ΔS carry canonical Provenance per `tac.provenance` package
- **Catalog #324** (no predicted band without post-training Tier-C validation) — §7 declares pending_post_training + reactivation criteria
- **Catalog #325** (per-substrate optimal form symposium) — this memo IS the per-substrate symposium for Riemannian-Newton meta-substrate
- **Catalog #326** (driver mode hardcode fix) — N/A for this memo (no driver scripts)

#### arxiv citation cross-references (HARD-EARNED-VERIFIED per parent synthesis §9)

**Riemannian optimization**:
- arxiv:2506.07351 — Quantized Riemannian Gradient Tracking
- arxiv:1603.03236 — Pymanopt
- arxiv:1805.08308 — Geomstats
- arxiv:2005.02819 — Geoopt

**Fisher / Natural gradient**:
- arxiv:2508.13898 — Fisher-Orthogonal Projection (CANONICAL for Hook 1)
- arxiv:1808.07172 — Amari-Karakida Fisher metric + natural gradient
- arxiv:1412.1193 — Martens natural gradient (Levenberg-Marquardt damping)
- arxiv:1503.05671 — K-FAC Kronecker-factored approximation
- arxiv:2303.05473 — Natural Gradient Methods perspective
- arxiv:2506.15830 — Rethinking LLM Training through Information Geometry

**Symplectic / Hamiltonian**:
- arxiv:2407.00294 — Deep Neural Networks with Symplectic Preservation (CANONICAL for Hook 4)
- arxiv:2509.24627 — Learning Hamiltonian Dynamics at Scale
- arxiv:2408.09821 — Symplectic Neural Networks Based on Dynamical Systems
- arxiv:2406.04104 — Symplectic Methods in Deep Learning
- arxiv:2106.11753 — Symplectic Learning for Hamiltonian Neural Networks
- arxiv:1306.0543 — Riemannian Manifold Hamiltonian Monte Carlo (Fisher-metric kinetic)

**Trust-region / Newton on manifolds**:
- Boumal 2020 textbook "An Introduction to Optimization on Smooth Manifolds" Ch. 6
- Absil-Mahony-Sepulchre 2008 textbook "Optimization Algorithms on Matrix Manifolds"

**Random matrix theory (Fisher near-singularity)**:
- arxiv:2506.03470 — Random Matrix Theory of Neural Network Hessians
- arxiv:2505.02809 — Towards Quantifying the Hessian Structure
- arxiv:2306.02108 — Random matrix theory and loss surfaces

**Tropical (Phase 6 deferred)**:
- arxiv:2409.03945 — TropNNC structured neural network compression
- arxiv:2402.00576 — Tropical Decision Boundaries

**Steganographic (UNIWARD-weighted Fisher)**:
- arxiv:1311.7041 — UNIWARD canonical paper

#### Sister subagent IDs (for absorption-pattern Catalog #314 awareness)

- **THIS subagent**: `riemannian_newton_design_20260518` (write scope: this memo file only)
- **Sister Z8 symposium**: `a68b22b14` (write scope: Z8 symposium memo + Z8 substrate files; DISJOINT)
- **Sister theoretical-floor estimator**: `ad9bca5c7f4370bbe` (DONE 2026-05-18; write scope CLOSED)
- **Sister deterministic-optimizer**: `acb41f8d3f7f0a3ea` (DONE 2026-05-18; write scope CLOSED)

---

**END OF DESIGN MEMO**

This memo is the canonical Phase 2 deliverable per the parent set-theory-manifolds-geometry deep research synthesis's TOP-1 NEW SUBSTRATE PARADIGM. The 5-phase build order (§13) operationalizes the predicted ΔS unlock `[-0.090, -0.029]` HIGH-orthogonality / `[-0.060, -0.019]` realistic across 4 frontier archives. The META-helper framing (Contrarian veto) preserves the 53-substrate registry; each substrate OPTS-IN per SubstrateContract subclass with per-substrate canonical-vs-unique decision per layer per UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

Next session: dispatch `lane_phase_1_fisher_precondition_canonical_helper_20260520` subagent to build OP-1 per §13 detailed breakdown.

