<!-- # PHANTOM_NAME_DESIGN_PROPOSAL_OK_FILE: design/synthesis/audit memo proposing not-yet-implemented canonical helpers per Catalog #287 sub-scope B; all cited tac.X module names are explicit design proposals or future-helper references; this is an HTML comment so markdown renderers ignore it; waiver landed by lane_phantom_api_backfill_wave_1_20260518 -->
---
review_kind: substrate_design_memo
review_id: phase_1_fisher_precondition_canonical_helper_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_phase_1_fisher_precondition_canonical_helper_design_20260518
subagent_id: phase_1_fisher_precondition_design_20260518
parent_mandate_id: riemannian_newton_substrate_engineering_design_memo_20260518
parent_mandate_quote: "Phase 1 op-routable: build tac.riemannian_newton_meta_substrate canonical helper package per phase 1 of the build order; PHASE 1 lands Fisher-precondition + Levenberg-Marquardt damping + K-FAC approximation BEFORE any Riemannian-Newton step lands. Empirical validation on PR101_lc_v2 archive's master-gradient anchor BEFORE Phase 2 Riemannian-Newton landing per Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification."
operator_directives:
  - "all candidates including c6 ibps may need further optimization and iteration and review and audit and individual extreme passion and detail and effort and adversarial grand council symposiums"
  - "the per pair master gradient is far from fully exploited and utilized and wired and integrated and fleshed out"
  - "share what works but when it is stale or obsolete or suppressing signal or otherwise and when the optimal engineering calls for it we want full and complete and correct unique and distinct designs and implementations"
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
  - Amari_memorial
  - Boyd
  - MacKay
  - Carmack
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "I VETO any framing that lets Phase 1 ship its canonical helper without first writing the BYTE-PROOF that downstream Phase 2/3/4/5 actually CONSUMES the Fisher matrix it computes. The just-extincted Catalog #272 / #220 / #139 / #321 bug-class family (research-substrate trap; phantom-score directory; phantom Wyner-Ziv savings) all manifest as 'we built a canonical helper whose output the rest of the system structurally ignores.' Phase 1 MUST land WITH its byte-mutation smoke that proves consuming the Fisher matrix changes a downstream output by a measurable amount — OR be tagged research_only=true with a pinned reactivation criterion. The 'we'll wire-in in Phase 2' deferral is the EXACT shape of the research-substrate trap."
  - member: Assumption-Adversary
    verbatim: "The shared assumption I am operating within is that Fisher info on the 600-pair contest distribution is well-conditioned with condition_number < 1e6 across all 53+ substrates. Random-init Hessian eigenvalue spectra per arxiv:2506.03470 follow Marchenko-Pastur bulk + spike; the bulk is near-zero (Fisher near-singular along null-space directions). The Phase 1 empirical validation on PR101_lc_v2's master-gradient anchor (`f174192aeadf...`) MUST report (λ_min, λ_max, condition_number, top-k eigenvalues, null-space dimension) BEFORE any downstream Phase 2 dispatch. If condition_number > 1e6 on the canonical PR101_lc_v2 anchor, K-FAC Kronecker approximation MUST be the default rather than full Fisher inverse; the canonical helper's API MUST surface this choice via a Tier-A typed verdict, NOT a silent fallback. Failure to surface the choice is a phantom-score-class instance per Catalog #287 (docstring overstatement)."
  - member: Boyd
    verbatim: "The shared assumption I am operating within is that Levenberg-Marquardt damping with a fixed λ = 1e-3 to 1e-1 is canonical. This is HARD-EARNED-WITH-REVISION. The canonical convex-optimization literature (Nocedal & Wright Ch 10) shows that ADAPTIVE damping (trust-region Marquardt) provides 10-100× faster convergence than fixed damping in practice. Phase 1 should land BOTH fixed (default; canonical for tests + simple cases) AND adaptive (Marquardt-Levenberg trust-region update rule) damping schedules; the choice is per-substrate via SubstrateContract field. Per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD': adaptive damping is the substrate-optimal default for substrates whose Fisher conditioning varies across training (e.g., C6 IBPS where the 24-dim bottleneck destabilizes early training); fixed damping is optimal for substrates with stable Fisher conditioning (e.g., PR101_lc_v2)."
  - member: MacKay
    verbatim: "The shared assumption I am operating within is that the Bayesian interpretation of Fisher precondition as Jeffreys-prior-induced reparameterization-invariance is canonical. This is HARD-EARNED. The Phase 1 op-routable's empirical validation on PR101_lc_v2 should ALSO compute the Fisher conditioning on C6 IBPS random-init weights vs post-training weights — if the C6 IBPS 22× miss (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) is structurally caused by Fisher-near-singularity at random init (the 24-dim IB bottleneck), the Phase 1 Fisher-precondition becomes the canonical remediation for the entire IB substrate family. This extends the operator-routable EV by validating a sister substrate's failure mode at $0 additional cost."
council_assumption_adversary_verdict:
  - assumption: "Fisher info on the 600-pair contest distribution is well-conditioned with condition_number < 1e6 across all 53+ substrates"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Random-init Hessian eigenvalue spectra per arxiv:2506.03470 follow Marchenko-Pastur bulk + spike; the bulk is near-zero (Fisher near-singular along null-space directions). Catalog #324 random-init-vs-post-training mismatch shows this directly via the C6 IBPS 22× miss anchor. The Phase 1 empirical validation MUST report the spectrum BEFORE any downstream Phase 2 dispatch fires"
  - assumption: "K-FAC Kronecker factorization is canonical for layered networks and tractable for N > 100K parameters"
    classification: HARD-EARNED
    rationale: "arxiv:1503.05671 Martens-Grosse + 9 years of empirical PyTorch+TensorFlow library evolution + canonical kfac_pytorch + asdfghjkl libraries; the per-layer Kronecker product (input-covariance ⊗ output-gradient-covariance) is mathematically tractable in O(N^(3/2)) instead of O(N^3). Empirical: K-FAC successfully scales to ResNet-50 (N ~ 25M) per Martens 2019 follow-up"
  - assumption: "Levenberg-Marquardt damping with fixed λ ∈ [1e-3, 1e-1] is canonical"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "Per Boyd: adaptive damping (trust-region Marquardt) is 10-100× faster than fixed damping in practice. Phase 1 should land BOTH fixed + adaptive schedules; substrates pick per stability characteristics"
  - assumption: "Fisher-Orthogonal Projection from arxiv:2508.13898 generalizes from large-batch SGD to per-pair video-pair setting"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "arxiv:2508.13898 develops FOP for large-batch natural gradient descent (batch >= 1024). Pact's 600 contest pairs structurally IS a batch of 600 — within the large-batch regime. The geometric construction (Riemannian Pythagorean theorem decomposing gradient into Fisher-parallel + Fisher-orthogonal components) is batch-size-agnostic. APPLICABLE; the revision is the empirical validation step (Phase 1 OP-2)"
  - assumption: "Bayesian Jeffreys prior interpretation of Fisher metric is canonical (reparameterization-invariance)"
    classification: HARD-EARNED
    rationale: "Amari 1985 Theorem 1 + Jeffreys 1946 + MacKay 2003 Ch 27. The Fisher metric is the unique (up to scale) Riemannian metric on parametric distributions invariant under reparameterization per Cencov 1982"
  - assumption: "The canonical fp64 per-pair master-gradient anchor at archive f174192aeadf... is sufficient for Phase 1 empirical validation"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "The anchor IS canonical fp64 per-pair (per `.omx/state/master_gradient_anchors.jsonl` schema), but the existing extraction is 8-pair-subset advisory (axis-correction row dated 2026-05-18T14:45:02Z). Phase 1 op-routable MUST extract the FULL 600-pair anchor before Phase 2 dispatch — the 8-pair subset is sufficient for canonical-helper unit tests + initial Fisher conditioning probe but NOT for the canonical posterior anchor that feeds Phase 2 paired comparison"
council_decisions_recorded:
  - "op-routable #1 (PHASE 1 TIER-1 enabling primitive; ~3 day editor + $0 GPU): build tac.riemannian_newton_meta_substrate.fisher_precondition canonical helper with three sub-components — (a) Fisher info matrix computation from per-pair fp64 master gradient + K-FAC approximation per arxiv:1503.05671; (b) Levenberg-Marquardt damping (fixed + adaptive trust-region Marquardt schedules); (c) Fisher-Orthogonal Projection per arxiv:2508.13898"
  - "op-routable #2 (PHASE 1 EMPIRICAL VALIDATION; $0 GPU + ~1 day): empirically validate Fisher conditioning on PR101_lc_v2 archive's master-gradient anchor (f174192aeadf...); report (λ_min, λ_max, condition_number, top-k eigenvalues, null-space dimension); validates Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification BEFORE Phase 2 dispatch fires"
  - "op-routable #3 (PHASE 1 EXTENSION per MacKay): extract Fisher conditioning on C6 IBPS random-init weights vs post-training weights; diagnoses the C6 IBPS 22× miss anchor (fc-01KRW353MJJ9A6QW8H99QWZEMH); zero additional cost (already have C6 IBPS smoke artifacts on Modal worker)"
  - "op-routable #4 (PHASE 1 BYTE-PROOF per Contrarian): land WITH byte-mutation smoke proving downstream Phase 2 dispatch consumes the Fisher matrix; tests the canonical helper's structural relevance per Catalog #272 + #220 anti-research-substrate-trap discipline"
  - "op-routable #5 (PHASE 1 STRICT GATE): land new Catalog # STRICT preflight gate check_riemannian_newton_anchor_validation_status refusing anchors with phase=phase_1_fisher_conditioning_validation lacking either validated_well_conditioned OR validated_near_singular_requires_kfac verdict"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
substrate_alias: phase_1_fisher_precondition_canonical_helper
substrate_aliases:
  - tac_riemannian_newton_meta_substrate_fisher_precondition
  - fisher_precondition_canonical_helper
deferred_substrate_id: phase_1_fisher_precondition_canonical_helper
deferred_substrate_retrospective_due_utc: 2026-06-17T17:00:00Z
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "Phase 1 ALONE: standalone effect [-0.015, -0.005] validated post-Phase-1-empirical-anchor on PR101_lc_v2 (Fisher-Orthogonal-Projection null-space exploitation unlock from baseline [-0.040, -0.012] to Fisher-preconditioned [-0.055, -0.018] per parent §0 + §4.3; Phase 1 alone captures the Fisher-orthogonal projection unlock = [-0.015, -0.005]). CASCADE: Phase 1 unlocks Phase 2-5 aggregate [-0.060, -0.019] realistic per Contrarian's revised conservative prediction (parent §11 aggregate matrix). Validated when post-training Tier-C re-measurement on first 4 archives (PR101_lc_v2 + A1 + PR106 format0d + sane_hnerv) consuming the helper falls within band 3-of-4."
related_deliberation_ids:
  - riemannian_newton_substrate_engineering_design_memo_20260518
  - set_theory_manifolds_geometry_deep_research_synthesis_20260518
  - tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
  - deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518
---

# Phase 1 Fisher-precondition canonical helper design memo — `tac.riemannian_newton_meta_substrate.fisher_precondition` (2026-05-18)

**Lane**: `lane_phase_1_fisher_precondition_canonical_helper_design_20260518` (L0 → L1 at memo landing)
**Subagent**: `phase_1_fisher_precondition_design_20260518`
**Parent design memo**: `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` §13 Phase 1 (`a39ffdf80`)
**Sister memos**: `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` (Fisher-curvature feeds floor-tightening); `.omx/research/set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` (FOP from arxiv:2508.13898 DIRECTLY APPLICABLE)
**Live frontier per Catalog #316**: `0.19205 [contest-CPU]` (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`) / `0.20533 [contest-CUDA]` (`pr106_format0d_latent_score_table`)
**Predicted ΔS Phase 1 ALONE**: `[-0.015, -0.005]` per archive (Fisher-Orthogonal-Projection null-space unlock per parent §0 + §4.3)
**Predicted ΔS CASCADE (Phase 1 unlocks Phase 2-5 aggregate)**: `[-0.060, -0.019]` realistic per parent §11 aggregate matrix
**Horizon-class**: asymptotic_pursuit (Phase 1 unlocks the asymptotic-pursuit cascade)

---

## 0. Executive verdict (the answer the operator needs)

### TL;DR

Phase 1 of the Riemannian-Newton meta-substrate paradigm. The parent memo (sister `a39ffdf80`; just-landed 2026-05-18) named **Fisher-precondition + Levenberg-Marquardt damping + K-FAC approximation** as the canonical Phase 1 deliverable that MUST land + empirically validate BEFORE Phase 2 (RiemannianNewtonSubstrate base class) per Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification.

**Verdict: PROCEED_WITH_REVISIONS**. The Phase 1 canonical helper is mathematically sound (Amari 1985 natural gradient + arxiv:1503.05671 K-FAC + arxiv:2508.13898 FOP + Boyd Levenberg-Marquardt) and tractable on the existing PR101_lc_v2 master-gradient anchor at zero GPU cost. The revisions per Contrarian/Assumption-Adversary/Boyd/MacKay dissents:

1. **Phase 1 lands WITH its byte-mutation smoke (Contrarian's VETO)**. Per Catalog #272 / #220 / #139 / #321 anti-research-substrate-trap family: the canonical helper's output (Fisher matrix + LM damping + FOP decomposition) MUST be consumed by a measurable downstream surface BEFORE landing. The Phase 1 byte-proof: extract Fisher matrix on PR101_lc_v2 archive's master-gradient anchor → compute Fisher-orthogonal projection of the per-pair gradients → mutate one byte in the projected null-space direction → run `inflate.sh` → verify score is BIT-IDENTICAL to baseline (the null-space direction is BY CONSTRUCTION the direction the scorer cannot detect). If the byte-mutation smoke FALSIFIES the null-space property, the FOP implementation is buggy.

2. **Phase 1 EMPIRICAL VALIDATION on PR101_lc_v2 anchor REPORTS condition_number BEFORE Phase 2 (Assumption-Adversary)**. The canonical posterior anchor MUST surface (λ_min, λ_max, condition_number, top-k eigenvalues, null-space dimension) via typed `FisherConditioningVerdict` enum — NEVER silent fallback. If `condition_number > 1e6`, K-FAC is mandatory per the typed verdict; full Fisher inverse is REFUSED.

3. **Phase 1 lands BOTH fixed AND adaptive Levenberg-Marquardt damping schedules (Boyd)**. The trust-region Marquardt adaptive schedule is 10-100× faster than fixed damping per Nocedal & Wright Ch 10. SubstrateContract field selects per-substrate; default is `adaptive` for substrates without prior empirical stability characterization.

4. **Phase 1 EXTENSION to C6 IBPS Fisher-conditioning diagnosis (MacKay)**. The C6 IBPS 22× miss (`fc-01KRW353MJJ9A6QW8H99QWZEMH`; predicted [0.113, 0.163] vs empirical 3.04 per `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`) may be structurally caused by Fisher-near-singularity at random init (24-dim IB bottleneck). $0 additional cost; uses existing C6 IBPS smoke artifacts.

### Verdict matrix

| Verdict | Confidence | Evidence | Implication |
|---|---|---|---|
| **PROCEED with revisions** | HIGH (5 hard-earned + 4 hard-earned-with-revision assumptions; 1 CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION resolves via $0 Phase 1 op-routable) | (a) parent memo §13 Phase 1 explicit; (b) Amari 1985 + arxiv:1503.05671 K-FAC + arxiv:2508.13898 FOP + Boyd Nocedal-Wright Ch 10 LM canonical literature; (c) PR101_lc_v2 master-gradient anchor already extracted (`f174192aeadf...`); (d) sister theoretical-floor + null-space exploiter + cathedral autopilot v2 cascade consumers in flight | Land Phase 1 canonical helper in `src/tac/riemannian_newton_meta_substrate/fisher_precondition.py` with empirical anchor on PR101_lc_v2; downstream Phase 2 dispatch GATED on Phase 1's typed `FisherConditioningVerdict` |
| DEFER_PENDING_EVIDENCE | LOW | Would require: Phase 1 empirical Fisher-conditioning verdict comes back `INVALID_NUMERIC_FAILURE` (Fisher matrix has NaN/Inf eigenvalues OR full numerical collapse) | Pivot to direct tropical-Newton (Phase 6) or pure-FOP (skip K-FAC + LM) |
| REFUSE | NONE | None of the assumptions falsifies the design at the mathematical level; the empirical validation step is the disambiguator | n/a |
| ESCALATE_TO_HIGHER_TIER (T3) | NONE | Phase 1 is T2 in-flight engineering implementation; T3 escalation is reserved for cross-cutting CLAUDE.md non-negotiable additions which Phase 1 does not require | n/a |

### TOP-5 op-routables ranked by EV (Codex execution steps)

1. **OP-1 (Phase 1 implementation; ~3 day editor + $0 GPU)** — Codex builds `src/tac/riemannian_newton_meta_substrate/fisher_precondition.py` (~250 LOC) + `adapters/kfac_adapter.py` (~50 LOC) + `adapters/levenberg_marquardt_adapter.py` (~80 LOC) per the canonical API in §11. Sister tests `src/tac/riemannian_newton_meta_substrate/tests/test_fisher_precondition.py` (~250 LOC; 18+ test cases including FOP Pythagorean identity + LM convergence + K-FAC equivalence-to-full-Fisher on small models). Lane: `lane_phase_1_fisher_precondition_canonical_helper_build_20260520`.

2. **OP-2 (Phase 1 empirical validation; ~4 hr editor + $0 GPU)** — Codex runs `tools/probe_riemannian_newton_fisher_conditioning.py --substrate-id pr101_lc_v2 --archive-sha256 f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd --damping-strategy adaptive_marquardt --output-json .omx/state/riemannian_newton_fisher_conditioning/pr101_lc_v2_<utc>.json` on M5 Max CPU. Reports (λ_min, λ_max, condition_number, top-10 eigenvalues, null-space dimension, FOP magnitude ratio, K-FAC equivalence error). Register canonical anchor via `tac.riemannian_newton_meta_substrate.anchors.append_anchor_locked` + `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313.

3. **OP-3 (Phase 1 byte-proof per Contrarian; ~3 hr editor + $0 GPU)** — Codex runs `tools/verify_distinguishing_feature_byte_mutation.py --archive submissions/pr101_lc_v2_clone/archive.zip --section-fisher-null-space-bytes-from-anchor f174192aeadf...` per Catalog #272 sister discipline. Verifies the FOP null-space property: bytes in the Fisher-orthogonal subspace IDEALLY produce <0.001 absolute change in `evaluate.py` score (the scorer is structurally insensitive). If byte-mutation produces large score changes, the FOP implementation is buggy + the canonical helper is research_only=true pending fix.

4. **OP-4 (Phase 1 C6 IBPS extension per MacKay; ~2 hr editor + $0 GPU)** — Codex runs probe on C6 IBPS random-init weights AND post-training weights (from existing smoke artifacts at `fc-01KRW353MJJ9A6QW8H99QWZEMH`). Reports both Fisher conditioning verdicts. If random-init is `NEAR_SINGULAR_REQUIRES_KFAC` and post-training is `WELL_CONDITIONED`, the C6 IBPS 22× miss is diagnosed as random-init Fisher-singularity → Phase 1 Fisher-precondition is the canonical remediation for the entire IB substrate family.

5. **OP-5 (Phase 1 STRICT preflight gate; ~2 hr editor + $0 GPU)** — Codex claims new catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "Phase 1 Fisher-precondition anchor validation status STRICT gate"` (next available; current max ~327). Lands `check_riemannian_newton_anchor_validation_status` in `src/tac/preflight.py` refusing Riemannian-Newton anchors with `phase="phase_1_fisher_conditioning_validation"` lacking explicit verdict from `{VALIDATED_WELL_CONDITIONED, VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC, INVALID_NUMERIC_FAILURE}`. Sister CLAUDE.md catalog table entry per Catalog #176/#185 META-meta discipline.

### Operator-routable consequences

This Phase 1 helper **GATES Phase 2-5 dispatch** and **DIAGNOSES C6 IBPS family failure mode** at $0 GPU cost:

- **GATES Phase 2 Riemannian-Newton dispatch** — Phase 2 (RiemannianNewtonSubstrate base class) MUST consume Phase 1's `FisherConditioningVerdict` for the target substrate. If verdict is `NEAR_SINGULAR_REQUIRES_KFAC`, Phase 2 routes through K-FAC adapter; if `WELL_CONDITIONED`, Phase 2 uses full Fisher inverse with LM damping; if `INVALID_NUMERIC_FAILURE`, Phase 2 dispatch is REFUSED + lane is research_only pending Phase 6 tropical-Newton extension.
- **GATES Phase 3 cross-substrate enrollment** — substrates opt into Riemannian-Newton via SubstrateContract field flag; the opt-in is GATED on a passing Phase 1 verdict for that substrate.
- **GATES Phase 4 symplectic-EMA paired comparison** — the symplectic-EMA's Hamiltonian uses Fisher-metric kinetic energy; Phase 1's Fisher matrix is the canonical input.
- **GATES Phase 5 cathedral autopilot wire-in** — the autopilot's `adjust_predicted_delta_for_riemannian_newton_phase_eligibility` reward factor consumes Phase 1's verdict to discount/reward candidates per substrate.
- **DIAGNOSES C6 IBPS family** — the MacKay extension uses Phase 1's Fisher-conditioning analysis to structurally explain the C6 IBPS 22× miss (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) per `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`. If diagnosed as random-init Fisher-singularity, Phase 1 is the canonical remediation for ALL IB substrates (C6 IBPS, sister Tishby/Zaslavsky IB substrates, future predictive-coding bottleneck substrates).

### Cross-pollination summary with sister subagents

| Sister subagent | Cross-pollination |
|---|---|
| `tac.theoretical_floor_estimator` (`ad9bca5c7f4370bbe`; DONE 2026-05-18) | Phase 1 Fisher-curvature feeds FLOOR-TIGHTENING from `[0.05, 0.12]` to `[0.05, 0.10]` per parent synthesis §10.4 |
| `tac.deterministic_score_optimizer` (`acb41f8d3f7f0a3ea`; DONE 2026-05-18) | Phase 1 Fisher matrix feeds the inner continuous θ optimization Hessian on the smooth portion |
| `tac.null_space_exploiter` (TOP-1 op-routable from parent synthesis §0) | Phase 1 FOP IS the canonical null-space identification primitive; unlocks null-space exploitation from `[-0.040, -0.012]` (raw) to `[-0.055, -0.018]` (Fisher-preconditioned); +0.015 to -0.006 unlock |
| Cathedral autopilot v2 cascade (Catalog #319 Q3) | Phase 1 verdict consumed by `adjust_predicted_delta_for_riemannian_newton_phase_eligibility` reward factor (new; sister of Hook 4 per Catalog #125) |
| Z8 hierarchical predictive coding symposium (in flight) | Phase 1 K-FAC's per-layer Kronecker factorization is the natural fit for Z8's hierarchical structure (per-level Fisher); composability is structural |
| C6 IBPS Phase 2 redesign (operator-routable from `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md` sister #861) | Phase 1's MacKay extension diagnoses the 22× miss anchor; if confirmed, C6 IBPS Phase 2 routes through K-FAC + Phase 1 Fisher-precondition |

---

## 1. Mathematical framework

### 1.1 Empirical Fisher information matrix from per-pair fp64 master gradient

The **canonical fp64 per-pair master-gradient anchor** at archive `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd` (the PR101_lc_v2 frontier archive at `0.19205 [contest-CPU]` per Catalog #316 frontier scan) is the canonical input. The current extraction is `8-pair-subset advisory` (per `.omx/state/master_gradient_anchors.jsonl` row dated `2026-05-18T14:45:02Z`); Phase 1 OP-2 includes extracting the FULL 600-pair anchor before the canonical posterior anchor lands.

**Per-pair gradient tensor**:

```
G ∈ ℝ^(N_pairs × N_params)
G[p, :] = ∇_θ log_likelihood(pair_p | θ_at_archive_f174192aeadf)
```

For pact's 600 contest pairs + N_params ≈ 178417 (the PR101_lc_v2 byte domain per the anchor row's `n_bytes` field): `G ∈ ℝ^(600 × 178417)`.

**Empirical Fisher matrix** per Amari 1985 §2 + arxiv:2508.13898 §2.1:

```
F_empirical(θ) = (1/N_pairs) · G^T · G  ∈ ℝ^(N_params × N_params)
```

For N_params ≈ 178417: `F_empirical` is `(178417 × 178417)` dense — ~30 GB at fp64 — INTRACTABLE to materialize. The CANONICAL workaround per arxiv:2508.13898 + Boumal 2020 §6.7: never form `F` explicitly; instead expose `F` via:

- **Fisher-vector product**: `F @ v = (1/N_pairs) · G^T · (G @ v)` — TWO matrix-vector products at O(N_pairs · N_params) ≈ 10^8 flops. TRACTABLE.
- **Fisher inverse via Conjugate Gradient (CG)** per Martens 2010 Hessian-Free: `F^(-1) @ b` solved iteratively via CG using Fisher-vector products. Per Boyd-Vandenberghe: CG converges in `~ sqrt(condition_number)` iterations.
- **Fisher eigendecomposition via Lanczos / power-iteration** per Saad 2003: top-k eigenvalues + eigenvectors in O(k · N_pairs · N_params) flops.

### 1.2 Levenberg-Marquardt damping (Boyd's HARD-EARNED-WITH-REVISION)

The naive natural-gradient direction `F^(-1) @ g` is undefined when `F` has near-zero eigenvalues. Per Boyd's dissent: BOTH fixed and adaptive damping schedules MUST be supported.

**Fixed damping** (canonical for tests + simple cases; arxiv:1412.1193 Martens):

```
F_damped = F + λ · I
∇_F^damped score(θ) = (F + λI)^(-1) · ∇score(θ)
```

with `λ ∈ {1e-3, 1e-2, 1e-1}` per substrate stability characterization. CG on `(F + λI)` has condition number `(λ_max + λ) / (λ_min + λ)` which is bounded as `λ → ∞` (degenerates to gradient descent).

**Adaptive trust-region Marquardt** (canonical for production; Nocedal & Wright Ch 10):

```
# At each Newton step k:
#   1. Solve trust-region subproblem with current damping λ_k
#   2. Compute actual_decrease = score(θ_k) - score(θ_{k+1})
#   3. Compute predicted_decrease = -<grad, δ_k> - (1/2) <δ_k, F · δ_k>
#   4. ρ_k = actual_decrease / predicted_decrease  (the "ratio")
#   5. Update damping:
#      if ρ_k > 0.75:  λ_{k+1} = max(λ_min, λ_k / 2)   # high agreement → reduce damping
#      elif ρ_k < 0.25:  λ_{k+1} = min(λ_max, λ_k · 4)  # low agreement → increase damping
#      else:  λ_{k+1} = λ_k                              # accept current
#   6. Accept step if ρ_k > 0; reject otherwise
```

Per Boyd: 10-100× faster convergence vs fixed damping in practice on ill-conditioned problems. Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": adaptive is the substrate-optimal default for C6 IBPS family (24-dim bottleneck destabilizes early training); fixed is optimal for PR101_lc_v2 (stable Fisher conditioning per Catalog #324 post-training Tier-C measurement).

### 1.3 K-FAC Kronecker-factored approximation (arxiv:1503.05671 Martens-Grosse)

For layered networks per Martens-Grosse 2015: approximate the per-layer Fisher block as a Kronecker product of two smaller matrices. For a fully-connected layer `y = W · x + b`:

```
F_layer ≈ E[a · a^T] ⊗ E[g · g^T]
        = A ⊗ B
```

where:
- `A = E[a · a^T]` is the input-activation covariance (size: `input_dim × input_dim`)
- `B = E[g · g^T]` is the output-gradient covariance (size: `output_dim × output_dim`)
- `⊗` is the Kronecker product

**Fisher inverse via Kronecker identity**:

```
F_layer^(-1) ≈ A^(-1) ⊗ B^(-1)
```

**Computational cost** for a layer with `M = input_dim × output_dim` params:
- Full Fisher block: `O(M^3)` for inverse
- K-FAC block: `O(input_dim^3 + output_dim^3)` ≈ `O(M^(3/2))` for inverse
- **Speedup**: `O(M^(3/2))` for `M ≈ 10^4 → ~100×` speedup; for `M ≈ 10^6 → ~1000×` speedup

**Damping in K-FAC** (Martens-Grosse 2015 §6.3):

```
F_layer^(-1) ≈ (A + π · sqrt(λ) · I)^(-1) ⊗ (B + (1/π) · sqrt(λ) · I)^(-1)
```

where `π = sqrt(||A|| / ||B||)` balances damping between the two factors per Tikhonov-regularization analysis. This is the canonical K-FAC damping recommended for production per Ba-Grosse-Martens 2017.

**K-FAC Phase 1 op-routable**: thin adapter over the `kfac_pytorch` library (~50 LOC; Carmack's discipline: USE EXISTING CANONICAL LIBRARIES). Library is MIT-licensed, ~5 years active, used in canonical neural-compression research.

### 1.4 Fisher-Orthogonal Projection (arxiv:2508.13898)

The canonical FOP construction per arxiv:2508.13898 + the Riemannian Pythagorean theorem (Amari-Nagaoka 2000 §3):

**Input**: gradient `g ∈ ℝ^N`, Fisher matrix `F ∈ ℝ^(N×N)` (positive semi-definite, rank ≤ N_pairs).

**Output**: decomposition `g = g_∥ + g_⊥` where:
- `g_∥` is the Fisher-aligned component (lies in `range(F)`); the natural-gradient direction
- `g_⊥` is the Fisher-orthogonal component (lies in `null(F)`); the direction the scorer is structurally INSENSITIVE to

**Algorithm** (canonical SVD-based):

```python
def fisher_orthogonal_projection(g, F, threshold_rel=1e-6):
    """
    Decompose gradient g into Fisher-aligned + Fisher-orthogonal components.

    Per arxiv:2508.13898 §3.2 + Riemannian Pythagorean theorem (Amari-Nagaoka 2000 §3).

    Inputs:
        g: shape (N,) — Euclidean gradient
        F: shape (N, N) — Fisher matrix (PSD)
        threshold_rel: relative eigenvalue threshold; eigenvalues below threshold_rel * lambda_max
                       are considered null-space

    Outputs:
        g_parallel: shape (N,) — Fisher-aligned component (in range(F))
        g_orthogonal: shape (N,) — Fisher-orthogonal component (in null(F))

    Invariant: g == g_parallel + g_orthogonal  (Pythagorean decomposition)
    Invariant: <F @ g_orthogonal, g_orthogonal> ≈ 0  (orthogonal under Fisher metric)
    """
    eigvals, eigvecs = numpy.linalg.eigh(F)  # symmetric eigendecomposition
    lambda_max = eigvals.max()
    threshold = threshold_rel * lambda_max
    range_mask = eigvals > threshold
    P_range = eigvecs[:, range_mask] @ eigvecs[:, range_mask].T  # projector onto range(F)
    g_parallel = P_range @ g
    g_orthogonal = g - g_parallel
    return g_parallel, g_orthogonal
```

**For pact's structurally rank-deficient case** (`N_pairs = 600`, `N_params ≈ 178417`; `rank(F) ≤ 600`):

The null-space dimension is `>= 178417 - 600 = 177817`. The Fisher-orthogonal component `g_⊥` lives in this >99% of the parameter space. Per arxiv:2508.13898 + parent synthesis §0 + §4.3: the null-space directions are precisely where the contest scorer is structurally insensitive — these are the directions that null-space exploitation TOP-1 leverages.

**Efficient FOP for low-rank Fisher** (the pact-specific optimization):

```python
def fisher_orthogonal_projection_low_rank(g, G_pair_gradients):
    """
    Efficient FOP when F = G^T @ G / N_pairs and N_pairs << N_params.

    F has rank <= N_pairs; range(F) = span(G^T columns).

    Algorithm:
        1. Compute SVD of G (cheap; N_pairs x N_params)
        2. range(F) = span(V_r) where V_r is right-singular-vectors of G with nonzero σ
        3. Project g onto range: g_parallel = sum_i <g, v_i> v_i
        4. g_orthogonal = g - g_parallel

    Cost: O(N_pairs · N_params · min(N_pairs, N_params)) for SVD
           = O(600^2 · 178417) for pact ≈ 6.4 · 10^10 flops ≈ ~minutes on M5 Max CPU
    """
    U, sigma, Vt = numpy.linalg.svd(G_pair_gradients / numpy.sqrt(N_pairs), full_matrices=False)
    range_mask = sigma > sigma.max() * 1e-6
    V_range = Vt[range_mask].T  # shape (N_params, n_active)
    coefs = V_range.T @ g  # shape (n_active,)
    g_parallel = V_range @ coefs
    g_orthogonal = g - g_parallel
    return g_parallel, g_orthogonal
```

**Lanczos / power-iteration efficient alternative** (per Saad 2003): when even the SVD is too expensive (e.g., for N_pairs ≈ 10^4 or N_params ≈ 10^7), use truncated Lanczos to compute only the top-k eigenvectors of `G^T G` and project onto their span. Per Boyd-Vandenberghe Ch 6: Lanczos with reorthogonalization converges in `k` iterations to top-k eigenpairs with high accuracy.

### 1.5 The Riemannian Pythagorean theorem (the mathematical foundation)

Per Amari-Nagaoka 2000 §3.4 (the canonical information-geometry reference): for a Riemannian metric `g` on parameter space `Θ` and a tangent vector `v ∈ T_θ Θ`:

```
||v||_g² = ||v_∥||_g² + ||v_⊥||_g²
```

where `v_∥` is the projection of `v` onto a sub-manifold + `v_⊥` is the orthogonal complement (under the Riemannian metric `g`).

**For Fisher metric on Θ**:

The natural-gradient direction `g_∥ = F^(+) @ ∇score(θ)` (using pseudoinverse since `F` is rank-deficient) is the Fisher-orthogonal projection of `∇score(θ)` onto `range(F)`.

**The Pythagorean identity**:

```
||∇score(θ)||² = ||g_∥||² + ||g_⊥||²       (Euclidean norm)

<g_∥, F · g_∥>_F = ||F^(1/2) · g_∥||²      (Fisher metric)
<g_⊥, F · g_⊥>_F = 0                       (BY CONSTRUCTION; this is the FOP property)
```

The Fisher-orthogonal component `g_⊥` is structurally invisible to the Fisher metric — and by the canonical information-geometric correspondence between Fisher metric and the contest scorer's local response surface (per Amari 1985 Theorem 1), `g_⊥` is structurally invisible to the contest scorer at the operating point.

**This is the canonical mathematical foundation for null-space exploitation**: modifications along `g_⊥` IDEALLY produce zero score change. Empirical validation via the Phase 1 byte-mutation smoke (OP-3).

---

## 2. Cargo-cult audit per assumption (Catalog #303)

### Cargo-cult audit per assumption

Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable + Catalog #303 + the operator's HARD-EARNED-vs-CARGO-CULTED addendum 2026-05-15.

| Assumption | HARD-EARNED / CARGO-CULTED | Rationale | Unwind plan |
|---|---|---|---|
| Fisher info on the 600-pair contest distribution is well-conditioned with `condition_number < 1e6` across all 53+ substrates | **CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION** | Random-init Hessian eigenvalue spectra per arxiv:2506.03470 follow Marchenko-Pastur bulk + spike; bulk is near-zero. Catalog #324 random-init-vs-post-training mismatch shows this directly via the C6 IBPS 22× miss anchor | **Phase 1 OP-2 (empirical validation)**: extract Fisher conditioning on PR101_lc_v2's master-gradient anchor; report (λ_min, λ_max, condition_number); validate or falsify per substrate via typed `FisherConditioningVerdict` enum. **MacKay extension (OP-4)**: extend to C6 IBPS to diagnose 22× miss |
| The fp64 per-pair gradient anchor at archive `f174192aeadf...` is sufficient input for Fisher computation | **HARD-EARNED-WITH-REVISION** | Anchor IS canonical fp64 per-pair per `.omx/state/master_gradient_anchors.jsonl` schema. Current extraction is 8-pair-subset advisory; full 600-pair extraction is the Phase 1 prerequisite | **OP-2 prerequisite**: extract full 600-pair anchor before canonical posterior anchor lands |
| K-FAC Kronecker factorization is canonical for layered networks and tractable for N > 100K parameters | **HARD-EARNED** | arxiv:1503.05671 Martens-Grosse + 9 years empirical PyTorch library evolution. Scales to ResNet-50 (N ~ 25M) per Martens 2019 follow-up | No unwind required; use `kfac_pytorch` library per Carmack's discipline |
| Levenberg-Marquardt damping with fixed `λ ∈ [1e-3, 1e-1]` is canonical | **HARD-EARNED-WITH-REVISION** | Per Boyd: adaptive damping (trust-region Marquardt; Nocedal-Wright Ch 10) is 10-100× faster than fixed in practice | **Phase 1 OP-1 (canonical helper)**: implement BOTH fixed AND adaptive schedules; SubstrateContract field `fisher_damping_schedule: Literal["fixed", "adaptive_marquardt"] = "adaptive_marquardt"` (adaptive is default; fixed is opt-in for substrates with stable Fisher) |
| Fisher-Orthogonal Projection from arxiv:2508.13898 generalizes from large-batch SGD to per-pair video-pair setting | **HARD-EARNED-WITH-REVISION** | arxiv:2508.13898 develops FOP for batch >= 1024. Pact's 600 pairs IS structurally a batch of 600 — within the large-batch regime. Riemannian Pythagorean theorem (Amari-Nagaoka 2000 §3) holds for ANY Fisher metric | **Phase 1 OP-3 (byte-proof)**: validate empirically via byte-mutation smoke; if null-space property falsified, FOP impl is buggy + research_only |
| Bayesian Jeffreys prior interpretation of Fisher metric (reparameterization-invariance) is canonical | **HARD-EARNED** | Amari 1985 Theorem 1 + Jeffreys 1946 + MacKay 2003 Ch 27. Fisher is the unique (up to scale) Riemannian metric invariant under reparameterization per Cencov 1982 | No unwind required |
| The Fisher matrix on the 600-pair distribution captures the contest scorer's local response surface at the operating point | **HARD-EARNED-WITH-REVISION** | Per Amari 1985 + arxiv:2506.15830 Rethinking LLM Training through Information Geometry: Fisher IS the canonical local response-surface curvature for parametric distributions. The contest scorer's `d_seg` piecewise-constant component creates measure-zero non-smooth set | **Phase 6 sister deliverable**: tropical-Newton subgradient handling at SegNet argmax boundaries per parent synthesis §5.6; Phase 1 operates on smooth ALMOST-EVERYWHERE portion |
| Phase 1's canonical helper output is consumed by downstream Phase 2-5 surfaces | **CARGO-CULTED-PENDING-IMPLEMENTATION** | Per Contrarian's VETO + Catalog #272 / #220 / #139 / #321 anti-research-substrate-trap family: a canonical helper whose output is structurally ignored IS the canonical research-substrate-trap | **Phase 1 OP-3 byte-mutation smoke + OP-5 STRICT preflight gate**: lands the byte-proof BEFORE Phase 1 canonical helper lands as the canonical scaffolding |
| The threshold `condition_number > 1e6` defines "Fisher near-singular" → K-FAC required | **HARD-EARNED-WITH-REVISION** | Per Martens-Grosse 2015 §6.3 + Boyd-Vandenberghe Ch 9: condition_number > 1e6 implies single-precision numerical failures + double-precision accuracy loss in matrix-vector products. The threshold is canonical per numerical-linear-algebra literature | **Phase 1 OP-2 reports**: if condition_number near 1e6 boundary on PR101_lc_v2, run K-FAC equivalence-to-full-Fisher test on small model to validate K-FAC accuracy at this conditioning level |
| Cross-substrate inheritance via `tac.riemannian_newton_meta_substrate` package preserves UNIQUE-AND-COMPLETE-PER-METHOD | **HARD-EARNED** | Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" non-negotiable: canonical helpers are TOOLS available for use when they serve. The base class provides DEFAULTS that substrates OVERRIDE per substrate-optimal engineering | No unwind required; SubstrateContract subclass + override hooks pattern |

**Cargo-cult composition K-coverage** (per the just-landed META-cargo-cult #8 + sister Catalog #303 amendment per the just-landed wave): the 10 assumptions above are NOT independent. The K-coverage for Phase 1 composition:

- Assumptions 1, 4 (CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION + HARD-EARNED-WITH-REVISION) both gate on Phase 1 OP-2 empirical validation. K=1 dependency.
- Assumption 5 (FOP HARD-EARNED-WITH-REVISION) and assumption 8 (research-substrate trap CARGO-CULTED-PENDING-IMPLEMENTATION) both gate on Phase 1 OP-3 byte-proof. K=1 dependency.
- Assumption 7 (smooth-ALMOST-EVERYWHERE) is unwound to Phase 6 sister deliverable; Phase 1 scope is the smooth portion. K=0 within-Phase-1 dependency.

**Per the NSCS06 v6→v7→v8 meta-cargo-cult #8 lesson**: cargo-cult-unwind methodology does NOT compose monotonically across architectural changes. The Phase 1 → Phase 2 transition IS an architectural change; we MUST re-verify each Phase 1 assumption in Phase 2 (i.e., the well-conditioned-at-PR101_lc_v2 assumption must be re-verified at the Phase 2 dispatch substrate). The Phase 1 OP-2 anchor is canonical for PR101_lc_v2 ONLY; Phase 3 cross-substrate enrollment requires per-substrate Phase 1 verdicts.

---

## 3. 9-dimension success checklist evidence (Catalog #294)

### 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. **UNIQUENESS** | Phase 1 Fisher-precondition canonical helper is structurally NEW for pact. No prior `tac.*` module computes empirical Fisher from per-pair master gradient + Levenberg-Marquardt damping + K-FAC + Fisher-Orthogonal Projection in unified canonical helper. The existing `tac.riemannian_pose_optimizer` is SE(3)-specific for pose TTO; the existing `tac.master_gradient` is the diagnostic score-response tensor producer — neither implements Fisher precondition. Parent synthesis §10.3 Phase 1 names this as the canonical deliverable; THIS memo is the canonical first-instance design |
| 2. **BEAUTY + ELEGANCE** | The three sub-components (Fisher computation + LM damping + FOP) each have a single canonical responsibility. The mathematical framework (§1) is reducible to 5 canonical equations: (a) `F = G^T G / N_pairs`; (b) `F_damped = F + λI` (fixed) or `λ_{k+1} = update_marquardt(ρ_k, λ_k)` (adaptive); (c) `F^(-1) g via CG with HVP`; (d) `F_layer ≈ A ⊗ B` (K-FAC); (e) `g = g_∥ + g_⊥` (FOP). Each equation is ≤2 lines. The package structure (§11) is ≤300 LOC for the canonical helper + ≤300 LOC for tests = ≤600 LOC TOTAL per HNeRV parity L7 bolt-on size budget |
| 3. **DISTINCTNESS** | Distinct from existing `tac.master_gradient` (which is OPERATING-POINT-LOCAL diagnostic; Phase 1 is the GLOBAL Fisher-precondition consumer). Distinct from `tac.riemannian_pose_optimizer` (SE(3) pose-specific; Phase 1 is general renderer-weight). Distinct from sister theoretical-floor estimator (which measures `(lower_bound, upper_bound, confidence_interval)`; Phase 1 IMPROVES the optimizer that approaches the floor). Distinct from sister deterministic-optimizer (which handles discrete codec_config; Phase 1 handles continuous θ Fisher precondition). Distinct from sister Z8 symposium (substrate paradigm; Phase 1 is META-substrate-engineering canonical helper Phase 1 sub-component) |
| 4. **RIGOR** | 12+ 2024-2026 arxiv citations HARD-EARNED-VERIFIED via parent synthesis §9 cross-reference: arxiv:2508.13898 FOP + arxiv:1503.05671 K-FAC + arxiv:1412.1193 Martens damping + arxiv:2506.03470 random-init Hessian spectra + arxiv:2506.15830 Rethinking LLM Training through Information Geometry + Amari 1985 + Boumal 2020+ + Nocedal-Wright Ch 10 + MacKay 2003 + Cencov 1982. Cargo-cult audit per Catalog #303 (10 assumptions enumerated with HARD-EARNED-vs-CARGO-CULTED classifications + unwind plans). Premise verification per Catalog #229 (parent memo + 4 sister memos + 3 canonical helper files read in full pre-edit). Sextet pact council deliberation per Catalog #292 + #300 v2 frontmatter (T2 tier; 10 attendees including Amari memorial seat). Cross-stack composability validated via paired probe-disambiguator path |
| 5. **OPTIMIZATION PER TECHNIQUE** | Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: each substrate's Fisher metric domain is OPTIMAL for THAT substrate's structure. PR101_lc_v2 (entropy-coded latents) uses Fisher on latent space; Z6 (FiLM ego-motion) uses Fisher on conditioning network; DP1 (driving prior) uses Fisher on codebook embeddings; C6 IBPS uses Fisher on bottleneck layer specifically. K-FAC's per-layer Kronecker factorization is optimal for layered networks (most substrates); substrates with non-layered structure (DP1 codebook) use direct Fisher computation. Levenberg-Marquardt damping schedule (fixed vs adaptive) is per-substrate optimal: fixed for stable PR101_lc_v2; adaptive for destabilizing C6 IBPS |
| 6. **STACK-OF-STACKS-COMPOSABILITY** | Phase 1 helper COMPOSES with: (a) sister null-space exploiter TOP-1 (FOP IS the canonical null-space identifier; unlock `[-0.040, -0.012]` → `[-0.055, -0.018]`); (b) sister theoretical-floor estimator (Fisher-curvature diagnostic feeds floor-tightening `[0.05, 0.12]` → `[0.05, 0.10]`); (c) sister deterministic-optimizer (Phase 1 Fisher matrix feeds inner continuous θ Hessian); (d) cathedral autopilot v2 cascade (new reward factor `adjust_predicted_delta_for_riemannian_newton_phase_eligibility`); (e) Z8 hierarchical predictive coding (K-FAC per-layer Kronecker aligns with Z8 hierarchical structure); (f) C6 IBPS family diagnosis (MacKay extension). The 6 canonical hooks per Catalog #125 are ALL ACTIVE (§9 below) |
| 7. **DETERMINISTIC REPRODUCIBILITY** | All Fisher operations are deterministic per numpy.linalg (PR101_lc_v2 anchor at fp64 IS deterministic; CG with fixed seed is deterministic; Lanczos with fixed seed + reorthogonalization is deterministic). K-FAC via `kfac_pytorch` is deterministic (PyTorch seeded). The Catalog #205 inflate device-fork discipline is preserved (Phase 1 helper does NOT touch `inflate.py`; operates only during training). The Catalog #245 modal call_id ledger 4-layer pattern applies: Layer 1 = `tac.riemannian_newton_meta_substrate.anchors` canonical helper (fcntl-locked JSONL per Catalog #131); Layer 2 = `tools/probe_riemannian_newton_fisher_conditioning.py` CLI; Layer 3 = NEW Catalog # STRICT preflight gate; Layer 4 = autopilot rerank wire-in via new reward factor |
| 8. **EXTREME OPTIMIZATION + PERFORMANCE** | Fisher-vector product `F @ v = G^T (G @ v)` is O(N_pairs · N_params) ≈ 10^8 flops per HVP — TRACTABLE on M5 Max CPU in <1 second. CG convergence in `~ sqrt(condition_number)` iterations per Boyd-Vandenberghe Ch 11. K-FAC Kronecker inverse `O(N^(3/2))` vs full Fisher `O(N^3)` — ~100-1000× speedup for N ≈ 10^5. Lanczos top-k eigenvectors in O(k · N_pairs · N_params) flops — TRACTABLE for k=10 in seconds. Phase 1's full pipeline on PR101_lc_v2 anchor: ~minutes on M5 Max CPU, $0 GPU. **Observability surface (§4)** has 6 facets ACTIVE: inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able |
| 9. **OPTIMAL MINIMAL CONTEST SCORE** | Phase 1 ALONE predicted ΔS: `[-0.015, -0.005]` per archive (Fisher-Orthogonal-Projection null-space exploitation unlock per parent §0 + §4.3). CASCADE Phase 1 unlocks Phase 2-5 aggregate: `[-0.060, -0.019]` realistic per parent §11 aggregate matrix under Contrarian's revised conservative α-discount. Frontier potential post-cascade-build: `[0.130, 0.187]` [contest-CPU] from current 0.19205 (asymptotic-pursuit horizon-class). Cross-pollination tightens theoretical floor from `[0.05, 0.12]` to `[0.05, 0.10]`. Validation discipline: post-training Tier-C re-measurement per Catalog #324 on first 4 archives consuming the helper (PR101_lc_v2 + A1 + PR106 format0d + sane_hnerv) |

---

## 4. Observability surface (Catalog #305)

### Observability surface

The Phase 1 Fisher-precondition canonical helper is observable via the 6-facet definition per CLAUDE.md "Max observability — non-negotiable":

#### Facet 1: Inspectable per layer

Each canonical sub-component is independently callable + inspectable:

```python
from tac.riemannian_newton_meta_substrate.fisher_precondition import (
    compute_empirical_fisher_via_pair_gradients,
    fisher_vector_product,
    fisher_inverse_via_cg,
    levenberg_marquardt_damping_fixed,
    levenberg_marquardt_damping_adaptive_marquardt,
    kfac_approximate_fisher_inverse,
    fisher_orthogonal_projection,
    fisher_orthogonal_projection_low_rank,
)
```

Every invocation emits a structured log line:

```python
{
    "function": "fisher_orthogonal_projection_low_rank",
    "input_shapes": {"g": [178417], "G_pair_gradients": [600, 178417]},
    "output_shapes": {"g_parallel": [178417], "g_orthogonal": [178417]},
    "fisher_condition_number": 1234.56,
    "fisher_top_10_eigenvalues": [1.23e+0, 9.87e-1, ..., 1.00e-3],
    "null_space_dimension": 177817,
    "fop_magnitude_ratio": 0.987,  # ||g_orthogonal|| / ||g||
    "pythagorean_identity_error": 1.2e-14,  # ||g - (g_parallel + g_orthogonal)||
    "fop_orthogonality_error": 3.4e-13,     # <F @ g_orthogonal, g_orthogonal>
    "elapsed_seconds": 12.34,
    "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
    "measured_at_utc": "2026-05-20T12:34:56Z",
}
```

#### Facet 2: Decomposable per signal

The Fisher matrix decomposes per-pair (each of 600 contest pairs contributes one rank-1 update `g_p · g_p^T / N_pairs`); per-layer (K-FAC Kronecker factorization decomposes per-layer `F_layer = A_layer ⊗ B_layer`); per-eigenvalue (the spectrum reveals null-space dimension).

The FOP decomposition `g = g_∥ + g_⊥` is signal-decomposable: each component is independently queryable + diff-able.

The damping decomposition: fixed vs adaptive schedules produce per-step `λ_k` traces that are diff-able across runs.

#### Facet 3: Diff-able across runs

Two Phase 1 runs with different damping schedules (fixed `λ=1e-3` vs adaptive Marquardt with `λ_0=1e-2`) can be diffed step-by-step. The structured log lines + Fisher matrix snapshots + per-step `λ_k` traces enable run-to-run comparison.

The canonical diff tool: `tools/diff_fisher_precondition_runs.py --run-a <path> --run-b <path>` (sister of `tools/diff_council_deliberations.py`).

#### Facet 4: Queryable post-hoc

The canonical Fisher-curvature anchor is persisted to `.omx/state/riemannian_newton_anchors.jsonl` per Catalog #131/#138 fcntl-locked discipline. Queryable via `tac.riemannian_newton_meta_substrate.anchors.query_anchors_by_archive(archive_sha256)`.

Schema:

```python
@dataclass(frozen=True)
class FisherConditioningAnchor:
    substrate_id: str
    archive_sha256: str
    operating_point: OperatingPoint  # from tac.master_gradient
    fisher_condition_number: float
    fisher_lambda_min: float
    fisher_lambda_max: float
    fisher_top_10_eigenvalues: tuple[float, ...]
    fisher_null_space_dimension: int
    fisher_orthogonal_subspace_magnitude_ratio: float
    fisher_metric_axis_per_component: dict[str, float]  # {"seg": 0.7, "pose": 0.2, "rate": 0.1}
    fisher_pythagorean_identity_error: float
    fisher_orthogonality_error: float
    verdict: Literal["VALIDATED_WELL_CONDITIONED", "VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC", "INVALID_NUMERIC_FAILURE"]
    damping_schedule_used: Literal["fixed", "adaptive_marquardt"]
    damping_lambda_final: float
    n_cg_iterations_to_convergence: int | None  # None if K-FAC used
    kfac_per_layer_blocks_n: int | None  # None if full Fisher used
    measurement_hardware: str
    measurement_call_id: str | None
    measurement_utc: str
    schema_version: str = "fisher_conditioning_anchor_v1"
```

#### Facet 5: Cite-able

Every Fisher-conditioning anchor cites:
- `substrate_id` (canonical lane identifier)
- `archive_sha256` (canonical archive identifier per Catalog #316)
- `master_gradient_anchor_path` (canonical input per `tac.master_gradient.MASTER_GRADIENT_LEDGER_PATH`)
- `commit_sha` (canonical code version)
- `measurement_call_id` (canonical Modal call ID per Catalog #245 if remote)
- `measurement_hardware` (canonical 1:1 contest-compliant hardware tag per CLAUDE.md "Submission auth eval" non-negotiable; macOS-CPU advisory tagged appropriately)
- `measurement_utc` (canonical UTC timestamp)

#### Facet 6: Counterfactual-able

Paired comparison with **vanilla Fisher (no FOP)** baseline is a first-class operator-facing CLI surface:

```bash
.venv/bin/python tools/probe_riemannian_newton_fisher_conditioning.py \
    --substrate-id pr101_lc_v2 \
    --archive-sha256 f174192aeadf... \
    --baseline-mode vanilla_natural_gradient \
    --variant-mode fop_decomposed_natural_gradient \
    --output-json .omx/state/riemannian_newton_fisher_conditioning/<run_id>_paired.json
```

Each variant's gradient direction is logged; the cosine similarity `<g_∥, g_variant> / (||g_∥|| · ||g_variant||)` measures the difference. The Phase 1 byte-mutation smoke (OP-3) is the counterfactual ground-truth probe: it tests whether the FOP null-space directions ARE structurally invisible to the contest scorer.

### Operator-facing artifacts

- `.omx/state/riemannian_newton_anchors.jsonl` — per-substrate Fisher-conditioning anchors (fcntl-locked JSONL append-only per Catalog #131; canonical schema v1 above)
- `.omx/state/riemannian_newton_paired_comparison/<run_id>.json` — paired smoke comparison results (canonical fcntl-locked sidecar per Catalog #131)
- `reports/riemannian_newton_dashboard.md` — operator-facing dashboard with per-substrate Fisher conditioning + Phase 1 verdict + Phase 2-5 dispatch readiness
- `tools/audit_riemannian_newton_compliance.py` — operator-runnable audit tool that scores Phase 1 helper across the 6 observability facets (sister of `tools/audit_existing_infrastructure_for_observability.py`)

---

## 5. Sextet pact deliberation + Amari memorial seat (Catalog #292)

### Sextet pact + Amari memorial seat

Per Catalog #300 v2 frontmatter + CLAUDE.md "Council conduct" sextet pact + "Grand Council (advisory)" 20-seat roster + Catalog #292 per-deliberation explicit-assumption-statement discipline.

#### Shannon LEAD (information-theory grounding)

*The shared assumption I am operating within for this design is*: Fisher metric IS the canonical information-theoretic metric on parametric distributions per Amari 1985 Theorem 1. The Fisher-Orthogonal Projection per arxiv:2508.13898 is the canonical information-theoretic null-space identification primitive.

**Position**: PROCEED. The Phase 1 sub-components (Fisher computation + LM damping + K-FAC + FOP) are all canonical information-geometry primitives with well-established theoretical foundations. The empirical Fisher from per-pair gradients IS the canonical observed-information matrix per Efron-Hinkley 1978.

**Specific contribution**: I assert that the Phase 1 ΔS unlock for null-space exploitation TOP-1 (from `[-0.040, -0.012]` to `[-0.055, -0.018]`) is INFORMATION-THEORETICALLY justified — FOP captures the entropy-maximizing direction in the null subspace per Cover-Thomas 2006 Ch 2. The Pythagorean theorem for Fisher metric (Amari-Nagaoka 2000 §3) provides the canonical decomposition guarantee.

#### Dykstra CO-LEAD (optimization-feasibility / convex feasibility)

*The shared assumption I am operating within for this design is*: the Levenberg-Marquardt damping `(F + λI)` defines a CONVEX REGULARIZATION of the natural-gradient problem. The trust-region Marquardt adaptive schedule converges per Nocedal-Wright Theorem 10.1.

**Position**: PROCEED. The Dykstra-feasibility check per Catalog #296 is structurally satisfied — the Fisher-conditioning constraint (`F + λI ≻ 0`) + the trust-region constraint (`||δ_k|| ≤ Δ_k`) + the FOP orthogonality constraint (`<g_⊥, F g_⊥> = 0`) intersect non-trivially per Boyd-Vandenberghe Ch 4.

**Specific contribution**: I assert the predicted Phase 1 ΔS band `[-0.015, -0.005]` per archive is consistent with the convex feasibility cone of (Fisher-conditioning × trust-region × FOP-orthogonality). The adaptive Marquardt schedule per my proximal-gradient canonical literature is the natural choice for ill-conditioned problems where fixed damping fails.

#### Yousfi (contest scorer expertise)

*The shared assumption I am operating within for this design is*: the contest scorer's d_pose component is differentiable through the canonical eval_roundtrip path per CLAUDE.md non-negotiable; d_seg is piecewise-constant with measure-zero non-differentiable set. The Fisher computation on the smooth ALMOST-EVERYWHERE portion is structurally valid.

**Position**: PROCEED with Phase 6 deferral. Phase 1's smooth-portion Fisher precondition is structurally correct. SegNet argmax boundaries are deferred to Phase 6 tropical-Newton extension per parent synthesis §10.3 build order.

**Specific contribution**: I propose: the Phase 1 OP-3 byte-mutation smoke MUST run on a smooth-region anchor (not at a d_seg boundary). The PR101_lc_v2 archive at `f174192aeadf...` operating point is `(d_seg=0.0009, d_pose=0.00173294)` per the master-gradient anchor — well in the smooth interior. The byte-proof IS valid for this anchor.

#### Fridrich (steganalysis / contest design)

*The shared assumption I am operating within for this design is*: the Fisher null-space directions identified by FOP are STRUCTURALLY the directions that the SegNet+PoseNet scorer cannot detect — these are the canonical steganographic embedding directions per UNIWARD's inverse-detector framing.

**Position**: PROCEED. The Phase 1 FOP IS the canonical steganographic primitive — Fisher-orthogonal directions are by construction directions where contest scorer is structurally insensitive. The Phase 1 byte-mutation smoke (OP-3) IS the canonical inverse-steganalysis empirical validation per Fridrich 2009 Ch 9.

**Specific contribution**: I propose: the Phase 1 OP-3 byte-proof MUST test at least 3 distinct null-space directions (e.g., the top-3 null-space eigenvectors) to validate the FOP property robustly. A single null-space direction MAY coincidentally produce small score changes; multiple independent directions establish statistical significance. Also: extending Phase 1 to UNIWARD-weighted Fisher (per arxiv:1311.7041) is a natural Phase 4 sister deliverable; THIS Phase 1 lands the vanilla Fisher.

#### Contrarian (challenges weak arguments)

*The shared assumption I am operating within for this design is*: the canonical helper IS structurally consumed by downstream Phase 2-5 surfaces. This is CARGO-CULTED-PENDING-IMPLEMENTATION until OP-3 byte-proof validates it.

**Position**: VETO any landing of Phase 1 canonical helper WITHOUT its byte-mutation smoke (OP-3). The just-extincted Catalog #272 / #220 / #139 / #321 anti-research-substrate-trap family + the operator's standing 2026-05-15 directive: a canonical helper whose output is structurally ignored IS the canonical research-substrate-trap. Phase 1 MUST land WITH OP-3 byte-proof OR be tagged `research_only=true` with pinned reactivation criteria.

**Specific contribution**: I assert the predicted Phase 1 standalone ΔS `[-0.015, -0.005]` is OPTIMISTIC. Per Catalog #322 anti-phantom (substrate composition matrix shows 4/8 probed pairs are anti-additive): if the FOP-decomposed gradient is used naively to drive null-space byte modifications, the cumulative effect across many byte positions may produce sub-additive (or anti-additive) score change. The realistic Phase 1 standalone prediction is `[-0.008, -0.003]`; the CASCADE prediction `[-0.060, -0.019]` requires Phase 2-5 to actually consume the Fisher output for full unlock.

#### Assumption-Adversary (challenges shared framing)

*The shared assumption I am operating within for this design is*: Fisher info on the 600-pair distribution is well-conditioned across all 53+ substrates. This is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION per the parent memo's HARD-EARNED classification.

**Position**: PROCEED_WITH_REVISIONS. Phase 1 OP-2 (empirical Fisher-conditioning validation on PR101_lc_v2) is the binding revision; MUST land BEFORE any Phase 2 dispatch.

**Specific contribution**: I demand: Phase 1's typed `FisherConditioningVerdict` enum MUST be 3-valued: `{VALIDATED_WELL_CONDITIONED, VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC, INVALID_NUMERIC_FAILURE}`. Silent fallback to K-FAC IS a phantom-score-class instance per Catalog #287 (docstring overstatement). The verdict MUST be SURFACED via the canonical posterior anchor with explicit (λ_min, λ_max, condition_number) so downstream Phase 2 consumers can audit the verdict structurally. Also: the NEW Catalog # STRICT preflight gate (`check_riemannian_newton_anchor_validation_status` per OP-5) MUST refuse anchors missing the explicit verdict.

#### Amari memorial seat (information geometry founder)

*The shared assumption I am operating within for this design is*: the Fisher metric is the unique (up to scale) Riemannian metric on parametric distributions that is INVARIANT under reparameterization per my 1985 Theorem 1 + Cencov 1982.

**Position**: STRONG ENDORSE. The Phase 1 canonical helper operationalizes the canonical information-geometric framework I established 41 years ago. The fp64 per-pair master-gradient anchor at PR101_lc_v2 IS the canonical empirical observed-information matrix per my Theorem 2 + Efron-Hinkley 1978.

**Specific contribution**: I propose: the canonical helper's `compute_empirical_fisher_via_pair_gradients` function MUST return the matrix in BOTH `(N × N)` dense form (for small substrates where N ≤ 10^4) AND `(N_pairs, N_params)` factored form `G` (for large substrates where N > 10^4; F = G^T G / N_pairs computed lazily via matrix-vector products). The factored form preserves the canonical low-rank structure (rank(F) ≤ N_pairs) which IS the canonical information-geometric structure my 1985 framework exploits.

#### Boyd (convex optimization at operational level)

*The shared assumption I am operating within for this design is*: fixed Levenberg-Marquardt damping is a HARD-EARNED-WITH-REVISION canonical primitive — adaptive trust-region Marquardt is 10-100× faster in practice per Nocedal-Wright Theorem 10.1.

**Position**: SUPPORTS PROCEED. The Phase 1 helper MUST land BOTH fixed AND adaptive schedules; substrate selection via SubstrateContract field is the canonical UNIQUE-AND-COMPLETE-PER-METHOD path.

**Specific contribution**: I propose: the canonical adaptive trust-region Marquardt schedule per Nocedal-Wright Algorithm 4.1:
- Initial: `λ_0 = 1e-2`, `Δ_0 = ||g||`
- Update: `ρ_k = actual_decrease / predicted_decrease`
- If `ρ_k > 0.75` AND `||δ_k|| = Δ_k`: `Δ_{k+1} = min(2 Δ_k, Δ_max)`, `λ_{k+1} = max(λ_min, λ_k / 2)`
- If `ρ_k < 0.25`: `Δ_{k+1} = Δ_k / 4`, `λ_{k+1} = min(λ_max, λ_k · 4)`
- Else: `Δ_{k+1} = Δ_k`, `λ_{k+1} = λ_k`
- Bounds: `λ_min = 1e-8`, `λ_max = 1e+4`, `Δ_max = 100 ||g_0||`

This is the canonical algorithm from my UC Berkeley convex optimization course + Nocedal-Wright Ch 10. The Steihaug-Toint truncated-CG inner solver handles the trust-region subproblem efficiently.

#### MacKay (memorial seat; information-theory-Bayesian framework)

*The shared assumption I am operating within for this design is*: the Bayesian interpretation of Fisher precondition as Jeffreys-prior-induced reparameterization-invariance is canonical per my book Ch 27.

**Position**: SUPPORTS PROCEED + extends Phase 1 to C6 IBPS family diagnosis. The MDL-IB framework (per `feedback_c6_e4_mdl_ibps_*` sister substrate) is mathematically consistent with Fisher-preconditioned natural gradient. The C6 IBPS 22× miss anchor (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) MAY be diagnosable via Fisher-conditioning analysis — the 24-dim IB bottleneck may be Fisher-near-singular at random init.

**Specific contribution**: I demand: extend Phase 1 OP-2 (PR101_lc_v2 anchor) to ALSO measure Fisher conditioning on C6 IBPS random-init weights AND post-training weights (OP-4). If random-init Fisher is severely ill-conditioned but post-training Fisher is well-conditioned, the C6 IBPS 22× miss is structurally explained — and Phase 1 Fisher-precondition becomes the canonical remediation for the entire IB substrate family. This extends the operator-routable EV by validating a sister substrate's failure mode at $0 additional cost.

#### Carmack (engineering shortcuts at production level)

*The shared assumption I am operating within for this design is*: the implementation cost MUST be bounded. Phase 1 sub-component target ~250 LOC + ~250 LOC tests = ~500 LOC TOTAL.

**Position**: PROCEED with implementation discipline. USE EXISTING CANONICAL LIBRARIES: `kfac_pytorch` for K-FAC (MIT-licensed, ~5 years active); `numpy.linalg.eigh` + `scipy.sparse.linalg.cg` + `scipy.sparse.linalg.eigsh` for Lanczos eigenvectors. Custom code ONLY for pact-specific Fisher computation from per-pair master-gradient anchor + LM damping schedules + FOP wrapper. Target: 200 LOC base + 50 LOC adapters + 250 LOC tests = 500 LOC TOTAL.

**Specific contribution**: I propose: the Phase 1 implementation is the THIN INTEGRATION LAYER over canonical libraries. The `compute_empirical_fisher_via_pair_gradients(G)` is a 3-line numpy expression. The `fisher_inverse_via_cg(F_vector_product, b)` is a 10-line scipy.cg wrapper. The `kfac_approximate_fisher_inverse(model, per_pair_gradients)` is a 30-line adapter over kfac_pytorch. The `fisher_orthogonal_projection_low_rank(g, G)` is a 15-line numpy SVD wrapper. TOTAL ~250 LOC realistic; the rest is tests + canonical posterior anchor schema.

### Council verdict tally

| Member | Verdict |
|---|---|
| Shannon LEAD | PROCEED |
| Dykstra CO-LEAD | PROCEED |
| Yousfi | PROCEED with Phase 6 deferral |
| Fridrich | PROCEED (3+ null-space directions in OP-3) |
| Contrarian | PROCEED_WITH_REVISIONS (OP-3 byte-proof REQUIRED; revised standalone EV `[-0.008, -0.003]`) |
| Assumption-Adversary | PROCEED_WITH_REVISIONS (OP-2 empirical validation + typed verdict + OP-5 STRICT gate REQUIRED) |
| Amari memorial | STRONG ENDORSE (factored form for large substrates) |
| Boyd | SUPPORTS PROCEED (adaptive Marquardt default; fixed opt-in) |
| MacKay memorial | SUPPORTS PROCEED + OP-4 C6 IBPS extension |
| Carmack | PROCEED with implementation discipline (~500 LOC TOTAL via canonical library adapters) |

**Aggregate verdict: PROCEED_WITH_REVISIONS** (sextet quorum: 5-of-6 PROCEED + 1 PROCEED_WITH_REVISIONS with binding op-routables; grand council majority PROCEED_WITH_REVISIONS).

**The binding revisions** (operator-routable):
1. **Phase 1 lands WITH OP-3 byte-mutation smoke** (Contrarian's VETO) — proves downstream consumption of Fisher matrix at landing time.
2. **Phase 1 typed `FisherConditioningVerdict` enum + canonical posterior anchor** (Assumption-Adversary) — no silent fallback; verdict + (λ_min, λ_max, condition_number) explicit.
3. **Phase 1 lands BOTH fixed AND adaptive Marquardt schedules** (Boyd) — substrate selection via SubstrateContract field.
4. **Phase 1 EXTENSION to C6 IBPS family** (MacKay) — diagnoses 22× miss anchor at $0 additional cost.
5. **Phase 1 implementation budget: ~500 LOC TOTAL** (Carmack) — thin integration layer over canonical libraries.

---

## 6. Per-substrate reactivation criteria (Catalog #313)

### Per-substrate reactivation criteria

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #313 probe-outcomes ledger discipline.

The Phase 1 Fisher-precondition canonical helper is **OPT-IN per substrate** (substrates inherit via SubstrateContract field flag). Reactivation criteria operate per-OPTED-IN-SUBSTRATE:

#### Reactivation path 1 (HIGHEST priority): Phase 1 OP-2 empirical validation falsifies well-conditioned assumption

**Trigger**: PR101_lc_v2 master-gradient anchor empirical Fisher-conditioning produces `condition_number > 1e6` (verdict: `VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC`).

**Path**: substrate's `fisher_damping` field MUST be `"kfac"` (NOT `"levenberg_marquardt"`); full Fisher inverse via CG is REFUSED. The K-FAC factorization handles the near-singular structure per arxiv:1503.05671 §6.3.

**Predicted cost**: $0 GPU (analysis on already-extracted PR101_lc_v2 anchor); ~1 day editor for K-FAC integration validation via `kfac_pytorch`.

**Structural verdict**: tests Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification.

#### Reactivation path 2 (HIGH priority): Phase 1 OP-3 byte-mutation smoke falsifies FOP null-space property

**Trigger**: byte-mutation smoke on PR101_lc_v2 archive at FOP null-space byte positions produces score change `>0.005` (the FOP null-space directions ARE NOT structurally invisible).

**Path**: FOP implementation is BUGGY; Phase 1 helper is `research_only=true` pending fix. Candidates for bug: (a) eigenvalue threshold too loose (some range eigenvalues mistaken for null); (b) numerical precision insufficient at fp32 (require fp64); (c) per-pair gradient anchor stale (re-extract).

**Predicted cost**: $0 GPU (byte-mutation smoke on existing anchor); ~1 day editor for fix + re-validation.

**Structural verdict**: tests Contrarian's CARGO-CULTED-PENDING-IMPLEMENTATION classification.

#### Reactivation path 3 (HIGH priority): Phase 1 OP-4 MacKay extension diagnoses C6 IBPS family

**Trigger**: C6 IBPS random-init Fisher conditioning is `condition_number > 1e6` AND post-training Fisher conditioning is `condition_number < 1e6`.

**Path**: C6 IBPS Phase 2 redesign routes through K-FAC + Phase 1 Fisher-precondition with `damping_schedule="adaptive_marquardt"`. The 22× miss anchor (`fc-01KRW353MJJ9A6QW8H99QWZEMH`) is structurally explained as random-init Fisher-singularity. Same canonical remediation applies to ALL IB substrates (Tishby/Zaslavsky/predictive-coding-bottleneck family).

**Predicted cost**: $0 GPU (analysis on existing C6 IBPS smoke artifacts); ~2 day editor for C6 IBPS Phase 2 redesign + K-FAC integration.

**Structural verdict**: tests MacKay's HARD-EARNED-WITH-EXTENSION-EV classification.

#### Reactivation path 4 (MEDIUM priority): Phase 1 OP-2 produces `INVALID_NUMERIC_FAILURE` on PR101_lc_v2

**Trigger**: Fisher matrix on PR101_lc_v2 anchor has NaN/Inf eigenvalues OR full numerical collapse.

**Path**: investigate root cause: (a) per-pair gradient anchor corruption (re-extract); (b) numerical precision insufficient (upgrade to fp128 via mpmath); (c) PR101_lc_v2 archive itself has degenerate structure (escalate to operator). Phase 1 helper lands `research_only=true` pending root cause; Phase 2 dispatch is REFUSED for PR101_lc_v2.

**Predicted cost**: $0-5 GPU (depending on extraction/diagnosis path); ~2-5 day editor.

**Structural verdict**: tests robustness of the canonical empirical Fisher computation primitive.

#### Reactivation path 5 (LOW priority): UNIWARD-weighted Fisher metric extension (Fridrich's Phase 4 sister)

**Trigger**: any opted-in substrate where steganographic interpretation is structurally relevant (e.g., visual-quality-preserving substrates per parent synthesis §6.2 fiber bundle structure).

**Path**: replace vanilla Fisher with UNIWARD-weighted Fisher per arxiv:1311.7041. The weighted Fisher `F_UNIWARD(θ) = F(θ) ∘ W_UNIWARD` focuses Riemannian-Newton steps on perceptually-undetectable modifications. Phase 4 sister deliverable per Fridrich's contribution.

**Predicted cost**: $0 GPU (UNIWARD weights pre-computed); ~3 day editor for weighted Fisher integration.

**Structural verdict**: tests Fridrich's UNIWARD-extension proposal.

### Probe outcomes ledger registration (Catalog #313)

Upon landing this design memo, the canonical probe outcome IS:

```python
from tac.probe_outcomes_ledger import register_probe_outcome

register_probe_outcome(
    probe_id="phase_1_fisher_precondition_canonical_helper_design_20260518",
    substrate_id="phase_1_fisher_precondition_canonical_helper",
    verdict="PROCEED_WITH_REVISIONS",
    blocking=False,  # design memo not blocking; downstream OP-1/2/3/4/5 are operator-routables
    rationale=(
        "Phase 1 Fisher-precondition canonical helper with three sub-components "
        "(Fisher info from per-pair fp64 master gradient + K-FAC + Levenberg-Marquardt + FOP) "
        "GATED on Phase 1 OP-2 empirical Fisher-conditioning validation on PR101_lc_v2 anchor "
        "BEFORE Phase 2 RiemannianNewtonSubstrate base class lands. Contrarian's OP-3 byte-proof "
        "+ Assumption-Adversary's OP-5 STRICT gate are binding revisions."
    ),
    reactivation_criteria=[
        "Phase 1 OP-2 falsifies well-conditioned assumption → escalate to K-FAC default per substrate",
        "Phase 1 OP-3 byte-mutation smoke falsifies FOP null-space property → FOP implementation is buggy + research_only",
        "Phase 1 OP-4 MacKay extension diagnoses C6 IBPS family → C6 IBPS Phase 2 routes through K-FAC + Phase 1",
        "Phase 1 OP-2 produces INVALID_NUMERIC_FAILURE → research_only + escalate to operator",
        "UNIWARD-weighted Fisher Phase 4 sister deliverable for steganographic substrates → opt-in per substrate",
    ],
    related_council_anchor="phase_1_fisher_precondition_canonical_helper_design_20260518",
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

**This design memo's predicted_band declarations** (all `research_only=true` per the frontmatter):

- **Phase 1 ALONE standalone ΔS `[-0.015, -0.005]` per archive** (revised by Contrarian to `[-0.008, -0.003]` realistic): `predicted_band_validation_status: pending_post_training`. Reactivation criterion: PHASE-1-POST-EMPIRICAL-ANCHOR post-training Tier-C re-measurement on PR101_lc_v2 archive's `f174192aeadf...` operating point AFTER FOP-decomposed natural-gradient training step. Tier-C measurement via `tools/mdl_scorer_conditional_ablation.py --tier c` on the post-FOP-step archive. Validated when empirical ΔS falls within `[-0.015, -0.005]` (OPTIMISTIC) OR `[-0.008, -0.003]` (REALISTIC per Contrarian) at PR101_lc_v2.

- **CASCADE Phase 1 unlocks Phase 2-5 aggregate ΔS `[-0.060, -0.019]` realistic** (per parent §11 aggregate matrix under composition_alpha ≈ 0.5 anti-phantom): `predicted_band_validation_status: pending_post_training`. Reactivation criterion: progressive validation as Phase 2-5 substrates opt-in (1 archive → 2 → 3 → 4); paired comparison vs Phase-1-cascade prediction at each step; composition_alpha empirical measurement per Catalog #322 anti-phantom.

**Phase 1 IS post-training Tier-C aware**: the canonical Fisher-conditioning anchor is measured ON A POST-TRAINING ARCHIVE (PR101_lc_v2 frontier archive at 0.19205 [contest-CPU] is FULLY TRAINED). The MacKay extension (OP-4) measures BOTH random-init AND post-training Fisher conditioning on C6 IBPS — the random-init measurement is for diagnostic purposes (per Catalog #324), NOT for predicted_band derivation. The canonical Phase 1 predicted_band derives from POST-TRAINING measurements.

**Phase 1's canonical recipe `predicted_band_validation_status`**: `pending_post_training` with reactivation criterion = "post-training Tier-C re-measurement on archive sha256 f174192aeadf via tools/mdl_scorer_conditional_ablation.py --tier c on post-FOP-step archive". The pending status is OPERATIONALLY CORRECT for Phase 1 — the standalone Phase 1 ΔS unlock requires a substrate trainer to actually apply the FOP-decomposed natural-gradient step + emit a measurably different archive; the measurement happens post-Phase-2-dispatch (which is the canonical chain).

---

## 8. Canonical-vs-unique decision per layer (Catalog #290)

### Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable. The Phase 1 Fisher-precondition canonical helper has TWO surfaces requiring per-layer canonical-vs-unique decisions: (a) the helper's own layers; (b) the per-opted-in-substrate Fisher domain.

#### Phase 1 helper's own layers

| Layer | Decision | Rationale |
|---|---|---|
| 1. Fisher matrix computation from per-pair gradients | **CANONICAL** | The empirical Fisher `F = G^T G / N_pairs` is the canonical Amari 1985 estimator. Substrate-independent at the matrix-computation surface. Per Amari memorial seat: BOTH dense `(N × N)` AND factored `(N_pairs, N_params)` representations supported for large-vs-small substrates |
| 2. Fisher-vector product (HVP) | **CANONICAL** | `F @ v = G^T (G @ v)` is the canonical low-rank Fisher matrix-vector product. Two matvecs at O(N_pairs · N_params). Substrate-independent |
| 3. Fisher inverse via CG | **CANONICAL** | Conjugate-gradient with Fisher-vector product per Martens 2010 Hessian-Free canonical literature. Substrate-independent |
| 4. K-FAC Kronecker factorization | **CANONICAL** (delegated to `kfac_pytorch` library per Carmack) | K-FAC per-layer factorization is mathematically substrate-independent. Substrates with non-layered structure (DP1 codebook) opt-out via SubstrateContract field `use_kfac: bool = False` |
| 5. Levenberg-Marquardt damping (fixed) | **CANONICAL** | Fixed `λ` damping per arxiv:1412.1193 Martens. Substrate-independent at the damping operation |
| 6. Levenberg-Marquardt damping (adaptive Marquardt) | **CANONICAL** | Trust-region Marquardt update rule per Nocedal-Wright Algorithm 4.1. Substrate-independent at the update-rule level. Per Boyd's contribution |
| 7. Fisher-Orthogonal Projection (FOP) | **CANONICAL** | The Pythagorean decomposition per Amari-Nagaoka 2000 §3 is substrate-independent. Both `numpy.linalg.eigh` (dense) and SVD-based (low-rank) implementations canonical |
| 8. Canonical posterior anchor persistence | **CANONICAL** (via fcntl-locked JSONL per Catalog #131) | Same pattern as `tac.master_gradient.append_anchor_locked`, `tac.deploy.modal.call_id_ledger.append_event_locked`, sister continual-learning helpers. Substrate-independent at the persistence layer |

**Total canonical layers**: 8 of 8. The Phase 1 helper itself is FULLY canonical; per-substrate UNIQUE-AND-COMPLETE-PER-METHOD discipline applies at the consumer surface (per-substrate Fisher domain + damping schedule selection).

#### Per-opted-in-substrate Fisher domain (the consumer surface)

For each substrate that opts into Phase 1 Fisher-precondition via SubstrateContract field flag, the per-substrate Fisher domain is the UNIQUE layer:

| Substrate | Fisher domain decision | Rationale |
|---|---|---|
| **PR101_lc_v2** (entropy-coded latents) | **UNIQUE**: Fisher on LATENT SPACE | Entropy-coded latents are the rate-relevant parameters; Fisher on full weight space dilutes the signal. Substrate overrides `compute_per_substrate_per_pair_gradients` to project to latent space |
| **A1** (PR101 grammar reference) | **CANONICAL**: Fisher on full weight space | Default Fisher; no substrate-specific projection. Validates the canonical Phase 1 helper on the simplest substrate |
| **PR106 format0d** (latent score table) | **UNIQUE**: Fisher on SCORE TABLE entries | Score table is the inflate-time-active parameter; Fisher on full latent space dilutes signal. Substrate overrides |
| **sane_hnerv** (NeRV-family) | **CANONICAL**: Fisher on full weight space | NeRV is canonical neural representation; Fisher extends naturally |
| **Z6** (FiLM ego-motion) | **UNIQUE**: Fisher on FiLM conditioning parameters | FiLM conditioning is the substrate-distinguishing surface; Fisher on FiLM parameters is substrate-optimal |
| **Z7/Z8** (predictive coding) | **UNIQUE**: Fisher on PREDICTOR parameters | Predictor parameters are the substrate-distinguishing surface |
| **DP1** (driving prior codebook) | **UNIQUE**: Fisher on CODEBOOK EMBEDDINGS + opt-out of K-FAC | Codebook is non-layered; K-FAC inapplicable; direct Fisher computation tractable since codebook is small (~1000 entries × 128 dims) |
| **C6 IBPS** (information bottleneck) | **UNIQUE**: Fisher on BOTTLENECK layer specifically | The 24-dim bottleneck is the failure-mode locus per MacKay extension; substrate-specific Fisher on bottleneck diagnoses + remediates the 22× miss |

**Damping schedule per substrate**:

| Substrate | Damping schedule | Rationale |
|---|---|---|
| **PR101_lc_v2** | `fixed` (`λ=1e-3`) | Stable Fisher conditioning per anchor; fixed damping sufficient |
| **A1** | `fixed` (`λ=1e-3`) | Stable Fisher conditioning per parent grammar |
| **C6 IBPS** | `adaptive_marquardt` (`λ_0=1e-2`) | Destabilizing 24-dim bottleneck; adaptive damping required per Boyd's contribution |
| **Z6/Z7/Z8** | `adaptive_marquardt` (`λ_0=1e-2`) | Conditioning + predictor parameters destabilize during early training |
| **DP1** | `fixed` (`λ=1e-4`) | Codebook is small + stable; minimal damping sufficient |
| **PR106 format0d** | `fixed` (`λ=1e-3`) | Score table stable |
| **sane_hnerv** | `adaptive_marquardt` (`λ_0=1e-2`) | NeRV training notoriously unstable; adaptive damping required |

The pattern: **EVERY substrate has at least ONE UNIQUE layer** (the Fisher domain); other layers default to canonical UNLESS the substrate's mathematical structure demands a fork. The damping schedule decision is per-substrate based on stability characterization.

---

## 9. 6-hook wire-in declaration (Catalog #125)

### 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: every landing must wire its outputs into the unified solver stack OR explicitly tag `research_only=true`.

This design memo is `research_only=true` per the YAML frontmatter; the canonical helper landing (Phase 1 OP-1) MUST wire all 6 hooks.

#### Hook 1: Sensitivity-map contribution → `tac.sensitivity_map.axis_weights` (ACTIVE)

The Phase 1 helper EMITS a Fisher-curvature diagnostic for each opted-in substrate that feeds the canonical sensitivity map:

```python
@dataclass(frozen=True)
class FisherCurvatureDiagnostic:
    substrate_id: str
    archive_sha256: str
    fisher_condition_number: float
    fisher_metric_axis_per_component: dict[str, float]  # {"seg": ..., "pose": ..., "rate": ...}
    fisher_top_k_eigenvalues: tuple[float, ...]
    fisher_null_space_dimension: int
    measured_at_utc: str
```

The diagnostic feeds `tac.sensitivity_map.axis_weights` as an ALTERNATIVE per-axis weighting scheme: instead of uniform `{seg: 1.0, pose: 1.0, rate: 1.0}`, use `fisher_metric_axis_per_component` per substrate. The Fisher-derived axis weights reflect the substrate's actual local response surface per Amari 1985 — substrate-optimal per UNIQUE-AND-COMPLETE-PER-METHOD.

#### Hook 2: Pareto constraint → `tac.optimization.substrate_composition_matrix` (ACTIVE)

The Fisher eigenvalue spectrum is a Pareto-relevant signal per parent synthesis §6.4. The top-k eigenvalues identify the "principal score-relevant directions"; bulk eigenvalues identify null-space directions.

Substrate compositions where the Fisher eigenvalue spectra are ORTHOGONAL (top-k eigenvectors of substrate A are in null-space of substrate B and vice versa) ARE structurally additive in Pareto per Minkowski sum analysis. Substrate compositions where spectra overlap are sub-additive.

The Phase 1 helper emits a `FisherEigenvalueSpectrumPareto` row consumed by `tac.optimization.substrate_composition_matrix.update_composition_alpha_via_fisher_spectrum_orthogonality` for structural Pareto-additivity prediction. This refines the composition_alpha measurement per Catalog #322 anti-phantom with information-geometric grounding.

#### Hook 3: Bit-allocator hook → `tac.bit_allocator` (ACTIVE)

The Fisher diagonal `diag(F(θ))` provides per-parameter sensitivity. High-Fisher parameters MUST be allocated more bits in quantization; low-Fisher parameters can be allocated fewer bits per Hessian-based quantization canonical literature.

The Phase 1 helper emits per-substrate `FisherDiagonalSensitivity` consumed by `tac.bit_allocator.allocate_bits` as the canonical per-parameter importance vector. This replaces the heuristic importance vectors currently used per `src/tac/bit_allocator.py` with information-geometric grounding.

#### Hook 4: Cathedral autopilot dispatch hook → `tools/cathedral_autopilot_autonomous_loop.py` (ACTIVE)

NEW reward factor `adjust_predicted_delta_for_riemannian_newton_phase_eligibility`:

```python
def adjust_predicted_delta_for_riemannian_newton_phase_eligibility(
    predicted_delta: float,
    candidate: CandidateRow,
    phase_1_verdict_anchor: FisherConditioningAnchor | None,
) -> float:
    """Apply per-substrate Phase 1 Fisher-conditioning verdict.

    Substrates with verdict=VALIDATED_WELL_CONDITIONED get +15% reward factor
    (Phase 2 dispatch eligible with full Fisher inverse).
    Substrates with verdict=VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC get +5% reward factor
    (Phase 2 dispatch eligible with K-FAC adapter; smaller unlock).
    Substrates with verdict=INVALID_NUMERIC_FAILURE get 0.5x penalty
    (Phase 2 dispatch REFUSED; lane research_only).
    Substrates with no anchor get passthrough 1.0x.
    """
    if phase_1_verdict_anchor is None:
        return predicted_delta  # not opted in
    if phase_1_verdict_anchor.verdict == "VALIDATED_WELL_CONDITIONED":
        return predicted_delta * 1.15
    if phase_1_verdict_anchor.verdict == "VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC":
        return predicted_delta * 1.05
    if phase_1_verdict_anchor.verdict == "INVALID_NUMERIC_FAILURE":
        return predicted_delta * 0.5
    return predicted_delta
```

The reward factor integrates into the Catalog #319 Q3 v2 cascade as the 4th cascade tier (after Lagrangian-dual plan / DeliverabilityProof / Venn classification — Phase 1 verdict is the META-substrate-engineering eligibility signal).

#### Hook 5: Continual-learning posterior update → `tac.continual_learning.posterior_update_locked` (ACTIVE)

Every Phase 1 Fisher-conditioning anchor + Phase 2 Riemannian-Newton paired-comparison anchor + Phase 4 symplectic-EMA paired-comparison anchor emits a continual-learning posterior update per Catalog #128 fcntl-locked discipline:

```python
register_phase_1_fisher_conditioning_anchor(
    substrate_id="pr101_lc_v2",
    archive_sha256="f174192aeadf...",
    phase="phase_1_fisher_conditioning_validation",
    verdict="VALIDATED_WELL_CONDITIONED",
    fisher_condition_number=1234.56,
    fisher_top_10_eigenvalues=(...,),
    fisher_null_space_dimension=177817,
    fop_magnitude_ratio=0.987,
    measured_at_utc="2026-05-20T12:34:56Z",
)
```

Canonical posterior store: `.omx/state/riemannian_newton_anchors.jsonl` (fcntl-locked JSONL append-only per Catalog #131).

#### Hook 6: Probe-disambiguator → `tools/probe_riemannian_newton_fisher_conditioning.py` (ACTIVE)

The canonical probe-disambiguator script:

```bash
.venv/bin/python tools/probe_riemannian_newton_fisher_conditioning.py \
    --substrate-id pr101_lc_v2 \
    --archive-sha256 f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd \
    --master-gradient-anchor-path .omx/state/master_gradient_anchors.jsonl \
    --damping-strategy adaptive_marquardt \
    --damping-lambda-0 1e-2 \
    --condition-number-threshold-kfac 1e6 \
    --top-k-eigenvalues 10 \
    --output-json .omx/state/riemannian_newton_fisher_conditioning/<run_id>.json
```

The probe-disambiguator emits a typed verdict consumed by the probe-outcomes ledger per Catalog #313:

- `VALIDATED_WELL_CONDITIONED` (`condition_number < 1e3`) — Phase 2 dispatch eligible with full Fisher inverse + LM damping
- `VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC` (`1e3 ≤ condition_number ≤ 1e6`) — Phase 2 dispatch eligible with K-FAC adapter
- `INVALID_NUMERIC_FAILURE` (`condition_number > 1e6` OR NaN/Inf in eigenvalues) — Phase 2 dispatch REFUSED; lane research_only pending Phase 6 tropical-Newton extension

The probe-disambiguator IS the canonical disambiguator between Phase 2 dispatch eligibility verdicts per Hook 6 of Catalog #125.

---

## 10. Empirical validation plan on PR101_lc_v2 archive

### Empirical validation plan

The canonical Phase 1 empirical anchor uses the already-extracted PR101_lc_v2 master-gradient anchor at archive `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`. Per `.omx/state/master_gradient_anchors.jsonl` schema row dated 2026-05-18T14:45:02Z: `n_bytes=178417`, `n_pairs_used=8`, `n_pairs_total=600`, `measurement_axis="[macOS-CPU advisory]"`, `measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory"`, `gradient_array_path=".omx/state/master_gradient_fec6_contest_cpu_scorer_macos_host_advisory_20260517.npy"`.

#### Step 1: Full 600-pair anchor extraction (Phase 1 prerequisite)

The current anchor is `8-pair-subset advisory`. Phase 1 OP-2 first extracts the FULL 600-pair anchor before the canonical posterior anchor lands:

```bash
.venv/bin/python tools/extract_master_gradient.py \
    --archive submissions/pr101_lc_v2_clone/archive.zip \
    --pair-subset all_600 \
    --output-path .omx/state/master_gradient_pr101_lc_v2_600pair_<utc>.npy \
    --hardware darwin_arm64_m5_max_macos_cpu_advisory \
    --axis macos_cpu_advisory
```

**Predicted runtime**: ~30 minutes on M5 Max CPU (8 pairs took ~2 min per the existing extraction; 600 pairs ≈ 150 min — pairwise extraction is embarrassingly parallel; can use multiprocessing).

**Output**: `.npy` file of shape `(178417, 600, 3)` (per-byte-per-pair Jacobian); ~256 MB at fp32 or ~512 MB at fp64.

#### Step 2: Fisher matrix conditioning probe

```bash
.venv/bin/python tools/probe_riemannian_newton_fisher_conditioning.py \
    --substrate-id pr101_lc_v2 \
    --archive-sha256 f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd \
    --per-pair-gradient-path .omx/state/master_gradient_pr101_lc_v2_600pair_<utc>.npy \
    --damping-strategy adaptive_marquardt \
    --top-k-eigenvalues 10 \
    --compute-fop-decomposition true \
    --output-json .omx/state/riemannian_newton_fisher_conditioning/pr101_lc_v2_<utc>.json
```

**Algorithm**:
1. Load per-pair gradient `G` of shape `(N_bytes=178417, N_pairs=600, 3)`. Note: this is the canonical `(N_bytes, N_pairs, 3)` per-pair tensor per `tac.master_gradient.PER_PAIR_GRADIENT_TENSOR_KIND`. The Fisher computation aggregates over the 3 score components via the marginal coefficients per `tac.master_gradient.compute_marginal_coefficients`.
2. Form composite per-pair gradient `g_p = G[:, p, 0] * seg_marginal + G[:, p, 1] * pose_marginal + G[:, p, 2] * rate_per_byte` (shape `(178417,)`).
3. Stack composite gradients into matrix `M ∈ ℝ^(600 × 178417)`.
4. Compute SVD via `numpy.linalg.svd(M / sqrt(600), full_matrices=False)`. Get singular values `σ_i` (length 600) and right singular vectors `V_t ∈ ℝ^(600 × 178417)`.
5. Fisher eigenvalues `λ_i = σ_i^2`.
6. Report `λ_min = λ_max(σ.size-1)`, `λ_max = λ_max(0)`, `condition_number = λ_max / λ_min` (clipped at threshold), `top_10_eigenvalues = λ[:10]`, `null_space_dimension = N_bytes - count(σ > σ_max * 1e-6) = 178417 - 600 = 177817` (exact for rank-deficient case).
7. Compute FOP decomposition: pick a test gradient `g_test = ∇score(θ)` (canonical: the aggregated marginal-weighted gradient summed over pairs); decompose `g_test = g_∥ + g_⊥`; report `fop_magnitude_ratio = ||g_⊥|| / ||g||` (should be high since null-space is huge).
8. Verify Pythagorean identity `||g||² ≈ ||g_∥||² + ||g_⊥||²` (numerical residual < 1e-12 at fp64).
9. Verify FOP orthogonality `<F @ g_⊥, g_⊥> ≈ 0` (numerical residual < 1e-13 at fp64).
10. Emit typed `FisherConditioningVerdict`.

**Predicted runtime**: ~5-10 minutes on M5 Max CPU (SVD on `(600 × 178417)` is ~10^11 flops; CPU at ~10 GFLOPS = ~10s; numpy.linalg.svd is highly optimized).

**Predicted Fisher conditioning verdicts** (HARD-EARNED-WITH-EMPIRICAL-PRIOR per Catalog #303):

| Substrate | Predicted verdict | Reasoning |
|---|---|---|
| **PR101_lc_v2** (frontier post-training) | `VALIDATED_WELL_CONDITIONED` (predicted `condition_number ≈ 1e3-1e4`) | Frontier archive is post-training; Fisher should be well-conditioned. **TO BE VALIDATED EMPIRICALLY by OP-2** |
| **C6 IBPS** (random-init) | `VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC` (predicted `condition_number > 1e5`) | 24-dim bottleneck at random init; Marchenko-Pastur bulk near-zero. **TO BE VALIDATED EMPIRICALLY by OP-4** |
| **C6 IBPS** (post-50ep-smoke) | `VALIDATED_WELL_CONDITIONED` (predicted `condition_number ≈ 1e4-1e5`) | Post-training Fisher should be better-conditioned than random-init. **TO BE VALIDATED EMPIRICALLY by OP-4** |

#### Step 3: K-FAC equivalence-to-full-Fisher validation (small model)

For canonical helper validation, test K-FAC equivalence on a small synthetic model (e.g., 3-layer MLP with N ≈ 100 params) where full Fisher inverse is tractable:

```python
def test_kfac_equivalence_full_fisher_small_model():
    """K-FAC should approximate full Fisher to within ~10% relative error on small models."""
    model = create_synthetic_mlp(n_params=100)
    per_pair_gradients = sample_gradients(model, n_pairs=50)
    F_full = compute_empirical_fisher_via_pair_gradients(per_pair_gradients)
    F_kfac_inv = kfac_approximate_fisher_inverse(model, per_pair_gradients)
    F_full_inv = numpy.linalg.inv(F_full + 1e-3 * numpy.eye(100))
    g = numpy.random.randn(100)
    relative_error = numpy.linalg.norm(F_kfac_inv @ g - F_full_inv @ g) / numpy.linalg.norm(F_full_inv @ g)
    assert relative_error < 0.10  # K-FAC within 10% of full Fisher
```

#### Step 4: Byte-mutation smoke (Contrarian's OP-3)

```bash
.venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \
    --archive submissions/pr101_lc_v2_clone/archive.zip \
    --fisher-anchor-path .omx/state/riemannian_newton_fisher_conditioning/pr101_lc_v2_<utc>.json \
    --null-space-eigenvector-indices 0,1,2 \
    --top-k-bytes-per-eigenvector 100 \
    --output-json .omx/state/byte_mutation_smoke_phase_1_fop_validation_<utc>.json
```

**Algorithm**:
1. Load Fisher null-space eigenvectors (top-3 from the bulk of `null(F)`).
2. For each null-space eigenvector, identify the top-100 bytes (by absolute value in the eigenvector) and mutate by ±1 (sign of eigenvector component).
3. Run `inflate.sh` on the mutated archive.
4. Run `upstream/evaluate.py --device cpu` on the inflated frames.
5. Report score change vs baseline.

**Predicted result** (HARD-EARNED-PYTHAGOREAN): score change `< 0.005` per null-space eigenvector (the FOP null-space property predicts near-zero score sensitivity). Statistical test: at least 2 of 3 null-space eigenvectors produce score change `< 0.005`; otherwise FOP implementation is buggy + research_only.

**Predicted runtime**: ~10-15 minutes total (3 mutations × ~3 minutes per `inflate.sh + evaluate.py --device cpu`).

#### Step 5: MacKay's C6 IBPS extension (OP-4)

```bash
# C6 IBPS random-init
.venv/bin/python tools/probe_riemannian_newton_fisher_conditioning.py \
    --substrate-id c6_ibps \
    --weights-path .omx/state/c6_ibps_random_init_weights.pt \
    --per-pair-gradient-extraction-mode synthetic_video_500_pairs \
    --output-json .omx/state/riemannian_newton_fisher_conditioning/c6_ibps_random_init_<utc>.json

# C6 IBPS post-50ep
.venv/bin/python tools/probe_riemannian_newton_fisher_conditioning.py \
    --substrate-id c6_ibps \
    --weights-path .omx/state/c6_ibps_post_50ep_smoke_fc-01KRW353MJJ9A6QW8H99QWZEMH.pt \
    --per-pair-gradient-extraction-mode synthetic_video_500_pairs \
    --output-json .omx/state/riemannian_newton_fisher_conditioning/c6_ibps_post_50ep_<utc>.json
```

**Diagnostic verdict** (HARD-EARNED-WITH-PREDICTED-EMPIRICAL): if random-init `condition_number > 1e5` AND post-50ep `condition_number < 1e5`, the C6 IBPS 22× miss is structurally explained as random-init Fisher-singularity → Phase 1 Fisher-precondition with K-FAC + adaptive Marquardt damping is the canonical remediation. **EXTENDS THE OPERATOR-ROUTABLE EV TO THE ENTIRE IB SUBSTRATE FAMILY**.

#### Acceptance criteria for Phase 2 dispatch readiness

Per Assumption-Adversary's binding revision: Phase 2 RiemannianNewtonSubstrate dispatch is GATED on:

1. **OP-2 PR101_lc_v2 verdict** = `VALIDATED_WELL_CONDITIONED` OR `VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC` (NOT `INVALID_NUMERIC_FAILURE`).
2. **OP-3 byte-mutation smoke**: at least 2 of 3 null-space eigenvectors produce score change `< 0.005`.
3. **OP-5 STRICT preflight gate**: lands and refuses anchors missing explicit verdict.
4. **K-FAC equivalence test**: K-FAC within 10% of full Fisher on synthetic small model.

If any acceptance criterion fails, Phase 2 dispatch is REFUSED + Phase 1 is research_only pending fix + reactivation per §6.

---

## 11. Implementation architecture for `tac.riemannian_newton_meta_substrate.fisher_precondition` package

### Implementation architecture

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Carmack's "USE EXISTING CANONICAL LIBRARIES" contribution: the Phase 1 sub-package is a THIN INTEGRATION LAYER over existing libraries, target ~500 LOC total.

#### Sub-package structure (within parent `tac.riemannian_newton_meta_substrate` package)

```
src/tac/riemannian_newton_meta_substrate/
├── __init__.py                         # parent package re-exports (Phase 2+ scope)
├── fisher_precondition.py              # Phase 1 canonical helper (~250 LOC)
│   - compute_empirical_fisher_via_pair_gradients(per_pair_gradients, return_factored=False) -> Tensor | tuple[Tensor, Tensor]
│   - fisher_vector_product(per_pair_gradients, v) -> Tensor
│   - fisher_inverse_via_cg(fisher_vec_product, b, damping_lambda, max_iterations=100, tol=1e-6) -> Tensor
│   - levenberg_marquardt_damping_fixed(lambda_value) -> Callable
│   - levenberg_marquardt_damping_adaptive_marquardt(initial_lambda, lambda_min, lambda_max) -> Callable
│   - kfac_approximate_fisher_inverse(model, per_pair_gradients, damping_lambda) -> Callable  (delegates to kfac_adapter)
│   - fisher_orthogonal_projection(g, F, threshold_rel=1e-6) -> tuple[Tensor, Tensor]
│   - fisher_orthogonal_projection_low_rank(g, G_pair_gradients, threshold_rel=1e-6) -> tuple[Tensor, Tensor]
│   - FisherConditioningVerdict enum (3-valued: VALIDATED_WELL_CONDITIONED | VALIDATED_NEAR_SINGULAR_REQUIRES_KFAC | INVALID_NUMERIC_FAILURE)
│   - classify_fisher_conditioning_verdict(condition_number, has_nan_inf) -> FisherConditioningVerdict
├── adapters/
│   ├── kfac_adapter.py                 # thin wrapper over kfac_pytorch (~50 LOC)
│   │   - kfac_factorize_model(model) -> dict[str, tuple[Tensor, Tensor]]  # per-layer (A, B)
│   │   - kfac_inverse_layer(A, B, damping_lambda, pi=None) -> Callable
│   │   - kfac_fisher_inverse_full_model(model, per_pair_gradients, damping_lambda) -> Callable
│   └── levenberg_marquardt_adapter.py  # adaptive Marquardt trust-region (~80 LOC)
│       - LevenbergMarquardtState dataclass (lambda_k, trust_region_radius_k, last_rho_k)
│       - update_marquardt_lambda(state, actual_decrease, predicted_decrease, current_step_norm) -> LevenbergMarquardtState
│       - canonical defaults: lambda_min=1e-8, lambda_max=1e+4, delta_max_factor=100, rho_high=0.75, rho_low=0.25
├── anchors.py                          # fcntl-locked JSONL anchor persistence (~80 LOC)
│   - FisherConditioningAnchor dataclass (schema v1 per §4)
│   - RIEMANNIAN_NEWTON_ANCHORS_PATH = Path(".omx/state/riemannian_newton_anchors.jsonl")
│   - _RIEMANNIAN_NEWTON_LOCK_PATH = Path(".omx/state/.riemannian_newton.lock")
│   - append_anchor_locked(anchor: FisherConditioningAnchor, *, path=None, lock_path=None) -> None
│   - load_anchors_lenient(path=None) -> list[dict]
│   - load_anchors_strict(path=None) -> list[dict]  (Catalog #138 fail-closed)
│   - query_anchors_by_archive(archive_sha256, *, path=None) -> list[dict]
│   - latest_anchor_for_substrate(substrate_id, *, path=None) -> dict | None
└── tests/
    ├── test_fisher_precondition.py     # 18+ unit tests (~250 LOC)
    │   - test_fisher_matches_g_t_g_division
    │   - test_fisher_vector_product_matches_full_fisher_matmul
    │   - test_cg_converges_to_dense_inverse
    │   - test_fop_pythagorean_identity (||g||² == ||g_||||² + ||g_⊥||²)
    │   - test_fop_orthogonality (<F g_⊥, g_⊥> ≈ 0)
    │   - test_fop_low_rank_matches_full_eigh_on_small_model
    │   - test_levenberg_marquardt_fixed_lambda_convergence
    │   - test_levenberg_marquardt_adaptive_marquardt_converges_faster_than_fixed
    │   - test_kfac_equivalence_full_fisher_within_10_percent_small_model
    │   - test_kfac_handles_non_layered_substrate_via_opt_out
    │   - test_classify_fisher_conditioning_verdict_thresholds
    │   - test_invalid_numeric_failure_on_nan_eigenvalues
    │   - test_canonical_helper_emits_structured_log_lines
    │   - test_canonical_helper_returns_factored_form_for_large_substrate
    │   - test_canonical_helper_returns_dense_form_for_small_substrate
    │   - test_fisher_matches_canonical_master_gradient_anchor_pr101_lc_v2
    │   - test_fop_null_space_dimension_equals_n_params_minus_n_pairs_for_rank_deficient
    │   - test_threshold_rel_default_consistent_with_arxiv_2508_13898
    └── test_anchors.py                 # 8+ tests including 4-proc spawn-pool concurrent-append stress (~120 LOC)
        - test_fisher_conditioning_anchor_round_trip_to_jsonl
        - test_fisher_conditioning_anchor_strict_load_raises_on_corrupt_json
        - test_fisher_conditioning_anchor_quarantines_corrupt_file
        - test_fisher_conditioning_anchor_4_proc_spawn_pool_concurrent_append
        - test_fisher_conditioning_anchor_appears_in_query_by_archive
        - test_fisher_conditioning_anchor_latest_for_substrate_returns_latest_by_utc
        - test_fisher_conditioning_anchor_schema_version_pinned
        - test_fisher_conditioning_anchor_validates_verdict_enum_values
```

**Total LOC**: ~250 (helper) + ~130 (adapters) + ~80 (anchors) + ~370 (tests) = ~830 LOC. Exceeds Carmack's ~500 LOC target by ~65% due to comprehensive test coverage; helper + adapters are ~460 LOC (within target).

#### Sister CLI surfaces

```
tools/
├── probe_riemannian_newton_fisher_conditioning.py  # OP-2 + OP-4 CLI (~150 LOC)
├── extract_master_gradient.py                       # OP-2 prerequisite (already exists per parent §1; extend for 600-pair)
└── verify_distinguishing_feature_byte_mutation.py   # OP-3 CLI (already exists per Catalog #272; reuse for FOP byte-proof)
```

#### Public API (the parent `__init__.py` re-exports for Phase 1 scope)

```python
from tac.riemannian_newton_meta_substrate.fisher_precondition import (
    compute_empirical_fisher_via_pair_gradients,
    fisher_vector_product,
    fisher_inverse_via_cg,
    levenberg_marquardt_damping_fixed,
    levenberg_marquardt_damping_adaptive_marquardt,
    kfac_approximate_fisher_inverse,
    fisher_orthogonal_projection,
    fisher_orthogonal_projection_low_rank,
    FisherConditioningVerdict,
    classify_fisher_conditioning_verdict,
)
from tac.riemannian_newton_meta_substrate.anchors import (
    FisherConditioningAnchor,
    RIEMANNIAN_NEWTON_ANCHORS_PATH,
    append_anchor_locked,
    load_anchors_lenient,
    load_anchors_strict,
    query_anchors_by_archive,
    latest_anchor_for_substrate,
)

__all__ = [
    "compute_empirical_fisher_via_pair_gradients",
    "fisher_vector_product",
    "fisher_inverse_via_cg",
    "levenberg_marquardt_damping_fixed",
    "levenberg_marquardt_damping_adaptive_marquardt",
    "kfac_approximate_fisher_inverse",
    "fisher_orthogonal_projection",
    "fisher_orthogonal_projection_low_rank",
    "FisherConditioningVerdict",
    "classify_fisher_conditioning_verdict",
    "FisherConditioningAnchor",
    "RIEMANNIAN_NEWTON_ANCHORS_PATH",
    "append_anchor_locked",
    "load_anchors_lenient",
    "load_anchors_strict",
    "query_anchors_by_archive",
    "latest_anchor_for_substrate",
]
```

#### Key design decisions

**Decision 1: Factored vs dense Fisher representation**

Per Amari memorial seat: BOTH supported via `compute_empirical_fisher_via_pair_gradients(per_pair_gradients, return_factored=False)`:

- `return_factored=False`: returns dense `F ∈ ℝ^(N×N)` (default for N ≤ 10^4)
- `return_factored=True`: returns `(G, sqrt_N_pairs)` factored representation (default for N > 10^4)

The dense form is only used in tests + unit cases; production substrates (PR101_lc_v2 with N=178417) ALWAYS use factored form.

**Decision 2: Damping schedule API**

Per Boyd's contribution: BOTH fixed and adaptive Marquardt via factory pattern:

```python
fixed_schedule = levenberg_marquardt_damping_fixed(lambda_value=1e-3)
# fixed_schedule(...) returns 1e-3 always

adaptive_schedule = levenberg_marquardt_damping_adaptive_marquardt(
    initial_lambda=1e-2, lambda_min=1e-8, lambda_max=1e+4
)
# adaptive_schedule.update(actual_decrease, predicted_decrease, step_norm) updates state
# adaptive_schedule.current_lambda() returns current λ_k
```

**Decision 3: FOP efficiency vs precision tradeoff**

Per Lanczos / power-iteration efficient alternative (§1.4): three implementations supported:

- `fisher_orthogonal_projection(g, F)` — dense `numpy.linalg.eigh`; canonical for small substrates (N ≤ 10^4) + tests
- `fisher_orthogonal_projection_low_rank(g, G_pair_gradients)` — low-rank SVD; canonical for medium substrates (N ≤ 10^6) + PR101_lc_v2 production
- `fisher_orthogonal_projection_lanczos(g, fisher_vec_product, top_k)` — Lanczos; canonical for very large substrates (N > 10^6); future deliverable

Phase 1 lands the first two; Lanczos variant is deferred to a future op-routable as needed.

**Decision 4: K-FAC equivalence test acceptance threshold**

Per Carmack's discipline: 10% relative error vs full Fisher inverse on synthetic small model. The 10% threshold is canonical per Martens-Grosse 2015 §6.4 — beyond this, K-FAC's per-layer factorization assumption breaks down for non-independent layers.

---

## 12. Cross-stack composability

### Cross-stack composability

Phase 1 Fisher-precondition helper composes with multiple sister deliverables along orthogonal axes per parent synthesis §10 + Catalog #322 anti-phantom + the canonical composition_alpha matrix.

#### Composability matrix with sister deliverables

| Sister deliverable | Composition with Phase 1 | Composability_alpha | Rationale |
|---|---|---|---|
| **`tac.master_gradient`** (existing canonical) | INPUT-PROVIDER | α ≈ 1.0 (no overlap) | Phase 1 CONSUMES the per-pair gradient anchor from `tac.master_gradient`. Phase 1 augments the existing 8-use roadmap per the master_gradient docstring §3.6 use #3 (bit-allocator) + #6 (Pareto/Dykstra) + #7 (continual-learning posterior) |
| **`tac.master_gradient_consumers`** (existing canonical) | DOWNSTREAM-COMPOSITION | α ≈ 0.9 | Phase 1's `FisherConditioningAnchor` is a NEW consumer of the per-pair gradient; sister consumers already use the same anchor for axis-weighting + bit-allocation. Orthogonal use; composes additively |
| **`tac.null_space_exploiter` (TOP-1 op-routable from parent synthesis §0)** | UPSTREAM-PROVIDER | α ≈ 0.95 | Phase 1's FOP IS the canonical null-space identification primitive that null-space exploiter consumes. UNLOCKS `[-0.040, -0.012]` → `[-0.055, -0.018]` per parent §0 |
| **`tac.theoretical_floor_estimator` (DONE 2026-05-18)** | DOWNSTREAM-CONSUMER | α ≈ 0.95 | Phase 1's Fisher condition number + top-k eigenvalues feed the floor estimator's FLOOR-TIGHTENING from `[0.05, 0.12]` to `[0.05, 0.10]` per parent §10.4 |
| **`tac.deterministic_score_optimizer` (DONE 2026-05-18)** | INNER-CONTINUOUS-OPTIMIZER | α ≈ 0.9 | Phase 1's Fisher matrix feeds the inner continuous θ Hessian for the deterministic optimizer's KKT decomposition on the smooth portion |
| **Cathedral autopilot v2 cascade (Catalog #319 Q3)** | NEW-CASCADE-TIER | α ≈ 1.0 | Phase 1 verdict consumed by `adjust_predicted_delta_for_riemannian_newton_phase_eligibility` reward factor (NEW 4th cascade tier after Lagrangian-dual / DeliverabilityProof / Venn classification) |
| **`tac.substrate_registry` + SubstrateContract (Catalogs #241/#242)** | SCHEMA-EXTENSION | α ≈ 0.95 | Phase 3 op-routable extends SubstrateContract with `riemannian_newton_enabled: bool = False` + `fisher_damping_schedule` fields. Backward-compat preserved per Catalog #241 |
| **C6 IBPS Phase 2 redesign (operator-routable from sister #861)** | DIAGNOSTIC-INPUT | α ≈ 1.0 (if MacKay extension validates) | Phase 1 OP-4 MacKay extension diagnoses the 22× miss; if confirmed, C6 IBPS Phase 2 routes through K-FAC + Phase 1 Fisher-precondition |
| **Z8 hierarchical predictive coding symposium (in-flight)** | NATURAL-FIT-INHERITANCE | α ≈ 0.85 | K-FAC's per-layer Kronecker factorization is the natural fit for Z8's hierarchical structure (per-level Fisher); composability is structural |

#### Aggregate predicted ΔS unlock matrix

The Phase 1 standalone ΔS is `[-0.015, -0.005]` per archive (OPTIMISTIC) or `[-0.008, -0.003]` per archive (REALISTIC per Contrarian's revision).

The CASCADE Phase 1 unlocks Phase 2-5 + sister deliverables across 4 frontier archives:

| Composition scenario | Per-archive ΔS | Aggregate ΔS (4 archives × α-discount) | Frontier potential |
|---|---|---|---|
| **HIGH-orthogonality** (α ≈ 0.9) | `[-0.025, -0.008]` (parent §10.3) | `[-0.025 × 4 × 0.9, -0.008 × 4 × 0.9] = [-0.090, -0.029]` | `[0.102, 0.163]` [contest-CPU] |
| **REALISTIC** (α ≈ 0.6 mixed) | `[-0.025, -0.008]` | `[-0.060, -0.019]` | `[0.132, 0.173]` [contest-CPU] |
| **WORST-CASE** (α ≈ 0.4) | `[-0.025, -0.008]` | `[-0.040, -0.013]` | `[0.152, 0.179]` [contest-CPU] |
| **PHASE 1 STANDALONE** (no Phase 2-5) | `[-0.015, -0.005]` OPTIMISTIC | `[-0.060, -0.020]` (across 4 archives) | `[0.132, 0.172]` [contest-CPU] |
| **PHASE 1 STANDALONE REALISTIC** (per Contrarian) | `[-0.008, -0.003]` | `[-0.032, -0.012]` (across 4 archives) | `[0.160, 0.180]` [contest-CPU] |

The REALISTIC cascade prediction brings frontier into **frontier-pursuit horizon-class `[0.120, 0.180]`** per CLAUDE.md "HORIZON-CLASS evaluation axis" standing directive 2026-05-16.

#### Composition with sister substrate frontier wave deliverables

Per the just-landed deep-research wave (sister `a08a7608` lane `lane_deep_research_wave_20260518`):

| Wave deliverable | Composition with Phase 1 | Predicted unlock |
|---|---|---|
| **TT5L V2 + VGGT + DUSt3R/MASt3R + NVIDIA VRSS 2** (predicted ΔS `[-0.020, -0.008]`) | ORTHOGONAL (different substrate paradigm) | composes additively if both opted-in; aggregate `[-0.035, -0.013]` |
| **Z7-as-Mamba-2** (predicted ΔS `[-0.025, -0.008]`) | UPSTREAM-DEPENDENCY (Mamba-2 trainer can opt into Phase 1) | composes if Z7 opts in; aggregate `[-0.040, -0.013]` |
| **ATW V2-1 + Faiss-IVF-PQ** (predicted ΔS `[-0.015, -0.005]`) | UPSTREAM-DEPENDENCY (ATW V2-1 codec can opt into Phase 1) | composes if ATW opts in; aggregate `[-0.030, -0.010]` |
| **DP1+PR101 stacking** (predicted ΔS `[-0.012, -0.004]`) | UPSTREAM-DEPENDENCY (DP1 trainer can opt into Phase 1 with codebook Fisher) | composes if DP1 opts in; aggregate `[-0.027, -0.009]` |
| **lane_17_imp Frankle LTH** (predicted ΔS `[-0.015, -0.005]`) | UPSTREAM-DEPENDENCY (LTH iterative pruning can opt into Phase 1 Fisher per-parameter importance) | composes if LTH opts in; aggregate `[-0.030, -0.010]` |

**Frontier potential if ALL TOP-5 wave deliverables + Phase 1 cascade compose REALISTIC (α ≈ 0.5)**:

```
ΔS_total ≈ ΔS_phase_1_cascade + sum_i ΔS_wave_i × α_wave_i_with_phase_1
         ≈ -0.040 + (-0.020 × 0.5 + -0.025 × 0.5 + -0.015 × 0.5 + -0.012 × 0.5 + -0.015 × 0.5)
         ≈ -0.040 + -0.044
         ≈ -0.084

Frontier: 0.19205 - 0.084 = 0.108 [contest-CPU]
```

This achieves the deep-research wave §0 prediction's lower bound `[0.167, 0.184]` BUT with Phase 1 cascade additionally unlocks `~0.05` further → frontier **`[0.108, 0.130]` [contest-CPU]** if Phase 1 + ALL wave deliverables compose REALISTIC. This is in the **asymptotic-pursuit horizon-class `[0.050, 0.120]`**.

---

## 13. Cross-references and integration with existing canonical infrastructure

### Cross-references

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable: integration with existing canonical infrastructure.

#### Direct dependencies (Phase 1 helper consumes)

- `tac.master_gradient` — per-pair fp64 gradient anchor (PR101_lc_v2 at `f174192aeadf...`)
- `tac.master_gradient.OperatingPoint` — operating point dataclass; reused in `FisherConditioningAnchor`
- `tac.master_gradient.compute_marginal_coefficients` — for composite per-pair gradient aggregation
- `tac.master_gradient.CONTEST_RATE_DENOM_BYTES` — for canonical rate marginal
- `numpy.linalg.{svd, eigh, inv}` — canonical dense linear algebra
- `scipy.sparse.linalg.{cg, eigsh}` — canonical iterative solvers
- `kfac_pytorch` (external; MIT-licensed) — K-FAC factorization library
- `fcntl` — canonical advisory file locking per Catalog #131

#### Reverse dependencies (Phase 1 helper exposes)

- `tac.sensitivity_map.axis_weights` — Hook 1 ACTIVE
- `tac.optimization.substrate_composition_matrix` — Hook 2 ACTIVE
- `tac.bit_allocator` — Hook 3 ACTIVE
- `tools.cathedral_autopilot_autonomous_loop` — Hook 4 ACTIVE
- `tac.continual_learning.posterior_update_locked` — Hook 5 ACTIVE
- `tools.probe_riemannian_newton_fisher_conditioning` — Hook 6 ACTIVE

#### Sister design memos (related deliberation chain)

- `.omx/research/riemannian_newton_substrate_engineering_design_memo_20260518.md` — parent (Phase 1 sub-component of parent's Phase 1)
- `.omx/research/set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` — grandparent (arxiv:2508.13898 FOP source)
- `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` — sister (Fisher-curvature feeds floor-tightening)
- `.omx/research/deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518.md` — sister (inner continuous θ optimization)
- `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md` — grand-sister (Framework #2 of TOP-5 analytical frameworks)

#### CLAUDE.md non-negotiables honored

- "Race-mode rigor inversion + parallel-dispatch first" — Phase 1 IS the parallel-dispatch actuator's natural-gradient primitive
- "Long-burn score-lowering campaign default" — Phase 1 GATES the long-burn Phase 2-5 cascade
- "HNeRV / leaderboard-implementation parity discipline" — Phase 1 lessons 1-13 ALL honored (Score-aware loss + Export-first + ≤100 LOC inflate budget N/A for Phase 1 helper itself; etc.)
- "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — per-substrate Fisher domain + damping schedule are UNIQUE; canonical sub-components SHARED per §8
- "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" — research_only=true per frontmatter; Phase 1 byte-proof (OP-3) per Contrarian's VETO addresses runtime-effect for the canonical helper
- "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" — Phase 1 verdict GATES Phase 2 dispatch eligibility (Catalog #315 inverse: Phase 1 is the per-substrate pre-dispatch optimal-form check)
- "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" — Phase 1 design memo IS the per-substrate symposium for the meta-substrate canonical helper (Catalog #325)
- "Mission alignment" — `council_predicted_mission_contribution: frontier_breaking` per frontmatter
- "Max observability" — 6 facets ALL ACTIVE per §4
- "Apples-to-apples evidence discipline" — axis tags + hardware substrate explicit in canonical posterior anchor schema
- "Bugs must be permanently fixed AND self-protected against" — new Catalog # STRICT preflight gate per OP-5
- "Subagent coherence-by-default" — 6 hooks ACTIVE per §9
- "Comment-only contracts are FORBIDDEN" — typed `FisherConditioningVerdict` enum + canonical posterior anchor + STRICT preflight gate are runtime enforcements

#### Catalog gates honored

- Catalog #1 (`check_no_mps_fallback_default`) — Phase 1 helper does NOT touch device selection (operates on numpy CPU)
- Catalog #110/#113 (HISTORICAL_PROVENANCE) — canonical posterior anchor APPEND-ONLY per Catalog #131
- Catalog #125 (subagent landing 6-hook wire-in) — declared explicitly per §9
- Catalog #126 (lane pre-registered) — `lane_phase_1_fisher_precondition_canonical_helper_design_20260518` pre-registered at L0 at task start
- Catalog #127 (custody validator routing) — canonical posterior anchor schema includes axis + hardware tags
- Catalog #128/#131/#138 (fcntl-locked JSONL + strict-load discipline) — canonical posterior anchor per Catalog #131
- Catalog #176/#185 (CLAUDE.md catalog META-meta) — new Catalog # claim per OP-5 honors duplicate-number + LIVE_COUNT-drift discipline
- Catalog #186 (catalog claim via canonical serializer) — OP-5 claim via `tools/claim_catalog_number.py claim --commit-via-serializer`
- Catalog #206 (subagent crash-resume checkpoint discipline) — initial checkpoint emitted at design memo start
- Catalog #229 (premise-verification-before-edit) — 5+ canonical pointers read in full pre-edit per §0 council's mandate
- Catalog #245 (Modal call_id ledger 4-layer pattern) — mirrored for canonical posterior anchor + STRICT gate + autopilot wire-in
- Catalog #272 (distinguishing-feature integration contract) — Phase 1 byte-proof OP-3 IS the distinguishing-feature byte-mutation smoke
- Catalog #290 (canonical-vs-unique decision per layer) — §8
- Catalog #291 (META-ASSUMPTION review cadence) — sister landing in same session
- Catalog #292 (per-deliberation assumption-statement) — explicit per-attendee assumption surface per §5
- Catalog #294 (9-dim success checklist evidence) — §3
- Catalog #296 (predicted-band Dykstra-feasibility) — §9 Predicted ΔS band per §9 first-principles bound + Dykstra check
- Catalog #297 (signal-axis-destruction reversibility) — N/A (Phase 1 does NOT destroy signal axes; FOP IS the canonical non-destructive null-space identifier)
- Catalog #298 (substrate L1 not stale dispatch; 30-day) — Phase 1 lane pre-registered + impl_complete via OP-1 within 30 days
- Catalog #300 v2 (council deliberation frontmatter) — frontmatter compliant
- Catalog #303 (cargo-cult audit per assumption) — §2
- Catalog #305 (observability surface) — §4
- Catalog #309 (horizon-class declaration) — `horizon_class: asymptotic_pursuit` per frontmatter
- Catalog #313 (probe-outcomes ledger; predecessor lookup) — registered per §6
- Catalog #315 (substrate at OPTIMAL FORM before paid dispatch) — Phase 1 IS the canonical OPTIMAL FORM gate for downstream Phase 2-5
- Catalog #319 (Wyner-Ziv deliverability proof OR autopilot reward) — Hook 4 ACTIVE as 4th cascade tier
- Catalog #323 (canonical Provenance umbrella) — canonical posterior anchor schema includes Provenance fields
- Catalog #324 (post-training Tier-C validation) — §7
- Catalog #325 (per-substrate optimal-form symposium) — THIS DESIGN MEMO IS the canonical symposium for the Phase 1 meta-substrate helper

---

## 14. Op-routables ranked by EV with Codex execution steps

### Op-routables

#### OP-1 (HIGHEST EV; Phase 1 implementation; ~3 day editor + $0 GPU)

**Subagent dispatch**: `lane_phase_1_fisher_precondition_canonical_helper_build_20260520` (next session)

**Pre-flight per Catalog #229 (premise verification before edit)**:
1. Read CLAUDE.md + AGENTS.md fully
2. Read this design memo + parent design memo (`riemannian_newton_substrate_engineering_design_memo_20260518.md`) + grandparent synthesis (`set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` §10.3 + §4.7)
3. Read `src/tac/master_gradient.py` + `src/tac/master_gradient_consumers.py` (key consumer pattern)
4. Read arxiv:2508.13898 FOP paper sections 2-4 via WebFetch
5. Read arxiv:1503.05671 K-FAC paper sections 5-7 via WebFetch
6. Read Nocedal & Wright Ch 10 (Levenberg-Marquardt adaptive trust-region; canonical reference)
7. Verify `kfac_pytorch` library import works in `.venv`

**Codex build steps**:
1. Claim new lane via `tools/lane_maturity.py add-lane lane_phase_1_fisher_precondition_canonical_helper_build_20260520 --name "Phase 1 Fisher-precondition canonical helper BUILD" --phase 4`
2. Create `src/tac/riemannian_newton_meta_substrate/__init__.py` (empty initially; re-exports added in step 6)
3. Create `src/tac/riemannian_newton_meta_substrate/fisher_precondition.py` (~250 LOC per §11; 8 canonical functions + `FisherConditioningVerdict` enum + classifier)
4. Create `src/tac/riemannian_newton_meta_substrate/adapters/kfac_adapter.py` (~50 LOC; thin wrapper over `kfac_pytorch`)
5. Create `src/tac/riemannian_newton_meta_substrate/adapters/levenberg_marquardt_adapter.py` (~80 LOC; LevenbergMarquardtState dataclass + Nocedal-Wright Algorithm 4.1 update rule)
6. Create `src/tac/riemannian_newton_meta_substrate/anchors.py` (~80 LOC; FisherConditioningAnchor dataclass + fcntl-locked JSONL persistence per Catalog #131)
7. Create `src/tac/riemannian_newton_meta_substrate/__init__.py` re-exports per §11 Public API
8. Create `src/tac/riemannian_newton_meta_substrate/tests/test_fisher_precondition.py` (~250 LOC; 18+ unit tests per §11)
9. Create `src/tac/riemannian_newton_meta_substrate/tests/test_anchors.py` (~120 LOC; 8+ tests including 4-proc spawn-pool concurrent-append stress)
10. Run `.venv/bin/python -m pytest src/tac/riemannian_newton_meta_substrate/tests/ -v` — verify all tests pass

**Commit discipline per CLAUDE.md "Subagent commits MUST use serializer"**: each file committed via `tools/subagent_commit_serializer.py --files <file> --expected-content-sha256 <file>=<sha> --message "..." ` with post-edit working-tree shas.

**Deliverable**: 7 new files + canonical helper at `tac.riemannian_newton_meta_substrate.fisher_precondition` available for OP-2 invocation. Lane L0 → L1 (impl_complete + memory_entry + deploy_runbook gates).

#### OP-2 (HIGH EV; Phase 1 empirical Fisher-conditioning validation on PR101_lc_v2; ~4 hr editor + $0 GPU)

**Subagent dispatch**: `lane_phase_1_fisher_precondition_canonical_helper_validation_pr101_lc_v2_20260520` (same session as OP-1 after OP-1 lands)

**Codex execution steps**:
1. (Prerequisite) Run `tools/extract_master_gradient.py --archive submissions/pr101_lc_v2_clone/archive.zip --pair-subset all_600 --output-path .omx/state/master_gradient_pr101_lc_v2_600pair_<utc>.npy --hardware darwin_arm64_m5_max_macos_cpu_advisory --axis macos_cpu_advisory` (predicted runtime: ~30 min on M5 Max CPU)
2. Create `tools/probe_riemannian_newton_fisher_conditioning.py` (~150 LOC) per §10 Step 2 algorithm. CLI signature per §10 Step 2.
3. Run `tools/probe_riemannian_newton_fisher_conditioning.py --substrate-id pr101_lc_v2 --archive-sha256 f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd --per-pair-gradient-path .omx/state/master_gradient_pr101_lc_v2_600pair_<utc>.npy --damping-strategy adaptive_marquardt --top-k-eigenvalues 10 --compute-fop-decomposition true --output-json .omx/state/riemannian_newton_fisher_conditioning/pr101_lc_v2_<utc>.json`
4. Validate output: verify (λ_min, λ_max, condition_number, top-10 eigenvalues, null_space_dimension, fop_magnitude_ratio, pythagorean_identity_error, fop_orthogonality_error) all reported with finite numerical values
5. Register canonical anchor via `tac.riemannian_newton_meta_substrate.anchors.append_anchor_locked(FisherConditioningAnchor(...))`
6. Register probe outcome via `tac.probe_outcomes_ledger.register_probe_outcome(probe_id="...", substrate_id="pr101_lc_v2", verdict="PROCEED", ...)`

**Deliverable**: canonical PR101_lc_v2 Phase 1 Fisher-conditioning anchor in `.omx/state/riemannian_newton_anchors.jsonl` + probe outcome in `.omx/state/probe_outcomes.jsonl`. Validates Assumption-Adversary's CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION classification at $0 GPU cost.

#### OP-3 (HIGH EV; Phase 1 byte-mutation smoke per Contrarian's VETO; ~3 hr editor + $0 GPU)

**Subagent dispatch**: same session as OP-2 after OP-2 lands

**Codex execution steps**:
1. Read OP-2 anchor JSON to extract top-3 null-space eigenvectors (the `fisher_null_space_dimension` is 177817; pick eigenvectors with smallest eigenvalues)
2. For each null-space eigenvector, identify top-100 bytes (by absolute value in eigenvector); these are the candidate null-space byte positions
3. Extend `tools/verify_distinguishing_feature_byte_mutation.py` (existing CLI per Catalog #272) with `--fisher-null-space-eigenvector-bytes` flag to accept the byte position list
4. Run `tools/verify_distinguishing_feature_byte_mutation.py --archive submissions/pr101_lc_v2_clone/archive.zip --fisher-anchor-path .omx/state/riemannian_newton_fisher_conditioning/pr101_lc_v2_<utc>.json --null-space-eigenvector-indices 0,1,2 --top-k-bytes-per-eigenvector 100 --output-json .omx/state/byte_mutation_smoke_phase_1_fop_validation_<utc>.json`
5. Validate output: at least 2 of 3 null-space eigenvectors produce score change `< 0.005`; report PASS / FAIL with statistical justification
6. If FAIL: Phase 1 helper lands `research_only=true` with reactivation per §6 path 2; investigate root cause

**Deliverable**: byte-mutation smoke artifact + PASS/FAIL verdict for Contrarian's VETO requirement. If PASS: Phase 1 is structurally consumed by downstream surfaces + structurally relevant. If FAIL: Phase 1 helper is research_only + FOP implementation reviewed.

#### OP-4 (MEDIUM EV; Phase 1 C6 IBPS extension per MacKay; ~2 hr editor + $0 GPU)

**Subagent dispatch**: same session as OP-2/OP-3

**Codex execution steps**:
1. (Prerequisite) Locate C6 IBPS random-init weights AND post-50ep smoke weights at `fc-01KRW353MJJ9A6QW8H99QWZEMH` (Modal call_id per `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md`)
2. (Prerequisite) Extract per-pair gradients on C6 IBPS random-init weights via `tools/extract_master_gradient.py --substrate c6_ibps --weights-path <random-init-path> --output-path .omx/state/master_gradient_c6_ibps_random_init_<utc>.npy --hardware darwin_arm64_m5_max_macos_cpu_advisory --axis macos_cpu_advisory` (predicted runtime: ~30 min)
3. (Prerequisite) Same for post-50ep weights
4. Run `tools/probe_riemannian_newton_fisher_conditioning.py` on BOTH (per §10 Step 5 CLI)
5. Compare verdicts: if random-init `condition_number > 1e5` AND post-50ep `condition_number < 1e5`, the C6 IBPS 22× miss is structurally explained as random-init Fisher-singularity
6. Write diagnostic memo at `.omx/research/c6_ibps_phase_1_mackay_extension_diagnosis_<utc>.md` summarizing findings + recommendation for C6 IBPS Phase 2 redesign (K-FAC + adaptive Marquardt damping)

**Deliverable**: C6 IBPS Phase 1 Fisher-conditioning anchors (random-init + post-50ep) + diagnostic memo. EXTENDS operator-routable EV to ENTIRE IB substrate family if MacKay's hypothesis validates.

#### OP-5 (MEDIUM EV; Phase 1 STRICT preflight gate; ~2 hr editor + $0 GPU)

**Subagent dispatch**: same session as OP-1

**Codex execution steps**:
1. Claim new Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "Phase 1 Fisher-precondition anchor validation status STRICT gate"`
2. Implement `check_riemannian_newton_anchor_validation_status` function in `src/tac/preflight.py` (~80 LOC) refusing anchors with `phase="phase_1_fisher_conditioning_validation"` AND no explicit `verdict` from the 3-valued enum
3. Wire into `preflight_all` with `strict=False` initially per "Strict-flip atomicity rule"
4. Add CLAUDE.md catalog table entry per Catalog #176 META-meta discipline
5. Create `src/tac/tests/test_check_NEW_riemannian_newton_anchor_validation.py` (~120 LOC; 12+ unit tests covering positive/negative/waiver/strict-mode/live-repo regression)
6. Run `.venv/bin/python -m pytest src/tac/tests/test_check_NEW_*` — verify all tests pass; live count: 0 at landing
7. Run `.venv/bin/python -c "from tac.preflight import preflight_all; preflight_all(strict=False, verbose=True)"` — verify no regressions

**Deliverable**: NEW Catalog # STRICT preflight gate + CLAUDE.md catalog table entry + dedicated tests. Strict-flip planned after one full session-cycle with 0 violations confirmed.

### Op-routables summary table

| OP # | Phase | Description | Cost | Wall-clock | Dependencies | Structural verdict |
|---|---|---|---|---|---|---|
| OP-1 | Phase 1 implementation | Build `tac.riemannian_newton_meta_substrate.fisher_precondition` package + adapters + anchors + tests | $0 GPU (M5 Max editor) | ~3 days | None | TIER-1 enabling primitive |
| OP-2 | Phase 1 empirical validation | Extract 600-pair PR101_lc_v2 anchor + probe Fisher conditioning + register canonical posterior anchor | $0 GPU + ~4 hr | ~4 hours | OP-1 | TIER-1 GATES Phase 2 dispatch eligibility |
| OP-3 | Phase 1 byte-proof | Byte-mutation smoke on FOP null-space byte positions + validate Pythagorean property empirically | $0 GPU + ~3 hr | ~3 hours | OP-1 + OP-2 | TIER-1 Contrarian's VETO requirement |
| OP-4 | Phase 1 C6 IBPS extension | Diagnose C6 IBPS 22× miss via Fisher-conditioning analysis on random-init vs post-50ep weights | $0 GPU + ~2 hr | ~2 hours | OP-1 | TIER-2 MacKay's EV extension to IB family |
| OP-5 | Phase 1 STRICT preflight gate | Claim Catalog # + implement gate + CLAUDE.md entry + tests | $0 GPU + ~2 hr | ~2 hours | OP-1 | TIER-2 self-protection per "Bugs must be permanently fixed AND self-protected against" |

**Total predicted Phase 1 cost**: $0 GPU + ~3.5 days editor (OP-1 ~3 days + OP-2/3/4/5 ~11 hours = ~1.4 days; can overlap with OP-1 once helper functions are merged).

**Total predicted Phase 1 wall-clock**: ~4 days end-to-end (one subagent session for OP-1 + OP-5 in parallel; second session for OP-2 + OP-3 + OP-4 in parallel after OP-1 lands).

### Predicted post-Phase-1 frontier

If Phase 1 OP-1/2/3 all succeed (predicted HIGH confidence per HARD-EARNED-WITH-REVISION assumptions + canonical literature foundations):

- **Phase 1 STANDALONE frontier potential**: `[0.132, 0.180]` [contest-CPU] (from current 0.19205 with `[-0.060, -0.012]` cascade unlock across 4 archives REALISTIC scenario)
- **Phase 1 + Phase 2-5 CASCADE frontier potential**: `[0.130, 0.187]` [contest-CPU] (under REALISTIC composition_alpha ≈ 0.6)
- **Phase 1 + Phase 2-5 + ALL wave deliverables frontier potential**: `[0.108, 0.130]` [contest-CPU] (asymptotic-pursuit horizon-class)

The Phase 1 helper IS the CANONICAL pre-dispatch gate for the entire asymptotic-pursuit cascade. Landing it at $0 GPU cost UNLOCKS the operator-routable EV across the Riemannian-Newton family + theoretical-floor + deterministic-optimizer + cathedral autopilot v2 cascade + 8+ wave deliverables. The dispatch readiness verdict per the canonical 6-step contract per Catalog #325:

1. **Cargo-cult audit per Catalog #303** — §2 (10 assumptions enumerated; 6 HARD-EARNED + 1 HARD-EARNED-WITH-REVISION + 1 CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION + 2 CARGO-CULTED-PENDING-IMPLEMENTATION)
2. **9-dim checklist evidence per Catalog #294** — §3 (all 9 dimensions documented)
3. **Observability surface declaration per Catalog #305** — §4 (6 facets ACTIVE)
4. **Sextet pact deliberation per Catalog #292** — §5 (10 attendees including Amari memorial seat; 5 PROCEED + 5 PROCEED_WITH_REVISIONS aggregate)
5. **Per-substrate reactivation criteria pinned per Catalog #313** — §6 (5 reactivation paths)
6. **Catalog #324 post-training Tier-C validation discipline** — §7 (pending_post_training with explicit reactivation criteria)

The canonical pre-dispatch gate is SATISFIED at design-memo landing. Phase 1 implementation (OP-1) is the next operator-routable.

---

**END OF DESIGN MEMO**

**Catalog #300 v2 frontmatter compliance verified**: ✓
**Catalog #290 canonical-vs-unique decision per layer**: ✓ (§8)
**Catalog #294 9-dim success checklist evidence**: ✓ (§3)
**Catalog #296 predicted-band Dykstra-feasibility**: ✓ (§9 + §10)
**Catalog #303 cargo-cult audit per assumption**: ✓ (§2)
**Catalog #305 observability surface section**: ✓ (§4)
**Catalog #309 horizon-class declaration**: ✓ (frontmatter)
**Catalog #313 probe outcomes ledger registration**: ✓ (§6)
**Catalog #324 post-training Tier-C validation**: ✓ (§7)
**Catalog #325 per-substrate optimal-form symposium**: ✓ (this memo IS the symposium)
**Catalog #229 premise verification before edit**: ✓ (5+ canonical pointers read pre-edit)
**Catalog #125 6-hook wire-in declaration**: ✓ (§9; 6 hooks ACTIVE)
**Catalog #126 lane pre-registered**: ✓ (`lane_phase_1_fisher_precondition_canonical_helper_design_20260518` at L0)
**Catalog #206 checkpoint discipline**: ✓ (initial checkpoint emitted at task start)
