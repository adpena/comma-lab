<!-- SPDX-License-Identifier: MIT -->

# META-LAGRANGIAN DUAL SOLVER PHASE 2 PRE-EXECUTION GATE REPORT 2026-05-26
<!-- # FORMALIZATION_PENDING:meta_lagrangian_dual_solver_per_axis_kkt_residual_v1_canonical_equation_proposed_in_landing_memo_pending_post_smoke_anchor_registration -->
<!-- # HISTORICAL_SCORE_LITERAL_OK:pose_avg_PR106_operating_point_3p4e_minus_5_referenced_per_CLAUDE_md_SegNet_vs_PoseNet_importance_section_2026-05-04_for_dykstra_polytope_baseline -->

**Subagent_id**: `meta-lagrangian-dual-solver-phase-2-per-axis-dual-variable-computation-pure-full-scorer-attack-mlx-first-numpy-portable-individually-fractal-20260526`
**Lane**: `lane_meta_lagrangian_dual_solver_phase_2_20260526`
**Apparatus surface**: `tac.findings_lagrangian` + `tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates`

## 3-strategy attack decomposition

Per 3-strategy directive elevation:

- **PRIMARY = FULL-SCORER pure** (apparatus-scope advancement). Phase 2 advances Catalog #355 Phase 1 mock per-candidate adjustment to actual per-axis dual-variable computation via Dykstra alternating projections on the 3D seg/pose/rate Pareto polytope.
- DISTORTION pure: N/A (Cascade B sister wave is the canonical distortion-attack subagent; this is apparatus, not substrate).
- RATE pure: N/A (apparatus-level Lagrangian dual; rate is one of the 3 polytope axes).

Sister-scope-disjoint from UNIWARD N+1 (substrate `uniward_per_pixel_distortion`) and Cascade B Path A sister wave 1 (substrate `hinton_distilled_scorer_surrogate`).

## Entropy-position declaration

P21 meta-Lagrangian-dual-variable-entropy per just-added § 9 sister entropy-position from T3 council § 9.

The dual variables λ_seg, λ_pose, λ_rate are a per-axis KKT residual signal whose entropy under the cathedral autopilot ranker's per-candidate posterior IS the operational entropy-position. Phase 1 collapses this to a 1-dim sigma; Phase 2 expands to 3-dim per-axis sigma.

## MLX-first → numpy-portable bridge contract

- **MLX-native**: Dykstra alternating projections kernel uses `mlx.core.array` for the 3D dual-variable updates. The projection onto each axis half-space is a closed-form 1-line MLX expression per Dykstra 1983 (Boyd-Dattorro 2006 modern treatment).
- **Numpy-portable**: the canonical helper accepts numpy `np.ndarray` inputs and returns numpy outputs via `mx.eval(arr)` + `np.asarray(arr)`. The MLX/numpy bridge lives at the boundary so CUDA-side cathedral autopilot ranker callers (when sister wave eventually lands a non-MLX environment) consume the same interface.
- **Cathedral autopilot ranker surface (not contest-archive direct)**: this is APPARATUS-SCOPE per Catalog #355; the ranker's per-candidate per-axis adjustment factor remains bounded [0.95, 1.05] per Phase 1 contract preservation.
- **MLX gradient discipline per `feedback_mlx_first_with_numpy_portable_bridge_standing_directive_20260526.md`**: the Dykstra primal-dual iteration only uses MLX for arithmetic; no gradients flow through MLX into PyTorch (per CLAUDE.md "MPS auth eval is NOISE" non-negotiable, MLX is research-signal only). All per-candidate per-axis adjustment factors carry `axis_tag="[predicted]"` + `score_claim=False` + `promotable=False` per Catalog #341.

## Individually-fractal decomposition

Per just-elevated GUIDING PRINCIPLE *"indivually fractally optimized"*:

- Layer 1: Ingredient #6 score-domain Lagrangian (Meta-Lagrangian Phase 2 IS the canonical apparatus-scope embodiment).
- Layer 2: Sub-ingredient per-candidate per-axis dual-variable computation (Phase 1 mock → Phase 2 actual).
- Layer 3: Sub-sub-ingredient Dykstra alternating projections convergence (per Dykstra CO-LEAD canonical primitive; Boyd grand-council operational level).
- Layer 4: Sub-sub-sub-ingredient per-axis KKT residual computation (sister of `tac.master_gradient` typed `CandidateModificationSpec` per Catalog #318).
- Layer 5: Sub-sub-sub-sub-ingredient cathedral autopilot ranker per-candidate per-axis adjustment factor application (Phase 1 wire-in extends to per-axis breakdown; Phase 2 surfaces `adjustment_factor_per_axis` dict alongside scalar adjustment).

## Canonical-vs-unique decision per layer

Per Catalog #290:

- **`tac.findings_lagrangian` (canonical)**: ADOPT. The Phase 1 wire-in already routes through `posterior_update_from_anchors` + `compute_findings_lagrangian` + `build_initial_partition`. Phase 2 EXTENDS via a NEW sister module `tac.findings_lagrangian.dual_solver_phase_2`; the existing 4-term Lagrangian becomes the SCALAR objective inside the Dykstra inner loop.
- **`tac.score_composition.AxisDecomposition` + `ComposedScoreDelta` (canonical)**: ADOPT. The Dykstra primal-dual iterations produce per-axis predictions; the existing `AxisDecomposition` Protocol extension IS the canonical Provenance-bearing serialization surface.
- **`tac.cathedral.consumer_contract` (canonical)**: ADOPT. The Phase 2 helper does NOT introduce a new consumer; it remains the meta-Lagrangian solver wire-in callsite (Catalog #355 surface) NOT a `cathedral_consumers/*` package (Catalog #335 surface) — the distinction is structural per "Phase 1 wire-in is invocation-callsite NOT auto-discovered consumer package".
- **Dykstra alternating projections (canonical primitive per Dykstra CO-LEAD inner council)**: ADOPT. Boyd-Dattorro 2006 "Alternating Projections on Manifolds" is the canonical reference; closed-form projections onto axis half-spaces (per-axis budget constraints) compose into the polytope intersection per Dykstra's correction-step convergence proof.
- **Per-axis KKT residuals (UNIQUE-PER-METHOD per Phase 2 advancement)**: FORK. The canonical `tac.master_gradient` typed `CandidateModificationSpec` produces RAW byte-mutation gradients per Catalog #318; Phase 2's per-axis KKT residuals operate at the AXIS-AGGREGATE level (per-candidate per-axis predictions), NOT at the raw-byte level. This is a sister scale, not raw-byte sister — distinct contract per Catalog #318 explicit refusal of raw-byte authority.
- **Bounded [0.95, 1.05] exposed adjustment SAFETY ENVELOPE (canonical)**: ADOPT from Phase 1. Phase 2 enables UNBOUNDED per-axis dual computation INTERNALLY but the EXPOSED `adjustment_factor` to cathedral autopilot ranker remains [0.95, 1.05] per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323 canonical Provenance + Catalog #341 routing markers. The new `adjustment_factor_per_axis` dict is OBSERVABILITY-ONLY (axis_tag="[predicted]"; not promotable; not score-claim).

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: Phase 2 dual-variable computation differs structurally from Phase 1 mock (uniform Gaussian residual proxy collapses into a single scalar; Phase 2 per-axis Dykstra produces 3 distinct dual variables per candidate).
2. **BEAUTY + ELEGANCE**: Dykstra alternating projections converges in O(log(1/ε)) iterations for the 3-axis polytope (Boyd 2004); the implementation fits in ~150 LOC for the core kernel.
3. **DISTINCTNESS**: Distinct from sister `tac.master_gradient` (raw-byte authority per Catalog #318 refusal); distinct from sister Cascade C' Lagrangian dual primitive (substrate-scope; THIS is apparatus-scope).
4. **RIGOR**: Closed-form Dykstra projection per axis (1-line each); convergence proof per Boyd-Dattorro 2006; per-axis KKT residuals provide canonical sensitivity signal.
5. **OPTIMIZATION PER TECHNIQUE**: MLX-native arithmetic for the Dykstra inner loop; numpy boundary at the interface; bounded [0.95, 1.05] exposed adjustment safety envelope.
6. **STACK-OF-STACKS-COMPOSABILITY**: Composable with Phase 1 invocation contract (sister Catalog #355 surface preservation); composable with `tac.score_composition` per Catalog #356 sister; composable with `tac.master_gradient` per Catalog #318 axis-aggregate-scale boundary.
7. **DETERMINISTIC REPRODUCIBILITY**: MLX `mx.random.seed(...)` for any stochastic primitive (Phase 2 is currently deterministic; no randomness in Dykstra alt-projections); numpy `np.random.seed(...)` at the bridge.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Dykstra inner loop bounded at 50 iterations (typical convergence in 10-20 per 3D polytope); per-candidate cost O(N_iters × 3) MLX ops; per-batch O(N_candidates × N_iters × 3) — fits in M5 Max budget.
9. **OPTIMAL MINIMAL CONTEST SCORE**: APPARATUS-GROWTH (not direct score-lowering). Per-candidate per-axis adjustment factor enables sharper ranking which DOWNSTREAM enables better dispatch decisions which DOWNSTREAM enables score-lowering. The chain is documented; no direct contest-score claim.

## Cargo-cult audit per assumption

Per Catalog #303:

- **Phase 1 mock Gaussian posterior assumption** (uses candidate's `predicted_score_delta` as single residual): HARD-EARNED for Phase 1 (operator approved bounded proxy to exercise wiring); CARGO-CULTED if extended to Phase 2 (collapses 3-axis polytope into 1D; loses per-axis structure). **Unwind**: Phase 2 replaces single-scalar residual with per-axis residuals derived from the candidate's `predicted_axis_decomposition` (when emitted per Catalog #356) OR canonical equation `contest_score_formula_v1` Jacobian inversion (when only scalar `predicted_score_delta` is available).
- **Bounded [0.95, 1.05] exposed adjustment safety envelope**: HARD-EARNED. Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323 canonical Provenance: the apparatus's prediction-vs-empirical posterior is the canonical promotion arbiter, NOT the Lagrangian dual. The bounded envelope structurally protects against future regressions where dual-variable explosion could leak into promotion.
- **3-axis polytope assumption (seg + pose + rate ONLY)**: HARD-EARNED. The contest score formula `S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37545489` is canonical per `contest_score_formula_v1`. No 4th axis exists at the apparatus level.
- **Dykstra alternating projections convergence assumption**: HARD-EARNED for convex polytopes (Boyd-Dattorro 2006 proof); CARGO-CULTED if applied to non-convex feasible sets. **Verification**: per-axis half-space constraints are linear → polytope is convex by construction.

## Observability surface

Per Catalog #305:

- Per-candidate per-axis dual variable values (`lambda_seg`, `lambda_pose`, `lambda_rate`) surfaced in the `invocations[].dual_variables_per_axis` field of the Phase 2 helper output.
- Per-candidate Dykstra convergence iteration count surfaced in `invocations[].dykstra_iterations_to_convergence`.
- Per-candidate per-axis KKT residual values surfaced in `invocations[].kkt_residual_per_axis`.
- Per-candidate per-axis adjustment factor distribution (dict keyed by `{seg, pose, rate}`) surfaced in `invocations[].adjustment_factor_per_axis`.
- Per-candidate scalar `adjustment_factor` PRESERVED from Phase 1 (the bounded [0.95, 1.05] envelope; cathedral autopilot ranker continues consuming the scalar; the new per-axis dict is observability-only).
- All emitted rows carry `axis_tag="[predicted]"` + `score_claim=False` + `promotable=False` per Catalog #341.

## Drift surface declaration

Per MLX↔CUDA bidirectional drift discipline (just-landed standing directive memo set):

- Dykstra convergence iterations MAY differ between MLX (M5 Max) and CUDA per substrate; the apparatus-scope ranker's per-candidate per-axis adjustment factor IS expected to be stable to within 1e-4 across substrates (bounded by the [0.95, 1.05] envelope's resolution).
- Per-axis KKT residual values MAY differ by ~1e-6 across substrates due to fp32 vs fp64 (MLX default is fp32; numpy default is fp64); the convergence criterion uses fp32 ε threshold per MLX-first directive.
- Per `mlx_cuda_bidirectional_drift_engineering_response_v1` (just-registered canonical equation): drift is engineering response (bridge contract) NOT noise; the apparatus's bounded envelope structurally absorbs any per-substrate convergence variance.

## Predicted ΔS band

Per Catalog #296 (Dykstra-feasibility check):

- **Apparatus-growth (NOT direct score-lowering)**: per-candidate per-axis adjustment factor IS bounded [0.95, 1.05] by SAFETY ENVELOPE per Phase 1 contract preservation; downstream ΔS effect on cathedral autopilot ranking decisions is ~0.05 × (top-candidate predicted_delta) per iteration = O(0.05 × 0.005) ≈ ±2.5e-4 per top-10 candidate.
- **Predicted apparatus efficiency improvement**: per-candidate per-axis decomposition enables Dim 1 Phase 4 (Dykstra alternating-projections on Pareto polytope) and Dim 6 Step 6.5 (Tier B score-contributing canonical contract sister) per cathedral autopilot smarter design blueprint § Dim 1 + Dim 3 + Dim 6.
- **Dykstra-feasibility check**: the 3-axis polytope (seg, pose, rate) is CONVEX by construction (linear half-space constraints); alternating projections converges to the polytope intersection per Boyd-Dattorro 2006 proof.
- **Apples-to-apples**: apparatus-scope ranker quality improvement IS observable via predicted-vs-empirical residuals AFTER sister wave actually dispatches the re-ranked top-K candidates and harvests results; THIS lane only lands the wiring + smoke + observability.

## Horizon-class declaration

Per Catalog #309:

- **Horizon class**: `apparatus-maintenance` + `frontier-protecting`. Phase 2 advances the canonical "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" surface per CLAUDE.md but DOES NOT directly lower the contest score.
- **Direct score impact**: 0 (apparatus-growth only; ranker quality improvement DOWNSTREAM enables better dispatch decisions DOWNSTREAM enables score-lowering).
- **Indirect score impact**: enables sister Dim 1 Phase 4 + Dim 3 Step 3.4 + Dim 6 Step 6.5 landings per cathedral autopilot smarter design blueprint.

## Catalog #344 canonical equation target

Proposed NEW canonical equation `meta_lagrangian_dual_solver_per_axis_kkt_residual_v1`:

- **In-domain context tokens**: `cathedral_autopilot_per_candidate_per_axis_adjustment` / `meta_lagrangian_dual_variables` / `dykstra_alternating_projections_3_axis_polytope` / `per_axis_kkt_residual_sensitivity_signal`.
- **Excluded context tokens**: `raw_byte_authority` (per Catalog #318 refusal) / `substrate_level_dual_variables` (substrate-scope is sister Cascade C' canonical primitive; this is apparatus-scope) / `direct_score_claim` (per Catalog #323 canonical Provenance: this is `[predicted]` apparatus-growth surface).
- **Producer surface**: `tac.findings_lagrangian.dual_solver_phase_2.compute_per_axis_dual_variables` (new in this lane).
- **Consumer surface**: `tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates` (Phase 2 extension).
- **Initial prediction**: Dykstra convergence in ≤ 20 iterations for 3-axis polytope on M5 Max MLX with ε = 1e-5; per-candidate per-axis adjustment factor bounded [0.95, 1.05] in exposed interface; per-axis KKT residual values ∈ [-1, +1] (normalized).
- **Registration**: deferred to landing memo after MLX-local smoke produces empirical anchor.

## Premise verification

Per Catalog #229:

- ✅ Read `tac.findings_lagrangian` Phase 1 helper at `tools/cathedral_autopilot_autonomous_loop.py:7165-7403`.
- ✅ Read `tac.findings_lagrangian/__init__.py` canonical package API (~150 LOC `__all__` enumeration).
- ✅ Read `tac.findings_lagrangian/lagrangian.py` 4-term Lagrangian (`compute_findings_lagrangian` returns `FindingsLagrangianResult` with `scalar` + `decompose` + `posterior_sigma_per_term`).
- ✅ Read `tac.findings_lagrangian/posterior.py` `GaussianPosterior` + `posterior_update_from_anchors` (1d conjugate; cycles residuals through dimensions if N > d).
- ✅ Read `tac.score_composition/__init__.py` `AxisDecomposition` + `ComposedScoreDelta` + `compose_score_from_axes` (canonical formula composer).
- ✅ Read `tac.cathedral.consumer_contract` `AxisDecomposition` Protocol extension.
- ✅ Verified MLX 0.31.2 available locally.
- ✅ Verified canonical equations registry API (`register_canonical_equation` + `update_equation_with_empirical_anchor`).
- ✅ Catalog #355 STRICT preflight gate contract: Phase 2 PRESERVES the canonical invocation tokens (`invoke_meta_lagrangian_on_candidates` / `discover_and_register_consumers` / `recommend_next_action_via_expected_information_gain`); the Phase 1 → Phase 2 advancement is INTERNAL to the helper.

## Sister-scope-disjoint declaration

Per Catalog #230 + #302 + #340:

- UNIWARD N+1 real-scorer empirical (#1369; substrate `uniward_per_pixel_distortion`): DISJOINT. THIS lane does not edit `src/tac/substrates/uniward_per_pixel_distortion/`.
- Cascade B Path A sister wave 1 (#1370; substrate `hinton_distilled_scorer_surrogate`): DISJOINT. THIS lane does not edit `src/tac/substrates/hinton_distilled_scorer_surrogate/`.
- (NSCS06 v8 Modal CUDA in flight, paid; operationally separate; not in working tree this turn.)

## Operator-routable next-step plan

After STEP 8 landing memo:
- (Phase 3) typed atom flow into solver: route `CandidateRow` objects through the Dykstra solver as typed atoms with per-axis Provenance.
- (Phase 4) per-element learned-optimal destination per META engineering vision.

Plan APPROVED for execution. Proceeding to STEP 2 (Phase 2 implementation).
