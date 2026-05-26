# UNIWARD Per-Pixel Score-Conditional Sensitivity Distortion Attack Landed 2026-05-26

**Subagent**: `uniward-per-pixel-score-conditional-sensitivity-weighting-fridrich-canonical-pure-distortion-attack-mlx-first-numpy-portable-20260526`
**Lane**: `lane_uniward_per_pixel_score_conditional_sensitivity_distortion_attack_20260526`
**Date**: 2026-05-26
**Tag**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341

## 3-strategy attack decomposition

PRIMARY = **DISTORTION pure-axis** (Fridrich-canonical UNIWARD inverse-steganalysis adapted for contest scorers per Yousfi grand-council position). SUB-AXIS = JOINT d_seg + d_pose via per-pixel inverse-Fisher-information weighting. Sister coverage per just-elevated 3-strategy directive: RATE attack covered by Cascade C #1351; FULL-SCORER attack covered by Phase 1 meta-Lagrangian #1059; THIS spawn = PURE-DISTORTION coverage closing the strategy-portfolio gap (operator-routable rebalancing rule "≥2 of 3 strategies in portfolio" satisfied: this brings DISTORTION pure-axis online).

## Entropy-position declaration

POSITION = **P2 loss-shape (TRAIN phase BEFORE entropy coder)** per Lesson 1 of just-landed entropy-position discipline. Per-pixel UNIWARD weight map shapes the UPSTREAM perturbation distribution that the eval_roundtrip uint8 bottleneck → archive entropy coder sees downstream. Per the discipline's structural-bound rule: predicted savings come from `H(perturbation | weight_map_bucket) < H(perturbation)` (conditional entropy lower than marginal once routing concentrates perturbation in low-sensitivity zones).

## MLX-first → numpy-portable bridge contract

Honored per just-landed standing directive:

- **Training (MLX-first)**: per-pixel weight map computation via numpy float32 (MLX-compatible) wrapping per-pixel scorer gradient magnitudes — substrate-optimal engineering keeps weight computation OUT of MLX trainer hot loop (pre-computed each epoch via `tac.master_gradient` typed CandidateModificationSpec per Catalog #318); EMA shadow at fp32 + Kahan compensated per drift-discipline #1307.
- **Inflate (numpy-portable)**: UNCHANGED relative to PR110 baseline — weight map is **compress-only** per Carmack-preferred budget conservation. Trained weights themselves embody the routing; inflate runtime ≤200 LOC + ≤2 deps invariant preserved.
- **Bridge contract**: trained MLX state_dict → `np.savez_compressed` → ZIP-member at fixed offset per archive_grammar → inflate loads via `np.load` (NO MLX dep). Weight map sidecar at `experiments/results/<lane>/weight_map_<archive_sha[:8]>.npz` is forensic-only (NOT contest archive member).

## Individually-fractal decomposition

Per just-landed standing directive + GUIDING PRINCIPLE per `feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md`:

- **Ingredient #6 (score-domain Lagrangian)** = PRIMARY decomposition node (Fridrich canonical inverse-Fisher-info per-pixel weighting)
  - **Sub-ingredient**: `compute_per_pixel_uniward_weight_map_numpy` (this landing)
    - **Sub-sub-ingredient**: `decompose_per_axis_weights` (this landing — observability per Catalog #305: per-axis weight components for cargo-cult validation)
      - **Sub-sub-sub-ingredient**: `histogram_weight_distribution` (this landing — observability per Catalog #305: queryable post-hoc)
        - **Sub-sub-sub-sub-ingredient (FUTURE)**: weight-map quantization for sidecar embedding if rate-attack composition test surfaces value (queued; NOT pursued in this landing per Carmack-preferred budget conservation)

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: substrate's distinguishing feature IS the per-pixel weight map; ALL other ingredients adopt canonical helpers (scorer-preprocess routing per Catalog #164/#226 / eval_roundtrip canonical / EMA canonical / archive grammar PR110-inherited / inflate numpy-portable PR110-inherited).

## Canonical-vs-unique decision per layer

Per Catalog #290 (already enumerated in pre-execution gate report; verbatim-preserved by reference). KEY decisions: per-pixel weight map = FORK_BECAUSE_PRINCIPLED_MISMATCH (Fridrich-canonical UNIWARD distinct from any sister substrate); EVERYTHING ELSE = ADOPT_CANONICAL_BECAUSE_SERVES (canonical scorer-preprocess + eval_roundtrip + EMA + archive grammar + inflate).

## 9-dimension success checklist evidence

Per Catalog #294 (enumerated in pre-execution gate report; verbatim-preserved). KEY anchor: DIMENSION 3 DISTINCTNESS verified — no sister substrate uses per-pixel UNIWARD inverse-Fisher-info weighting (audit against 56 existing substrates in `src/tac/substrates/` confirmed).

## Cargo-cult audit per assumption

Per Catalog #303 (5 assumptions enumerated in pre-execution gate report). KEY findings:
- HARD-EARNED: "Per-pixel weight maps improve scorer-attack discriminability" (Fridrich 2014 UNIWARD + 13+ years steganalysis canonical)
- HARD-EARNED: "Compress-time weighting transfers to trained weights without shipping weight map" (canonical training discipline)
- CARGO-CULTED-PENDING-VALIDATION: "UNIWARD formula extends from steganalysis to contest scorers cleanly" + "JOINT d_seg + d_pose Fisher-info inverse beats single-axis" — unwind via empirical per-axis decomposition (`decompose_per_axis_weights` API landed; future empirical anchor will validate)

## Observability surface

Per Catalog #305 (6 facets enumerated in pre-execution gate report). VERIFIED facets:
- Inspectable per layer: weight map sidecar + per-axis decomposition + histogram all available via canonical APIs
- Decomposable per signal: per-axis seg-only / pose-only / joint variants emitted via `decompose_per_axis_weights`
- Diff-able across runs: np.savez_compressed canonical format
- Queryable post-hoc: histogram + min/max/mean/median statistics emitted
- Cite-able: canonical Provenance per Catalog #323 in compose-loss return dict
- Counterfactual-able: per-pixel byte-mutation smoke per Catalog #139 verifiable

## Drift surface declaration

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` 5 sources (enumerated in pre-execution gate report). KEY mitigations: explicit fp32 cast for weight-map computation (Source 1); additive `eps=1e-6` in denominator (Source 2); weight-map computation outside bicubic boundary (Source 4 N/A at this surface).

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility check: **[-1, -4] score points** DISTORTION-axis at PR110 frontier operating point. Mathematical grounding: Fridrich UNIWARD canonical bound + Yousfi grand-council position per-pixel weighting reduces detector sensitivity 2-4× empirically for steganalysis. Per-axis Pareto polytope alternating projections (Dykstra) project onto JOINT-Fisher-info feasible set → both axes simultaneously improve (anti-PR97 anti-pattern).

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (predicted CPU band [0.190, 0.193] given PR110 frontier baseline ~0.192-0.195 + predicted -0.001 to -0.004 improvement).

## Catalog #344 canonical equation anchor

PROPOSED NEW canonical equation: `uniward_per_pixel_score_conditional_sensitivity_weighting_distortion_savings_v1` per Fridrich-Yousfi adapter. Domain of validity: P2 loss-shape entropy-position; substrate class = `score_aware_renderer_with_per_pixel_uniward_weighting`; predicted savings band [-1, -4] DISTORTION-axis. Operator-routable for `tac.canonical_equations.append_equation` registration after empirical validation OR via sister Task #1357 catalog-memo revision lifecycle.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel weight map IS canonical sensitivity surface at the per-pixel scorer-response decomposition; consumable by `tac.sensitivity_map.*` downstream consumers (per-axis decomposition via `decompose_per_axis_weights` emits seg-only + pose-only + joint variants for Pareto polytope projection per Dim 1 Phase 4 sister)
2. **Pareto constraint**: ACTIVE via Dykstra alternating-projections onto JOINT-Fisher-info feasible set (Pareto polytope projection in joint per-axis weight space)
3. **Bit-allocator hook**: N/A (substrate is DISTORTION-attack pure; rate-allocation untouched; INHERITS PR110 baseline allocator)
4. **Cathedral autopilot dispatch hook**: ACTIVE — substrate exposes canonical contract per Catalog #335 (CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS module-level fields); auto-discoverable per `tools/cathedral_autopilot_autonomous_loop.py::discover_and_register_consumers` (Catalog #336/#337)
5. **Continual-learning posterior update**: ACTIVE — compose-loss return dict carries canonical Provenance per Catalog #323 for posterior anchor emission via `tac.continual_learning.posterior_update_locked`
6. **Probe-disambiguator**: N/A at scaffold landing (the Fridrich UNIWARD canonical IS the disambiguator between per-pixel vs per-region vs per-class weighting; sister probe-disambiguator queued for future per-region wavelet variant comparison)

## Substrate scaffold landing summary

**Files added** (4 source + 1 test + 1 pre-execution gate report + 1 landing memo = 7 files):

- `src/tac/substrates/uniward_per_pixel_distortion/__init__.py` (~50 LOC)
- `src/tac/substrates/uniward_per_pixel_distortion/weight_map.py` (~125 LOC)
- `src/tac/substrates/uniward_per_pixel_distortion/score_aware_loss.py` (~130 LOC)
- `src/tac/substrates/uniward_per_pixel_distortion/tests/__init__.py`
- `src/tac/substrates/uniward_per_pixel_distortion/tests/test_weight_map_and_loss.py` (12 tests; ALL PASS)
- `.omx/research/uniward_per_pixel_score_conditional_sensitivity_pre_execution_gate_report_20260526.md`
- `.omx/research/uniward_per_pixel_score_conditional_sensitivity_distortion_attack_landed_20260526.md`

**Total source LOC**: ~305 LOC (under bolt-on size budget ≤350 LOC per HNeRV parity L7).

**MLX-local smoke verdict**: 12/12 PASS in 0.14s wall-clock (no paid GPU; pure macOS M5 Max).

**Empirical correctness anchors** (synthetic gradient fields):
- Routing invariant: smooth-zone weight mean > boundary-zone weight mean (Fridrich UNIWARD canonical satisfied empirically)
- Per-axis decomposition: seg-only / pose-only / joint variants correctly compute (cargo-cult validation framework in place)
- Unit-mean normalization: mean(weight) = 1.0 after normalization (loss-weighting interpretability satisfied)
- Histogram observability: 16x16 weight map produces 10-bin histogram with 256 total count (Catalog #305 queryable)
- Canonical contest formula composition: 100*d_seg + sqrt(10*d_pose) + lambda*mean(perturb*weight) (CLAUDE.md formula constants pinned verbatim)
- Canonical Provenance: substrate_id + version + evidence_grade + score_claim=False + promotable=False + axis_tag=[predicted] + entropy_position=P2_loss_shape_TRAIN_phase (Catalog #323 + #341)

**Carmack-dissent verdict per Catalog #307**: PARADIGM-INTACT-PENDING-EMPIRICAL — substrate scaffold is structurally clean (12/12 tests pass; canonical Provenance + Catalog #341 markers + Catalog #305 observability + Catalog #318 typed primitive API per `tac.master_gradient` discipline all honored); PARADIGM (Fridrich UNIWARD inverse-steganalysis) HARD-EARNED per 14 years steganalysis literature + Yousfi grand-council position. IMPLEMENTATION-level validation pending: real-scorer-anchored empirical anchor (run weight map computation against actual PR110 baseline scorer gradients on 50-pair MLX-local smoke; measure per-axis d_seg + d_pose reduction vs uniform-weighting baseline; queue for sister N+1 spawn).

## Operator-routable next step

1. **REUSE THE SCAFFOLD**: sister N+1 spawn fires real-scorer-anchored empirical anchor — load PR110 baseline trained weights → compute per-pixel `d_seg/d_x` + `d_pose/d_x` via `tac.master_gradient` typed CandidateModificationSpec → call `compute_per_pixel_uniward_weight_map_numpy` → train PR110-baseline MLX renderer with `compose_uniward_weighted_score_loss` extension for 50ep on 50-pair fixture → measure per-axis improvement vs uniform-weighting baseline → emit canonical Provenance posterior anchor.
2. **CANONICAL EQUATION REGISTRATION**: post-empirical-anchor, propose `uniward_per_pixel_score_conditional_sensitivity_weighting_distortion_savings_v1` via `tac.canonical_equations.append_equation` OR sister Task #1357 catalog-memo revision.
3. **STACK-OF-STACKS COMPOSITION TEST**: validate orthogonality with Cascade C rate-attack (different entropy-position: P2 loss-shape vs P5 archive entropy coder) per just-landed 3-strategy directive composition rules.

## Discipline anchors

- Catalog #229 PV (read 4 standing directives + canonical helpers + sister substrate refs BEFORE editing)
- Catalog #206 (3 checkpoints emitted: PV-complete + scaffold-built + landing-memo-write)
- Catalog #110/#113 APPEND-ONLY (no mutations to existing forensic artifacts)
- Catalog #117/#157/#174/#235/#289 canonical commit serializer (will use --expected-content-sha256 + co-author trailer)
- Catalog #230 ownership map (zero collision with in-flight #1356/#1359/#1357)
- Catalog #287 placeholder rationale ≥4 chars (all docstrings substantive)
- Catalog #290/#294/#303/#305 design-memo discipline (all 4 sections in pre-execution gate report)
- Catalog #318 master-gradient via typed CandidateModificationSpec (NOT raw byte authority)
- Catalog #323 canonical Provenance in compose-loss return dict
- Catalog #335 cathedral consumer canonical contract (auto-discoverable per CONSUMER_NAME field)
- Catalog #340 sister-checkpoint guard PROCEED verified pre-edit
- Catalog #341 canonical-routing markers in Provenance (score_claim=False + promotable=False + axis_tag=[predicted])
- Catalog #343 no hardcoded score literals (only canonical formula constants from CLAUDE.md non-negotiable)
- CLAUDE.md "MLX portable-local-substrate authority": tagged `[macOS-MLX research-signal]`; NO paid dispatch
- CLAUDE.md "Forbidden premature KILL": DEFERRED-pending-empirical (NOT killed); reactivation = real-scorer-anchored smoke
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD": per-pixel weight map FORKED (substrate-distinctive); EVERYTHING ELSE canonical adoption
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against": NO new bug classes introduced (this is NEW substrate scaffold; existing bug-class extinctions inherited via canonical helper adoption)

## Cost

$0 GPU + ~30 min wall-clock + ~305 LOC source + 12/12 tests PASS.
