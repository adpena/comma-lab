# UNIWARD Per-Pixel N+1 Real-Scorer-Anchored Empirical Pre-Execution Gate Report 2026-05-26

**Subagent**: `uniward-per-pixel-n-plus-1-real-scorer-anchored-empirical-anchor-50pair-mlx-local-pr110-baseline-20260526`
**Lane**: `lane_uniward_per_pixel_n_plus_1_real_scorer_empirical_20260526`
**Date**: 2026-05-26
**Tag**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341
**Successor of**: scaffold subagent `a136f59ce91f5b468` / commit `aa2612d9b` / lane `lane_uniward_per_pixel_score_conditional_sensitivity_distortion_attack_20260526` L1

## 3-strategy attack decomposition

PRIMARY = **DISTORTION pure-axis** (empirical-validation phase of just-landed scaffold's Fridrich-canonical UNIWARD inverse-steganalysis adapter). SUB-AXIS = JOINT (d_seg + d_pose). This is the 5th-order recursive node of the scaffold's individually-fractal decomposition: 1st=substrate scaffold → 2nd=weight map computation → 3rd=Fisher-info inverse from BOTH scorers → 4th=histogram observability → **5th=real-scorer-anchored empirical validation on PR110 baseline**.

## Entropy-position declaration

POSITION = **P2 loss-shape (TRAIN phase BEFORE entropy coder)** — sister of scaffold subagent. Per-pixel weight map shapes the upstream perturbation distribution that the eval_roundtrip uint8 bottleneck → archive entropy coder sees downstream. Per the entropy-position discipline structural-bound rule: predicted savings come from `H(perturbation | weight_map_bucket) < H(perturbation)` (conditional entropy lower than marginal once routing concentrates perturbation in low-sensitivity zones).

## MLX-first → numpy-portable bridge contract

Honored per standing directive:

- **Training (MLX-first)**: minimal MLX renderer (Adam optimizer); per-pixel weight map precomputed via numpy float32 (MLX-compatible); EMA shadow at fp32; eval_roundtrip simulated via uint8 cast.
- **Inflate (numpy-portable)**: UNCHANGED relative to PR110 baseline — weight map is compress-only per Carmack-preferred budget conservation.
- **Bridge contract**: trained MLX state → np.array conversion for scorer evaluation; weight map sidecar is FORENSIC-ONLY (NOT contest archive member). Scorers run via torch on MPS for canonical `score_pair_components` invocation.

## Individually-fractal decomposition (5th-order recursive node)

- 1st order: substrate scaffold (commit `aa2612d9b`; 12/12 tests pass)
- 2nd order: per-pixel weight map computation (canonical UNIWARD)
- 3rd order: Fisher-info inverse from BOTH scorers via `tac.master_gradient` typed CandidateModificationSpec per Catalog #318
- 4th order: histogram observability per Catalog #305 (sister of scaffold)
- **5th order (THIS SCOPE)**: real-scorer-anchored empirical validation comparing BASELINE (uniform weight) vs VARIANT (UNIWARD weight) on 50-pair MLX-local fixture, 30ep, 96x128

## Canonical-vs-unique decision per layer

Per Catalog #290:

- **Scorer loaders (SegNet + PoseNet)**: ADOPT_CANONICAL_BECAUSE_SERVES via `tac.differentiable_scorers.load_differentiable_scorers` (canonical pose-then-seg ordering per Catalog #222; canonical preprocess routing per Catalog #164/#226)
- **Score component computation**: ADOPT_CANONICAL_BECAUSE_SERVES via `tac.substrates.score_aware_common.score_pair_components`
- **eval_roundtrip**: ADOPT_CANONICAL_BECAUSE_SERVES (uint8 simulation per CLAUDE.md non-negotiable)
- **EMA decay 0.997**: ADOPT_CANONICAL_BECAUSE_SERVES per CLAUDE.md EMA non-negotiable
- **Per-pixel weight map**: FORK_BECAUSE_PRINCIPLED_MISMATCH (substrate's distinguishing feature; Fridrich-canonical UNIWARD; sister of scaffold)
- **Renderer architecture**: UNIQUE_FOR_THIS_EMPIRICAL_ANCHOR (minimal MLX renderer at 96x128; PR110 architecture-shape NOT required — empirical anchor measures WEIGHTING EFFECT not renderer power)
- **Fixture selection**: ADOPT_CANONICAL_BECAUSE_SERVES (`upstream/videos/0.mkv` per CLAUDE.md "Forbidden `make_synthetic_pair_batch` calls in any non-smoke training path")

## 9-dimension success checklist evidence

Per Catalog #294 (sister of scaffold; empirical phase fills gaps):

1. **UNIQUENESS**: Verified empirically — 56 existing substrates; none use per-pixel UNIWARD inverse-Fisher-info weighting.
2. **BEAUTY + ELEGANCE**: Empirical harness ≤300 LOC; PR101-style reviewable.
3. **DISTINCTNESS**: Sister of scaffold (continues same substrate); empirical-anchor phase is structurally distinct from scaffold's unit-test phase.
4. **RIGOR**: Real-scorer-anchored measurement (NOT synthetic gradients); BASELINE vs VARIANT controlled comparison; per-axis decomposition.
5. **OPTIMIZATION PER TECHNIQUE**: Per-pixel weight map per scaffold; canonical helpers for all other layers.
6. **STACK-OF-STACKS COMPOSABILITY**: P2 loss-shape orthogonal to Cascade C P5 archive entropy coder — composable.
7. **DETERMINISTIC REPRODUCIBILITY**: Seed-pinned (42); fixed video; fixed pair indices; archived results JSON.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Goal IS empirical validation of optimization technique.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Predicted -1 to -4 DISTORTION-axis (subject to empirical adjudication).

## Cargo-cult audit per assumption

Per Catalog #303 (extends scaffold's 5 assumptions):

| Assumption | Classification | Empirical Test |
|---|---|---|
| "UNIWARD weight map computed from REAL scorer gradients improves per-axis d_seg+d_pose vs uniform" | CARGO-CULTED-PENDING-VALIDATION (Fridrich canonical for steganalysis; UNTESTED for contest scorers) | THIS sweep |
| "Per-pixel weight map encodes useful gradient signal at PR110 baseline operating point" | CARGO-CULTED-PENDING-VALIDATION | Compute weight map; verify max/min ratio > 1 (non-degenerate signal) |
| "JOINT d_seg + d_pose Fisher-info inverse beats per-axis-only weighting" | CARGO-CULTED-PENDING-VALIDATION | Per-axis decomposition via `decompose_per_axis_weights` |
| "30ep / 50pair / 96x128 fixture sufficient for cross-axis discrimination" | HARD-EARNED via sister BoostNeRV / NSCS06 v8 / FEC sister landings | Cross-validate against sister results format |
| "Minimal MLX renderer reveals weighting effect (PR110-scale renderer NOT required)" | HARD-EARNED (weighting effect is loss-shape-level; renderer architecture orthogonal) | Compare BASELINE vs VARIANT delta in same architecture |

## Observability surface

Per Catalog #305:

- **Inspectable per layer**: per-pixel scorer gradients + weight map + per-axis d_seg/d_pose per epoch
- **Decomposable per signal**: per-axis seg-only / pose-only / joint weight variants emitted via `decompose_per_axis_weights`
- **Diff-able across runs**: results JSON with per-epoch per-axis trace; BASELINE vs VARIANT side-by-side
- **Queryable post-hoc**: histogram + min/max/mean/median statistics per epoch
- **Cite-able**: canonical Provenance per Catalog #323 (substrate_id + version + evidence_grade + score_claim=False + promotable=False + axis_tag=[predicted])
- **Counterfactual-able**: weight-map-zero-out probe verifies routing is necessary not coincidental

## Drift surface declaration

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` 5 sources:

- **Source 1 (fp16/bf16)**: explicit fp32 cast for weight map; MLX renderer in fp32
- **Source 2 (softmax epsilon)**: additive `eps=1e-6` in weight map denominator (canonical scaffold)
- **Source 3 (AdamW state)**: MLX Adam; state preserved across arms
- **Source 4 (bicubic boundary)**: N/A (no resize during loss computation)
- **Source 5 (EMA)**: EMA decay 0.997 applied identically across arms

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility check (preserved from scaffold): **[-1, -4] score points** DISTORTION-axis at PR110 frontier operating point. Mathematical grounding: Fridrich UNIWARD canonical bound + Yousfi grand-council position. Per-axis Pareto polytope alternating projections (Dykstra) project onto JOINT-Fisher-info feasible set. The 5th-order empirical anchor ADJUDICATES whether this prediction transfers from steganalysis canonical to contest-scorer specific.

**ADJUSTED for empirical reality**: this 50-pair 30ep MLX-local smoke produces RELATIVE per-axis improvement signal (variant_d_seg / baseline_d_seg ratio + variant_d_pose / baseline_d_pose ratio), NOT direct contest-CUDA score delta. The ratio<1 verdict on at least one axis (without other-axis regression > 5%) is the PARADIGM-VALIDATED signal.

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (predicted CPU band [0.190, 0.193] given PR110 frontier baseline ~0.192-0.195 + predicted -0.001 to -0.004 improvement).

## Catalog #344 canonical equation target

`uniward_per_pixel_score_conditional_sensitivity_weighting_distortion_savings_v1` — REGISTRATION ON PARADIGM-VALIDATED. The empirical anchor's per-axis improvement ratios become the equation's first EmpiricalAnchor row. If FALSIFIED (no axis improvement), equation NOT registered; sister 6th-order iteration explores alternative-Fisher-info-axis (e.g., per-region wavelet) per Catalog #307 IMPLEMENTATION-LEVEL classification.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — real scorer gradient magnitudes consumed via canonical `tac.master_gradient` discipline
2. **Pareto constraint**: ACTIVE via per-axis improvement ratio measurement (anti-PR97 verification: BOTH axes monitored for trade-off detection)
3. **Bit-allocator hook**: N/A (DISTORTION-attack pure)
4. **Cathedral autopilot dispatch hook**: ACTIVE — substrate inherits scaffold's canonical contract per Catalog #335
5. **Continual-learning posterior update**: ACTIVE — results JSON emitted with canonical Provenance for posterior anchor emission
6. **Probe-disambiguator**: ACTIVE — BASELINE vs VARIANT controlled comparison IS the disambiguator between routing-effect and coincidental-noise

## Operator-routable next step

EXECUTE empirical training script → emit sweep results JSON → write landing memo with per-axis BASELINE vs VARIANT comparison + canonical equation #344 registration verdict.
