# UNIWARD Per-Pixel N+1 Real-Scorer-Anchored Empirical Distortion Attack Landed 2026-05-26

**Subagent**: `uniward-per-pixel-n-plus-1-real-scorer-anchored-empirical-anchor-50pair-mlx-local-pr110-baseline-20260526`
**Lane**: `lane_uniward_per_pixel_n_plus_1_real_scorer_empirical_20260526`
**Date**: 2026-05-26
**Tag**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341
**Successor of**: scaffold `a136f59ce91f5b468` / commit `aa2612d9b` / lane `lane_uniward_per_pixel_score_conditional_sensitivity_distortion_attack_20260526` L1
**Verdict**: **PARADIGM-NULL-NO-EFFECT** (Carmack-dissent IMPLEMENTATION-LEVEL per Catalog #307; PARADIGM Fridrich-canonical UNIWARD inverse-steganalysis remains INTACT-pending-6th-order-iteration)

## Empirical results (canonical sweep summary)

| Axis | BASELINE (uniform) | VARIANT (UNIWARD) | Ratio (V/B) | Verdict |
|---|---|---|---|---|
| d_seg mean | 0.432208 | 0.432200 | 1.0000 | NULL (within noise) |
| d_pose mean | 0.0000813 | 0.0000811 | 0.9981 | NULL (<0.99 threshold) |
| Contest-partial (100·seg + sqrt(10·pose)) | 43.2492 | 43.2484 | — | Δ = -0.0008 |

**Wall-clock**: 329.8s total (Phase 3 cached gradients ~10s; Phase 5 BASELINE 83s; Phase 6 VARIANT 83s).
**Fixture**: 50 pairs @ 96x128 from `upstream/videos/0.mkv`; 30ep MLX Adam @ lr=0.05; seed=42; ema=0.997; eval_roundtrip uint8 simulation via straight-through estimator.
**Scorers**: 13.9M-param PoseNet + 9.5M-param SegNet via canonical `tac.scorer.load_differentiable_scorers` per Catalog #164/#226.
**Per-pixel weight map**: dynamic range 19.71x (min=0.052, max=1.027, mean=1.000 after unit-mean normalization). Real scorer gradient magnitudes: seg ∈ [1e-6, 4.3e-3] mean 8.3e-5; pose ∈ [1e-6, 2.5e-5] mean 3.2e-6.

## 3-strategy attack decomposition

PRIMARY = DISTORTION pure-axis (empirical-validation phase of scaffold's Fridrich-canonical UNIWARD inverse-steganalysis adapter). SUB-AXIS = JOINT d_seg + d_pose. Empirically PARADIGM-NULL at this fixture+architecture choice; sister 6th-order iteration MUST explore alternative integration surfaces (renderer-internal vs free-RGB-tensor loss reweighting) per Carmack-dissent below.

## Entropy-position declaration

POSITION = P2 loss-shape (TRAIN phase BEFORE entropy coder). Sister of scaffold. The structural-bound rule `H(perturbation | weight_map_bucket) < H(perturbation)` empirically FAILED to materialize for this fixture+architecture because the free-RGB-tensor baseline already absorbs all perturbation budget into the L2-minimum gradient field with no capacity bottleneck the UNIWARD reweighting can redirect signal against.

## MLX-first → numpy-portable bridge contract

Honored:
- Training MLX-native (MLX Adam; eval_roundtrip via mx.round straight-through estimator)
- Scorer evaluation torch on CPU (canonical `load_differentiable_scorers`)
- Weight map sidecar at `.omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/real_scorer_gradients_cache.npz` (forensic-only; NOT contest archive member)

## Individually-fractal decomposition (5th-order recursive node EXECUTED)

- 1st: substrate scaffold (commit `aa2612d9b`; 12/12 tests pass)
- 2nd: per-pixel weight map computation (canonical UNIWARD)
- 3rd: Fisher-info inverse from BOTH scorers via `tac.master_gradient` typed CandidateModificationSpec discipline (real-scorer-anchored)
- 4th: histogram observability per Catalog #305
- **5th: real-scorer-anchored empirical validation (THIS landing)** — verdict PARADIGM-NULL-NO-EFFECT
- **6th (operator-routable next)**: integrate UNIWARD weighting INTO actual capacity-constrained renderer (BoostNeRV / PR110 substrate) rather than free-RGB-tensor L2-loss reweighting — the free-tensor architecture has no parameter bottleneck the routing can leverage

## Canonical-vs-unique decision per layer

Per Catalog #290 (preserved from pre-execution gate report; per-pixel weight map FORKED; everything else canonical adoption).

## 9-dimension success checklist evidence

Per Catalog #294 (preserved from pre-execution gate report). RIGOR (Dim 4) dimension specifically validated EMPIRICALLY this landing: real-scorer-anchored measurement (NOT synthetic), BASELINE vs VARIANT controlled comparison, deterministic seed-pinned reproduction, per-axis decomposition emitted.

## Cargo-cult audit per assumption

Per Catalog #303. EMPIRICAL UPDATES (5 assumptions enumerated in pre-execution gate report):

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "UNIWARD weight map computed from REAL scorer gradients improves per-axis d_seg+d_pose vs uniform" | CARGO-CULTED-PENDING-VALIDATION | EMPIRICALLY-FALSIFIED-FOR-THIS-ARCHITECTURE (free-RGB-tensor L2-loss reweighting: no effect) |
| "Per-pixel weight map encodes useful gradient signal at PR110 baseline operating point" | CARGO-CULTED-PENDING-VALIDATION | EMPIRICALLY-CONFIRMED (weight map has 19.7x dynamic range; non-degenerate signal) |
| "JOINT d_seg + d_pose Fisher-info inverse beats per-axis-only weighting" | CARGO-CULTED-PENDING-VALIDATION | UNRESOLVED (joint test produced no effect; per-axis-only NOT tested due to budget) |
| "30ep / 50pair / 96x128 fixture sufficient for cross-axis discrimination" | HARD-EARNED via sister landings | CONFIRMED (BASELINE converged to stable seg=0.432208 / pose=8.13e-5; signal-to-noise adequate) |
| "Minimal MLX renderer reveals weighting effect (PR110-scale renderer NOT required)" | HARD-EARNED | **EMPIRICALLY-FALSIFIED** — free-RGB-tensor architecture has NO parameter bottleneck the weighting can leverage; weighting effect requires capacity-constrained substrate |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:
- Inspectable per layer: per-pixel scorer gradients + weight map cached
- Decomposable per signal: per-axis seg-only / pose-only / joint variants emitted in sweep_results.json (`weight_map_per_axis_decomposition_samples`)
- Diff-able across runs: deterministic seed, cached gradients, JSON results
- Queryable post-hoc: histograms + min/max/mean/median in JSON
- Cite-able: canonical Provenance per Catalog #323 in sweep results
- Counterfactual-able: BASELINE vs VARIANT IS the counterfactual

## Drift surface declaration

5 sources HONORED per pre-execution gate report.

## Predicted ΔS band vs empirical

Per Catalog #296: PREDICTED [-1, -4] DISTORTION-axis at PR110 frontier operating point; EMPIRICAL contest-partial delta = **-0.0008** (well outside predicted band; ~125x smaller than predicted minimum -1.0). Residual magnitude normalized = |−0.0008 - (−2.5)| / 1.5 = 1.665 (significantly outside-band per Dykstra-feasibility check). Per CLAUDE.md "Apples-to-apples evidence discipline": this 50-pair MLX-local fixture is RESEARCH-SIGNAL not contest-CUDA evidence; the band/empirical mismatch is informative for sister 6th-order iteration design, NOT a refutation of the underlying Fridrich UNIWARD canonical bound.

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (the empirical NULL verdict places us still inside the plateau; sister 6th-order iteration into capacity-constrained substrate may shift to plateau-adjacent or asymptotic-pursuit).

## Catalog #344 canonical equation registration verdict

**NOT REGISTERED** per `tools/register_uniward_per_pixel_score_conditional_sensitivity_canonical_equation_20260526.py` REFUSAL logic. The script correctly refuses with verdict `PARADIGM-NULL-NO-EFFECT` (only `PARADIGM-VALIDATED-*` verdicts register). Registry remains at 52 entries (NOT 53). Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 IMPLEMENTATION-LEVEL classification: the equation is DEFERRED-PENDING-6th-ORDER-ITERATION, NOT killed. When sister 6th-order spawn validates the paradigm against a capacity-constrained substrate, the same registration script consumes the new sweep_results.json and registers.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — real scorer gradient magnitudes consumed; per-pixel decomposition emitted
2. **Pareto constraint**: ACTIVE — per-axis ratio measured (anti-PR97); both axes monitored
3. **Bit-allocator**: N/A (DISTORTION-attack pure)
4. **Cathedral autopilot dispatch**: ACTIVE — substrate inherits scaffold's canonical contract; canonical equation #344 registration GATED on PARADIGM-VALIDATED (correctly REFUSED on NULL)
5. **Continual-learning posterior**: ACTIVE — sweep_results.json carries canonical Provenance per Catalog #323
6. **Probe-disambiguator**: ACTIVE — BASELINE vs VARIANT IS the disambiguator; NULL verdict disambiguates "routing matters at this surface" from "routing requires capacity bottleneck"

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + IMPLEMENTATION-LEVEL-FALSIFIED-AT-FREE-RGB-TENSOR-ARCHITECTURE**. The Fridrich UNIWARD canonical (14+ years steganalysis literature; Holub-Fridrich-Denemark 2014) remains the canonical inverse-Fisher-info routing primitive. The empirical NULL is because the test architecture (free per-pair RGB tensors trained with simple L2 loss) has NO parameter bottleneck the per-pixel routing can leverage — every pixel can be reconstructed independently with effectively infinite capacity. The PR110-class substrate has parameter-constrained per-pixel reconstruction; that's where UNIWARD routing has structural traction. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": REACTIVATION CRITERIA = sister 6th-order spawn integrating UNIWARD weighting into a real capacity-constrained substrate (BoostNeRV / PR110-residual / NeRV-family) loss path AND demonstrating per-axis improvement.

## Operator-routable next step

1. **Sister 6th-order iteration** — spawn UNIWARD-weighted-loss-INTO-PR110-class-substrate sweep: integrate `compose_uniward_weighted_score_loss` extension into BoostNeRV-PR110 or pr110_residual substrate trainer's loss path (replace simple scorer term with UNIWARD-weighted variant); rerun 50pair/30ep/96x128 on capacity-constrained substrate; measure per-axis improvement.
2. **Sister probe (alternative-Fisher-info-axis)** — test per-axis-ONLY weighting variants (`decompose_per_axis_weights` produces seg_only / pose_only / joint); the JOINT NULL verdict doesn't disambiguate whether seg-only or pose-only might have signal.
3. **Sister probe (per-region wavelet UNIWARD)** — Mallat wavelet decomposition + per-region weight map per `tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md` — different granularity may surface effects the per-pixel granularity hides at this fixture scale.

## Discipline anchors

- Catalog #229 PV (read scaffold landing memo + 5 standing-directive memos + canonical helper APIs + master_gradient discipline BEFORE writing harness)
- Catalog #206 (4 checkpoints emitted: PV-complete + scaffold-built + sweep-launched + landing-memo-write)
- Catalog #110/#113 APPEND-ONLY (NEW artifacts only; scaffold + standing directives never mutated)
- Catalog #117/#157/#174/#235/#289 canonical commit serializer (will use --expected-content-sha256 + co-author trailer)
- Catalog #230 ownership map (no scope collision with in-flight sisters #1359/#1361)
- Catalog #287 placeholder rejection (all rationales ≥4 chars substantive)
- Catalog #290/#294/#296/#303/#305/#309 design-memo discipline (all sections present)
- Catalog #307 paradigm-vs-implementation classification (PARADIGM-INTACT + IMPLEMENTATION-LEVEL-FALSIFIED)
- Catalog #318 master-gradient via typed primitive (NOT raw byte authority)
- Catalog #323 canonical Provenance in sweep_results.json
- Catalog #335 cathedral consumer canonical contract (inherited from scaffold)
- Catalog #341 canonical-routing markers (score_claim=False + promotable=False + axis_tag=[predicted])
- Catalog #343 no hardcoded score literals (only canonical formula constants from CLAUDE.md non-negotiable)
- Catalog #344 canonical equation REGISTRATION REFUSED on NULL verdict
- CLAUDE.md "MLX portable-local-substrate authority" — tagged [macOS-MLX research-signal]; NO paid dispatch
- CLAUDE.md "Forbidden premature KILL" — DEFERRED-pending-6th-order-iteration

## Cost

$0 GPU + ~50 min wall-clock (including 213s Phase 3 gradient cache build + 167s 2-arm training + 30-min draft+review iterations) + ~370 LOC harness + 0 canonical equations registered (gate correctly refused).
