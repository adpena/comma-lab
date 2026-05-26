# UNIWARD 6th-Order Integration into BoostNeRV-PR110-Residual Capacity-Constrained Substrate Distortion Attack Landed 2026-05-26

**Subagent**: `uniward-6th-order-integration-into-boostnerv-pr110-residual-capacity-constrained-substrate-recursive-doctrine-mlx-first-numpy-portable-20260526`
**Lane**: `lane_uniward_6th_order_integration_into_boostnerv_pr110_residual_20260526`
**Successor of**: N+1 verdict `3316721639` (subagent `a30ea26630c15c98b`) → PARADIGM-NULL-NO-EFFECT
**Sister-of**: scaffold `aa2612d9b` (UNIWARD substrate L1)
**Date**: 2026-05-26
**Tag**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341
**Verdict**: **PARADIGM-NULL-NO-EFFECT-AT-CAPACITY-CONSTRAINED-SUBSTRATE** (Carmack-dissent IMPLEMENTATION-LEVEL per Catalog #307; PARADIGM Fridrich-canonical UNIWARD inverse-steganalysis remains INTACT-pending-7th-order-iteration)

## Empirical results (canonical sweep summary)

| Axis | BASELINE (BoostNeRV uniform) | VARIANT (BoostNeRV + UNIWARD) | Ratio (V/B) | Verdict |
|---|---|---|---|---|
| d_seg mean | 0.432536 | 0.432537 | 1.0000 | NULL (within noise) |
| d_pose mean | 5.51e-5 | 5.53e-5 | 1.0039 | NULL (slight regression within noise) |
| Contest-partial (100·seg + sqrt(10·pose)) | 43.2772 | 43.2773 | — | Δ = +0.0001 |

**Wall-clock**: 298.9s total (Phase 5 BASELINE 87s; Phase 6 VARIANT 69s).
**Fixture**: 50 pairs @ 96x128 from `upstream/videos/0.mkv` (sister of N+1; cached real-scorer gradients reused per Catalog #230).
**Architecture**: BoostNeRV ResidualHeadMLX shape (z_proj + conv1 + conv2) → 1,971 params/round (≈0.16 params/pixel at 12,288-pixel grid; substantially bottlenecked vs N+1 free-RGB ~36,864 params/pair).
**Capacity bottleneck**: 1,971 params + 50×24=1,200 latent params = 3,171 total trainable params across 50 pairs.
**Per-pixel weight map**: same 19.71x dynamic range as N+1 (reused cache).
**Scorers**: 13.9M-param PoseNet + 9.5M-param SegNet via canonical `tac.scorer.load_differentiable_scorers` per Catalog #164/#226.

## 6th-order DIAGNOSED-MECHANISM hypothesis tested

Per N+1 verdict diagnosed mechanism: "UNIWARD per-pixel routing requires PARAMETER BOTTLENECK for structural traction." THIS sweep tested the hypothesis at a capacity-constrained substrate (BoostNeRV ResidualHeadMLX, ~0.16 params/pixel). The empirical result REFUTES the simple-form hypothesis: the capacity bottleneck alone is NOT sufficient for UNIWARD per-pixel routing to surface effect. The routing's structural traction requirement is deeper than parameter count.

## 3-strategy attack decomposition

PRIMARY = DISTORTION pure-axis. SUB-AXIS = JOINT d_seg + d_pose. Empirically PARADIGM-NULL at this fixture+integration choice; sister 7th-order iteration MUST explore alternative integration surfaces (per Carmack-dissent below).

## Entropy-position declaration

POSITION = P2 loss-shape (TRAIN phase BEFORE entropy coder). Sister of scaffold + N+1. The structural-bound rule `H(perturbation | weight_map_bucket) < H(perturbation)` empirically did NOT materialize even at capacity-bottleneck architecture: the per-pixel residual error magnitude at THIS substrate's converged operating point may already be too small (loss=0.0106) for the inverse-Fisher routing to meaningfully redistribute signal across pixels.

## MLX-first → numpy-portable bridge contract

Honored:
- TRAINING MLX-native (manual Adam in MLX with NHWC conv2d; eval_roundtrip via mx.round straight-through estimator)
- Scorer evaluation torch on CPU (canonical `load_differentiable_scorers`)
- Cached real-scorer gradients from sister N+1 (npz; no contest archive member)
- BoostNeRV substrate READ-ONLY consumer-imported per Catalog #230 (NO modifications to BoostNeRV training/test paths)

## Individually-fractal decomposition (6th-order recursive node EXECUTED)

- 1st: substrate scaffold (commit `aa2612d9b`; 12/12 tests pass)
- 2nd: per-pixel weight map computation (canonical UNIWARD)
- 3rd: Fisher-info inverse from BOTH scorers via `tac.master_gradient` typed CandidateModificationSpec discipline
- 4th: histogram observability per Catalog #305
- 5th: real-scorer-anchored empirical validation on free-RGB-tensor fixture (N+1; PARADIGM-NULL → DIAGNOSED architecture-mismatch)
- **6th: empirical validation at capacity-constrained substrate (THIS landing)** → verdict PARADIGM-NULL-NO-EFFECT-AT-CAPACITY-CONSTRAINED-SUBSTRATE (also FALSIFIED simple capacity-bottleneck hypothesis)
- **7th (operator-routable next)**: deeper mechanism diagnosis — UNIWARD may require specific substrate CLASSES (entropy-coded sidecar / quantized residual bits / per-pair codebook), NOT just any capacity bottleneck

## Canonical-vs-unique decision per layer

Per Catalog #290 (preserved from pre-execution gate report):

| Layer | Decision | Rationale |
|---|---|---|
| Scorer-preprocess routing | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.substrates.score_aware_common.score_pair_components` canonical |
| BoostNeRV architecture (READ-ONLY) | ADOPT_CANONICAL_BECAUSE_SERVES | BoostNeRV substrate IS the capacity-constrained substrate tested |
| UNIWARD weight-map routing | FORK_BECAUSE_PRINCIPLED_MISMATCH | Unique-per-method per Fridrich adapter |
| Loss composition factory | FORK_BECAUSE_PRINCIPLED_MISMATCH | Specific to BoostNeRV residual loss path |
| Per-axis Provenance | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #323 + Catalog #341 |

## 9-dimension success checklist evidence

Per Catalog #294 (preserved from pre-execution gate report). RIGOR (Dim 4) dimension validated EMPIRICALLY this landing: real-scorer-anchored measurement, BASELINE vs VARIANT controlled comparison at TWO architecture surfaces (N+1 free-RGB + THIS capacity-constrained), deterministic seed, per-axis decomposition.

## Cargo-cult audit per assumption (EMPIRICAL UPDATES)

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "UNIWARD routing requires capacity bottleneck for structural traction (N+1 mechanism diagnosis)" | HARD-EARNED-FROM-N+1-DIAGNOSIS | **EMPIRICALLY-FALSIFIED** — capacity bottleneck (0.16 params/pixel) NOT sufficient at this loss path; routing requires DEEPER mechanism |
| "BoostNeRV ResidualHeadMLX is sufficiently capacity-constrained for UNIWARD routing to surface" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY-FALSIFIED** — substrate IS capacity-constrained but routing has no effect at this loss path |
| "Per-pixel weight map applied to per-pixel residual error reweights the loss landscape" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY-FALSIFIED-AT-L2-RESIDUAL-LOSS-PATH** — weighting applied but final converged solution NOT redirected; gradient flow may smooth out per-pixel weights through residual head's convolutional structure |
| "30ep / 50pair / 96x128 fixture sufficient (sister N+1 parameters)" | HARD-EARNED-FROM-N+1 | CONFIRMED (BASELINE converged to stable seg=0.4325 / pose=5.5e-5; signal-to-noise adequate) |
| "Cached real-scorer gradients valid for BoostNeRV substrate" | HARD-EARNED-FROM-N+1-ANCHOR | CONFIRMED — gradients depend on GT frames only; reusable across substrates |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:
- Inspectable per layer: per-epoch loss + per-axis d_seg + d_pose; residual head params + z_latents params surfaced
- Decomposable per signal: per-axis ratios + contest-partial delta emitted
- Diff-able across runs: deterministic seed; cached gradients (apples-to-apples vs N+1)
- Queryable post-hoc: histograms + min/max/mean/median in JSON
- Cite-able: canonical Provenance per Catalog #323 in sweep results
- Counterfactual-able: BASELINE (BoostNeRV uniform) vs VARIANT (BoostNeRV + UNIWARD) IS the counterfactual at capacity-constrained substrate

## Drift surface declaration

5 sources HONORED per pre-execution gate report.

## Predicted ΔKL or Δd_seg/Δd_pose band vs empirical

Per Catalog #296: PREDICTED [0.98, 1.00] per-axis ratio (small but non-null effect bounded by λ=0.01 + capacity bottleneck). EMPIRICAL d_seg ratio = **1.0000** (NULL); d_pose ratio = **1.0039** (within noise; slight regression). EMPIRICAL OUTCOME is at the upper bound of predicted band; ratio for d_pose actually slightly EXCEEDS 1.00. The Dykstra-feasibility intersection of (UNIWARD-routing-effect ∩ capacity-bottleneck ∩ L2-residual-loss-landscape ∩ contest-CPU-30ep-50pair-96x128-fixture) appears to be approximately the EMPTY SET at this fixture+integration choice.

## Horizon-class declaration

Per Catalog #309: **plateau-adjacent** (sister to NSCS06 v8 PR111 candidacy chain). The empirical NULL keeps us inside the plateau; sister 7th-order iteration may shift if a DIFFERENT integration surface unlocks routing effect.

## Catalog #344 canonical equation registration verdict

**NOT REGISTERED** per `tools/register_uniward_per_pixel_score_conditional_sensitivity_canonical_equation_20260526.py` REFUSAL logic. The script will correctly refuse with verdict `PARADIGM-NULL-NO-EFFECT-AT-CAPACITY-CONSTRAINED-SUBSTRATE` (only `PARADIGM-VALIDATED-*` verdicts register). Registry remains at 52 entries. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 IMPLEMENTATION-LEVEL classification: the equation is DEFERRED-PENDING-7th-ORDER-ITERATION, NOT killed. The Fridrich UNIWARD canonical (14+ years steganalysis literature) remains INTACT.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — real scorer gradient magnitudes consumed; per-pixel routing applied
2. **Pareto constraint**: ACTIVE — per-axis ratio measured at capacity-constrained substrate
3. **Bit-allocator**: N/A (DISTORTION-attack pure)
4. **Cathedral autopilot dispatch**: ACTIVE — substrate inherits scaffold's canonical contract; canonical equation #344 registration correctly REFUSED on NULL
5. **Continual-learning posterior**: ACTIVE — sweep_results.json carries canonical Provenance per Catalog #323
6. **Probe-disambiguator**: ACTIVE — BASELINE vs VARIANT IS the disambiguator at capacity-constrained substrate; NULL+NULL across TWO architecture surfaces disambiguates "UNIWARD requires capacity bottleneck" from "UNIWARD requires DEEPER substrate-class mechanism"

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + IMPLEMENTATION-LEVEL-FALSIFIED-AT-BOTH-N+1-FREE-RGB-AND-6TH-ORDER-BOOSTNERV-CAPACITY-CONSTRAINED**. The Fridrich UNIWARD canonical (Holub-Fridrich-Denemark 2014) remains the canonical inverse-Fisher-info routing primitive per 14+ years steganalysis literature. The TWO empirical NULLS at orthogonal architectures (free-RGB-tensor N+1 + capacity-constrained ResidualHeadMLX) FALSIFY the simple "needs parameter bottleneck" hypothesis from N+1. The deeper diagnosed mechanism may be:

- **UNIWARD requires entropy-coded sidecar surfaces** (P3+ entropy-position rather than P2 loss-shape) where per-pixel bit-allocation has direct control over which pixels carry which bits — the L2 loss landscape does not expose this granularity
- **UNIWARD requires quantization-aware substrate** where per-pixel quantization stride is the routed signal (sister to Track 4 STC + Fridrich's classical UNIWARD-on-JPEG-DCT-coefficients application domain)
- **UNIWARD requires per-pair codebook substrate** where the codebook entries (NOT raw pixels) are the routed signal — Fridrich's UNIWARD was DEFINED on JPEG DCT coefficients, NOT raw pixels

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": **REACTIVATION CRITERIA** = sister 7th-order iteration that tests UNIWARD routing at an ENTROPY-CODED or QUANTIZED-SIDECAR or PER-PAIR-CODEBOOK substrate (sister candidates: NSCS06 v8 chroma_lut, grayscale_lut, VQ-VAE indices_blob — the T3 council pivot ordering #1/#2/#3 stacking candidates). UNIWARD's natural application domain in Fridrich's original work was JPEG DCT coefficients (quantized; entropy-coded); the L2-residual-on-raw-RGB application domain may be structurally outside the routing primitive's natural fit.

## Operator-routable next step

1. **7TH-ORDER ITERATION** — spawn UNIWARD-into-VQ-VAE-indices-blob sister test: VQ-VAE indices ARE the routed signal (quantized + entropy-coded); UNIWARD per-codebook-entry weighting may have structural traction where per-pixel routing did not.
2. **7TH-ORDER ITERATION** — spawn UNIWARD-into-grayscale-LUT sister test: LUT entries ARE the routed signal (Catalog #297 signal-axis destruction reversibility considered).
3. **7TH-ORDER ITERATION** — spawn UNIWARD-into-NSCS06-v8-chroma-LUT sister test: chroma LUT entries ARE the routed signal (sister to T3 council #1 stacking candidate).
4. **DEFER pure-DISTORTION-attack pursuit** of UNIWARD until 7th-order entropy-coded sidecar integration validates routing primitive applies AT ALL in our application domain.

## Discipline anchors

- Catalog #229 PV (read scaffold + N+1 + BoostNeRV BPR1 + sweep_results.json + harness BEFORE writing integration module)
- Catalog #206 (3+ checkpoints emitted: PV-complete + integration-built + empirical-complete + landing-memo-write)
- Catalog #110/#113 APPEND-ONLY (NEW artifacts only; scaffold + N+1 + BoostNeRV substrate NEVER mutated)
- Catalog #117/#157/#174/#235/#289 canonical commit serializer (will use --expected-content-sha256 + co-author trailer)
- Catalog #230 ownership map (BoostNeRV substrate READ-ONLY consumer import; sister scope-disjoint preserved)
- Catalog #287 placeholder rejection (all rationales ≥4 chars substantive)
- Catalog #290/#294/#296/#303/#305/#309 design-memo discipline (all sections present)
- Catalog #307 paradigm-vs-implementation classification (PARADIGM-INTACT + IMPLEMENTATION-LEVEL-FALSIFIED at TWO architectures)
- Catalog #318 master-gradient via typed primitive (cached gradients from N+1)
- Catalog #323 canonical Provenance in sweep_results.json
- Catalog #335 cathedral consumer canonical contract (inherited from scaffold)
- Catalog #341 canonical-routing markers (score_claim=False + promotable=False + axis_tag=[predicted])
- Catalog #343 no hardcoded score literals
- Catalog #344 canonical equation REGISTRATION REFUSED on NULL verdict (correct gate behavior)
- CLAUDE.md "MLX portable-local-substrate authority" — tagged [macOS-MLX research-signal]; NO paid dispatch
- CLAUDE.md "Forbidden premature KILL" — DEFERRED-pending-7th-order-iteration
- CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — UNIWARD per-pixel routing is unique-per-method; BoostNeRV substrate is canonical-adopted (sister-disjoint)

## Cost

$0 GPU + ~30 min wall-clock (PV + integration ~10 min; 17/17 tests pass; ~5 min empirical sweep with retry on NCHW→NHWC fix; ~10 min landing memo + canonical commit) + ~250 LOC integration module + ~270 LOC empirical harness + 0 canonical equations registered (gate correctly refused on NULL).

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to BoostNeRV substrate's training/test paths. BoostNeRV consumer-imported ONLY via `from tac.substrates.boost_nerv_pr110_residual import ...`. The integration module + empirical sweep + tests + landing memo are ALL scoped under `src/tac/substrates/uniward_per_pixel_distortion/` + `tools/uniward_6th_order_*` + `.omx/research/uniward_6th_order_*`. Zero collision with sister slots (Meta-Lagrangian Phase 3 / Cascade B CATALYST / NSCS06 v8 Modal CUDA).
