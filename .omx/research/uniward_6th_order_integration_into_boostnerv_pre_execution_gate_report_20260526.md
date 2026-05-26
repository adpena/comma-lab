# UNIWARD 6th-Order Integration into BoostNeRV-PR110-Residual Pre-Execution Gate Report 2026-05-26

**Subagent**: `uniward-6th-order-integration-into-boostnerv-pr110-residual-capacity-constrained-substrate-recursive-doctrine-mlx-first-numpy-portable-20260526`
**Lane**: `lane_uniward_6th_order_integration_into_boostnerv_pr110_residual_20260526`
**Successor of**: N+1 verdict `3316721639` (subagent `a30ea26630c15c98b`; PARADIGM-NULL-NO-EFFECT + DIAGNOSED capacity-bottleneck-mechanism)
**Sister-of**: scaffold `aa2612d9b` (UNIWARD substrate L1)
**Date**: 2026-05-26
**Tag**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority"

## The 6th-order diagnosed mechanism (from N+1 verdict)

UNIWARD per-pixel weight map IS VALID (19.71x dynamic range; real-scorer gradients non-degenerate). BUT N+1 test architecture (free per-pair RGB tensors with simple L2 loss) has NO parameter bottleneck — every pixel reconstructs independently → UNIWARD reweighting has nothing to redirect against. PARADIGM Fridrich UNIWARD inverse-steganalysis INTACT per 14+ years steganalysis literature; THIS test architecture was the bottleneck.

## The 6th-order fix

Integrate `compose_uniward_weighted_score_loss` INTO **BoostNeRV-PR110-residual capacity-constrained substrate**'s loss path. BoostNeRV BPR1 has a parameter-constrained ResidualHeadMLX (~3K params/round for hidden_dim=12); UNIWARD per-pixel routing should have structural traction at this bottleneck.

## Sister-disjoint discipline per Catalog #230

- UNIWARD scope (THIS lane): NEW integration module `src/tac/substrates/uniward_per_pixel_distortion/boostnerv_integration.py` + sister test file
- BoostNeRV scope (READ-ONLY consumer): import-only via `from tac.substrates.boost_nerv_pr110_residual import ...`; NO modifications to BoostNeRV substrate; STAY OUT of training/test paths

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Scorer-preprocess routing | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.substrates.score_aware_common.score_pair_components` canonical for both scaffold + N+1 + this integration |
| BoostNeRV architecture (READ-ONLY) | ADOPT_CANONICAL_BECAUSE_SERVES | BoostNeRV substrate IS the capacity-constrained substrate to test against |
| UNIWARD weight-map routing | FORK_BECAUSE_PRINCIPLED_MISMATCH | Unique-per-method per Fridrich adapter; not a candidate for canonical helper extraction |
| Loss composition factory | FORK_BECAUSE_PRINCIPLED_MISMATCH | Specific to BoostNeRV-PR110 composition (rgb_pr110_base + residual); not generalizable as canonical helper yet |
| Per-axis Provenance | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #323 canonical Provenance + Catalog #341 routing markers |

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence |
|---|---|
| UNIQUENESS | UNIWARD per-pixel routing layered into BoostNeRV BPR1 trained-loss-path is novel composition; no sister lane has this integration |
| BEAUTY+ELEGANCE | Factory pattern: `compose_uniward_into_boostnerv_loss(boostnerv_loss_fn, weight_map, *, lambda_uniward) -> Callable`; ≤200 LOC integration module |
| DISTINCTNESS | Distinct from scaffold (compute weight map; no integration) + N+1 (free-RGB-tensor test; no substrate) + BoostNeRV (uniform per-pixel weighting; canonical baseline) |
| RIGOR | Real-scorer-anchored cached gradients from N+1; deterministic seed; BASELINE vs VARIANT controlled comparison at capacity-constrained substrate |
| OPTIMIZATION-PER-TECHNIQUE | UNIWARD weight-map per Fridrich-canonical Holub-Fridrich-Denemark 2014; BoostNeRV substrate-engineering frozen-base+residual per HNeRV parity discipline |
| STACK-OF-STACKS-COMPOSABILITY | UNIWARD per-pixel routing IS the unique optimization layered on BoostNeRV's canonical substrate-engineering foundation; sister composition to Cascade B's stacking discipline |
| DETERMINISTIC-REPRODUCIBILITY | Seed-pinned; cached real-scorer gradients from N+1 `real_scorer_gradients_cache.npz`; MLX-deterministic |
| EXTREME-OPTIMIZATION-PERFORMANCE | Bounded UNIWARD coefficient λ=0.01 SAFE band; ResidualHeadMLX capacity-constrained ~3K params/round |
| OPTIMAL-MINIMAL-CONTEST-SCORE | If 6th-order PARADIGM-VALIDATED at capacity-constrained substrate: canonical equation #344 first anchor REGISTERED; if NULL again: 7th-order iteration |

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Empirical adjudication plan |
|---|---|---|
| "UNIWARD routing requires capacity bottleneck for structural traction (N+1 mechanism diagnosis)" | HARD-EARNED-FROM-N+1-DIAGNOSIS | Adjudicated by THIS sweep VARIANT-vs-BASELINE on capacity-constrained substrate |
| "BoostNeRV ResidualHeadMLX is sufficiently capacity-constrained for UNIWARD routing to surface" | CARGO-CULTED-PENDING-VALIDATION | ~3K params/round on (96,128) residual grid = ~3K params per 12,288 pixels = ~0.24 params/pixel; substantially bottlenecked vs free-RGB N+1 ~36,864 params/pair |
| "Per-pixel weight map applied to per-pixel residual error reweights the loss landscape" | CARGO-CULTED-PENDING-VALIDATION | Adjudicated empirically; if effect surfaces, weighting succeeds |
| "30ep / 50pair / 96x128 fixture sufficient (sister N+1 parameters)" | HARD-EARNED-FROM-N+1 | Reuse N+1's exact fixture for apples-to-apples comparison |
| "Real scorer gradients cached from N+1 are valid for BoostNeRV substrate (not re-computed)" | HARD-EARNED-FROM-N+1-ANCHOR | Cached gradients depend on GT frames only (not on substrate); reusable across substrates |

## Observability surface (per Catalog #305)

All 6 facets HONORED:
- **Inspectable per layer**: BoostNeRV residual output + UNIWARD-weighted loss + per-axis d_seg + d_pose all emitted per epoch
- **Decomposable per signal**: Per-axis d_seg + d_pose + BPR1 sidecar bytes + total contest-partial all reported
- **Diff-able across runs**: Deterministic seed; cached gradients; JSON results sister to N+1 schema
- **Queryable post-hoc**: Per-axis comparison + verdict + provenance in canonical JSON
- **Cite-able**: Canonical Provenance per Catalog #323 in sweep_results.json
- **Counterfactual-able**: BASELINE (BoostNeRV uniform per-pixel) vs VARIANT (BoostNeRV + UNIWARD) IS the counterfactual

## Predicted ΔKL or Δd_seg/Δd_pose band (per Catalog #296)

Per Dykstra-feasibility intersection on capacity-constrained substrate:
- **Predicted band per-axis ratio**: d_seg + d_pose VARIANT/BASELINE ∈ [0.98, 1.00] (small but non-null effect; bounded by λ=0.01 SAFE band + capacity-constrained substrate enables routing to traffic some signal redirected against)
- **First-principles**: UNIWARD weight-map has 19.71x dynamic range; BoostNeRV ResidualHeadMLX has ~0.24 params/pixel bottleneck; effect magnitude bounded by min(dynamic_range_log, capacity_bottleneck_log)
- **Mathematical grounding**: Cramer-Rao bound on per-pixel Fisher information × residual capacity per pixel = expected sensitivity-axis traction
- **Sister probe-disambiguator**: BASELINE vs VARIANT at same capacity bottleneck IS the disambiguator; if VARIANT/BASELINE = 1.0 (full NULL): 7th-order iteration

## Horizon-class declaration (per Catalog #309)

**plateau-adjacent** (sister to NSCS06 v8 PR111 candidacy chain via Cascade B Path A; the substrate-stacking discipline targets the 0.193 cluster floor; UNIWARD-into-substrate is the unique optimization layered on canonical substrate-engineering).

## Entropy-position declaration

**P2 loss-shape** (TRAIN phase BEFORE entropy coder; sister of scaffold + N+1). The UNIWARD weight map is compress-only training-time signal; not shipped to inflate per Carmack-preferred budget conservation (HNeRV parity L4).

## MLX-first → numpy-portable bridge contract

Honored:
- **TRAINING**: MLX-native (MLX Adam; eval_roundtrip via mx.round straight-through estimator) — sister of N+1
- **INFLATE**: numpy-portable per HNeRV parity L4 (BoostNeRV substrate already honors)
- **BRIDGE**: UNIWARD weight map consumed at TRAINING (compress) time only; NOT shipped to archive bytes; BoostNeRV inflate path UNCHANGED

## Drift surface declaration (per MLX↔CUDA bidirectional directive)

5 sources:
1. **fp16/bf16**: weight computation explicitly fp32-cast (inherited from scaffold weight_map.py)
2. **softmax epsilon**: additive eps=1e-6 in denominator (sister of scaffold)
3. **AdamW state**: N/A at UNIWARD weight computation surface
4. **bicubic**: N/A at UNIWARD weight computation surface
5. **EMA**: ema_decay=0.997 in trainer loop per CLAUDE.md EMA non-negotiable

## Individually-fractal decomposition

- **1st order**: scaffold (commit `aa2612d9b`; PROCEED_PENDING_EMPIRICAL)
- **2nd order**: per-pixel weight map computation (Fisher-info inverse from BOTH scorers)
- **3rd order**: real-scorer gradient via `tac.master_gradient` per Catalog #318
- **4th order**: histogram observability per Catalog #305
- **5th order**: real-scorer-anchored empirical validation on free-RGB-tensor fixture (N+1; PARADIGM-NULL-NO-EFFECT; DIAGNOSED architecture-mismatch)
- **6th order (THIS lane)**: integrate UNIWARD compose_uniward_weighted_score_loss into BoostNeRV-PR110-residual capacity-constrained substrate's loss path

## Discipline anchors

- Catalog #229 PV (read scaffold + N+1 + BoostNeRV BPR1 + sweep_results.json + harness BEFORE implementation)
- Catalog #206 (checkpoints emitted every ~10 tool uses)
- Catalog #110/#113 APPEND-ONLY (NEW artifacts only; sisters never mutated)
- Catalog #117/#157/#174/#235/#289 canonical commit serializer (with --expected-content-sha256 + co-author trailer)
- Catalog #230 ownership map (BoostNeRV substrate READ-ONLY consumer scope; sister scope-disjoint)
- Catalog #287 placeholder rejection (rationales ≥4 chars substantive)
- Catalog #290/#294/#296/#303/#305/#309 design-memo discipline (all sections present)
- Catalog #307 paradigm-vs-implementation classification (will adjudicate on empirical verdict)
- Catalog #318 master-gradient via typed primitive (real-scorer gradients cached from N+1)
- Catalog #323 canonical Provenance in sweep_results.json
- Catalog #335 cathedral consumer canonical contract (inherited from scaffold substrate)
- Catalog #340 sister-checkpoint guard before edits
- Catalog #341 canonical-routing markers (score_claim=False + promotable=False + axis_tag=[predicted])
- Catalog #343 no hardcoded score literals
- Catalog #344 canonical equation REGISTRATION GATED on PARADIGM-VALIDATED verdict
- CLAUDE.md "MLX portable-local-substrate authority" — tagged [macOS-MLX research-signal]; NO paid dispatch
- CLAUDE.md "Forbidden premature KILL" — DEFERRED-pending-7th-order-iteration if NULL again
- CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — UNIWARD per-pixel routing is the unique-per-method optimization layered on BoostNeRV's canonical substrate-engineering foundation
