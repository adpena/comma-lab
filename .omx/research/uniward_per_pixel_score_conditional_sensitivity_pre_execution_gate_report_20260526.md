# UNIWARD Per-Pixel Score-Conditional Sensitivity Pre-Execution Gate Report 2026-05-26

**Subagent**: `uniward-per-pixel-score-conditional-sensitivity-weighting-fridrich-canonical-pure-distortion-attack-mlx-first-numpy-portable-20260526`
**Lane**: `lane_uniward_per_pixel_score_conditional_sensitivity_distortion_attack_20260526`
**Date**: 2026-05-26
**Sister scope**: Disjoint from in-flight BoostNeRV C-i' (#1356; loss-shape axis ⊥), Cascade B Hinton-KL CATALYST (#1359; CATALYST-composition not PURE-distortion), Catalog-memo revision over-cap (#1357; non-compute apparatus).

## 3-strategy attack decomposition

PRIMARY = DISTORTION pure (Fridrich-canonical UNIWARD inverse-steganalysis adapted to contest scorers per Yousfi grand-council position). SUB-AXIS = JOINT d_seg + d_pose via per-pixel inverse-Fisher-information weighting (`weight[h,w] = 1.0 / (eps + (d_seg_grad[h,w])^2 + (d_pose_grad[h,w])^2)`). Sister strategies UNATTACKED here (RATE attack covered by Cascade C frontier #1351; FULL-SCORER attack via Lagrangian dual covered by Phase 1 meta-Lagrangian wire-in #1059) — this is PURE-DISTORTION coverage per just-elevated 3-strategy directive's strategy-coverage rule (≥2 of 3 strategies in portfolio).

## Entropy-position declaration

POSITION = P2 loss-shape (TRAIN phase) per `feedback_entropy_position_discipline_in_full_stack_pipeline_standing_directive_20260526.md` Lesson 1 (BEFORE entropy coder wins). The per-pixel UNIWARD weight map reshapes the UPSTREAM perturbation distribution that the rendered RGB takes BEFORE the eval_roundtrip uint8 bottleneck → BEFORE archive entropy coder → so the rate-attack downstream codec sees a perturbation distribution that has already been routed away from scorer-sensitive zones. Predicted savings respect the BEFORE-entropy-coder position's structural floor (the conditional entropy `H(perturbation | weight_map_bucket)` is LOWER than the marginal `H(perturbation)` because routing concentrates perturbation in low-sensitivity zones where the scorer-response distribution has lower entropy).

## MLX-first → numpy-portable bridge contract

Per `feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md`:

- **Training**: MLX-native renderer + MLX-native PoseNet/SegNet forward (or PyTorch fallback if MLX scorer port not yet landed); per-pixel weight map computed via `mlx.core.grad` against per-pixel inputs OR via `tac.master_gradient` typed CandidateModificationSpec per Catalog #318 (NOT raw byte authority).
- **Inflate**: numpy-portable ≤200 LOC + ≤2 deps (numpy + pyav); weight map consumed AT TRAINING TIME (compress-only) and NOT shipped to inflate. This is Carmack-preferred budget conservation: the weight map's value is in routing training-time perturbation; the trained weights themselves embody the routing, so inflate runtime is UNCHANGED relative to baseline (PR110 family inflate).
- **Bridge contract**: trainer exports trained MLX state_dict → `np.savez_compressed(...)` → ZIP-member at fixed offset per archive_grammar → inflate loads via `np.load(...)` with NO MLX dep. The weight-map itself is forensic-only (saved as separate sidecar JSON at `experiments/results/<lane>/weight_map_<archive_sha[:8]>.npz` for post-hoc audit) — NOT part of the contest archive.

## Individually-fractal decomposition

Per just-elevated GUIDING PRINCIPLE + 13-ingredient parity discipline:

- **Ingredient #6 (score-domain Lagrangian)** = PRIMARY decomposition node addressed (UNIWARD weighting IS the canonical inverse-Fisher-information score-domain weighting per Fridrich's 2012 SS&P paper)
  - **Sub-ingredient**: per-pixel weight map computation
    - **Sub-sub-ingredient**: Fisher-info inverse from BOTH scorers (SegNet contribution + PoseNet contribution)
      - **Sub-sub-sub-ingredient**: weight-map quantization for sidecar embedding (preferably compress-only; sub-sub-sub-sub-ingredient queued if rate-attack composition test surfaces value)
- **Ingredient #5 (full renderer)** = UNCHANGED (lightly-modified PR110 baseline MLX renderer)
- **Ingredient #8 (eval_roundtrip + diff-yuv6)** = CANONICAL (canonical scorer-preprocess routing per Catalog #164/#226; NO substrate-specific fork needed)
- **Ingredient #9 (runtime closure)** = INHERITS baseline inflate (no new deps per Carmack-preferred budget conservation)

Per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD: substrate's distinguishing feature IS the per-pixel weight map; ALL other ingredients are canonical adoption because they SERVE not SUPPRESS substrate-optimal engineering.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| score_aware_loss (canonical scorer-preprocess routing) | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #164/#226 enforces canonical helper; substrate just EXTENDS via per-pixel weighting (not forks) |
| per-pixel weight map computation | FORK_BECAUSE_PRINCIPLED_MISMATCH (UNIQUE-PER-METHOD per Fridrich-Yousfi-adapter) | Fridrich UNIWARD canonical formula `weight = 1 / (eps + Fisher_info)` is substrate-distinctive; no sister substrate uses it |
| eval_roundtrip | ADOPT_CANONICAL_BECAUSE_SERVES | CLAUDE.md non-negotiable; canonical `apply_eval_roundtrip_during_training` |
| EMA shadow (decay=0.997) | ADOPT_CANONICAL_BECAUSE_SERVES | CLAUDE.md non-negotiable; canonical `tac.training.EMA` |
| archive grammar (ZIP) | ADOPT_CANONICAL_BECAUSE_SERVES | PR110 baseline grammar inherited; weight map NOT shipped |
| inflate.py (numpy-portable) | ADOPT_CANONICAL_BECAUSE_SERVES (inherits baseline) | Carmack-preferred budget conservation; weight-map effect is in trained weights |
| `tac.master_gradient` per-pair typed CandidateModificationSpec | ADOPT_CANONICAL_BECAUSE_SERVES per Catalog #318 | Typed primitive API; NO raw byte authority |
| MLX-first training (Catalog #192/#317) | ADOPT_CANONICAL_BECAUSE_SERVES | "Remember all on MLX" standing directive |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS** (class-shift not within-class): Fridrich UNIWARD canonical formula NOT used by any sister substrate; substrate ID is `uniward_per_pixel_distortion` — distinct from per-instance per-region wavelet variants per audit.
2. **BEAUTY + ELEGANCE**: weight map formula is ONE LINE (`weight = 1 / (eps + d_seg_grad**2 + d_pose_grad**2)`); 30-sec reviewable.
3. **DISTINCTNESS**: explicitly different from sister substrates (BoostNeRV residual-correction class; Cascade B Hinton-KL CATALYST class; this is PURE DISTORTION + canonical-Fridrich-UNIWARD class).
4. **RIGOR**: premise verification (read 4 standing directives + Cascade C P19 methodology + `tac.master_gradient` canonical helper API); adversarial-review-ready; cargo-cult audit per section below.
5. **OPTIMIZATION PER TECHNIQUE**: substrate-optimal MLX gradient backprop through BOTH scorers for per-pixel sensitivity (not approximate); explicit eps for numerical stability.
6. **STACK-OF-STACKS COMPOSABILITY**: orthogonal to Cascade C rate-attack (DIFFERENT entropy-position: P2 loss-shape vs P5 archive entropy coder); composable with PR110 substrate-engineering frontier.
7. **DETERMINISTIC REPRODUCIBILITY**: MLX seed pinned per `tac.local_acceleration.mlx_seed_pin`; np.savez_compressed produces byte-stable export.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: MLX-local smoke <3h on M5 Max; no paid GPU.
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted ΔS [-1 to -4] score points via DISTORTION-axis (d_seg + d_pose joint improvement); empirical anchor will validate post-smoke.

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Classification | Unwind path |
|---|---|---|
| "Per-pixel weight maps improve scorer-attack discriminability" | HARD-EARNED per Fridrich 2012 SS&P + 13+ years steganalysis empirical anchors | N/A |
| "UNIWARD formula extends from steganalysis to contest scorers cleanly" | CARGO-CULTED (steganography uses STATIC content-adaptive cost; contest scorers are LEARNED and use gradient-reachable forward) | Validate via empirical smoke: measure if per-pixel Fisher-info inverse weighting actually reduces per-pixel scorer sensitivity vs uniform-weighting baseline |
| "JOINT d_seg + d_pose Fisher-info inverse beats single-axis" | CARGO-CULTED (joint Fisher-info matrix may have cross-axis cancellation that obscures per-axis sensitivity) | Validate via per-axis decomposition: log per-pixel `d_seg_only_weight` + `d_pose_only_weight` + `joint_weight` and compare empirical score reduction |
| "Compress-time weighting transfers to trained weights without shipping weight map" | HARD-EARNED per canonical training discipline (score-aware loss shapes the gradient → shapes the trained weights) | N/A but verify via paired compress-only-vs-sidecar-shipped smoke |
| "MLX-native per-pixel gradient backprop is computationally feasible at 384x512" | UNCLEAR — empirical test required (M5 Max memory + MLX softmax kernel performance) | First smoke runs at 96x128 fixture; scale up if successful |

## Observability surface

Per Catalog #305 6-facet definition:

1. **Inspectable per layer**: weight map saved as np.savez_compressed sidecar at `experiments/results/<lane>/weight_map_<archive_sha[:8]>.npz`; per-pixel histogram emitted in training logs; trainer dumps per-epoch weight map for visual inspection.
2. **Decomposable per signal**: per-axis weight components (d_seg-only + d_pose-only + joint) logged separately; final score decomposed via canonical Provenance per Catalog #323.
3. **Diff-able across runs**: weight maps from sister UNIWARD variants comparable via canonical np.savez format; per-pixel diff visualizable.
4. **Queryable post-hoc**: weight map sidecar + training metrics JSON + canonical posterior anchor in `.omx/state/continual_learning_posterior.jsonl`.
5. **Cite-able**: canonical Provenance per Catalog #323 (archive sha + commit + call_id + config + random_seed).
6. **Counterfactual-able**: per-pixel byte-mutation smoke per Catalog #139 verifies weight-map application affects rendered RGB at the expected pixel locations.

## Drift surface declaration

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`:

- **Source 1 (Bfloat16/fp16 precision)**: MLX defaults fp32; per-pixel gradient backprop kept at fp32; numpy export at fp32. RISK: PR110 baseline may use fp16; mitigate via explicit fp32 cast for weight-map computation.
- **Source 2 (softmax / LSE epsilon)**: per-pixel weight formula uses additive `eps=1e-6` in denominator (`1 / (eps + grad**2)`); MLX `mlx.core.softmax` with `axis=-1` for SegNet outputs uses default epsilon. RISK: low; weight formula has explicit eps.
- **Source 3 (AdamW β state)**: MLX AdamW β₁=0.9 / β₂=0.999 per Z6 anchor; m/v buffers preserved across MLX↔PyTorch boundary per Catalog #1309.
- **Source 4 (F.interpolate bicubic)**: PR110 baseline uses bicubic resize; not bit-identical CPU↔CUDA per PR #110 disclosure. RISK: weight-map computation happens at training resolution (NOT bicubic-resize boundary); inflate-time bicubic UNCHANGED from baseline.
- **Source 5 (EMA Kahan precision)**: EMA shadow at fp32 + Kahan compensated summation per #1307 canonical fix.

Bidirectional anticipation: when porting MLX-trained substrate → PyTorch/CUDA for paired auth-eval, EMA shadow MUST be exported at fp32; AdamW state buffers MUST be preserved exactly; bicubic-resize-related ops MUST have paired CPU+CUDA inflate parity test.

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility check:

Predicted DISTORTION-axis improvement: **[-1, -4] score points** total on combined (d_seg + d_pose) axis at PR110 frontier operating point. Mathematical grounding: Fridrich UNIWARD canonical bound + Yousfi grand-council position that per-pixel inverse-steganalysis weighting reduces detector sensitivity by 2-4× empirically for steganalysis (translating to ~2-4× per-pixel d_seg reduction in low-sensitivity zones; per-pixel d_pose reduction smaller due to sqrt non-linearity per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"). Dykstra-feasibility check: per-pixel weighting projects onto JOINT-Fisher-info feasible set; the per-axis Pareto polytope alternating projections converge to a feasible point WHERE both axes simultaneously improve (vs PR97 anti-pattern where seg-for-pose trade was destructive).

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (predicted CPU band [0.190, 0.193] given PR110 frontier baseline ~0.192-0.195 and predicted -0.001 to -0.004 improvement). Per CLAUDE.md "HORIZON-CLASS Consequence 4": plateau-adjacent substrates DO advance the frontier when paired with sister rate-attack ASYMPTOTIC pursuits (Cascade C #1351); this work is the DISTORTION-pure-axis sister to Cascade C's rate-pure-axis effort.

## Catalog #344 canonical equation target

Proposed NEW: `uniward_per_pixel_score_conditional_sensitivity_weighting_distortion_savings_v1` (Fridrich-canonical with our specific scorer-adapter per Yousfi grand-council position; pending empirical validation + sister Task #1357 catalog-memo revision OR direct `tac.canonical_equations.append_equation` registration). Domain of validity: P2 loss-shape entropy-position; substrate class = `score_aware_renderer_with_per_pixel_uniward_weighting`; predicted savings band [-1, -4] DISTORTION-axis score points.

## Pre-execution gate verdict

**PROCEED** with substrate scaffold per all 4 directive frameworks: 3-strategy (DISTORTION-pure) + entropy-position (P2 loss-shape BEFORE entropy coder) + MLX-first-numpy-portable bridge contract (compress-only weighting; inflate UNCHANGED) + individually-fractal (per-substrate full GUIDING PRINCIPLE decomposition).
