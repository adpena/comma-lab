# Theoretical floor analyzer v2 refresh — 2026-05-11

**Status**: planning_only. NO score claim. Per CLAUDE.md "Forbidden score
claims" + "Meta-Lagrangian/Pareto solver" non-negotiable.

**Cross-references**:
- `tac.optimization.theoretical_floor_substrate_refresh` (this refresh module)
- `tac.optimization.substrate_composition_matrix` (Deliverable 1)
- `tac.optimization.autopilot_dispatch_ranking` (Deliverable 2)
- `tools.theoretical_floor_solver_v2` (original v2 solver this refresh consumes)
- `feedback_grand_council_fields_medal_theoretical_floor_20260509.md`
- `feedback_hinton_distilled_scorer_saliency_masked_l2_encoders_landed_20260511.md`
- `feedback_nerv_family_expansion_blocknerv_ffnerv_dsnerv_hinerv_tcnerv_landed_20260511.md`
- `feedback_self_compression_family_scpp_hessian_mdl_landed_20260511.md`
- `feedback_anr_token_renderer_categorical_full_substrate_landed_20260511.md`
- `feedback_pose_axis_lanes_full_scaffolds_landed_20260511.md`
- `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
- `feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md`
- `feedback_magic_codec_auto_selector_landed_20260511.md`

## Refresh inputs

The v2 Grand Council Fields-Medal Theoretical Floor (2026-05-09) estimated:
- Median: **0.140**
- 95% CI: **[0.128, 0.152]**
- Std: 0.012

The 2026-05-11 substrate landings introduce new evidence:
1. **24 non-HNeRV substrates** classified across 6 substrate_classes
   (residual / renderer_replacement / self_compression /
   pose_axis_sidechannel / meta_codec / bolt_on).
2. **Hinton-distilled SegNet+PoseNet surrogate** (LL landing) breaks the
   YUV6 MSE proxy dominance (603.78 -> 0.64 on smoke); adds an IB-Lagrangian
   distillation-gap floor adjustment of -0.001.
3. **19 packet-compiler primitives** (magic codec auto-selector + pr101_*
   + pr103_*) contribute up to 500 bytes of byte-only re-encoding savings
   (ALPHA * 500 / N_REF = 0.000333 score units).

## Refreshed floor estimate

Per `tac.optimization.theoretical_floor_substrate_refresh.refreshed_floor_estimate()`:

| Quantity | Value |
|---|---|
| **Refreshed median** | **0.13867** |
| Refreshed 95% CI | [0.12847, 0.14887] |
| Refreshed std | 0.0052 |
| v2 baseline median | 0.140 |
| Hinton-distilled adjustment | -0.001 |
| Most aggressive substrate floor | 0.133 (hessian_block_fp) |
| Substrates predicting below v2 floor | 3 / 24 |
| Packet-compiler max byte savings | 500 bytes -> 0.000333 score |

The refreshed median is **5e-4 below the v2 council baseline** because the
3 self-compression substrates (hessian_block_fp 0.133, mdl_fp4_tto 0.134,
scpp_substrate 0.135) predict floors below 0.140. The Hinton adjustment
is a small (-0.001) additional shift.

Per CLAUDE.md "Forbidden score claims": this is a PLANNING estimate, not a
measurement. The first empirical anchor below 0.155 (in `[contest-CUDA]`)
will be integrated via `tac.continual_learning.posterior_update_locked`
on its harvest.

## Per-substrate predicted floor (24 rows)

The 5 most-aggressive predictions:

| Rank | substrate_id | predicted_floor | class |
|---|---|---|---|
| 1 | hessian_block_fp | 0.13300 | self_compression |
| 2 | mdl_fp4_tto | 0.13375 | self_compression |
| 3 | scpp_substrate | 0.13500 | self_compression |
| 4 | nerv_as_renderer | 0.14850 | renderer_replacement |
| 5 | mnerv | 0.15050 | renderer_replacement |

The 5 LEAST-aggressive predictions:

| Rank | substrate_id | predicted_floor | class |
|---|---|---|---|
| 24 | anr_token_renderer_v62 | 0.19550 | renderer_replacement |
| 23 | categorical_substrate | 0.17400 | renderer_replacement |
| 22 | dsnerv | 0.15850 | renderer_replacement |
| 21 | ffnerv | 0.15500 | renderer_replacement |
| 20 | blocknerv | 0.15500 | renderer_replacement |

Self-compression substrates dominate the LOW end of the floor distribution
because they don't need to LOWER the substrate's intrinsic distortion
floor — they shift the R(D) curve so the same score is reachable at
fewer bytes, which directly cuts the rate term.

## Pareto frontier (24 substrates)

Per `tac.optimization.theoretical_floor_substrate_refresh.refreshed_pareto_frontier()`:
each substrate maps to a (bytes_midpoint, predicted_score_at_midpoint)
point. The frontier is sorted by bytes_midpoint ascending.

Bolt-ons (cost=0, byte_budget=(0,0)) inherit PR106_R2_BYTES = 178,750
as their midpoint and shift the predicted score by their delta_mid.

The frontier is consumed by Deliverable 2 (autopilot dispatch ranker)
via the substrate composition matrix.

## Minimum-marginal-byte-EV thresholds

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent",
at PR106 r2 frontier (d_pose ~ 3.4e-5) the marginal-value-per-unit-
distortion ranking is:

| Axis | score_per_unit | threshold_for_1e_3_score |
|---|---|---|
| rate | 6.66e-7 score/byte | 1502 bytes |
| seg | 100 score/unit | 1.00e-5 |
| pose (at PR106 r2 d_pose) | 271 score/unit | 3.69e-6 |

**Pose-axis dominates seg by 2.71x at PR106 r2 frontier.** Per CLAUDE.md
operating-point note this is the canonical inversion vs the OLD 1.x score
operating point where SegNet was 7x more cost-effective per unit.

## Updated S_floor estimate (was 0.132 +/- 0.014; refresh below)

| Layer | Estimate |
|---|---|
| **Refreshed median** | **0.13867** |
| Refreshed CI95 | [0.12847, 0.14887] |
| Refreshed std | 0.0052 |
| (was 2026-05-09 v2 median) | (0.140) |
| (was 2026-05-09 v2 CI95) | ([0.128, 0.152]) |

The refresh narrows the CI by ~15% (0.012 -> 0.0102 half-width) and shifts
the median down by 1.3e-3 due to Hinton adjustment + self-compression
substrate priors.

## Architectural class jump opportunities

Per Grand Council §8 A1 gap decomposition (0.193 - 0.140 = 0.053):
- 0.027 from byte axis (joint hyperprior + cross-tensor MI)
- 0.011 from d_seg (boundary-aware allocation + UNIWARD)
- 0.002 from d_pose (super-additive)
- 0.013 from architectural-class jump (88K -> 128-256K params)

The 0.013 architectural-class-jump component aligns with the difference
between ANR's per_class_floor (0.193) and HNeRV's (0.140). The 24-substrate
matrix gives the operator a structured menu of architectural-jump
candidates ranked by predicted floor + EV/$.

## Operator decisions surfaced (by NOT YET pin)

Per `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`:
- **NOT YET item 2** (Phase 2 GPU $223-303): the refreshed floor confirms
  the Phase 2 envelope is justified — 3 substrates predict below 0.140,
  with self-compression substrates as the most aggressive.
- **NOT YET item 3** (Phase 3 GPU $600-1200): the refreshed floor shows
  the architectural-class-jump component (0.013) is realizable via
  ANR/categorical substrate-replacement; Phase 3 dispatches one of those
  substrate trainers.
- **NOT YET item 1** (A1 PR submission): unchanged; refresh does not
  change the A1 dual-axis dispatch decision.

## CLAUDE.md compliance audit

- **Forbidden score claims**: every output tagged `[predicted; ...]`;
  `score_claim=False` invariant on every dataclass.
- **Forbidden /tmp paths**: `write_refresh_report` refuses /tmp /var/tmp
  /private/tmp paths; durable artifact at
  `experiments/results/cathedral_autopilot_dispatch_ranking_20260512T000000Z/theoretical_floor_refresh.json`.
- **HNeRV parity discipline lesson 6** (score-domain Lagrangian): per-
  class floor priors derive from score-aware lane registry rows; weight-
  domain proxies are NOT used.
- **Catalog #123** (no weight-domain saliency on score-gradient substrate):
  the Hinton-distilled adjustment uses the LL surrogate's score-gradient
  KL distance, not weight-domain mean-of-squares.
- **Catalog #124** (representation lane archive grammar): every substrate
  in the matrix has the 8 archive-grammar fields declared in its landing
  memo (verified by Catalog #124 strict gate).
- **Catalog #125** (subagent landing has solver wire-in): see "6-hook
  wire-in declaration" in the landing memo.

## 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" Catalog #125:

1. **Sensitivity-map contribution**: per-substrate predicted_floor + per-
   class architectural entropy bound feed `tac.sensitivity_map.*` as priors.
2. **Pareto constraint**: refreshed_pareto_frontier() IS the Pareto
   constraint surface for the 24-substrate space.
3. **Bit-allocator hook**: minimum_marginal_byte_ev_threshold() returns
   the thresholds the bit-allocator should use; pose-axis dominance
   factor 2.71x is the bit-allocator priority.
4. **Cathedral autopilot dispatch hook**: the refresh consumes the same
   matrix as Deliverable 2; the autopilot's ranked dispatches are
   evaluated against this floor.
5. **Continual-learning posterior update**: when an empirical anchor lands
   below 0.155, it routes through `posterior_update_locked` and the
   refresh re-runs to update the median.
6. **Probe-disambiguator**: per-substrate-class predicted floors form a
   probe-disambiguator pair (hnerv vs nerv-family vs anr vs categorical
   vs vqvae); empirical Phase 2 dispatches will resolve the regime.
