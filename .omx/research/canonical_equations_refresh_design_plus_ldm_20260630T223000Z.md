# Canonical equations registry REFRESH — design refinements + LDM grounding

**UTC** 2026-06-30T22:30:00Z · **authority** `[equations / math view — advisory · NON-PROMOTABLE]`
**pointer UNMOVED 0.19110** · score_claim **false** · promotable **false** · ready_for_exact_eval_dispatch **false**
**Scope** CPU-only registry edit (no GPU, no launch, no live-run / residual-pipeline / `tac.lie` source touch).

Operator flagged the canonical equations registry (`tac.canonical_equations`) as out of date. This landing folds in
BOTH (a) the 2026-06-30 design refinements (the gauntlet 3-pass + screw/twist research + pose=screw/canonicalize
insights) and (b) the LDM theoretical grounding from the just-read paper. This is the **EQUATIONS view** of the
triality (DAG ↔ DSL ↔ equations); the design-refine step verifies cross-view consistency (note at the bottom).

## LDM paper folded in (THEORETICAL ANCHOR — cited, NOT measured)

**Mikulasch & Zenke, "Understanding SSL via Latent Distribution Matching", ICML 2026.** Three load-bearing results:

1. **`F_LDM = −D_KL[R(z,z′) ‖ P_θ(z,z′)] = alignment ⟨log P_θ(z′|z)⟩ + uniformity H_R[z,z′]`** → our action
   `S_τ = distortion (d_seg/d_pose) + rate (entropy/MDL)`. **alignment ↔ distortion, uniformity ↔ rate.** The
   uniformity term is an entropy estimator → grounds rate-as-entropy-estimator (our E7).
2. **Theorem 1 (identifiability):** predictive model + (nonlinear) Gaussian predictor + invertible encoder covering
   latent space ⇒ recovers the true latent **UP TO AFFINE** → grounds **pose = the identifiable ego-motion latent
   (screw/twist ξ) up to an affine map**; the physical-ξ ↔ PoseNet-6-vector gap is an AFFINE calibration (the B3
   read-back), not a mystery.
3. **Discussion:** SSL latents high-dim but intrinsic-dim "a few tens"; SSL = **geometric reparameterization, NOT
   lossy compression**; the entropy estimator is the key term → grounds our intrinsic-dim/residual-ID work (the
   ~8–9-D manifold, mod-dim DERIVED) + canonicalize-to-ground-frame (ego-removal IS the reparameterization).

## Equations registered (APPEND-ONLY — 7 NEW ids; registry 178 → 185)

Every new id REFERENCES the prior E0–E12 rows via an `annotates` key in `domain_of_validity`. The originals are
**NOT mutated and NOT re-registered** (proven below).

| # | equation_id | tier | annotates / grounds |
|---|---|---|---|
| 1 | `witness_action_ldm_alignment_uniformity_correspondence_v1` | **THEORETICAL-ANCHOR** (LDM, cited) | E0 master + E7 rate + E6 IB |
| 2 | `pose_ego_screw_twist_identifiable_up_to_affine_v1` | **DERIVED + MEASURED**(warp,research-signal) **+ THEORETICAL-ANCHOR**(affine, LDM Thm 1) | E9 pose + Wyner-Ziv pose |
| 3 | `witness_canonicalize_to_ground_frame_residual_v1` | **SOLVED** (structural; pending realized-through-R) | E0 + E7 |
| 4a | `ego_motion_cumulative_se3_bspline_v1` | **DERIVED** (Sommer-Usenko arXiv:1911.08860; anchors pending) | temporal factor of #3 |
| 4b | `dual_quaternion_screw_blend_annulus_seam_v1` | **DERIVED / ASPIRATIONAL** (Kavan TOG 2008; measure through R) | per-class warp seam |
| 4c | `movables_stored_out_of_inr_multibody_v1` | **SOLVED + DERIVED**(d_seg ~0.0008 estimate; pending) | E7 capacity allocation |
| 5 | `residual_manifold_intrinsic_dim_whitney_v1` | **DERIVED**(Whitney 2m+1) **+ MEASURED**(lane-orbit dim ~8) **+ THEORETICAL-ANCHOR**(LDM few-tens) | E0 residual architecture |

**MEASURED anchors attached** (real research-signal, advisory/pre-R — `[macOS-MLX research-signal]`):
- #2 `grok_pose_warp_dual_use_dseg_modulation_feed_ja_20260630` — the grok pose-warp $0 CONFIRMED (commit 2f83e0b9e,
  FEED-ja): Road ground-homography +15% d_seg, calibration closes; pose is FREE dual-use.
- #5 `lane_orbit_manifold_dim_8_decisive_20260623` — d_seg islands = lane markings = ~8-dim nonlinear manifold
  (DECISIVE 2026-06-23, 9 lines converge).

**THEORETICAL / DESIGN-pending (empty or theoretical anchors — honestly flagged):** #1 (pure LDM correspondence),
#2's affine read-back (`pose_identifiable_up_to_affine_ldm_theorem1` anchor, `INFERRED_FROM_DOMAIN_LITERATURE`,
read-back validation PENDING — fallback = store the PoseNet-6-vector), #3 / #4a / #4b / #4c (design-stage; anchors
pending realized-through-R), #5's residual-specific TwoNN/MLE ID ($0 measurement PENDING).

**NO-FAKE:** every theoretical row carries `provenance_tag` + `tier` in `domain_of_validity` explicitly distinguishing
THEORETICAL_ANCHOR (cited, not measured) from DERIVED/SOLVED/MEASURED. The LDM grounding is NEVER presented as an
exact-eval result; pointer 0.19110 moves only on the byte-closed exact row.

## APPEND-ONLY proof (Catalog #110/#113 HISTORICAL_PROVENANCE)

Snapshotted the 4 annotated originals' `to_dict()` sha256 BEFORE and AFTER registration:

```
UNMUTATED witness_unified_action_fixed_fisher_background_v1   554ac38ec2c0d4f7 -> 554ac38ec2c0d4f7
UNMUTATED pose_sqrt_concave_coupling_sidecar_v1               ece3ad614872bf0b -> ece3ad614872bf0b
UNMUTATED rate_mdl_cosmological_constant_reverse_waterfill_v1 5fb9e99a69bca8af -> 5fb9e99a69bca8af
UNMUTATED indirect_rd_logloss_equals_information_bottleneck_v1 6f10f6a34e68580d -> 6f10f6a34e68580d
APPEND-ONLY honored (no original mutated): True
```

The registry ledger appended 7 `registered` events; no `register` re-fired on any existing id (the script skips
already-present ids). 248 `canonical_equations` tests pass.

## Triality-consistency note (for the design-refine step to verify)

This is the EQUATIONS view. The DSL view (`tac.witness_dsl`) must carry the matching constructs and the DAG view
(`sub015_DAG_*`) the matching FEED entries; per the design-refine requirements memo
(`thetastar_residual_inr_config_update_requirements_*`) the three are ONE object in three views. Consistency map:

| equation (this landing) | DSL construct (to verify present/consistent) | DAG FEED (to verify present) |
|---|---|---|
| #2 pose=screw | `per_class_warp{...}` + stored-pose ξ | FEED-ja (grok pose-warp) |
| #3 canonicalize-to-ground-frame | `ground_frame_canonicalize(screw)` + `residual_mode(target)` | FEED (canonicalize) |
| #4a SE(3) B-spline | `se3_bspline(controls)` | screw research (a7eda614) |
| #4b screw-blend | `per_class_warp{blend=dual_quaternion}` | screw research |
| #4c movables-stored | `store_movables(codec)` | design-refine req |
| #5 mod-dim Whitney | `mod_dim=derived` | $0 residual-ID FEED |
| #1 LDM correspondence | (theoretical annotation; informs the rate/distortion terms) | this thread |

**Consistency invariant:** the DSL program compiles to the command the DAG records, governed by these equations. The
design-refine verifies all three agree before R2/fire ($0 check). means≠ends: equations are a MEANS.

## Reproduce / audit
```
.venv/bin/python tools/register_witness_design_ldm_equations.py   # idempotent: skips already-present ids
.venv/bin/python tools/list_canonical_equations.py | grep -E "ldm_alignment|ego_screw_twist|ground_frame_residual|se3_bspline|screw_blend|movables_stored|intrinsic_dim_whitney"
```
Registry ledger: `.omx/state/canonical_equations_registry.jsonl` (fcntl-locked APPEND-ONLY).
Script: `tools/register_witness_design_ldm_equations.py`.
