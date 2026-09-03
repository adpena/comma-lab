---
title: "ddm_vr1 — the v7–v11 era, recalled at source and priced against the two live doors: 7 FOLD-NOW rows with landed code the born trainer and the fold-back do not consume, and a second independent proof that capacity is closed"
unit: ddm_vr1
charter: .omx/research/charters/ddm_vr1_v7_v11_signal_recall_20260903.md
date_utc: 2026-09-03
axis: "read-only recall + S-arithmetic. NO scorer, NO Modal, NO Metal, NO builds, NO edits to live-arm files."
score_claim: false
promotion_eligible: false
pointer_moved: false
own_vehicle_frontier: "afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED by this unit"
verdict_scope_default: "per-row INSTANCE; family verdicts only where the cited receipt is n600 on the frozen scorer"
repo_head_at_write: 0509b1757
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_vr1 — the v7–v11 signal recall

Craft contract: `docs/operating_manual_craft_handoff.md` (verify by RE-DERIVING from primary
artifacts; label MEASURED / DERIVED / INFERRED out loud; attack your own conclusion). Every number
below was opened at its file, not recalled (m44). A row with no receipt is UNPRICED and unranked.

## §0 ANSWER FIRST

The operator is right that the era holds signal, and the reason it is still signal is measurable:
**the live doors consume almost none of it.** I grepped the two live trainers and the population
pipeline for every v-era surface. What today's born trainer
(`experiments/ddm_qbt1_qbflow_trainer.py`, 42d322db5), the fold-back trainer
(`src/tac/pr130_lift/train_semantic_quantized_resumable.py`, 42d322db5) and the population stage
(`src/tac/semantic_pipeline/stages/train.py`) actually import from the v7–v11 substrate is:

- `tac.witness_dsl.curriculum_dsl.EmaDecayCalibrated` + the `ema_decay_run_geometry_v1` LawRef
  (all three doors) — the EMA law, and only that;
- `tac.boundary_math.power_diagram_witness.open_stored_npy_memmap` (qbt1:53) — a memmap utility,
  not a lever;
- `tac.witness_dsl.activation_ledger` via `src/tac/pr130_lift/live_lever_activation.py` (2a83341b2).

**Zero** loss terms, weight allocators, render kernels, basis frames, or capacity-routing primitives
from the era reach either door *by import*. (The doors are not primitive-less — qbt1 carries its own
r7 constrained-margin dual, its own birth gate, and its own along-tangent features, all authored in
the ddm line in August. The point is narrower and checkable: none of it came from the v-era
substrate, and the v-era substrate's own levers sit unconsumed beside it.) The charter's PRIOR-LAW
PREDICTION (≥ 5 FOLD-NOW rows with landed code the live doors do not consume) **holds with 7**; the
falsifier (< 3) does not fire.

The era also carries one gestalt-level fact worth more than any single lever: it measured the
capacity verdict on a completely different vehicle, four weeks before gc1 measured it on the born
form, and got the same answer (§5).

## §1 INVENTORY AT SOURCE — what the era is, in code and receipts

### 1.1 The SPECs (memo-level, read in full)

| artifact | what it is | the load-bearing content I consumed |
|---|---|---|
| `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` | sealed single-trunk launch vehicle | §2 sealed constants w/ provenance; §8 the OPERATING CONTRACT (resumability P0; ALREADY-SETTLED table; no-stray; execution guardrails; cathedral invariant). §8B is a durable do-not-re-derive list. |
| `.omx/research/SPEC_v8_perclass_decomposition_20260708.md` | edge-centric per-class carriers | §1 tropical argmax of decoupled SDF fields; §2 carrier table (MyCar IoU 0.994; Lane band d_seg 0.00087; Road+Undriv 20–50 KB DERIVED; net 27–57 KB vs 114 KB); §3 **merge→diff→correct, chroma-first / luma-reserved**; §4 the six seam risks. |
| `bregman_v9_all_surfaces_{build_spec,measurement,binding}_20260714.*` | the v9 Bregman/metric stack | consumed into `policy_bindings/optimal_metric`; vh1 §4 already recorded "no branch-relevant residue". Confirmed: `bregman_*` equations carry 1 anchor each and 3 of 6 have **zero consumers**. |
| `.omx/research/BUILD_SPEC_v10_compiler_receiver_20260718.md` | typed compiler + counted receiver | the frozen 7-section `v10_counted_program.v2` wire format and its exact factor custody; factors `2` and `10` remain explicit `MISSING` / `consumer_id=BLOCKED`. Apparatus certificate, not a score mover. |
| `codex_findings_ddm_v11_obligation_vocabulary_solve_20260722_codex.md` | scorer-obligation vocabulary | measured n64/n256 windows: bytes **rose** (52,204→52,523; 72,933→73,508) for d_seg deltas of 5.6e-5 / 1.6e-5. The obligation vocabulary admitted 1/24 and 3/32 bundles. |
| `.omx/research/ddm_vh1_v8v9v10_harvest_20260730.md` | the PRIOR harvest of this same era | 16 ranked rows + §2 the cross-vehicle law taxonomy + §4 nothing-more verdicts. **Its consumers were TR1/BR-A..D/burn-3 — all retired.** vr1 re-aims the same corpus at qbt1 / ft1 / fpc3 and does not repeat its rows. |
| `.omx/research/ddm_hr1_realization_engineering_20260811.md` | the four-arm realization race | the direct ancestor of ft1 (§4 below). |
| `.omx/research/ddm_lr2_legal_realization_ladder_20260804.md` | the legal realization ladder | §0: every rung that SHIPS an offset field loses against its own bar (η ≤ 0.12 for every deterministic transport realizer measured); the live descendant is per-block SOLVED paint with no offset field. |

### 1.2 The DSL leg — MEASURED: it describes a retired vehicle

```
tac.witness_dsl.lever_registry.completeness()   (lever_registry.py 7f6301ed2)
  trainer_path          = experiments/train_levelset_witness_realized_through_R_mlx.py
  vehicle_label         = [RETIRED vehicle: train_levelset_witness_realized_through_R_mlx.py]
  describes_live_vehicle= False        LIVE_TRAINER_BASENAME = "train_tr1_partition_renderer_mlx.py"
  trainer_total 443 · dsl_referenced 372 · mapped 363 · unmapped 80 · stale 3 · coverage 0.8194
tac.witness_dsl.activation_ledger.never_fired() → 209 rows
.omx/state/lever_activation_ledger.jsonl        → 340 rows, last written 2026-08-22
```

The registry's own idea of "live" (`train_tr1_partition_renderer_mlx.py`) is itself retired, so the
registry is **two generations behind** the born trainer. `src/tac/pr130_lift/live_lever_activation.py`
(2a83341b2, 2026-08-17) already cured this **for the fold-back trainer only** — deriving the lever set
by AST from that trainer's own `add_argument` calls: **18 live levers, 4 never-fired**
(`carrier_rank_penalty`, `carrier_tensors`, `distill_max_seg`, `distill_weight`). The born trainer
`ddm_qbt1_qbflow_trainer.py` is covered by **neither** registry.

### 1.3 The equations leg — 88 of 465 rows are v-era topical

`tools/list_canonical_equations.py --json` → 465 rows; 88 match the era's topic keywords
(margin/flip/fisher/uniward/curvelet/shearlet/parabolic/capacity/waterfill/kkt/persistence/eikonal/
chan_vese/laguerre/island/birth/bregman/costate/realization/geometric_rate/ema_decay/screw/aa_sdf/
boundary). Registered-but-never-anchored (`empirical_anchors == []`, i.e. a law with no measurement):
`certified_layer_precision_waterfill_v1`, `frozen_scorer_fisher_pullback_metric_v1`,
`score_marginal_lagrange_multipliers_v1`, `witness_measured_reverse_waterfill_v1`,
`ema_decay_substrate_stage_aware_v1`, `top1_ordinal_margin_minimality_v1`,
`ddm_lp1_deepest_home_context_waterfill_v1`, `ddm_tolerance_capped_min_score_waterfill_v1`,
`hope_bn_capacity_per_stratum_codebook_v1`, `dual_quaternion_screw_blend_annulus_seam_v1`,
`ego_motion_cumulative_se3_bspline_v1`, `mipod_..._fisher_information_cost_matrix_savings_v1`,
`uniward_standalone_no_op_..._v1`, `segnet_input_costate_injection_v1`,
`cls_lowres_downsample_policy_boundary_preservation_v1`. Zero-consumer (orphan) rows:
`compact_shearlet_parabolic_capacity_v1` (its own domain field says
`dsl_wire_status: OWED_SERIALIZED_NO_LIVE_CONSUMER_ASSERTED`) and three `bregman_*` rows.

### 1.4 `src/tac/boundary_math/` — landed, tested, and pointed at retired consumers

Grep of non-test importers (`grep -rl "boundary_math\.<m>|boundary_math import <m>" --include="*.py"
src tools experiments submissions`):

| module (commit) | tests | non-test importers | reaches a LIVE door? |
|---|---|---|---|
| `aa_sdf_observation_render.py` (492503662) | `tests/test_aa_sdf_observation_render.py` | `curriculum_dsl`, the RETIRED levelset trainer, `tools/levelset_byte_close_and_eval.py`, its verify tool | **no** |
| `chroma_boundary_match.py` (7d3be7e00) | `tests/test_chroma_boundary_match.py` | only its own canonical-equation module + `tools/seed_lever_relative_significance.py` | **no** |
| `horizon_weighted_margin.py` (5b049c73f) | `tests/test_horizon_weighted_margin.py` | same two | **no** |
| `persistence_topology_loss.py` (83e0df8a4) | `tests/test_persistence_topology_loss.py` | `witness_dsl.gauge`, `levelset_micro_batch_loss`, `metal_persistence_pool`, RETIRED trainers | **no** |
| `island_protection.py` (e1bfb97d0) | `tests/test_island_protection.py` | 13 files — all retired trainers/probes/`cuda_v9_island_runtime` | **no** |
| `laguerre_logit_offset.py` (aab422c5c) | `tests/test_laguerre_logit_offset.py` | 10 files, all v-era probes/trainers | **no** |
| `analytic_lane_render_band.py` (c219841d8) | (via v8 equation modules) | 40+ tools/probes; `src/tac/preflight.py` | **no** |
| `lane_sdf_component.py` / `hood_static_component.py` | — | 40+ / 10; `optimization/ddm_hood_static_reassert.py` is the newest | **no** |
| `windowed_curvelet_frame.py` / `compact_shearlet_frame.py` / `localized_basis_frames.py` | (equation modules) | `basis_control`, `boundary_coordinate_joint_solve`, v-era packet `inflate.py`s | **no** |
| `boundary_solver.py` (7237d3eee) | — | `dense_raster_lzma_baseline`, `legal_frame_bridge`, 2 smoke tools | **no** |

### 1.5 The costate organ — the one v-era organ that IS current

`src/tac/ddm_costate_organ.py` (cd678f402, 2026-09-03, 3,478 lines) is the live successor; its SENSE
surface is the DDM receipt fleet (dv1, g3, g4, v19-family), NOT the witness-training-era digest. It
is imported by 10 non-test modules and `tools/costate_digest.py`, which is wired as a SessionStart
hook (`.claude/settings.json:85`). Its `_legacy_duty_count()` (organ:555) still reads
`tac.witness_dsl.activation_ledger.duty_to_measure_ranked` — i.e. the duty queue it surfaces for
levers is the RETIRED-vehicle queue of §1.2.

## §2 THE RANKED FOLD TABLE (25 rows)

Ranking = (probability the row moves the exact score) × (readiness: landed code + a live consumer
that lacks it) ÷ cost. Exchange rates used throughout, DERIVED_EXACT from
`score_atomic_flip_byte_exchange_v1`: **1 argmax site = 8.477105034722222e-07 S = 1.273108215332031 B;
1 archive byte = 6.658589531221714e-07 S.** afr1's seg axis = 2.0139e-4 × 117,964,800 = **23,757 sites
= 0.020139 S**.

| # | artifact (era) | status | plugs into | expected effect (derivation) | verdict |
|---:|---|---|---|---|---|
| 1 | **`_live_margin_weight`** — per-pixel realized-margin loss allocator, mean-1 normalized + `stop_gradient`, three allocators (`inverse` / `exp` / `bottom-k`). `experiments/train_witness_realized_through_R_mlx.py:1086` (5d27d3354) | LANDED CODE (MLX; needs a ~30-line torch port) | **qbt1 `expected_flip_margin_loss` :537-538** — today `per_sample = sigmoid(-margin/tau).mean(dim=(1,2))`, a UNIFORM pixel mean. Also fpc3. | Concentration is n600-MEASURED and scorer-side: `waterfill_annulus_through_r_..._v1` anchor 0 — annulus fraction **0.0571** captures **flip_capture 0.9823**, enrichment **17.203×**, residual outside 7.9e-5. FEED-bp (DAG:1205, n=60 advisory): witness-hard px margin median **0.42** vs global **5.79** (14×); **89.2%** of d_seg in the bottom-5%-margin pixels, 47.8% in bottom-1%. FEED-bq (DAG:1215) quantified the re-route as a **3.55× fixed-byte d_seg downshift**. On the born field (bz2 d_seg 0.01299522) 3.55× → 0.003661, still **26.8×** above qn1's 1.3646784205e-4 target. Real lever, not the whole 95.2× gap. | **FOLD-NOW** |
| 2 | **AA-SDF footprint-integrated render** (supersample-box ss=2 / mip-NeRF IPE). `src/tac/boundary_math/aa_sdf_observation_render.py` (492503662) + tests + `tools/aa_sdf_observation_render_verify_n600.py` | LANDED CODE + **MEASURED LAW** `aa_sdf_observation_footprint_render_dseg_v1` | **qbt1 `QBFLOWTorch.forward` :477-479** — `rgb = torch.sigmoid(_linear(render_state, ...))` is POINT-sampled at exactly `EVAL_H, EVAL_W = 384, 512` (:68), then bicubic→camera→uint8 STE. | n600, frozen CPU-torch SegNet, through contest R, **at grid 384**: `point_dseg` **0.0054940456814236115** → `aa_dseg` **0.0008598581949869792** = **6.389×**; recall 0.7734 → 0.9755. `aa_dseg_gain` 0.004634187486436632 = **0.4634 S** at 100×. HONEST SCOPE: measured on the real-frame achievable signal (a confound-free upper bound on RENDER LEGIBILITY), not on a trained field — so it bounds the manufactured-at-native-render class that `mst1` prices at 78.71% of today's seg error. The born field (0.01299522) sits **2.37× ABOVE** the point-sampled bound, so the mechanism is in range for qbt1. | **FOLD-NOW** |
| 3 | **Chan-Vese one-sided area constraint** — `E = (λ_c/2)·relu(A_c − A_c^GT)²`, `λ_c = W_birth/(δ·A_GT_c)`. Equation `chan_vese_area_constraint_birth_balance_v1`; DSL `AreaConstraintBirth` | MEASURED LAW (anchor 0) + DERIVED λ (anchor 1, A/B OWED) | **qbt1 `dual_ascent_margin_constraints` :593** — the live constraint set is **RECALL-ONLY** (Lane werr ≤ 0.12, Movable ≤ 0.009). Nothing caps rare-class over-paint. | n600 read at ep125: part_frac Lane **13.761× GT**, Movable **4.581× GT**; majority deficit **0.1189** ≈ rare excess **0.1191** (exact mass conservation) ⇒ **Road d_seg floored at 0.398**. The law: one-sided rare-class pressure with no area cap imposes a majority-class floor, capacity- and time-INDEPENDENT. Corroborated on today's object: gc1 measured square atoms adding **~39,000 Movable errors**. qbt1's dual ascent is exactly the un-capped pressure this law describes. | **FOLD-NOW** |
| 4 | **Per-class-pair margin normalization** from the exact rank-4 head. Equation `segnet_head_rank4_linear_flipdist_v1`; producer `experiments/results/segnet_fractal_20260715/stage_b1_gates_head.py` | MEASURED LAW (2 anchors) | **qbt1 `expected_flip_margin_loss` :538** uses ONE scalar `tau` for every class pair; `tau_for_step` anneals 0.15→0.05 globally. | The centered head is **exact rank 4** (rank-4 recon max-abs err **5.96e-8**; singvals [3.128, 2.154, 2.025, 1.796, **0.0**]), so flip distance has the closed form `|margin| / ‖w_c − w_c'‖`. Measured median `flipdist_feat`: Undrivable-Movable **0.081**, Road-Movable 0.106, Road-Lane **0.131**, Road-MyCar 0.171, Road-Undrivable **0.177** — a **2.185×** spread. A scalar τ therefore mis-scales the modelled flip probability by up to 2.19× across edges, systematically under-weighting the Undriv↔Movable edge. Cost: one frozen (5,5) constant matrix. | **FOLD-NOW** |
| 5 | **Chroma-first / luma-reserved channel routing** (SPEC_v8 §3, the near-triangular correction Jacobian). Code: `tac.boundary_math.chroma_boundary_match` (7d3be7e00) + DSL `SegChromaBoundary`; equations `chroma_boundary_annulus_match_hinge_v1`, `rgb_chroma_necessity_per_boundary_pair_v1` | LANDED CODE (orphan) + MEASURED LAW | **ft1's fine-tune** — its named risk is "a changed render shifts PoseNet". Also qbt1's `rgb` head. | VERIFIED AT SOURCE (`upstream/frame_utils.py:51-72`): `rgb_to_yuv6` box-averages U,V over 2×2 → chroma reaches PoseNet **4× low-passed**, while SegNet reads RGB at full resolution (`modules.py` `SegNet.preprocess_input`). Measured n600: `d_pose_delta_desat_annulus` **0.000426** vs `d_pose_delta_keep_annulus` **0.00957** (**22.46×**) — annulus-confined chroma edits are the pose-cheap channel; chroma survives R at gain **0.98–1.00**; `d_seg_equiv_desat_annulus` 0.002972. **93.4%** of chroma-flips lie in the fragile annulus (GT top-2 margin < 1); removing chroma costs Lane→Road **0.0754**, Movable→Undriv **0.0438**; chroma is **0.212** of the margin-gradient energy. HONEST NUANCE that cuts against the fold: chroma-dominant boundary fraction is Movable **0.30–0.36** > horizon 0.24 > hood 0.22 > **Lane 0.08** — the class holding 33.56% of the bits is the class chroma decides LEAST. | **FOLD-NOW** |
| 6 | **Margin-band satisficing cap** — `m_safe = headroom · δ_R`. Equation `margin_band_satisficing_threshold_v1`; DSL `MarginBandSatisficing` + `lawref.resolve`; stub `fh1_margin_satisfice_cap` (fh1_adapted_force_levers_20260731.py:58, 7f6301ed2) | MEASURED LAW + DESIGNED-STUB | **qbt1's loss is unbounded** — `sigmoid(−m/τ)` keeps buying margin at already-safe sites forever. | n96, frozen CPU-torch: **δ_R = 0.019590163230895963**, derived headroom **2.0**, **m_safe = 0.039180326461791926**, full-R annulus p95 0.03712. Above `m_safe` the margin cannot be un-done by R, so gradient spent there is spent on nothing. Composes with row 1 as a CAP on the allocator (fh1 R3), never a finisher stage. `headroom_3_status: OPEN_UNMEASURED`; `treatment_delta_s: UNMEASURED`. Re-run `tools/measure_delta_R_noise_floor.py` at n600 — never rebuild it (SPEC_v75 §8B). | **FOLD-NOW** |
| 7 | **Persistence / soft-clDice island-recall loss**. `src/tac/boundary_math/persistence_topology_loss.py` (83e0df8a4) + tests; equation `persistence_topology_cldice_betti_island_recall_v1` | LANDED CODE (MLX) + MEASURED LAW | **qbt1's Lane term** — Lane is 0.59% of area, **33.56% of the model bits**, IoU 0.263. qbt1 has NO thin-structure term. | numpy-fp32 on real n600 GT: **clDice erasure sensitivity 8.826 vs CE 0.07983 = 110×**; **island recall gain +0.4433**; `bulk_dseg_delta 0.0` (it does not disturb the bulk). CLAUDE.md warrant: mf1 RETIRED Morse–Smale but **persistence SURVIVES** on F1b (diagram well-defined without perturbation: 0.00000% ties, 100% isolated) — cite F1b, never Morse–Smale. Cost: a torch port of an MLX loss. | **FOLD-NOW** |
| 8 | **Costate-driven schedule** (state-gated coordinated cascade). `src/tac/ddm_costate_organ.py` (cd678f402) + `tac.witness_control.{lambda_net,costate_panel,control_alphabet,prototype_router}`; equation `costate_lambda_marginal_ds_v1` anchors 2–3 | LANDED CODE + MEASURED (walk-forward backtest, advisory) | **fpc3** ("its objective should be the fold-back objective, not the PR130 curriculum") and qbr1 cell selection. | #430 replay: the organ's state-gated cascade beats the hand-scheduled #205 curriculum on **all 3** replay models — **∫d_seg·dep 6.283 vs 8.649 = −27.4%**; self-replay MAE 0.0054, final gap 1.6e-4. The tri-gate (LOO ∧ walk-forward ∧ binding-floor 0.8) measured and extincted the look-ahead bias that inflated the earlier n=5 claim. λ = (100, 5/√(10·d_pose), 25/37545489). | **FOLD-AFTER-BURN** (firing it now would confound the running discriminator) |
| 9 | **Along-tangent frequency ladder + parabolic normal companion**. Equations `windowed_curvelet_parabolic_capacity_v1`, `shearlet_nterm_upper_bounds_task_rate_v1`; code `boundary_math/windowed_curvelet_frame.py`, `compact_shearlet_frame.py` | LANDED CODE + MEASURED LAW | **qbt1 `forward` :462-468** — `along_features` are sin/cos of `π·f·u_tangent` at FIXED f ∈ {8,16,24,32}, oriented by the road-probability gradient tangent. Normal-direction information reaches the flow only as raw `road_condition` features (`gx, gy, tangent_x, tangent_y`, :439-441); there is **no across-tangent frequency companion and no parabolic scale ladder.** | The along-tangent deficit is **3.125 (~3.2×)** and codex-verified NOT attributable to R attenuation (sampled R-chain anisotropy < **0.056%** — `codex_findings_ddm_gc2_oracle_gap_closure_20260724_codex.md:36-37`). The parabolic law `σ_n = (σ_t/aniso)²/w0` is implemented and certified monotone; measured boundary-spectrum anisotropy n600 **41.0**. Realized capacity gain is MODEST and honestly reported: spectral reverse-waterfill 1.7–2.0× (linear upper bound) but **spatial OMP N-term only 1.09× (n600) / 1.23× (n96)**. Shearlet N-term is a PROVEN UPPER BOUND on the task rate at O(N⁻²(log N)³); tightness UNPROVEN. | **FOLD-AFTER-BURN** (a form change; race, never adopt) |
| 10 | **v8 class-matched carriers**: analytic lane ground-frame band + Movable Hungarian slot-track + MyCar static clamp. `boundary_math/{analytic_lane_render_band,lane_sdf_component,hood_static_component,curve_relative_offset_coder}.py`; equation `v8_geometric_rate_decomposition_v1` (8 anchors) | LANDED CODE + MEASURED LAW (n600, bit-exact real coders) | The class-matched-atom question gs3 ADDENDUM 1 leaves open ("square atoms barely improve Lane and add ~39,000 Movable errors → class-protected anisotropic/curve atoms"). | n600 on `gt_n600.npz['lstars']` with real bit-exact coders: whole-scene bitmap **0.339** → geometric-dominant **0.061** (5.5× under bitmap, 1.9× under the then-frontier 0.118) → lossless-complete **0.140**. In bytes: dominant **91,611 B**, complete **210,255 B** = **1.854× the shipped HPAC token stream (113,411 B) for the same field.** Carriers: MyCar IoU 0.994 (~0.1–0.5 KB), Lane band d_seg 0.00087 (~1–2 KB), Movable ~2–6 KB. Boundary cover 72.5% at 1.0 px fit residual. | **FOLD-AFTER-BURN** as an ATOM FAMILY for the born generator; the COMPLETE-replacement reading is SUPERSEDED (row 21). |
| 11 | **Optimizer/objective transition warm-start law**. Equation `muon_finisher_schedule_warmstart_and_lr_anneal_v1` (3 anchors) | MEASURED LAW | **qbt1's `curriculum_mode: "ce_birth_then_margin"`** switches objective discontinuously at step `birth_max_steps` (:2242). ft1 switches four times (`float_ce → ce → softplus_margin → expected_flip`, :325-337). | Two independent receipts for the same mechanism: a cold optimizer transition costs a **d_seg spike of +0.000357**; at mod32cap ep726 the cold Muon fire quenched d_seg **0.0034139 → 0.0043514 (+27.5%)** and the finisher **never re-beat the ep650 best 0.0033662**, ending +11% above it. Mechanism: "cold first orthogonalized step = wild unit-norm direction from one noisy gradient → boundary thrash." The transferable law is not about Muon: **any discontinuous change of the descent direction near a piecewise-constant argmax field thrashes the boundary; warm-start the momentum across every stage boundary.** | **FOLD-AFTER-BURN** |
| 12 | **Expected-flip margin law, scalar top1−top2 form**. Equation `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` | MEASURED LAW (n600 EXACT) | qbt1 `expected_flip_margin_loss` :523; ft1 `_phase_for_step` terminal phase. | **ALREADY-IN, and now warranted at the strongest available scale:** `gap13 − gap12 ≥ 0` at **every one of 118M pixels** (`gap13_minus_gap12_min = 0.0`), bit-exact against the cached margins on the frozen scorer. Only the 2nd-place class can flip the argmax first, so the scalar margin misses no flip-ONSET fragility and the multi-class-simplex lever is a clean NEGATIVE. Triple junctions are a flip-STRUCTURE DOF (car corners, lane-facet fraction 0.414), not the lane tail. | **ALREADY-IN** |
| 13 | **Pose in the loop at step zero**. | ALREADY-IN | qbt1 `joint_objective` :628 — `total = 100·(realized_seg + interface_seg) + sqrt(10·pose_mse)` from the first step. | The v7.5 POSE LAUNCH GATE is the receipt for why: run-1 measured `d_pose ≈ 1.79` flat ⇒ **√(10·1.79) ≈ 4.24 of S from pose alone**, and SPEC_v75 §1 states plainly that the 3.4e-5 is ancestor full-RGB-photometric and non-transferable. qbt1 already obeys the successor law. | **ALREADY-IN** |
| 14 | **EMA decay resolved through run geometry**. Equation `ema_decay_run_geometry_v1`; lever `EmaDecayCalibrated` | ALREADY-IN | qbt1 :57/:2331/:3498; `semantic_pipeline/stages/train.py:27`; `pr130_lift/train_semantic_quantized_resumable.py:61`. Strict gate `check_ema_executable_law_matches_sealed_law` (wc3). | The era's own rule, still binding: hr1 states "the active EMA must be re-anchored whenever the event-run geometry changes; carrying a smoke-window decay into a longer continuation is refused." This is the **one** v-era law all three live doors consume. | **ALREADY-IN** |
| 15 | **Per-class primal-dual (KKT-flavored) capacity routing**. `dual_ascent_margin_constraints` :593 + `MARGIN_CONSTRAINT_LANE_MOVABLE` pins | ALREADY-IN (one-sided) | qbt1 :2255-2268; bounds Lane 0.12 / Movable 0.009, η_λ 0.11387788414126129 (derived from the retained r6 endpoint Lane werr 0.9981336319522209), λ_max 5.0 reusing the ddm_lg1 natural-unit ceiling. | The projected dual ascent on REALIZED within-class error is a genuine KKT routing and it is live. What the era says is missing is the **other** side of the constraint set (row 3) and the per-PIXEL allocator (row 1). The 5-regime waterfill SPEC_v8 §2 promised generalizes this, and remains unbuilt. | **ALREADY-IN** |
| 16 | **RG1 debt-proportional per-pixel band weight**. `src/tac/pr130_lift/band_objective.py` (60aefac08) | ALREADY-IN, default OFF, **FIRED ONCE and measured NEGATIVE at formulation scope** | ft1 trainer :980-983, :1178-1251; `--band-objective-weight` (α ∈ [0,1]), table sha recorded only when active. | `W_e = flips_e / band_px_e` per unordered class pair, same rule off-band; `band_weight_field` returns `(1−α)·1 + α·W/mean(W)` whose mean is **exactly 1 algebraically**, so it cannot rescale the effective LR. Grounding: rt1 — **99.22% of the seg axis sits exactly ON the transmitted label boundary** (zero archive bytes to address); av3 — `peak_flips ~ ‖dw‖^0.458`, R² **0.9969**, the diffusive signature of a piecewise-constant argmax field. **The firing receipt, which I found in the ledger and which changes the prior:** `.omx/state/lever_activation_ledger.jsonl:278` (fired α=1.0, 2026-08-17) and `:319` (measured 2026-08-22, `codex:ddm_nl1_never_fired_levers`) — "[macOS-MPS training-signal], n600 semantic replacement: 600 activations rotated the realized step **66.3 degrees**, but **best exact-seg stayed at init** and the **residual judge was structurally underpowered**; 40,252 packed bytes. FORMULATION scope." Verdict: `.omx/research/ddm_jr1_band_objective_judge_repair_20260817.md`. So the term DOES move the descent direction materially; what failed was the judge that had to admit the move — and MPS is never authority. Its docstring distinguishes it from row 1: this weights by measured per-edge DEBT, row 1 by margin MAGNITUDE. **Different variables; race them, never stack blindly (m164: union ≠ sum of legs, 3.705×).** | **ALREADY-IN — RE-RACE with a repaired judge, not "fire it"** |
| 17 | **Live-vehicle lever activation ingester**. `src/tac/pr130_lift/live_lever_activation.py` (2a83341b2) | ALREADY-IN for ft1 only | the (c) DSL door. | It already solved the §1.2 staleness for the fold-back trainer by deriving levers by AST at call time ("it cannot describe a vehicle the code does not have"). **It does not cover `ddm_qbt1_qbflow_trainer.py`.** Extending `LIVE_TRAINER_PATH` to a set is the cheapest available repair of the DSL leg. | **FOLD-NOW (apparatus)** |
| 18 | **Directional / curvelet basis −48% d_seg** | **SUPERSEDED** | — | The −48% was measured on **circular SYNTHETIC GT with an oracle orientation** at n96 (`curvelet_directional_basis_dseg_reduction_v1` anchor 0, whose own domain field says "do NOT consume as a realized lever"); the realized transfer is **170–350× off** and the matched realized A/B was UNCOMPUTABLE (the directional arm crashed). CLAUDE.md's 2026-07-27 routing note makes self-orient default OFF. Row 9 is the survivable residue: a form to RACE, never a number to cite. | **SUPERSEDED** |
| 19 | **Laguerre-OT / Menon class-prior head offsets** | **SUPERSEDED (measured negative, n600, twice)** | a caution for qbt1's `derive_balanced_class_weights` (:686, consumed ONLY as `birth_class_weights` at :2601 — the CE-birth stage, not the margin stage) | Full n600, realized-through-R on the frozen CPU SegNet: `no_offset` **0.0031436** < `menon` **0.0033119** (+1.68e-4) < `ot_newton` **0.0048921**, with the OT solve itself EXACT (max mass err 2.82e-11, 8 iters). The flip-mass reformulation is far worse: `flip_weighted` **0.0196734**, `flip_median` **0.0215612** (6×), Lane per-class 0.6129 vs 0.21182. **Matching GT class FREQUENCY is measured to hurt realized d_seg.** qbt1's inverse-frequency balanced weights are the loss-weight cousin of the Menon offset — a different mechanism aimed at the same quantity, and confined to the birth stage where "make every class exist" is the actual objective, so the negative does not transfer as a verdict. Treat as a WATCH row: race the CE-birth class weighting against uniform once, and never carry frequency-matching into the margin stage. | **SUPERSEDED / WATCH** |
| 20 | **Island seed + containment protection** | **SUPERSEDED for born vehicles** | — | The law is real (n600, [contest-CPU advisory]: lane erased-recall **0.5646** → seeded **0.9304**, seed-birth gain **+0.3658**; containment gain +0.157, wash 0.9531 vs 0.7961). But vh1 §1 row 11 measured its PRECONDITION ABSENT on the successor vehicle from existing telemetry: Lane components **20 → 476** (GT 985) and Movable **68 → 105** (GT 134) birthed **unaided**, smallest surviving lane component 89 px → 1 px. qbt1 carries its own birth gate (`realized_ce_birth_objective` :703, `evaluate_birth_verdict` :803, existence-majority mode). Non-reactivation, scope: formulation-on-witness-INR. | **SUPERSEDED** |
| 21 | **Generator + honest residual as a rate replacement** (v8 complete-lossless; the curve-relative offset coder) | **SUPERSEDED by its own measurement, and re-measured today** | — | v8's own anchor 3 records the negative: the curve-relative signed-offset coder did NOT deliver the conjectured 4–5× residual shrink; complete stays ≈0.135–0.140 = "a wash-with-frontier". In bytes, v8 complete **210,255 B vs the HPAC's 113,411 B** for the same exact field = **1.854×**. gc1 (2026-09-03) measured **401,537 B vs 113,411 B = 3.5×** on the born form. Two generator families, eight weeks apart, same sign and the same verdict. | **SUPERSEDED (and CORROBORATED)** |
| 22 | **Store-side targeted waterfill paste** | **SUPERSEDED (scope-guarded)** | — | n600 through-R: realized best = **FULL-STORE**; targeted paste dominated at realization efficiency **η_R ≈ 0.34992**; residual_realized_tau12 0.003088. The equation's own domain field states the store-side negative **does NOT transfer to witness INR capacity allocation** — which is why row 1 (a TRAINING allocator) survives while the paste does not. Sister receipt: lr2 §0, every offset-shipping rung loses (η ≤ 0.12). | **SUPERSEDED** |
| 23 | **Distillation as a finishing-stage lever** (hr1 §"DW1 and QA75/KD-#74") | **SUPERSEDED (measured refusal)** | ft1's never-fired `--distill-weight` / `--distill-max-seg` | Six-form mini-race on the TR1 E2 endpoint selected attack-weighted Hinton KD (T=2, weight 100). The 12-epoch winner reached **0.0050507** and transiently 0.004995 — then the matched long window REVERSED: plain control ended **0.0051147** (slope −6.80e-6/ep), the distillation arm was refused after 29 epochs at **0.0054967** (slope **+1.37e-5/ep**), a deficit of **3.82e-4 = 12.8× the measured 2.99e-5 noise floor**. Head-range relaxation did not rescue it. **The second, more portable lesson: a short mini-race selected the WRONG objective** — binding on ft1's stop rule and on qbr1's six-cell selection window. | **SUPERSEDED / CAUTION** |
| 24 | **Closed-scorer viscosity-HJ / SE(3) / entropy KKT stationarity**; `certified_layer_precision_waterfill_v1`; `frozen_scorer_fisher_pullback_metric_v1`; `score_marginal_lagrange_multipliers_v1` | **MEMO-ONLY-NEEDS-CODE** | — | `closed_scorer_viscosity_kkt_stationarity_v1`'s only anchor reads `predicted: NUMERIC_WITNESS_OWED`, `empirical: NOT_MEASURED`, axis "[DERIVED; archive witness owed]". The other three carry `empirical_anchors == []`. These are the era's promissory notes; none is a lever. Do not cite them as measured. | **MEMO-ONLY-NEEDS-CODE** |
| 25 | **hr1's other two arms — counted low-rank adapters, and joint token+renderer descent** | **MEMO-ONLY-NEEDS-CODE** (`execution_allowed=false`; never run) | ft1 runs arm 2 of hr1's four | hr1 (2026-08-11) specified a full-n600 four-arm race under one common receiver: frozen renderer / full renderer fine-tune / **counted low-rank adapters** / **joint token-plus-renderer descent**, with every learned treatment seeing the frozen scorers only after the camera-resolution uint8 cliff and the complete resize path. Its collapse falsifier did NOT fire: the retained PR135 renderer disagrees with its own semantic token plane at **d_seg = 0.00029639352578669786** (fidelity 99.9703606474%, below the 99.99% threshold). ft1 is arm 2. Arms 3 and 4 are designed, gated, and unrun. | **MEMO-ONLY-NEEDS-CODE** |

## §3 THE FOLD-NOW ROWS WITH THEIR EXACT CODE PATHS

Seven rows, each a named edit at a named line, none of them a new invention:

1. **Per-pixel margin weight** — port `_live_margin_weight`
   (`experiments/train_witness_realized_through_R_mlx.py:1086`, MLX) to torch and multiply it into
   `expected_flip_margin_loss`'s per-pixel term at
   `experiments/ddm_qbt1_qbflow_trainer.py:538` (`per_sample = torch.sigmoid(-margin / tau).mean(dim=(1, 2))`
   → a `(w * p).sum() / w.sum()` reduction). Preserve BOTH invariants that make it honest:
   **mean-1 normalization** (re-allocates the budget, never adds to it) and **stop-gradient** (the
   margin map is an allocation PRIOR; a differentiable weight lets the field game its own margin
   instead of fixing the flip — a NO-FAKE boundary the original author wrote into the docstring).
2. **AA-SDF footprint render** — `src/tac/boundary_math/aa_sdf_observation_render.py` into qbt1's
   render at `experiments/ddm_qbt1_qbflow_trainer.py:477-479` (before
   `roundtrip_to_camera_uint8_ste` at :495). ss=2 box supersampling is 4× the render points; an
   analytic footprint is the cheaper form the module already implements.
3. **Chan-Vese area cap** — add the one-sided `relu(A_c − A_c^GT)²` penalty beside
   `dual_ascent_margin_constraints` (`:593`) and the per-class block in `joint_objective` (`:663-681`),
   with `λ_c = W_birth/(δ·A_GT_c)` derived live from `derive_balanced_class_weights`' own bincount
   (`:686-702`) — no hand-typed area constants.
4. **Per-edge margin scale** — replace the scalar `tau` in `expected_flip_margin_loss` (`:538`) and
   `per_class_expected_flip_margin_loss` (`:564`) with `tau · ‖w_c − w_c'‖` from the frozen
   `segmentation_head.0.weight` (weights sha `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`).
5. **Chroma-first routing** — `src/tac/boundary_math/chroma_boundary_match.py` (currently imported by
   nothing but its own equation module) into ft1's loss, plus a luma-change penalty; the mechanism is
   `upstream/frame_utils.py:65-72`.
6. **Satisficing cap** — `tac.witness_dsl.curriculum_dsl.MarginBandSatisficing` +
   `tools/measure_delta_R_noise_floor.py` re-run at n600, as a CAP on row 1's allocator
   (`fh1_margin_satisfice_cap` is the designed stub).
7. **Persistence/clDice** — torch port of
   `src/tac/boundary_math/persistence_topology_loss.py` into qbt1's Lane term.

Plus the apparatus row: extend `LIVE_TRAINER_PATH` in
`src/tac/pr130_lift/live_lever_activation.py:51` from a single path to the live SET
(`+ experiments/ddm_qbt1_qbflow_trainer.py`), so the born trainer's levers stop being invisible to
both the ledger and the costate organ's duty head (`ddm_costate_organ.py:555`).

**Composition caution, stated up front:** rows 1, 3, 4, 6 and 16 all act on the same per-pixel
gradient budget. m164 measured union ≠ sum of legs at 3.705×, and hr1's DW1 receipt shows a short
mini-race picking the wrong objective. Race them singly against a matched control before any
composition.

## §4 APPEND-ONLY — ADDITIONS TO `ddm_fb1_foldback_program_20260903.md`

fb1's map is correct and its seven discovery→lever rows stand. These are the rows the era supplies
that fb1's table does not carry. MAIN folds them; I have not edited fb1.

- **fb1 item 3 needs a correction and a split.** fb1 names the lever "Fisher/UNIWARD margin
  surrogate as a hard-site weight". Two measured facts change it:
  (a) **The UNIWARD/texture half is DEAD, measured.** `margin_saliency_reachability_replaces_texture_proxy_v1`:
  the texture proxy is orthogonal to through-R reachability at chance (`texprox_vs_sR_pearson −0.033`,
  top-5% Jaccard **0.024** vs chance **0.026**) and it mildly MISDIRECTS (`texprox_vs_grad_margin −0.215`).
  Do not fold UNIWARD.
  (b) **The Fisher half needs no Fisher computation.** `frozen_scorer_fisher_curvature_margin_colocation_v1`
  (n=96, byte-faithful: argmax mismatch 0.0, margin delta max 4.8e-7) measured Pearson(curvature, −margin)
  **0.978** in the 2px band (all-px 0.814, Spearman 0.908), Fisher trace vs spec-norm 0.997, boundary
  anisotropy 9.56, **96.8% of flip mass inside the 2px band**. The margin field the loss already computes
  **is** the Fisher surrogate. And the decisive ranking is measured against real flips: the margin weight
  `w = exp(−m/τ)` scores **AUC 0.991 / flip-mass-in-top-5% 0.973**, while the S_R reachability multiplier
  scores only **AUC 0.767 / 0.221**. So: fold the **margin weight** (row 1), treat S_R as a SECONDARY
  multiplier at best, and drop UNIWARD.
  (c) fb1's actual open variable — the **coded-price** weight — is a THIRD variable, it is already
  built for one door (`src/tac/pr130_lift/band_objective.py`), and **fb1's "unmeasured as a training
  weight anywhere" is now false**: it FIRED at α=1.0 on 2026-08-17 and was measured on 2026-08-22
  (`.omx/state/lever_activation_ledger.jsonl:278,:319`) — the 600 activations rotated the realized step
  **66.3°** but best exact-seg **stayed at init** because the residual judge was structurally
  underpowered, on the MPS training-signal axis, FORMULATION scope (`ddm_jr1_...20260817.md`). The
  correct fb1 status is RE-RACE-WITH-A-REPAIRED-JUDGE, not unmeasured. Race margin-magnitude against
  coded-debt; do not assume they agree.
- **fb1 item 5 ("class-routed capacity/atoms — parked") has landed code and n600 receipts.** The v8
  carrier family (row 10) is exactly the class-matched atom set gc1's addendum asks for, with
  bit-exact real-coder measurements: MyCar IoU 0.994, Lane band d_seg 0.00087, dominant-only rate
  0.061 (91,611 B). Its COMPLETE form is closed (row 21) — but as an atom family for a born generator
  it was never raced.
- **fb1 item 6 (the full-pipeline fold) inherits a measured schedule alternative.** The costate organ
  beats the hand-scheduled curriculum by **−27.4% on ∫d_seg·dep** across all three walk-forward replay
  models (row 8). fpc3's line "its objective should be the fold-back objective, not the PR130
  curriculum" has a landed, backtested candidate.
- **NEW row for fb1's map: the render's own sampling is a training lever.** fb1 attributes 95% of seg
  error to the render path (td1/rt1/mst1) but proposes only a loss change. The era measured the
  *sampling* half at n600 on the exact grid the born trainer uses: point 0.0054940 → footprint
  0.00085986 at g384 (row 2). That is a change to `forward`, not to the loss.
- **NEW row: two of ft1's four never-fired levers have a v-era measured refusal.**
  `--distill-weight` / `--distill-max-seg` are 2 of the 4 never-fired levers on the live fold-back
  trainer, and hr1's DW1 receipt refused that family at the finishing stage with a deficit of 12.8×
  the noise floor (row 23). The other two (`carrier_rank_penalty`, `carrier_tensors`) have no such
  precedent and are genuinely unmeasured.
- **NEW row: ft1 is arm 2 of a four-arm race that was designed and never run** (row 25). Arms 3 and 4
  (counted low-rank adapters; joint token+renderer descent) are hr1-specified with
  `execution_allowed=false`.
- **NEW row: the loss-space hold is closed; the hold must be realized.** hr1's JD-line precedent:
  JD1's loss-space hold allowed live `d_seg` to worsen from ≈0.00357 to ≈0.00599 while pose improved.
  ft1's and qbt1's stop rules must latch the Seg floor in REALIZED space, never in loss space.
- **NEW row: the pose ceiling correction is era-consistent.** SPEC_v75's POSE LAUNCH GATE (d_pose ≈ 1.79
  ⇒ ≈4.24 of S) and gs3 ADDENDUM 4 item 3 (same-object promotion pose ceiling ≈ 1.694e-5, not 1.25e-4)
  are the same discipline at two operating points: **never carry a pose number across vehicles.**

## §5 GESTALT LINE

**The era does not move where sub-0.12 lives; it proves the location twice and hands over the two
levers gs3 says are still open.**

The proof: on 2026-06-26, FEED-bq fitted `d_seg = E + A/N^α` with **α = 1.504** on two clean converged
realized points of the HNeRV/witness family and concluded "**pure capacity scaling does NOT reach
sub-0.15**; BASIS-CHANGE MANDATORY", with `N_opt = 219,719 params → S = 0.1931` as the family minimum.
On 2026-09-03, gc1 fitted a local exponent of **−1.228** on the born generator (1.59× bytes → 1.84×
fewer mismatches, crossing extrapolating to a **9.62× cap** packet) and concluded CAPACITY-CLOSED.
**Two vehicles, two independent exponents, one verdict.** The same convergence appears on the rate
side: v8's complete generator+residual measured **210,255 B vs 113,411 B for the HPAC on the same
field (1.854×)** in July; gc1 measured **401,537 B vs 113,411 B (3.5×)** in September. And on the
serialize-vs-generate side, `realization_necessity_preimage_per_stratum_v1` measured on 2026-07-15
that a DP-simplified per-class contour description at ε=1 px, int16-delta-packed + brotli q11, costs
**K_ladder_edge_bytes_n600 = 143,552.5 B** — **1.040× the entire 137,986 B cap**, for the boundary
curves alone, before any renderer, pose carrier, or residual (K/H = 0.47370; road+lane = 0.61 of H).
That is an independent v-era receipt for the August "explicit boundary floor ≥ archive ⇒ GENERATE,
don't serialize" law.

What the era adds to gs3 is not a new door but the two cures FEED-bq pre-registered and quantified
against exactly this wall: **(1) a 3.55× fixed-byte d_seg downshift from the margin-weighted annulus
re-route** — landed code (row 1), never consumed by the born trainer; and **(2) a basis whose
manifold exponent α ≥ ~2.9** — the class-matched-form question gc1 and gf2 left open, with the v8
carrier family (row 10) and the parabolic frames (row 9) as the built candidates. Applied to the born
field, cure (1) alone takes d_seg 0.01299522 → 0.003661, still **26.8×** above qn1's 1.3646784205e-4
target; the total needed factor is **95.2×**. So the era supplies real levers and no shortcut: the
bar stays qn1's — a born field wrong on ≲16,000 of 117,964,800 sites at ≤137,986 B with pose
≤1.25e-4.

One honest cross-era correction in the other direction: the era's "render-legibility ceiling 0.00086"
(`aa_dseg_floor_g384`) is **not** a ceiling for today's frontier object. afr1's shipped semantic
renderer sits at d_seg 2.0139e-4 — **4.27× BELOW** that v-era floor. The floor was scoped to the
SDF-raster render family; the PR130 lineage beat it. Cite it as a bound on the born SDF-style raster
(row 2), never on afr1.

## §6 RECALL EVIDENCE (what I consulted before concluding)

- `tools/graph_memory_recall.py "v7 v75 v8 v9 v10 v11 spec era levers kernels laws"` (39,744 nodes /
  162,651 edges) — surfaced `ddm_vh1_v8v9v10_harvest_20260730.md`, which I read IN FULL before writing
  a single row, so this unit does not repeat the 2026-07-30 harvest. vh1's consumers were TR1 and the
  BR-A..D branches (all retired); vr1's are qbt1 / ft1 / fpc3 / the DSL / the equations leg.
- Primary artifacts opened at source: SPEC_v75, SPEC_v8, BUILD_SPEC_v10, the v11 codex findings, hr1,
  lr2, the sub015 DAG blocks FEED-bp (:1205) / FEED-bq (:1215) / FEED-br, `upstream/modules.py`,
  `upstream/frame_utils.py:51-72`, `experiments/ddm_qbt1_qbflow_trainer.py`,
  `src/tac/pr130_lift/{train_semantic_quantized_resumable,band_objective,live_lever_activation}.py`,
  `experiments/train_witness_realized_through_R_mlx.py:1086`,
  `src/tac/witness_dsl/{lever_registry,fh1_adapted_force_levers_20260731}.py`,
  `src/tac/ddm_costate_organ.py`, `codex_findings_ddm_gc2_oracle_gap_closure_20260724_codex.md`.
- Registry: `tools/list_canonical_equations.py --json` (465 rows), filtered and dumped anchor-by-anchor
  for the 25 equations cited. Every number in §2 came out of that dump or out of a file I opened.
- Live-vehicle greps (the basis of §0): `boundary_math` / `witness_dsl` / `canonical_equations`
  importers of `ddm_qbt1_qbflow_trainer.py`, `ddm_qbr1_*`, `ddm_ft1_*`, `src/tac/semantic_pipeline/`,
  `src/tac/pr130_lift/`.
- Carried forward, not re-litigated: CLAUDE.md's 2026-07-27 self-orient routing note (row 18); the mf1
  Morse–Smale retirement with persistence surviving on F1b (row 7); `ddm_ww1_walls_that_werent`,
  `ddm_rn1_n600_reopen_sweep`, memory m34, m44, m164, m110.

## §7 NEXT_IF_RESUMED

1. **Race row 1 against row 16 on the ft1 door** — margin-magnitude weight vs measured coded-debt
   weight, matched control, single lever at a time. Both are landed; neither has fired. Report B/H/W.
2. **Price row 2 with a $0 read** — render 8 born pairs point-sampled and ss=2 at 384×512 through the
   real R and the frozen SegNet; the delta is the born vehicle's share of the 6.389× the era measured
   on the real frame. No training required.
3. **Land the row-17 apparatus fix** (`LIVE_TRAINER_PATH` → a set) so the born trainer's levers enter
   the ledger and the costate duty head; today they are invisible to both.
4. **Re-run `tools/measure_delta_R_noise_floor.py` at n600** (row 6) — the tool exists; δ_R 0.0196 is
   an n96 number and SPEC_v75 §8B says re-run, never rebuild.
5. **Fold row 3 into the qbt1 constraint set** — the recall-only dual is the exact configuration the
   Chan-Vese law says floors the majority class.
6. **Hand rows 8, 9, 10, 11 to the post-burn adjudication**, not before: firing a schedule or form
   change now confounds the running discriminator.

## §8 DEAD-ENDS (do not re-open silently)

- **UNIWARD / texture-cost as a site weight** — orthogonal to through-R reachability at chance and
  mildly misdirecting. Also `uniward_standalone_no_op_on_bitstream_dominated_by_sli1_decoder_cost_v1`
  has zero anchors. Closed.
- **Class-frequency head offsets (Menon / Laguerre-OT) and their flip-mass reformulations** — n600
  measured negative, reproduced at n2/n6/n24/n48/n600 (row 19).
- **Island seed/containment protection on a vehicle that births unaided** — precondition measured
  absent (row 20).
- **Distillation as a finishing stage** — measured refusal at 12.8× the noise floor (row 23).
- **Store-side targeted waterfill paste / any offset-field-shipping realizer** — η_R ≈ 0.35 and
  ≤ 0.12 respectively (rows 22, lr2 §0).
- **The −48% directional-basis figure** — synthetic circular GT with oracle orientation; never a
  production routing input (row 18).
- **Complete generator+residual as a rate replacement for the HPAC** — measured 1.854× (v8) and 3.5×
  (gc1) worse for the same exact field (row 21).
- **Morse–Smale language for the boundary** — retired by mf1; persistence survives on F1b only.
- **Citing any v-era pose magnitude on this vehicle** — SPEC_v75 §1 and gs3 ADDENDUM 4 both say no.

---

Pointer honesty: this unit measured nothing on the scorer, built nothing, and moved nothing. Its
product is a table other arms consume.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — UNMOVED.
