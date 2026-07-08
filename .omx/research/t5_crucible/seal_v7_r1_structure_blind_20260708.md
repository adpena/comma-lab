# SEAL v7 R1 — STRUCTURE ROUND (allowlist-blinded)

> Contract: MECHANICAL BLINDING by allowlist. Phase-1 derives the optimal training-program
> SHAPE from physics ALONE, committed BEFORE any Phase-2 read of the authored crucible_v7 doc /
> witness_autoconfig / feature modules. The Phase-1 commit timestamp IS the blinding proof.
> Every Phase-2 divergence between the blind derivation and the authored v7 is a REVISE finding
> by contract — never rationalized.

## Store separation (the evidence)

- **PHASE 1 reads (allowlist ONLY):** `CLAUDE.md` (session context) · `docs/triality_dag_dsl_equations_deepmath.md` ·
  the canonical equations via `tools/list_canonical_equations.py --json` (all 513 rows, id+summary) ·
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` metadata (shapes only:
  `gt_f0/f1 (600,874,1164,3) uint8`, `lstars (600,384,512) int64`, `margins (600,384,512) f32`,
  `gt_poses (600,6) f64`, `n_pairs () int64`).
- **PHASE 1 did NOT read:** the authored `crucible_v7` doc, `witness_autoconfig.py`, any feature
  module, any position/seal/synthesis doc, the ORCHESTRATION_LEDGER, any other `.omx/research` file.
- **PHASE 2 reads (added after the Phase-1 commit):** `crucible_v7_authored_20260708.md` ·
  `src/tac/witness_autoconfig.py` (v7 derive) · the 5 feature-module docstrings.

---

## §PHASE-1 — the blind structural derivation (physics only)

### P1.0 The one-line shape

The optimal program is the numerical integration of **ONE continuous τ=ε=ħ anneal of a single
frozen-scorer-Fisher-metric level-set action**, run on a **fixed directional step-native basis set
BEFORE training**, where **every phase transition is SENSOR/DERIVED-triggered (readiness), never
clock-triggered**; island-birth protection runs **concurrently with** the anneal (not after);
the terminal phase is a **warm-started Muon spectral finisher** (LR→floor + Polyak averaging,
optional full-P curvature-aware head solve); **pose and rate compose at byte-close, not as
curriculum stages**. The "stages" of a PR95-style schedule are, physically, *regime labels on one
continuous flow* — only the optimizer-geometry switch is a true discontinuity.

### P1.1 Why it is ONE continuous flow, not a stage list

The action is `witness_unified_action_fixed_fisher_background_v1`: contest S = one variational
action S_τ stationary in the **frozen-scorer Fisher pullback metric** G
(`frozen_scorer_fisher_pullback_metric_v1`), fixed background, no back-reaction. Descent in that
metric is natural gradient (`dm3_natural_gradient_steepest_descent_under_norm_v1`); CE is exactly
the mirror-descent/Bregman flow of the categorical-entropy potential
(`ce_softmax_mirror_descent_natural_gradient_v1`). The curriculum is therefore a **homotopy of
relaxations of ONE energy** — graduated non-convexity — parametrized by a single scalar:

- τ = ε = ħ is simultaneously the Maslov/tropical Planck constant, the Modica–Mortola interface
  half-width, and the mirror-descent temperature (`tau_eps_hbar_one_dequantization_two_scales_v1`).
- The τ:1.0→0.05 anneal **IS** the (+,×)→(max,+) semiring/dequantization limit
  (`maslov_dequantization_bound_v1`: softmax_τ→argmax costs ≤ τ·ln5) and the Γ-limit to the weighted
  perimeter of the argmax partition (`multiphase_modica_mortola_perimeter_gamma_limit_v1`).

⇒ **CONTINUOUS backbone:** τ(t) annealed coarse→fine. The apparent CE→softplus→…→finisher "stages"
are which action terms dominate at which τ-scale — one flow, not resets.

⇒ The ONLY genuine discontinuity is the **descent-norm switch** (AdamW diagonal → Muon spectral),
because it changes the metric of steepest descent itself (`dm3_...`: Adam=diagonal, Muon=spectral).

### P1.2 What is CONTINUOUS vs STAGED

**Continuous (schedules on the one flow):**
- τ(t) anneal (the backbone). Coarse→fine.
- eps(t): CFL edge tracker — DERIVED, not fixed. `adaptive_eps_cfl_edge_tracking_v1`:
  eps(t)=clamp(|c_a(t)|·√(η·λ_eik/8)·(1+margin),floor,upper); a FIXED eps falls below the RISING
  CFL edge as sharpening grows ⇒ eps must be adaptive.
- EMA window: `ema_window_pi_group_v1` — ρ does NOT transfer; π_ema=window/stage does. Continuous
  within a regime, re-scaled at the finisher (early=tracking, finisher=Polyak ~0.1–0.3× stage).
- The seg loss / natural-gradient flow itself; logit-adjustment; boundary-weight field.

**Staged (discrete, each a SENSED trigger — see P1.3):**
- Nucleation gate (CE forms every class + plateaus) → open the τ-anneal.
- Optimizer-geometry switch (AdamW → Muon finisher).
- Terminal head solve / stop.

### P1.3 Every transition is a SENSOR/DERIVED trigger, NOT a clock (the load-bearing derivation)

This is the sharpest physics result and the one a PR95-echo schedule gets structurally wrong.

1. **Open-anneal trigger = per-class nucleation readiness, not an epoch.**
   `curriculum_handoff_critical_nucleus_v1`: CE→τ is a per-class READINESS trigger; MCF erodes
   sub-nucleus classes, so CE must form EVERY class + plateau FIRST. Fire when: all 5 classes
   nucleated ∧ CE plateaued.
2. **Muon-engage trigger = partition-formed ∧ τ anneal-COMPLETE, never before nucleation.**
   `muon_switch_conditioning_criterion_v1`: PR95 stage-8 is the right ORDER, wrong CLOCK — engage is
   a trigger (partition formed + τ plateau). `anneal_truncation_fixed_clock_defect_v1`: a fixed-clock
   anneal + early consumer freeze silently truncates τ/β endpoints (β 3.18/4.0, τ 0.216/0.05 at a
   premature Muon freeze) ⇒ the finisher entry MUST carry an **anneal-complete precondition**.
   `finisher_transient_budget_and_meat_exhaustion_v1`: finisher viable ONLY if recovery τ_e <
   remaining budget AND warm anneal-complete entry (else it ends ABOVE its entry best).
3. **Stop trigger = power-law-tail meat extrapolation, not window slope.**
   `weak_kam_powerlaw_tail_exit_v1`: the late tail is power-law (exp floor broken ~22%);
   exponential/window-slope plateau detectors fire EARLY. Exit on extrapolated remaining meat under
   a+b·t^−α. (`dseg_stretched_exponential_anneal_trajectory_v1` gives the coarse global fit; the tail
   is the binding stop signal.)

### P1.4 Island-birth runs CONCURRENTLY with the anneal (not a late stage)

Arithmetic necessity: `islands_necessity_floor_big3_only_v1` — un-born islands (movable 44.8% +
lane 19.1%) carry 63.9% of d_seg; big-3-only floors at ~0.00215 ≫ 0.00092 T_3 need ⇒ island birth
is NECESSARY, not optional. And `mcf_minority_erasure_inevitability_v1`: the perimeter-gradient flow
is motion-by-mean-curvature, so high-curvature thin Lanes erase FIRST and inevitably (95.7% of
smoothing cost is Lane). ⇒ if island protection is bolted on AFTER the anneal, the anneal has
already erased them. Protection must be ACTIVE THROUGHOUT the anneal:
- `island_finest_scale_protection_survival_v1`: sparse GT-appearance seed on the self-detected
  island band births the class through the frozen SegNet; freeze protected pixels under bulk wash.
- `persistence_topology_cldice_betti_island_recall_v1`: soft-clDice(β0/β1) + persistence-weighted
  recall births the finest-scale erasure tail the topology-blind CE drops.
- `logit_adjustment_class_prior_law_v1`: logits_c += τ·log(prior_c) shifts gradient to rare
  Lane/Movable classes at ZERO archive bytes (deployed argmax reads raw logits).

### P1.5 The representation is set BEFORE the flow (basis-match is prior to capacity)

- `curvelet_directional_basis_dseg_reduction_v1`: directional all-class basis = −48% d_seg vs
  isotropic; basis-match is PRIOR to capacity.
- `anisotropic_basis_along_tangent_frequency_deficit_v1`: freq_along ≤ 8 cyc vs dash ~25 ⇒ 3.2×
  deficit ⇒ the basis literally cannot represent dashes; they erase finest-first at ANY capacity.
  Fix = along-tangent frequency, not more epochs. `anisotropic_basis_two_regime_allocation_v1`:
  freq_along = √(freq_across) on the free rule-118 band; 26 when carrying the dash comb.
- `step_native_activation_edge_optimality_v1` + `hosc_activation_saturation_trainability_v1`:
  step-native charts put all error at the edge (no Gibbs) for a piecewise-constant argmax; hosc is
  trainable ONLY with siren-init OR β-anneal 1→4; fixed β≥4 random-walks. ⇒ step_basis, or
  β-annealed hosc — never fixed β=4.
- `residual_manifold_intrinsic_dim_whitney_v1`: mod-dim from measured intrinsic-dim m via Whitney
  2m+1 (~8→17–19), NOT inherited mod-16 (under-embeds).
⇒ Basis (direction, along-tangent freq, step-native activation, mod-dim) is a PRIOR fixed at build
time, not a curriculum stage.

### P1.6 What to DROP from a PR95-echo schedule (defects)

- **l7 / L∞ sharpening:** `l7_linf_sharpening_defect_in_smoothing_flow_v1` — l7 moves d_seg only
  −0.00012; L∞ sharpening decouples from d_seg inside a viscosity/smoothing flow. DROP.
- **L28 channel offsets:** `l28_channel_offset_does_not_transfer_to_levelset_witness_v1` — ancestor
  offsets RAISE witness d_seg. DROP.
- **Fixed-clock anneal denominators / early consumer freeze** (P1.3 #2). DROP for sensed triggers.
- **Cold Muon start / flat Muon LR:** `muon_finisher_schedule_warmstart_and_lr_anneal_v1` — cold
  start spikes d_seg +0.000357; flat LR plateaus (NS fixes magnitude). REPLACE with warm-start
  momentum (v←AdamW m) + cosine LR→floor; `rewarmup_beta2_memory_window_v1`: LR rewarmup must span
  1/(1−β2) or full-LR steps divide by unconverged moments.
- **Focal γ:** `focal_gradient_concentration_v1` — γ*=0 (weak bulk-boundary lever). Not a stage.

### P1.7 Terminal phase

Warm-started **Muon spectral/Stiefel finisher** (`dm1_stiefel_isometry_rank_preservation_v1`: WᵀW=I
preserves rank by construction, byte-free), LR cosine→floor, finisher-EMA = Polyak averaging
(~0.1–0.3× stage window). Then, because the head chart is near-quadratic but NOT 2nd-order-exhausted
(`gn_hessian_spectrum_indefinite_at_ema_best_v1`: indefinite Ritz, |λ_−|=2.65×λ_max at EMA-best), an
optional **full-P curvature-aware head TerminalSolve** — full-P ONLY: `quadratic_head_chart_subset_
solve_gap_v1` shows K=8 subset transfers +5.1% WORSE. STOP on the P1.3 #3 power-law-tail criterion.

### P1.8 Composition order (what composes with what, in order)

1. **Build-time PRIOR:** directional step-native basis + along-tangent freq + Whitney mod-dim
   (P1.5). Also the openpilot ego-ξ physical prior as INIT/source for both axes
   (`openpilot_unified_physical_prior_both_scored_axes_v1`, rule-118 free, net-S gated).
2. **Nucleate:** CE / mirror-descent, AdamW, until all-class nucleation + plateau (P1.3 #1).
3. **Anneal τ continuously** (coarse→fine) WITH, concurrently: island-birth protection (P1.4),
   logit-adjustment, boundary-weight (w≈0.2, `boundary_distance_weight_calibration_v1`), adaptive
   eps (P1.2). l7/L28/focal excluded (P1.6).
4. **Finisher:** on anneal-complete ∧ partition-formed → warm-started Muon, LR→floor, Polyak EMA,
   + train-time weight-entropy rate penalty (`weight_entropy_rate_in_loss_lever_v1`: −19.6% byte
   floor) so rate is shaped WHILE finishing.
5. **Terminal:** optional full-P head solve; stop on power-law-tail meat exhaustion (P1.7).
6. **Byte-close composition (NOT stages):** pose = stored-ξ carrier
   (`pose_sqrt_concave_coupling_sidecar_v1`, `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`,
   `warp_real_luma_frame0_pose_carrier_dpose_v1`) — dual-use ξ (`pose_ego_screw_twist_identifiable_
   up_to_affine_v1`) also warps the partition, so it feeds d_seg source AND is the pose carrier;
   rate = reverse-waterfill / grammar-rev2 at byte-close (`rate_mdl_cosmological_constant_reverse_
   waterfill_v1`). Both are the SECOND wall after d_seg (`pose_second_wall_t1_feasibility_bound_v1`).
7. **Cross-cutting invariants (all phases):** resumable + per-stage EMA-shadow checkpoints; the
   costate monitor runs continuously and can trigger rollback (`costate_lambda_marginal_ds_v1`:
   WATCH/EROSION backtested on #205); every load-bearing verdict measured THROUGH byte-close
   (axis-9 / POWERPLAY Correctness Demonstration).

### P1.9 The blind shape, compressed to invariants (for the Phase-2 table)

- I-1 ONE continuous τ=ε=ħ anneal backbone; only the descent-NORM switch is a true discontinuity.
- I-2 Transitions are SENSED (nucleation-readiness / anneal-complete / power-law meat), never clocked.
- I-3 Island-birth protection CONCURRENT with the anneal (MCF erases minority first), not late.
- I-4 Basis (direction + along-tangent freq + step-native + Whitney mod-dim) fixed BEFORE training.
- I-5 Finisher = warm-started Muon (v←AdamW m) + cosine LR→floor + Polyak EMA + β2-window rewarmup.
- I-6 Terminal = optional FULL-P curvature-aware head solve; stop on power-law tail, not window slope.
- I-7 DROP l7, L28, focal-γ, fixed-clock anneal, cold-Muon/flat-LR.
- I-8 eps + EMA-window are DERIVED/stage-scaled, not fixed constants.
- I-9 Pose (stored dual-use ξ) + rate (reverse-waterfill + train-time weight-entropy) compose at
  byte-close, not as curriculum stages; both are the second wall; ξ is also a both-axes prior.
- I-10 Cross-cutting: resumable per-stage EMA checkpoints + continuous costate rollback + axis-9
  through-byte-close for every load-bearing verdict.

<!-- PHASE-2 APPENDED BELOW THIS LINE AFTER THE PHASE-1 COMMIT -->
