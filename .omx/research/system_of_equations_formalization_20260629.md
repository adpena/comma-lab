# Witness unified-action SYSTEM OF EQUATIONS — formalization (2026-06-29)

**Operator directive (verbatim):** *"We want to formalize our system of equations and all that math and
ensure no signal loss and integration with canonical equations and all related"* + fold-ins: *"remember the
theta-star docs and research too and calibration — all is related signal."*

**Status:** 13 canonical equations (E0 master + E1–E12 terms) registered into
`.omx/state/canonical_equations_registry.jsonl` (165 → 178 entries) via the canonical helper
`tac.canonical_equations.register_canonical_equation` (fcntl-locked, append-only). Builder:
`tools/register_witness_action_system.py` (reproducible). **NO exact score moved — this is
SYSTEM-INTELLIGENCE (means), not goal progress (ends). Pointer stays contest-CPU 0.19110.** Per CLAUDE.md
"Canonical equations + models registry" + "Results must become system intelligence" non-negotiables.

This formalizes the session's deep-math (DAG FEEDs gz→ih) as the operator's GR-style system of equations:
**the contest is ONE variational action `S_τ`, stationary in the FIXED frozen-scorer Fisher metric — matter
on a fixed curved background (NOT full GR: no back-reaction).**

---

## The system (E0 master; E1–E12 its terms)

| id | equation | tier / anchor |
|---|---|---|
| **E0** `witness_unified_action_fixed_fisher_background_v1` | `S_τ = 100·D_seg^τ + √(10·D_pose) + 25·B/N`; `δS_τ/δφ=0` in metric `G` | MASTER (theoretical; terms carry anchors) |
| **E1** `frozen_scorer_fisher_pullback_metric_v1` | `G(θ)=E_x[Jᵀ F_x J]`, `F_x=diag(p)−ppᵀ` (fixed background) | theoretical; validated by E3 |
| **E2** `dm1_stiefel_isometry_rank_preservation_v1` | `WᵀW=I ⇒ PR(M=code·Wᵀ)=PR(cov(code))`; `E_spec=−β·log[(tr C)²/‖C‖_F²]` | **MEASURED** (collapse PR 3.34→1.19, `[macOS-MLX research-signal]`) + **PROVEN by test** (commit 07dd971d8, `test_levelset_dm1_stiefel_entropy.py`, `[empirical]`) |
| **E3** `frozen_scorer_fisher_curvature_margin_colocation_v1` | `corr(tr F, −margin)_∂Σ=0.978`; margin = byte-faithful Fisher surrogate; anisotropy 9.56:1 | **MEASURED** `[macOS-CPU advisory]` (commit b0bee924e, n96, byte-faithful) — the live calibration baseline |
| **E4** `dm2_lane_ipm_polynomial_geodesic_v1` | lane = deg-3 IPM polynomial (1.28px), ~7× tangent/cross; annulus = geodesic | `[macOS-MLX research-signal]`; refine: Finsler (1612.00343) |
| **E5** `dm3_natural_gradient_steepest_descent_under_norm_v1` | optimizer = steepest descent under a norm (Adam=diag / Muon=spectral / SinkGD=doubly-stochastic); flow = Dykstra; curriculum = homotopy | theoretical framework |
| **E6** `indirect_rd_logloss_equals_information_bottleneck_v1` | seg log-loss remote-coding = IB exactly (Courtade–Weissman 1110.3069) | theoretical EXACT (surrogate); refine: posterior-coding (2410.09018), Wasserstein-Dice (1707.00478) |
| **E7** `rate_mdl_cosmological_constant_reverse_waterfill_v1` | rate = 25·B/N = Λ/MDL; reverse-water-fill; FREE-generic vs COUNTED-learned boundary | theoretical; refine: eikonal-SDF generator (Ballé), contour-coding (1604.08001), MDL (1006.3679) |
| **E8** `dm1_decisive_smoke_falsification_gate_v1` | BINDING ⟺ `PR(M)≥3.0 ∧ Δd_seg^adv≤−0.10`; else SYMPTOM | **FORMALIZATION_PENDING** (awaits the decisive $0 smoke — the live calibration anchor) |
| **E9** `pose_sqrt_concave_coupling_sidecar_v1` | `√(10·d_pose)`, marginal `5/√(10·d_pose)→∞`; ⇒ stored-target sidecar not carrier | `[macOS-MLX research-signal]` (realized d_pose 0.094→0.018) |
| **E10** `theta_star_eikonal_length_boundary_energy_v1` | `E_bdy=0.01∫(‖∇φ‖−1)²+0.001∫‖∇H(φ)‖` (eikonal+Chan-Vese length) = θ* operational terms of E0/E4 | `[macOS-MLX research-signal]` (live weights 0.01/0.001) |
| **E11** `decode_determinism_integer_arithmetic_v1` | bit-identical decode ⟺ integer/fixed-point arithmetic (Ballé); ≠ McMullen contraction | discipline / theoretical |
| **E12** `vcm_task_rd_headroom_above_floor_v1` | headroom `= S_achieved/S_floor`; `S_floor≈0.118`, `S_pointer=0.19110`, `T_3=0.15` | FORMALIZATION_PENDING target (VCM 2505.14980, lit-hunt-cited) |

**θ* unification (operator fold-in):** θ* (`theta_star_witness_lever_stack_and_variational_levelset_frame_20260627`,
campaign tasks #183/#184/#185, DSL `src/tac/witness_dsl` #189) is the **OPERATIONAL lever-stack instantiation
of E0** — the witness as the viscosity solution of the variational level-set PDE. E10 is the bridge term
(eikonal+length = the live boundary energy of the same action `S_τ`).

---

## No-signal-loss coverage ledger (DAG FEEDs gz→ih → equation)

| session finding (FEED) | equation |
|---|---|
| DM1 rank-collapse + multiplicative resonance (panel gz→hk) | E2 (+ E1) |
| DM2 lane = IPM-poly annulus (hv) | E4 (+ E10) |
| DM3 penalty-flow / per-group / Muon (hu) | E5 (+ E10) |
| 5-paper synthesis: SinkGD/Hebbian/Fisher-Rao/Attractor schedule (hw/hx/hy/hz/ib) | E5 (conditioning → E2) |
| latent-SDE FALSE-FRIEND (ib) | recorded as REJECTED transfer (no equation; E2 lens note) — preserved, not dropped |
| GR-research unified action (ia) | E0 + E1 + E6 + E7 + E9 |
| $0 co-location CONFIRMED ×3 (id, b0bee924e) | **E3** |
| optimizer design: Stiefel isometry byte-free + decisive smoke (ic) | E2 + E5 + **E8** |
| McMullen review (ie): paradigm validated, NOT a mover; contraction nugget | E11 cross-ref (distinct from decode-determinism) |
| lit-hunt H1 whole-system / H3 generator (ig): posterior-coding, eikonal-SDF, VCM headroom | E6 / E7 / **E12** |
| lit-hunt H2 boundary (ih): Finsler, Wasserstein-Dice, contour-coding/MDL | E4 / E6 / E7 |
| DM1-fix BUILD landed, PR identity proven (07dd971d8) | E2 (second anchor) |
| θ* frame + campaign #183/#184/#185 + DSL #189 (operator fold-in) | E0 + E10 |

Every measured finding maps to a registered equation OR an explicit recorded rejection. The latent-SDE
false-friend + McMullen not-a-mover verdicts are PRESERVED (recorded as such) — no signal loss.

---

## Cross-ref graph (integration with the existing 165 equations)

- **E0** master ← E1…E12 (terms). E0 consumers: `tac.unified_action`, `tac.contest_score`,
  `tac.boundary_math.lever_b_levelset_generator`.
- **E5** (flow) ↔ existing `ema_decay_substrate_stage_aware_v1`, `dykstra_pareto_polytope_intersection_compounding_v1`,
  `anti_pattern_polytope_exclusion_dykstra_compounding_v1`.
- **E6** (IB) ↔ `categorical_blahut_arimoto_rate_distortion_v1`,
  `wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1`.
- **E7** (rate/Λ) ↔ `master_gradient_null_space_byte_fraction_v1`, `per_byte_leverage_uniformly_distributed_v1`,
  `brotli_cascade_bounded_per_stream_v1`.
- **E9** (pose) ↔ low-rank pose codec (#140), `wyner_ziv...posenet...`.
- **E12** (floor) ↔ the measured `S_floor≈0.118` (Lever-F derivation, task #53).

---

## Calibration wiring (operator fold-in)

Each equation carries the canonical `EmpiricalAnchor` + `predicted_vs_empirical_residual` + a
`next_recalibration_trigger`. The **live calibration anchors** are:
- **E3** (co-location) — residual 0.0 baseline; recalibrates `when_3+_new_empirical_anchors_in_domain`
  (re-run on SegNet/frame-cache change).
- **E8** (decisive-smoke gate) — PENDING; the smoke's measured `(PR, Δd_seg)` is its first anchor and will
  recalibrate the BINDING-vs-SYMPTOM verdict via `tools/recalibrate_equation.py` + the Catalog #344/#371
  auto-recalibrator.
Future anchors (the v2 training rows) auto-update these residuals — the registry becomes the continual-learning
memory the operator asked for (no tribal knowledge).

---

## Honesty firewall (NO-FAKE)

- **MEASURED** (real anchor): E2 (collapse + proven-by-test), E3 (co-location, byte-faithful), E4/E9/E10
  (`[macOS-MLX research-signal]`). All advisory — **NOT contest scores**.
- **THEORETICAL / EXACT-for-surrogate**: E0, E1, E5, E6 (Courtade–Weissman theorem), E7, E11.
- **FORMALIZATION_PENDING**: E8 (awaits the smoke), E12 (VCM headroom, lit-hunt-cited not independently read).
- **Lit-hunt cross-refs** (Finsler 1612.00343, posterior-coding 2410.09018, Wasserstein-Dice 1707.00478,
  contour-coding 1604.08001, MDL 1006.3679, VCM 2505.14980) are tagged *advisory — from lit-hunt FEED-ig/ih,
  not independently verified in this formalization*.
- **No fabricated calibration**: theoretical equations carry empty anchors (is_well_calibrated=False = honest
  "land first anchor" cue), not invented residuals.
- **means≠ends**: this makes the system smarter; it moves no exact score. Pointer 0.19110.

**Pillar status (canonical-helper 6-pillar):** wired (registry consumers/producers) · tested (existing
registry test suite + the E2 proven-by-test anchor) · Provenance-routed (every equation+anchor carries
canonical Provenance) · memo-anchored (this doc) · registered (178 entries) · retro-sweep N/A (no new STRICT
gate).
