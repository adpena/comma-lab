# #433 — Anisotropic-coupled per-class λ (P0 physics-model directive) + 3 measured threads (2026-07-11)

**Agent:** Fable builder (organ extension; #427 seal + #430/#428 wave inherited; the earlier
SOL dispatch of this task CRASHED with zero output — this is the real landing). **Cost:** $0
(cached tensors only: `gt_n96.npz` margins/lstars + σ_cc′ #382 + the durable comma10k prior;
NO scorer forward; the live #205 run / pid 88030 untouched — all reads were `daemon.log`
telemetry). **Pointer 0.19108282 [contest-CPU] UNMOVED** — the organ is MEANS; every number
here `[macOS advisory] NON-PROMOTABLE, score_claim=false`.

**Code:** `src/tac/witness_control/aniso_perclass_lambda.py` (profiles + flip temperature +
openpilot prior + `PhysicsPriorMeanAdjoint` + arms N/O/P/Q/R/S + SAO trust region) wired into
`lambda_net.{ARCHITECTURES,make_model}` + the CostateAgent DSL (2 new pipeline stages +
arm→lens routing). **Tool:** `tools/aniso_perclass_lambda_backtest.py`. **Tests:**
`src/tac/tests/test_aniso_perclass_lambda.py` (14; a planted-pair test CAUGHT a real
row-stochasticity bug — classes with no measured pair partners silently lost boundary mass —
fixed before landing). **Artifact:**
`experiments/results/costate_organ_backtests/aniso_perclass_lambda_backtest_20260711T144317Z.json`
+ a SUPERSEDING complete organ-ledger record (my first record was partial-arm-set and knocked
the prototype family out of the deduped arbitration view — caught + remedied append-only).

## 0. The operator directive (P0, verbatim-anchored)

"Bulk × sensitivity × impact on d_seg are ONE interacting per-class ANISOTROPIC dynamical
system; each class has a very different profile and they interact; also remember anisotropic.
Our V9·CGauge carrier work models that beautifully and we have all the equations." The
per-class-λ now CONSUMES the registered equations — none re-derived:

| ingredient | registered law consumed | realization |
|---|---|---|
| bulk vs sensitivity | `cgauge_master_action_v1` A2 (Fisher two regimes; margin ρ 0.978) | per-class interior/annulus split of σ′(m/ε)/ε |
| per-pair tension | `junction_young_angle_sigma_fit_v1` via #382 `resolve_length_sigma_matrix` | 1/σ_cc′ pair weighting |
| class interaction | `argmax_of_sdf_is_additively_weighted_power_diagram_v1` (#284) | boundary-pair adjacency measured from the cached partition |
| anisotropy | `anisotropic_basis_along_tangent_frequency_deficit_v1` + `cgauge_curvelet_parabolic_bank_v1` | per-pair along-tangent/across-normal margin-gradient energy |
| impact | ARM H smoothed-argmax EXACT gradient (ε=τ=ħ family, L75) | g^ε per class |

## 1. The measured physics (the coverage report — UNION residual, never Lane-only)

**Flip temperature (new instrument):** at ε = global-median margin (5.80) the sensitivity
field DEGENERATES TO AREA SHARES (measured: shares [.233,.007,.494,.016,.250] ≈ areas) —
classes-as-area-knobs, the exact failure the directive forbids. The flip-relevant scale is the
advection-ball rank-matched margin threshold (L85 dominant flip mode, two independent cached
sources): **ε_flip = 1.048**. All physics below is at ε_flip.

**Fisher two-regime split (per class, bulk/boundary):** annulus susceptibility **0.577 in
5.6% area** — independently RE-DERIVES the #333 annulus concentration from cache (prompt
memory: annulus share 0.57). Per-class profiles (the "very different profiles", measured):

| class | susc share | bulk/boundary | g^ε_flip | op-addressable |
|---|---|---|---|---|
| Road | 0.458 | 0.326 / 0.674 | 6.98e-3 | 0.939 |
| Lane | 0.077 | 0.000 / 1.000 | 1.20e-3 | 0.978 |
| Undrivable | 0.267 | 0.717 / 0.283 | 3.92e-3 | 0.229 (the horizon crack) |
| Movable | 0.059 | 0.283 / 0.717 | 8.87e-4 | 0 (model scope) |
| MyCar | 0.139 | 0.469 / 0.531 | 2.07e-3 | 1.000 (static hood) |

**Coupling C_phys** (row-stochastic BY CONSTRUCTION; boundary half split ½ own / ½ partner —
DERIVED_EXACT from m = φ_top1−φ_top2 symmetry): Road→Lane 0.320, **Lane→Road 0.494** (≡ the
K-arm's independent adjacency⊙1/σ 0.499 — convergent), Movable→{Road .195, Undriv .128},
MyCar→{Road .169, Lane .097}. **Anisotropy** (along/across margin-gradient energy, geomean-1
gauge): Road–Lane 0.181 vs Road–Undriv 0.019 — the Road–Lane interface carries ~10× the
relative along-tangent structure (the dash comb), so it is up-weighted in the coupling;
registered deficit 25/8 = 3.125 consumed as the scale anchor.

## 2. P0 VERDICT — aniso-coupled vs isotropic-independent (walk-forward tri-gate)

Formulation discipline first (measured, this wave + #430 wave): φ-rescale/coupling into an
unconstrained ridge re-fit is ABSORBED (N ≡ A ≡ K to 4 decimals — reconfirmed here). The
physics-consuming formulation is the **prior-mean** (shrink-to-prior) solve with the
**score-law-pinned scale**: direction + relative magnitudes fully pinned by physics
(C_phys ∘ g^ε), intercept prior = per-channel median drift, exactly ONE global κ ≥ 0 by 1-dof
projection (vs arm L's 7-param empirical-Bayes that injected variance at n=2).

| arm | WF MAE | early-fold WF | LOO | AUROC |
|---|---|---|---|---|
| persistence heuristic | **0.002792** | 0.004715 | 0.003698 | — |
| **Q_priormean_iso (M0=I ablation)** | **0.003067** | **0.004024** | 0.002858 | 0.92 |
| **P_priormean_aniso (THE P0 arm)** | 0.003182 | 0.004265 | 0.002906 | 0.92 |
| S_priormean_openpilot | 0.003777 | 0.005112 | 0.003234 | 0.96 |
| R_priormean_c10k_scorelaw | 0.003895 | 0.005268 | 0.003269 | 0.96 |
| A_ridge_solve (isotropic-independent incumbent) | 0.003902 | 0.005201 | 0.003296 | 1.0 |
| N_aniso_coupled / K / O (reweight family) | 0.003902–3 | 0.005200 | 0.003295 | 1.0 |
| L_priormean_comma10k (the old κ failure) | 0.009120 | 0.022165 | 0.003337 | 1.0 |

**Honest P0 answer, two halves:**
1. **The physics FORMULATION wins decisively over the isotropic-independent baseline**: P
   beats A by −18% WF / −18% early-fold, and the prior-mean family is the **first
   ridge-family arm ever to beat persistence at early folds** (0.004024/0.004265 vs
   0.004715) — the n=1-fragility regime this arm was built for. The old failure L is beaten
   4–5× at early folds: the score-law-pinned κ cure is real.
2. **The anisotropic-coupled DIRECTION is forecast-neutral at this n**: the isotropic
   ablation Q (same machinery, M0=I) is equal-or-better (Δ 0.000115 WF, ~3.7%, within the
   30× fold-variance noise). verdict_scope: INSTANCE (1 trajectory, 9 plateau-dominated
   intervals; a transient-rich window is the discriminating data). C_phys remains the
   **structural** per-class λ readout the #430 composer consumes (Lane→Road coupling is a
   measured fact independent of the forecast gate) — physics as structure, not yet as
   forecast edge. Nobody passes the full WF gate at n=9: the whole family still loses to
   persistence (plateau posture; meta-λ prefer_persistence stands).

## 3. Thread 1 — the OPEN comma10k family: verdict

Arm R = rarity direction × Fisher impact × σ_cc′ coupling, scale pinned by the score law
(κ 1-dof). Measured: the SCALE CURE works (early-fold 0.00527 vs L's 0.0222) but R ≈ plain
ridge and is WORSE than the direction-free ablation Q ⇒ the comma10k rarity DIRECTION adds
nothing to forecasting on this trajectory. **Family verdict: CLOSED for forecast use at this
n on this vehicle** (verdict_scope now: formulation ×2 [φ-rescale, EB-κ] + direction ×1
[score-law-pinned] — all measured). The rarity prior stays alive as a SENSOR/duty-queue
instrument (its Lane 3.5× converges with the flip crux). Reopen trigger: ≥3 organ-ledger
records / regime-rich intervals.

## 4. Thread 2 — openpilot ISOLATED arm: verdict

Built isolated (never again bundled): measured horizon row 186 / hood row 286 from the cached
partition; ego-addressable susceptibility per class (Movable = 0 BY MODEL SCOPE — ego motion
does not explain object motion, L83). Reweight arm O ≡ ridge (absorption, expected);
prior-mean arm S 0.003777 — better than ridge but WORSE than the no-openpilot ablation Q.
**Clean isolated verdict: NO distinct organ win; openpilot stays witness-side-only as held**
(verdict_scope: formulation ×2 on this trajectory; the geometry itself — horizon/hood rows,
addressability — is now a cached organ sensor).

## 5. Thread 3 — RL/post-training deepening: honest verdict

- **GEPA cycle 2** executed on the extended 9-arm tournament: **11/11 candidates REFUSED**
  (best challenger Q at 0.003067 vs incumbent E_prototype_bregman 0.002839). Incumbent
  stands; the reflect→measure→dispose loop works and correctly declines.
- **SAO single-rollout trust region** on Λ between walk-forward refits: **INERT at the
  pre-registered radii** 0.25/0.5/1.0 (≡ plain ridge 0.003902 — fold-to-fold coefficient
  drift is <25% of norm). Positive control at r=0.001–0.05 BINDS and worsens (0.0078→0.0043)
  ⇒ the clamp is live and the null is real: **the WF error is model BIAS, not update
  variance** — an anchored update cannot fix it.
- **Curiosity acquisition**: PowerPlay ranking already runs in the organ; no new measured
  value at n=1 (no new probes became measurable this wave).
- **Verdict: RL/post-training does NOT help NOW at n=1.** Every gain this wave came from
  solve-formulation physics, not reflective/RL machinery. It needs the trajectory ledger to
  accrue (≥3 records; the VAPO crossover is DETECTED never scheduled). Data-blocked → SOL's
  synthetic-data-generation survey is the named unblock; when it lands, GEPA/SAO re-run per
  accrual automatically via the arbitration.

## 6. Fragility (honest)

n=1 trajectory, 9 intervals, plateau-dominated (transient folds are where the physics prior
should differentiate — unmeasurable until a regime-rich window). P-vs-Q margin (1.15e-4)
within fold noise. ε_flip is advection-radius-1 rank-matched — a different flip model would
shift it (the susceptibility ORDERING was stable across ε in the two temperatures measured).
The openpilot arm is diag-only (no coupling) by isolation design. The organ-ledger incident
(partial record superseding the full one) is remedied append-only but shows the compounding
path trusts record completeness — a duty item for a record-completeness guard.

## 7. Triality legs + stores consulted

**DSL:** 2 new `TrainingPipelineSpec` stages (`aniso_perclass_physics`,
`openpilot_isolated_prior`, both EXECUTED_$0 with measured rows) + GEPA stage updated with
cycle 2 + SAO + the RL verdict + 6 new arms routed in the arbitrated derivation. **Equations:**
5th `EmpiricalAnchor` (`aniso_perclass_lambda_433_backtest_20260711`) registered on
`costate_lambda_marginal_ds_v1` (residual = the P-vs-Q gap 1.15e-4). **DAG:** FEED-433 block
appended to the sub015 DAG + the superseding FEED-426-organ ledger record. **STORES:** organ
envelope #427 · scorer-arms memo #430 · `cgauge_master_action_20260711` derivation tree ·
`cgauge_parametrization_optima` (deficit 25/8) · σ_cc′ #382 (both presets read; fitted-20260707
used) · textured power diagram #284 · gt_n96 cache · comma10k prior artifact · organ ledger.

**Pointer 0.19108282 [contest-CPU] UNMOVED.**
