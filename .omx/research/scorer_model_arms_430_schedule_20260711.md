# Scorer-model arms (P0) + #430 coherent-schedule backtest — BUILT + MEASURED (2026-07-11)

**Agent:** Fable builder (organ extension wave; seal #427 inherited). **Cost:** $0 compute
(cached tensors + 192 real comma10k masks fetched once, ~1.6 MB; NO scorer forward; pid
88030 / the live #205 run untouched — all scorer signal came from `gt_n96.npz` caches).
**Pointer 0.19108282 [contest-CPU] UNMOVED** — everything here is MEANS (the organ never
ships, never scores). Every number `[macOS advisory] NON-PROMOTABLE, score_claim=false`.

**Code:** `src/tac/witness_control/scorer_model_arms.py` (arms H/I/J/K/L/M) +
`src/tac/witness_control/schedule_backtest.py` (#430) + `tools/schedule_bundle_backtest.py`
(one-command durable artifact) + `tools/build_comma10k_regime_prior.py` (bounded fetch).
**Tests:** `src/tac/tests/test_scorer_model_arms_430.py` (17) + the organ suite (25) green.
**Artifacts:** `experiments/results/costate_organ_backtests/schedule_430_backtest_20260711T134845Z.json`
+ `costate_organ_backtest_20260711T13{4347,5448}Z.json` +
`experiments/results/comma10k_regime_prior/{comma10k_class_prior.json,fetch_manifest.json,masks/}`.

## 1. What the backtest EARNED (each measured, none asserted)

### ARM H — smoothed-argmax metric relaxation (#428 survey #1; BUILT + VERIFIED EXACT)
Perturbed-optimizer/Gumbel smoothing of the d_seg argmax metric THROUGH the real frozen
SegNet's cached margin field. The per-class gradient of the ε-smoothed metric is
analytic and **finite-difference verified: rel gap 5.6e-7** — zero model error, no
surrogate-vs-teacher gap to audit. ε = median margin (the τ=ε=ħ knob, L75). Top-2
(pairwise-tropical) approximation stated. Measured per-class gradient at ε=5.80:
Road 7.97e-3 · Lane 2.50e-4 · Undriv 1.69e-2 · Movable 5.38e-4 · MyCar 8.55e-3 —
the EXACT ∂(smoothed d_seg)/∂δ_c (mass-weighted; the aggregate-flip λ direction).

### ARM J — adversarial-boundary geometry (survey #4; BUILT + FAITHFULNESS-AUDITED)
Advection-ball minimal-flip susceptibility (the measured dominant flip mode, L85) with
class-pair weighting by the fitted Young σ_cc′ (#382 — REUSED, not rebuilt). Measured:
**Lane susceptibility 2.67 = 4.07× uniform — the flip-hot class recovered independently**
from pure boundary geometry (converges with the L2/L66 crux). **Ball-agreement
faithfulness acceptance (2306.04431 transposed): IoU 0.732, precision/recall 0.844 —
PASS** (margin flip-set vs actual advection-ball label-change set; two independent
cached sources; rank-matched so calibration can't fake it). Composes the
margin-polytope free-budget picture (#47 first-order flip system, `margin_polytope`).

### ARM I/L — comma10k regime model (trajectory-INDEPENDENT; BUILT + HONEST NEGATIVE)
Durable prior from **192 real comma10k masks** (SegNet's actual training set; 0 contest
frames, L80; 0.0% unmatched palette). Measured class shares Road .200 / Lane .0071 /
Undriv .521 / Movable .0206 / MyCar .252 — near-identical to the contest-video shares
(same rig). **Rarity prior: Lane 3.51× / Movable 1.22×** — a trajectory-free instrument
that independently points at the measured flip crux.
**Did it cure the n=1 fragility? MEASURED NO — in both built formulations**
(verdict_scope: formulation ×2, family open):
- φ-class-rescale into the ridge solve (arm I): **INERT** — WF ≡ plain ridge to 4
  decimals; the unconstrained re-fit absorbs any feature rescale (same mechanism as the
  seal's G-arm neutrality, now understood structurally).
- shrink-to-PRIOR ridge (arm L, the neg↔cure adjacency): **WORSE at early folds**
  (0.0222 vs ridge 0.0052) — the prior DIRECTION is trajectory-free but its SCALE κ is
  not; empirical-Bayes κ at n=2 injects variance instead of removing it.
Stopped there per the no-tuning-until-it-wins discipline. The arms' measured wins are
elsewhere: **binding AUROC 1.0 restored** (all six new arms; prototype family sits at
0.96) + the sensors themselves + the duty-queue ranking use.

### ARM K — per-class λ-heads + v8 reconcile (BUILT; feeds #430)
Cross-class coupling C = ½I + ½·rownorm(boundary-pair-adjacency ⊙ 1/σ_cc′), adjacency
MEASURED from the cached argmax partition (#284 pair structure). Measured coupling:
Lane→Road 0.499 (the shared Road–Lane boundary carries half of Lane's response) —
the physics the diagonal φ missed. `perclass_lambda()` is the per-class marginal-ΔS
readout the #430 composer consumes. Forecast-neutral (≡ ridge family), AUROC 1.0.

### Distillation verdict (the operator's question)
**Metric-relaxation WON; the learned surrogate stays properly BLOCKED.** Survey order
followed: #1 (smoothed argmax) is built, exact, and $0; a learned Jacobian/Sobolev
surrogate (#2) now has to beat a ZERO-MODEL-ERROR baseline to justify a multi-hour
training job — its `TrainingPipelineSpec` stage stays BLOCKED with that named blocker.
No fake build.

## 2. #430 — the coherent synergistic schedule (GO executed; MODEL-BASED backtest)

Shape per the coordinator fold (**arXiv 2607.08716** "Remember When It Matters":
selective intervention MEASURED to beat always-on/passive, +8.3pp Terminal-Bench):
the organ's schedule is a **state-gated cascade** — island-birth → boundary-form →
τ-sharpen⊕repair → finish — with gates DERIVED from the measured trajectory (recorded
in the artifact) and coordinated bundles ordered by per-class λ (arm K), NOT
independent epoch/dwell floors.

**Backtest protocol (honesty tier stated on every artifact):** 4-policy counterfactual
replay (hand = the measured #205 shares · selective · always-on · uniform) through the
response model, on the real 9-interval #205 trajectory; model trust quantified by
SELF-REPLAY (hand-policy replay must reproduce the measured trajectory).

| model (replay) | self-replay MAE / final gap | hand ∫d_seg·dep | selective | always-on | uniform |
|---|---|---|---|---|---|
| E_prototype_bregman (WF winner, state-dep) | 0.0054 / 1.6e-4 | 8.649 | **6.283 (−27.4%)** | 6.258 | 11.362 |
| A_ridge_solve (sensitivity) | 0.0060 / 1e-5 | 8.849 | 8.805 | 8.805 | 8.910 |
| K_perclass_v8 | 0.0060 / 1e-5 | 8.849 | 8.805 | 8.804 | 8.908 |

**Measured verdicts:** (1) the organ's coordinated schedule **beats the hand-scheduled
#205 curriculum on ALL THREE replay models** (decisively, −27%, on the
walk-forward-winning state-dependent model; marginally on the affine models — an
affine model structurally can't see timing). (2) **selective ≈ always-on in-model**
(Δ0.4%): on this transient-only prefix a cascade gate is active at EVERY verdict, so
the two policies nearly coincide; the in-model replay CANNOT resolve the gate question
— 2607.08716's external measurement + the off-policy-trust argument carry it; claimed
as exactly that, not as our win. (3) The **OperatorGoTicket is emitted**
(`mutate_live_config`, actuation NONE, measured rows embedded); **gates_owed: the LIVE
A/B + the witness-DSL compile of the bundle** — the organ recommends, the operator
launches. d_pose is outside this replay (pose-blind until ep726 by design).

## 3. Live posture changes observed during the build (free out-of-sample)

- The #205 trajectory grew 9→10 verdicts during the build. At 9 intervals **the whole
  model family now loses walk-forward to persistence** (E_bregman 0.002839 vs 0.002792)
  — the envelope §3's plateau-regime prediction confirmed out-of-sample. The
  arbitration correctly falls back to the incumbent default; **meta-λ
  prefer_persistence is the operative posture**; the organ ledger holds the live state.
- The **PRISM faithfulness audit FAILS at the newest state** (max_rel_gap 0.398 > 0.35)
  — the plateau shifted the routing and the renormalization is doing hidden work; the
  audit catching this live is the instrument working. Watch item for the next check-in.

## 4. Fragility (honest)

Replay verdicts are MODEL-BASED counterfactuals on n=1 trajectory (instance scope);
trust bounded by the self-replay residual; the always-on baseline shares mechanics with
selective (same λ, same budget) but its excursion is farther off-policy. The comma10k
negative is formulation-scoped — a prior-as-REGULARIZER with trajectory-free SCALE
(e.g., κ pinned by the score law rather than fit) is the named open follow-up, duty-
queued, not built. The cascade's later stages (τ⊕repair, finish) never fire in the
replay window (trajectory hasn't reached them) — their bundles are template-inherited,
not replay-validated.

## 5. Triality legs + stores consulted

DSL: training pipeline statuses updated honestly (5 EXECUTED_$0 stages incl. the two
new arms + #430; learned surrogate BLOCKED with named blocker); new arms mapped in the
arbitrated derivation. Equations: 4th EmpiricalAnchor on `costate_lambda_marginal_ds_v1`
(`scorer_model_arms_430_backtest_20260711`), registered. DAG: FEED-426-organ record
compounded (latest-per-run); FEED block appended to the sub015 DAG. Papers-checked:
2607.08716 appended. STORES: envelope #427 · #428 survey · gt_n96 cache · σ_cc′ #382
(`length_sigma`) · margin_polytope #47 · comma10k (fetched) · organ ledger.

**Pointer 0.19108282 [contest-CPU] UNMOVED.**
