# The Transient Forge — BUILT + MEASURED honest-gate verdict (Task #434, 2026-07-11)

**Status:** the engine is BUILT ($0, deterministic, numpy-only) and the honest adoption
gate is MEASURED. **[MEASURED] verdict: HONEST NEGATIVE — synthetic data does NOT yet
confer real chronological walk-forward skill on the #205 plateau trajectory.** The engine
runs, the gate is honest, the iteration target is named. **Pointer 0.19108282 [contest-CPU]
UNMOVED — this is MEANS; no score claim.** Every number `[macOS advisory] NON-PROMOTABLE,
score_claim=false`. Tier-2 (real witness micro-runs) is operator-GO and was **NOT fired**;
the live #205 run (pid 88030) was untouched.

Design consumed as-is: `.omx/research/synthetic_data_nvidia_sota_organ_434_20260711.md`
(THE SPEC). This memo is the BUILD + MEASUREMENT of it.

---

## 1. What was built (the engine, per the memo's §2 fidelity ladder)

Module: `src/tac/witness_control/transient_forge.py` · CLI: `tools/transient_forge_backtest.py`
· tests: `src/tac/witness_control/tests/test_transient_forge.py` (13, all green).

- **tier-0 — surrogate replay** (`tier0_replay_windows`): consumes the in-tree #430
  `schedule_backtest.replay_policy` machinery over randomized control policies on the real
  prefix (REPRESENTATION/COVERAGE only; the memo's stated bias caveat — never the sole
  adoption signal).
- **tier-1 — the multi-class CGauge simulator** (`simulate` / `sample_sim_params`): a
  deterministic numpy-fp32 multiphase relaxed gradient flow on the registered master
  action's PHYSICS — per-class relaxation toward equilibria (UNCONDITIONALLY-STABLE
  exponential update; see §3 round-1 fix) + σ_cc′-seeded pair coupling with the measured
  **C_phys 0.494 Lane↔Road** anchor + Chan–Vese island-birth source (Movable) + MCF
  minority (Lane) erosion, lever-modulated through the SAME `lever_features` design surface
  the organ learns over. Genuinely creates transients (island birth, boundary formation,
  Lane reversal) OUTSIDE the observed plateau. Grounded on the real prefix (shrinks the
  sim2real gap); transient DOF are the new physics.
- **tier-2 — real witness micro-runs**: DESIGNED, operator-GO, **NOT fired** (compute owned
  by the live run; the only bias-free source; the only records that count toward the
  ≥3-record graduation gate).
- **UED-regret teacher** (`window_regret`): regret = arm-ensemble jackknife disagreement +
  learning-potential (ridge-vs-persistence residual on the window). Plateau windows score
  ≈0 and die; the `reversal` regime drives the C_phys coupling with asymmetric Lane/Road
  lever emphasis (the #433 unblock, in the DATA).
- **BIRD/QD diversity gate** (`QDArchive`): admits a batch only when regime-descriptor
  archive coverage GROWS (new/improved cells) AND effective rank does not collapse
  (redundancy audit); plus a nearest-centroid **memorization probe** (accuracy − chance;
  measured ≈ −0.05, i.e. safely below memorizable).
- **PDR × RQGM loop** (`forge_corpus`): PARALLEL generate M candidate trajectories across
  regimes → UED regret-select (keep top-frac) → BIRD diversity-gate → DISTILL to a bounded
  workspace. The within-fold evaluation (regret, diversity) is FROZEN and consults ONLY the
  real prefix ≤ k (RQGM: no look-ahead; the structural cure for the flattery class).
- **TRAK / exact-LOO influence pruning** (`influence_prune`): per-source influence on a
  held-out real probe (the last prefix interval, no test look-ahead) — the per-window
  flattery detector, REPORTED as a diagnostic (`n_pruned_out`).
- **Optimal-form augmentation** (`ForgeAugmentedRidge`, the VeLO/TabPFN/PFN pattern): the
  synthetic corpus is fit to a PRIOR MEAN; the real solve is shrunk toward it with strength
  λ selected PAST-ONLY by prefix LOO (λ=0 recovers real-only; λ→∞ recovers the prior). This
  is the codebase's measured non-absorbable prior injection (shrink-to-prior, not
  φ-rescale). Naive concat (synthetic as equal-weight volume) is kept only as a diagnostic.

## 2. The adoption gate — MEASURED (the acid test, §3 of the SPEC)

Real trajectory: `experiments/results/levelset_v752_baseline_20260710T185913Z` (10 verdicts,
9 intervals, 22 levers, plateau-dominated). Artifact:
`experiments/results/transient_forge_backtests/transient_forge_backtest_20260711T171116Z.json`.
Chronological walk-forward, null-relative, prior-mean arm, λ selected past-only:

| arm | real WF MAE | vs persistence |
|---|---|---|
| **persistence (null)** | **0.002792** | — |
| incumbent E_prototype_bregman | 0.002839 | loses |
| real-only ridge (c, ablation) | 0.003902 | loses |
| **forge prior-mean (d, PRIMARY)** | **0.003882** | **loses** |
| forge naive-concat (diagnostic; synthetic swamps) | 0.167173 | loses badly |

**beats persistence = FALSE · beats incumbent = FALSE · beats real-only = TRUE → ADOPTED =
FALSE.** Per-fold λ-selection: 6/7 folds pick λ=0 (weight synthetic zero); ONE transient
fold (ep150) picks λ=0.03 and improves 0.00279→0.00265 — a marginal, within-noise, single-
fold effect. Aggregate: the Forge nudges ridge 0.5% better than real-only but does NOT
rescue it past persistence.

**#433 aniso acid test** (P aniso-reversal corpus vs Q iso-plateau corpus, same arm, real
folds): P 0.003880 vs Q 0.003902, separation **+2.3e-5 — within noise**. The manufactured
anisotropic transients do NOT separate from the isotropic ablation on the real folds ⇒ the
coupling is either genuinely absent on this trajectory OR the simulator cannot express it
discriminably. **At 0 tier-2 runs these two are not distinguishable — stated, not hidden.**

## 3. Round-1 adversarial self-review (attacking the hardest links)

- **(a) Is the CGauge simulator realistic, or a self-fulfilling toy the arm memorizes?**
  Attack + defense: if it were a memorizable toy, synthetic-fold skill would be high while
  real skill stayed low AND the arm would eagerly weight it. Measured: past-only λ-selection
  weights synthetic ≈0 on 6/7 folds (monotone LOO in λ — verified directly: prefixLOO
  0.000115→0.000379 as λ 0→3), and the memorization probe is below chance (−0.05). The arm
  is NOT fooled. Residual honest risk: the non-Lane↔Road coupling entries come from the
  prefix dxdt covariance (n=5 prefix ⇒ noisy); only the C_phys entry is externally measured.
- **(b) Does synthetic beat REAL-ONLY, or only persistence (which real-only already loses)?**
  Real-only ridge LOSES to persistence (0.0039 vs 0.0028); forge marginally beats real-only
  but STILL loses to persistence. The treatment does not rescue the arm to competitiveness.
  No inflated headline.
- **(c) Is the sim2real gap real and closable, or a wall?** Synthetic-fold MAE is tiny
  (~1e-4) vs real fold MAE 0.0039 — a large gap. This is the memo §5 weakest link made
  measured: the simulator fits its own dynamics far better than the real dynamics. Whether
  the gap is closable is UNKNOWN without tier-2 (bias-free) trajectories or a transient-rich
  real test trajectory. The deepest honest ceiling: the adoption TEST is still n=1 (one
  plateau run); no amount of manufactured trajectory diversity is provable-useful until the
  TEST set contains the regime the training manufactures — i.e. a transient-rich real
  trajectory or tier-2 referee.
- **Round-1 bug caught + fixed:** explicit-Euler integration diverged (log-bytes → 1e6 when
  k·Δep>2); replaced with the unconditionally-stable exponential relaxation backbone +
  bounded sources + physical clamps (regression-tested). Also: the spurious macOS Accelerate
  divide/overflow FP flags on finite matmuls (envelope §6 #8) are suppressed with a
  fail-loud finite guard (`_fit_ridge`). Also: influence pruning that emptied the corpus was
  degenerating the primary arm to real-only and HIDING the λ arbiter — the primary now sees
  the full diverse corpus and λ-selection is the transparent arbiter (pruning stays a
  reported diagnostic).

## 4. Named iteration target (the honest forward path)

The engine is built and the gate is honest. To make synthetic data confer real skill:
(1) **tier-2 micro-runs** (operator-GO) — 3–5 short real witness runs give the bias-free
referee that disambiguates "coupling absent" vs "simulator can't express it," AND accrue the
≥3-record graduation gate; (2) **regime-match the test** — the value is unprovable on a
plateau-only test set; a transient-rich real trajectory (or tier-2) is the missing TEST
regime; (3) **tighten sim2real** — reduce the reported synthetic-vs-real gap (the Replicator
5→87 iteration loop transposed) before re-testing. Synthetic-fold wins remain NEVER adoption
evidence (NO-FAKE #3); tier-2 records are the only graduation-counting data.

## 5. Triality legs (landed this pass)

- **DSL:** `transient_forge_synthetic_trajectories` promoted PROPOSED→**EXECUTED_$0** in
  `tac.witness_dsl.costate_agent_dsl.DEFAULT_TRAINING_PIPELINE` with the measured
  honest-negative row (matches the `comma10k_regime_prior` / `openpilot_isolated_prior`
  EXECUTED_$0-with-honest-negative precedent — EXECUTED_$0 = the $0 stage RAN and produced a
  measured row, win or negative). No trainer flag invented; the backtest CLI is standalone.
- **DAG:** FEED-434-forge-build appended to
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **Equations:** N/A this pass — the first candidate law was the MEASURED sim2real transfer
  curve, registered only IF synthetic confers real WF skill; it does NOT (adoption=False),
  so no law (consistent with the memo §4 + the #427 seal's ≥5-run anchor stance). The engine
  CONSUMES `cgauge_master_action_v1` (the physics) — none re-derived.

**[MEASURED] Pointer 0.19108282 [contest-CPU] UNMOVED — MEANS work; no score claim.**
