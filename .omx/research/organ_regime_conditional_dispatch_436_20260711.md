# The #426 costate organ becomes a REGIME-CONDITIONAL SELF-DISPATCHER (task #436, 2026-07-11)

**Author:** decision-systems subagent. **Status:** MEANS. **Pointer 0.19108282 [contest-CPU]
UNMOVED — no score claim; every number `[macOS advisory] NON-PROMOTABLE, score_claim=false`.**
Operator #436: *"the costate organ is smart enough to know what to use and when to use it, or
it should be."* This memo closes the gap between "or it should be" and "is", MEASURES the
result on the sealed #205 trajectory, and states the honest verdict + limits.

---

## 1. The gap (not a rebuild — an extension of the built skeleton)

The organ ALREADY had: the meta-λ self-monitor (`self_monitor.py`, senses plateau →
`prefer_persistence`), the PRISM prototype router (`prototype_router.py`, names regimes:
lane-erosion / mixed-Lane-Road / movable-island-unborn), the arbitration layer
(`continual_costate.arbitrate_architecture` + DSL `RoutingSpec` SINGLE_BEST). **The gap: the
arbiter picks a GLOBAL single-best arm** (the seal chose prototype; the ledger picks the best
cumulative walk-forward arm). But the closed-form GP-costate result
(`closed_form_gp_costate_posterior_20260711.md` §4) MEASURED that the optimal tool is
**REGIME-CONDITIONAL**: `T_gp_costate_posterior` is the best arm on the TRANSIENT folds (first
arm to beat persistence walk-forward, −34% mean, 2–4× on the early transient) yet LOSES to
persistence on the PLATEAU (every learned arm does; envelope §3). **A global-single-best
arbiter structurally cannot express that.** #436 builds the per-STATE dispatcher.

## 2. What was built (`tac.witness_control.regime_dispatch`, $0 numpy, no actuation surface)

- **`classify_regime(past, comp, lever_names, *, meta_lambda_guard=True)` — PAST-ONLY.**
  Discriminator = the self_monitor plateau component verbatim (τ=1.0, NOT tuned): the
  trajectory is on a **plateau** when the latest OBSERVED class-weighted `|d_seg slope|` is
  below its running median, else **transient**; `<2` observed intervals → **uncertain**. It
  also reads the PRISM prototype router's **named regime** + **routing entropy** at the latest
  past state (interpretive context). Verified past-only: it consumes `intervals[:hold]` and
  never the fold target `intervals[hold]`.
- **The meta-λ defer governor (task-directed "know when to use nothing").** In a nominal
  transient, if the self-monitor's **model-surprise** signal fires — the interpretable head,
  fit on all-but-the-latest OBSERVED interval, mispredicted the latest OBSERVED interval by
  `>1.5×` the persistence error (self_monitor component-1, threshold verbatim) — the organ
  **distrusts its own model** and downgrades the regime to `uncertain` → defer to persistence.
- **`dispatch_decision` → the OBSERVATORY row (Rudin, hard req).** Every dispatch emits WHICH
  regime, WHICH tool, WHY (the deciding past-only signal), the meta-λ surprise ratio, the PRISM
  regime name + entropy, and the cited per-regime WF-ranking prior. `explain()` is the readback.
- **`backtest_dispatch` — the arbiter** (walk-forward, past-only): per fold, classify from
  `intervals[:hold]`, route, forecast, score vs the MEASURED slope; compare the dispatcher mean
  WF MAE to persistence AND to the **global-single-best fixed arm** (strongest single member of
  the pool `{persistence, T_gp, E_prototype, E_prototype_bregman, F_bsf}`). Reports the meta-λ
  guard ablation, per-fold rows, and a look-ahead oracle diagnostic (reported, never used to route).

The policy `{transient→T_gp_costate_posterior, plateau→persistence, uncertain→persistence}` is a
**STRUCTURAL PRIOR** (from GP memo §4 + envelope §3); the backtest **tests** it, never fits it —
the dispatcher has **zero parameters fit on this trajectory** (both thresholds inherited from
self_monitor; the routing table fixed).

## 3. THE MEASURED VERDICT (sealed #205: 10 verdicts / 9 intervals / 7 walk-forward folds)

| forecaster | walk-forward MAE (class-weighted d_seg) | vs persistence | vs global-single-best |
|---|---|---|---|
| persistence (incumbent) | 0.002792 | — | — |
| **global-single-best fixed arm** = `T_gp_costate_posterior` | **0.001852** | −34% | — (the arbiter to beat) |
| E_prototype_bregman / F_bsf / E_prototype | 0.00284 / 0.00284 / 0.00297 | loses | loses |
| **dispatcher (meta-λ guard ON)** | **0.001596** | **−43%** | **−13.8% (BEATS)** |
| dispatcher (meta-λ guard OFF, ablation) | 0.001738 | −38% | −6.2% (BEATS) |

Per fold (past-only classification → route):
```
ep50  transient → T_gp   err .00143  (oracle T_gp        ✓)
ep75  transient → T_gp   err .00088  (oracle T_gp        ✓)
ep100 plateau   → persist err .00306  (oracle E_bregman .0001 — 2-arm policy leaves value)
ep125 transient → T_gp   err .00109  (oracle T_gp        ✓)
ep150 plateau   → persist err .00072  (oracle persist    ✓)
ep175 plateau   → persist err .00182  (oracle T_gp — a miss)
ep200 UNCERTAIN → persist err .00217  (oracle persist    ✓; meta-λ surprise ×2.6 fired)
```
**The dispatcher BEATS BOTH persistence AND the global-single-best arm walk-forward, routing
5/7 folds to the oracle arm.** The meta-λ guard flips exactly the ep200 fold (the head was
surprised ×2.6 there) and improves it; both guarded and unguarded variants beat global-single-
best, so the conclusion is robust to the guard choice.

## 4. THE HONEST VERDICT — BEATS-BOTH-BUT-PROVISIONAL (not an honest-negative, not an adoption)

The task's arbiter question — *does per-state dispatch beat picking one arm?* — answers **YES on
this trajectory**. It is NOT the honest-negative branch (the dispatcher does beat global-single-
best, so it is not merely over-fitting labels into a loss). **But it is NOT a significance-cleared
adoption either**, and I refuse to narrate it as one:

- **Margin within noise.** 13.8% (guarded) / 6.2% (unguarded) over global-single-best at n=7
  folds. The dispatcher differs from global-single-best only on the folds it routed to
  persistence (ep100/150/175/200): **3 wins, 1 loss, sign-test p≈0.31 — NOT significant.**
- **In-sample-derived policy (the deepest limit — attack (c)).** The transient→GP /
  plateau→persistence rules AND both thresholds were derived from analysis of THIS SAME
  trajectory (GP memo §4, self_monitor). Confirming them here is **consistent-with**, not
  **out-of-sample-of**, the motivating data. The mechanism (past-only classify → route) is
  sound and leak-free; the **policy's transfer is unmeasured**. The generalization test is owed
  at ≥2 trajectories (re-runs per record accrual via the organ ledger).
- **The 2-arm policy leaves value on the table.** At ep100 the true oracle was
  `E_prototype_bregman` (0.0001) but the plateau route chose persistence (0.00306). A 3-arm
  policy (plateau → the prototype family when it is confident) is a measured owed extension.

**Banks regardless:** the regime→tool RULES + the dispatcher MECHANISM + the leak-free
walk-forward arbiter are structural knowledge that compound. The dispatcher is wired advisory
(the DSL default) but its ADOPTION as a superior-to-single-best policy stays **verdict_scope:
instance, provisional-until-accrual** — exactly the seal's standing for every n=1 organ verdict.

## 5. Self-awareness (task step 4) — the organ correctly DEFERS

At the **live #205 state** the discriminator says transient (recent slope 1.86e-4 ≥ median
9.64e-5) BUT the meta-λ model-surprise fires ×2.6 → the organ downgrades to `uncertain` →
**defers to persistence**. It also defers on insufficient history (`<2` intervals) and on
plateau. "Knowing what to use when" includes **knowing when to use nothing** — demonstrated,
tested (`test_meta_lambda_guard_defers_when_model_surprised`).

## 6. Round-1 adversarial review (attacking the hardest links)

- **(a) Past-only, or peeks at the fold outcome?** CLEAN. `classify_regime` consumes
  `intervals[:hold]`; the meta-λ surprise fits on `past[:-1]` and predicts `past[-1]` (the last
  OBSERVED interval, never the target). The GP query epoch = the boundary (last training obs) —
  boundary extrapolation, not leakage (identical to the shipped `backtest` WF convention; GP
  memo §6c). Test-guarded (`test_classification_is_past_only_no_target_leak`).
- **(b) Beats global-single-best, or overfits ~7–9 labels?** BEATS it with **zero
  trajectory-fit parameters** (thresholds inherited; table fixed) — it is not fitting labels,
  it applies parameter-free inherited rules. But the margin is within noise (p≈0.31); the honest
  claim is "nominal robust win, not significance-cleared." Not overclaimed.
- **(c) Does the transient/plateau boundary generalize?** UNKNOWN — the policy is in-sample-
  derived; out-of-sample test owed at ≥2 trajectories. Stated as the primary limitation (§4).
- **Self-attacks:** the meta-λ guard uses the E_prototype incumbent's surprise as a *global*
  model-distrust signal even when routing GP (defensible — surprise = regime-shift argues for
  the conservative incumbent — but a GP-own-surprise variant is more targeted; owed sensitivity
  check). The 2-arm policy is minimal by design; the 3-arm extension is owed. Determinism +
  containment (no subprocess/trainer imports, advisory dataclasses only) verified.

## 7. Triality legs + STORES CONSULTED

- **DSL:** `DispatchPolicySpec` (typed, `never-invent`-validated regime→tool, meta-λ-guard flag)
  wired into `CostateAgentProgram` + `CompiledCostateOrgan.dispatch()/dispatch_backtest()` +
  `describe()`/`render_lines()`; provenance carries the measured verdict.
  (`tac.witness_dsl.costate_agent_dsl`.)
- **DAG:** `FEED-436-regime-dispatch` appended to
  `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** **N/A-with-reason.** n=1 trajectory, per-fold sign-test p≈0.31 (below the
  ≥5-run anchor bar per the seal's standing stance); the dispatcher + backtest are APPARATUS,
  not a registered law. Reactivation: register a `regime_conditional_dispatch_v1` law only when
  (i) ≥5 trajectory records show per-state dispatch beats global-single-best out-of-noise AND
  (ii) the per-fold significance clears.
- **Tests:** `src/tac/witness_control/tests/test_regime_dispatch.py` (15: policy shape,
  past-only, defer cases, observatory, the arbiter backtest, determinism, meta-λ guard measured
  + defer, DSL leg + never-invent). All pass; ruff-F clean.
- **Tool:** `tools/lambda_net_backtest.py` gains the `regime-conditional dispatch backtest`
  section + JSON payload (default ON, $0).
- **STORES CONSULTED:** `closed_form_gp_costate_posterior_20260711.md` (§4 regime→tool rules) ·
  `costate_organ_capabilities_limits_envelope_20260711.md` (the seal: §2 prototype WF, §3
  plateau/transient thesis, §4 n=1 fragility) · `self_monitor.py` (plateau + surprise
  components, thresholds) · `prototype_router.py` (PRISM regime names + entropy) ·
  `lambda_net.py` (arms, walk-forward gate, Interval API) · `continual_costate.py`
  (arbitrate_architecture — the global-single-best arbiter this extends). Live pids: none
  disturbed (no training running; $0 numpy).

**Pointer 0.19108282 [contest-CPU] UNMOVED — all MEANS.**
