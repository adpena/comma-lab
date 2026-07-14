# COSTATE ORGAN ROUTER STABILITY — standalone DAG feed

**Node:** `FEED-costate-router-stability-20260714`  
**Status:** `BUILT`, `REAL-#205-BACKTESTED`, `ADVISORY`, `research_only=true`  
**Pointer delta:** `0`  
**Shared-DAG action:** intentionally none; main owns collision-safe merge.

## Node contract

```text
#426 costate organ + #436 regime dispatcher
  -> deterministic NumPy-fp32 gate + selection-margin certificate
  -> content-addressed DECIDE ledger
  -> exact APPLY replay / MISMATCH_ALARM
  -> sequential route-match calibration diagnostic
  -> forecast-driven K>1 shadow allocation
  -> custodied self-normalized clipped/masked IS architecture evaluation
```

## Incoming edges

- `#205 real trajectory`: 10 verdicts / 9 intervals / 7 walk-forward folds.
- `#436 regime policy`: transient -> `T_gp_costate_posterior`; plateau/uncertain -> persistence.
- `#434 synthetic costate`: training/proposal feed only; cannot satisfy real density or adoption custody.
- `surrogate_vjp_fidelity_metric`: supplies sister density-custody semantics only; no metric duplication.
- Molt prior art: fp32 router, route replay, IS pattern; no framework code or thresholds imported.

## Measured node outputs

- Exact zero fp32 gate margins at epochs 75 and 125: `2/7` folds.
- Dispatcher WF `0.0015959393896760557` vs global-single-best `0.00185206618604584` vs persistence `0.002791931483929152`.
- DECIDE/APPLY `REPLAY_MATCH`; historical mismatch prevalence `UNMEASURED_NO_HISTORICAL_APPLY_LEDGER`.
- Calibration `MIS_CALIBRATED_INSTANCE`: stable `3/5`, within-roundoff `2/2`, high-minus-low match rate `-0.4`.
- Sequential diagnostic posterior: explicit assumed `Beta(1,1)` -> terminal `Beta(6,3)`, mean `2/3`.
- Compute receipt: `K2_SHADOW_A_RIDGE_SOLVE_OWED_FOR_ALL_FOLDS`; current route unchanged; actuation none.
- IS: `BLOCKED_DISTRIBUTION_CUSTODY`.

## Triality

### Equation

```text
m_t = f32(s_t) - f32(median_f32(s_<=t))
a_apply(d_t) := a_decide[d_t]
q | y_1:t ~ Beta(1 + sum y_i, 1 + t - sum y_i)
w_i proportional to M_i clip_[l,u](p_live(g_i)/p_bt(g_i))
```

Canonical sidecar: `src/tac/canonical_equations/costate_router_stability_20260714.py`, equation ID `costate_router_stability_v1`.

### DSL

`RouterStabilitySpec` declares `gate_dtype=numpy.float32`, exact replay, unfrozen learning, self-normalized clipped/masked IS, required density custody, no default clip bounds, explicit Beta prior provenance, and `k2_shadow_with_A_ridge_solve`. No CLI/trainer flag.

### DAG

This file is the standalone feed. Main may merge only after checking current hot-DAG ownership.

## Six-hook wire-in

1. **Sensitivity contribution:** gate margin and posterior route-match uncertainty annotate which costate recommendation is numerically fragile.
2. **Pareto constraint:** no score/byte promotion effect; advice is dominated unless replay-consistent and custody-complete.
3. **Bit allocator:** none direct; router is MEANS, not witness payload.
4. **Cathedral/autopilot:** consumes blocker or K2 shadow request only; cannot actuate heavy/live work.
5. **Continual learning:** future custodied route/outcome rows update calibration; synthetic rows cannot satisfy real adoption authority.
6. **Probe disambiguator:** selected-tool vs `A_ridge_solve` same-checkpoint K2 shadow comparison when calibration is not trustworthy.

## Outgoing gates

- `FEED-costate-router-replay-consumption`: require DECIDE/APPLY bijection and zero unacknowledged mismatch alarms.
- `FEED-costate-router-density-custody`: require hashed live/backtest densities, common regime schema, support, executed decision rows, and provenance-derived clip bounds.
- `FEED-costate-router-transfer`: require at least two independent real trajectories; rerun confidence calibration and IS-weighted architecture evaluation.
- `FEED-costate-router-fallback-adoption`: require K2 same-checkpoint outcome rows showing when `A_ridge_solve` improves realized forecast debt. Current `n=1` does not license fallback replacement.
- `FEED-costate-router-live-apply`: remains operator-GO gated after all advisory gates; this node supplies no launch authority.

## Reactivation criteria

Re-open the distribution-shift verdict only when the custody gate is complete. Re-open margin calibration after a second independent real trajectory or new executed route/outcome rows. Do not add FORE/HCM/TOFU/CL estimators here; treat them as future custody/admission feeds.

No negative here closes a family. `MIS_CALIBRATED_INSTANCE` queues past-only margin-to-reliability calibration plus K2 same-checkpoint comparison; `BLOCKED_DISTRIBUTION_CUSTODY` queues real density/support/executed-row custody. The optimal-form router/forecast family remains intact.
