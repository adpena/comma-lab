# Codex findings — costate-organ router stability and forecast calibration

**Date:** 2026-07-14 UTC  
**Lane:** `costate_organ_router_stability`  
**Status:** `BUILT + $0 REAL-#205 BACKTESTED`, `ADVISORY`, `research_only=true`  
**Pointer:** **UNCHANGED** — `0.1910828242 [contest-CPU Linux x86_64]`, archive SHA prefix `ad02b0124cbb`; defensive borrowed-lineage bank `0.1880443979880752 [contest-CPU]`. This landing is MEANS. No byte-closed exact evaluator row exists.

## Executive verdict

The Molt router-stability patterns fit the costate organ, but the real #205 re-derivation found a sharper result than a generic port:

1. **Gate drift pathology: PRESENT at a boundary.** Epochs 75 and 125 are exact NumPy-fp32 `recent_slope == running_median` ties: `2/7` folds have zero router margin. The other five folds are far beyond fp32 roundoff. The legacy comparison was repeatable only through an implicit `>=`; the new certificate makes dtype, distance, ULP guard, policy digest, and tie law explicit.
2. **DECIDE/APPLY mismatch: STRUCTURALLY POSSIBLE before, now guarded.** There was no historical APPLY ledger, so a historical mismatch rate is **UNMEASURED**, not zero. The new content-addressed ledger replays the exact selected arm, appends `MISMATCH_ALARM`, and fails closed on divergence while leaving later router learning unfrozen.
3. **Backtest/live shift: `BLOCKED_DISTRIBUTION_CUSTODY`.** The canonical OPE receipt has one independent real trajectory, zero production causal manifests, zero executed decision rows, zero coverage rows, and no separately hashed visited-live density or derived clip bounds. The self-normalized clipped/masked IS implementation refuses to invent uniform weights.

Top discovery: the margin certificate is **not calibrated as correctness confidence on this instance**. Fp32-stable folds matched the fold oracle `3/5` with mean error `0.0018393701`; both roundoff-tie folds matched `2/2` with mean error `0.0009873600`. Thus high-minus-low match rate is `-0.4`, not positive. Scoped verdict: `MIS_CALIBRATED_INSTANCE`; no transfer claim from `n=1`.

This does **not** close the margin-calibration, router, forecasting, or costate-organ family. The measured object is the exact current #436 rule in its fully deterministic form, but the calibration sample is still one trajectory. Reformulation queue: (a) accrue independent real trajectories and executed route/outcome rows; (b) fit/calibrate a margin-to-reliability transform on past-only folds; (c) run same-checkpoint K2 selected-tool versus `A_ridge_solve`; (d) apply custodied IS before any live-distribution claim. `BLOCKED_DISTRIBUTION_CUSTODY` is an admission gate, not a family verdict.

## Real #205 receipt

- Input: `experiments/results/levelset_v752_baseline_20260710T185913Z`, 10 verdicts, 9 intervals, 7 walk-forward folds, seed 0.
- Run custody: `daemon.log` SHA256 stayed `7fdc44d19946121fb18e35060f5146bf1f48dea81c08891f8f4477d42b0bed82` before and after. The run directory was read-only.
- Output: `experiments/results/costate_organ_router_stability_20260714/costate_organ_backtest_20260714T115500Z.json`, SHA256 `939af8df48b644646bfed406564519cb064e13ca3b7747c2203871db4b134e23`.
- Replay: `costate_router_replay_20260714T115500Z.jsonl`, SHA256 `34441f31876406353445a00e75ec01813e5c5f9ebd27144da951057115949c23`; decision `3b9b225c617e58324e242183a433397df9e53d58c4f1796418379830870821d0`; `REPLAY_MATCH`.
- Legacy #436 reproduced: dispatcher WF MAE `0.0015959393896760557`; no-meta guard `0.0017378015724245231`; global-single-best `T_gp_costate_posterior` `0.00185206618604584`; persistence `0.002791931483929152`; oracle-route matches `5/7`.

All numbers above are **MEASURED on this macOS-CPU advisory backtest**, not a contest score.

## Forecasting frame, bounded honestly

The organ forecasts marginal trajectory movement and uses that forecast to allocate route and compute. The new narrow diagnostic updates a sequential Beta-Bernoulli posterior over the **walk-forward route-match path**:

```text
q | y_1:t ~ Beta(alpha_0 + sum_i y_i, beta_0 + t - sum_i y_i)
```

With the explicit **ASSUMED**, data-neutral `Beta(1,1)` prior and five matches in seven folds, the terminal diagnostic is `Beta(6,3)`, mean `2/3`, standard deviation `0.1490712`. This is not a posterior over the full physical training trajectory and is not a promotion threshold.

Because the confidence direction failed, every fold emits an advisory `K=2` same-checkpoint shadow request `{selected tool, A_ridge_solve}`. The current route is unchanged and `actuation=NONE`. This measures whether the architectural incumbent is a sound fallback before any deferral policy exists. Per the later fleet directive, FORE/HCM/TOFU/CL remain future custody/admission feeds, not duplicate estimators.

## Built stabilizers

- `router_stability.py`: deterministic NumPy-fp32 gate, selection-margin/ULP certificate, content-addressed DECIDE/APPLY replay, durable mismatch alarm, custodied self-normalized clipped/masked IS, sequential calibration diagnostic, and no-launch K>1 allocation receipt.
- `regime_dispatch.py`: the #436 dispatcher consumes the exact fp32 certificate and emits gate plus calibration telemetry for every fold.
- Typed DSL: `RouterStabilitySpec` fixes gate dtype, requires replay, forbids freezing learning, requires density custody, leaves IS clip bounds unset, records the assumed Beta prior, and names `k2_shadow_with_A_ridge_solve`. No trainer flag was invented.
- Backtest tool: emits the replay ledger, IS blocker, forecast calibration, and compute allocation in one receipt.
- Canonical equation: `costate_router_stability_v1` records margin, replay, sequential route reliability, and IS laws. Registration is explicit only; import has no shared-registry side effect.

## Canonical laws

```text
m_t = f32(s_t) - f32(median_f32(s_<=t))
g_t = plateau if m_t < 0 else transient
a_apply(decision_id) := a_decide[decision_id]
r_i = p_live(g_i) / p_backtest(g_i)
w_i = n_M M_i clip_[l,u](r_i) / sum_j M_j clip_[l,u](r_j)
L_hat(a) = sum_i w_i loss_i(a) / sum_i w_i
```

Exact ties use the preregistered #436 law: slope equality selects transient; surprise equality does not defer. IS requires real density hashes, shared regime schema, support, and provenance-derived clip bounds.

## Composition with #434 and surrogate arm

Synthetic #434 rows may train or propose experts, but they cannot manufacture visited-live density, support, or real adoption authority. Real-only walk-forward/OPE remains the gate. The distilled-surrogate arm's quotient-VJP metric remains owned by the sister surface; this landing consumes only its density-custody requirement and does not duplicate its estimator.

## Prior art, not code lift

[NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt) documents fp32 router logits, exact routing replay without freezing router learning, and clipped/masked importance correction. The [PR2 paper](https://arxiv.org/abs/2606.00395) supplies the routing-replay context. Pact imports no Molt framework code, CUDA assumption, LLM token semantics, or thresholds; only the three stability patterns are mapped.

## Lateral finding

The current PRISM explanation faithfulness audit is `FAILED_CURRENT_INSTANCE`: max relative suppression gap `0.3980379017 > 0.35`. Verdict scope: prototype-explanation counterfactual on this current run only. It neither falsifies the router-stability mechanisms nor licenses live adoption; it is a separate observability debt.

## Round-1 adversarial review

- **PASS:** no launch, subprocess, live-run mutation, score claim, or operator-GO bypass exists in the new module.
- **PASS:** APPLY validates the content address and selected tool; mismatch writes an alarm before raising.
- **PASS:** DECIDE and APPLY rows are flushed with `fsync` before returning.
- **PASS:** future decisions remain learnable (`router_learning_frozen=false`).
- **PASS:** missing density, hashes, support, or clip provenance fails closed; no uniform fallback.
- **PASS:** hindsight oracle labels appear only in the backtest calibration diagnostic, never the dispatcher.
- **PASS:** custom Beta priors require explicit provenance; the default receipt cannot falsely label a caller-supplied prior as `Beta(1,1)`.
- **PASS:** canonical equation import does not mutate the hot shared registry.
- **OPEN/BLOCKED:** no historical APPLY rows, so mismatch prevalence is unmeasured.
- **OPEN/BLOCKED:** no live density/support/executed decision custody, so IS robustness and cross-distribution agreement are unmeasured.
- **OPEN/INSTANCE:** margin confidence is directionally miscalibrated on `n=1`; K2 shadow measurement is owed before fallback adoption.

Assumption-challenge axis: the shared assumption “greater distance from the fp32 boundary implies more reliable expert selection” is falsified **for this instance**. Violating it does not remove margin telemetry; it turns margin into a calibration feature whose mapping must be learned and checked from real route/outcome custody.

Verification: `81 passed in 18.73s`; Ruff PASS; `py_compile` PASS; `git diff --check` PASS.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/organ_regime_conditional_dispatch_436_20260711.md`
- `.omx/research/adaptivebayes_costate_intrinsictime_20260713.md`
- `.omx/research/organ_ope_support_first_20260713.json`
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`
- `reports/latest.md`, `.omx/state/master_gradient_anchors.jsonl`
- live per-arm and fleet inboxes through `2026-07-14T11:44:35Z`

## Verdict scope and pointer delta

`INSTANCE`: one real #205 trajectory; advisory costate-router selection; `[macOS-CPU advisory]`; no contest score, no live schedule mutation, no heavy actuation, no cross-trajectory claim. Pointer delta: **0**.
