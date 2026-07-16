# Costate-organ elevation — exact-factorized adjoint and consumption audit

**Task:** #516  
**Date:** 2026-07-16 UTC  
**Verdict:** `BUILT + BACKTESTED-PASS-DEVELOPMENT[#205 scalar/binding] + CONSUMED-IN-EXISTING-SHADOW-PATH`  
**Scope:** `[macOS advisory] NON-PROMOTABLE`; `score_claim=false`; `actuation=NONE`  
**Pointer:** unchanged. No run, stop, paid dispatch, checkpoint, config, or archive bytes were touched.

## Outcome

The costate organ now has one hybrid adjoint whose class direction is not learned:

\[
K=p_{\mathrm{visible}} B^\top
  \operatorname{diag}\!\left(\frac{\|\Delta w_{cc'}\|_2}{G_{cc'}}\right)B,
\qquad
\Lambda_{\mathrm{hyb}}(x,\phi,t)=\alpha_\psi(x,\phi,t)(-K\phi).
\]

`B` is the signed class-pair incidence matrix. The exact operator is symmetric positive semidefinite, rank four, and annihilates the all-class gauge direction. The learned part is restricted to the existing closed-form differentiated-RBF temporal posterior plus five shared non-negative amplitude coefficients (constant, two bounded temporal sensors, boundary occupancy, regularizer occupancy). It has **zero learned class-direction parameters**.

This arm extends the existing `lambda_net`/shadow controller. It is not a second controller. A share proposal may enter the existing recommendation ranker only after the established LOO + past-only walk-forward + binding-AUROC gate passes, and only levers with measured share variation can authorize DECIDE. Unvaried feature-structured values remain duty-to-measure.

## Exact / derived / learned provenance

| Quantity | Label | Value | Authority |
|---|---|---:|---|
| SegNet centered head rank | MEASURED | 4 | `segnet_head_rank4_linear_flipdist_v1` |
| certified zero-weight camera coordinates | MEASURED | 0.226969 | `realization_necessity_preimage_per_stratum_v1` |
| visible camera support used by aggregate organ | DERIVED | 0.773031 | `1 - 0.226969` |
| Road-Lane inverse-gain lambda ratio vs other-major median | DERIVED | 2.089623x | `lane_gain_chain_composed_v1`; `lambda proportional 1/G` |
| Road-Lane head/gain factor vs other-major median | DERIVED | 2.939451x | measured head normals composed with measured gains |
| factorized operator rank | DERIVED/VERIFIED | 4 | deterministic float64 incidence-Laplacian check |
| all-class gauge null residual, Linf | DERIVED/VERIFIED | 2.22e-16 | deterministic float64 check |
| learned amplitude coefficient count | BUILT | 5 | `ExactFactorizedAdjoint` |
| learned class-direction count | BUILT | 0 | structural test |
| residual ridge | POST-HOC DEVELOPMENT CONSTANT | 10.0 | selected during #205 build; independent compatible trajectory owed |
| Lane island persistence median | MEASURED | 0.625 | `lane_gain_chain_composed_v1`, n600 |
| Lane islands below persistence 0.5 | MEASURED | 0.419 | same |
| births / deaths per scored frame step | MEASURED upper bounds | 9.43 / 9.50 | same; 5% overlap/interpolation tolerance |
| event turnover per interior island-step | DERIVED upper-bound proxy | 0.99110 | `(births+deaths)/19.1`; not an exact saddle identity |

The explicit `apply_ker_a_mask` projection refuses shape broadcasting and zeros only coordinates certified invisible by the exact resize/preimage artifact. The controller-level aggregate uses the measured visible mass; pixelwise consumers must supply the actual mask.

## Backtest — honest axes and stop rules

Durable receipt: `.omx/research/costate_organ_elevation_backtest_20260716.json`, SHA-256 `267fd618e2377a7b6c6b32a0be42ca3b28a3072ea972b2fb012fa274e63eda3b`.

### #205 / v7.5.2 baseline — compatible

Source: `/Users/adpena/Projects/pact/experiments/results/levelset_v752_baseline_20260710T185913Z`; 10 verdicts, 248 dense loss rows, 22 levers, 9 joined intervals. The source `run.log` SHA-256 was identical before and after the audit.

| Gate | Model | Persistence | Verdict |
|---|---:|---:|---|
| scalar d_seg LOO MAE | 0.00324552 | 0.00369778 | BACKTESTED-PASS |
| scalar d_seg past-only WF MAE (7 folds) | 0.00185207 | 0.00279193 | BACKTESTED-PASS |
| binding AUROC | 0.82 | magnitude heuristic 1.00 | PASS floor `>=0.80`; heuristic remains better |
| per-class LOO MAE | 0.04253537 | 0.05788957 | wins |
| per-class past-only WF MAE | 0.02664125 | 0.01082251 | **LOSES** |

Therefore `BACKTESTED-PASS-DEVELOPMENT` means the existing scalar/binding tri-gate only. The residual ridge `10.0` was selected during this #205 build, so #205 is a development set, not an independent validation set. It does not authorize a claim that the arm generalizes per class or across runs. The current advisory proposal's noise band is shown in the shadow row; feature-structured, unvaried levers cannot authorize it.

### mod32cap — incompatible schema

The run has 41 verdicts and 1000 loss rows but yields zero interval records because the verdict telemetry lacks interval-aligned `d_seg_by_class`. Verdict: `UNAVAILABLE_INSUFFICIENT_INTERVAL_SCHEMA`. No #205 equivalence is inferred.

### C2 — rows pending

The current `c2_perclass_stratum_20260716` and `c2_witness_own_decomp_20260716` surfaces are analysis artifacts without `run.log`. Verdict: `PENDING_NO_RUN_LOG`. The backtest tool discovers these paths and will harvest them when compatible rows arrive; it does not touch the live C2 run or fabricate a result.

## Consumption audit

| Surface | Verdict | Last fired / evidence | Actual decision consequence |
|---|---|---|---|
| always-on shadow estimate/classify/recommend | CONSUMED | existing sidecar last row `2026-07-11T15:31:00Z`, ep225, `WATCH_NO_ACTION`; launcher -> observer -> `build_shadow_report` | owns ranked recommendation persisted to `costate_shadow.jsonl` |
| producer bridge, cross-run posterior, duty rank | CONSUMED | invoked by every shadow report | axis EV and owed-probe ordering enter SENSE/DECIDE |
| exact-factorized adjoint | CONSUMED | in-memory #205 shadow pass in receipt | entered the same recommendation list after BACKTESTED-PASS; identified levers only |
| Morse-Smale + #344 NCDE | CONSUMED | in-memory #205 row; #344 reported stable, no fire | next-stage-boundary advisory row; `predicted_dS=null`, `actuation=NONE` |
| CostateAgent DSL/panel | ORPHANED from always-on production | test/tool invocation only | now truthfully registers the factorized expert, but has no production authority |
| regime dispatcher | INERT for always-on recommendation | digest/backtest only | no causal shadow recommendation change |
| digest + witness introspection + dashboard | CONSUMED | schema/test verified; visible after MAIN landing and next refresh | exposes exact/learned split, admission confidence, event warning, and recommendation why |

The key repair is therefore not merely another tournament arm: `_factorized_overlay` and `_merge_factorized_candidate` live in `shadow_controller.py`, the call path already launched by `launch_witness_run.py`. The old DSL/router remains explicitly non-authoritative until an always-on consumer is intentionally routed through it.

## Event intelligence

The shadow report now composes two distinct instruments without conflating them:

1. Morse-Smale evidence: low Lane persistence and measured birth/death turnover yield a DERIVED saddle-node warning proxy and the Road-Lane critical pair-weight prior.
2. #344 NCDE: supplies trajectory asymptote/basin timing.

Their row is eligible only at a stage boundary. It never contains a fabricated score delta and cannot mutate a schedule. On the audited #205 snapshot the NCDE fit was stable (`R2=0.99699`) and did not fire; the event row is a next-boundary watch, not an alarm.

## Durable integration

- Code: `factorized_adjoint.py`; architecture `V_exact_factorized_residual` in the existing `lambda_net` registry.
- Production consumer: `shadow_controller.py`; fields persist inside the existing `ShadowReport` schema.
- DSL: existing `CostateAgentProgram` gains a canonical-equation sensor and the existing expert panel gains the factorized lens. No witness lever/flag was created.
- Equation: `hybrid_exact_factorized_costate_adjoint_v1`, registered append-only in `.omx/state/canonical_equations_registry.jsonl`.
- DAG: `.omx/research/costate_organ_elevation_DAG_FEED_20260716.md`.
- Observability: digest, witness-run introspection, and live dashboard render the exact/derived/learned split and event row.
- Reproducible audit: `tools/costate_organ_elevation_backtest.py`; source-log hash-before/hash-after certification.

## Tests

The final-tree focused command ran the factorization, λ-organ, production shadow controller, dashboard, witness introspection, and all three source-LawRef suites. Three consecutive clean passes each reported `122 passed, 10 skipped`; no implementation or artifact fix occurred between those passes. The command was:

```text
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/tests/test_costate_factorized_adjoint.py \
  src/tac/tests/test_lambda_net_costate_organ.py \
  src/tac/tests/test_witness_control_costate.py \
  src/tac/tests/test_dashboard_server.py \
  tools/test_witness_run_introspect.py \
  src/tac/canonical_equations/tests/test_segnet_head_rank4_flipdist_20260715.py \
  src/tac/canonical_equations/tests/test_realization_necessity_preimage_20260715.py \
  src/tac/canonical_equations/tests/test_lane_gain_chain_composed_20260716.py
```

## Stores consulted

- delegated authority prompt SHA-256 `ac866c8f5a0e721059e6eea9ded6cbffe606948b290735ffefdafc3b56e9df87`;
- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`;
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, `.omx/state/master_gradient_anchors.jsonl`, `.omx/state/modal_call_id_ledger.jsonl`, cost-band/continual-learning posteriors, canonical task status, and frontier scan surfaces;
- latest costate-controller, amortized-operator, producer-bridge, regime-dispatch, scorer-factorization, necessity/preimage, temporal-advection, and router-stability memos;
- graph-memory recall for `costate organ exact factorized adjoint rank-4 ker(A) gain prior event bifurcation trajectory controller consumption`;
- exact source artifacts `segnet_fractal_20260715/stage_a.json`, `lane_channel_refactor_20260716/s1_gain_chain.json`, `s3_dash_geometry.json`, `s4_events_t1_audit.json`, and `necessity_solver_20260715/asupport.json`;
- read-only #205/mod32cap/C2 experiment surfaces named in the backtest receipt;
- latest per-arm and broadcast inboxes before each checkpoint.

No new web/literature search was needed: the build consumed already-canonical scorer laws and the existing T/#344 temporal instruments. This avoids adding a paper-shaped novelty when the missing work was production consumption and scope discipline.

## MAIN landing review required

This worktree is not the source of truth. MAIN review must verify the serializer commit, inspect the hot state/DAG/equation appends, rerun the focused suite, and only then land. Global `lane_maturity validate` remains blocked by 110 pre-existing evidence paths absent from this isolated worktree; the task lane's own `impl_complete` and `strict_preflight` evidence is present, so this landing does not rewrite unrelated historical lanes. Until MAIN lands and the next observer refresh emits a durable row, the live sidecar remains the pre-task schema and the pointer remains unchanged.
