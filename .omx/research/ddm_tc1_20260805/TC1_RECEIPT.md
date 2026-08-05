# TC1 Receipt - IG1-F1 Cross-Step Coupling Table From Recorded Solves

## Answer First

TC1 did not recover a true cross-step interaction matrix
`C[t,t'] = DeltaS({t,t'}) - DeltaS({t}) - DeltaS({t'})`.

Verdict: `RECORD-CENSORED / ALREADY-EMBODIED`. The recorded receipts preserve cumulative solver
trajectories and TJ1 stop-law fields, but they do not preserve the counterfactual fields a true
`C[t,t']` table requires: independently replayed single-step deltas, paired-step deltas, or
per-step gradients/secants. The Fisher-conditioned analogue is also unsupported: gradients,
secants, or Fisher-vector payloads are absent in every completed source inspected.

The weaker supported table is a cumulative secant/curvature table over recorded proxy-flip
trajectories. It shows strong diminishing returns, not a new off-diagonal routing law:

| source | denominator | proxy objective gain 0->25 | 25->50 | 50->75 | 75->100 | interpretation |
|---|---:|---:|---:|---:|---:|---|
| OD3 terminal truth curves | n32 for 0->25; n22 for 25->50; n1 for 50->75 | 8164 | 1101 | 26 | n/a | Terminal rows are already in TJ1-style `marginal_below_bar` territory. |
| SQ1/CW1 uncap50 | n32 | 18531 | 1692 | n/a | n/a | Cap-bound floor improved; TJ1 already consumed this shape. |
| SQ1/SQ2 uncap100 | n32 | 18531 | 1692 | 724 | 449 | Longer solve still diminishes; SQ2 gate refused promotion on pose/R8 and cap class. |
| OD3 k8/k12 worst-4 panels | n4 for 0->25; n3 for 25->50 | 967 | 161 | n/a | n/a | Same Stage-1 curves as terminal worst panel; k changes pose carriage, not cross-step solve coupling. |

Does it reorder candidates versus current practice? No admissible reorder. OD3's weak curve
diagnostics are not an independent stable ordering signal: on the n32 terminal rows, observed
last 5-step gain versus TJ1 marginal score has Spearman rho `-0.011` (n32), after-25 gain versus
projected remaining score has rho `0.245` (n32), and after-25 gain versus steps-run has rho
`0.824` (n32), meaning the largest weak signal mostly says "this row ran longer." The ST2 prior
still has zero consumable overlap with the OD2/OD3 pair ids. Therefore IG1-F1 closes as already
embodied by `trajectory_derived_stopping_law_v1`; no new solver/controller lands.

## Data Coverage Denominator

| source | rows inspected | cumulative curve present | single-step counterfactual delta | paired-step counterfactual delta | gradient/secant/Fisher fields | TC1 use |
|---|---:|---:|---:|---:|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_js1_n32_terminal_seg100_cprime_k4.json` | 32 | 32/32 `stage1.selected_curve`; common steps 0..25, variable to 75 | 0/32 | 0/32 | 0/32 | Weak secant table only; true `C` unsupported. |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k8_worst4.json` | 4 | 4/4; common steps 0..45 | 0/4 | 0/4 | 0/4 | Weak worst-panel secants; pose-k diagnostic only. |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k12_worst4.json` | 4 | 4/4; common steps 0..45 | 0/4 | 0/4 | 0/4 | Same Stage-1 weak secants; k12 is not a cross-step measurement. |
| `.omx/research/ddm_od3_20260805/OD3_AGGREGATE.json` | aggregate | terminal aggregates only | absent | absent | absent | Used for denominators, live frontier arithmetic, and OD3 boundary. |
| `.omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json` | 32 | 0/32 selected curves; terminal Stage-1 fields only | 0/32 | 0/32 | 0/32 | Terminal comparison only; no weak curve table. |
| `.omx/research/ddm_od2_20260805/OD2_AGGREGATE.json` | aggregate | terminal aggregates only | absent | absent | absent | Used to compare OD2 cap-bound state versus OD3 terminality. |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap50_cw1.json` | 32 | 32/32 `solved_convergence_curve`; common steps 0..50 | 0/32 | 0/32 | 0/32 | Weak secant table only. |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json` | 32 | 32/32 `solved_convergence_curve`; common steps 0..100 | 0/32 | 0/32 | 0/32 | Weak secant table only. |
| `.omx/research/ddm_tj1_20260805/trajectory_replay.json` | 2 controls + SQ2 target | aggregate trajectory replay controls | absent | absent | absent | Confirms current consumer/law already uses these curves scorer-free. |
| `.omx/research/ddm_od9_20260805/od9_js1_persist_n32_cprime_k4.json` | 26 live rows found | live/unlanded persistence rows | not consumed | not consumed | not consumed | Found during recall, not used for verdict because OD9 is live/unlanded and is the named consumer. |

## Weak Secant Table

All entries below are proxy-objective changes from recorded curves, not exact score moves and not
true `C[t,t']` entries. Score-unit conversions use the exact flip exchange
`100/(600*384*512) = 8.477105034722222e-07 S/flip`.

| source | interval | rows with both checkpoints | total proxy objective gain | mean gain/row | total S-equivalent |
|---|---:|---:|---:|---:|---:|
| OD3 terminal | 0->25 | 32 | 8164 | 255.125 | 0.006920708550 |
| OD3 terminal | 25->50 | 22 | 1101 | 50.045455 | 0.000933329264 |
| OD3 terminal | 50->75 | 1 | 26 | 26.000000 | 0.000022040473 |
| OD3 k8 worst-4 | 0->25 | 4 | 967 | 241.750000 | 0.000819736057 |
| OD3 k8 worst-4 | 25->50 | 3 | 161 | 53.666667 | 0.000136481391 |
| OD3 k12 worst-4 | 0->25 | 4 | 967 | 241.750000 | 0.000819736057 |
| OD3 k12 worst-4 | 25->50 | 3 | 161 | 53.666667 | 0.000136481391 |
| SQ1/CW1 uncap50 | 0->25 | 32 | 18531 | 579.093750 | 0.015708923340 |
| SQ1/CW1 uncap50 | 25->50 | 32 | 1692 | 52.875000 | 0.001434326172 |
| SQ1/SQ2 uncap100 | 0->25 | 32 | 18531 | 579.093750 | 0.015708923340 |
| SQ1/SQ2 uncap100 | 25->50 | 32 | 1692 | 52.875000 | 0.001434326172 |
| SQ1/SQ2 uncap100 | 50->75 | 32 | 724 | 22.625000 | 0.000613742405 |
| SQ1/SQ2 uncap100 | 75->100 | 32 | 449 | 14.031250 | 0.000380622016 |

This table supports only the existing trajectory-stopping conclusion: late increments become small
relative to early increments, so safety caps must not be called convergence, and terminal OD3 rows
must be stopped by marginal score/compute rather than cap-best wording. That is exactly TJ1's
current `trajectory_derived_stopping_law_v1` surface.

## Recall Evidence

| query / read | scope searched | finding beyond charter seeds | plan impact |
|---|---|---|---|
| `tc1_prompt`, `_common_contract`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, operating manual, live hot state | required governing files | TC1 is scorer-free, OD9 owns successor work, pointer line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`. | Kept this as a report-only receipt with no source edits and no scorer calls. |
| `sha256sum /Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_js1_n32_terminal_seg100_cprime_k4.json` | OD3 raw source | SHA matched charter exactly: `5f7f934e6bafa440572577509e0e733ab3c5e80940d8cf11178e91ecb93bd4ca`. | Consumed OD3 as the primary source. |
| `IG1-F1`, `C[t,t']`, `trajectory coupling`, `Fisher-conditioned` | `.omx/research/ddm_ig1_20260805`, arm queue, hot state | IG1-F1 was queued only after OD3 landed; falsifier says near-diagonal or no reorder closes IG1 already-embodied. | Set verdict target to reorder/no-reorder, not a new equation. |
| `trajectory_derived_stopping_law_v1`, `marginal_below_bar`, `safety_bound_REPORTED` | `.omx/research/ddm_tj1_20260805`, `src/tac/optimization/trajectory_stopping.py`, canonical equations list | Existing canonical law already consumes cumulative trajectories and labels safety caps separately from convergence. | Classified the weak secant table as already embodied, not a new controller. |
| `v19b synergy joint remeasure stack amplification`, `+0.0805` | DAG and research memos | v19b remains the prior that true joint replay can reveal non-additivity, but it was a joint remeasurement stack, not reconstructable from TC1 cumulative curves. | Prevented laundering weak curve curvature into a true `C` claim. |
| `ST2 scorer-native Fisher targeter` plus OD3 ST2 intersection sidecar | ST2 and OD3 receipts | OD3 found zero usable ST2 overlap for the OD2 pair ids. | No ST2 reorder can be claimed on this denominator. |
| `fd1 scorer metric GN fd2 realization gap uint8` and canonical equations Fisher rows | research memos and `tools/list_canonical_equations.py --json` | Fisher/margin/natural-gradient geometry is already represented by existing rows including `frozen_scorer_fisher_curvature_margin_colocation_v1`, `ce_softmax_mirror_descent_natural_gradient_v1`, `bregman_dual_metric_squared_hessian_v1`, and `trajectory_derived_stopping_law_v1`. | No duplicate IG equation; TC1 stays diagnostic. |
| `.omx/research/ddm_od9_20260805/od9_js1_persist_n32_cprime_k4.json` | full-corpus recall beyond seeds | Found 26 live/unlanded persistence rows. | Not consumed into TC1 evidence; OD9 remains the named consumer if future instrumentation is needed. |

## Evidence And SHA Table

| path | bytes | sha256 |
|---|---:|---|
| `.omx/tmp/codex_runs/tc1_prompt.md` | 2648 | `190139b4198d58f8b5eb7d2e76b42882a9b524847b0eebc492202a2456a32508` |
| `.omx/tmp/codex_runs/_common_contract.md` | 4124 | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` |
| `.omx/research/ddm_ig1_20260805/IG1_CROSSWALK_RECEIPT.md` | 15393 | `da2d7d7714562355f0b525edcbbb137fcae38ec6bc7fe2350f71a518e5df4335` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_js1_n32_terminal_seg100_cprime_k4.json` | 312293 | `5f7f934e6bafa440572577509e0e733ab3c5e80940d8cf11178e91ecb93bd4ca` |
| `.omx/research/ddm_od3_20260805/OD3_AGGREGATE.json` | 16678 | `3dc164c4301cb8fd6e75ce9f8a1b85b3a2a1ab38df0bd435871d36a6b5e531c3` |
| `.omx/research/ddm_od3_20260805/OD3_TERMINALITY_RECEIPT.md` | 9618 | `549fca629c8273839e040bdbc13c2391669a59aa9cd94cc7f48de76a23259670` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k8_worst4.json` | 40292 | `f16b2f62c531f2c1e694ba543a387c789192519eb70812eac9a7e543eade68c4` |
| `/Volumes/VertigoDataTier/pact/ddm_od3_20260805/od3_capacity_sweep_k12_worst4.json` | 40312 | `b9b5ea03810124d4fd99ca266f4427b57431bc768c250ab08251d540d8211581` |
| `.omx/research/ddm_od2_20260805/od2_js1_n32_cprime_k4.json` | 103690 | `fd1016751e4668ff786692f52f91d924be97081a70a20d11e470150aaf85c6af` |
| `.omx/research/ddm_od2_20260805/OD2_AGGREGATE.json` | 31831 | `43c97e844c23c00b5ad7367e147735587e00dec21b2f274ebfef7770b32a3ace` |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap50_cw1.json` | 131241 | `b412bb4d1a31e83a4d31a0600102dba0b1549bc02b27c89ade725d3b89676998` |
| `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32_uncap100_sq2.json` | 188487 | `dc7ecfe5c1578cc6a7f2668c070f04251b7e570a3e288d2789364d4e8ecead0b` |
| `.omx/research/ddm_tj1_20260805/trajectory_replay.json` | 15101 | `f55bbc2be252806d81fa411d35aaf2aec3a3cfd73d6fdb2962279f2aeefcd1ce` |
| `.omx/research/ddm_sq2_20260804/sq2_gate_verdict.json` | 2340 | `455e8b9f51b33a2f17c9460d480d896a6b923169c512460442f3fe6529709970` |
| `.omx/research/ddm_cw1_cap_artifact_uncap_and_guard_20260804.md` | 5817 | `60960a16e62fc041f3ba5dabaf1bd091c0bd082a6bb041f46f105db07ac4ce0b` |

## Follow-On Disposition

- IG1-F1 trajectory coupling table: `FIRED_AND_CLOSED` by this receipt as
  `RECORD-CENSORED / ALREADY-EMBODIED`.
- New solver/controller from IG1-F1: `FOLDED_NONE`. No true `C[t,t']`, no Fisher-conditioned
  analogue, and no admissible reorder were measured.
- Future true-C instrumentation: `FOLDED_INTO_OD9_OR_NEXT_TRAJECTORY_PERSISTENCE_TOUCH`, not a
  standalone launch. Exact fire order if that code is otherwise being edited: persist per-step
  action ids, independent single-step replay deltas, paired-step replay deltas, and gradient/secant
  vector hashes under the same receiver/coder denominator. Until those fields exist, do not call a
  cumulative curve a cross-step interaction matrix.

## Next If Resumed

1. Do not rerun OD3 or SQ1/SQ2/CW1 for TC1; their hashes above are the consumed record.
2. If OD9 lands a completed persistence receipt, inspect it as a new producer only after it is
   landed and hash-custodied. Treat the live 26-row untracked file as non-authority until then.
3. If a future arm needs true `C[t,t']`, instrument the producer before the solve runs. Required
   fields are independent single-step deltas, paired-step deltas, and either gradients/secants or
   an explicit `fisher_conditioned_unavailable` field per step.
4. Keep IG1 closed as already embodied unless a future completed producer shows a true
   counterfactual interaction table that reorders OD9 leg B / shared parametrization versus TJ1,
   ST2, fd1/fd2, and current Fisher-margin practice.

```json
{
  "schema": "ddm_tc1_ig1_f1_receipt.v1",
  "axis": "[research-signal] scorer-free replay over recorded receipts",
  "score_claim": false,
  "promotion_eligible": false,
  "scorer_forwards_executed": 0,
  "evaluate_py_run": false,
  "true_cross_step_C": {
    "status": "UNSUPPORTED_RECORD_CENSORED",
    "required_fields_absent": [
      "independent_single_step_delta",
      "paired_step_delta",
      "per_step_gradient_or_secant",
      "per_step_fisher_vector"
    ],
    "sources_with_required_fields": 0,
    "sources_inspected": 9
  },
  "weak_secant_table": {
    "status": "SUPPORTED_CUMULATIVE_CURVES_ONLY",
    "verdict": "diminishing_returns_already_embodied_by_trajectory_derived_stopping_law_v1",
    "reorders_current_practice": false,
    "od3_rank_checks": {
      "last_gain_vs_tj1_marginal_s_spearman": -0.011363636363636465,
      "after25_gain_vs_projected_remaining_s_spearman": 0.24486803519061584,
      "after25_gain_vs_steps_run_spearman": 0.8244134897360704,
      "n": 32
    }
  },
  "follow_on_disposition": {
    "ig1_f1": "FIRED_AND_CLOSED_RECORD_CENSORED_ALREADY_EMBODIED",
    "new_solver_or_controller": "FOLDED_NONE",
    "future_true_C_instrumentation": "FOLDED_INTO_OD9_OR_NEXT_TRAJECTORY_PERSISTENCE_TOUCH"
  },
  "frontier": {
    "own_vehicle": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]",
    "contest_pointer": "borrowed/unmoved"
  }
}
```

## Boundary

Measured in this TC1 unit: zero new scorer measurements; a scorer-free structural audit of recorded
trajectory fields; weak cumulative secant totals over OD3/SQ1 curves; OD3 n32 rank diagnostics from
recorded curves only.

Not measured: any true counterfactual `C[t,t']`, any Fisher-conditioned cross-step matrix, any new
archive bytes, any receiver-closed OD9 packet, any n600 scorer job, or any `upstream/evaluate.py`
score.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
