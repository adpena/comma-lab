---
title: Z1 scorer-conditional MDL operator surface landing
date: 2026-05-14
lane_id: lane_zen_floor_scorer_conditional_mdl_ablation_20260514
status: L1_IMPL_SURFACE_LANDED
score_claim: false
research_only: true
promotion_eligible: false
ready_for_exact_eval_dispatch: false
gpu_spend: false
evidence_axes:
  - parser-proven-archive-entropy
  - proxy-planning
  - cli-test
source_documents:
  - .omx/research/zen_floor_field_medal_grade_council_20260514.md#Decision-Z1
  - .omx/research/grand_council_maximize_value_with_time_traveler_seat_20260514.md#Decision-1
---

# Z1 scorer-conditional MDL operator surface landing

## Summary

This landing adds the operator-facing surfaces for Decision Z1 without touching
the parent-owned core module `src/tac/analysis/scorer_conditional_mdl.py`.

Added:

- `tools/compute_scorer_conditional_mdl_ablation.py`
- `tools/probe_zen_floor_disambiguator.py`
- `src/tac/tests/test_scorer_conditional_mdl_cli.py`

The compute CLI wraps the parent Z1 manifest builder. It accepts archive specs
as `label=path` or `label=path,parser=name`, accepts optional eval JSON specs
as `label=path`, and writes both JSON and markdown artifacts. All emitted
success and fail-closed payloads set:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- `gpu_required=false`

The probe consumes Z1 JSON and arbitrates between:

- static-floor interpretation: floor is primarily source plus scorer plus
  runtime conditional;
- substrate-engineering-scope interpretation: floor moves with better substrate
  and byte-to-scorer feature binding.

The probe remains `proxy_planning_only` unless a true scorer-feature binding
file exists and explicitly asserts readiness. Even then it remains a planning
artifact, not a score claim or dispatch authorization.

## Solver wire-in

- Sensitivity map: compute output preserves
  `tac.sensitivity_map.scorer_conditional_entropy_map_v1`.
- Pareto constraint: output is a lower-bound planning surface only; exact
  Pareto movement requires a byte-closed codec candidate and exact eval.
- Bit allocator: compute output emits `allocator_hook` with prioritized entropy
  rows.
- Cathedral autopilot: compute and probe outputs emit fail-closed
  `autopilot_rows` with zero dispatch cost and explicit blockers.
- Continual learning: no empirical score anchor is added in this landing.
- Probe disambiguator: `tools/probe_zen_floor_disambiguator.py` is the Z9 probe
  consuming Z1 output.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_scorer_conditional_mdl_cli.py -q
.venv/bin/python tools/compute_scorer_conditional_mdl_ablation.py --help
.venv/bin/python tools/probe_zen_floor_disambiguator.py --help
.venv/bin/python -m py_compile tools/compute_scorer_conditional_mdl_ablation.py tools/probe_zen_floor_disambiguator.py src/tac/tests/test_scorer_conditional_mdl_cli.py
```

Result:

```text
4 passed in 0.17s
```

Test coverage includes:

- archive CLI success path with `label=path` and optional `label=eval.json`;
- proxy-safe invariants on compute output;
- fail-closed missing archive handling with JSON and markdown artifacts;
- probe output fields and autopilot rows;
- true scorer-feature binding follow-up path that changes probe authority while
  preserving `score_claim=false`.

## Contest compliance

No remote dispatch, no GPU spend, no archive candidate, no score claim, and no
promotion language. The only axis material preserved from optional eval JSON is
archive-level metadata used as a scalar proxy. It is not promoted to true
scorer-conditional entropy.

## Remaining blockers

1. True scorer-feature binding is still missing. The current Z1 parent module
   correctly marks `true_scorer_feature_available=false`; conversion from proxy
   planning to true scorer-conditioned MDL needs byte-to-penultimate-feature
   saliency or component-response curves.
2. The lane registry entry for
   `lane_zen_floor_scorer_conditional_mdl_ablation_20260514` remains a planning
   registry row until the operator or parent marks this implementation surface
   as landed. This memo records the evidence for that update.
3. No A1/PR106 production archive run was executed in this patch; the tools are
   ready for the parent/operator to point at canonical archives and eval JSONs.
