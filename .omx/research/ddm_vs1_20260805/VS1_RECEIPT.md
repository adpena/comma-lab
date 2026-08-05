# VS1 receipt - four-row hygiene batch

**Date:** 2026-08-05
**Cost:** $0
**Scorer:** none
**Launches:** none
**Score claim:** none

## Answer-first verdicts

| row | verdict | what changed | boundary |
|---|---|---|---|
| #846 | DONE | Re-scoped overbroad negatives at source: post-hoc pose-value storage is FORMULATION-scoped, exact reversible L3 raster residual storage is FORMULATION-scoped, and GR1 "no middle ground" is INSTANCE-scoped. | No new measurement; wording/registry only. |
| #839 | DONE | Added the four-name scorer-invisible convention and retagged the live priced rate instance as `COUNTED_PAYLOAD_RATE_CREDIT`. | Only `COUNTED_PAYLOAD_RATE_CREDIT` can feed waterfill rate columns. |
| #890 | CLASSIFIED | The detailed three-item #890 content is absent from repo stores; PH4 O1/O2/O3 are the consumable queue: O1 scorer-gated, O2 conditional on O1, O3 stale/done. | No scorer fired; no missing content invented. |
| #877 | STALE-CLOSED | Current code already treats printed `Final score` as rounded display and recomputes canonical score from components. | No code edit needed in VS1. |

## Row evidence

### #846

Recall found `ddm_ub1_untagged_verdict_scope_audit_20260801.md` and
`ddm_ng1_20260805/ng1_negative_results_audit.md` as the source audits. The source fixes landed in:

| file | fix |
|---|---|
| `.omx/research/ddm_gc14_first_descent_20260731.md` | Replaced "post-hoc store-apply RULED OUT by law" with a formulation-scoped pose-value-storage limitation; no seg/TR1 law is claimed. |
| `.omx/research/generator_description_crux_synthesis_20260719.md` | Replaced "plane-storage family closed" with `FORMULATION:EXACT_REVERSIBLE_L3_RASTER_RESIDUAL_RATE_DEAD`. |
| `.omx/research/ddm_gr1_granularity_rerace_20260730.md` | Marked the QA07 nested-rung/no-middle-ground result as INSTANCE scope and named the owed lower-convex-hull allocation. |
| `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` | Updated the DAG plane-storage section and equations sync line to the formulation anchor. |
| `.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md` | Re-scoped the SPEC note to exact reversible L3 raster residual storage only. |
| `.omx/state/canonical_equations_registry.jsonl` | Renamed the anchor to `exact_reversible_l3_raster_residual_rate_dead_formulation_20260719` and set verdict scope to `FORMULATION:EXACT_REVERSIBLE_L3_RASTER_RESIDUAL_RATE_DEAD`. |

### #839

Recall found four quantities sharing "scorer-invisible":

| canonical name | quantity | rate meaning |
|---|---|---|
| `RESIZE_KERNEL_NULLITY_DOF` | 80.67% resize-kernel dimension. | Not bytes. |
| `CERTIFIED_ZERO_WEIGHT_BLIND_MASK` | 22.70% / 230,904 camera pixels read by neither scorer. | Not bytes by itself. |
| `RANGE_A_COMPLEMENT_RENDER_ENERGY` | Approx. 52% measured range(A)-complement render/head-norm energy. | Precision/gauge signal, not a rate column. |
| `COUNTED_PAYLOAD_RATE_CREDIT` | Actual archive bytes removed from parser-consumed payload. | The only admitted waterfill rate column. |

The convention is now in `.omx/research/ddm_vs1_20260805/SCORER_INVISIBLE_NAMING.md`.
`P_NULL_GAUGE` in `src/tac/canonical_equations/ddm_m4_rate_floor_20260723.py` names
`COUNTED_PAYLOAD_RATE_CREDIT` explicitly, and the `pantheon_synergy` F5 row no longer consumes the
~52% gauge/energy number as bytes.

### #890

Recall sources:

| source | finding |
|---|---|
| `.omx/research/harness_tasklist_bridge_20260803.jsonl` | Has bare pending row #890: "physics vs photometrics scorer-readout asymmetry - 3 censored items on v4c/v4d photometric stage". |
| `.omx/research/ddm_ph4_physics_photometrics_dynamics_20260803.md` | Records a bounded absence of detailed #890 content in repo-visible stores and names O1/O2/O3 owed measurements. |
| `.omx/research/ddm_qj1_followon_backlog_join_20260804.json` | Repeats that #890 needs content quoted into repo; bare ID cannot be consumed. |

Classification:

| item | classification | fire order |
|---|---|---|
| PH4 O1, blind-set aimed d_pose A/B with bit-identical d_seg positive control | `SCORER_GATED_QUEUE` | First scorer slot after the current holder; measure `d_pose` and `d_seg` together at n600. |
| PH4 O2, byte question for rank-k blind-set correction | `SCORER_GATED_CONDITIONAL` | Fire only after O1 proves aimability; otherwise retire. |
| PH4 O3, Q3 visible-complement warp pass-through | `STALE_DONE_SCORER_FREE` | Already measured as `0.8902`; artifact `.omx/research/ddm_ph4_visible_set_warp_passthrough_cx1_20260803.json`, sha256 `ec850cfd8481b45e8fd1b3e21e53dfbf02bbfd0aef8bb5ea08761c43255449e7`. |

### #877

Recall found `.omx/research/ddm_mi1_measurement_integrity_20260804.md`; current code already carries the
cure:

| file | current behavior |
|---|---|
| `experiments/contest_eval.py` | Parses `Final score` into `reported_final_score_display_rounded`; sets `score` from `score_recomputed_from_components`. |
| `src/comma_lab/evaluate.py` | Recomputes `current_score` from SegNet, PoseNet, and rate components. |
| `src/tac/eval/auth_eval.py` | `best_score` prefers `computed_score`; `final_score` is a fallback only when components are absent. |
| `src/tac/tests/test_measurement_integrity.py` | Tests that rounded display is not canonical. |

## RECALL EVIDENCE

Sources searched: VS1 prompt, common contract, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
`docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `ddm_ub1`, `ddm_ng1`,
`ddm_mi1`, `ddm_ph4`, `harness_tasklist_bridge_20260803.jsonl`,
`ddm_qj1_followon_backlog_join_20260804.json`, and `.omx/state/canonical_equations_registry.jsonl`.

Queries used: `#846`, `post-hoc stored corrections`, `plane-storage`, `No middle ground pays`, `#839`,
`scorer-invisible`, `ker(A)`, `blind mask`, `range(A)`, `rate column`, `#890`, `physics vs photometrics`,
`scorer-readout asymmetry`, `censored`, `v4c`, `v4d`, `#877`, `Final score`,
`reported_final_score_display_rounded`, and `score_recomputed_from_components`.

Found beyond seed docs: NG1 had already summarized #846 but not fixed source wording; PH4 had already
proved #890's detailed content absent from repo stores and exposed the O1/O2/O3 queue; MI1 had already
fixed #877 in current code.

## Validation

Commands:

```bash
.venv/bin/python -m pytest src/tac/canonical_equations/tests/test_ddm_m4_rate_floor_20260723.py src/tac/tests/test_measurement_integrity.py
.venv/bin/python tools/review_tracker.py mark-file src/tac/canonical_equations/ddm_m4_rate_floor_20260723.py --status reviewed
.venv/bin/python tools/review_tracker.py mark-file src/tac/canonical_equations/ddm_m4_rate_floor_20260723.py --status reviewed
.venv/bin/python tools/review_tracker.py status | rg -n "ddm_m4_rate_floor_20260723|canonical_equations/ddm_m4"
```

Result: 10 pytest tests passed. Review tracker shows
`src/tac/canonical_equations/ddm_m4_rate_floor_20260723.py` at 8 reviewed / 8 total entities.

## NEXT_IF_RESUMED

1. For #890, wait for the scorer slot; fire PH4 O1 before O2.
2. For #839, retag old uses on touch using the four canonical names, not a broad sweep.
3. For #846, do not cite family/law scope unless a new measurement earns it.
4. For #877, keep exact-ranking consumers on `score_recomputed_from_components`; never sort or ledger by the rounded display line.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
