# DAG FEED — confound-pass follow-on hardening

Date: 2026-07-15
Catalog: NOT CLAIMED (stale counter collision at already-occupied #405)
Lane: `p0_confound_hardening_20260715`
Authority: BUILD + local verification only; score claim = false; launch = none.

## Triality edges

| Producer | Typed edge | Consumer | Containment |
|---|---|---|---|
| `VerdictLiveGap(every=K)` DSL Lever | `--verdict-live-gap-every K` | explicit all-run trainer observer | inference timing only; never consumed by loss/controller/checkpoint/archive |
| trainer default `-1` | `U_warm = ceil(2/(1-beta_ema))` and accepted EMA-update count | async + sync live-vs-EMA verdict observer | automatic warmup telemetry; `0` remains explicit opt-out |
| accepted-batch liveness | `0.02 < accepted_frac < 0.5` | `partial_freeze` L1 WARN | emit only; no mutation or halt |
| deterministic descending d_seg trace | strict descent predicate + verdict-trend canary | setup + baseline/async/sync verdict stamps + L3 clearance classification | pure Python known-effect signal; failed clearance alerts only |
| source wiring | three dated follow-on gates | `CONFOUND_GATES` and `_CONFOUND_STRICT` | live-count zero at atomic fix+gate landing |

Canonical-equations note: `src/tac/canonical_equations/confound_observability_20260715.py`.
Dependency-free executable predicates: `src/tac/confound_observability.py`.

## Byte-identity proof

Round-1 adversarial trace found no dataflow edge from an added value to model parameters,
loss, backward, gradients, optimizer inputs/state, EMA values, curriculum/controller decisions,
checkpoint selection, quantization, archive payload, or d_seg/d_pose calculation. The only new
runtime side effects are JSON telemetry and, during the EMA warmup observation window, an extra
read-only verdict inference over copied live parameters. The canary uses deterministic synthetic
Python dictionaries and does not import MLX. The accepted EMA-update counter is observational,
unpersisted, and is read only by the live-gap cadence predicate.

Therefore training/model/archive bytes are identity-preserved by construction. This is not a
score measurement and grants no promotion authority.

## Verification surfaces

- `src/tac/canonical_equations/tests/test_confound_observability_20260715.py`
- `src/tac/witness_control/tests/test_verdict_trend_alarm.py`
- `src/tac/tests/test_confound_gates.py`
- strict real-tree execution of the three dated follow-on gates
- Python compile of every changed Python source and test

The live C0 run directory is outside this DAG and was not accessed or modified.

## Review-tracker custody

`tools/review_tracker.py scan` and `mark-file ... --status reviewed` were invoked,
but this isolated worktree has no `duckdb` Python module and network access is
disabled. The tool failed closed before mutating its database/JSON snapshot. The
review itself is represented by the strict counterexample harness, focused lint,
compile checks, and this round-1 receipt; MAIN still owes canonical review-tracker
ingest if its merge environment supplies DuckDB.
