# Exact-Eval Custody Authority Hardening - 2026-05-16

## Scope

This landing consolidates repeated exact-eval custody predicates into
`tac.exact_eval_custody` and wires that authority into:

- `tac.optimization.prediction_band`
- `tac.optimization.l5_v2_probe_disambiguator`
- `tac.optimizer.exact_readiness`
- `tools/parallel_dispatch_top_k.py`
- `tools/operator_briefing.py`

## Why

The same SHA-256, archive/runtime identity, metric-field, and score-formula
checks were hand-rolled in several L5/Cathedral/dispatch surfaces. That made it
easy for one surface to become stricter while another still accepted incomplete
custody as rank, architecture-lock, or paid-dispatch authority.

## Changes

- Added shared helpers for SHA normalization, runtime-tree SHA extraction,
  exact contest score computation, and axis evidence validation.
- Kept existing public blocker vocabularies stable in prediction-band and
  L5-v2 code while delegating the low-level evidence checks to the shared
  authority.
- Tightened L5-v2 probe arbitration: architecture lock now requires every
  required candidate to have an eligible paired exact-eval observation, not just
  a present candidate ID plus one eligible winner.
- Tightened paid fan-out authority in `parallel_dispatch_top_k`: broad
  `contest_generalized` context is no longer enough. Paid dispatch requires the
  literal `contest_exact_eval` target marker.
- Tightened generic exact-ready fan-out: selected exact-ready rows now run
  through the same live-custody/terminal-evidence audit even when the input file
  uses a noncanonical `dispatch_ready` schema.
- Tightened terminal-claim replay protection: same-lane/same-archive rows with
  a runtime-tree SHA mismatch remain blocked unless the candidate explicitly
  declares `score_affecting_runtime_changed=true`.
- Tightened prediction-band evidence: landed exact anchors must use canonical
  `contest_cpu` / `contest_cuda` axes with inflate/evaluate device custody; a
  `mixed` landed band requires paired CPU and CUDA observations.
- Tightened operator briefing: `ready_for_submit` now requires the complete
  local gate conjunction (static preflight, compliance, payload diff, dry run,
  missing-env, blockers, and terminal evidence), not the packet flag alone.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_exact_eval_custody.py \
  src/tac/tests/test_prediction_band.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py \
  src/tac/tests/test_optimizer_exact_readiness.py \
  src/tac/tests/test_dispatch_command_builder_shapes.py \
  src/tac/tests/test_operator_briefing.py::test_exact_eval_packet_ready_for_submit_requires_all_static_and_env_gates \
  src/tac/tests/test_operator_briefing.py::test_briefing_json_composite_has_all_three_keys \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py \
  tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q
```

Result: `108 passed`.

```bash
.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q
```

Result: `17 passed`.

```bash
.venv/bin/python -m ruff check \
  src/tac/exact_eval_custody.py \
  src/tac/optimization/prediction_band.py \
  src/tac/optimization/l5_v2_probe_disambiguator.py \
  src/tac/optimizer/exact_readiness.py \
  tools/parallel_dispatch_top_k.py \
  tools/operator_briefing.py \
  src/tac/tests/test_exact_eval_custody.py \
  src/tac/tests/test_prediction_band.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py \
  src/tac/tests/test_optimizer_exact_readiness.py \
  src/tac/tests/test_dispatch_command_builder_shapes.py \
  src/tac/tests/test_operator_briefing.py \
  tests/test_parallel_dispatch_top_k_exact_ready_audit.py
```

Result: `All checks passed!`.

```bash
.venv/bin/python -m py_compile \
  src/tac/exact_eval_custody.py \
  src/tac/optimization/prediction_band.py \
  src/tac/optimization/l5_v2_probe_disambiguator.py \
  src/tac/optimizer/exact_readiness.py \
  tools/parallel_dispatch_top_k.py \
  tools/operator_briefing.py
```

Result: passed.

## Evidence Semantics

This is a hardening landing only. It claims no score movement and dispatches no
provider jobs. It reduces false-authority risk in L5-v2 staircase selection,
Cathedral prediction-band rank reward, and exact-eval paid dispatch fan-out.
