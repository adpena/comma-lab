# L5 v2 Duplicate Axis Row Hardening - 2026-05-16

## Scope

Closed a fail-open evidence-shape bug in the L5 v2 staircase. Probe observations
and gate artifacts previously normalized axis rows into a dictionary keyed by
axis. Duplicate `contest_cpu` or `contest_cuda` rows therefore collapsed with
last-write-wins semantics and could hide a bad row behind a later good row.

## Fix

- `evaluate_l5_v2_probe(...)` now emits
  `l5_v2_probe_axis_evidence_duplicate:<axis>` when duplicate exact-axis rows
  appear in one observation.
- `l5_v2_dispatch_readiness(...)` now emits
  `l5_v2_gate_artifact_semantics_invalid:<gate>:<section>:duplicate_axis:<axis>`
  when paired-axis gate artifacts contain duplicate axis rows.
- The stale L5 v2 staircase research-basis assertion now matches the canonical
  `time_traveler_l5_v2` source-basis list, including the source-basis sidecar
  citations.

## Evidence Axis

Planning/evidence-custody hardening only. No score claim, no promotion claim,
and no dispatch authorization.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_staircase_v2.py -q
.venv/bin/ruff check src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_staircase_v2.py
```

Results:

- `64 passed`
- `All checks passed`
