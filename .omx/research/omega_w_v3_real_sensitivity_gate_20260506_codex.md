# Omega-W-V3 Real Sensitivity Gate - 2026-05-06 Codex

## Scope

Hidden-gem registry key: `omega_w_v3_real_sensitivity_gate`.

This tranche tightens OWV3 dispatch readiness. A map that is merely CUDA-tagged,
non-stub, and source-SHA-matched is not enough for production dispatch anymore:
`tools/dispatch_dryrun_omega_w_v3.py --require-real-sensitivity` must also pass
the certified sensitivity metadata contract.

## Patch

- `tools/dispatch_dryrun_omega_w_v3.py` now requires
  `component_sensitivity_map_certification_v1` metadata for real-sensitivity
  dispatch readiness.
- The gate cross-checks both top-level source archive identity and certified
  `baseline_archive_*` identity against the selected PR106 source archive.
- `src/tac/tests/test_dispatch_dryrun_omega_w_v3.py` covers certified accept,
  stale archive rejection, and clean-but-uncertified rejection.
- `src/tac/hidden_gems.py` records the hidden-gem item as implemented and
  moves the next step to certified-map candidate building.

## Evidence Discipline

Evidence grade: `empirical` guardrail and dispatch-readiness structure only.

This does not claim score movement. Sensitivity maps remain optimizer feedback;
archive promotion still requires exact CUDA auth eval on exact archive bytes.
