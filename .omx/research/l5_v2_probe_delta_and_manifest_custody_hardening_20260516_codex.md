# L5 v2 Probe Delta and Manifest Custody Hardening

Date: 2026-05-16
Owner: Codex
Status: landed

## Bug Classes

1. L5 v2 side-info and exact-anchor gates required inflated-output manifest
   paths plus aggregate SHA fields, but did not verify that the aggregate SHA
   in the gate row matched the manifest content.
2. The L5 v2 probe disambiguator could select an architecture from a finite
   freeform `predicted_or_measured_delta` even when paired CPU/CUDA axis rows
   did not bind that delta.

## Fix

- L5 v2 gate validation now parses inflated-output manifests and rejects
  aggregate-SHA mismatches for both temporal side-info byte-mutation proofs and
  exact anchor pairs.
- L5 v2 probe observations now require per-axis `score_delta` evidence, accept
  either a top-level axis field or `component_deltas.score_delta`, and require
  the observation delta to match the conservative paired CPU/CUDA value.
- Probe selection now uses the paired-axis score delta and emits
  `selected_delta_source=paired_axis_score_delta`.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_probe_disambiguator.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py src/tac/tests/test_autopilot_dispatch_ranking.py src/tac/tests/test_research_basis.py -q`

## Reactivation Criteria

If L5 v2 later allows CPU-only or CUDA-only architecture decisions, that must
be a separate explicit evidence axis with its own selector and blocker labels.
The paired disambiguator must not silently fall back to handwritten deltas.
