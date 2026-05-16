# L5 Substrate Scaffold Registry Wire-In

Date: 2026-05-16
Owner: Codex
Status: landed

## Bug Class

L5/Z staircase substrate packages existed under `src/tac/substrates/` and were
visible through canonical inventory/ranking, but the package-level
`SUBSTRATE_SCAFFOLDS` registry still reflected an older 15-entry snapshot.
That split-brain made scaffold/package discovery depend on which registry an
operator or subagent read first.

## Fix

- Added package-level scaffold mappings for:
  - `z3_balle_hyperprior_bolton`
  - `z4_cooperative_receiver_loss`
  - `z5_predictive_coding_world_model`
  - `c1_world_model_foveation`
  - `c6_e4_mdl_ibps`
  - `time_traveler_l5_autonomy`
- Replaced the stale exact-count test with a direct L5 staircase registry
  inclusion test.
- Kept the WAVE-A-2 non-leak check so legacy renderer rows still do not get
  mislabeled as package-level scaffolds.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/__init__.py src/tac/tests/test_wave_a_2_taxonomy_inventory_wire_in.py`
- `.venv/bin/python -m pytest src/tac/tests/test_wave_a_2_taxonomy_inventory_wire_in.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_registered_substrate.py src/tac/tests/test_autopilot_dispatch_ranking.py -q`

## Reactivation Criteria

Any new promoted substrate package with `archive.py`, `inflate.py`, or
`registered_substrate.py` must either enter `SUBSTRATE_SCAFFOLDS` or carry a
dated exclusion rationale that names its alternate discovery surface.
