# Codex Findings: Archive-Bound Contract Migration For PR103, DQS1, Public Intake

UTC: 2026-05-28T20:33:33Z

## Landing

Migrated the next active archive/candidate emitters onto
`tac_archive_bound_candidate_contract.v1`:

- PR103/byte-range entropy recode materializer reports.
- PR103/byte-range executable chain manifests after adapter + receiver proof.
- Public-frontier archive intake profiles.
- DQS1 local-first harvest observations and downstream byte-shaving signal units.

The contract remains false-authority: MLX/macOS advisory and byte-only intake rows
are acquisition signal only, exact CPU/CUDA dispatch remains blocked, and receiver
proof/runtime custody remain required before promotion.

## Verification

- `.venv/bin/ruff check --fix` on touched files.
- `.venv/bin/python -m py_compile` on touched modules.
- `.venv/bin/pytest src/tac/tests/test_byte_range_entropy_recode_materializer.py src/tac/tests/test_public_frontier_intake.py src/tac/tests/test_byte_shaving_signal_surface_builder.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_dqs1_local_first_harvest_observations.py -q`
- `.venv/bin/pytest src/tac/tests/test_mlx_dynamic_sweep_observations.py -q`

## Remaining Contract Migration

Next useful targets are exact-ready bridge intake rows, materializer-chain harvest
normalization, public PR runtime replay manifests, and family-agnostic materializer
outputs that already have byte-closed archives but still expose bespoke readiness
fields.
