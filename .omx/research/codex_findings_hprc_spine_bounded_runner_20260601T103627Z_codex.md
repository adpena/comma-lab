# HPRC Spine Bounded Runner Findings

Timestamp: 2026-06-01T10:36:27Z

## Verdict

The compact-representation lane now has a contract-first bounded-runner surface
instead of separate family-specific readiness interpretation. The runner
consumes `hprc_spine_acquisition_report.v1`, requires every compact base row to
emit a representation spine, and keeps receiver proof, full-video MLX replay,
and exact CPU/CUDA gating as hard follow-up gates.

## Live Artifact

- Runner plan:
  `/Volumes/VertigoDataTier/pact/hprc_spine_bounded_runner_live_20260601T103459Z/hprc_spine_bounded_runner_plan.json`
- Inputs:
  `/Volumes/VertigoDataTier/pact/hprc_spine_acquisition_live_20260601T044157Z/hprc_spine_acquisition_queue.json`
  plus three MLX component-neutralization profiles.
- Output shape:
  9 compact-base sweep rows, 13 section-value rows, 7 residual-token admission
  rows, and 3 selected runner rows.

## Runner Findings

- `hnerv_packed` at 178,258 bytes and `pr95_hnerv` at 178,417 bytes are the
  current full-coverage candidates below the 216k/285k ceilings and route to
  receiver proof, full-video MLX replay, and exact gate.
- The available `pact_nerv_vq` row remains blocked for base comparison because
  it declares only 32 pairs; it routes to full-coverage training/export before
  byte-ceiling comparison.
- Residual transforms with measured negative advisory objective deltas now route
  to receiver proof instead of sitting in profile files.
- `neutralize_residual_rc` is preserved as a durable demotion signal for the
  existing residual section because the measured removal row improves advisory
  score while saving bytes.

## Authority

All rows remain false-authority. MLX profiles route local work and posterior
demotion only. No row is score-authority or exact-dispatchable until a
receiver-proven byte-closed archive passes the exact CPU/CUDA gate.

## Verification

- `ruff check` on touched code passed.
- `pytest src/tac/substrates/hprc/tests/test_spine_bounded_runner.py src/tac/substrates/hprc/tests/test_spine_acquisition.py -q` passed: 6 tests.
- `pytest src/tac/substrates/hprc/tests -q` passed: 76 tests.
- Review policy checks passed for the new runner module, CLI, and tests.
