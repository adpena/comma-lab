# Codex Findings: Serialized Archive Delta Materializer Feedback

- UTC: 2026-05-25T11:46:22Z
- Lane: `codex_serialized_archive_delta_materializer_feedback_20260525`
- Status: integrated; planning-only; not score authority

## Finding

Single-run materializer manifests should not need bespoke schema support before
they can become local acquisition signal. The durable byte-economics contract is
already `serialized_archive_delta_contract.v1`; if a future HNeRV, NeRV-family,
or non-NeRV materializer emits that contract plus materializer/receiver identity
and false-authority fields, queue observation and dynamic sparse feedback should
accept it without another hardcoded selected-delta key.

## Landed Integration

- Taught `tac.optimization.materializer_feedback.materializer_archive_delta(...)`
  to read canonical `serialized_archive_delta` payloads when no materializer
  family-specific delta section is present.
- Normalized feedback rows now preserve the selected
  `serialized_archive_delta` payload and only mark rate-positive when realized
  savings are explicitly true, avoiding modeled-or-inconsistent byte deltas
  becoming acquisition signal.
- Preserved existing family-specific sections as the preferred source when they
  exist, so current archive-section recode, packet-member recompress, and tensor
  factorization behavior remains stable.
- Extended `experiment_queue_observer` to preserve
  `serialized_archive_delta.schema` and to surface unknown future materializer
  manifests as succeeded feedback artifacts only when they also carry
  materializer identity, a canonical serialized archive delta, a passing
  postcondition, and explicit false-authority proof.
- Added regression tests proving future-family manifests flow into observation
  rows and queue feedback without score/promotion authority, including a strict
  failure when a nested serialized delta carries a truthy `score_claim`.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/materializer_feedback.py src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_experiment_queue_observer.py --no-cache`
- `git diff --check`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_inverse_steganalysis_acquisition.py::test_queue_materializer_delta_reads_family_local_archive_delta_fields src/tac/tests/test_inverse_steganalysis_acquisition.py::test_queue_observation_receiver_negative_materializer_artifact_blocks_water_bucket -q --durations=30 --durations-min=0.01`
- `.venv/bin/python tools/lane_maturity.py validate`

## Remaining Work

- Have future materializers emit `serialized_archive_delta_contract.v1` by
  default, even when they also expose family-specific sections.
- Add a runner-level standard artifact for dynamic sparse materializer feedback
  hints after queue observation harvest.
- Keep exact auth score authority separate from these local feedback rows.
