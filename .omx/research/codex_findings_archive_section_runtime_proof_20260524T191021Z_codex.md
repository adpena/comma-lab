# Codex Findings: Archive Section Runtime Proof

Date: 2026-05-24T19:10:21Z

## Verdict

`archive_section_entropy_recode_v1` now emits a file-backed
`family_agnostic_runtime_consumption_proof_v1` artifact, but it only satisfies
the receiver contract when selected Brotli sections preserve both raw decoded
payload and section length. Length-changing entropy recodes remain blocked until
a stronger receiver/runtime proof exists.

This is intentionally conservative: raw payload identity is useful signal, but
it is not enough to prove runtime consumption when offsets or section lengths
move.

## Landed Integration

- `materialize_archive_section_entropy_recode_candidate(...)` accepts
  `runtime_consumption_proof_out` and writes a deterministic proof file.
- `tools/run_family_agnostic_materializer.py` auto-emits a sibling
  `*.runtime_consumption_proof.json` for archive-section entropy recode rows
  when no external proof is supplied.
- `byte_shaving_campaign_queue` tracks the archive-section proof as an
  output/pullback artifact.
- Harvested archive-section rows carry the proof status and verified
  source/candidate member custody fields into the exact-readiness path.
- Proof sections record source/candidate section SHA, raw decoded SHA, offset,
  length, raw-payload identity, and length-preservation status.

## Safeguards

- Length-changing recodes keep
  `section_length_changed_requires_runtime_consumption_proof`.
- Generated proofs keep score/promotion/rank/kill and dispatch authority false.
- Materializer rows remain non-authoritative and still require exact auth eval
  before any score claim.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_wraps_archive_section_entropy_recode_adapter -q`
  - `13 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - `197 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `git diff --check`
- `.venv/bin/python tools/lane_maturity.py validate`
- `tools/review_tracker.py policy-check` clean on touched Python surfaces after
  two Codex review passes.

## Remaining Work

The next proof layer is a runtime-route proof for length-changing archive-section
recodes. Until then, the generated proof is a calibration/harvest artifact and a
length-preserved receiver proof, not a universal archive-section promotion path.
