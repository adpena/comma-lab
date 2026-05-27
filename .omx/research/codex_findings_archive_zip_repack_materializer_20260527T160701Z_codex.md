# Codex Findings - Archive ZIP Repack Materializer Wiring

UTC: 2026-05-27T16:07:01Z

## Landing

Archive-wide ZIP repack is now a first-class family-agnostic materializer at
the entropy-coder boundary. It explores uniform and greedy per-member ZIP
compression plans, emits a byte-closed candidate archive, and writes a
payload-identity receiver proof that verifies every decoded ZIP member SHA-256
against the source archive before any exact-readiness consumer can trust it.

The target is wired through the executable registry, byte-shaving work queues,
family-agnostic materializer CLI, empirical sweep CLI, materializer feedback
normalization, frontier bootstrap defaults, entropy-position labeling, queue
observation, and materializer-chain harvest family classification.

## Authority

No score movement is claimed. The manifest, observation rows, and queue rows
stay false-authority: no score claim, promotion eligibility, rank/kill
authority, or exact-dispatch readiness. Exact auth eval is still required for
any contest score claim.

## Verification

- `ruff check` on the touched archive-repack, queue, registry, feedback,
  entropy-position, sweep, and test surfaces passed.
- `py_compile` on the touched Python surfaces passed.
- `pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializer_sweep.py::test_materializer_empirical_sweep_supports_archive_zip_repack src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_registry_has_family_agnostic_fail_closed_targets src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_wraps_archive_zip_repack src/tac/tests/test_repair_campaign_scorer.py -q`
  passed: 53 tests.
- `pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py::test_frontier_bootstrap_propagates_declared_video_scope_to_queue -q`
  passed.
- `tools/lane_maturity.py validate` passed: 1438 lanes validated cleanly.
- Review policy checks passed for all touched tracked Python files in this
  landing.
- Local CPU queue smoke wrote
  `.omx/research/frontier_final_rate_attack_archive_zip_repack_20260527Tsmoke/`.
  The frontier archive repack row preserved receiver proof/custody and failed
  closed as `candidate_not_rate_positive` with zero realized byte delta.

## Next Action

Run archive ZIP-repack sweeps against the current frontier archive as a local
CPU materializer campaign, ingest observations into materializer feedback, then
let the autonomous chain planner compare ZIP-repack against packet-member
recompress and ZIP-header elide under the same receiver-proof and exact-eval
handoff gates.
