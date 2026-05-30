# Codex Findings: Archive-Bound Candidate Adapter Spine

Date: 2026-05-30T18:36:50Z

## Finding

Candidate readiness was still partially duplicated across materializer harvest,
exact-ready bridge, public/frontier-style archive intakes, and acquisition
inputs. That created signal-loss risk: one emitter could update archive/runtime
custody while another consumer continued reading stale readiness fields.

## Landing

- Added a shared row-to-contract migration helper in
  `tac.optimization.archive_bound_candidate_contract`.
- Wired materializer-chain harvest, family-agnostic materializer harvest, and
  repair-family exact-ready bridge outputs to emit
  `tac_archive_bound_candidate_contract.v1` plus its contract surface.
- Made cross-family acquisition consume archive-bound contracts and adapter
  packages directly, with entropy-stage penalties and posterior negative
  demotion.
- Added `ArchiveBoundCandidateAdapter` and
  `build_archive_bound_candidate_adapter_package(...)` so new substrates emit
  one candidate row and automatically receive replay bundles, MLX advisory
  triage requests, receiver-proof gates, exact-axis blockers, and posterior
  update hooks.
- Documented the canonical pipeline in
  `docs/archive_bound_candidate_pipeline.md`.

## Verification

- `ruff check` on touched Python files: passed.
- `py_compile` on touched Python files: passed.
- `pytest src/tac/tests/test_archive_bound_candidate_adapter_spine.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_repair_family_materializers.py -q`: 89 passed.
- `git diff --check` on touched files: passed.

## Residual

`tools/check_tac_terminology.py --strict` still fails on pre-existing stale
public-doc links to `https://github.com/adpena/tac` outside this slice. The new
archive-bound pipeline doc and docs index entry did not introduce those
violations.
