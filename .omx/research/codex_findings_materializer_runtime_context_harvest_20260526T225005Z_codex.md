# Codex Findings: Materializer Runtime Context Harvest

UTC: 2026-05-26T22:50:05Z

## Finding

Family-agnostic materializer manifests can be byte-valid and receiver-proofed
while still losing the work-row runtime binding during harvest. The affected
path was:

1. `targeted_component_correction_chain_materializer_work_queue.json` carried
   `receiver_runtime_binding_context` and
   `materializer_context_closure_plan.receiver_proof_request`.
2. `harvest_materializer_chain_manifests()` accepted the materializer manifest.
3. `build_candidate_queue(accepted_paths)` rebuilt rows from manifest paths only,
   so source/candidate runtime context from the work queue was not propagated.

That left later closure/readiness stages dependent on archive-shape inference
instead of queue-owned custody.

## Fix Landed

`src/comma_lab/scheduler/materializer_chain_harvest.py` now carries accepted
work-queue runtime context forward by manifest path and overlays only missing
path fields onto harvested source-queue rows. The overlay preserves fail-closed
authority semantics and records `materializer_harvest_runtime_context_sources`
plus applied fields for auditability.

The extractor treats materializer-source runtime as distinct from reference
runtime. In the live FEC8 work queue, `source_archive_path` referenced the older
FEC6 baseline while the materializer source runtime was the FEC8 submission
runtime. The fix keeps the FEC8 source runtime, submission dir, and `inflate.sh`
internally coherent.

## Live Proof

Artifact:

`.omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/`

Commands executed:

```bash
.venv/bin/python tools/harvest_materializer_chain_candidates.py \
  --repo-root . \
  --work-queue .omx/research/frontier_rate_attack_feedback_refresh_fec8_rate_packet_bridge_20260526_codex_v2/targeted_component_correction_chain_materializer_work_queue.json \
  --source-queue-out .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/source_queue.json \
  --report-out .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/harvest_report.json \
  --allow-unfinished-state \
  --require-accepted \
  --overwrite

.venv/bin/python tools/build_materializer_submission_closure.py \
  --source-queue .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/source_queue.json \
  --submission-dir-out .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/submission_closure/submission_dir \
  --closed-source-queue-out .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/submission_closure/closed_source_queue.json \
  --closure-report-out .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/submission_closure/closure_report.json \
  --overwrite

.venv/bin/python tools/run_materializer_exact_readiness_bridge.py \
  --source-queue .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/submission_closure/closed_source_queue.json \
  --exact-readiness-out-dir .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/exact_readiness \
  --bridge-report-out .omx/research/codex_runtime_context_harvest_regression_20260526T224917Z_codex/exact_readiness_bridge_report.json \
  --overwrite
```

Observed harvested row:

- `source_runtime_dir`: `experiments/results/pr101_frame_exploit_selector_fec8_static_second_order_k16_clean_20260526_codex/submission_dir`
- `source_submission_dir`: same FEC8 runtime dir
- `source_inflate_sh_path`: same FEC8 `inflate.sh`
- `candidate_submission_dir`: same FEC8 runtime dir
- `runtime_consumption_proof_status`: `present`
- `receiver_contract_satisfied`: `true`

Exact-readiness now skips this candidate before promotion because it is
non-rate-positive:

- `ready_candidate_count`: `0`
- `skipped_candidate_count`: `1`
- `blocked_candidate_count`: `0`
- row verdict: `skipped_non_rate_positive_materializer`
- blocker: `materializer_candidate_not_rate_positive_for_exact_readiness`

No source-runtime-missing closure blocker remained, and zero/negative
materializer economics no longer produce unrelated exact-readiness blockers
such as `score_affecting_change_proof_missing`.
The skipped per-candidate report still preserves archive/runtime custody facts,
including source runtime path, source archive SHA, receiver proof status, and
runtime tree hashes when present.
The follow-up narrowing pass also removed a false runtime-adapter blocker by
not copying work-row `candidate_runtime_dir` into static family-agnostic ZIP
transforms. The closure now classifies this live candidate as
`source_runtime_static_closure_with_candidate_archive`; no
`runtime_consumption_proof_runtime_tree_sha_missing` blocker remains.

## Regression Coverage

Added
`test_harvest_work_queue_runtime_context_reaches_family_agnostic_source_queue`
covering a work-queue `packet_member_zip_header_elide_v1` row whose manifest
lacks runtime context but whose queue row carries the receiver binding.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q
.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py
```

Results:

- `58 passed`
- `ruff`: all checks passed

## Next Integration Hooks

The next blocker is not harvest custody. It is exact-readiness economics and
proof completeness for this specific candidate class:

1. Keep `candidate_not_rate_positive` as a refusal, not an implementation error.
2. Add score-affecting change proof and runtime-tree SHA fields where they are
   relevant for positive materializer candidates.
3. Feed rate-negative materializer outcomes into the chain planner as negative
   acquisition data, not exact-eval candidates. The bridge now emits explicit
   `skipped_non_rate_positive_materializer` rows so acquisition can consume the
   result without polluting exact-readiness blocker telemetry.
