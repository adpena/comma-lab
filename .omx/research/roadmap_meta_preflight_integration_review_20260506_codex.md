# Roadmap / Meta Preflight Integration Review - 2026-05-06

Scope: review-only hardening for `tools/build_frontier_roadmap_status.py`,
`tools/build_field_meta_dispatch_selection.py`, and focused tests. No GPU or
remote dispatch was attempted. No score claim or lane claim was made.

## Findings

- Candidate packet count remains four for the f6c0035a field/meta inputs.
- The previous selected packet,
  `hnerv_lowlevel_repack_pr106_q10_20260506_codex`, has local archive,
  runtime, and strict-preflight closure, but its manifest source does not expose
  a `lane_id` or `job_name` / `instance_job_id`.
- Treating that row as `candidate_static_preflight_ready=true` created a
  false-readiness risk: the selector could say one packet was static-ready even
  though no matching Level-2 claim could be formed from the manifest.
- The selector now reports this as local-only readiness:
  `candidate_local_preflight_ready_count=1`,
  `candidate_static_preflight_ready_count=0`,
  `ready_candidate_count=0`, and report-level
  `ready_for_exact_eval_dispatch=false`.
- The next proof for the selected packet is now
  `manifest_lane_id_and_instance_job_id_for_level2_claim`, not a generic active
  claim request.
- Live roadmap dirty state during review showed one dirty-blocked frontier row:
  `joint_admm_balle_arithmetic_stack`, blocked by unrelated WIP in
  `src/tac/joint_codec_stack_orchestrator.py`. The pre-existing public-intake
  nested gitlinks and raw Kaggle checkout were left untouched.

## Hardening

- `candidate_local_preflight_ready`: archive, runtime, and strict local
  candidate preflight are closed.
- `candidate_static_preflight_ready`: local preflight is closed and the manifest
  exposes lane/job identity needed for a later Level-2 claim.
- `ready_for_exact_eval_dispatch`: static preflight is ready and a matching
  active Level-2 lane claim is present.
- Next-tranche workstreams now expose `dirty_blocked_keys`,
  `unblocked_keys`, and `all_keys_safe_to_touch_now` so strategic keys do not
  hide shared-worktree blockers.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_build_field_meta_dispatch_selection.py src/tac/tests/test_build_frontier_roadmap_status.py -q`
  passed, 12 tests.
- `.venv/bin/python -m ruff check tools/build_field_meta_dispatch_selection.py tools/build_frontier_roadmap_status.py src/tac/tests/test_build_field_meta_dispatch_selection.py src/tac/tests/test_build_frontier_roadmap_status.py`
  passed.
- `git diff --check -- tools/build_field_meta_dispatch_selection.py tools/build_frontier_roadmap_status.py src/tac/tests/test_build_field_meta_dispatch_selection.py src/tac/tests/test_build_frontier_roadmap_status.py .omx/research/roadmap_meta_preflight_integration_review_20260506_codex.md`
  passed for tracked diff paths; the new ledger was also checked for trailing
  whitespace with `rg`.
