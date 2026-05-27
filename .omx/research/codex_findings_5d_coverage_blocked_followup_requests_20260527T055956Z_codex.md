# Codex Findings: 5D Coverage Blocked Follow-Up Requests

UTC: 2026-05-27T05:59:56Z

## Finding

The 5D coverage acquisition queue correctly emitted acquisition plans, but the
blocked work-order classes were still too prose-like:

- `populate_missing_paired_cpu_cuda_axis_anchors` exposed a fake-looking
  `dispatch_modal_paired_auth_eval.py --submission-bundle ...` template that
  does not match the canonical paired dispatcher contract.
- `acquire_negative_delta_cells_before_operator_fanout` named MLX scorer
  response work but did not expose a typed request with concrete cache inputs,
  output custody, and the canonical scorer-response runner.

That created an automation gap between coverage audit signal and the exact-axis
/ MLX-cache acquisition lanes that should consume it.

## Fix

Coverage acquisition plans now carry typed `followup_lane_requests`:

- `pair_frame_5d_canvas_exact_axis_anchor_request.v1`
  - Requires a `submission_bundle_v1_20260526` sidecar.
  - Routes dry-run planning through `tools/paired_auth_eval_cli.py`.
  - Exposes the canonical `tools/dispatch_modal_paired_auth_eval.py` paired
    dispatcher template with `--expected-runtime-tree-sha256 auto` and
    `--skip-axis-if-promotable-anchor-exists`.
  - Runs the paired-dispatch command-contract blocker check and records the
    blocker list in the plan.
- `pair_frame_5d_canvas_mlx_negative_delta_request.v1`
  - Requires reference/candidate MLX cache dirs plus archive byte size.
  - Routes local scorer-response work through
    `tools/run_mlx_scorer_response_cache.py`.
  - Stamps output/ component paths and keeps local MLX signal false-authority.

The old fake `--submission-bundle` dispatch template is gone.

## Verification

- `ruff check --fix` on touched files: pass.
- Focused pytest:
  - `src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py`
  - `src/tac/tests/test_pair_frame_5d_extended_operator_queue.py`
  - `src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py`
  - `src/tac/tests/test_modal_paired_dispatch_contract.py`
  - result: `16 passed in 2.80s`
- Live coverage queue smoke against the current PR110 5D coverage audit:
  - queue validation: `valid=true`, `experiment_count=6`, `step_count=9`
  - plan worker: `failure_count=0`, `success_count=5`
  - exact-axis request blocker check: `[]`
  - MLX request points at `tools/run_mlx_scorer_response_cache.py`

## Remaining Work

The follow-up requests are now typed and queue-emitted, but they are still
requests. The next bridge is an executor that binds real
`submission_bundle_result.json` and real MLX cache dirs into child queues, then
uses the existing paired-auth and MLX execution queues as actuators.
