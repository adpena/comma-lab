# Codex findings: targeted repair materializer binding

Generated: 2026-05-26T18:06:34Z

## Scope

This checkpoint advances the paired-reference targeted repair rows from local
MLX acquisition signal into bounded local materializer execution surfaces. It
preserves false authority throughout: no score claim, no promotion claim, no
rank/kill, no paid dispatch, and no exact-eval authority.

## What changed

- Bound `renderer_payload_dfl1_v1` context for targeted correction chain
  materializers when the source archive has the required renderer payload
  members.
- Added full-frame file-list context binding from
  `upstream/public_test_video_names.txt` for DFL1 parity follow-up.
- Added DFL1 archive-member preflight blockers so packet-member archives that do
  not contain `renderer.bin`, `masks.mkv`, and `optimized_poses.pt` do not get
  misclassified as executable native renderer payload materializers.
- Extended the targeted component harvest CLI to emit materializer work queues
  and materializer execution queues, not just handoff JSON.
- Preserved sidecar full-frame parity overlays in materializer-chain harvests
  while removing stale receiver/parity blockers from queue rows.

## Evidence

Primary repair-dynamics artifact set:

- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_handoff.json`
- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_work_queue.json`
- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_execution_queue.json`

The work queue now has 14 rows, 2 executable rows, and 12 blocked rows. The
executable rows are:

- `renderer_payload_dfl1_v1`
- `packet_member_zip_header_elide_v1`

The execution queue has 2 experiments and 11 total steps. The DFL1 path includes
local candidate materialization, runtime-consumption proof, shell full-frame
parity sidecar, submission closure, exact-readiness bridge, dispatch plan, and
dispatch queue artifacts under:

- `experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_targeted_component_correction_chain_materializers/`

Earlier cache-reuse artifact set:

- `.omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_bound_materializer_work_queue.json`
- `.omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_bound_materializer_execution_queue.json`
- `.omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_bound_materializer_execution_worker_result.json`
- `.omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_bound_materializer_execution_worker_result_rerun.json`

The first worker result captured a failing materializer execution probe; the
rerun preserved the partial recovery with 3 successes and 1 failure. These are
kept intentionally as signal, not overwritten.

## Compliance verdict

The materializer path remains encoder/compression-side only. Receiver artifacts
are deterministic consumption/parity/closure proofs, not optimizer authority.
MLX and local materializer signals remain acquisition and readiness evidence
only. Exact CPU/CUDA auth eval, claimed dispatch lifecycle, and promotion gates
remain required before any score/rank/promotion claim.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/comma_lab/scheduler/materializer_chain_harvest.py tools/harvest_frontier_targeted_component_correction_response.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
