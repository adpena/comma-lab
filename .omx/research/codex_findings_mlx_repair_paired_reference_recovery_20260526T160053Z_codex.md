# Codex findings: MLX repair paired-reference recovery

Generated: 2026-05-26T16:00:53Z

## Scope

This landing converts the May 26 repair-dynamics queue from candidate-only MLX
reuse into paired candidate-vs-source reference evidence. It stays local and
false-authority throughout: no score claim, no promotion claim, no rank/kill,
no paid dispatch, and no exact-eval authority.

## What changed

- Recovered source reference eval context for targeted component-correction rows
  from receiver-closed materializer closure reports when the acquisition row no
  longer carries top-level source archive/runtime fields.
- Passed that recovered context through the work-order CLI so queue worker rows
  preserve source archive path, source archive SHA, source inflate path, and
  reference component eval metadata.
- Added explicit candidate-vs-reference score delta summaries for local CPU and
  local MLX harvest rows.
- Gated repair-dynamics palette probe matrix emission to correction families
  whose family id starts with `repair_dynamics_`. Generic SegNet/PoseNet
  waterfill rows may inherit the repair prior but do not require the palette
  matrix.

## Queue evidence

Failed probe preserved:

- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155219Z/`
- Failure class: `segnet_posenet_waterfill_region_repair` inherited
  `repair_dynamics_prior_active=true`, so the queue tried to run
  `tools/build_repair_dynamics_palette_probe_matrix.py` for a non-repair-dynamics
  family. That tool correctly refused to emit a repair-dynamics matrix for that
  row, causing the queue postcondition to fail.

Successful corrected run:

- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/`
- Queue:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_queue.json`
- Worker:
  `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_queue.json run-worker --execute --max-steps 18 --max-experiments 1 --max-parallel 1`
- Worker result: 18 started, 18 succeeded, 0 failures.
- Probe matrix emission: 3 repair-dynamics rows only.
- Non-repair waterfill row:
  `repair_dynamics_prior_active=true`,
  `repair_dynamics_probe_required=false`,
  `repair_dynamics_palette_probe_matrix_path=null`.

## Harvest signal

Aggregate:

- `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_response_harvest.json`
- Rows: 4
- Negative local Lagrangian rows: 4
- Local acquisition recommended rows: 4
- Ready-for-budget-spend rows: 0
- Paired MLX-vs-local-CPU Lagrangian delta drift max absolute: 0.0
- Absolute MLX-vs-local-CPU score offset max absolute: 0.017644694706653752

All four selected rows have:

- local CPU component delta: 0.0
- local MLX component delta: 0.0
- receiver-closed archive byte delta vs source reference: -258
- receiver-closed rate delta: -0.0001717916099055202
- measured local Lagrangian delta: -0.0001717916099055202

Interpretation: the paired CPU and MLX candidate-vs-reference deltas agree for
the receiver-closed byte win. This is useful local acquisition signal only; it
does not authorize score claims or promotion.

## Downstream artifacts

- Materialization requests:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_materialization_requests.json`
- Materialization queue:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_materialization_queue.json`
- Operation-chain work orders:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_operation_chain_work_orders.json`
- Operation-chain compiler queue:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_operation_chain_queue.json`
- Chain materializer handoff:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_handoff.json`
- Chain materializer work queue:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_work_queue.json`

The handoff now has 14 materializer work rows and 1 executable row. Budget spend
remains blocked by receiver-consumed correction materializer, full-frame inflate
parity, exact-axis component response, exact auth eval, and local-signal-only
gates.

## Compliance verdict

Contest-compliance status is fail-closed and aligned with the encoder-side
repair contract:

- The repair/action search is encoder/compression-side only.
- The receiver remains deterministic consumption/inflate only.
- MLX rows are tagged local acquisition signal, not contest CPU/CUDA score
  authority.
- The remaining promotion path requires receiver-consumed materialization,
  full-frame inflate parity, and claimed exact CPU/CUDA auth eval.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py tools/build_frontier_targeted_component_correction_work_order.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_queue.json validate`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_materialization_queue.json validate`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_operation_chain_queue.json validate`

