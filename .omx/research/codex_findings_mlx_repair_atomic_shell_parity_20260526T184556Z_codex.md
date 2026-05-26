# Codex Findings - MLX Repair Atomic Shell Parity Closure

UTC: 2026-05-26T18:45:56Z

## Scope

Follow-up adversarial pass on the MLX repair materializer execution queue after
the DFL1 shell-parity rerun exposed a proof-custody bug class: `--overwrite`
could remove the previous proof directory before the replacement proof had
completed.

## Findings And Fixes

- `tools/prove_shell_inflate_parity.py` now treats CLI overwrite as an atomic
  proof-directory replacement. Existing proof directories are moved aside,
  replacement proof artifacts are written into the canonical output path, and
  any exception restores the prior proof.
- The overwrite guard still refuses symlink outputs, dangerous root/home/repo
  roots, and non-empty directories without shell-parity marker files.
- Regression coverage now proves both successful overwrite replacement and
  failed-overwrite restoration of the prior proof.
- The queue was rewound at
  `prove_renderer_payload_dfl1_shell_parity` and rerun with one local
  IO-heavy lane. Final state: 11/11 steps succeeded.

## Current Evidence

- Queue state:
  `.omx/state/experiment_queue_frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z_targeted_chain_materializer_execution_r2.sqlite`
- Queue definition:
  `.omx/research/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/targeted_component_correction_chain_materializer_execution_queue.json`
- DFL1 shell parity proof:
  `experiments/results/frontier_mlx_repair_dynamics_paired_reference_20260526T155611Z/frontier_targeted_component_correction_chain_materializers/targeted_component_chain_materializers/targeted_component_chain_targeted_component_materialization_packet_member_merge_cbe7d79124ba_001/renderer_payload_dfl1_v1/exact_eval_handoff/renderer_payload_dfl1_shell_parity/shell_inflate_parity.json`
- DFL1 proof summary:
  `full_frame_inflate_output_parity_claim=true`, `cmp_equal=true`,
  `output_sha256_match=true`, `blockers=[]`, `left_bytes=3662409600`,
  `right_bytes=3662409600`.

## Authority Boundary

This is exact-handoff and local proof-custody plumbing, not a score claim.
The generated dry-run dispatch plans remain false-authority until the lane has
an explicit dispatch claim, exact-eval readiness gates clear, and contest
CPU/CUDA auth eval returns a result.
