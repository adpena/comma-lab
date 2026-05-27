# Codex Findings - PR95 MLX Full 3000-Epoch Control

Date UTC: 2026-05-27T02:44:47Z
Agent: Codex
Lane: lane_pr95_hnerv_mlx_reproduction

## Finding

The PR95/HNeRV MLX reproduction lane now has an explicit full-execute CLI path
and a first end-to-end local control run through the byte-closed archive and
upstream CPU advisory scorer. This is not a frontier candidate; it is a
reproducibility and integration control.

## Artifacts

- Training root:
  `experiments/results/pr95_mlx_full_rgb_mse_3000ep_20260527T0052Z/`
- Training report:
  `experiments/results/pr95_mlx_full_rgb_mse_3000ep_20260527T0052Z/report.json`
- Final checkpoint:
  `experiments/results/pr95_mlx_full_rgb_mse_3000ep_20260527T0052Z/checkpoints/stage08_converge_low_lr_epoch003000_20260527T023014Z.pt`
- Packaged archive:
  `experiments/results/pr95_mlx_full_rgb_mse_3000ep_20260527T0052Z/stage08_epoch3000_latents_from_pt_package/archive.zip`
- Runtime consumption proof:
  `experiments/results/pr95_mlx_full_rgb_mse_3000ep_20260527T0052Z/stage08_epoch3000_runtime_consumption.json`
- Advisory score artifact:
  `experiments/results/pr95_mlx_full_rgb_mse_3000ep_20260527T0052Z/local_cpu_advisory_eval/raw_advisory_eval.json`

## Measurements

- MLX training mode: `executed_local_mlx`
- Stage count: 8
- Configured epochs: 3000
- Telemetry rows: 3001
- Final recorded loss: `0.00100373150780797`
- Peak MLX memory from final telemetry row: `4538579136` bytes
- Checkpoints emitted: 8
- Archive bytes: `216419`
- Archive SHA-256:
  `27d5199b5cffca661cfbe9e8bfdeb8d039b59b409f83e0762a5c9ebe5661dddd`
- Raw output bytes: `3662409600`
- Raw output SHA-256:
  `86bfcdfc8cc425e0600bf3098a2d235659c895489735a2f2f6a5320c24ff4d79`
- Runtime consumption proven: `true`
- Local advisory scorer elapsed: `460.6435425000964` seconds

## Advisory Score

Axis: `[macOS-CPU advisory]`

- canonical_score: `45.4913178090151`
- avg_posenet_dist: `163.02104187`
- avg_segnet_dist: `0.04971349`
- rate_unscaled: `0.00576418`
- archive_size_bytes: `216419`

This score has no rank, promotion, or dispatch authority. The advisory artifact
correctly carries blockers:

- `raw_eval_advisory_not_full_archive_inflate_custody`
- `not_contest_cuda`

## Interpretation

The loop is operational:

1. Native MLX long training runs the configured PR95/HNeRV curriculum.
2. PyTorch export emits canonical PR95 decoder keys plus full latents.
3. The exported checkpoint packages into the PR95 archive grammar.
4. The public PR95 inflate runtime consumes the archive into full raw output.
5. The canonical raw advisory scorer evaluates the preserved raw output without
   duplicating another 3.4 GB raw blob.

The score is intentionally bad relative to the frontier. The next gap is not
archive packaging or runtime consumption; it is source-faithful PR95 training
recipe fidelity and scorer-aware objective/curriculum fidelity. The first full
control was RGB MSE-heavy and should be treated as a calibrated bad baseline for
Muon/AdamW parity, LR schedule parity, QAT/noise, full-data sampling, and
scorer-routed loss work.

## Repair-Rate Attack Sidecar Signal

The harvested read-only subagent audit found the rate-only parent materializer
path already receiver-consumed while child repair rows remain blocked on direct
candidate-chain materializer manifests. Highest-EV next implementation target:
insert a bounded local queue step between
`emit_repair_budget_materialization_plan` and
`bind_repair_budget_materializer_execution` that emits a direct child manifest
only when a nonzero repair operator and receiver proof exist, otherwise fails
closed with a typed blocker. Target functions:

- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py::build_frontier_repair_budget_materialization_plan`
- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py::build_frontier_repair_budget_materializer_binding_report`
- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py::build_frontier_repair_budget_waterfill_queue`

This should use the new `tac.submission_packet` substrate where possible and
must avoid retained raw outputs.

## Next Actions

1. Keep the full-execute CLI guard committed so full local MLX training cannot
   be confused with smoke execution.
2. Add a canonical PR95 source-recipe parity stage: Muon/AdamW exact optimizer
   knobs, source LR schedule, source sampling, and faithful stage semantics.
3. Add a scorer-aware/frozen-teacher objective lane after source-faithful
   parity, not before.
4. Convert the repair-rate sidecar finding into a child materializer queue step
   with fail-closed direct-manifest semantics.
5. Use exact CPU/CUDA auth eval only after the local advisory score approaches a
   plausible frontier band.
