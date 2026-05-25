# Distortion-Axis Probe Feedback Replan

Generated: 2026-05-25T16:46:12Z

## Verdict

The Probe 9 learned-sweep bridge now has a local feedback edge. It no longer
stops at a ranked plan: the durable plan can be harvested into an append-only
`mlx_dynamic_sweep_observation.v1` row, then replanned so the already-harvested
candidate/config/pass is suppressed by canonical observation feedback.

## What Changed

- Added `tac.optimization.distortion_axis_probe_learned_sweep_feedback`
- Added `tools/run_distortion_axis_probe_learned_sweep_feedback.py`
- Preserved advisory semantics through
  `tac.optimization.mlx_dynamic_sweep_observations`
- Extended `tools/operator_briefing.py` Phase 6h to report feedback observation
  and replan suppression counts

## Durable Artifacts

- Observation JSONL:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_observations.jsonl`
  - SHA-256:
    `c7c6e940333aa43b84dd3886b63e861fb5621a885bb53f9d2cc939d9cfbc3b22`
- Feedback summary:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_feedback_summary.json`
  - SHA-256:
    `5bf078fb793dc763903e79b72b511803938a49effe5d646b0b82419f1d102a5b`
- Replan JSON:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_replan.json`
  - SHA-256:
    `1a75c77db62062ea23fc87f2739cafa9dd1aa729ebc3fdcbf137def0043baeef`
- Replan Markdown:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_replan.md`
  - SHA-256:
    `9d759d89010c489540f446e66363f66ea98e562f69c11e370a52ed15a1f4a13e`

## Observed Feedback

- Candidate:
  `distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1`
- Sweep config: `macos_cpu_advisory`
- Optimization pass: `smoke`
- Observed axis: `macos_cpu_advisory`
- Advisory score delta: `-0.010`
- Replan suppressed observed row count: `1`
- Replan local-ready row count: `4`

## Authority Boundary

This is not scorer execution, not an archive materialization, and not exact CPU
or CUDA evidence. The observation uses a hash identity for the probe verdict,
not a submission archive SHA. It is replanning-only signal with:

- `score_claim = false`
- `promotion_eligible = false`
- `rank_or_kill_eligible = false`
- `ready_for_exact_eval_dispatch = false`

## Remaining Edge

The bridge now closes the local feedback/replan loop for Probe 9 evidence. The
next executable edge is a real substrate runner that spends the derived repair
budget on SegNet/PoseNet training or archive bytes, then emits full macOS-CPU
advisory or MLX scorer-response artifacts rather than probe-derived feedback.
