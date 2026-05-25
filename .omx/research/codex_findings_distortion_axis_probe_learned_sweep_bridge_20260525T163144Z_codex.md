# Distortion-Axis Probe Learned-Sweep Bridge

Generated: 2026-05-25T16:31:44Z

## Verdict

Probe 9/10 distortion-axis evidence is now converted into a durable learned-sweep
planning payload instead of remaining a standalone verdict note.

The bridge adapts Probe 9
`POSITIVE_SIGNAL_BREAKS_THRESHOLD` into one planning candidate:
`distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1`.
It suppresses Probe 10
`NEGATIVE_MOTION_NEUTRAL` as a motion-weighted Hinton-KL enhancement, while
preserving the non-motion temporal-context family as separate future signal.

## Durable Artifacts

- Candidate payload:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_candidates.json`
  - SHA-256:
    `89619c075cdbb4da7e5d932680378baeaaa989c739d1e45356564d7d58d50061`
- Learned-sweep plan:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_plan.json`
  - SHA-256:
    `f79db2343ab70ad567039105b16f557d02ff35b5ef48ff94997446a54b23c475`
- Rendered plan:
  `.omx/research/distortion_axis_probe_learned_sweep_bridge_20260525T1625/distortion_axis_probe_learned_sweep_plan.md`
  - SHA-256:
    `f79881d4e610b6da10523a43cae4ddc6926e63315b5aa835d4ed8f2070579ac3`

## Non-Authoritative Budget Signal

Using the conservative predicted delta from the Probe 9 recommendation
(`-0.010` score units), the bridge computes a planning-only repair budget of
`15018.1956` byte-equivalent score units via the canonical rate term
`25 / 37545489`.

This is not score evidence and not dispatch authority. It is a waterbucket
planning signal for asking whether a candidate can spend rate budget on SegNet
or PoseNet repair while still improving total score.

## Queue And Operator Wiring

New reusable surfaces:

- `tac.optimization.distortion_axis_probe_learned_sweep_adapter`
- `tools/adapt_distortion_axis_probes_to_learned_sweep.py`
- `tools/operator_briefing.py` Phase 6h:
  `pact.distortion_axis_learned_sweep_bridge_summary.v1`

Operator briefing now reports:

- `payload_count = 1`
- `plan_count = 1`
- `adapted_candidate_count = 1`
- `suppressed_candidate_count = 1`
- `local_ready_row_count = 5`
- `score_claim = false`
- `ready_for_exact_eval_dispatch = false`

## Remaining Integration Blocker

The bridge is intentionally planning-only. It names the next missing executable
edge explicitly:

`planning_payload_only_selection_adapter_required_before_local_actuation`

That means the learned-sweep planner can rank the distortion candidate, but
`run_mlx_dynamic_learned_sweep_local.py` still needs a compatible selection
adapter or substrate-specific local actuator before execution is automatic.
