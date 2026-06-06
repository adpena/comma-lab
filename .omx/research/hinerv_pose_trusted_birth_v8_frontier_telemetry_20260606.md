# HiNeRV Pose-Trusted Birth v8 Frontier Telemetry

UTC: 2026-06-06. Agent: Codex.
Authority: `[macOS-MLX research-signal]` only. `score_claim=false`,
`promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.

## Run

Run root:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_birth_real_smoke_20260606/hinerv_witness_readiness_short_smoke_v8_frontier_telemetry`

Command class: bounded real-video HiNeRV one-pair witness-readiness smoke,
same controls as v6/v7:

- real video: `upstream/videos/0.mkv`
- `--num-pairs 1 --epochs 1 --batch-pairs 1`
- `--pose-trust-required`
- real SegNet teacher and real PoseNet teacher active
- `--scorer-domain-bootstrap-steps 16`
- `--scorer-domain-bootstrap-learning-rate 5.0e-4`
- `--scorer-domain-bootstrap-segnet-hard-birth-weight 8.0`
- `--scorer-domain-bootstrap-segnet-hard-birth-min-ratio-floor 0.02`

The run completed and exported:

`compact_renderer_mlx_spine_runner_report.json`

## Code Landed In This Slice

1. `hi_nerv_target_region_birth_candidate_frontier_telemetry.v1` is now
   emitted by the HiNeRV target-region birth actuator and copied into the
   `hi_nerv_target_region_birth_receipt.v1` receipt. It records rejected
   candidate geometry rather than changing admission:
   candidate count, region progress count, pose-cap count, joint-score count,
   max receiver/argmax motion, margin/support deltas, and min/max PoseNet
   output movement.
2. `live_birth_survival` metadata attachment now strips canonical
   authority/readiness keys before entering `substrate_artifact_metadata`.
   The JSON evidence files still carry explicit false-authority flags on disk;
   the nested metadata copy keeps a single canonical custody surface.

This fixes the v7 failure:

`substrate_artifact_metadata.score_aware_training.live_birth_survival.score_claim`

## v8 Birth Receipt

Path in `training_artifact.json`:

`substrate_artifact_metadata.score_aware_training.short_scorer_teacher_smoke_readiness.output_head_target_init_gate.metadata.scorer_domain_bootstrap.target_region_birth_actuator`

Key fields:

- `accepted=false`
- `accepted_step_count=0`
- `rejected_step_count=3`
- blocker: `hinerv_target_region_birth_no_accepted_step`
- `pose_guard.available=true`
- `pose_guard.pose_input_contest_resolution=true`
- `pose_guard.max_pose_output_delta_l2=0.05`
- `pose_guard.pose_guard_rejected_step_count=3`
- `exact_nonrate.delta_score_nonrate=0.0`
- `argmax_transitions.net_target_support_delta=0`
- `argmax_transitions.target_hard_won_count=0`
- `argmax_transitions.argmax_changed_count_region=0`

## Candidate-Frontier Telemetry

The rejected candidates localize the crux:

- `candidate_attempt_count=27`
- `pose_rejected_candidate_count=27`
- `joint_rejected_candidate_count=27`
- `region_progress_candidate_count=0`
- `pose_cap_satisfied_candidate_count=0`
- `joint_score_improved_candidate_count=0`
- `no_progress_candidate_count=27`
- `max_candidate_receiver_uint8_changed_pixels_region=50568`
- `max_candidate_argmax_flipped_pixels_region=50161`
- `max_candidate_region_hard_won_delta=0`
- `max_candidate_region_hard_ratio_delta=0.0`
- `max_candidate_margin_mean_improvement=0.0`
- `max_candidate_margin_p50_improvement=0.0`
- `max_candidate_nonrate_improvement=0.0`
- `min_candidate_pose_output_delta_l2=0.26197221875190735`
- `max_candidate_pose_output_delta_l2=1.552881121635437`

Interpretation: this is not just a short-step-count issue. The actuator can
move receiver uint8 heavily, but its candidate direction creates large churn
without improving the unsolved target tail, and every receiver-visible
candidate violates the PoseNet cap by at least about 5.24x.

## Launch Gate

Command:

`uv run python tools/validate_nerv_long_run_gate.py --family hinerv --run-root <v8 root> --frontier-pointer .omx/state/canonical_frontier_pointer.json --advisory`

Verdict:

- `approved=false`
- `highest_level="none"`
- blocker: `real_video_birth_receipt_missing`

The gate discovers `hi_nerv_target_region_birth_receipt.v1` in the run, but
correctly refuses it because no accepted real-video birth exists.

## Next Action

Do not launch the long MLX run. The next high-EV implementation is not more
iterations of this same region-gradient actuator. Build a target-geometry
split actuator that optimizes the unsolved tail of the selected connected
component while preserving already-won pixels and measuring PoseNet movement
before admission. The already-won/tail split is now visible in telemetry:
`unsolved_tail_pixel_count=19229`, `already_won_region_pixel_count=31339`.

The next smoke should require:

- positive unsolved-tail margin movement before pose compensation is attempted;
- zero or bounded already-won loss;
- PoseNet cap satisfaction before admission;
- exact nonrate improvement;
- then fakequant/hysteresis survival on the same action id.

## Codex Follow-Up Landed On Main

After this v8 run, Codex landed the target-geometry split in `main`:

- `bfe6e7f9a Target HiNeRV birth updates at unsolved tails`
- `492b04cc9 Trace HiNeRV birth candidate attempts`

The actuator now optimizes the initial unsolved tail of the selected connected
component, penalizes regressions on already-won pixels, admits progress through
net target support or tail-margin improvement, and persists per-attempt records
with receiver-uint8, argmax, PoseNet, joint-score, and decision fields. The next
bounded smoke should be v9 from `492b04cc9` or later; v8 remains pre-v9
diagnostic evidence and is not long-run authority.
