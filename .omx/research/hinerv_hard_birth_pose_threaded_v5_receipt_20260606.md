# HiNeRV hard-birth pose-threaded v5 receipt (partner ask #1)

UTC: 2026-06-06. Agent: `swarm_f_pose_threaded_v5_20260606` (swarm-F pose-thread).
Authority: **`[macOS-MLX research-signal]` — false authority, NON-PROMOTABLE.**
`score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.
Runner commit: `0b9a1ec99`.

## What this is

The v5 short witness-readiness smoke on the REAL contest video
(`upstream/videos/0.mkv`), run after threading the REAL PoseNet pair teacher
into the HiNeRV target-region birth actuator so the actuator emits a populated
`pose_guard` + `exact_nonrate` term instead of running pose-blind.

Run root:
`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_birth_real_smoke_20260606/hinerv_witness_readiness_short_smoke_v5`

## Root cause that v5 fixes (the latent build crash)

The bootstrap pose-teacher build site in
`tools/run_compact_renderer_mlx_spine_runner.py` (inside
`_run_hi_nerv_mlx_scoreaware_smoke`, the `pose_direct_live_distillation_weight
> 0.0` guard) constructed the teacher bundle as a bare `SimpleNamespace` with
only `target_rgb_0` / `target_rgb_1`. `build_mlx_posenet_pair_teacher` reads
`pose_dims = int(bundle.pose_dims)` with **no default fallback**
(`src/tac/substrates/_shared/mlx_score_aware/loss.py:3684`), so the bare
namespace raised `AttributeError: 'SimpleNamespace' object has no attribute
'pose_dims'`. The wiring committed earlier (`e00fac86d`) already passed
`pose_teacher=bootstrap_pose_scorer_teacher` + `require_pose_trust=...` into
`fit_target_region_birth_from_segnet`, but the teacher object never built — so
`pose_teacher` was effectively `None` and `pose_guard.available` stayed
`false`. The v4 receipt confirms this stale-code state: `pose_guard.available
= false`, `action_id = null`, blocker
`hinerv_target_region_birth_pose_trust_telemetry_missing`, and no
`exact_posenet_target_pose` / `actuators_enabled_effective` keys at all.

The v5 fix (commit `0b9a1ec99`, 1 hunk, runner only):
1. Add `pose_dims=6` to the bootstrap teacher bundle (the documented default;
   the contest PoseNet head uses the first 6 of its 12 pose dims). This makes
   `build_mlx_posenet_pair_teacher` actually succeed on contest-size targets.
2. Wrap the build in `try/except`: on any failure (resolution mismatch, missing
   upstream weights, device fault) fall back to `pose_teacher=None` and record
   `posenet_pair_teacher_build_failed:<exc>` in the bootstrap metadata blockers
   — the runner never crashes (partner requirement: "never crash the runner").

`require_pose_trust` at the callsite is `bool(pose_trust_required)`, which is
`False` for this advisory run (no `--pose-trust-required` flag). The actuator
emits `exact_nonrate` + `pose_guard` regardless, as specified.

## v5 command (verbatim)

```
uv run python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --planner-row-id "hi_nerv::witness_readiness_short_smoke" \
  --allow-bounded-planner-row-timing-smoke-waiver \
  --allow-unscored-research-smoke \
  --source-video-path upstream/videos/0.mkv \
  --output-dir <v5 dir> --overwrite \
  --num-pairs 1 --epochs 1 --batch-pairs 1 \
  --segnet-direct-live-distillation-weight 0.5 \
  --segnet-direct-live-target-min-ratio-floor-weight 1.0 \
  --pose-direct-live-distillation-weight 0.25 \
  --scorer-domain-bootstrap-steps 2 \
  --scorer-domain-bootstrap-segnet-hard-birth-weight 2.0 \
  --coder-aware-qat \
  --receiver-cache-quality-mlx-scorer-response-device-type cpu
```

Runner exit 0. `mode = "executed_hi_nerv_mlx_scoreaware_and_exported"`,
`ready_for_exact_eval_dispatch = false`.

## v5 target_region_birth_actuator receipt (verbatim subset)

Path in `training_artifact.json`:
`substrate_artifact_metadata/score_aware_training/short_scorer_teacher_smoke_readiness/output_head_target_init_gate/metadata/scorer_domain_bootstrap/target_region_birth_actuator`

```json
{
  "schema": "hi_nerv_target_region_birth.v1",
  "enabled": true,
  "accepted": false,
  "accepted_step_count": 0,
  "rejected_step_count": 3,
  "action_id": "4449d108f2924864ffd7092a36302231d557630a48c70272ed8301a29647f804",
  "blockers": [
    "hinerv_target_region_birth_no_accepted_step"
  ],
  "birth_class_index": 0,
  "before_region_hard_ratio": 0.0,
  "after_region_hard_ratio": 0.0,
  "pose_guard": {
    "available": true,
    "final_pose_output_delta_l2": 0.0,
    "input_convention": "concat_yuv6_pair_nhwc255_frame0_then_frame1",
    "max_accepted_pose_output_delta_l2": 0.0,
    "max_pose_output_delta_l2": 0.05,
    "pose_guard_rejected_step_count": 3,
    "pose_input_contest_resolution": true,
    "pose_input_height": 384,
    "pose_input_width": 512
  },
  "exact_nonrate": {
    "authority": "batch_local_live_mlx",
    "delta_score_nonrate": 0.0,
    "new_d_pose_batch": 194.20106506347656,
    "new_d_seg_batch": 0.49237060546875,
    "new_nonrate_score": 93.30531046259965,
    "normalization_scope": "batch_local",
    "old_d_pose_batch": 194.20106506347656,
    "old_d_seg_batch": 0.49237060546875,
    "old_nonrate_score": 93.30531046259965,
    "pose_term_available": true
  },
  "joint_score_rejected_step_count": 0,
  "argmax_transitions": {
    "argmax_changed_count_region": 0,
    "net_target_support_delta": 0,
    "target_hard_lost_count": 0,
    "target_hard_won_count": 0,
    "target_to_wrong_count": 0,
    "wrong_to_target_count": 0,
    "wrong_to_wrong_count": 0
  },
  "human_visual_fidelity_objective": false,
  "runtime_sidecar_bytes": 0
}
```

Bootstrap pose-teacher metadata (`exact_posenet_target_pose`) is now
`enabled: true`, `source: "real_mlx_posenet_teacher_cache"`,
`pose_dims: 6`, `teacher_surface: "teacher_pose_for_yuv6_pair_nhwc"`,
`upstream_posenet_safetensors_sha256:
"0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"`.

## v4 → v5: the surface that flipped

| field | v4 | v5 |
|---|---|---|
| `pose_guard.available` | **false** | **true** |
| `pose_guard.pose_input_contest_resolution` | (absent) | **true** (384×512) |
| `pose_guard.pose_guard_rejected_step_count` | 0 | **3** |
| `exact_nonrate.pose_term_available` | (no `exact_nonrate`) | **true** |
| `exact_posenet_target_pose` | None (key absent) | enabled, real PoseNet sha |
| `action_id` | **null** | `4449d108…7804` |
| `accepted` | **true** (pose-blind) | **false** (pose-guarded reject) |
| birth blocker | `pose_trust_telemetry_missing` | `no_accepted_step` |

v4 ran on the pre-fix codebase where the pose teacher never built (the
`action_id=null` + absent `exact_posenet_target_pose` keys are the tell). Its
`accepted=true` was a **pose-BLIND** acceptance — there was no PoseNet trust
gating the step. v5's acceptance flipped to `false` precisely BECAUSE the pose
guard is now live and rejected all 3 candidate steps for moving the PoseNet
output beyond the `max_pose_output_delta_l2=0.05` trust cap
(`pose_guard_rejected_step_count=3`, `joint_score_rejected_step_count=0`). This
is the intended L3 trust behavior, not a regression: the actuator now refuses
pose-unsafe births.

## Honest verdict

- **`available` requirement: MET.** `pose_guard.available=true`,
  `pose_input_contest_resolution=true`, `exact_nonrate.pose_term_available=true`.
  The REAL PoseNet teacher (real upstream safetensors sha) is now threaded into
  the birth actuator. This was the partner's ask #1 and the gate's
  `pose_input_contest_resolution` precondition.
- **`pose_trust_pass`: FALSE for this smoke** — not because pose trust is
  unavailable, but because no step was accepted. The birth was NOT accepted
  (`accepted=false`, `accepted_step_count=0`,
  blocker `hinerv_target_region_birth_no_accepted_step`). The worst SegNet
  target region never crossed the frontier
  (`before/after_region_hard_ratio=0.0`, `target_hard_won_count=0`) in this
  deliberately tiny 1-pair / 1-epoch / 2-bootstrap-step / max_steps=8 smoke,
  and the 3 candidate steps it did try were rejected by the pose guard.
  `exact_nonrate.delta_score_nonrate=0.0` follows from no accepted step (old ==
  new). `_pose_trusted` (the launch gate's L3 predicate) requires
  `delta_score_nonrate < 0.0`, which a zero-delta non-accepted birth does not
  satisfy — so this run is NOT pose-trusted at the gate level either, which is
  correct and fail-closed.
- The mechanism that determines whether the birth gets ACCEPTED with positive
  pose-trusted target support is now the right one (real PoseNet trust cap), and
  it is actively gating. Getting an ACCEPTED pose-trusted birth needs a longer
  bootstrap (more `--scorer-domain-bootstrap-steps`, larger `max_steps`, and a
  pair where the worst region is actually crossable within the 0.05 pose cap) —
  the smoke config is not sized for acceptance. No fakery: the actuator honestly
  reports it crossed nothing and accepted nothing.

## Launch gate (advisory)

```
.venv/bin/python tools/validate_nerv_long_run_gate.py \
  --family hi_nerv --run-root <v5 dir> \
  --frontier-pointer .omx/state/canonical_frontier_pointer.json --advisory
```

Verdict: **`approved: false`**, `highest_level: "none"`, blocking evidence
`["real_video_birth_receipt_missing"]`. Frontier pointer fresh
(`last_refreshed_utc 2026-06-06T18:22:01Z`). The gate's `_accepted_live_birth`
requires `accepted_step_count > 0` with positive target support; v5's birth was
not accepted, so the gate correctly stays at L-none. The v5 birth receipt IS
discovered by the gate (it appears in `evidence_index` under
`hi_nerv_target_region_birth_receipt.v1`), but it does not clear L2 because no
step was accepted. This is the fail-closed answer: pose teacher availability is
necessary (now satisfied) but not sufficient — an accepted, pose-trusted,
positive-target-support birth is still required for L2→L3.

## Wire-in (Catalog #125; observability/false-authority only)

- Sensitivity-map: N/A — this is a runner plumbing fix, not a per-axis byte
  saving.
- Pareto: N/A.
- Bit-allocator: N/A (`runtime_sidecar_bytes=0`, ordinary model state).
- Cathedral autopilot: N/A (research smoke, non-promotable).
- Continual-learning posterior: N/A (no score claim; advisory only).
- Probe-disambiguator: this receipt + the v4→v5 diff IS the disambiguator for
  "is the pose teacher actually threaded into the birth actuator" → YES, the
  flip of `pose_guard.available false→true` with a real upstream PoseNet sha
  resolves it empirically.

## Mission

`frontier_breaking_enabler` — unblocks the L3 pose-trusted-birth gate by making
`pose_guard.available` reachable on the real video. Does not itself claim a
score; horizon = `frontier_pursuit`.
