# 05 — Joint Seg/Pose dynamics and trust-region rule

## Incidence

| Frame | SegNet | PoseNet | Implication |
|---|---:|---:|---|
| frame 0 | no direct term | yes | Use for pose-only repair; SegNet drift should be zero except shared decoder coupling. |
| frame 1 | yes | yes | SegNet repairs can harm pose; require trust region. |

## Local score units

Let `g_pose=5/sqrt(10*d_pose)`.

For a proposed update:

`Delta S_local = 100*Delta d_seg + g_pose*Delta d_pose + (25/N)*Delta bytes`.

Since bytes are usually unchanged inside one optimizer step, training trust region begins with Seg/Pose and later adds section bytes after export.

## Trust-region rule

For SegNet-focused frame-1 update:

```text
seg_gain = -100 * delta_d_seg
pose_harm = pose_marginal * max(delta_d_pose, 0)
accept if:
  seg_gain > 0
  pose_harm <= rho * seg_gain + pose_harm_abs_floor
  joint_delta_score < -min_joint_improvement
else backtrack lr or reject update
```

Recommended defaults:

```json
{
  "rho": 0.35,
  "pose_harm_abs_floor": 0.005,
  "min_joint_improvement": 0.001
}
```

For Pose-focused update:

```text
pose_gain = -pose_marginal * delta_d_pose
seg_harm = 100 * max(delta_d_seg, 0)
accept if:
  pose_gain > 0
  seg_harm <= rho * pose_gain + seg_harm_abs_floor
  joint_delta_score < -min_joint_improvement
```

For frame-0-only update:

```text
abs(100*delta_d_seg) <= seg_frame0_leak_budget
pose_gain > 0
```

## Patch target

`src/tac/substrates/_shared/mlx_score_aware/adapter.py`

Add after pre/post step component capture:

```python
def _seg_pose_trust_region_verdict(pre, post, config):
    ...
```

Required telemetry:

```text
dynamics_delta_seg_score_units
dynamics_delta_pose_score_units
dynamics_pose_marginal
dynamics_seg_gain_score_units
dynamics_pose_harm_score_units
dynamics_joint_delta_score_units
dynamics_seg_pose_trust_ratio
dynamics_update_admitted_by_joint_score
```
