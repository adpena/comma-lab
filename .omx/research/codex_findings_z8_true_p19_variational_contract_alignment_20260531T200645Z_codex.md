# Codex Findings: Z8 true-P19 variational contract alignment

UTC: 2026-05-31T20:06:45Z

## Finding

The Z8 full-video VJP acquisition contract had advanced to the true-P19 gate:

- `true_per_axis_posenet_jacobian_mahalanobis_surface`
- archive-runtime candidate custody
- full-video exact reduction before KKT/Dykstra budget spend

But the joint variational driver test still asserted the older seven-item
contract and the driver metadata still described the implicit Dykstra allocator
as `pending_follow_on`.

That was a real integration drift: the materializer and acquisition paths were
already stricter, while the high-level driver contract lagged behind them.

## Change

- Updated the joint variational P19 contract to name the true six-axis PoseNet
  Jacobian surface directly.
- Replaced the older generic `posenet_null_subset_pair_ids` / generic
  Mahalanobis wording with executable measurements:
  - `per_axis_posenet_vjp_or_jvp`
  - `contest_inverse_variance_mahalanobis_weights`
  - `pose_null_mask_from_mahalanobis_jacobian_norm`
- Added explicit P19 authority requirements:
  - six pose axes
  - positive finite inverse-variance weights
  - archive-runtime candidate custody
  - full-video exact reduction before budget spend
- Marked the implicit KKT/Dykstra allocator status as wired through the
  full-video VJP reduction path rather than pending.
- Updated the contract tests so stale seven-item acquisition contracts fail.

## Authority posture

This does not claim a score and does not promote Z8 output. It closes a
contract/readiness drift so future Z8 codec work cannot silently reason from
the old scalar/proxy P19 language after the executable true-P19 gate landed.

## Verification

- `ruff check joint_variational_driver.py test_joint_variational_driver.py`
  -> passed
- `pytest test_joint_variational_driver.py -q`
  -> 4 passed
- `pytest test_full_video_vjp_acquisition.py test_joint_coefficient_waterfill.py -q`
  -> 35 passed

## Next required work

The orphaned `tools/z8_p18_p19_freeze_vs_implicit_kkt_comparison.py` still
contains useful allocator-characterization signal but is not yet canonical TAC
pipeline code. The next safe migration is to extract its comparison result
schema and matched-operating-point logic into a reusable TAC module, then leave
the heavyweight DistortionNet driver as a thin CLI over that API.
