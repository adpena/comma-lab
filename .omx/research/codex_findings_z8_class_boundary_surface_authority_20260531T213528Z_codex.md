# Codex Findings: Z8 Class/Boundary Surface Authority

UTC: 2026-05-31T21:35:28Z

## Finding

The Z8 full-video VJP lane had begun carrying a real P18 SegNet class/region and boundary-protection modifier, but the coefficient materializer's NPZ reader still accepted older full-video surfaces that only proved P19/gradient-reduction/budget authority. That left a stale-contract path where a pre-class-boundary bundle could still spend coefficient budget if it had the legacy `budget_spend_authority=true` fields.

## Fix Landed

- `joint_p18_p19_waterfill` now treats the P18 surface as `100 * class_boundary_weight_i * |dL_seg/dx_i|`, excludes boundary-protect atoms from safe rate spend, and reports class histograms and boundary counts.
- `full_video_vjp_acquisition` builds deterministic SegNet class-region weights plus low-margin/adjacent-class boundary masks on the full pair RGB atom grid, pixel-mean normalizes the weights, writes the authority into bundle NPZ files, and requires that modifier before budget spend.
- `joint_coefficient_waterfill` now rejects stale pre-class-boundary surfaces by default, including old NPZ surfaces whose other readiness fields claim full-video budget authority.

## Verification

- `ruff check` passed on the touched Z8/waterfill files.
- `pytest src/tac/tests/test_joint_p18_p19_waterfill.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_full_video_vjp_acquisition.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_joint_coefficient_waterfill.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_per_subband_rd_waterfill_solver.py -q` passed: 87 tests.

## Remaining Work

The next high-EV Z8 work is to run the refreshed full-video bundle through the live 600-pair RD-waterfill schedule and compare the exact byte-closed replay against the prior schedule, now with the class/boundary authority path included in the materializer gate.
