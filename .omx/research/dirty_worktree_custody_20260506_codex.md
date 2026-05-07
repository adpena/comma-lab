# Dirty Worktree Custody - 2026-05-06

Purpose: preserve signal during the adversarial-review takeover tranche by
separating owned code/research changes from pre-existing forensic/public-intake
state that must not be trampled.

## Owned Takeover Tranche

These files are part of the current reviewed tranche and are candidates for
staging after focused tests pass:

- Arithmetic/HNeRV entropy custody:
  `src/tac/arithmetic_terminal.py`,
  `tools/build_hnerv_entropy_candidate_packet.py`,
  `src/tac/hnerv_entropy_candidate_packet.py`,
  `experiments/results/hnerv_entropy_packet_discovery_20260506_codex/*`,
  and related tests.
- Foveation/LA-pose:
  `src/tac/hyperbolic_foveation.py`,
  `src/tac/foveation_readiness.py`,
  `src/tac/raft_radial_pose.py`,
  `experiments/optimize_poses.py`,
  `src/tac/lapose_foveation_*`,
  and related tests.
- Beta sensitivity/water-fill:
  `src/tac/sensitivity_map.py`,
  `experiments/build_sensitivity_map_pr106.py`,
  `experiments/repack_pr106_with_water_filling.py`,
  `tools/dispatch_dryrun_omega_w_v3.py`,
  `experiments/pipeline.py`,
  `scripts/remote_lane_*sensitivity_weighted.sh`,
  and related tests.
- Categorical/openpilot:
  `src/tac/categorical_*`,
  `tools/build_categorical_candidate_fixture.py`,
  and related tests.
- Meta selector/field equations:
  `src/tac/optimization/meta_lagrangian_allocator.py`,
  `tools/build_field_meta_dispatch_selection.py`,
  `tools/build_frontier_roadmap_status.py`,
  and related tests.
- Submission runtime package markers:
  `submissions/__init__.py` and `submissions/apogee_v2/__init__.py`.

## Pre-existing Dirty Forensic State - Do Not Stage In This Tranche

These dirty entries are public-intake/recovered trees or raw external checkouts.
They are left untouched and must not be normalized inside this commit:

- `experiments/results/public_pr100_intake_20260504_codex/source`
- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source`
- `experiments/results/public_pr103_intake_20260504_codex/source`
- `experiments/results/public_pr105_kitchen_sink_intake_20260504_codex/source`
- `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source`
- `experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/repo`
- `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/repo`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_src/repo`
- `reports/raw/kaggle_ingest/kaggle-dilated-h64-long1000-retry-v6-20260410T234220Z/comma_video_compression_challenge`

## Index Hygiene Note

During takeover, the index contained staged deletions of recovered fail-loud
stubs in `src/tac/henosis_pr82_transfer.py` and `src/tac/sjkl_basis.py`. Those
deletions were not promoted because they would remove recovery signal without a
test-backed replacement. The files remain dirty only if a later worker has
additional reviewed changes; otherwise preserve the fail-loud stubs.
