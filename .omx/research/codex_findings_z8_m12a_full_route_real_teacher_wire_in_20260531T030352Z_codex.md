# Codex Findings: Z8 M12a Full-Route Real-Teacher Wire-In

UTC: 2026-05-31T03:03:52Z

## Scope

Adversarial review of the Z8 M12a operator recipe, remote driver, and MLX
trainer path after the Yousfi proceed-with-revisions landing. The review
focused on no-fake-implementation compliance, active-route consistency, MLX
local score binding, and provenance-clean predictive-coding execution.

## Finding

The M12a recipe claimed `Z8_TRAINER_MODE=full` as the active long-training
route, with score-aware optimizer semantics. The remote driver still defaulted
to `canonical_quadruple`, and the full trainer path constructed a
`RendererBundle` with `distillation_weight > 0` but no real scorer teacher. In
the current fail-closed harness, that route would either fail before training
or require an explicit mock-teacher bypass. It therefore was not a real active
M12a score-aware path.

The same recipe also implied Canonical Revision #2 (`Z8_M7_SOURCE=
empirical_from_master_gradient`) was active. That signal is not consumed by the
full-mode trainer today. Keeping it in the active recipe would overstate the
mathematical binding depth and would be a provenance/authority bug.

## Fix

- `experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py`
  now builds real MLX SegNet and PoseNet pair teachers for `--full` by
  default, plus learnable SegNet and pose student heads.
- `--allow-mock-scorer-teacher` is the explicit escape hatch for research-only
  local smoke; the default path stays real-teacher/fail-closed.
- Full-mode training now carries PoseNet distillation weight, EMA 0.997,
  gradient clipping, LR warmup, weight decay, and optimizer selection through
  `run_mlx_score_aware_full_main`.
- `scripts/remote_lane_substrate_z8_hierarchical_predictive_coding.sh`
  defaults to `Z8_TRAINER_MODE=full`, passes the stabilizer knobs, records them
  in provenance, and harvests `training_artifact.json` without treating MLX
  advisory rows as contest score authority.
- The M12a recipe now names the real active contract:
  `mlx_score_aware_full_main + real_segnet_hinton_t2 +
  real_posenet_pose_mse + z8_hierarchical_renderer`.
- The recipe removes the false active M7 scorer-sensitivity claim and marks
  Rev #2 as M12c reactivation work: wire empirical master-gradient sensitivity
  into the active Z8 loss surface before claiming the Rev #1+#2 predicted band.
- The M12a predicted band was made more honest for Revision #1-only execution:
  `[contest-CPU] [0.185, 0.205]`, `[contest-CUDA] [0.199, 0.219]`, with the
  deep Yousfi band preserved as M12c-conditional.
- A stale canonical-quadruple observability test was updated to include the Rev
  #3 Wyner-Ziv side-info provenance key already emitted by the implementation.

## Verification

- `.venv/bin/python -m ruff check experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py src/tac/tests/test_z8_m12a_full_route_contract.py src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py`
  passed.
- `bash -n scripts/remote_lane_substrate_z8_hierarchical_predictive_coding.sh`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_z8_m12a_full_route_contract.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_yousfi_revisions_3_4_5.py src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py -q`
  passed: 55 tests.
- `.venv/bin/python tools/lane_maturity.py validate` passed: 1557 lanes.
- `.venv/bin/python tools/review_tracker.py policy-check experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py src/tac/tests/test_z8_m12a_full_route_contract.py src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py`
  passed: 0 violations.

## Residual Blockers

- M12a is now a real MLX-local advisory full-route trainer, not exact score
  authority.
- M12b remains the paired CPU/CUDA exact-axis calibration gate.
- M12c must wire empirical master-gradient scorer sensitivity, PoseNet
  side-info, the four-level blind-spot extension, and the UNIWARD finite-
  difference primitive into the active trainer before the deep Yousfi band can
  be claimed.

## Subagent Note

An xhigh subagent spawn was attempted for deeper Z8/Dreamer/Z7 provenance
audit, but the thread limit was reached. This landing was therefore completed
as a local Codex adversarial pass.
