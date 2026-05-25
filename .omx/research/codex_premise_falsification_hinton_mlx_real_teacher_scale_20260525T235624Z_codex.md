# Hinton MLX Real-Teacher Scale Premise Falsification

Date: 2026-05-25T23:56:24Z
Lane: lane_hinton_mlx_real_teacher_refire_20260525
Status: code-fix-landed-locally-pending-refire

## Finding

The uncommitted Hinton real-teacher refire patch originally built the
`RealSegNetTeacherLogitsCache` from `frames_thwc_uint8.astype(np.float32) /
255.0`. That is not 1:1 with upstream contest SegNet input handling.

`upstream/modules.py::DistortionNet.preprocess_input` rearranges decoded video
frames to float NCHW and then calls `SegNet.preprocess_input`; SegNet only
selects the last frame and resizes. It does not divide RGB values by 255.
Therefore the real-teacher cache must feed raw `0..255` RGB floats to match the
contest scorer.

## Impact

Any empirical real-teacher loss verdict produced by the pre-fix cache is
invalidated as contest-teacher evidence. In particular, the untracked
`.omx/research/hinton_mlx_bundle_landed_20260525.md` memo should be treated as
historical scratch until the real-SegNet 100-epoch smoke is refired with the
correct input scale.

This does not falsify the Hinton distillation idea. It falsifies the premise
that the current real-teacher refire measurements were hydrated with the same
teacher input distribution as the upstream scorer.

## Fix

`build_real_segnet_teacher_cache` now preserves the upstream `0..255` RGB scale
and documents the scorer-parity reason. A regression test monkeypatches
`tac.scorer.load_default_scorers` and asserts the fake SegNet receives max RGB
value `255.0`, not `1.0`.

## Verification

- `.venv/bin/ruff check src/tac/substrates/hinton_distilled_scorer_surrogate src/tac/substrates/hinton_distilled_scorer_surrogate/tests/test_mlx_loss.py tools/run_hinton_mlx_long_training_smoke.py`
- `.venv/bin/python -m pytest -q src/tac/substrates/hinton_distilled_scorer_surrogate/tests/test_mlx_loss.py src/tac/substrates/hinton_distilled_scorer_surrogate/tests/test_convergence_classifier.py`

## Next Action

Refire the real-SegNet teacher smoke before using the Hinton bundle verdict for
dispatch, ranking, or substrate-roadmap decisions. If the corrected teacher
still plateaus, proceed to the learnable student-head variant. If it improves,
re-open the numpy parity proof gate on the corrected student artifact.
