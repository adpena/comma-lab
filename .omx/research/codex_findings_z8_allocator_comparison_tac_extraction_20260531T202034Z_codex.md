# Codex Findings: Z8 allocator comparison TAC extraction

UTC: 2026-05-31T20:20:34Z

## Finding

`tools/z8_p18_p19_freeze_vs_implicit_kkt_comparison.py` was an orphaned
experiment actuator. It contained reusable allocator logic, archive mutation,
matched operating-point classification, and advisory replay wiring, while also
importing another tool module for baseline build/replay helpers.

That shape violated the pipeline-over-tools direction: a useful allocator
characterization signal could not be consumed by TAC runners or acquisition
without re-importing a script.

## Change

- Added reusable TAC module:
  `tac.substrates.z8_hierarchical_predictive_coding.allocator_comparison`.
- Moved the freeze allocator, implicit-KKT/Dykstra allocator, archive mutation,
  matched operating-point rows, advisory replay helpers, and result schemas into
  TAC.
- Converted the comparison script into a thin CLI over TAC APIs.
- Added a no tool-to-tool import regression guard.
- Added a public `flatten_detail_coefficients` helper for Z8 detail pyramids so
  comparison code does not rely on a private underscore helper.

## Reviewer fixes

The read-only reviewer found five concrete issues, all patched:

- KKT solver blockers now propagate into arm rows and block winner verdicts.
- Matched operating points now match by charged ZIP bytes/rate first, with
  dead-zone fraction retained as a diagnostic.
- SegNet P18 saliency is applied only to frame 1, the frame SegNet actually
  scores; frame 0 keeps PoseNet protection but gets zero SegNet saliency.
- CLI provenance is written as a sidecar after the final result hash is known,
  instead of rewriting `result.json` after hashing it.
- The archive mutation test now proves archive bytes change and verifies at
  least one previously nonzero detail coefficient becomes zero.

## Authority posture

This comparison is still `[macOS-CPU advisory]` and false-authority. It uses the
legacy single-norm PoseNet characterization surface so old freeze-vs-KKT results
remain apples-to-apples. The emitted report now states that true materializer
budget authority requires `per_axis_posenet_jacobian_mahalanobis_v1`.

## Verification

- `ruff check allocator_comparison.py joint_p18_p19_deadzone_rate_attack.py
  test_allocator_comparison.py z8_p18_p19_freeze_vs_implicit_kkt_comparison.py`
  -> passed
- `pytest test_allocator_comparison.py -q`
  -> 6 passed
- `pytest test_allocator_comparison.py test_joint_p18_p19_deadzone_rate_attack.py
  test_joint_variational_driver.py -q`
  -> 24 passed
- `tools/z8_p18_p19_freeze_vs_implicit_kkt_comparison.py --help`
  -> passed

## Next required work

Run the extracted comparison on a small live archive smoke, then queue a
full-600-pair advisory run only if the smoke confirms runtime is acceptable.
Keep any result false-authority until receiver proof and exact CPU/CUDA replay.
