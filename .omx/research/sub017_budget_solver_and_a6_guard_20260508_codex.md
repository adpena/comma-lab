# sub-0.17 budget solver + A6 guard — 2026-05-08

Timestamp: 2026-05-08T20:51:52Z

## Scope

Operator requested continued score-lowering, stacking, mathematical solving,
and proof work after the A2 packet-parity greenup. This note records three
concrete landed artifacts:

1. Closed-form sub-target byte-budget solver in `src/tac/score_geometry.py`.
2. A6 Selfcomp block-FP × hyperprior fail-closed decoder hardening.
3. A2 inflate-parity temp-work cleanup on nonzero/exception failure paths.

No remote dispatch was attempted. No score claim is made.

## Artifact 1 — closed-form target byte budget

New reusable helper:

- `target_byte_budget_for_score(...)`
- dataclass: `TargetByteBudget`
- evidence grade: `[prediction; closed-form target byte budget]`
- promotion/rank/kill/dispatch flags: all false

The helper solves the exact contest formula:

```text
B_max = floor((S_target - 100*d_seg_floor - sqrt(10*d_pose_floor)) * N / 25)
```

This converts hand-written sub-0.17 planning arithmetic into a typed solver
primitive. Callers must pass the distortion floors for the score axis being
planned; CPU-axis values remain planning unless exact `[contest-CPU]` evidence
backs them, and CUDA promotion still requires exact `[contest-CUDA]`.

Focused regression:

- target score: `0.17`
- floor assumption: `d_seg=6.0e-4`, `d_pose=3.5e-5`
- current bytes: `178,392`
- expected byte budget: `136-138 KB`
- required savings: `40-43 KB`

## Artifact 2 — A6 fail-closed guard

A6 current measured disposition remains narrow:

- current PR101 max-abs-scale conditional Gaussian proxy is a
  measured-config negative;
- best known estimate remains above PR101 brotli;
- no family kill, no score claim, no runtime packet, no exact eval.

Code hardening landed:

- `split_into_blockfp(...)` now handles empty non-int8 arrays before `min/max`
  reduction.
- `decompose_blockfp_with_hyperprior(...)` now rejects ChARM chunk symbol-count
  mismatches before decoding blocks.

Reactivation path: build a byte-closed runtime packet that consumes A6 bytes
through `inflate.sh`, preferably with true Selfcomp per-channel block-FP or
learned/tensor-aware PMFs rather than the current max-abs proxy, then run local
inflate parity before any claimed exact eval dispatch.

## Artifact 3 — A2 inflate-parity failure cleanup

Recursive review found that `verify_inflate_parity(...)` cleaned
`.inflate_parity_work` on the success path but left it behind when `inflate.sh`
returned nonzero. The helper now removes temp work before nonzero-return,
timeout, and pre-comparison exception records. This does not change score or
packet bytes; it closes a hygiene bug class in the packet closure evidence
path.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_score_geometry.py -q
35 passed

.venv/bin/python -m pytest src/tac/tests/test_a6_blockfp_hyperprior_compose_unit.py -q
35 passed

.venv/bin/python -m pytest tests/test_build_a2_sensitivity_weighted_pr101_packet.py src/tac/tests/test_score_geometry.py src/tac/tests/test_a6_blockfp_hyperprior_compose_unit.py -q
84 passed

.venv/bin/python -m py_compile tools/build_a2_sensitivity_weighted_pr101_packet.py src/tac/score_geometry.py src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py
PASS

git diff --check
PASS
```

## Evidence status

These are solver and guard artifacts only. They improve future exact-evaluable
stack selection and decoder safety, but they do not promote any candidate
archive.
