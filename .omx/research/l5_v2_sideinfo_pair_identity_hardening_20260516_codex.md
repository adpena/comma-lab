# L5 v2 side-info pair identity hardening

- date: `2026-05-16`
- agent: `codex`
- scope: `L5 v2 TT5L side-info effect curve`
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`

## Why

The TT5L runtime-mismatch packet showed the exact failure mode this gate must
not allow: a real single-axis `contest_cuda` result can share the archive SHA
with a planned paired run but still have a different runtime contract. The
side-info effect curve previously checked that all CPU/CUDA cells existed, but
did not prove that CPU and CUDA cells for the same side-info variant were the
same archive/runtime pair.

That can create a false paired curve by mixing cells from different runtimes or
different variant archives.

## Fix

- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py` now normalizes
  `runtime_content_tree_sha256` from direct evidence or nested runtime custody.
- `src/tac/optimization/l5_v2_measurement_schedule.py` now rejects each
  side-info variant unless:
  - `contest_cpu` and `contest_cuda` use the same `archive_sha256`; and
  - either their `runtime_content_tree_sha256` matches, or their
    `runtime_tree_sha256` matches when no content tree is available.
- Added focused regressions for archive mismatch and runtime-content mismatch.
- Backfilled the TT5L trained CUDA seed/effect-curve cell with recovered
  runtime content tree
  `105fc0834cfb8a54b8f46edb81a030d076369c3062f3066c1800602f9d6035f5`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_measurement_schedule.py -q
.venv/bin/python -m ruff check src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/optimization/l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_measurement_schedule.py
.venv/bin/python -m py_compile src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/optimization/l5_v2_measurement_schedule.py tools/build_l5_v2_sideinfo_effect_curve.py tools/build_l5_v2_lattice_measurement_schedule.py tools/build_l5_v2_paired_measurement_dispatch_plan.py
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py -q
```

Observed:

- `16 passed`
- `All checks passed!`
- py_compile clean
- broader L5 staircase regression: `123 passed`

## Current State

The L5 v2 side-info curve remains fail-closed. The only seeded cell is TT5L
`trained` on `contest_cuda`; the missing `contest_cpu` trained cell and all
zero/random-LSB/shuffled/ablated controls still block architecture lock and any
promotion.
