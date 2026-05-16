# L5 v2 CPU Axis And Required-Axes Hardening

- date: `2026-05-16`
- agent: `codex`
- trigger: readonly L5-v2 adversarial review P2 findings
- score_claim: `false`
- promotion_eligible: `false`

## Fixes

1. Contest-CPU hardware evidence now requires explicit Linux provenance in addition to CPU/x86 tokens and forbidden macOS/GPU token rejection. This prevents an unlabelled `x86_64 cpu host` row from being accepted as contest-CPU evidence.
2. Paired CPU/CUDA axis-plan validation still accepts per-axis `inflate_device=cpu` and `eval_device=cpu` without requiring those device fields themselves to say Linux. Linux provenance belongs on the hardware field.
3. L5-v2 paired measurement dispatch planning now blocks schedule rows that omit `required_axes`; it no longer silently defaults malformed rows to CPU/CUDA.

## Verification

- `.venv/bin/python -m ruff check src/tac/exact_eval_custody.py src/tac/optimization/l5_staircase_v2.py src/tac/optimization/l5_v2_paired_measurement_dispatch_plan.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py src/tac/tests/test_l5_staircase_v2.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_paired_measurement_dispatch_plan.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_exact_eval_custody.py -q`

The broad L5-v2 slice passed with `164 passed`.
