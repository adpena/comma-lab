# L5 v2 Side-Info Effect-Curve Producer

- date: `2026-05-16`
- agent: `codex`
- scope: Time-Traveler L5 v2 staircase, side-info architecture-lock gate
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Why

The L5 v2 staircase now correctly blocks TT5L architecture lock until a paired
CPU/CUDA side-info effect curve exists. The missing piece was a canonical
producer for that artifact. Without it, operators could know the next blocker
but still have no byte-closed, axis-labelled, repeatable way to close it.

## Landed

Added `tac.optimization.l5_v2_sideinfo_effect_curve` plus
`tools/build_l5_v2_sideinfo_effect_curve.py`.

The producer consumes one or more cell JSON files. Each cell carries:

- `axis`: `contest_cpu` or `contest_cuda`
- `variant`: `zero`, `random_lsb`, `shuffled`, `trained`, or `ablated`
- exact-eval custody fields, either at top level or under `evidence`

For every axis/variant cell it runs `validate_exact_eval_evidence` with:

- archive/runtime SHA validation
- 600-sample exact axis requirement
- official score formula recomputation
- hardware/device checks for CPU vs CUDA axes
- artifact path and log path existence
- inflated-output manifest SHA and aggregate raw-output SHA checks

The artifact sets `predicate_passed=true` only when:

- all ten paired cells are present;
- every cell has valid exact-eval custody;
- the trained side-info variant beats or ties the best control on both CPU and
  CUDA axes;
- no score/promotion/dispatch authority flag is true.

This is a causal usefulness gate, not a score promotion gate.

## Operator Command

```bash
.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py \
  --cell-json <paired_tt5l_sideinfo_cells.json> \
  --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json
```

Then feed the output into:

```bash
.venv/bin/python tools/build_l5_v2_lattice_measurement_schedule.py \
  --probe-intake-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json \
  --sideinfo-effect-curve-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json
```

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_sideinfo_effect_curve.py tools/build_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py`
- `.venv/bin/python -m py_compile src/tac/optimization/l5_v2_sideinfo_effect_curve.py tools/build_l5_v2_sideinfo_effect_curve.py src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_staircase_v2.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_sideinfo_effect_curve.py -q` -> `4 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py -q` -> `109 passed`
- `git diff --check`
