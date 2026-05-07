# Apogee int6 predispatch current gate

Date: 2026-05-07
Agent: codex
Scope: `apogee_int6` exact-eval readiness and predispatch sanity

## Summary

The current `tools/predispatch_sanity.py` result for
`experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip`
correctly blocks exact dispatch without override.

The blocker is not the obsolete rule that a lossy archive can never beat a
lossless baseline. The current rule computes the official rate term plus the
SHA-tied scorer-basin parity component penalty. For this candidate:

- predicted band: `[0.190, 0.204]`
- lossless baseline anchor: `0.20945673`
- official rate delta: `-0.010513`
- parity component penalty: `+0.107727`
- SHA-tied rate-distortion floor: `0.3067`

Since `predicted_high=0.2040` is below the SHA-tied floor, the prediction is not
supported by the current readiness evidence. The parity evidence remains useful
as calibration and basin evidence, but it is not score-lowering evidence.

## Command

```bash
.venv/bin/python tools/predispatch_sanity.py \
  --archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip \
  --predicted-low 0.190 \
  --predicted-high 0.204 \
  --rel-err-pct 1.55 \
  --lane-class apogee_intN \
  --distortion-proxy-ran \
  --readiness-evidence-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json \
  --json
```

Output artifact:
`experiments/results/apogee_int6_basin_parity_20260507_claude/predispatch_sanity_with_parity.json`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_predispatch_sanity.py \
  src/tac/tests/test_build_field_meta_dispatch_selection.py::test_field_meta_selector_summarizes_operator_next_steps_without_dispatch \
  -q
```

Result: `14 passed`.

```bash
.venv/bin/python -m ruff check \
  tools/predispatch_sanity.py \
  tools/build_field_meta_dispatch_selection.py \
  src/tac/tests/test_predispatch_sanity.py
```

Result: passed.

## Next action

Do not dispatch Apogee int6 as score-lowering evidence from this parity packet.
Treat it as a calibration target unless a new artifact supplies
score-lowering component evidence or an explicit operator override is recorded
for calibration exact eval.
