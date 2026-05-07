# PR91 HPM1 Reference Teacher-Forcing Seed Prefix Probe - 2026-05-07

## Scope

Local CPU-only HPM1/categorical grammar probe. No archive was built, no lane was
claimed, no GPU or remote dispatch was attempted, and no score is claimed.

## Command

```bash
.venv/bin/python tools/audit_pr91_hpm1_reference_teacher_forcing_probe.py \
  --spatial-order-candidates phase_major_row_major \
  --range-prefix-probe \
  --range-prefix-window-symbols 1 \
  --range-prefix-seed-symbol-counts 1,2,4,8,16,32,64,128 \
  --range-prefix-replay-symbol-limit 128 \
  --range-prefix-max-target-decoded-before 20000 \
  --json-out .omx/research/pr91_hpm1_reference_teacher_forcing_seed_prefix_20260507_codex.json
```

## Findings

- phase-major reference forcing remains live;
- candidate progress status: `phase_major_reference_forcing_remains_live_if_present`;
- range-prefix classification: `submitted_stream_reference_symbol_mismatch_in_seed_prefix`;
- seed prefixes `1`, `2`, and `4` decode the forced reference prefix but already
  produce a different submitted-word prefix;
- first submitted reference-symbol replay failure occurs at `first_8_symbols`;
- first mismatch: symbol index `7`, decoded symbol `2`, reference symbol `0`;
- full decode proven: `false`;
- byte-exact reencode proven: `false`;
- dispatch allowed: `false`.

## Interpretation

The blocker is no longer just spatial traversal. Phase-major PR85/QMA9 context
can advance the semantic failure row, but the submitted PR91 token/range stream
diverges almost immediately under local re-encoding. The next patch target is
to recover the true PR91 encoder semantic tokens or a bridge from the
phase-major prior into the submitted PR91 symbol stream before treating HPM1 as
byte-reencodeable.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py src/tac/tests/test_all_lanes_pr91_gate.py -q`
  - `32 passed in 168.31s`
- `.venv/bin/ruff check src/tac/pr91_hpm1_codec.py src/tac/tests/test_pr91_hpm1_codec.py tools/audit_pr91_hpm1_reference_teacher_forcing_probe.py`
  - passed
- `.venv/bin/python tools/all_lanes_preflight.py`
  - all 23 checks passed before this HPM1 commit
