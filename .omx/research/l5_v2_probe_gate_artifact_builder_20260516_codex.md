# L5 v2 probe gate artifact builder - 2026-05-16

## Scope

The L5 v2 readiness surface required the `c1_z5_tt5l_probe_disambiguator` gate
to carry a wrapped `probe_disambiguator` artifact with observations, a
recomputable verdict, and a verdict SHA-256. The operator-facing next action
previously emitted only the raw verdict JSON, which could not satisfy the gate.

No score claim, promotion claim, or dispatch authorization is made here.

## Fixes landed

- Added `tools/build_l5_v2_probe_gate_artifact.py`.
- Added `build_l5_v2_probe_gate_artifact()` in
  `src/tac/optimization/l5_v2_probe_disambiguator.py`.
- L5 v2 readiness now points the next action at the gate artifact builder:
  `.omx/research/l5_v2_probe_gate_artifact_20260516_codex.json`.
- Added canonical probe-gate artifact discovery. Valid artifacts can be
  auto-consumed by `l5_v2_dispatch_readiness`; blocked artifacts are skipped and
  leave the probe gate unsatisfied.

## Current artifact status

Generated from the existing empty template:

```bash
.venv/bin/python tools/build_l5_v2_probe_gate_artifact.py \
  --input-json .omx/research/l5_v2_probe_template_20260516_codex.json \
  --output-json .omx/research/l5_v2_probe_gate_artifact_20260516_codex.json
```

Result: exit code `1`, as expected.

The artifact is intentionally blocked:

- `architecture_lock_allowed=false`
- `score_claim=false`
- missing paired exact CPU/CUDA observations for C1, Z5, and TT5L
- missing byte-closed archive artifacts for all candidates

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_v2_probe_disambiguator.py \
  src/tac/tests/test_l5_staircase_v2.py -q
```

Result: `108 passed in 0.70s`.

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_v2_probe_disambiguator.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  tools/build_l5_v2_probe_gate_artifact.py
```

Result: `All checks passed!`.
