# L5-v2 Probe Axis-Evidence Hardening

Date: 2026-05-16
Owner: codex
Scope: `src/tac/optimization/l5_v2_probe_disambiguator.py`

## Finding

The L5-v2 probe-disambiguator could treat paired axis labels as enough
architecture-lock evidence. A row with `exact_axes=["contest_cpu",
"contest_cuda"]`, valid-looking archive/runtime hashes, and side-info flags
could become eligible without per-axis exact-eval evidence.

That was too thin for the L5-v2 staircase: architecture lock should require
axis-specific custody, sample count, hardware/device, command/log trace,
component distances, archive bytes, and score formula closure.

## Fix

- Added `axis_evidence` to each `L5V2ProbeObservation`.
- Required one evidence object for each required axis: `contest_cpu` and
  `contest_cuda`.
- Validated per-axis archive/runtime SHA, score, SegNet distance, PoseNet
  distance, archive bytes, sample count, hardware, inflate device, eval device,
  auth-eval command, and log path.
- Recomputed the contest formula per axis and blocked formula mismatches.
- Updated the emitted template so future probe artifacts carry the required
  per-axis fields.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_disambiguator.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_disambiguator.py`
- `.venv/bin/python -m py_compile src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_disambiguator.py`
- `git diff --check -- src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_disambiguator.py`

## Result

The probe remains planning-only and still cannot claim score movement, but
architecture lock now requires paired exact-axis evidence rather than labels.
