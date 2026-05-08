# Preflight Profile No-Output Diagnosis - Worker P2 - 2026-05-08

generated_at_utc: 2026-05-08T13:25:00Z
scope: `.venv/bin/python -m tac.preflight --profile q_faithful_dilated_88k`
status: patched with focused scan-scope and verbosity improvement

## Symptom

The profile-level preflight command produced no stdout/stderr for at least
30 seconds and had previously been stopped after more than two minutes.
The control command with codebase checks disabled completed quickly:

- `.venv/bin/python -u -m tac.preflight --profile q_faithful_dilated_88k --no-codebase`: `PREFLIGHT PASSED` in 0.47s.

## Diagnosis

`faulthandler.dump_traceback_later(12)` pinned the silent path to the first
codebase scan:

- `preflight_all`
- `check_codebase_drift`
- `_scan_python_for_forbidden`
- `Path.read_text`
- `tac.preflight_fs_cache._cached_read_text`
- `Path.resolve`

The live tree contains roughly 12,004 Python files under `experiments/`, with
roughly 11,619 under `experiments/results/`. Those result bundles include
public clones, recovered runtime trees, and provider artifacts. They are not
source launch surfaces for the drift check; they are covered by separate
custody/release/audit gates.

A later bounded stack sample also showed another expensive monolithic scanner:
`preflight_loader_format_safety` was parsing Python files in
`_scan_python_for_unsafe_renderer_loader` at the 45s mark. That check already
emits a normal completion line once it finishes; the zero-output symptom came
from the first drift check doing work before any visible progress line.

`tools/all_lanes_preflight.py` passed because it runs smaller explicit gates
and does not call the monolithic `tac.preflight.preflight_all()` codebase scan.

## Patch

- `check_codebase_drift` now accepts `repo_root` and `verbose` keyword
  arguments.
- The profile CLI now passes `verbose` into `check_codebase_drift`, so the first
  long-running check emits an immediate `[codebase-drift]` line.
- The Python drift scan skips `experiments/results/**` artifacts while still
  scanning source files under `experiments/`, `scripts/`, `src/tac/contrib`,
  `src/tac/deploy`, and `src/tac/experiments`.
- Focused regression tests cover the skip boundary, the still-scanned source
  boundary, and the new verbose scope line.

## Verification

- Focused `check_codebase_drift(strict=False, verbose=True)` on the live tree
  now reports scope immediately and finishes in 3.94s, scanning 492 source
  Python files.
- A 15s bounded profile-command run now emits progress through the arity check
  instead of producing zero output.
- A 120s bounded profile-command run still did not complete the full monolithic
  preflight on this dirty tree, but it produced continuous progress through
  `training-needs-auth-eval`. This patch intentionally does not broaden cleanup
  or rewrite the full preflight scheduler.

## Follow-Up Boundary

This does not weaken dispatch/release hygiene for generated result bundles:
untracked-source inventory, orphan/release index audits, public-release hygiene,
and reverse-engineering custody gates remain the correct surfaces for
`experiments/results/**` artifacts.

## Codex Verification Update - 2026-05-08T13:36Z

- `.venv/bin/python -u -m tac.preflight --profile q_faithful_dilated_88k`
  now emits immediate progress, including the bounded codebase-drift scope line
  and `OK: scanned 492 source Python file(s)`.
- The same full command still did not finish quickly on the dirty tree; it
  reached the filename contract after about 37s before the local command
  session ended. Treat this as remaining full-preflight runtime work, not the
  original zero-output hang.
- `.venv/bin/python -u -m tac.preflight --profile q_faithful_dilated_88k --no-codebase`:
  `PREFLIGHT PASSED`.
- `.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings`: all 25
  operator gates passed after strict untracked-source dispositions were updated.
