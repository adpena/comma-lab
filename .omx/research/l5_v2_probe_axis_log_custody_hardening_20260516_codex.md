# L5 v2 probe axis-log custody hardening (2026-05-16)

## Scope

- `src/tac/optimization/l5_v2_probe_disambiguator.py`
- `src/tac/tests/test_l5_v2_probe_disambiguator.py`

## Finding

The L5 v2 probe-disambiguator required per-axis exact-eval rows to name a
`log_path`, but it did not bind those paths to durable repo-local files. A
malformed observation could therefore pass the axis evidence shape with a stale,
missing, transient, or outside-repo log path as long as the scalar score fields,
axis labels, devices, and formula closure were otherwise valid.

That is not sufficient for L5-v2 architecture lock-in. The disambiguator is the
gate that arbitrates C1/Z5/TT5L and must be able to trace every CPU/CUDA axis
row back to durable artifacts, not just JSON literals.

## Change

- Pass `artifact_base_dir=repo_root` into the shared exact-eval custody
  validator so `log_path` must exist.
- Add L5-specific blockers for transient and outside-repo axis logs:
  - `l5_v2_probe_axis_log_path_transient:<axis>`
  - `l5_v2_probe_axis_log_path_outside_repo:<axis>`
  - existing validator mapping now also emits
    `l5_v2_probe_axis_log_path_file_missing:<axis>`
- Update tests so eligible probe fixtures create repo-local CPU/CUDA log files,
  and add regression coverage for missing, `/tmp`, and outside-repo axis logs.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_disambiguator.py -q
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_autopilot_dispatch_ranking.py -q
.venv/bin/python -m ruff check src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_disambiguator.py
.venv/bin/python -m py_compile src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_disambiguator.py
```

Results:

- `19 passed`
- `57 passed`
- `ruff`: clean
- `py_compile`: clean

## Evidence Semantics

This is a fail-closed custody hardening only. It does not create a score claim,
does not authorize dispatch, and does not promote any L5-v2 candidate. It
prevents C1/Z5/TT5L architecture lock-in from accepting per-axis exact-eval rows
that cannot be traced to durable CPU/CUDA log artifacts.
