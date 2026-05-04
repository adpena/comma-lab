# Dispatch CLI And Shell Bug Hardening - 2026-05-04

## Scope

This pass hardens repeated low-level bug classes that slowed the final sprint.
It does not dispatch remote work and does not make score claims.

Owned write scope:

- `.omx/research/dispatch_cli_and_shell_bug_hardening_20260504_codex.md`
- `tools/check_dispatch_cli_shell_hazards.py`
- `src/tac/tests/test_dispatch_cli_shell_hazards.py`
- narrow regression tests for existing dispatch helpers.

## Real CLI Surfaces Inspected

Verified with live argparse help before changing tests or examples:

```text
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --help
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-ssh --help
.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status --help
.venv/bin/python tools/claim_lane_dispatch.py claim --help
```

Important result: `--required-device` and `--required-samples` belong to
`scripts/adjudicate_contest_auth_eval.py`, not to
`scripts/launch_lightning_batch_job.py exact-eval`. The Lightning launcher
already emits the adjudicator command internally when `--adjudicate` is passed.

## Bugs Found

### 1. Stale adjudicator flags passed to the Lightning launcher

Failure mode: a command copied adjudicator-only flags into
`launch_lightning_batch_job.py exact-eval`, producing strict argparse failure
instead of a queued exact eval.

Permanent guard added:

- `scripts/launch_lightning_batch_job.py` now gives explicit stale-flag
  diagnostics for `--required-device` and `--required-samples`.
- `src/tac/tests/test_lightning_batch_jobs.py` now asserts those flags are
  rejected with the specific adjudicator-only hint.
- `tools/check_dispatch_cli_shell_hazards.py` scans future operational snippets
  for those flags when they are attached to `launch_lightning_batch_job.py`.

### 2. zsh special variable `path` clobbers command lookup

Failure mode: in zsh, assigning or iterating with the variable name `path`
mutates the shell command lookup path. A one-liner using `path` made commands
such as `dirname` and `mkdir` disappear during the sprint.

Permanent guard added:

- `tools/check_dispatch_cli_shell_hazards.py` flags `path=` / `for path in` /
  `read ... path` in zsh-facing snippets and docs.
- The scanner strips heredoc bodies before scanning so Python code inside a
  shell heredoc can still use normal Python variables named `path`.

Recommended operator rule:

```text
In zsh snippets, use file_path, item_path, artifact_path, rel_path, or archive_path.
Do not use path as a shell variable.
```

### 3. macOS `find -printf` incompatibility

Failure mode: GNU `find -printf` works on Linux remotes but fails on macOS/BSD
local shells.

Permanent guard added:

- `tools/check_dispatch_cli_shell_hazards.py` flags local/macOS-facing
  `find ... -printf` commands.
- The scanner permits remote Linux contexts such as `/workspace/...` over SSH,
  where the command is intentionally executed on Linux.

Recommended local replacement:

```text
Use Python pathlib/stat for local file timestamps and byte accounting, or a
POSIX/BSD-safe find+xargs/stat form verified on macOS.
```

### 4. Remote dispatch claim closure hygiene

Failure mode: dispatch claims are append-only; terminal rows must close only
the matching older active job. If a terminal row for a different job closed a
whole lane, a live duplicate could be hidden.

Permanent guard added:

- `src/tac/tests/test_claim_lane_dispatch.py` now covers the adversarial case:
  a terminal row for `job_done` must not close an older active row for
  `job_still_active` in the same lane.

## Verification Commands

```text
.venv/bin/python -m py_compile scripts/launch_lightning_batch_job.py tools/claim_lane_dispatch.py tools/check_dispatch_cli_shell_hazards.py
.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py::test_batch_job_cli_rejects_adjudicator_only_flags_with_specific_hint src/tac/tests/test_claim_lane_dispatch.py::test_claim_helper_terminal_row_does_not_close_different_active_job src/tac/tests/test_dispatch_cli_shell_hazards.py -q
.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --repo-root . --scan-path scripts --scan-path docs --scan-path reports --json-out .omx/state/dispatch_cli_shell_hazards_20260504_codex.json
git diff --check -- scripts/launch_lightning_batch_job.py tools/check_dispatch_cli_shell_hazards.py src/tac/tests/test_lightning_batch_jobs.py src/tac/tests/test_claim_lane_dispatch.py src/tac/tests/test_dispatch_cli_shell_hazards.py .omx/research/dispatch_cli_and_shell_bug_hardening_20260504_codex.md
```

Results:

- Python compilation passed.
- Focused pytest passed: `6 passed in 0.62s`.
- Strict scanner passed across `scripts`, `docs`, and `reports` with
  `hazard_count=0`; JSON artifact:
  `.omx/state/dispatch_cli_shell_hazards_20260504_codex.json`.
- `git diff --check` passed for touched files.

## Dispatch Status

No remote jobs were dispatched. No dispatch claims were created or closed by
this hardening pass.
