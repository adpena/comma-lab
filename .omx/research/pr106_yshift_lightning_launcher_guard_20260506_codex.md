# PR106 Yshift Lightning Launcher Guard

Date: 2026-05-06
Agent: Codex
Evidence grade: empirical

## Finding

`tools/lightning_dispatch_pr106_yshift_score_table.py --print-only` could emit
claim, stage, and SSH preflight commands with an empty `--ssh-target` when the
operator environment did not define `LIGHTNING_SSH_TARGET`. A real dispatch
would fail later in staging, after the operator had already inspected a command
surface that looked complete.

## Change

The launcher now fails before claim/stage dispatch when SSH staging is required
and no SSH target is configured. Explicit pre-staged Lightning Batch command
generation remains available with:

```text
--backend batch --skip-ssh-check --skip-stage
```

In print-only mode, skipped SSH/stage phases are no longer printed.

## Guard

`src/tac/tests/test_lightning_dispatch_pr106_yshift_score_table.py` now covers:

- missing SSH target fails before claim/stage dispatch.
- pre-staged Batch print-only mode omits SSH/stage commands and still prints
  the claim plus batch command.

## Dispatch Status

No GPU dispatch, remote staging, or lane claim was attempted. This is launch-DX
hardening only.
