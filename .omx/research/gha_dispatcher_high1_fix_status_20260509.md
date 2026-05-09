# GHA dispatcher HIGH 1 fix status — coordination for a3c89347

<!-- generated_at: 2026-05-09T00:00:00Z, from_state_hash: codex_round2_fix_in_flight -->

## Status: LANDED (lane_codex_round2_custody_concurrency_fix at L0)

Codex round-2 review found that `tools/dispatch_cpu_eval_via_github_actions.py:329-418`
uses unbounded substring match on `submission_name` for both
`run_log_mentions_submission` and `download_artifact`. With existing names
like `apogee` and `apogee_stack_b100`, dispatching `apogee` can attach a
`[contest-CPU]` score from a concurrent `apogee_stack_b100` run.

## Coordination requirement for a3c89347

If a3c89347 is dispatching the 12 A1 bias-correction variants (which all
share an `apogee_stack_*` name prefix), the substring-match bug WILL
produce cross-attribution under any concurrency.

**Hold dispatch until this status file shows `STATUS: LANDED`** OR
serialize dispatches strictly (one variant active at a time) AND mark
all results `[contest-CPU candidate; pending custody verification]` to
re-validate post-fix.

## Fix scope

1. `run_log_mentions_submission(run_id, repo, name)`: use regex with
   `\b<name>\b` boundary OR explicit `--submission-dir <path>` parsing
   where `Path(path).name == name`.
2. `download_artifact(...)`: same exact-match logic against
   `submission_dir:` lines in artifact metadata.
3. Fallback artifact selection: FAIL CLOSED on ambiguity by raising
   `AmbiguousSubmissionMatchError` with both candidates.
4. 8-12 new regression tests: `apogee` vs `apogee_stack_b100`,
   `apogee_stack_b100` vs `apogee_stack_b100_v2`, exact match still
   works, ambiguous concurrent runs raise the new error, empty/whitespace
   name rejected.

## Status updates

- 2026-05-09T00:00:00Z — IN-FLIGHT: fix lane pre-registered at L0;
  catalog #127 (`check_authoritative_tag_requires_custody_metadata`) +
  #128 (`check_continual_learning_writes_use_lock`) claimed.
- 2026-05-09T06:10:00Z — LANDED: dispatcher uses exact `submission_dir`
  token identity for run logs and artifact fallback; duplicate matching
  reports fail closed via `AmbiguousSubmissionMatchError`. Concurrent A1 CPU
  eval harvesters may proceed only after running the focused regression suite
  and preserving report custody fields.
- 2026-05-09T07:00:00Z — COMPLETE: HIGH 2 (custody validator with
  `CustodyVerdict` typed taxonomy) + MEDIUM (locked transactional writes via
  fcntl) also landed alongside HIGH 1. STRICT preflight gates #127
  (`check_authoritative_tag_requires_custody_metadata`) + #128
  (`check_continual_learning_writes_use_lock`) are strict in `preflight_all()`
  at live count 0. Round-2 adversarial hardening removed whole-file validator
  and lock co-owner accepts; #127 now requires line-local custody routing or an
  explicit same-line waiver, and #128 now requires a local `_posterior_lock`
  context for bare `save_posterior(...)`. 21 focused #127/#128 tests pass.
  Canonical landing memo:
  `feedback_codex_round2_custody_concurrency_fix_landed_20260509.md`.
