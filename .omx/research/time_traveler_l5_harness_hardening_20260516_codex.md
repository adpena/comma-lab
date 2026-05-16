# Time-Traveler L5 Harness Hardening - 2026-05-16

## Scope

Hardened `tools/smoke_time_traveler_l5_autonomy_macos_cpu.py` after adversarial
review found three evidence bugs in the free macOS-CPU companion harness:

- The harness exposed a `--samples` knob even though
  `experiments/contest_auth_eval.py` enforces the contest 600-sample count.
- Missing TT5L runtime fell back to `submissions/exact_current/inflate.sh`,
  which could evaluate the wrong runtime and suppress packet/runtime bugs.
- macOS-CPU advisory scores were classified with `falsification` language,
  despite CLAUDE/AGENTS evidence rules requiring paired contest-axis evidence
  before promotion, ranking, family retirement, or falsification.

## Landing

- Removed the sample-count CLI surface. The harness now reports
  `samples_evaluated` only from the evaluator payload `n_samples`.
- Added archive-local TT5L runtime resolution. Default lookup requires
  `submission_dir/inflate.sh` adjacent to the archive or output tree. Missing
  runtime is now `tt5l_inflate_sh_not_found_no_exact_current_fallback`.
- Replaced falsification verdicts with
  `escalate_above_threshold_requires_contest_axis_recheck`.
- Added blocker `macos_cpu_advisory_cannot_falsify_architecture` to every
  summary.
- Updated the neighboring Time-Traveler trainer regression so it verifies the
  canonical shared auth-eval gate owns fail-closed `subprocess.run(...,
  capture_output=True, ...)` semantics.

## Evidence

Commands run:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_smoke_time_traveler_l5_autonomy.py \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests

.venv/bin/python -m ruff check \
  tools/smoke_time_traveler_l5_autonomy_macos_cpu.py \
  src/tac/tests/test_smoke_time_traveler_l5_autonomy.py \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py

git diff --check -- \
  tools/smoke_time_traveler_l5_autonomy_macos_cpu.py \
  src/tac/tests/test_smoke_time_traveler_l5_autonomy.py \
  src/tac/tests/test_train_time_traveler_full_cpu_mode.py
```

Results:

- `131 passed, 1 skipped`
- `ruff`: all checks passed
- `git diff --check`: clean

## Next

The harness still produces no score claim and no promotion-ready artifact.
Any TT5L result that looks materially good or bad must be paired on the exact
same archive/runtime packet across `[contest-CPU]` Linux x86_64 and
`[contest-CUDA]` before it can move lane status.
