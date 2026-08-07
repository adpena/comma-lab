# ddm_ab1 Receipt

status: COMPLETE-SCOPED
arm: ddm_ab1
utc: 2026-08-07T13:23:09Z
charter: `.omx/tmp/codex_runs/ab1_prompt.md`
common_contract: `.omx/tmp/codex_runs/_common_contract.md`

## Counts First

| item | count / status |
|---|---:|
| #254 residual warnings before AB1 | 140 |
| #254 residual warnings after AB1 | 0 |
| Heavy trainers scanned after AB1 | 150 |
| Adopted guards | 140 |
| Same-line waivers | 0 |
| Scanner vocabulary fixes | 0 |
| Survivors | 0 |
| Scorer / archive / remote launches | 0 |

## Classification

All 140 residual rows were classified as LIVE-FIREABLE heavy entrypoints under the existing
`check_heavy_witness_trainers_call_admission_guard` scope. Each file had a `main()` and an
argument-parsing point, so AB1 adopted `assert_governed_admission("<file-stem>")` immediately
after argument parsing. No row needed an experimental-dead waiver and no row was a scanner
misdetect.

The adoption set is exactly the pre-edit residual set emitted by:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ... check_heavy_witness_trainers_call_admission_guard(strict=False, verbose=True) ... PY`

Representative checked insertions:

- `experiments/train_cnerv_as_renderer.py`: guard after `args = parse_args(argv)`.
- `experiments/train_levelset_witness_realized_through_R_torch.py`: guard after `args = build_parser().parse_args(argv)`.
- `src/tac/mps_gap_experiment/train_on_mps_cli.py`: guard after `args = parse_args(argv)`.

AB1 also added `test_ab1_real_repo_admission_backlog_drained`, which asserts the real repo gate
returns `[]`.

## Recall Evidence

Read before edits:

- Governing files: `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` (byte-identical), `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Charter files: `.omx/tmp/codex_runs/ab1_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- AH1 handoff: `.omx/research/ddm_ah1_20260807/AUDIT_TABLE.md`, `RECEIPT.md`, `NEXT_IF_RESUMED.md`.
- Source surfaces: `src/tac/preflight.py`, `src/tac/admission_guard.py`, `src/tac/tests/test_admission_coverage_gate.py`, the five AH1 exemplar trainers.
- Corpus searches: `.omx/research`, `.omx/state`, `docs`, `src`, `experiments`, `tools`, canonical equations JSON for `admission|guard|memory|governor|#254|train`.

Findings beyond the charter seeds:

- RR9 and mx1c receipts documented the original top-tier five residuals and confirmed the guard is opt-in per entrypoint.
- Canonical equation search found related Metal mem-probe and launch-protocol admission laws, but no extra AB1-specific adoption constraint beyond the existing #254 scanner.
- Memory quick pass found no `ddm_ab1` / #254-specific registry hit beyond the already-loaded Pact discipline and the #899 unrelated memory surface.

What changed in plan:

- Because every residual had a concrete `main()` and argument parse point, AB1 used mechanical adoption for all residuals instead of issuing waivers.
- Because the scanner drained to zero, AB1 added a zero-backlog regression instead of a survivor table.

## Verification

Commands run:

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ... ast.parse(...) ... PY`
  - Result: `ast_parse_ok 144 files`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_admission_coverage_gate.py`
  - Result: `13 passed in 5.83s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/tests/test_codex_arm_watch.py src/tac/tests/test_no_silent_failure_launch_hardening.py src/tac/tests/test_admission_coverage_gate.py`
  - Result: `38 passed in 9.05s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY' ... check_heavy_witness_trainers_call_admission_guard(strict=True, verbose=True) ... PY`
  - Result: `OK (150 heavy trainer(s) scanned); remaining=0`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file ...`
  - Result: `ab1-pass1_ok 141`, `ab1-pass2_ok 141`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_admission_coverage_gate.py`
  - Result: `15 entities compliant, 0 violations`.

## Authority Boundary

This arm is apparatus-only. It did not run `upstream/evaluate.py`, did not build `archive.zip`,
did not dispatch remote/GPU work, and did not claim a contest or macOS score row.

Own-vehicle frontier line remains the hot-state line observed for this arm:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

## Commit Attempt

Serializer command attempted with:

- 143 explicit files (140 trainer entrypoints, 1 test file, 2 receipt files)
- `--no-co-author`
- `--triality-legs none`
- `--triality-reason "apparatus-only admission hardening [no-triality] [p0-ledger-ok]"`
- one `--expected-content-sha256` per file

Outcome: blocked before commit at `git add` with rc=128:

```text
[subagent-commit-serializer] git add failed (rc=128):
error: unable to create temporary file: Operation not permitted
error: experiments/train_cnerv_as_renderer.py: failed to insert into database
error: unable to index file 'experiments/train_cnerv_as_renderer.py'
fatal: updating files failed
```

Post-failure index check:

`git diff --cached --name-status`

returned empty output, so no AB1 file remained staged.

Commit replay artifacts:

- `COMMIT_INTENT.md`
- `POST_EDIT_SHA256SUMS.txt`
