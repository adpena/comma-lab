# PR106 Yshift Lightning Batch Dispatch

Date: 2026-05-06

## Scope

Hardened and launched the PR106 yshift `score_table` path through Lightning
Batch Jobs instead of the interactive Studio tmux path.

The interactive Studio SSH target was reachable, but the current Studio runtime
had no CUDA device. Batch Jobs are therefore the correct route for promotion
work because the job requests an explicit T4 machine while still using SSH for
source and claim-ledger staging.

## Code

- `tools/lightning_dispatch_pr106_yshift_score_table.py`
  - New canonical launcher for the score-table lane.
  - Records a stable active claim for `lane_pr106_yshift_score_table`.
  - Stages the repo, the PR106 archive, and the freshly updated claim ledger.
  - Supports `--dry-run-batch` and closes dry-run claims as `completed_dry_run`.
  - Defaults to Lightning Batch Jobs; Studio tmux remains an emergency/manual
    backend.
- `scripts/remote_lane_pr106_yshift_sidechannel.sh`
  - Fixed stale `NO_NVDEC_NEEDED` header.
  - Defines `log` before the NVDEC/DALI failure path can call it.
  - Skips NVDEC probe only for `zero` CPU smoke.
  - Allows `PR106_YSHIFT_LOG_DIR` so Batch Jobs can write into the canonical
    Lightning artifact directory.
  - Fails early when score-table generation lacks a stable instance/job id for
    claim verification.

## Evidence

- `[empirical:test]` `.venv/bin/python -m pytest src/tac/tests/test_lightning_dispatch_pr106_yshift_score_table.py src/tac/tests/test_lightning_dispatch_pr106_stack.py src/tac/tests/test_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_sidechannel.py -q`
  - Result: passing at time of ledger creation.
- `[empirical:lint]` `.venv/bin/python -m ruff check tools/lightning_dispatch_pr106_yshift_score_table.py src/tac/tests/test_lightning_dispatch_pr106_yshift_score_table.py`
  - Result: passing at time of ledger creation.
- `[empirical:syntax]` `bash -n scripts/remote_lane_pr106_yshift_sidechannel.sh && .venv/bin/python -m py_compile tools/lightning_dispatch_pr106_yshift_score_table.py`
  - Result: passing at time of ledger creation.
- `[empirical:dry-run-batch]` job `lane_pr106_yshift_score_table_dryrun_20260506T0731Z`
  - SSH auth passed.
  - Remote manifest verification passed.
  - `file_count=1497`, `total_bytes=209584578`.
  - Claim was closed as `completed_dry_run`.
- `[external:lightning-submit]` job `lane_pr106_yshift_score_table_20260506T0732Z`
  - Submitted via Lightning Batch Jobs on T4.
  - Initial SDK status: `Pending`.
  - Active claim: `lane_pr106_yshift_score_table` /
    `lane_pr106_yshift_score_table_20260506T0732Z`.

## Promotion Status

Evidence grade: `empirical` until the Batch Job produces and we harvest
`contest_auth_eval.json`.

No score claim is made from the scorer table itself. Promotion requires the
charged yshift archive and exact CUDA auth eval JSON from the submitted Batch
Job, followed by adjudication/custody review.
