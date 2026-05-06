# PR106 Yshift Score-Table Builder

Date: 2026-05-06

## Scope

Canonicalized the PR106 yshift sidechannel lane so a precomputed CUDA scorer
candidate table can be reduced into deterministic charged archive bytes without
loading scorers at inflate time and without turning the table into a score
claim.

## Code

- `experiments/build_pr106_yshift_sidechannel.py`
  - Added `--search-mode score_table`.
  - Added `--score-table-npy`, `--score-table-manifest`, and
    `--candidate-radius`.
  - Added `choose_yshift_candidates_from_score_table_file(...)`.
  - Records source/archive/table SHA-256 custody, candidate-grid diagnostics,
    no-op retention behavior, and fail-closed dispatch blockers.
- `scripts/remote_lane_pr106_yshift_sidechannel.sh`
  - Accepts `PR106_YSHIFT_MODE=score_table`.
  - Requires `PR106_YSHIFT_SCORE_TABLE_NPY` for that mode.
  - Carries candidate radius and pair count through to the builder.
- `tools/dispatch_dryrun_pr106_sidechannels.py`
  - Checks the new yshift score-table flags and help surface.
- `src/tac/tests/test_pr106_yshift_sidechannel.py`
  - Covers table-shape validation, deterministic score-table reduction,
    archive emission, sidechannel roundtrip, and fail-closed metadata.

## Evidence

- `[empirical:pytest]` `.venv/bin/python -m pytest src/tac/tests/test_pr106_yshift_sidechannel.py src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q`
  - Result: `37 passed`
- `[empirical:ruff]` `.venv/bin/python -m ruff check --select F401,F821 experiments/build_pr106_yshift_sidechannel.py src/tac/tests/test_pr106_yshift_sidechannel.py tools/dispatch_dryrun_pr106_sidechannels.py`
  - Result: passed
- `[empirical:py_compile]` `.venv/bin/python -m py_compile experiments/build_pr106_yshift_sidechannel.py tools/dispatch_dryrun_pr106_sidechannels.py`
  - Result: passed
- `[empirical:bash-n]` `bash -n scripts/remote_lane_pr106_yshift_sidechannel.sh`
  - Result: passed
- `[empirical:preflight]` `.venv/bin/python -m tac.preflight --no-codebase`
  - Result: passed
- `[empirical:dryrun]` `.venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py --json`
  - Result: `ok=true`, `dispatch_attempted=false`, `score_claim=false`

## Promotion Status

Evidence grade: `empirical`.

This is not score evidence. Any archive emitted from `score_table` mode remains
`ready_for_exact_eval_dispatch=false` until a lane claim exists, the score-table
manifest is present, and the exact archive is scored through the canonical CUDA
auth-eval path.

Required next step: produce the actual CUDA candidate score table against the
exact PR106 source archive, build the full 600-pair yshift sidechannel archive
from that table, then run exact CUDA auth eval on the emitted archive.
