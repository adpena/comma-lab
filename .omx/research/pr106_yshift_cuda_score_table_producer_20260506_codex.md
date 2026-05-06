# PR106 Yshift CUDA Score-Table Producer

Date: 2026-05-06

## Scope

Added the missing producer side for the PR106 yshift `score_table` builder path.
The new profiler turns the full candidate grid into a deterministic
`score_table.npy` artifact by scoring one yshift candidate at a time against the
official two-frame scorer pair.

## Code

- `experiments/build_pr106_yshift_score_table.py`
  - New canonical producer for `score_table.npy`.
  - Provides `--dry-run-plan` for CI and custody planning without scorer loads.
  - Real mode requires `--device cuda`, `torch.cuda.is_available()`, and a
    matching active row in `.omx/state/active_lane_dispatch_claims.md`.
  - Scores each candidate as `100*seg_dist + sqrt(10*pose_dist)` without the
    archive-rate constant, because all rows share the same sidechannel
    container before entropy coding.
  - Perturbs one frame at a time inside the official two-frame scorer pair, so
    PoseNet and SegNet semantics match upstream `evaluate.py`.
- `scripts/remote_lane_pr106_yshift_sidechannel.sh`
  - `PR106_YSHIFT_MODE=score_table` can now either consume an existing
    `PR106_YSHIFT_SCORE_TABLE_NPY` or build one first in a claimed CUDA job.
- `tools/dispatch_dryrun_pr106_sidechannels.py`
  - Checks the new score-table producer surface.
- `src/tac/tests/test_pr106_yshift_score_table.py`
  - Covers dry-run manifest emission, lane-claim parsing, contest formula
    without rate, and torch yshift integer-shift semantics.
- `src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py`
  - Adds the producer to the missing-builder fail-closed set.

## Evidence

- `[empirical:dry-run-plan]` `.venv/bin/python experiments/build_pr106_yshift_score_table.py --pr106-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip --out-dir /tmp/pr106_yshift_score_table_plan_codex --candidate-radius 1 --n-pairs 2 --dry-run-plan`
  - Result: wrote candidate grid and manifest with expected table shape `[4, 27]`.
- `[empirical:pytest]` `.venv/bin/python -m pytest src/tac/tests/test_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_sidechannel.py src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q`
  - Result: passing at time of ledger creation.
- `[empirical:ruff]` `.venv/bin/python -m ruff check experiments/build_pr106_yshift_score_table.py src/tac/tests/test_pr106_yshift_score_table.py experiments/build_pr106_yshift_sidechannel.py tools/dispatch_dryrun_pr106_sidechannels.py`
  - Result: passing at time of ledger creation.
- `[empirical:syntax]` `.venv/bin/python -m py_compile experiments/build_pr106_yshift_score_table.py experiments/build_pr106_yshift_sidechannel.py tools/dispatch_dryrun_pr106_sidechannels.py && bash -n scripts/remote_lane_pr106_yshift_sidechannel.sh`
  - Result: passing at time of ledger creation.

## Promotion Status

Evidence grade: `empirical`.

This producer creates scorer-backed profile data, not score evidence. A table
becomes useful only after:

1. A lane claim is active for the CUDA producer run.
2. The full table is generated against the exact PR106 archive.
3. `experiments/build_pr106_yshift_sidechannel.py --search-mode score_table`
   emits the charged archive.
4. The charged archive passes exact CUDA auth eval through
   `archive.zip -> inflate.sh -> upstream/evaluate.py`.
