# PR106 UNIWARD runtime packet dispatch repair - 2026-05-08

## Scope

This ledger supersedes the earlier no-go finding for the PR106 UNIWARD
runtime packet dispatch path. The byte candidate itself was not changed:

- Archive: `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
- Bytes: `150511`
- SHA-256: `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- Evidence grade before CUDA returns: `empirical` plus exact-eval dispatch in flight

## Blocker Found

Worker A correctly found that the generated packet's `submission_dir/inflate.sh`
was not self-contained when used as an external inflate path. It tried to import
through the contest `submissions/<name>` module layout, so canonical
`experiments/contest_auth_eval.py --inflate-sh .../submission_dir/inflate.sh`
could fail before reaching the candidate logic.

## Repair

`tools/build_pr106_uniward_runtime_packet.py` now emits a self-contained
`inflate.sh` that:

- resolves its own directory,
- selects Python from `PR106_UNIWARD_PYTHON`, `PYTHON`, repo `.venv/bin/python`,
  `python`, then `python3`,
- calls the colocated `inflate.py` directly, and
- avoids `PYTHON_INFLATE`, which collides with the robust-current config guard.

`scripts/lightning_exact_eval_repro.py` and
`scripts/launch_lightning_batch_job.py` now recursively include and require
non-hidden external inflate runtime files, so nested helpers such as
`submission_dir/src/model.py` and `submission_dir/src/codec.py` are part of the
staged Lightning closure.

## Verification

Local checks:

- `python tools/build_pr106_uniward_runtime_packet.py --rms-target 0.05 --output-dir experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke`
- `python tools/verify_pr106_uniward_runtime_packet_sha256.py`
- `PATH="$PWD/.venv/bin:$PATH" python experiments/contest_auth_eval.py --archive experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip --inflate-sh experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/submission_dir/inflate.sh --upstream-dir upstream --device cpu --work-dir <tmp> --inflate-timeout 10 --evaluate-timeout 10`

The no-GPU auth-eval probe now reaches candidate code and fails only on the
intentional CUDA requirement (`inflate requires GPU`), not on Python lookup or
module import.

Focused tests:

- `python -m pytest src/tac/tests/test_build_pr106_uniward_runtime_packet.py src/tac/tests/test_lightning_exact_eval_repro.py src/tac/tests/test_lightning_batch_jobs.py src/tac/tests/test_build_cross_paradigm_admm_x_op1_finalizer.py src/tac/tests/test_dispatch_command_builder_shapes.py src/tac/tests/test_dual_layer_stc_av1_codec.py src/tac/tests/test_zipwire_archive.py -q` -> `205 passed`
- `python -m ruff check ...` on the touched Python implementation and test files -> clean
- `cargo fmt -p zipwire --check && cargo test -p zipwire` -> clean
- `git diff --check` on owned paths -> clean

## Dispatch State

Predispatch sanity refused the first unanchored PR106 Lagrangian calibration
run because no prior CUDA anchors existed and no distortion proxy file was
available. The override was logged explicitly as a first-calibration dispatch:

`Operator requested first PR106 Lagrangian calibration exact CUDA run after byte
closed runtime and import path repair; proxy unavailable because anchor file is
absent.`

Active exact-eval job:

- Lane: `pr106_uniward_lagrangian_runtime_packet`
- Job: `pr106-uniward-rms005-exact-20260508T083555Z`
- Lightning user/teamspace/studio: `adpena` / `comma-lab` /
  `lossy-compression-challenge`
- Remote: `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`
- Status at `2026-05-08T08:41:02Z`: `Running`
- Expected archive bytes/SHA match the candidate above
- Adjudication baseline: PR106 `0.20454`, archive bytes `186239`

This remains non-promotable until the Lightning CUDA auth eval returns
`contest_auth_eval.json` and adjudication artifacts for the exact archive bytes.

## Supersession note - exact CUDA result harvested

The Lightning exact CUDA auth eval returned terminal artifacts for the exact
packet above:

- Artifact dir:
  `experiments/results/lightning_batch/pr106-uniward-rms005-exact-20260508T083555Z`
- Result JSON:
  `experiments/results/lightning_batch/pr106-uniward-rms005-exact-20260508T083555Z/contest_auth_eval.adjudicated.json`
- Archive bytes: `150511`
- Archive SHA-256:
  `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- Hardware: Tesla T4, `device=cuda`, `n_samples=600`
- SegNet distortion: `0.0019625`
- PoseNet distortion: `0.00016559`
- Canonical recomputed score: `0.3371617511972341`
- Adjudication status: `REGRESSION_REVIEW_REQUIRED`
- Paper/science grade: `A-negative scoped forensic`

Interpretation: the byte-closed PR106 UNIWARD-Lagrangian `rms_target=0.05`
implementation is an exact CUDA regression relative to the active PR106-family
frontier. The smaller archive did not compensate for the SegNet/PoseNet
distortion increase. This retires only the measured `rms_target=0.05` runtime
packet configuration; it does not kill UNIWARD, Lagrangian allocation,
score-aware weighting, training-time UNIWARD, or future scorer/Jacobian-weighted
variants.

Dispatch claim closure is already recorded in
`.omx/state/active_lane_dispatch_claims.md` as
`completed_regression_review_required` for
`pr106-uniward-rms005-exact-20260508T083555Z`, with `no promotion/no family
kill` in the notes.
