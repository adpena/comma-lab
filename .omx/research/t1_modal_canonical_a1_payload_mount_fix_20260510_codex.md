# T1 Modal canonical A1 payload mount fix

Generated: `2026-05-10T09:45:00Z`

## Verdict

The T1 Modal actuator now refuses or prevents the missing-`A1_canonical`
failure that stopped `t1_balle_modal_guard_a3311268_20260510T0831Z` before
training.

## Fix

- `experiments/modal_t1_balle_endtoend.py` mounts the resolved local
  `experiments/results/A1_canonical` payload into
  `/workspace/pact/experiments/results/A1_canonical`.
- It also mounts `.omx/state/canonical_a1_designation.md` into the remote
  `.omx/state` directory.
- Local plan metadata now records canonical A1 archive/checkpoint/latent/memo
  paths, sizes, and SHA-256s.
- Plan validation fails closed if any required canonical A1 payload component
  is missing.
- The Modal worker now checks for the remote canonical A1 directory and
  designation memo before running the remote lane script, returning
  `stage=missing_canonical_a1_payload` instead of spending train/eval time on a
  doomed run.

## Local canonical A1 payload snapshot

- Archive: `178262` bytes,
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Checkpoint: `925250` bytes,
  `5846043656d8261855d58de6d9c3568c7b2c4ecabdc3b4b1729aef913f7cb272`
- Extracted latents: `69088` bytes,
  `5ba13604837e27b834867d2ace06d7c21228e8b97d241da6744020bee9f79090`
- Designation memo: `3617` bytes,
  `064842006b08e5d7d0527bfa3dc06c112ed3de97d423a1fd43f25a6169c2ea45`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_modal_t1_balle_endtoend.py`
  -> `12 passed in 0.54s`
- `.venv/bin/python experiments/modal_t1_balle_endtoend.py plan --label t1-plan-canonical-check --epochs 1 --batch-size 1 --timeout-hours 24 --train-timeout-hours 2 --max-target-pairs 4`
  -> `ready_for_modal_dispatch_command=true`, `canonical_a1_payload.ready_for_modal_mount=true`
- `git diff --check -- experiments/modal_t1_balle_endtoend.py src/tac/tests/test_modal_t1_balle_endtoend.py`
  -> clean

## Rerun gate

Do not rerun T1 until the red-team overclaim fixes also land. After that, a
fresh T1 claim may use the guarded short-run shape again, but the next dispatch
must be treated as a new measured configuration with no promotion unless exact
CUDA auth-eval blockers are zero.
