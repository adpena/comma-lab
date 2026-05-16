# NSCS06 v8 Partner WIP Adversarial Review

- date: `2026-05-16`
- reviewer: `codex`
- reviewed_state: unstaged partner WIP on top of `13f05d3944999cd44039d3d7a4e3b47e2265ceef`
- reviewed_diff_sha256: `40fa52252997c2ad9a6d2d1e07f4572bb2d3312777735a0a947388adf4678ed1`
- scope: `experiments/train_substrate_nscs06_v8_path_b_wavelet.py`, `scripts/remote_lane_substrate_nscs06_v8_path_b_wavelet.sh`, `src/tac/substrates/nscs06_v8_path_b_wavelet/tests/test_v8_substrate.py`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Reviewed Claim

The partner WIP fixes an auth-eval custody bug where Modal CPU advisory output
could be looked for or logged as if it were a contest-CUDA result. The patch
makes the trainer derive `contest_auth_eval_{device}.json` from the actual
auth-eval device and makes the remote completion marker read score axis and
claim validity from the auth-eval JSON before printing a contest-axis marker.

## Findings

No blocking bug found in the current diff.

The trainer path now preserves both CUDA-claim and non-CUDA advisory results:

- `return_non_cuda_result=True` is passed to the canonical auth-eval helper.
- `auth_eval_score_axis`, `auth_eval_device`, `auth_eval_evidence_grade`, and
  `auth_eval_score_claim_valid` are persisted in the stats payload.
- `score_claim` is true only when the helper returns `auth_eval_score_claim`
  and `auth_eval_score_claim_valid`.

The remote script now derives completion labels from raw auth-eval JSON fields:

- `score_axis`
- `score_claim`
- `score_claim_valid`
- `promotion_eligible`
- `device`

This matches `experiments/contest_auth_eval.py`, which emits diagnostic
non-CUDA semantics with `score_claim_valid=false` for advisory CPU/MPS paths
and valid `[contest-CUDA]` only for completed CUDA exact eval.

## Residual Risk

The regression tests added by the partner WIP are primarily source-string
checks. They are useful as static guards, but they do not execute a fixture
through the shell completion-summary parser. A future refactor could preserve
the watched strings while still breaking the JSON-to-marker behavior.

Recommended follow-up if this WIP is expanded:

1. Factor the remote completion-summary Python snippet into a tiny script or
   shell-callable helper.
2. Add fixture tests for:
   - diagnostic CPU JSON -> `[training-artifact-no-score-claim]`
   - contest CPU JSON -> `[contest-CPU]`
   - contest CUDA JSON -> `[contest-CUDA]`
   - malformed/missing JSON -> fail-closed or explicit missing-artifact log

Do not block the current recovery fix on that follow-up; the existing bug class
is materially reduced and the current shell syntax is valid.

## Verification

- `.venv/bin/python -m pytest src/tac/substrates/nscs06_v8_path_b_wavelet/tests/test_v8_substrate.py -q` -> `23 passed`
- `bash -n scripts/remote_lane_substrate_nscs06_v8_path_b_wavelet.sh` -> clean
- `.venv/bin/python -m py_compile experiments/train_substrate_nscs06_v8_path_b_wavelet.py` -> clean
- `.venv/bin/python -m ruff check experiments/train_substrate_nscs06_v8_path_b_wavelet.py src/tac/substrates/nscs06_v8_path_b_wavelet/tests/test_v8_substrate.py` -> clean

Note: `ruff` was not run against the shell script because it parses `.sh` as
Python; `bash -n` is the relevant syntax check for that file.
