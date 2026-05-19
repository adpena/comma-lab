# Codex Findings: Auth-Eval Roundtrip Pair-Group Contract

Task:
`codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`.

Context:
the PR101 OP-7 score-response matrix had reached the exact-eval launch step.
The first launch attempt failed before provider spawn because the generated
direct Modal wrapper commands did not include the newer paired-by-default
`--pair-group-id` metadata.

Failure observed:

- `experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py`
  fail closed unless each direct wrapper run has either a shared
  `--pair-group-id` or an explicit `--single-axis-waiver-reason`.
- The generated contest CPU/CUDA rows from
  `tac.auth_eval_roundtrip_matrix` omitted both fields.
- The local Modal entrypoints created app objects, then exited with:
  `FATAL: Modal auth eval is paired-by-default...`
- No score claim, recovered eval JSON, or promotion authority was produced.

Fix:

- `src/tac/auth_eval_roundtrip_matrix.py`
  - contest CPU and contest CUDA wrapper commands now carry the same
    deterministic pair group per archive/runtime matrix side;
  - diagnostic Modal rows now carry
    `--single-axis-waiver-reason diagnostic_non_promotional_roundtrip_axis`;
  - the matrix candidate metadata exposes `modal_pair_group_id`.
- Regression coverage:
  - `src/tac/tests/test_auth_eval_roundtrip_matrix.py` verifies contest
    CPU/CUDA pair-group equality and diagnostic waiver semantics;
  - `src/tac/tests/test_master_gradient_pr101_score_response_matrix.py`
    verifies the PR101 baseline and candidate command pairs carry shared
    CPU/CUDA pair groups.

Regenerated PR101 OP-7 matrix:

- JSON:
  `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.json`
- JSON SHA-256:
  `36a28e32f70d5dec053287f397829cb6ecd0f1a663718cb27a793262ee5561a2`
- Markdown:
  `experiments/results/pr101_pose_axis_score_response_matrix_20260519T092500Z_codex/score_response_matrix.md`
- Markdown SHA-256:
  `654fac6497fcaec2b9b9701a2684bb65d7123b8606741059441e28d1cdeae90d`

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- This is a command-contract fix only. The next score-bearing step remains
  paired contest CPU/CUDA exact eval plus score-response probe review.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_auth_eval_roundtrip_matrix.py \
  src/tac/tests/test_master_gradient_pr101_score_response_matrix.py
```

Result: `12 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/auth_eval_roundtrip_matrix.py \
  src/tac/master_gradient_pr101_score_response_matrix.py \
  src/tac/tests/test_auth_eval_roundtrip_matrix.py \
  src/tac/tests/test_master_gradient_pr101_score_response_matrix.py
```

Result: passed.

