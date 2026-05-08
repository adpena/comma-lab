# ADMM No-Dead-K Static Preflight - 2026-05-08

Scope: exact-eval/preflight readiness hardening for the selected-K no-dead-K
CPU artifact.

No dispatch was attempted. No score was claimed.

## Helper

Added:

```bash
tools/run_admm_no_dead_k_static_preflight.py
```

The helper is read-only. It verifies the build manifest, archive bytes/SHA,
single-member ZIP safety, generated runtime manifest, and `inflate.sh` shell
syntax. It preserves CPU-build blocker semantics by keeping
`ready_for_exact_eval_dispatch=false`, `score_claim=false`, and
`dispatch_attempted=false`.

## Command

```bash
.venv/bin/python tools/run_admm_no_dead_k_static_preflight.py \
  --build-manifest experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T092353Z/build_manifest.json \
  --json-out reports/raw/selected_ks_no_dead_k_static_preflight_20260508_codex/static_preflight_no_score.json \
  --fail-if-static-closure-broken
```

## Result

- Static archive/runtime closure: passed.
- Archive:
  `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T092353Z/archive.zip`
- Archive bytes: `159576`
- Archive SHA-256:
  `efc87556699abd6520921b0c888a395f2c95de2090e888ed5843ac35fc134e89`
- ZIP member: `x`
- Runtime tree SHA-256:
  `24630173228005ae6ed1aadeac8519d2687b9de90326836303b42475367d1004`
- Static blockers: none.
- Runtime cache check: passed; `submission_dir` contains no `__pycache__` or
  `.pyc` files after the smoke-import cleanup.
- Readiness remains blocked by CPU-build proxy semantics, missing exact CUDA
  auth eval, missing score-promotion custody, the
  `apogee_int6_contest_cuda_anchor_required_first` blocker, incomplete release
  packet staging, diagnostic-source selected-K semantics, and the required
  active dispatch claim before any dispatch.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_run_admm_no_dead_k_static_preflight.py -q
.venv/bin/python -m pytest src/tac/tests/test_build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/tests/test_run_admm_no_dead_k_static_preflight.py -q
.venv/bin/python -m py_compile tools/run_admm_no_dead_k_static_preflight.py src/tac/tests/test_run_admm_no_dead_k_static_preflight.py
.venv/bin/ruff check tools/run_admm_no_dead_k_static_preflight.py src/tac/tests/test_run_admm_no_dead_k_static_preflight.py
```
