# Preflight Dirty-State Drift Review - Worker P1 - 2026-05-08

## Scope

Adversarial review of strict preflight and all-lanes dirty-state drift after the
half-frame scanner fix. Focus areas:

- missing helper files referenced by `src/tac/preflight.py` and
  `tools/all_lanes_preflight.py`
- stale deleted tool/test paths
- strict checks failing on orphaned/deleted files
- false positives that would block exact dispatch

Branch check: `git rev-parse --abbrev-ref HEAD` returned `main`.

## Commands And Findings

### Repository state

Command:

```bash
git status --short --branch
```

Finding:

- Worktree is heavily dirty and ahead of origin.
- Several source/test/tool paths are staged as deleted while the same paths
  exist as untracked working-tree replacements.
- This is partner work and was not reverted.

Command:

```bash
git diff --cached --name-status
```

Relevant staged deletions:

- `tools/check_evidence_implementation_matches_model_spec.py`
- `tools/build_admm_x_lossy_coarsening_path_b_step6.py`
- `tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py`
- `tools/pr101_omega_opt_joint_admm_allocation_empirical.py`
- `src/tac/tests/test_preflight_implementation_model_match.py`
- multiple `src/tac/tests/test_pr101_*.py` files

Command:

```bash
for p in tools/pr101_omega_opt_joint_admm_allocation_empirical.py \
  tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py \
  tools/build_admm_x_lossy_coarsening_path_b_step6.py \
  experiments/results/admm_x_lossy_coarsening_path_b_step6_20260508T060435Z/submission_dir/inflate.py \
  tools/check_evidence_implementation_matches_model_spec.py \
  reports/hstack_vstack_multipass_plan_20260507.json; do
  if [ -e "$p" ]; then printf 'EXISTS %s\n' "$p"; else printf 'MISSING %s\n' "$p"; fi
done
```

Finding:

- Every checked helper/candidate path exists in the working tree.
- The drift is index-vs-working-tree shadowing, not a current filesystem
  missing-helper failure.

Command:

```bash
git diff --cached --name-only --diff-filter=D > /tmp/pact_staged_deletions.txt
rg -nF -f /tmp/pact_staged_deletions.txt src/tac/preflight.py tools/all_lanes_preflight.py src/tac/tests/test_preflight_meta_bugs.py || true
```

Matches:

- `src/tac/preflight.py:22376` references
  `tools/pr101_omega_opt_joint_admm_allocation_empirical.py`
- `src/tac/preflight.py:22377` references
  `tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py`
- `src/tac/preflight.py:22605` references
  `tools/build_admm_x_lossy_coarsening_path_b_step6.py`

All three currently exist in the working tree.

### Half-frame scanner

Command:

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_meta_bugs.py::TestHalfframeArchiveTrainedProfile -q
```

Result:

```text
4 passed in 0.48s
```

Finding:

- The half-frame scanner constant/profile handling is currently green.
- `q_faithful_dilated_88k` resolves to `mask_half_sim_prob=1.0` and
  `use_zoom_flow=True`.

### False positive found and fixed

Command before fix:

```bash
.venv/bin/python - <<'PY'
from tac.preflight import check_evidence_row_has_falsification_scope_when_negative
v = check_evidence_row_has_falsification_scope_when_negative(strict=False, verbose=True)
print('violations', len(v))
for item in v:
    print(item)
PY
```

Failure:

- 3 violations in `reports/cathedral_autopilot_evidence.jsonl:36-38`
- Each row was a positive/proxy `[CPU-build]` row with
  `family_falsified=false`, not a negative/retired result.
- The check name and doc intent are "when negative", but the implementation
  required `falsification_scope` for every row where `family_falsified` was
  `false`.

Fix:

- Updated `check_evidence_row_has_falsification_scope_when_negative()` so
  positive/proxy CPU-build rows with `family_falsified=false` do not block
  strict preflight.
- The check still fails negative/retired evidence rows with
  `family_falsified=false` and no `falsification_scope`.
- The check still fails any `family_falsified=true` row.

Post-fix command:

```bash
.venv/bin/python - <<'PY'
from tac.preflight import check_evidence_row_has_falsification_scope_when_negative
v = check_evidence_row_has_falsification_scope_when_negative(strict=False, verbose=True)
print('violations', len(v))
PY
```

Result:

```text
[harden-falsification-scope] OK: 40 evidence row(s) across 1 file(s) all consistent
violations 0
```

### Focused tests

Command:

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_meta_bugs.py::TestEvidenceFalsificationScopeGuard -q
```

Result:

```text
4 passed in 0.58s
```

Command:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py::TestHalfframeArchiveTrainedProfile \
  src/tac/tests/test_preflight_meta_bugs.py::TestEvidenceFalsificationScopeGuard -q
```

Result:

```text
8 passed in 0.49s
```

Command:

```bash
.venv/bin/python - <<'PY'
from tac.preflight import check_evidence_row_has_falsification_scope_when_negative
print(check_evidence_row_has_falsification_scope_when_negative(strict=True, verbose=False))
PY
```

Result:

```text
[]
```

Command:

```bash
.venv/bin/python - <<'PY'
from tac.preflight import (
    check_admm_lagrangian_bisection_convergent,
    check_codec_pipeline_op_order_deterministic,
    check_per_tensor_K_side_info_matches_decoder_expectation,
    check_evidence_row_has_falsification_scope_when_negative,
    check_137531_candidate_decoder_path_wired,
)
for check in [
    check_admm_lagrangian_bisection_convergent,
    check_codec_pipeline_op_order_deterministic,
    check_per_tensor_K_side_info_matches_decoder_expectation,
    check_evidence_row_has_falsification_scope_when_negative,
    check_137531_candidate_decoder_path_wired,
]:
    result = check(strict=True, verbose=False)
    print(f'{check.__name__}: {len(result)} violation(s)')
PY
```

Result:

```text
check_admm_lagrangian_bisection_convergent: 0 violation(s)
check_codec_pipeline_op_order_deterministic: 0 violation(s)
check_per_tensor_K_side_info_matches_decoder_expectation: 0 violation(s)
check_evidence_row_has_falsification_scope_when_negative: 0 violation(s)
check_137531_candidate_decoder_path_wired: 0 violation(s)
```

Command:

```bash
git diff --check src/tac/preflight.py src/tac/tests/test_preflight_meta_bugs.py
```

Result:

- No whitespace errors.

### Full preflight status

Incorrect command attempted first:

```bash
.venv/bin/python -m tac.preflight --profile q_faithful_dilated_88k --skip-artifact-checks
```

Result:

```text
preflight.py: error: unrecognized arguments: --skip-artifact-checks
```

Correction:

```bash
.venv/bin/python -m tac.preflight --help
```

Confirmed CLI options:

- `--renderer`
- `--masks`
- `--poses`
- `--archive`
- `--no-codebase`
- `--profile`
- `--tto-frames`
- `--gt-poses`

Command:

```bash
.venv/bin/python -m tac.preflight --profile q_faithful_dilated_88k
```

Result:

- Failed before reaching the new HARDEN checks.
- Failure class: unrelated score-tag drift.

Failure:

```text
MetaBugViolation: SCORE LINES WITHOUT LANE TAG:
  - .ralph/run_log.md:559
  - .omx/research/findings.md:52
```

### All-lanes preflight status

Command:

```bash
.venv/bin/python tools/all_lanes_preflight.py --jobs 1
```

Result:

- Exit code 3.
- Gate #10 failed: untracked source inventory, 85 undispositioned source-like
  artifacts.
- Gate #11 failed: orphan recovery canonicalization, 28 source-like staged
  deletions outside the recovery tree.
- Gate #15 failed: release index/worktree split, shadowed staged rollback
  entries where local checks read the working tree while a commit would
  publish the staged rollback.
- Gates #20 and #21 passed.
- Lane dry-runs passed or self-protected.

Conclusion:

- Current all-lanes failure is real dirty-state custody drift, not a missing
  helper referenced by `all_lanes_preflight.py`.
- Exact dispatch should not be blocked by the fixed falsification-scope false
  positive.
- Exact dispatch remains blocked by broader dirty-state custody gates until
  partner staged deletions/untracked replacements are dispositioned.

## Changed Files

- `src/tac/preflight.py`
  - narrowed `check_evidence_row_has_falsification_scope_when_negative()` to
    negative/retired evidence rows.
- `src/tac/tests/test_preflight_meta_bugs.py`
  - added focused regression tests for positive CPU-build rows, exact-negative
    rows with and without scope, and explicit family-falsified rows.
- `.omx/research/preflight_dirty_state_drift_review_20260508_worker_p1.md`
  - this ledger.
