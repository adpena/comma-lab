# Z3 Modal Mount Scope Abort And Fix (2026-05-15)

## Summary

Attempted a fail-fast Z3 Ballé hyperprior smoke through the canonical
`tools/run_modal_smoke_before_full.py` path:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch \
  --smoke-epochs 20 \
  --smoke-gpu T4 \
  --smoke-timeout-hours 0.25 \
  --smoke-only
```

The command was aborted before provider acceptance. The local Modal CLI stayed
CPU-bound and `lsof` showed it had opened a generated raw artifact:

```text
experiments/results/m5max_sweep_constrained_coord_search_20260509T142645Z/_calibration/_calibration_anchor/work/inflated/0.raw
size=3,662,409,600 bytes
```

No `modal_call_id.txt` was written for the attempted Z3 smoke.

## Classification

- axis: provider-dispatch-local-pre-spawn
- score_claim: false
- promotion_eligible: false
- failure_class: modal_mount_stability_scope_mismatch
- terminal claim: `stopped_local_modal_mount_scope_abort`

The problem was not Z3 model quality. The canonical mount builder correctly
passes `ignore=["results/**"]` when mounting `experiments/`, but the
mtime-stability fingerprint walked `experiments/` without honoring that same
ignore scope. The guard therefore read large generated artifacts that Modal
would not upload.

## Custody And Reproducibility

| Field | Value |
| --- | --- |
| observed_at_utc | 2026-05-15T03:47Z-2026-05-15T04:02Z |
| base_git_sha | `87872fb0f2428182efd797b0a20d7f9a9673ddf3` |
| worktree_state_at_fix | dirty by this fix only: `src/tac/deploy/modal/mount_manifest.py`, `src/tac/tests/test_check_165_modal_mount_mtime_stability.py`, this ledger |
| modal_version | `1.4.2` |
| abort_evidence_command | `lsof -p <local_modal_cli_pid> | rg 'experiments/results|0.raw|m5max'` |
| abort_evidence | local Modal process opened `experiments/results/m5max_sweep_constrained_coord_search_20260509T142645Z/_calibration/_calibration_anchor/work/inflated/0.raw` (`3,662,409,600` bytes) |
| provider_acceptance | none; no `modal_call_id.txt` produced |
| terminal_claim_status | `stopped_local_modal_mount_scope_abort` |
| score_claim | false |

## Fix

`src/tac/deploy/modal/mount_manifest.py` now carries ignore patterns into the
fingerprint path. File inclusion is delegated to Modal's own
`FilePatternMatcher`, with safe early pruning for descendant-only ignored
trees such as `results/**`, so the guard hashes the same upload-relevant file
set without reading ignored generated outputs.

Regression test:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_check_165_modal_mount_mtime_stability.py \
  src/tac/tests/test_mount_manifest.py -q
```

Result: `65 passed` after adding coverage for nested
`experiments/results/**` churn through `build_training_image`.

Performance sanity:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from time import perf_counter
from tac.deploy.modal.mount_manifest import _hash_mount_set_fingerprint
start = perf_counter()
fp, missing = _hash_mount_set_fingerprint([(Path("experiments"), ("results/**",))])
print({"elapsed_sec": round(perf_counter() - start, 3), "missing": len(missing), "sha16": fp[:16]})
PY
```

Expected shape after the fix: no reads from `experiments/results/**`; local
hashing stays in low single-digit seconds on the current tree.

## Reactivation

Retry the same Z3 smoke only after this fix is committed and the worktree is
clean, because the recipe uses `--require-clean-head`.
