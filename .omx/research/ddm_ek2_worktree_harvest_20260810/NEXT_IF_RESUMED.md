# NEXT IF RESUMED

## Current State

The EK2 harvest is code/test green but not committed. The managed sandbox blocked creation of the assigned branch:

```text
codexwt/ddm_ek2_worktree_harvest
```

The branch failure was:

```text
fatal: cannot lock ref 'refs/heads/codexwt/ddm_ek2_worktree_harvest': Unable to create '.git/refs/heads/codexwt/ddm_ek2_worktree_harvest.lock': Operation not permitted
```

Do not commit this package on the predecessor EK branch. First restore Git ref-write permission or move the exact worktree state into a Git-writable checkout, then create the assigned branch and use the serializer.

## Before Serializer

1. Confirm the two cold-store directories still match `cold_store_manifest.json`.
2. Re-run the focused suite:

```text
PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py \
  src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py \
  src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py
```

3. Re-run direct guards:

```text
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
import tac.preflight as p
root = Path.cwd()
print("check_154", len(p._check_154_manifestless_cleanup_identity(root)))
print("check_344", len(p._check_344_anchor_roundtrip_integrity(root)))
print("check_351", len(p._check_351_canonical_producer_identity(root)))
PY
```

4. Recompute post-edit SHA-256 for every committed file and call:

```text
.venv/bin/python tools/subagent_commit_serializer.py \
  --message "ddm_ek2: harvest Einstein-Kolmogorov worktree residue [no-triality] [p0-ledger-ok]" \
  --no-co-author \
  --expected-content-sha256 <path>:<post-edit-sha256> \
  --files <paths...>
```

No `REVIEW_GATE_OVERRIDE=1` may be used for the Python files.

## Boundaries

EK2 did not run a scorer, exact replay, paid dispatch, CUDA dispatch, or Modal job.

The required frontier line remains:

```text
S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]
```

Contest pointer remains borrowed/external and unmoved by this work.

## Secondary Residue

The charter allowed secondary residue only after the EK worktree was handled. Because branch creation and serializer landing are blocked, no secondary arm was executed. If resumed after landing EK2, use the requested fire order:

```text
v7-waterfill first
v8-margin-gated second
ratecrush third
```

Before any remote/GPU/eval dispatch, claim the lane and obey the governed launcher and resumability rules.
