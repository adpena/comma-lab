# V9 CGauge CUDA Modal fire-readiness closeout (2026-07-12)

Status: **SOURCE/LOCAL CUSTODY CLOSED; FIRE REMAINS CONDITIONAL ON FOUR LIVE OPERATOR CHECKS**

Authority: Task438/Task381 pre-fire closeout. No `modal run`, `modal volume put`,
GPU dispatch, paid function, or provider mutation was performed. This receipt is
training-advisory and makes no contest-score claim.

Source commits: `eb8b06c9fb` (`fix(modal): seal V9 fire-readiness custody`) and
`3eae6e3bb5` (`docs(modal): record conditional fire closeout`)

## Fire-readiness checklist

### 1. GT cache staging and remote custody

**CLOSED — local source bytes and staging contract.** The unchanged plan-first
command emitted `witness_asset_stage_readiness.v1` with `status=passed`,
`provider_contacted=false`, and `staging_executed=false`. It rehashed and bound:

- local: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
- bytes: `5,078,017,610`
- SHA-256: `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- Volume: `comma-train-lane-results`
- Volume-relative destination:
  `assets/v9_cgauge/gt_cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6.npz`
- mounted path:
  `/modal_results/assets/v9_cgauge/gt_cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6.npz`

The launcher now refuses any non-canonical Task438 digest at its CLI, binds the
exact byte count into `witness_cloud_plan.v7`, and proves that the advisory
`asset_stage_argv` destination and mounted runtime path are the same object.
Execution still never runs `asset_stage_argv`; `Volume.from_name` remains
`create_if_missing=False`; the execution path performs a read-only exact-path and
byte-count lookup, then the bounded same-image CPU preflight rehashes the mounted
bytes before any H100 allocation.

**OPERATOR-MUST-CONFIRM — live remote presence.** The read-only CLI lookup did
not return within 30 seconds in this restricted session, so remote presence is
not claimed. Run exactly:

```bash
.venv/bin/modal volume ls --json comma-train-lane-results assets/v9_cgauge/gt_cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6.npz
```

Confirm exactly one regular-file row at that path with `5,078,017,610` bytes.
If absent, remain DEFER; staging is a separate operator-authorized transaction
and must not be inserted into the fire command.

### 2. Clean main

**OPERATOR-MUST-CONFIRM.** This closeout was selectively serialized without
absorbing unrelated sister work. The launcher enforces the same condition again
immediately before provider read and immediately before dispatch. After all
sister work lands, run exactly:

```bash
test "$(git branch --show-current)" = main && test -z "$(git status --porcelain=v1 --untracked-files=all)" && test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
```

Any nonzero result means DEFER.

### 3. Deployment version, wheel, flags, and required assets

**CLOSED.** `pyproject.toml` and `tac.__version__` are `0.2.0rc2`, replacing the
stale rc1 identity after post-rc1 `src/tac` changes. A wheel is rebuilt from a
committed-HEAD archive, not the dirty shared working tree. Final wheel receipt:

```text
path=dist/tac-0.2.0rc2-py3-none-any.whl
bytes=27,631,647
sha256=af8710c95dd9072e609f699faa48db2ae99eef33556ec525405f164fb949786f
metadata_version=0.2.0rc2
witness_launcher_in_wheel=true
build_source=git archive 3eae6e3bb5 (not the dirty shared worktree)
```

`REQUIRED_DATASET_ASSETS` names `tac-0.2.0rc2-py3-none-any.whl` with the measured
expected-size anchor `27,631,663`; the final committed-archive rebuild differs by
16 bytes and passes the consumer's 1% tolerance. The wheel contains
`tac/deploy/witness_cloud_launcher.py` and its METADATA reports
`Version: 0.2.0rc2`.

`deploy_config.BASE_FLAGS` was independently compared to the real Click parser
in `experiments/train_renderer_fridrich.py`: **34/34 flags exist; missing=[]**.
The V9 Modal route does not consume that asymmetric-renderer flag list: it uses
the typed V9 DSL plus the reviewed remote driver and mounts committed source via
the Modal image. The parity check therefore closes the general deployment
checklist without inventing or injecting V9 flags.

### 4. Authenticated workspace, budget/IAM, and image cache

**OPERATOR-MUST-CONFIRM.** Read-only local CLI evidence resolved the active
profile as `adpena`; it did not establish owner role, remaining hard budget, or
the exact image's cache/build receipt. Run exactly:

```bash
.venv/bin/modal profile current && .venv/bin/modal environment billing report --for "this month" --show-resources
```

In Modal Usage & Billing, confirm the selected workspace and environment have
at least `$3.30` remaining under their effective hard budgets and that the
identity is least-privilege for this lane. Also confirm a build/cache receipt for
the exact image definition and that neither `MODAL_FORCE_BUILD` nor
`MODAL_IGNORE_CACHE` is set. Source execution refuses those two bypass variables,
but source cannot prove a provider-side cache hit.

### 5. Volume retention and harvest custody

**OPERATOR-MUST-CONFIRM.** Modal Volumes are persistent until deletion; source
cannot choose an account retention policy. Inspect the live inventory with:

```bash
.venv/bin/modal volume list --json && .venv/bin/modal volume ls --json comma-train-lane-results
```

Before GO, record the policy: preserve all stage/epoch checkpoints through exact
call-ID harvest and receipt review; then certify or block any cleanup using the
repository artifact-lifecycle contract. Do not delete the 5 GB GT asset or smoke
output merely to make this checklist green. Harvest the exact spawned call ID
within 24 hours using the command printed by `modal_train_lane.py`.

## Validation seal

- exact local GT custody: SHA-256 and byte count matched
- unchanged plan-first compile: `execution_allowed=true`, exact `H100!`, typed
  horizon 3000, `stop_after_epochs=3`, child timeout 1500 s, invocation timeout
  1800 s, planned total `$3.296256`, internal refusal maximum `$5.00`
- focused deploy/launcher suite: **242 passed**
- Ruff: clean on all changed Python surfaces
- Ty: no errors on changed production surfaces; only the repository's existing
  `possibly-unbound` configuration warning
- `bash -n scripts/remote_v9_cgauge_cuda.sh`: pass
- `py_compile`: pass
- `git diff --check`: pass
- review tracker: changed Python entities reviewed; standard-policy second pass
  completed for the wheel/deploy builder

## Exact conditional fire command — unchanged from money-safety review

Do not run until every `OPERATOR-MUST-CONFIRM` item above passes. The command
contains no staging operation.

```bash
GT_SHA=cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6
LABEL=v9-cgauge-cuda-438-smoke-20260712-r2
PLAN_FILE="$(mktemp)"

PYTHONPATH=src .venv/bin/python tools/launch_witness_cloud.py \
  --provider modal \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --gt-cache-sha256 "$GT_SHA" \
  --label "$LABEL" \
  --gpu 'H100!' \
  --epochs 3000 \
  --num-pairs 600 \
  --output "$PLAN_FILE"

PLAN_SHA="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_sha256"])' "$PLAN_FILE")"
rm -f "$PLAN_FILE"

PYTHONPATH=src .venv/bin/python tools/launch_witness_cloud.py \
  --provider modal \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --gt-cache-sha256 "$GT_SHA" \
  --label "$LABEL" \
  --gpu 'H100!' \
  --epochs 3000 \
  --num-pairs 600 \
  --expected-plan-sha256 "$PLAN_SHA" \
  --execute \
  --operator-go-token GO-CLOUD-381
```

Post-fix planned ceiling remains **`$3.296256` (restated as `$3.30`)**, with the
repository's internal **`$5.00` refusal maximum**. The `$0.50` image/staging
allowance remains a conservative source assumption, not a provider-enforced
all-in cap.
