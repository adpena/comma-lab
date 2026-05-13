# PR101 LC V2 Smoke RC13 Harvest - 2026-05-13

## Scope

Harvested Modal training call:

- lane_id: `lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513`
- instance/job: `substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch_20260513T215933Z__smoke__100ep`
- call_id: `fc-01KRHNMT4SEB794HFPH5GNTHFP`
- axis: no score axis; training smoke failed before trainer execution
- score_claim: `false`
- promotion_eligible: `false`

## Result

The Modal smoke returned `rc=13` with one harvested artifact and no numeric
elapsed-time cost anchor. A terminal claim was appended by the harvester as:

`failed_modal_training_rc_13`

No archive, no auth-eval output, and no score claim were produced.

## Classification

`failed_worker_sentinel_not_mounted`

The harvested `modal_worker_head_ledger.json` shows:

```text
.omx/operator_authorize_recipes/substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch.yaml:
  local=76d380b3c938
  worker=MISSING_WORKER
```

The worker-side Catalog #166 sentinel gate behaved correctly: it refused
training before GPU work because one requested sentinel was not present in the
Modal mounted source tree.

## Root Cause

The failed dispatch included an operator-side recipe file under
`.omx/operator_authorize_recipes/` in `--sentinel-files`. That path is not part
of `mount_manifest.STRUCTURAL_MINIMUM_DIRS`, and `modal_train_lane.py` copies
only mounted runtime source into `/tmp/pact`. Therefore the worker can never
hash that recipe path, and the gate returns `MISSING_WORKER`.

This is not a model result and not evidence against the PR95 enhanced
curriculum substrate.

## Current Guard State

Current `tools/operator_authorize.py` carries Catalog #201: worker-side sentinel
paths are restricted to the canonical Modal mount set (`src/`, `scripts/`,
`upstream/`, `submissions/`, `experiments/`, `tools/`, `pyproject.toml`).
Operator-side recipe drift must be checked operator-side, not by the worker
sentinel hash gate.

## Reactivation Criteria

Relaunch only after:

1. `tools/operator_authorize.py` emits no `.omx/` paths in `--sentinel-files`.
2. `tools/run_modal_smoke_before_full.py` plan/smoke path shows the same lane
   id and sentinel set before dispatch.
3. The dispatch claim is clean and no active duplicate lane exists.

Any relaunch remains non-promotable until a byte-closed archive plus exact
auth-eval artifacts exist on the appropriate contest axis.
