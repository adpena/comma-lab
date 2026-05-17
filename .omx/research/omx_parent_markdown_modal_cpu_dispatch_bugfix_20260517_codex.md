# OMX Parent Markdown Modal CPU Dispatch Bugfix - 2026-05-17

## Why This Exists

The operator warned that relevant OMX/Claude signal might live outside
`.omx/research`. A fresh no-ignore scan of `.omx/**/*.md` found an active
dispatch failure in `.omx/state/active_lane_dispatch_claims.md` that would have
been missed by research-only reading.

## Scan Surface

Fresh scan command class:

```bash
rg --files --hidden --no-ignore .omx -g '*.md'
rg -n -i --hidden --no-ignore \
  'l5|tt5l|time[-_ ]?trav|staircase|cargo[-_ ]?cult|local minima|rule #?6|fec6|master gradient|score[-_ ]?lower|frontier' \
  .omx --glob '*.md' --glob '!.omx/research/**'
```

Observed Markdown inventory at `main` commit
`f4a8de3437c7ab7268c4d5acbeabcf4e9a408e4d` before this fix:

| Bucket | Markdown files |
|---|---:|
| total `.omx/**/*.md` | 2404 |
| non-research `.omx` Markdown | 636 |
| `.omx/research` | 1768 |
| `.omx/auto_memory_snapshot_20260504T230223Z` | 562 |
| `.omx/state` | 22 |
| `.omx/context` | 28 |
| `.omx/tmp` | 16 |
| `.omx/plans` | 4 |
| root `.omx/*.md` | 2 |

Root `.omx/*.md` files remain:

- `.omx/notepad.md`
- `.omx/release_manifest_v0.2.0-rc1.md`

Neither root file supersedes current L5-v2 / Rule #6 routing. The active
current routing remains concentrated in `.omx/state/current_focus.md`,
`.omx/state/next_experiments.md`, and
`.omx/state/active_lane_dispatch_claims.md`.

## Actionable Finding

The parent-scope scan found the newest active-claims rows:

```text
2026-05-17T17:46:50Z active_dispatch
lane_op_routable_1_master_gradient_extractor_20260517
master_gradient_fec6_modal_cpu_dispatch_20260517T174650Z

2026-05-17T17:47:56Z failed_dispatch_rc_2
operator-authorize native dispatch returned non-zero rc=2
```

The recipe correctly declared a Modal CPU tool dispatch:

- `dispatch_kind: tool`
- `gpu: "CPU"`
- `min_smoke_gpu: "CPU"`
- `modal.lane_script: scripts/operator_authorize_master_gradient_fec6_modal_cpu.sh`
- `modal.cost_band_trainer: tools/extract_master_gradient.py`

The dispatch protocol had already been patched to allow CPU for tool
dispatches, but `experiments/modal_train_lane.py` still routed only:

- `T4`
- `A10G`
- `A100`
- `H100`

Therefore `operator_authorize.py --recipe master_gradient_fec6_modal_cpu_dispatch`
could pass the protocol, open a claim, call `modal_train_lane.py --gpu CPU`,
then fail with rc=2 at the Modal entrypoint before useful work began.

## Fix

Patched `experiments/modal_train_lane.py` to add:

- `run_lane_training_cpu` Modal function with no GPU allocation.
- `gpu in ("CPU", "cpu", "Cpu")` routing to that function.
- Updated unsupported-GPU message to include CPU.
- Updated the entrypoint docstring to list CPU as a valid dispatch target.

Patched `src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py` to
pin:

- five Modal wrappers, not four;
- CPU wrapper payload threading;
- main-entrypoint CPU routing;
- the user-facing error message includes CPU.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py \
  src/tac/tests/test_dispatch_protocol_tool_scope.py -q
```

Result: `51 passed`.

```bash
.venv/bin/python -m ruff check \
  experiments/modal_train_lane.py \
  src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py
```

Result: `All checks passed!`.

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe master_gradient_fec6_modal_cpu_dispatch --dry-run
```

Result: dry-run renders the Modal CPU recipe successfully. No claim and no
dispatch were opened by the dry-run.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

This landing fixes a dispatch actuator bug. It does not produce a score,
archive, gradient sidecar, or promotion claim. A future real master-gradient
dispatch still requires a fresh active lane claim and normal operator/provider
custody.
