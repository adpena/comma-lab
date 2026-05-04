---
name: Azure dispatch wired as 4th GPU platform — $200 free credits ready to deploy
description: 2026-04-30 — wired Azure VM-spot dispatch alongside Vast.ai + Modal + Lightning. New module src/tac/deploy/azure/azure_dispatch.py + scripts/launch_lane_azure.py + 25 passing tests + cost_projection.md ledger. User must run `az login` once before non-dry-run dispatches. NO VMs spawned in this wiring session per "infrastructure first, dispatches when needed" mandate.
type: project
originSessionId: f0d211b9-718f-4fc9-a752-00eac703aca2
---

## Why

User has $200 free Azure credits unused (per CLAUDE.md "GPU budget"). Per the
2026-04-30 mandate ("ALL-IN — push 6-month plan aggressively"), every available
GPU platform should be wired so we can scale to 10+ parallel dispatches across
Vast.ai + Modal + Lightning + Azure.

## What landed

| Path | Purpose |
|------|---------|
| `src/tac/deploy/azure/__init__.py` | Module docstring describing the VM-spot pattern |
| `src/tac/deploy/azure/azure_dispatch.py` | provision_spot_vm / ssh_in / run_lane / harvest / deprovision + AzureVMSpec/Handle dataclasses + pricing table + active-VM tracker |
| `scripts/launch_lane_azure.py` | Retry wrapper analog to `launch_lane_with_retry.py`. **Defaults to dry-run** — passes `--no-dry-run` to actually spawn |
| `src/tac/tests/test_azure_dispatch.py` | 25 tests covering pricing sanity, name sanitization, login pre-flight, dry-run no-side-effects, SSH command construction, active-VM tracker round-trip, cost estimation, $200 budget cap, Pattern A nohup wrapper |
| `.omx/state/cost_projection.md` | NEW operator-facing cost ledger; documents Azure pricing table + $200 free-credit allocation plan + cross-platform reference |

## Architecture choices

- **VM-spot pattern, not Azure ML SDK.** The heavyweight `azure-ai-ml` SDK is
  overkill for single-instance lane scripts. Mirroring the Vast.ai pattern
  (provision → SSH → run lane Pattern A → harvest → deprovision) gives us
  identical surface area to debug across platforms.
- **Spot pricing default** (60-80% off on-demand). Default eviction policy:
  `Deallocate` so spot interrupt doesn't destroy the OS disk.
- **Active VM tracker** at `.omx/state/azure_active_vms.json` — analog of
  `.omx/state/vastai_active_instances.json`. Cleanup tooling can detect
  orphans by reading this JSON.
- **Pattern A nohup detach** in `run_lane()` — verified by test
  `test_run_lane_emits_pattern_a_detach_wrapper` that the remote command
  contains `nohup`, `disown`, `/dev/null` redirection, and `tee`-style log.
- **Fail-loud on no `az login`.** `ensure_azure_logged_in()` is called before
  every side-effecting operation (NOT in dry-run, so dry-run works on
  unauthenticated machines).

## Pricing reference (Azure US East 2026-04-30)

| SKU | GPU | On-demand | Spot |
|-----|-----|-----------|------|
| Standard_NC6s_v3 | V100 16GB | $3.06/hr | ~$0.50/hr |
| Standard_NC24ads_A100_v4 | A100 80GB | $3.67/hr | ~$1.10/hr |
| Standard_NC40ads_H100_v5 | H100 80GB | $6.98/hr | ~$2.30/hr |

Spot is dynamic — operators MUST verify with `az vm list-skus` or the Azure
pricing API before any large dispatch.

## Tests passing

```
PYTHONPATH=src:upstream:. .venv/bin/python -m pytest \
    src/tac/tests/test_azure_dispatch.py -x -v
# 25 passed in 0.08s
```

Coverage:
1. Pricing table sanity (no zero/negative rates, spot < on-demand)
2. AzureVMSpec name sanitization (Azure 1-64 alnum+hyphen rule)
3. ensure_azure_logged_in raises AzureCLIMissing / AzureNotLoggedIn correctly
4. provision_spot_vm dry-run does not call `az`
5. SSH command construction includes StrictHostKeyChecking=no
6. Active VM tracker round-trip (register + unregister via JSON file)
7. estimate_cost rounds correctly for V100/A100/H100
8. remaining_budget_usd respects the $200 cap (floors at 0 when overspent)
9. run_lane builds Pattern A nohup detach wrapper (verified literal markers)

Smoke-test of launcher in dry-run:
```
PYTHONPATH=src:upstream:. .venv/bin/python scripts/launch_lane_azure.py \
    --lane-script scripts/launch_lane_with_retry.py --label test-lane --max-retries 1
# ✓ DISPATCHED: vm=test-lane-a1 label=test-lane attempts=1 (DRY-RUN)
```

## Required user action before first real dispatch

1. **`az login`** — interactive; opens a browser. Required ONCE per machine.
   Without this, `--no-dry-run` invocations fail with `AzureNotLoggedIn`
   (NOT silent — fail-loud per CLAUDE.md non-negotiable).
2. **Verify** `az account list | head -5` shows an active subscription.
3. (Optional) **Pre-create the resource group** to pin region: `az group
   create --name pact-gpu-rg --location eastus`. The dispatcher does this
   idempotently anyway.
4. (Optional) **Quota request** — by default Azure free credits include 4
   vCPU GPU quota in some regions. Heavy GPUs (A100/H100) may require
   quota request via the Azure portal.

## What's NOT done (deliberately, per "infrastructure first" mandate)

- No VMs spawned in this session (DO NOT spawn — user said low-priority
  vs. Lightning).
- SCP / tarball wiring is stubbed in `launch_lane_azure.py` — when the
  first real Azure dispatch happens, port the Vast.ai phase2-scp + extract
  flow to Azure (template exists in `scripts/launch_lane_on_vastai.py`).
- No Azure budget tracker yet — analog of
  `src/tac/deploy/vastai/budget.py`. Add when actual spend starts.
- No quota probe at provision time — Azure raises clear errors when quota
  is exceeded; we surface them via `AzureCommandError`.

## Cross-refs

- CLAUDE.md "GPU budget and compute resources" — pricing intro + $200 cap
- `feedback_full_six_month_plan_aggressive_no_shortcuts_20260430.md` — mandate
- `experiments/modal_train_lane.py` — Modal dispatch reference
- `scripts/launch_lane_with_retry.py` — Vast.ai dispatch reference
- `.omx/state/cost_projection.md` — operator-facing planning ledger
