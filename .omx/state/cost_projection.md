# Cost projection — multi-platform GPU dispatch ledger

Last updated: 2026-04-30

This is the operator-facing source of truth for **estimated** cost per
platform. Actual spend is tracked per-platform in their respective
trackers (Vast.ai, Modal, Lightning, Azure budget files). This file is
for **planning** new dispatches against remaining credits.

## Platform pricing reference

| Platform     | GPU                           | $/hr (on-demand) | $/hr (spot/best) | Notes |
|--------------|-------------------------------|------------------|------------------|-------|
| Vast.ai      | RTX 4090 (24GB)               | n/a              | $0.25–$0.40      | Primary; ~85% NVDEC bad-host rate; use launch_lane_with_retry.py |
| Vast.ai      | H100 (80GB)                   | n/a              | ~$2.00           | Heavy lanes only |
| Modal        | T4 (16GB)                     | $0.59            | n/a              | Reliable; mirrors training surface |
| Modal        | A10G (24GB shared)            | $1.00            | n/a              | OOM if lane needs >22GB |
| Lightning.ai | T4 / A100                     | varies           | varies           | Free tier + paid; managed |
| **Azure**    | **Standard_NC6s_v3** (V100 16GB) | **$3.06**     | **~$0.50**       | **NEW — wired 2026-04-30** |
| **Azure**    | **Standard_NC24ads_A100_v4** (A100 80GB) | **$3.67** | **~$1.10** | **NEW — wired 2026-04-30** |
| **Azure**    | **Standard_NC40ads_H100_v5** (H100 80GB) | **$6.98** | **~$2.30** | **NEW — wired 2026-04-30** |
| AWS          | g4dn.xlarge (T4)              | $0.526           | $0.22            | Free credits available |

## Credit balances (planned vs remaining)

| Platform     | Hard cap   | Spent so far | Remaining | Source of truth |
|--------------|------------|--------------|-----------|-----------------|
| Vast.ai      | $25.00     | tracked elsewhere | varies | `vastai` CLI + budget tracker |
| Modal        | $30.00/mo  | tracked elsewhere | varies | Modal dashboard |
| AWS          | $100.00    | $0 (unused)  | $100.00   | AWS console |
| **Azure**    | **$200.00**| **$0 (unused)** | **$200.00** | **Azure portal post-`az login`** |

## Azure dispatch wiring (2026-04-30)

- Module: `src/tac/deploy/azure/azure_dispatch.py`
- CLI:    `scripts/launch_lane_azure.py` (default dry-run; pass `--no-dry-run` to spawn)
- Tests:  `src/tac/tests/test_azure_dispatch.py` (25 tests, all green)
- Active VM tracker: `.omx/state/azure_active_vms.json` (analog of
  `.omx/state/vastai_active_instances.json`)

### How to allocate the $200 free Azure credits

A reasonable burn plan:
- **$100 reserve** for a deadline-week burst on H100 (~43 hr at $2.30/hr spot)
- **$60 mid-tier** for A100 lanes (~55 hr at $1.10/hr spot)
- **$40 smoke / cheap experiments** on V100 (~80 hr at $0.50/hr spot)

Before any non-dry-run dispatch:
1. `az login` (interactive — required ONCE per machine)
2. Verify `az account list` shows an active subscription
3. Smoke-test: `python scripts/launch_lane_azure.py --lane-script <X> --label smoke` (defaults to dry-run)
4. Real dispatch: append `--no-dry-run`

### Cost projection examples (Azure spot)

| Lane scope                    | GPU type                  | Hours | Spot cost |
|-------------------------------|---------------------------|-------|-----------|
| Smoke / probe                 | NC6s_v3 (V100)            | 0.5   | $0.25     |
| Light lane (e.g. Lane G clone) | NC6s_v3 (V100)            | 4     | $2.00     |
| Heavy lane (e.g. Lane Ω-W)    | NC24ads_A100_v4 (A100)    | 6     | $6.60     |
| Full IMP cycle                | NC24ads_A100_v4 (A100)    | 80    | $88.00    |
| Deadline-week H100 burst      | NC40ads_H100_v5 (H100)    | 24    | $55.20    |

## Cross-refs
- CLAUDE.md "GPU budget and compute resources" section (canonical pricing intro)
- Memory: `project_azure_dispatch_wiring_20260430.md` (this wiring's design notes)
- `experiments/modal_train_lane.py` (Modal dispatch reference)
- `scripts/launch_lane_with_retry.py` (Vast.ai dispatch reference)
