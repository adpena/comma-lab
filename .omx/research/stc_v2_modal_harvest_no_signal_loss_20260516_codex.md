# STC v2 Modal Harvest No-Signal-Loss Closure

Date: 2026-05-16

Lane: `lane_stc_clean_source_v2_substrate_build_20260516`

Instance/job: `substrate_stc_v2_modal_t4_dispatch_20260516T213028Z`

Modal call id: `fc-01KRSB76H04HM4958V2HX2JZZ4`

## Classification

The STC v2 Modal T4 smoke dispatch is closed as an infrastructure/config
failure, not a model result and not a method falsification.

- terminal status: `failed_modal_training_rc_25`
- elapsed: `1.6170410570000087` seconds
- estimated cost: `$0.00026501506211944583`
- artifacts harvested: `5`
- score claim: `false`
- promotion eligible: `false`
- rank/kill eligible: `false`

The worker failed before archive construction or auth eval:

```text
FATAL: Lane A anchor archive missing at /tmp/pact/experiments/results/lane_a_landed/archive_lane_a.zip
STC v2 swap-archive requires renderer.bin + optimized_poses.pt
from Lane A; the lane has no fallback path.
```

This means the measured configuration is missing a required upstream anchor
artifact. The STC v2 lane remains reactivatable by supplying the Lane A anchor
archive/runtime inputs under the path expected by the remote worker or by
changing the recipe/remote script to mount the correct byte-closed anchor path.

## No-Signal-Loss Actions

- Ran `tools/backfill_terminal_claim_evidence.py` for the three earlier
  `failed_dispatch_rc_2` STC v2 terminal rows. It appended three no-score
  cathedral evidence rows.
- Ran `tools/harvest_modal_calls.py --from-ledger --execute` and recovered the
  active STC v2 Modal call before cache loss.
- The harvester appended:
  - terminal dispatch claim row for `failed_modal_training_rc_25`
  - cost-band anchor row
  - Modal call-id ledger terminal row
  - cathedral autopilot terminal evidence row
- No archive bytes or score artifacts were produced by this STC v2 call.

## Evidence

Primary harvested artifacts:

- `experiments/results/lane_substrate_stc_v2_modal_t4_dispatch_20260516T213028Z_modal/harvested_artifacts/_harvest_summary.json`
- `experiments/results/lane_substrate_stc_v2_modal_t4_dispatch_20260516T213028Z_modal/harvested_artifacts/lane_stc_clean_source_v2_results/run.log`
- `experiments/results/lane_substrate_stc_v2_modal_t4_dispatch_20260516T213028Z_modal/modal_training_terminal_claim.json`
- `.omx/state/modal_call_id_ledger.jsonl`
- `reports/cathedral_autopilot_evidence.jsonl`

Verification commands:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/backfill_terminal_claim_evidence.py \
  --earliest-timestamp-utc 2026-05-16T21:20:00Z --max-rows 10 --dry-run
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/harvest_modal_calls.py \
  --from-ledger --execute --get-timeout-seconds 30
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/claim_lane_dispatch.py \
  summary --live-only --format json
.venv/bin/python tools/operator_briefing.py --json
.venv/bin/python tools/all_lanes_preflight.py
```

Observed verification:

- `claim_lane_dispatch.py summary --live-only --format json`: `active_count=0`.
- `backfill_terminal_claim_evidence.py --dry-run`: `missing_terminal_claims=0`.
- `operator_briefing.py --json`: `dispatch_claim_summary.active_count=0`,
  `active_gated_lanes=[]`, `active_composition_lanes=[]`.
- `tools/all_lanes_preflight.py`: Gates #27, #28, and #29 now pass.
- `tools/all_lanes_preflight.py` still has the pre-existing Gate #10
  `untracked source inventory` failure for `experiments/results/` runtime-source
  baseline drift; that is not introduced by this STC v2 closure.

## Reactivation Criteria

Before another STC v2 paid or Modal smoke dispatch:

1. Provide a byte-closed Lane A anchor archive and companion `renderer.bin` /
   `optimized_poses.pt` inputs in the remote worker contract.
2. Verify the mounted path in the recipe or driver before provider launch.
3. Keep `score_claim=false` until an archive is built and exact review classifies
   the result on a named CPU/CUDA axis.
