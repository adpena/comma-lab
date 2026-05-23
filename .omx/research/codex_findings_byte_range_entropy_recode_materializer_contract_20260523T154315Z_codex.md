# Codex Findings - Byte-Range Entropy Recode Materializer Contract

UTC: 2026-05-23T15:43:15Z
Lane: `byte_range_entropy_recode_materializer_contract`
Authority axis: planning-only / false authority

## Finding

The byte-shaving master-gradient backlog had a high-value `byte_range` /
`entropy_recode` family, but the queue compiler could only report
`target_kind_required`. That made the DAG unable to distinguish "no known
contract exists" from "a contract exists but the source rows have not named it
or supplied runtime-consumption proof."

## Landing

- Registered `byte_range_entropy_recode_adapter` in the byte-shaving
  materializer registry.
- The adapter is deliberately `executable=false`.
- Receiver contract:
  `byte_range_entropy_recode_receiver.v1`
- Receiver kind:
  `archive_charged_byte_range_entropy_recode`
- Required context fields before execution:
  `archive_member_name`, `archive_byte_range`, `runtime_consumption_proof`
- Queue backlog rows now include `suggested_materializer_count` and
  `suggested_materializers` so target-kind-missing rows can still route
  implementation work to a concrete fail-closed contract.
- Explicit byte-range entropy target-kind rows classify as
  `adapter_not_executable` with
  `receiver_contract_registered_but_adapter_not_executable`.

## Smoke Artifact

Command:

```bash
.venv/bin/python tools/build_byte_shaving_campaign_queue.py \
  --plan .omx/research/byte_shaving_campaign_master_gradient_plan_20260523T144718Z.json \
  --materialization-out .omx/research/byte_shaving_campaign_master_gradient_byte_range_suggested_contract_smoke_20260523T154232Z.json \
  --portfolio-out .omx/research/byte_shaving_campaign_master_gradient_byte_range_suggested_contract_portfolio_20260523T154232Z.json \
  --action-summary-out .omx/research/byte_shaving_campaign_master_gradient_byte_range_suggested_contract_action_summary_20260523T154232Z.json \
  --materializer-backlog-out .omx/research/byte_shaving_campaign_master_gradient_byte_range_suggested_contract_materializer_backlog_20260523T154232Z.json \
  --repo-root . \
  --candidate-limit 8
```

Result:

- executable rows: 0
- blocked rows: 36
- backlog rows: 3
- queue: null
- score authority: false
- promotion eligible: false
- rank/kill eligible: false
- ready for exact eval dispatch: false

Top backlog:

1. `byte_range` / `entropy_recode`: `target_kind_required`,
   `candidate_saved_bytes_sum=268542`, suggested materializer
   `byte_range_entropy_recode_adapter`, target kind
   `byte_range_entropy_recode_v1`, executable false.
2. `byte_range` / `null_remove_or_seed`: `target_kind_required`,
   no suggested materializer.
3. `byte_range` / `delta_encode`: `target_kind_required`,
   no suggested materializer.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py
.venv/bin/python -m ruff check \
  src/comma_lab/scheduler/byte_shaving_materializer_registry.py \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  src/comma_lab/scheduler/__init__.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py
```

Result: `16 passed`; ruff clean.

## Remaining Gaps

- Map master-gradient byte ranges onto exact archive member names and byte
  offsets.
- Prove runtime consumption for rewritten byte ranges before any archive output
  is queueable.
- Implement a local byte-range recode smoke over a real member payload using
  byte-closed archive read/write helpers.
- Register separate contracts for `null_remove_or_seed` and `delta_encode` only
  after their archive/runtime semantics are precise enough to fail closed.
