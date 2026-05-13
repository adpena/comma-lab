# Exact-ready queue retraction refresh (2026-05-13, codex)

## Summary

The exact-ready queue audit was refreshed after Modal custody closure.

Raw audit without suppression still reports 5 stale ready rows across 5
historical queue files. All 5 are already covered by the canonical retraction
manifest:

```text
.omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json
```

Suppressed audit result:

```text
passed True
raw 5 suppressed 5 remaining 0
```

## Command

```bash
.venv/bin/python tools/audit_exact_ready_queues.py \
  --suppression-manifest .omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json \
  --format json \
  --output reports/exact_ready_audit_suppressed_20260513.json
```

The transient report was inspected and removed; this ledger is the durable
state.

## Dispatch conclusion

No historical exact-ready queue row is currently dispatchable. Future dispatch
must regenerate a fresh exact-ready queue from live archive/runtime custody and
must claim the lane before any remote eval.

