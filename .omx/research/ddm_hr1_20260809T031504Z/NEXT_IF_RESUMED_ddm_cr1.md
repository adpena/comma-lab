# HR1 durable follow-on backfill — ddm_cr1

Canonical typed provenance: `HR1_ROUTING.jsonl`. This file is a bridge into the
costate-readable arm queue, not a second disposition engine.

## NEXT_IF_RESUMED

- `cr1_p2_edge_conditioned`; disposition=QUEUED-WITH-FIRE-ORDER; owner=#984 edge-stream receiver owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=Exact label/annulus residual object exists; require exact decode and receiver parse-back before score claim.; action=edge-conditioned representation beat pooled baseline; receiver/scorer survival still unclaimed; source=.omx/research/ddm_cr1_20260808/CR1_ROWS.jsonl (line 2, commit e3c3fb35418ff000c812d3390c5d52c8f877f470, sha256 423857d24c99eeb1ef06b6e3f7aebcb6824a27f16ae85d7f9812122f9da2f791).
