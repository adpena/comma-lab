# HR1 durable follow-on backfill — ddm_rr18

Canonical typed provenance: `HR1_ROUTING.jsonl`. This file is a bridge into the
costate-readable arm queue, not a second disposition engine.

## NEXT_IF_RESUMED

- `rr18_endpoint_facet_provenance_repair`; disposition=QUEUED-WITH-FIRE-ORDER; owner=ddm_mx1g endpoint-facet writer owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=before the endpoint facet corpus is used as a self-contained downstream policy input; action=Patch the endpoint facet writer and rerun, or commit a cache-bound endpoint addendum carrying input/target paths and SHA-256 values, pair IDs, source head, replay argv, axis, and score_claim=false.; source=.omx/research/ddm_rr18_20260808/ROUND18_FINDINGS.md (lines 39-44; lines 174-179, commit 3c3348fd3593de23a6b537973721b7f5f7b080ea, sha256 279736e755f3d8ea1f383601fe1d4ac94287f22716ed84f852badd03672c698a).
- `rr18_armveh_own_curve`; disposition=DEFERRED; owner=MX1 ARM-CAP/ARM-VEH receiver-discriminator owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=ARM-VEH re-enters n120 selection, wins the receiver comparison, or otherwise becomes material; action=If ARM-VEH wins or remains material to n120 selection, obtain its own curve before consuming the ARM-CAP-derived step recommendation; the CAP knee is only a starting prior.; source=.omx/research/ddm_rr18_20260808/ROUND18_FINDINGS.md (lines 157-160; lines 177-179, commit 3c3348fd3593de23a6b537973721b7f5f7b080ea, sha256 279736e755f3d8ea1f383601fe1d4ac94287f22716ed84f852badd03672c698a).
