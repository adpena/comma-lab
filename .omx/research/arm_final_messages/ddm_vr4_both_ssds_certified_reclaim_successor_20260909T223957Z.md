**Blocked: 0 of 120 GiB reclaimed.** Both SSDs were fully inventoried; the [435-row ledger](/Users/adpena/Projects/pact/.omx/research/ddm_vr4_reclaim_ledger_20260909.jsonl) records every large-file blocker. All 435 files remained unchanged.

- **Safety blocker:** `pgrep` cannot enumerate processes; `ps` is denied. No deletion or movement was safe to authorize.
- **Measured lead:** 20 retained-reproducer chains passed, covering **68.218 GiB**. Current raw hashes and live-use clearance remain unverified.
- **Storage:** AP ≈38 GiB free; Vertigo ≈65 GiB. Prefer Vertigo for bounded new writes after fresh preflight.
- **Landing:** Git writes were denied. Verified fallback bundle `de0cc3f7b2` awaits MAIN consumption. [Full report](/Users/adpena/Projects/pact/.omx/research/ddm_vr4_both_ssds_certified_reclaim_successor_20260909.md).

Existing frontier unchanged: **S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN. Consumer store: main Git. Trigger: Git-writable harvest; consume the [verified bundle receipt](/Volumes/VertigoDataTier/pact/ddm_vr4_reclaim_20260909/serializer/20260909T223906.512772Z-76600/receipts.jsonl).
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN/storage custodian. Consumer store: VR4 reclaim ledger. Trigger: working process visibility; rehash and clear live references before certified reclaim.

**LIVE-HYPOTHESES**

- The 20 retained-reproducer candidates may release 68.218 GiB because their archive/runtime/decode chains passed.
- Alternate legacy receiver receipts may certify additional queued raws; the neighbor search located those documents.

**DEAD-ENDS**

- Today’s five AP T4 directories contain only **2,257,407 B**, not the predicted ≥40 GiB of raws.
- Empty `lsof` output cannot establish safety while process enumeration fails.
- Moving bytes between SSDs alone creates no combined capacity.

