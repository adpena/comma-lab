Both SSD reserve targets are restored. All payload bytes were retained.

| Volume | Before free | After serializer | Target |
|---|---:|---:|---:|
| Vertigo | 35.82 GiB | **85.05 GiB** | 80 GiB |
| APDataStore | 32.02 GiB | **62.72 GiB** | 60 GiB |

- **25 verified hardlinks:** Vertigo’s dedup-phase free-space gain was **86,679,662,592 B**.
- **9 verified moves:** retained **32,961,686,400 payload B**; AP’s transfer-phase gain was **32,959,234,048 B**.
- Snapshot pruning: **0 B**. Certified deletion list: **empty**.

Phase gains include metadata and concurrent activity. Final custody checks passed for all 25 hardlinks, nine copies, and 25 MOVED certificates. [Completed memo](/Users/adpena/Projects/pact/.omx/research/ddm_sr4_ssd_reserve_reclaim_20260910.md).

Serializer **rc 17**: sandbox Git-object writes were denied. The fallback commit `e2c5fbe115dbf1217544f1c3ee762bbb4fe53aa8` and all **91 committed file hashes** were verified; the shared index is unchanged. Main landing remains outstanding. [Verification receipt](/Users/adpena/Projects/pact/.omx/research/ddm_sr4_20260910/SERIALIZER_RESULT.json).

No score was measured. Unchanged by this work: **composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: **MAIN**; consumer store: **Pact Git history**; trigger: **MAIN harvest with Git-write access**. Land the verified fallback commit identified in the receipt.

**LIVE-HYPOTHESES**

- Additional cold duplicates may yield space: historical hashes and sampled groups remain unconsumed. Their current equality and physical reclaim are untested.

**DEAD-ENDS**

- Same-volume moves cannot restore that volume’s reserve.
- Already-hardlinked raws provide no new allocation savings.
- No inspected snapshot met the three-day expiry rule.
- Partial samples and MOVE labels cannot prove payload custody.

