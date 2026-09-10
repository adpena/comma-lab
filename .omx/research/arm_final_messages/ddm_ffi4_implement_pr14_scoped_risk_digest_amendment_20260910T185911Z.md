Implemented and verified; **MAIN landing remains pending**. The single serializer attempt returned **rc 17** (Git object write denied). Bundle commit `8948b1706687290b81e7208f5db9153b6c7a2ddc` independently verified all 14 intended files.

- **135 tests passed**, including all six named controls; Ruff clean.
- Two review passes completed per changed Python file.
- Both real trees have 49 byte-identical normalized rows:

| Tree | Risk digest |
|---|---|
| RLC1 reference | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` |
| RLC4 candidate | `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890` |

[Memo and clause mapping](/Users/adpena/Projects/pact/.omx/research/ddm_ffi4_implement_pr14_scoped_risk_digest_amendment_20260910.md) · [Verified bundle handoff](/Users/adpena/Projects/pact/.omx/research/ddm_ffi4_20260910/FINAL_HANDOFF.json) · [Freeze-row draft](/Users/adpena/Projects/pact/.omx/research/ddm_ffi4_20260910/FREEZE_APPEND_DRAFT.json)

The draft leaves only the implementation commit and manifest SHA placeholders. No Modal, authorization, fire, timing window, n600 run, encoder, decoder, scorer, or payload cleanup occurred. No volume writes. Upstream, PR/sealed trees, legacy timing code, frozen receipt, sj1 directories, common-contract protected files, and staged index remain untouched. Exclusion is confined to the risk digest; full custody remains manifest-inclusive.

Unchanged: **composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer: verified bundle handoff; trigger: harvest and hash/test revalidation. Land the implementation.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer: freeze-row draft and frozen receipt; trigger: implementation landed and committed-source manifest verified. Append and commit the amendment.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN/rlc5; consumer: reused RLC4 evidence and new risk receipts/intent; trigger: both landings committed, evidence pins valid, pointer still move 42. Re-emit and validate the intent.

## LIVE-HYPOTHESES

- The real producer may now pass after the freeze append: retained risk digests agree. The producer-to-MAIN custody path remains untested.

## DEAD-ENDS

- Broader manifest exclusion is closed: full custody and legacy timing retain it.
- Non-pin `inflate.py` byte changes refuse risk inheritance.
- Pre-amendment intents, stale/uncommitted freeze rows, and forged source manifests refuse.
- Re-timestamping old evidence is unnecessary: pinned older receipts pass the amended chronology controls.