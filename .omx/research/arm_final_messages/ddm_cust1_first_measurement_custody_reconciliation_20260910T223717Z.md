**Partial: custody artifacts are verified; two ledger events remain blocked.**

| Gap | Receipt / result |
|---|---|
| Run3 reconciliation | pr16-schema receipt written; failure remains non-promotable |
| Run1/run4 claims | Two corrective rows added; historical rows preserved |
| Run5 reconciliation | Success receipt written and linked to MAIN’s v3 seal |
| Authorizations v1–v8 | All eight hashed and registered |

[Memo and verification table](/Users/adpena/Projects/pact/.omx/research/ddm_cust1_first_measurement_custody_reconciliation_20260910.md) · [Final handoff](/Users/adpena/Projects/pact/.omx/research/ddm_rlc5_20260910/custody/FINAL_HANDOFF.json)

**Ledger events written: 0.** The canonical helper rejects both reconciliation event types. Fixing its whitelist exceeds the charter’s prohibition on source edits.

**Serializer rc: 17.** Git object writes were denied; fallback commit `f14eecadf999fb0f4f37ce4904d619dddd7a588f` is retained in a verified bundle. Checkpoint marked complete; shared index unchanged.

Only custody checks were performed—no evaluation, dispatch, completion, or volume writes. MAIN independently advanced the pointer. Final canonical line:

> S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store:** bundle referenced in `custody/FINAL_HANDOFF.json`; **trigger:** harvest in a Git-writable environment. Land the verified custody patch.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store:** `custody/LEDGER_EVENT_ATTEMPTS.json` → canonical call ledger; **trigger:** authorized event-taxonomy support lands and references rehash. Append both reconciliation events once.

## LIVE-HYPOTHESES

- A narrow taxonomy extension should close both ledger gaps: both calls failed at the whitelist before mutation. This remains untested.

## DEAD-ENDS

- Current helper: rejects the required event types.
- Run4 promotion: forbidden by pr17’s stale-intent ruling.
- Run3 success or nonce reuse: contradicted by retained failure and permanent reservation.
- Direct landing here: sandbox denies Git object writes; verified bundle retained.

