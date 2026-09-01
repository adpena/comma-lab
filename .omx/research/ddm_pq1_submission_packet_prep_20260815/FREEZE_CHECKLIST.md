# Freeze checklist — packet generation 7, AFR1

State: **FROZEN, NOT PUBLISHED**. Remaining-gate owner: **operator**.

| Property | Frozen value |
|---|---|
| Archive | `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 180,002 B |
| Runtime | 38 rows, tree `6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58` |
| Exact score | `0.14797617125559104` `[contest-CUDA]` T4 n600 |
| CPU axis | RECORD-WITH-REASON; no AFR1 score, none inherited |
| E2E rebuild | not re-run; VERIFIED remains generation-3-only |
| Packet custody | `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/` |
| Receipt custody | `/Volumes/APDataStore/pact/ddm_pq12/receipts/` |

## The one remaining gate

**BLOCKED-ON-OPERATOR-CONFIRM-AND-TEXT.** The operator must personally read the
current contest LLM policy and retained input memos, decide that the publication
path is acceptable, write the full PR description and every public-facing
comment in their own words with honest authorship/attribution disclosure, and
then give the one-line authorization permitting hosting and any PR/open/push
action.

Until then, the hosted-manifest check stays red and no network publication action
is authorized. `PR_BODY_DRAFT.md` is input material only.

## Closed before freeze

- AFR1 archive and 38-row runtime rehashed from disk.
- Tolerance-zero packet seal validated.
- Stager reproduced the evaluated enumerated runtime tree.
- `MANIFEST.sha256` regenerated from the authority receipt.
- Accounting section 11 appended without rewriting historical sections.
- Generation-7 census and strict compliance receipt retained; exact outcomes are
  in the pq12 freeze memo.

The historical 0/5 review counter remains a research-QA ledger, not a second
publication authority under the pq12 charter.
