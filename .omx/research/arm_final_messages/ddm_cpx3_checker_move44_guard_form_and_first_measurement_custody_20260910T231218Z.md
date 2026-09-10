Completed `ddm_cpx3`: **81/93 → 91/93**, strict **rc 1**.

| Checks | Result |
|---|---|
| CPU refusal and policy: 7 | PASS |
| Dispatch custody: 3 | PASS |
| Import hygiene and hosting: 2 | Still FAIL |

**222 tests passed**, Ruff passed, and two review passes completed per Python file. Receiver, README, volumes, upstream, frozen consumers, and shared index remain unchanged. No publishing, Modal calls, scorer runs, or new score measurement.

Serializer ran once: **rc 17**, blocked by sandbox Git writes. Commit `a272bab138e` is retained in a bundle and independently verified **38/38 files**; MAIN landing remains pending.

[Memo](/Users/adpena/Projects/pact/.omx/research/ddm_cpx3_checker_move44_guard_form_and_first_measurement_custody_20260910.md) · [Landing receipt](/Users/adpena/Projects/pact/.omx/research/ddm_cpx3_20260910/SERIALIZER_STATUS.json)

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer `.omx/research/ddm_cpx3_20260910/SERIALIZER_STATUS.json`; trigger verified-bundle harvest: land the exact commit.
- **FOLDED** — owner operator + MAIN; consumer `submissions/_staging_move44_pr140_swap/_packet/BLOCKERS.json`; trigger receiver decision: resolve import hygiene and README; obtain fresh exact evidence for receiver edits.
- **FOLDED** — owner operator + MAIN; consumer `submissions/_staging_move44_pr140_swap/SWAP_COMMANDS.md`; trigger non-hosting clearance and explicit publishing authorization: host, verify, and require strict PASS before swapping.

**LIVE-HYPOTHESES**

- Removing fallback imports may preserve receiver behavior because packaged relative imports are tried first. This requires fresh receiver validation.

**DEAD-ENDS**

- Move-43 receipts cannot validate move 44: archive/runtime bindings differ.
- Old guard text and line numbers cannot establish current receiver behavior; actual source and traceback must agree.
- An older harvested claim alone cannot establish completed-seal custody.
- **91/93 is not release clearance**; import hygiene and hosting remain unresolved.