# ddm_ip1 — pre-build lv2's FOUR decision-integrity ports (scorer-free, fail-closed, hr2 pattern)

## Mission (lv2 PORT-NOW rows 1-4, 81ca1affef — build NOW during the solve window so the js1
reseal binds finished apparatus, not build debt)

Build the four ports as typed, fail-closed, tested code. NONE may execute against terminal
objects (they don't exist); ALL follow hr2's pattern: typed schemas + refusal-until-bound +
byte-identical no-op when unused. Consumer stores per lv2's NEXT_IF_RESUMED.

1. **m94 CLAIM-UNIT INSTRUMENT SCOPING** → extend the verdict-emission surface (recall FIRST:
   tac.verdicts.verdict_payload() landed 3f2dbb2c14 + vw1's adoption-decay lesson — extend the
   EMBEDDABLE half, do NOT invent a parallel verdict surface): typed fields
   {instrument_capacity, object_capacity, claim_units, scope_ok} with fail-closed validation
   (claim > capacity ⇒ REFUSE the verdict emission). 15+ tests incl. both-direction controls.
2. **m37 SAME-PARENT FRESHNESS ENFORCEMENT** → a typed helper (likely beside hr2's content
   binder in src/tac/witness_dsl/hr1_prestage.py or a sibling module): assert
   producer_parent_sha == consumer_parent_sha for any {fit, map, selector, correction} object
   pair; REFUSE on mismatch or absent parent; no waiver strings.
3. **ACTIVATION-LEDGER TERMINAL JOIN** → a query/report tool (extend, never fork, the existing
   lever_activation_ledger surface — recall its reader): given a compiled DSL config, join
   every non-default lever to {FIRED/FOLDED/queued} evidence from
   .omx/state/lever_activation_ledger.jsonl (251 rows) and emit a typed receipt; REFUSE
   (rc≠0) when any non-default lever has no row. Runnable against ANY compiled config now
   (test with a current lc2/v752-era config as the positive control), terminal config at fire.
4. **EG1 STOP-POLICY INTERFACE PORT** → port the same-parent total-score stop/continue/handoff
   INTERFACE from eg1's E2 deliverable (recall its receipts; grammar invariants ONLY — TR1
   constants/code paths are lv2 dead-ends): typed StopVerdict {continue, handoff, stop} from
   same-parent complete-score dominance receipts; components parameterized, no numeric
   constants baked.

## Boundaries

Scorer-FREE. No Modal. Respect the review gate (.py = 2 clean passes, no override). Serializer
commits w/ post-edit shas, [no-triality] [p0-ledger-ok], --no-co-author. If sandbox git-write
fails (the BLOCKED-GIT class hr1/rvs2 hit), leave a NEXT_IF_RESUMED fire-order for the MAIN
handoff — do NOT bypass custody. Durable memo
`.omx/research/ddm_ip1_decision_integrity_ports_20260811.md` w/ NEXT_IF_RESUMED + DEAD-ENDS.

## OPTIMAL FORM

Pins: lv2 memo+JSONL (81ca1affef) · vw1 verdict receipts (09b361bcc8/3f2dbb2c14) · hr2
(436edf452c) · eg1 E2 receipts · lever_activation_ledger.jsonl (251 rows). SCOPE = all four
ports complete w/ tests; no stubs-as-done (#819 law). PRIOR-LAW PREDICTION (derived from vw1):
the m94 port's hardest risk is ADOPTION not construction — so port 1 MUST land as an extension
of the existing embeddable verdict_payload (which has a named producer contract), not a new
surface; FALSIFIER: if verdict_payload cannot carry the fields without breaking its pinned
contract, land the smallest compatible extension + a migration note, never a parallel API.
