# ddm_hr1 — harvest-routing: the 29 landed arms' follow-ons, from prose to STORES

Operator directive 2026-08-09 verbatim: *"Also must ensure no signal loss or orphan signal."*

MAIN drained the 29-arm harvest backlog to zero at the **queue-state** level (all marked
`landed`, 3 orphaned findings rescued in commit `b08fd86f87`). The **routing** half is
unpaid, and it is the half that matters: the machine-readable NEXT_IF_RESUMED surface
(`.omx/state/codex_arm_queue.next_if_resumed.jsonl`, 67 rows) holds **ZERO rows for any of
these 29 arms** — MAIN verified this. Their follow-ons therefore exist ONLY as prose inside
final messages and findings memos. That is the orphan surface, and it is measured, not
suspected.

## THE POPULATION (denominator = 29, named)

`ddm_aa1 · ddm_cf2 · ddm_cr1 · ddm_gc21 · ddm_gdl1 · ddm_hb2_hpac_pack_roundtrip · ddm_lx1 ·
ddm_m1c1 · ddm_m1r2 · ddm_m1r3 · ddm_m1r4a_mechanics · ddm_m1r4b_science · ddm_m1r4c_arith ·
ddm_m1r5a · ddm_m1r5b · ddm_m1r5c · ddm_ng1 · ddm_oh1 · ddm_pk1 · ddm_pk2 · ddm_rr16 ·
ddm_rr17 · ddm_rr18 · ddm_rv2 · ddm_tr2_trot_crosswalk · ddm_tr2p1 · ddm_wc1 · ddm_wc2 · ddm_zc1`

Sources, in this order: (1) `.omx/research/arm_final_messages/<name>_*.md` (one per arm,
byte-faithful post-exit capture); (2) each arm's committed findings dir under
`.omx/research/ddm_<name>_*/`; (3) the commit each names.

## OPTIMAL FORM (required block)

- REFERENCE form = every follow-on in the population extracted, adjudicated, and landed on a
  machine-readable STORE a real consumer reads, with the routing denominator reported.
- SCOPE reductions (legal, declare them): bound by wall-clock; report the exact unrouted
  remainder with its denominator and reason.
- MECHANISM reductions (require a TOY-BRACKET declaration): summarizing instead of routing;
  writing a new `.md` as the terminus; counting rows without naming their consumer.
- Provenance: every routed row cites arm name + source file + line/heading + commit sha.

## METHOD

1. **EXTRACT.** For each of the 29, enumerate every forward-looking item: named next
   measurements, "remaining", "owed", "MAIN to review", "queued", "fire-order", blocked
   verdicts with a resolving action, and *un-flagged* ones stated only in prose. A findings
   memo's own "next" section counts. Report per-arm counts.
2. **DEDUPE.** Many will already be done, already tracked, or superseded — several of these
   arms are review rounds of the same M1 ticket. Check the task ledger, `probe_outcomes_ledger`,
   and git log before filing anything as open. A duplicate filed as new is signal *noise*, which
   is the opposite of the job.
3. **ADJUDICATE + ROUTE.** Every surviving row exits with an OWNER and exactly one disposition:
   `FIRED` (done now, $0 and in-charter) · `FOLDED` (written into a real consumer surface) ·
   `QUEUED-WITH-FIRE-ORDER` (named trigger + named owner) · `DEFERRED` (named MEASURED blocker
   + reopening condition) · `SUPERSEDED` (cite what superseded it) · `ALREADY-DONE` (cite the
   commit). **"Unowned", "MAIN to route", and "your call" are FORBIDDEN.**
4. **ROUTE TO STORES, NOT PROSE.** A `.md` alone is a bridge artifact and must name the store
   that should absorb it next. Real consumer surfaces, in preference order:
   `.omx/state/probe_outcomes_ledger.jsonl` (23 callers, 662 live rows — the surface that WON) ·
   the canonical task ledger · `tac.canonical_equations` · a DSL `Lever` · the lane registry ·
   the costate duty-queue · `.omx/state/codex_arm_queue.next_if_resumed.jsonl` (the surface
   these arms *should* have populated).
5. **FIX THE GENERATOR, not just the instance.** The reason these 29 produced zero
   NEXT_IF_RESUMED rows is a real defect in the extractor's contract or in the arm contract that
   feeds it (`tac.subagent_contract`). Diagnose it at source. If the cure is a contract line or
   an extractor predicate, land it — with an executed positive control proving the new path
   catches a block the old one missed, and a negative control proving it does not fire on prose
   that is not a follow-on. **A detector that would report the same thing after the cure is
   applied is not a cure** — check it against that test explicitly.
6. **REPORT THE DENOMINATOR.** rows extracted / deduped / routed / unrouted, and why. A silent
   remainder is the failure; a stated one is fine.

## HONESTY BARS

- Cite everything at source; no claim from working memory.
- Negative-existence claims ("this arm had no follow-ons") require the words "did not find in
  <scope>" — our #1 false-claim class.
- Do not invent a follow-on to fill a row, and do not invent a consumer that does not read the
  store you write to. Verify the reader exists.
- If an arm's finding was a review verdict (several are `FINDINGS_RESET_COUNTER` on the same M1
  ticket), the follow-on is the *named blocking defect*, not the verdict label. Route the defect.
- Six of these arms hit a sandbox `git add` blocker. MAIN rescued 3 orphaned files
  (`b08fd86f87`); verify no OTHER uncommitted artifact from any of the 29 remains on disk, and
  commit-or-certify anything found per the certify-or-block rule.

## BOUNDARIES

Scorer-free. No Metal/MPS/CUDA, no scorer slot, no eval, dispatch, launch, archive build, or
promotion. No upstream edit. No public-PR-intake edit. Commit via
`tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`, tags
`[no-triality] [p0-ledger-ok]`, and NO AI/Co-Authored-By trailer. Two `review_tracker.py
mark-file` passes for any `.py`.

## DELIVERABLE

`.omx/research/ddm_hr1_<UTC>/` with `HR1_ROUTING.jsonl` (one typed row per follow-on: arm,
source, sha, text, disposition, owner, consumer_store, fire_trigger), `HR1_FINDINGS.md`
(denominators, per-arm table, the generator defect + its cure + controls), `RECEIPT.json`.
Final message: rows extracted / routed / unrouted with reasons, the generator defect and
whether it is cured, and the top-3 routed rows by stakes.
