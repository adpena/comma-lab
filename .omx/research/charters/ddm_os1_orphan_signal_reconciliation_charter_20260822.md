# ddm_os1_orphan_signal_reconciliation — memos that exist only on local disk are invisible to the instrument that has redirected three arms today

## MANDATE

Routed finding (MAIN, 2026-08-22): **recall-before-decide redirected THREE arms in a single
session** — it killed a partition-R(D) charter by surfacing `ddm_tk1_20260806/RECEIPT.md`
(all-lossless ladder, Route S dominated by the live pointer); it stopped a duplicate NR1
charter by surfacing the existing one; and it stopped a cross-stream charter by surfacing
`ddm_rb1_rate_bound_decomposition_20260822.md`'s explicit warning that *"calling cross-stream
jointness a saving before real serialization would merely move uncounted payload between
labels."* Each save cost one grep. That instrument is measurably the highest-leverage
apparatus we have right now.

**And it is running on an incomplete index.** A prior audit found 42 research memos existing
only on local disk — invisible to git, to the graph builder, and to the corpus. MAIN measured
the live figure at the moment this charter was written: **21 untracked `.md` files under
`.omx/research/`**, most of them `arm_final_messages/*` byte-for-byte captures. Whether those
are deliberate (np1 forensic captures with a standing reason to be untracked) or orphaned
(landed research nobody committed) is EXACTLY the question — and nobody has adjudicated it.

An orphaned memo is worse than an unwritten one: the work was done, the tokens were spent,
and the next arm re-derives it blind. This arm measures the real population, adjudicates each
class, and closes the birth path so a memo cannot be born invisible.

## SCOPE

1. **Measure the POPULATION, with its denominator.** Not "42" — the live count, by class:
   untracked-in-git · tracked-but-absent-from-the-graph-index · present-in-graph-but-
   unreachable-by-recall (`tools/graph_memory_recall.py`). These are THREE different
   invisibilities and a file can have any subset. State each count against its own
   denominator (m50: a count without its denominator is not a measurement). MAIN's live
   reading was 21 untracked `.md` under `.omx/research/` — reproduce or correct it, and say
   which.
2. **Adjudicate each class, do not blanket-commit.** `arm_final_messages/*` are np1
   byte-for-byte forensic captures; mutating them destroys evidentiary value (#878), and
   there may be a standing reason they are untracked. Determine the INTENT for each class
   from the code that writes them, not from guesswork. Then apply the certify-or-block rule:
   commit what should be committed, CERTIFY what should stay out (with the machine-readable
   reason on record), and never silently delete.
3. **The recall-reachability leg — the one that actually matters.** For a sample of memos
   that ARE tracked, verify they are reachable by `tools/graph_memory_recall.py` on a query
   whose answer they contain. A memo that git holds but recall cannot surface is invisible
   where it counts. Report the reachability rate with its denominator and its sampling method
   (seeded random, NOT a prefix — prefix-of-a-skewed-population is a different population).
4. **Close the birth path.** If a producer writes memos to a location that is gitignored or
   graph-invisible by default, that is the structural defect; a one-time sweep would leave the
   class alive. Land the cure at the producer or name precisely why it cannot be fixed there.
5. **Do NOT rebuild the graph as a side effect of measuring it** — measure first, report the
   before/after separately if you rebuild at all, so the reachability numbers are not
   self-confounded.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire, NO scorer runs, NO Metal fires, NO local advisory
  launches. $0. Metal controls are MAIN-fire-only.
- The live JO r9 run directory is SACRED — read nothing from it, write nothing into it.
- **Do NOT mutate any `arm_final_messages/*` file.** They are byte-for-byte forensic captures
  (#878). You may commit, certify, or classify them; you may not edit them.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_os1_orphan_signal_reconciliation/`.
- File ownership: parallel arms own RC1 (rate_crush), NR1 (quotient build), VF1 (token census).
  Do not touch their memos or retained trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_nl1_never_fired_levers_20260822.md` — 31 of 37 unmeasured lever names pointed at
  RETIRED vehicles. The lesson generalizes: an inventory that does not check whether each row
  still points at a live object produces a worklist of ghosts. Check liveness before counting.
- `ddm_rb1_rate_bound_decomposition_20260822.md` — the "moving payload between labels" trap.
  Its analogue here: re-classifying an orphan as tracked without changing reachability is
  relabelling, not curing. The test is whether RECALL surfaces it, not whether git holds it.
- `ddm_db1_decode_boundary_families_20260822.md` — DB1 self-reported an ALWAYS-KEEP-THE-PAYLOAD
  violation during development (a transcript discarded after decoder failure, then
  reconstructed and retained) and cured it with fsynced per-group checkpoints + durable
  success/failure receipts. That is a FRESH instance of the class today, from a live arm, and
  a proven cure pattern — consider whether the same durability shape applies to memo birth.
- The wrong-denominator genus has bitten this campaign repeatedly: a headline count of "42"
  that was never re-derived is exactly the shape that produced a "0 of 36" claim which turned
  out to be 5/5 = 100% on the true population. Re-derive; do not inherit the number.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_nl1_never_fired_levers_20260822.md`,
  sha a11e56b228513c066b803cb6c03e7ce31d2af40d7271b812abaff5e16b5ced3a — it corrected its own
  charter's stale count (275 vs the true 279), kept the distinct denominator that WAS right,
  drained the queue to zero unmeasured non-retired rows, and recorded four retirements
  honestly. Match that bar: correct the inherited number, state every denominator, exit with
  zero UNKNOWN rows.
- Provenance pins (verify each at start; refuse if the tree drifted):
  `.omx/research/ddm_nl1_never_fired_levers_20260822.md`=a11e56b228513c066b803cb6c03e7ce31d2af40d7271b812abaff5e16b5ced3a
  `.omx/research/ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09
  `.omx/research/ddm_db1_decode_boundary_families_20260822.md`=08fd9c4b5d4e583293c3977a8a98abb0205b0a0fc0443e67bd5247aed2de86af
  `.omx/research/ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd
- SCOPE reductions declared per row (sampling the reachability leg is legal if the sampling
  method is stated and seeded). MECHANISM reductions FORBIDDEN — a sweep that commits files
  without adjudicating intent, or that reports git-tracking as if it were recall-reachability,
  is the relabelling fake this charter exists to refuse.
- **PRIOR-LAW PREDICTION (falsifiable):** the orphan population is dominated by ONE producer
  writing to ONE default location (the #821 law: N instances usually trace to one fact counted
  N times), so a single producer-side cure closes most of it. **FALSIFIER:** if the orphans
  are spread across ≥4 distinct producers with no shared default, the one-cure hypothesis is
  refuted and the honest answer is a per-producer table, not a structural fix. Count it plainly
  either way.

## DELIVERABLE

`.omx/research/ddm_os1_orphan_signal_reconciliation_20260822.md` — the three-class population
with per-class denominators + the per-class intent adjudication (commit / certify-with-reason /
leave-with-reason) + the recall-reachability rate with its sampling method + the producer-side
cure (landed, or precisely why not) + the explicit verdict on the prior-law prediction with
verdict_scope at the NARROWEST level the evidence supports. Commit via the serializer. End with
the own-vehicle frontier line.
