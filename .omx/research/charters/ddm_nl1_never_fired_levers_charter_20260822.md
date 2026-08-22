# ddm_nl1_never_fired_levers — drain the built-but-never-fired live-vehicle levers and the fired-but-never-measured firings (owning memo: ddm_rb1_rate_bound_decomposition_20260822.md for the byte bar)

## MANDATE

Operator 20260822: *"Remember that Codex is available"*, under the standing P0 *"Pursue todo
as p0."* Two measured orphan populations sit on the live vehicle and have sat for days:
TWELVE never-fired live-vehicle levers, and THIRTY-SEVEN lever firings with NO measurement in
the ledger (275 firings, 7 measurements). Both are the campaign's own recorded defect classes:
a lever built and never fired is orphaned signal, and a lever fired without a measurement is
worse — it consumed a window and produced nothing citable. rb1 has now made the bar concrete:
the archive must shed 42,382 B and tested incumbent headroom is 0 B, so any lever with a real
byte or distortion effect is worth strictly more than it was when it was parked.

## SCOPE

1. INVENTORY both populations from the ledgers, not from memory — the never-fired live-vehicle
   levers and the fired-without-measurement firings. State each population's DENOMINATOR (the
   vacuity law: a count without its denominator is not a measurement).
2. Per row: is the lever live-vehicle-relevant TODAY (the instruments-point-at-a-retired-
   vehicle hazard is a known class), what would firing it cost, and what would it measure?
3. FIRE the scorer-free subset that can be measured on retained receipts / real coders now,
   and REPORT the measurements. Rows needing the scorer become a ranked queue with fire-orders
   for MAIN, never silent.
4. Every row exits OWNED: measured · queued-with-a-fire-order · or retired with a reason.
   No row may exit UNKNOWN.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- **THE jo1 r9 RUN DIR IS SACRED AND LIVE** (pid 76768) — do not read-lock, write, move, or
  clean anything under experiments/.scratch/ddm_jo2_joint_objective_solve/**.
- **SCORER-FREE.** r9 holds the single Metal slot; every measurement is $0 CPU over RETAINED
  receipts or real coder runs on retained payloads. A scorer run from this arm is a breach.
- SHIPPED RECEIVER BYTES ARE CUSTODY — read, never edit.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to the APDataStore tier under your own arm name.
- FILE OWNERSHIP: you own `experiments/ddm_nl1_*`, your own memo, and ledger APPENDS.
  Do NOT edit rc1/db1/xt1 surfaces, nr1 surfaces, or shared coder modules. Ledger writes are
  APPEND-ONLY — never rewrite or quarantine shared state.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- The lever instruments point at a RETIRED vehicle and 8 live-vehicle levers are mis-filed for
  want of one declaration — check vehicle-relevance BEFORE pricing any row, or the whole
  inventory measures the wrong object.
- ddm_rb1_rate_bound_decomposition_20260822.md: rate-side incumbent headroom is 0 B, so a
  lever whose only claim is a re-code of existing streams is refuted before firing.
- ddm_ec2_collateral_suppressed_conditioner_20260822.md: even a PERFECT zero-collateral EC1
  reaches only 0.137984 — a seg-side lever must be priced against that ceiling, not against
  the raw gap.
- same_defect_negatives_masquerade_as_family_convergence_20260805: N parked levers sharing one
  parking reason is ONE instance, not N verdicts.

## OPTIMAL FORM

- Family exemplar (reference): ddm_ec2_collateral_suppressed_conditioner_20260822.md, landed
  37d9474c1a — reported the confirmed half and the failed half of its own prediction, banked
  6 dead-ends, and refused to promote an unpriced artifact to candidate. Match that bar per
  row: a fired lever without a reported number is exactly the defect this charter exists to
  drain, so do not create more of it.
- Provenance pins:
  .omx/research/ddm_rb1_rate_bound_decomposition_20260822.md=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09
  .omx/research/ddm_tl1_teacher_ledger_20260822.md=d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15
- SCOPE reductions declared per row (a bounded subset is legal if the DENOMINATOR and the
  skipped rows are named). MECHANISM reductions FORBIDDEN.
- **PRIOR-LAW PREDICTION (falsifiable):** the campaign's parked-lever population is dominated
  by rows parked for VEHICLE-RELEVANCE reasons rather than for measured failure, so a
  vehicle-relevance re-check should reclassify a substantial share as never-actually-tested
  rather than tested-and-dead. FALSIFIER: the large majority of both populations turn out to
  carry a real measured failure on the LIVE vehicle — then the parking was correct, the
  populations are honestly drained, and the orphan framing was wrong. Count it plainly.

## DELIVERABLE

`.omx/research/ddm_nl1_never_fired_levers_20260822.md` — both inventories with denominators,
per-row vehicle-relevance + cost + what-it-measures, the scorer-free subset FIRED with
reported numbers, and a ranked fire-order queue for the rest. Zero UNKNOWN exits. Commit via
the serializer. End with the own-vehicle frontier line.
