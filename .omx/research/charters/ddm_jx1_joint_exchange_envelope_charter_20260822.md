# ddm_jx1_joint_exchange_envelope — the JOINT (rate × seg × pose) lower envelope at the live operating point, and the ranked fire table that reaches 0.12 or names the exact residual

## MANDATE

Routed finding (`ddm_rb1_rate_bound_decomposition_20260822.md`): the sub-0.12 demand is
**shed 42,382 B** — but that figure is computed **at FIXED distortion**. The score has a
measured exchange rate: 0.001 S of distortion buys 1,502 B of rate budget. Nobody has
solved the JOINT problem. Every arm this session optimized ONE axis and reported a
single-axis ceiling: RB1 found 0 B rate headroom holding distortion fixed; EC2 found even
a PERFECT zero-collateral seg conditioner reaches only 0.137984 holding rate fixed. Two
single-axis ceilings do not compose into a joint ceiling, and the campaign has been
pricing moves against whichever axis the arm happened to own.

This arm builds the joint envelope from MEASURED moves only and emits a **ranked fire
table**: the ordered, jointly-priced composition that reaches S ≤ 0.12, or — if no
composition of measured moves reaches it — the exact residual mass that a NEW
representation must supply, stated in bytes and in S, with the axis it must come from.
That residual number is the campaign's actual target and it does not currently exist.

This is a DECISION deliverable, not a survey. A table of interesting rows that does not
end in an ordered fire list, or in a named residual, has not done the job.

## SCOPE

1. **Verified arithmetic (check it, then use it; do NOT re-derive from scratch).**
   Pointer DX2: S 0.14821987563243377 @ 180,368 B, archive sha
   976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674.
   rate = 25·180368/37545489 = 0.120100 · seg = 100·0.00020139 = 0.020139 ·
   pose = sqrt(10·6.37e-6) = 0.007981 · sum 0.148220 (matches to 6 s.f.).
   Distortion 0.028120. Gap to target 0.028220. **The rate term ALONE is 0.120100 —
   already above 0.12 at zero distortion**, which is why rate-representation is mandatory
   rather than optional. Marginal values at THIS operating point (derive and state them):
   d(S)/d(byte) = 25/37545489 = 6.658e-7 · d(S)/d(d_seg) = 100 ·
   d(S)/d(d_pose) = 5/sqrt(10·d_pose) = 5/0.007981 = 626.5. Report any disagreement with
   your own recomputation as the finding.
2. **The measured-move ledger.** Every move with a RETAINED measured (Δbytes, Δd_seg,
   Δd_pose) triple on the DX2/RC2 lineage — banked micro-edits, admitted and refused
   candidates, token drops, carrier re-solves, pose re-solves, compensated semantic edits.
   Per row: the triple · its repeat-noise floor · its ΔS AT THIS OPERATING POINT using the
   marginals above (NOT the ΔS quoted in its own memo, which was computed against a
   different, now-superseded baseline — a delta without its baseline is unanchored, and
   baselines moved repeatedly this session) · whether it is still REACHABLE on the DX2
   body or was measured on a superseded body.
3. **The joint envelope.** Solve for the min-S composition subject to receiver closure.
   **UNION ≠ SUM OF LEGS — measured at 3.705× in this campaign.** Any composed figure
   built by summing legs is an UPPER BOUND ONLY and must be labelled so on its face. Where
   joint remeasure receipts exist, use them and say so; where they do not, the row is
   projection, not measurement, and must carry that label in the table itself, not in a
   footnote.
4. **The fire table.** Ordered by realized-ΔS-per-unit-risk, each row carrying: the move ·
   its joint price · its receiver-closure status · its owner · its fire trigger. If a
   prefix of the table reaches S ≤ 0.12, that prefix IS the campaign answer — say so in
   the first line of the memo and hand MAIN a fire order.
5. **The residual, if the table does not reach.** State it three ways: bytes that a new
   representation must supply at fixed distortion · the equivalent distortion reduction at
   fixed rate · the joint frontier point that minimizes total new mass. Name which axis
   (token field / semantic / carrier / pose) it must come from, with the per-stream anatomy
   from RB1 as the constraint. This number is the deliverable's whole point.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- NO scorer runs, NO Metal fires, NO local advisory launches — $0, retained receipts only.
  Metal controls are MAIN-fire-only. You may RECOMMEND a fire; you may not fire.
- The live JO r9 run directory is SACRED — read nothing from it, write nothing into it.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_jx1_joint_exchange_envelope/`.
- File ownership: parallel arms own RC1 (rate_crush), DB1 (decode boundary), VF1
  (evaluator-visible floor). Do not touch their memos or retained trees; CITE their landed
  rows. VF1 is measuring the load-bearing token mass — if VF1 lands before you finish,
  consume its census as a constraint on your residual rather than re-deriving it.
- Do NOT invent a move that has no retained measurement. An unmeasured row in a fire table
  is the borrowed-number fake.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ec2_collateral_suppressed_conditioner_20260822.md` — falsifier SPLIT: separation
  CONFIRMED (H fell 98.08%) but no-benefit-shrink FAILED (B retention 55.24%); and **even
  a perfect zero-collateral EC1 reaches only 0.137984**, 0.018 above target. The seg axis
  has a measured ceiling well short of 0.12 on its own. Do not model seg as unbounded.
- `ddm_rb1_rate_bound_decomposition_20260822.md` — 0 B tested headroom across all seven
  streams at fixed distortion; tokens 113,777 B = 63% of the archive. The rate axis has a
  measured ceiling on its own. Two ceilings, one envelope — that is the arm's whole point.
- `ddm_xt1_exact_solve_teacher_student_20260822.md` — CLOSED: linear post-hoc overlays
  (43–997 B, heldout negative-or-zero at zero repeat noise) · finishing-stage solve-field
  KD (lost by 12.8× its noise floor) · explicit solved-value tails · incumbent stream
  recoding. Do not resurrect these as fire-table rows.
- `ddm_tk1_20260806/RECEIPT.md` — Route S at 168,892 B projects S 0.157386, **worse than
  the live 0.14822 despite being 11,476 B smaller**: a cheaper representation that costs
  distortion can lose outright. This is the single sharpest reason the envelope must be
  solved jointly rather than per-axis.
- `ddm_fp1_class_field_projection_20260731.md` — flat-paint floor f′ = 0.008305,
  FORMULATION-scoped: the partition alone does not determine the argmax.
- `ddm_nl1_never_fired_levers_20260822.md` — 31 of 37 unmeasured lever names pointed at
  RETIRED vehicles. Verify every ledger row targets the incumbent DX2 HPAC body before
  admitting it to the table; a move measured on a superseded body is a different quantity.
- Banked-union precedent: qs2 (−4.375e-6, +34 B) ∪ re1 (−1.207e-6, 0 B) projected ≈
  −5.6e-6, BELOW the naming bar — unions of micro-moves have repeatedly under-delivered
  against their summed legs. Treat the 3.705× union-vs-sum gap as the prior, not the tail.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_rb1_rate_bound_decomposition_20260822.md`,
  sha fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 — the per-stream
  anatomy that turned "the archive is too big" into seven numbered streams with tested
  headroom each. Match that bar on the joint axis: every row a number, every number a
  receipt, every ceiling stated as a ceiling.
- Provenance pins (verify each at start; refuse if the tree drifted):
  `.omx/research/ddm_ec2_collateral_suppressed_conditioner_20260822.md`=466d75ad05b7cd3489c7c345ba32a5cf7a92a91386be6fa03fe39658dbdb9715
  `.omx/research/ddm_xt1_exact_solve_teacher_student_20260822.md`=6437bc53d96e527049c3fd6cd60b91af220305881a7bcc68195fece15a728867
  `.omx/research/ddm_tk1_20260806/RECEIPT.md`=5519cce5a986ffd1536233c2f0865a1ce2f95996293f230cb8a0da0f30e09861
  `.omx/research/ddm_tl1_teacher_ledger_20260822.md`=d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15
  `.omx/research/ddm_nl1_never_fired_levers_20260822.md`=a11e56b228513c066b803cb6c03e7ce31d2af40d7271b812abaff5e16b5ced3a
- SCOPE reductions declared per row (retained-receipt-only, projection-where-no-joint-
  receipt-exists — both legal if LABELLED ON THE ROW). MECHANISM reductions FORBIDDEN — a
  fire table whose composed ΔS is a sum of legs presented as a measurement is the fake this
  charter exists to refuse.
- **PRIOR-LAW PREDICTION (falsifiable):** given RB1's 0 B rate headroom, EC2's 0.137984
  perfect-seg ceiling, and the union-vs-sum 3.705× prior, NO composition of currently
  measured moves reaches S ≤ 0.12; the envelope will terminate short and the residual will
  require new-representation mass on the order of 20,000–40,000 B. **FALSIFIER:** if a
  jointly-priced prefix of the fire table reaches S ≤ 0.12 — or comes within one
  admissible move of it — the prediction is refuted, and that prefix must be surfaced to
  MAIN as a fire order in the memo's FIRST line, not buried in a table. Report either
  outcome plainly; a refutation here ends the campaign and is the best possible result.

## DELIVERABLE

`.omx/research/ddm_jx1_joint_exchange_envelope_20260822.md` — the marginal-value table at
the live operating point + the measured-move ledger (triple · noise floor · re-priced ΔS ·
reachability on DX2) + the joint envelope with every projected row labelled UPPER-BOUND on
its face + the ranked fire table with owners and triggers + the residual stated three ways
with its required axis named + the explicit verdict on the prior-law prediction with
verdict_scope at the NARROWEST level the evidence supports. Commit via the serializer. End
with the own-vehicle frontier line.
