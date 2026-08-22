# ddm_ni1_nr1_k32_receiver_distortion — K32 clears the ceiling by 2,391 B, which buys a 7.9% distortion budget for 1,558,833 changed tokens; measure it

## MANDATE

NR1 (`ddm_nr1_taskcell_quotient_prebuild_20260822.md`, commit b3647a73ca) landed the first
EXECUTABLE task-cell quotient — real QPARAM/QCTX/QPAIR/QEVENT, not a spec. K32 measures
**69,004 B packet · 98.6786% token agreement · 2,391 B BELOW the token-only ceiling**, and
projects to **135,595 B** whole-archive if every non-token DX2 byte and distortion hold.

NR1 then stated the open question precisely: *"NR1 is not killed by rate, but matched distortion
is unmeasured: K32 changes 1,558,833 tokens."* No scorer ran; no score was claimed.

**MAIN's derived bar (check it, then use it).** Pointer DX2 S 0.14821987563243377 @ 180,368 B:
rate 0.120100 · seg 0.020139 · pose 0.007981 · distortion 0.028120. Sub-0.12 needs archive
≤ **137,986 B** (STRICT ⇒ FLOOR of 137,986.88). K32's projected 135,595 B leaves **2,391 B** of
slack = **0.001592 S** of additional distortion budget at 6.658e-7 S/B. Holding pose fixed, the
admissible d_seg ceiling is **(0.029712 − 0.007981)/100 = 0.00021731** — only **1.079× the
current 0.00020139**. K32 must change 1.56M tokens and raise d_seg by **less than 7.9%**.
That is the whole question. If your recomputation disagrees, STOP and report the disagreement.

## SCOPE

1. **Verify inherited state, refuse on drift.** `src/tac/optimization/nr1_taskcell_quotient.py`,
   `experiments/ddm_nr1_taskcell_quotient_prebuild.py`, K32 result at
   `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1/RESULT.json`.
   K32 attribution: QPARAM 239 B · QCTX 152 B · QPAIR 52,124 B · QEVENT 16,489 B.
2. **Integrate into the SHIPPING full-RGB DX2 receiver.** NR1's own dead-end warns the July
   quotient ABI is NOT a shortcut (no genuine QCTX, synthetic rank-one renderer, n24/no-op
   fixtures) — do not route through it. The receiver you measure through must be the one that
   would ship; verifying against a non-shipping receiver is the #417 fake.
3. **Measure the distortion NR1 could not.** Advisory n600 d_seg + d_pose through the real
   R/uint8/frozen-scorer path, PER-CLASS d_seg broken out. **Lane (class 1) on its own row** —
   it is 0.59% of area, IoU 0.263 in GT, ~19% of all flips, and the sister candidate RC1 shows
   a rare-class collapse to 0.146 under a different quotient. A headline that averages Lane away
   is the wrong object.
4. **Report against the 0.00021731 bar explicitly.** Pass/fail with the measured number, not a
   qualitative read. If pose moves, re-derive the seg ceiling at the realized pose rather than
   reusing the fixed-pose figure above.
5. **Retain everything.** Repeats + mutation controls + all coder losers + decoded fields.
   ALWAYS KEEP THE PAYLOAD binds at the typing moment: fsynced checkpoints and durable
   success/failure receipts from the START (a sister arm self-reported violating this rule
   today and cured it exactly that way). Emit a sealed fire-order for MAIN; do NOT fire.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm. NO Metal fires (MAIN-fire-only). Local
  advisory launches ONLY via the canonical firer.
- The jo1 r9 run directory is SACRED. NOTE: r9 terminated BY SELF-REFUSAL (typed blocker
  `POSE_CAP_EXCEEDED / COLLATERAL_CAP_EXCEEDED / EXACT_DELTA_NONNEGATIVE`) — there is NO improved
  endpoint. NR1's own fire-order assumes refitting to a frozen primary endpoint; that assumption
  is now false. Work from the CURRENT DX2 body and say so; do not wait on r9.
- Shipped receiver bytes are CUSTODY — integrate additively, never edit in place.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- Receipts to `/Volumes/APDataStore/pact/ddm_ni1_nr1_k32_receiver_distortion/`.
- File ownership: RI1 owns the RC1 candidate's identical measurement on a DIFFERENT payload;
  OS1 owns orphan reconciliation. Do not touch their memos or retained trees. If RI1 lands
  first, CONSUME its per-class method for comparability — do not re-invent the breakout.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_nr1_taskcell_quotient_prebuild_20260822.md` — K128 CLOSED by rate (exceeds the combined
  token+HPAC ceiling by 6,720 B). K64 is 8,481 B above the token-only ceiling but clears the
  COMBINED ceiling by 5,034 B — a live alternative, not a closed one. Integrity-only mutations
  and four named no-op consumers are CLOSED as receiver evidence: a mutation that changes bytes
  without changing what the receiver renders proves nothing.
- `ddm_rc1_rate_crush_20260822.md`=dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d —
  the sister candidate: 113,006 B, 98.795970% token agreement, and class-1 IoU 0.146. Its
  closing dead-end is the one most likely to trap you: *"Overall token agreement cannot be
  promoted as evaluator evidence."* 98.68% agreement is NOT a distortion measurement.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** There is NO retained
  token-level sensitivity corpus to lean on; the quotient-invisibility prediction is
  INCONCLUSIVE, neither confirmed nor refuted. Measure what you claim.
- `ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd —
  the measured envelope terminates at DX2; no composition of measured moves reaches 0.12.
  Summing independently measured legs without a joint receiver receipt is invalid.
- `ddm_db1_decode_boundary_families_20260822.md`=08fd9c4b5d4e583293c3977a8a98abb0205b0a0fc0443e67bd5247aed2de86af —
  a live arm self-reported an ALWAYS-KEEP-THE-PAYLOAD violation today and cured it with fsynced
  per-group checkpoints + durable receipts. Inherit the cure.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_nr1_taskcell_quotient_prebuild_20260822.md`,
  sha e1ae945821f60d0c0fc2de062b6325c2773fde24125dbb1975862bc3c296c64d — it built four REAL
  exact-once-consumed surfaces, measured three K rungs with per-surface byte attribution,
  refused every score claim, and named its own unmeasured leg. Match that bar.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — measuring through a
  non-shipping receiver, or reporting token agreement in place of measured d_seg, is the fake
  this charter refuses.
- **PRIOR-LAW PREDICTION (falsifiable):** the rare-class-collapse law plus the extreme tightness
  of the budget (1.079×, 7.9%) predicts K32's 1,558,833 changed tokens raise d_seg ABOVE
  0.00021731 — byte-feasible, distortion-infeasible, same verdict shape as its sister.
  **FALSIFIER:** measured n600 d_seg ≤ 0.00021731 at unchanged-or-better pose makes K32 a LIVE
  sub-0.12 candidate, and that must be the memo's FIRST line with the exact archive bytes and S
  recomputed FROM COMPONENTS. Report either outcome plainly — a refutation here would be the
  campaign's biggest result to date, and a confirmation honestly closes the quotient's rate
  hypothesis on distortion rather than on bytes.

## DELIVERABLE

`.omx/research/ddm_ni1_nr1_k32_receiver_distortion_20260822.md` — the shipping-receiver
integration (path named + verified) + measured n600 d_seg with per-class breakout and Lane on
its own row + measured d_pose + explicit pass/fail against 0.00021731 (re-derived if pose moved)
+ exact archive bytes + S recomputed from components if score-worthy + repeats and mutation
controls retained + sealed MAIN fire-order or honest closure + the verdict on the prior-law
prediction with verdict_scope at the NARROWEST level supported. Commit via the serializer. End
with the own-vehicle frontier line.
