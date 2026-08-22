# ddm_ri1_rc1_full_rgb_receiver — RC1 clears the byte demand with 24,980 B to spare and has UNKNOWN distortion; build the receiver that turns bytes into a score

## MANDATE

RC1 (`ddm_rc1_rate_crush_20260822.md`, commit 35710b32bd) produced the first representation
that COVERS the full sub-0.12 byte demand: K=2,048 terminal temporal program, counted payload
**59,884 B**, complete shadow archive **113,006 B**, cut vs DX2 **67,362 B**, headroom below
the 137,986 B ceiling **24,980 B**, token agreement **98.795970%**.

RC1 then refused to promote it, correctly: *"this is not a score result. The receiver currently
stops at reconstructed tokens, and class-1 IoU is only 0.146; no full-RGB render, scorer, Modal
dispatch, or evaluator ran."* Its own dead-ends say it plainly: *"Overall token agreement cannot
be promoted as evaluator evidence; K=2,048's rare-class collapse makes the exact receiver/scorer
leg mandatory."*

**Class 1 is Lane** (canonical comma10k order: 0=Road, 1=Lane, 2=Undrivable, 3=Movable,
4=MyCar). Lane is already the weakest class in the GT at IoU 0.263 and carries ~19% of all
d_seg flips. **0.146 is rare-class collapse on the single most d_seg-sensitive class.** The
arithmetic 0.1033663 RC1 quotes is a BYTE statement in score notation and must never be cited
as a score.

This arm builds the missing leg: integrate RC1's retained payload into a full-RGB DX2 receiver
so the candidate can be scored. It does NOT fire the scorer — MAIN owns that.

## SCOPE

1. **Verify the inherited state before building.** RC1 payload sha `eab66bad…e61164`, canonical
   result `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/RESULT.json`, sealed
   fire-order sha `0d683cd3…eae56`, module `src/tac/optimization/rc1_terminal_program_vq.py`,
   materializer `experiments/ddm_rc1_rate_crush.py`. Refuse if any drifted.
2. **Build the full-RGB integration.** RC1's receiver stops at reconstructed tokens; DX2's
   shipping receiver renders RGB. Extend/adapt so the RC1 payload decodes through the ACTUAL
   shipping DX2 render path to full RGB frames. The receiver you integrate into must be the one
   that would SHIP — verifying against a receiver that will not ship is the #417 fake.
3. **Measure the distortion RC1 could not.** Advisory n600 d_seg + d_pose through the real
   R/uint8/frozen-scorer path on the integrated receiver, with PER-CLASS d_seg broken out.
   Lane (class 1) gets its own row — the 0.146 IoU is the named alarm and a headline that
   averages it away is the wrong object.
4. **Retain everything, controls included.** Exact repeats + all-paid-section mutation controls,
   per RC1's fire-order. ALWAYS KEEP THE PAYLOAD binds at the typing moment: never a scalar-only
   artifact when bytes exist in memory. A sister arm self-reported a violation of this rule
   TODAY (see PRIOR NEGATIVE SIGNAL) — write fsynced checkpoints and durable success/failure
   receipts from the start, not after a failure.
5. **Emit a sealed fire-order for MAIN, do not fire.** If the measured distortion makes the
   candidate score-worthy, say so with the exact archive bytes and the recomputed S from
   COMPONENTS (never a rounded display field). If it does not, say THAT — a byte win with
   catastrophic Lane is a closed row, and closing it honestly is the correct outcome.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- NO Metal fires (MAIN-fire-only). Local advisory launches ONLY via the canonical firer.
- Shipped receiver bytes are CUSTODY — never edit them in place; integrate additively.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- Bulky receipts to `/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/`.
- File ownership: parallel arms own NR1 (quotient build) and OS1 (orphan reconciliation). Do
  not touch their memos or retained trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rc1_rate_crush_20260822.md` — K=4,096 is byte-dead (105,811 B payload / 158,933 B
  shadow). Fixed-RC64 coder/context races cannot supply the cut (88 B ceiling already shipped).
  PR130 memoryless bounds are NOT transferable DX2 entropy floors. Do not re-open these.
- `ddm_jx1_joint_exchange_envelope_20260822.md` — the measured envelope TERMINATES at DX2; no
  composition of measured moves reaches 0.12, residual is the full 42,382 B. JX1 independently
  flagged RC1's shadow container and refused it credit for exactly the gap this arm closes:
  no shipping full-RGB receiver, no measured Δd_seg/Δd_pose. Do not repeat that omission.
- `ddm_vf1_evaluator_visible_floor_20260822.md` — **0 of 117,964,800 token positions carry
  qualifying current-DX2 evidence**; the load-bearing/inert census is entirely UNMEASURED and
  the quotient prediction is INCONCLUSIVE. Consequence for you: there is NO retained token-level
  sensitivity corpus to lean on. Measure what you claim; do not infer distortion from agreement.
- `ddm_db1_decode_boundary_families_20260822.md` — DB1 self-reported an ALWAYS-KEEP-THE-PAYLOAD
  violation during development (a partial transcript discarded after a decoder failure, then
  reconstructed and retained) and cured it structurally with fsynced per-group checkpoints and
  durable receipts. A live arm broke a DEF-CON-1000 rule today; inherit the cure, not the defect.
- Token agreement ≠ evaluator evidence. This is RC1's own closing dead-end and the single
  easiest way for this arm to produce a fake result.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_rc1_rate_crush_20260822.md`,
  sha dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d — it built the largest
  byte win of the campaign and REFUSED to call it a score, naming its own blocking leg. Match
  that bar: the measurement decides, and a byte number never wears score notation.
- Provenance pins (verify each at start; refuse if the tree drifted):
  `.omx/research/ddm_rc1_rate_crush_20260822.md`=dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d
  `.omx/research/ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd
  `.omx/research/ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4
  `.omx/research/ddm_db1_decode_boundary_families_20260822.md`=08fd9c4b5d4e583293c3977a8a98abb0205b0a0fc0443e67bd5247aed2de86af
- VERIFIED ARITHMETIC (check once, then use): pointer DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.120100 · seg 0.020139 · pose 0.007981. Distortion 0.028120 →
  S<0.12 needs archive ≤ **137,986 B** (STRICT inequality ⇒ FLOOR of 137,986.88, not round-up)
  → shed **42,382 B**. Exchange rate 0.001 S distortion = 1,502 B. RC1's 113,006 B sits
  24,980 B under that ceiling — which is exactly why its distortion is the whole question.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — integrating into a
  non-shipping receiver, or reporting agreement/arithmetic in place of measured distortion,
  is the fake this charter exists to refuse.
- **PRIOR-LAW PREDICTION (falsifiable):** the rare-class-collapse law (Lane is 0.59% of area,
  IoU 0.263 in GT, ~19% of flips) predicts RC1's 0.146 class-1 IoU produces a d_seg increase
  that overwhelms the 24,980 B of headroom — i.e. the candidate is byte-feasible and
  distortion-infeasible. **FALSIFIER:** measured n600 d_seg on the integrated receiver lands
  within the 24,980 B headroom's S-equivalent (1.663e-2 S at 6.658e-7 S/B) — if it does, RC1 is
  a live sub-0.12 candidate and MAIN must be told in the memo's FIRST line. Report either
  outcome plainly; a refutation here would be the campaign's biggest result to date.

## DELIVERABLE

`.omx/research/ddm_ri1_rc1_full_rgb_receiver_20260822.md` — the integration (shipping-receiver
path named and verified) + measured n600 d_seg WITH per-class breakout and Lane on its own row
+ measured d_pose + exact archive bytes + recomputed-from-components S if score-worthy + repeats
and mutation controls retained + the sealed MAIN fire-order or the honest closure + the explicit
verdict on the prior-law prediction with verdict_scope at the NARROWEST level the evidence
supports. Commit via the serializer. End with the own-vehicle frontier line.
