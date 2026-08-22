# ddm_lq1_lane_quotient_representability — Lane gets 8.97× its area share of the codebook and still collapses to 14.7%; if that is representational rather than budgetary, BOTH live candidates die

## MANDATE

CB2 (`ddm_cb2_class_balanced_dictionary_20260822.md`) refuted the reweighting hypothesis with a
measurement that inverts the whole framing:

| class | area | K=2,048 capacity | capacity/area | agreement given true class | share of 1,420,331 mismatches |
|---|---:|---:|---:|---:|---:|
| Road | 23.233107% | 44.672445% | 1.9228× | 98.909066% | 21.050797% |
| **Lane** | **0.585848%** | **5.252197%** | **8.9651×** | **14.729089%** | **41.490540%** |
| Movable | 1.238046% | 19.927002% | 16.0955× | 80.155540% | 20.405103% |
| Undrivable | 49.517502% | 26.790446% | 0.5410× | 99.681074% | 13.116309% |
| MyCar | 25.425497% | 3.357910% | 0.1321× | 99.813551% | 3.937251% |

The K=256→2,048 increment was *more* Lane-heavy still (5.974981% of added capacity). So capacity
was ALREADY allocated away from area and toward Lane, aggressively, and Lane still fails. **A
resource that is over-supplied 8.97× and still starving is not failing from scarcity.**

**Why this is now the campaign's decisive question, not an RC1 post-mortem.** RI1's advisory
measured RC1 at d_seg 0.01605413 against a 0.000367727 ceiling — 43.66× over. Lane is 41.49% of
mismatches, so even a PERFECT Lane (every Lane token correct, at zero byte cost) leaves
0.01605413 × (1 − 0.4149054) = 0.009394, still **25.5×** over. RC1 is dead in this family with or
without a Lane cure; that arithmetic is not the question. The question is whether the *mechanism*
that kills Lane is a property of QUOTIENT/DICTIONARY REPRESENTATION ITSELF — because NR1's K32
task-cell quotient is the same representational class, and if the answer is yes, both live
candidates are dead and the campaign needs a third representation, not another tuning of these two.

## SCOPE

1. **Verify inherited state, refuse on drift.** CB2's memo
   `.omx/research/ddm_cb2_class_balanced_dictionary_20260822.md` · RC1 payload sha
   `eab66bad…e61164` · `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/RESULT.json` ·
   modules `src/tac/optimization/rc1_terminal_program_vq.py`,
   `src/tac/optimization/nr1_taskcell_quotient.py`. Reproduce the five capacity/agreement rows
   above from the retained artifacts; any disagreement IS the finding, report it first.
2. **Diagnose WHY Lane fails at 8.97× capacity.** Distinguish at least these, with a measurement
   per hypothesis and its denominator (m50): (a) **intrinsic diversity** — Lane's 691,095 source
   positions occupy far more distinct temporal programs than its 64,539 slots can cover, so the
   quantizer is genuinely under-provisioned in program-space despite being over-provisioned in
   slot-share; (b) **assignment loss** — Lane programs ARE in the codebook but the nearest-centroid
   assignment routes Lane positions to non-Lane centroids (measure the confusion, not just the
   agreement); (c) **objective geometry** — the fitting distortion metric treats a Lane/Road swap
   as cheap because Lane's positions are few, so centroids drift off Lane regardless of slot count.
   These have different cures and only measurement separates them.
3. **Test representability, not tuning.** The load-bearing question is whether ANY member of the
   quotient/dictionary family can carry Lane at feasible bytes. Construct the honest upper bound:
   what agreement does Lane reach if you give it an ORACLE assignment (correct centroid chosen per
   position, ignoring the encoder) at the existing K? If oracle-assignment Lane is high, the defect
   is (b) assignment and is curable; if oracle-assignment Lane is ALSO low, the codebook does not
   CONTAIN Lane's programs and the defect is (a)/(c) — a representational limit. Say which.
4. **Cross-apply to NR1-K32 — the reason this arm exists.** NR1's K32 task-cell quotient is the
   same representational class on the same token field (98.6786% overall agreement,
   1,558,833 changed tokens, 122,250 B realized). Measure NR1-K32's PER-CLASS agreement with Lane
   on its own row, using CB2's exact method so the two are comparable. If Lane collapses there too,
   that is a FAMILY-level fact about quotient representations of this token field and it must be
   the memo's first line. NI1's advisory (MAIN-fired, in flight) will supply the scorer-side d_seg;
   do NOT wait on it and do NOT substitute agreement for it.
5. **Report what it would take, or say it is closed.** If Lane is representable but mis-assigned,
   name the cure and its byte cost. If it is not representable in this family at feasible K, say so
   with verdict_scope FAMILY and name what a third representation would have to do differently —
   that redirects the campaign and is a complete result.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO scorer runs. NO Metal fires (MAIN-fire-only). NO local
  advisory launches — the single n600 lane is occupied by MAIN's NI1 K32 advisory. This arm is
  SCORER-FREE by construction; that is a declared SCOPE reduction, not a mechanism reduction.
- **Token agreement is NOT d_seg.** CB2 labelled every agreement figure PROXY-NOT-SCORE and so must
  this arm. RC1's own dead-end: *"Overall token agreement cannot be promoted as evaluator evidence."*
  Reporting a per-class agreement improvement as a distortion improvement is the fake this charter
  refuses.
- The jo1 r9 run directory is SACRED. r9 is terminal by SELF-REFUSAL (`EXACT_DELTA_NONNEGATIVE`);
  there is no improved endpoint and nothing to wait on.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES; it does not re-cut
  or re-fit either live payload.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every confusion matrix, oracle-assignment field, and per-class
  decomposition persists with sha256 + bytes. Scalar-only artifacts while the fields exist in
  memory are forbidden AT THE TYPING MOMENT.
- Receipts to `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/`. **NOTE: the
  APDataStore tier is at ~11 GiB free.** Check before writing bulk; route to
  `/Volumes/VertigoDataTier/pact/` (59 GiB free) if the receipt set is large, and say which you used.
- File ownership: RI1 owns RC1's receiver/distortion memo · NI1 owns NR1-K32's · AD2 owns the
  addressing-vs-payload decomposition. Do not touch their memos or retained trees; CITE their rows.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_cb2_class_balanced_dictionary_20260822.md` — **the reweighting hypothesis is REFUTED**
  (`PRIOR_LAW_REFUTED_AT_STEP_2_NO_REFIT`, verdict_scope FORMULATION). Do NOT propose a class-balanced
  or flip-weighted refit; capacity already exceeds area share 8.97× for Lane and 16.10× for Movable.
  Any successor that reallocates capacity is re-running a refuted mechanism.
- `ddm_rc1_rate_crush_20260822.md`=dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d —
  **K=4,096 is byte-dead** (105,811 B payload / 158,933 B shadow). Raising K is not the answer and
  costs exactly the bytes the score needs. Fixed-RC64 coder/context races are CLOSED (88 B ceiling
  shipped). Cross-vehicle prior: shared-VQ K=64/K=256 measured **ANTI-PARETO at λ=1.0** — codebook
  size/composition changes have produced non-monotone quality before.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** No retained token-level
  sensitivity corpus exists. Do not assume any position class is inert; derive or measure.
- `ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd —
  UNION ≠ SUM OF LEGS, measured 3.705× in this campaign. Per-class mismatch shares do NOT compose
  additively into d_seg predictions; any composed figure is an UPPER BOUND, labelled on its face.
- `ddm_dc1s_sparse_grid_sweep_20260821.md` — a family CLOSED honestly at full n600 scope
  (388,326 B vs a 113,777 B member, all 190 groups negative). The bar for closing a family here is
  that same bar: measured, at scope, with the mechanism named.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_cb2_class_balanced_dictionary_20260822.md` — it stopped at its
  own step-2 gate, refused to refit on a refuted premise, gave every number an explicit denominator,
  declined to invent a per-class byte partition of a jointly-compressed stream, and labelled every
  agreement figure PROXY-NOT-SCORE. Match that bar exactly. This arm inherits its method and asks
  the next question.
- VERIFIED ARITHMETIC (check once, then use): pointer DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.120100 · seg 0.020139 · pose 0.007981 · distortion 0.028120 →
  S<0.12 needs archive ≤ **137,986 B** (STRICT ⇒ FLOOR of 137,986.8388) → shed **42,382 B**;
  6.658e-7 S/B. RC1 113,006 B ⇒ d_seg ceiling 0.000367727; MEASURED 0.01605413 = 43.66× over.
  Perfect-Lane bound: 0.01605413 × (1 − 0.4149054) = 0.009394 = 25.5× over. NR1-K32 realized
  122,250 B ⇒ ceiling 0.000306175 (1.52× DX2), scorer measurement in flight.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — an oracle-assignment bound
  computed on a subset while reported as the population bound, or a representability claim made
  without actually testing whether the codebook CONTAINS Lane's programs, are the two fakes here.
- **PRIOR-LAW PREDICTION (falsifiable):** Lane's failure is REPRESENTATIONAL, not assignment —
  oracle-assignment Lane agreement at the existing K stays well below the other classes (say
  <60%), because Lane's thin, high-diversity temporal programs are not clustered in the codebook at
  all; and NR1-K32 shows the same Lane collapse, making this a FAMILY property of quotient
  representations over this token field.
  **FALSIFIER:** oracle-assignment Lane agreement lands high (>90%) ⇒ the codebook DOES contain
  Lane's programs and the defect is the encoder's assignment rule — a cheap, curable, byte-free
  target, and that would be the campaign's best news in days. Report either outcome plainly and
  put the decisive number in the FIRST line.

## DELIVERABLE

`.omx/research/ddm_lq1_lane_quotient_representability_20260822.md` — the per-hypothesis diagnosis
(intrinsic diversity vs assignment vs objective geometry, each with its measurement and
denominator) + the oracle-assignment upper bound with its exact construction + NR1-K32's per-class
agreement with Lane on its own row by CB2's method + the explicit verdict on the prior-law
prediction with verdict_scope at the NARROWEST level the evidence supports + either the named cure
with its byte cost or the FAMILY closure with what a third representation must do differently.
Every agreement figure labelled PROXY-NOT-SCORE. Commit via the serializer. End with the
own-vehicle frontier line.
