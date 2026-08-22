# ddm_cb2_class_balanced_dictionary — RC1's dictionary is weighted by AREA while the score is paid in FLIPS; Lane is under-weighted ~32× and that is the named cause of its 0.146 collapse

## MANDATE

Routed finding (`ddm_rc1_rate_crush_20260822.md`, LIVE-HYPOTHESIS 1, verbatim):
*"A class-balanced or boundary-debt temporal dictionary could preserve rare class-1/3 programs
far better at the same K, because the current population-weighted objective spends most
distortion on majority classes and the assignment stream is only 10,900 B at K=2,048."*

RC1 produced the campaign's largest byte win — K=2,048, counted payload 59,884 B, complete
shadow archive **113,006 B**, **24,980 B under the 137,986 B ceiling** — and refused to promote
it, naming class-1 IoU **0.146** as the blocking alarm. Class 1 is **Lane** (canonical comma10k
order 0=Road, 1=Lane, 2=Undrivable, 3=Movable, 4=MyCar).

**The mismatch is arithmetic.** Lane is **0.59% of pixel area** but carries **~19% of all d_seg
flips**. A population-weighted dictionary allocates codebook capacity by how OFTEN a program
appears; the evaluator pays for FLIPS. That is a ~32× disagreement between what the fit optimizes
and what the score charges. Road (23.2% area) and Undrivable (49.5%) dominate the objective and
starve the class that actually moves d_seg.

**The cure is byte-free, which is why it is worth doing now.** Re-weighting changes WHICH programs
the codebook retains, not HOW MANY: same K, same 48,920 B codebook, same 10,900 B assignment map,
same 113,006 B archive. It spends none of the 24,980 B headroom — headroom the campaign needs for
the score, not for fidelity. This arm is therefore strictly preferable to raising K.

## SCOPE

1. **Verify inherited state, refuse on drift.** Module `src/tac/optimization/rc1_terminal_program_vq.py`,
   materializer `experiments/ddm_rc1_rate_crush.py`, payload sha `eab66bad…e61164`, canonical
   result `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/RESULT.json`, section
   anatomy codebook 48,920 B / spatial map 10,900 B. Refuse if any drifted.
2. **CONFIRM the mechanism BEFORE re-fitting it.** Decompose the EXISTING K=2,048 fit's
   agreement and its codebook-capacity allocation PER CLASS. The claim under test is that
   capacity tracks AREA rather than flip-incidence. State the per-class numbers with their
   denominators (m50: a count without its denominator is not a measurement). If capacity does
   NOT track area, the whole premise is refuted at step 2 and the arm reports that and stops —
   which is a complete and valuable result, not a failure.
3. **Source the weighting field from RETAINED measurement, never invent it.** A flip-propensity
   or boundary-debt weight must come from an existing receipt joinable to RC1's program-site
   coordinates — candidates: the g4 spatial-stationarity per-pixel flip-frequency map (per
   stratum, across 600 pairs), the jg3 three-way {edit,drop,keep} n600 rows, and the
   token-by-token waterfill rows (768 cells × 4 rungs under joint remeasure, per-cell measured
   Δbytes/Δd_seg/Δd_pose; locate them by that content in `.omx/research/` — do NOT search for a
   bare harness task id, which does not resolve against the repo ledger). If NO retained field
   joins to RC1's program-site coordinates, say so plainly and name the exact join that is
   missing; do not synthesise a weight and do not fall back to a uniform class prior while
   calling it sensitivity.
4. **Re-fit at FIXED K=2,048 and hold the byte budget.** The re-fitted payload MUST come in at
   or below 113,006 B — report exact bytes per section. A re-weighting that grows the archive
   has changed the object and is not this experiment.
5. **Report agreement PER CLASS, and label it as a proxy — never as a score.** RC1's own closing
   dead-end binds: *"Overall token agreement cannot be promoted as evaluator evidence."* This
   arm is SCORER-FREE by construction (the single n600 lane is contended by two live arms), so
   its verdict on d_seg is OWED, not delivered. Emit a sealed fire-order naming the exact
   scorer run that would decide it; do NOT fire, and do NOT let a class-1 agreement improvement
   be written as a distortion improvement.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm. NO scorer runs. NO Metal fires
  (MAIN-fire-only). NO local advisory launches — this arm is scorer-free by design.
- The jo1 r9 run directory is SACRED. NOTE: r9 terminated BY SELF-REFUSAL (typed blocker
  `EXACT_DELTA_NONNEGATIVE`); there is no improved endpoint to wait on. Work from the current
  DX2 body and RC1's retained payload; do not gate on r9.
- Shipped receiver bytes are CUSTODY — integrate additively, never edit in place.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): the re-fitted codebook, assignment map, and every losing
  weighting variant persist with sha256 + bytes. A scalar-only artifact when payload bytes
  exist in memory is forbidden AT THE TYPING MOMENT, not at review.
- Receipts to `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/`.
- File ownership: RI1 owns integrating RC1's EXISTING payload into a shipping full-RGB receiver;
  NI1 owns NR1's K32 measurement. Do not touch their memos or retained trees. If RI1 lands
  first, CONSUME its per-class d_seg breakout as the calibration for your weighting field —
  a measured per-class flip cost beats any proxy you can build.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rc1_rate_crush_20260822.md`=dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d —
  **K=4,096 is byte-dead** (105,811 B payload / 158,933 B shadow); RC1 states this "closes that
  point, not the family." Do NOT re-open K-raising: it buys fidelity with the exact bytes the
  score needs. Also closed: fixed-RC64 coder/context races (88 B ceiling already shipped),
  PR130 memoryless bounds as DX2 floors, literal C1 plane storage.
- **Cross-vehicle VQ prior (honest caveat, different lineage):** an earlier shared-VQ-codebook
  measurement recorded **K=64 and K=256 ANTI-PARETO at λ=1.0** — codebook-size and codebook-
  composition changes have produced non-monotone quality before. Treat a "re-weighting must
  help" intuition as UNPROVEN; the class-conditional decomposition in scope item 2 is what
  makes this falsifiable rather than hopeful.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** There is NO retained
  token-level sensitivity corpus to lean on. This directly constrains scope item 3: verify your
  weighting field EXISTS and JOINS before designing around it.
- `ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd —
  the measured envelope terminates at DX2; UNION ≠ SUM OF LEGS (measured 3.705× in this
  campaign). Any composed figure built by summing legs is an UPPER BOUND and must be labelled
  so on its face.
- `ddm_db1_decode_boundary_families_20260822.md`=08fd9c4b5d4e583293c3977a8a98abb0205b0a0fc0443e67bd5247aed2de86af —
  a live arm self-reported an ALWAYS-KEEP-THE-PAYLOAD violation today (a transcript discarded
  after a decoder failure) and cured it with fsynced per-group checkpoints + durable
  success/failure receipts. Inherit the cure, not the defect.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_rc1_rate_crush_20260822.md`,
  sha dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d — it produced the
  campaign's largest byte win, refused to call it a score, and named its own blocking leg and
  its own best next hypothesis. Match that bar; this arm executes the hypothesis RC1 wrote down.
- VERIFIED ARITHMETIC (check once, then use): pointer DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.120100 · seg 100·0.00020139 = 0.020139 · pose √(10·6.37e-6) =
  0.007981. Distortion 0.028120 → S<0.12 needs archive ≤ **137,986 B** (STRICT inequality ⇒
  FLOOR of 137,986.88, not round-up) → shed **42,382 B**. Exchange rate 0.001 S distortion =
  1,502 B; 6.658e-7 S/B. RC1's 113,006 B leaves **24,980 B** = **0.016633 S** of distortion
  budget, i.e. a d_seg ceiling of **0.00036772** (1.83× the current 0.00020139) at fixed pose.
  Report any disagreement with your own recomputation as the finding.
- SCOPE reductions declared per row (scorer-free is a declared SCOPE reduction — say so on the
  verdict). MECHANISM reductions FORBIDDEN — inventing a sensitivity weight that no retained
  receipt supports, or reporting per-class agreement as if it were per-class d_seg, is the fake
  this charter exists to refuse.
- **PRIOR-LAW PREDICTION (falsifiable):** codebook capacity in the existing K=2,048 fit tracks
  CLASS AREA rather than flip-incidence, so Lane (0.59% area / ~19% of flips) holds a share of
  the 48,920 B codebook far below its flip share; re-weighting at fixed K raises class-1 program
  preservation materially without growing the archive. **FALSIFIER:** if the measured capacity
  allocation does NOT track area (scope item 2), or if re-weighting fails to raise class-1
  preservation, or if it raises class-1 only by degrading total flip-weighted agreement, the
  population-weighting hypothesis is REFUTED and RC1's Lane collapse needs a different cure —
  report that plainly, because it redirects the campaign's largest byte win.

## DELIVERABLE

`.omx/research/ddm_cb2_class_balanced_dictionary_20260822.md` — the per-class capacity-vs-area
decomposition of the EXISTING fit (with denominators) + the weighting-field provenance (retained
receipt named, or the missing join named) + the re-fitted payload at ≤113,006 B with exact
per-section bytes + per-class agreement labelled PROXY-NOT-SCORE + the sealed scorer fire-order
that would decide it + the explicit verdict on the prior-law prediction with verdict_scope at the
NARROWEST level the evidence supports. Commit via the serializer. End with the own-vehicle
frontier line.
