# ddm_jf1_joint_field_model_refit — five arms measured the OFF-DIAGONAL cells of a 2×2 and named the result a local optimum; the DIAGONAL (field AND model move together) is the one cell none of them entered

## MANDATE

Routed finding, five memos, no operator verbatim.

**(1) The five-arm law, stated exactly.** The shipped 19-member HPAC/RC64 model and the dx2 token
field sit at a joint local optimum that is SHARP in every direction any arm this wave tested:

| arm | what MOVED | what was HELD | result |
|---|---|---|---|
| `ddm_ad2` | input ORDER | model + field | reorder-favourable +34.5%, but only as a context-model SUBSTITUTE |
| `ddm_to2` | input ORDER | model + field | 196–687% WORSE |
| `ddm_ef1` | the ESTIMATOR | field | generic estimators 3.21× worse (365,322 B) |
| `ddm_oe1` | model (ADD one member) | field | all rungs larger; best +10,818 B, selectivity 0.175524 |
| `ddm_ld1` | field (lossy Lane coarsen) | **model UNCHANGED** (its charter says so verbatim) | all six rungs LARGER (+196 … +1,528 B) |

**(2) The structural reading nobody wrote down.** Those five rows populate exactly two cells of a
2×2 — {field fixed × model moved} and {field moved × model fixed}. **The diagonal — field moved AND
model refit to the moved field — is not among them.** And the sharp-optimum law is precisely the
reason the diagonal is the only cell that could win: the shipped model was FIT to THIS field, so
perturbing the field while freezing the model is *guaranteed* to raise cost. LD1's +196 … +1,528 B
is not evidence that the field is incompressible; it is the arithmetic consequence of pricing a
changed field under a stale model. **A coarsened field has a lower entropy under ITS OWN refit model
than the original field has under the original model — or it does not. **No memo among the five named
above reports that measurement; each names the axis it FROZE.**

**(3) Why it is worth an arm now.** The token stream is 113,777 B and the demand is 42,382 B. The
coding level is closed AT FIXED FIELD; the token allocation is closed AT FIXED MODEL (`ddm_ld1` +
`ddm_lx2`). Both closures share the same freeze. If the diagonal also loses, then the closure is
complete and unconditional and the campaign routes to a body change (`ddm_nr1_taskcell_quotient_prebuild_20260822.md` (task-cell body rebase)) with no ambiguity
left. If it wins, it is the only measured route to the demand on this body. **Both outcomes are
decisive; there is no empty result.**

## SCOPE

1. **Verify pins; reproduce the incumbent; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token
   stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B · TO2
   decoded field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. Re-encode
   the UNMODIFIED field under the SHIPPED model and reproduce 113,777 B exactly before perturbing
   anything; a disagreement IS the finding. Do NOT re-derive the exchange rate —
   `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived `25/37,545,489 = 6.658590e-07`; cite and use it.
2. **Establish the REFIT instrument and prove it on the null.** Existing HPAC fitting surfaces:
   `tools/train_ddm_cl1_hpac_capacity.py`, `tools/fit_ddm_cl1_hpac_capacity.py`,
   `tools/pr86_hpac_codec.py`. **Positive control, mandatory and reported: refit the model on the
   UNMODIFIED field and re-encode.** If the refit reproduces ≤ 113,777 B the instrument is
   trustworthy; if the refit of the incumbent's own field lands ABOVE 113,777 B, the instrument is
   weaker than the shipped fit and every subsequent row is bounded by that deficit — **say so and
   report the deficit as the arm's first number.** No diagonal row is admissible without this control.
3. **Run the DIAGONAL, ≥4 rungs, and price the MODEL BYTES.** Each rung takes an LD1-class field
   coarsening (reuse LD1's own rungs where the retained payloads permit — cite, do not re-derive)
   and **refits the 19-member model to the coarsened field**, then re-encodes. Per rung report:
   **stream bytes (real re-encode, never an estimate) · model bytes (the refit model is itself
   stored — its size may change and MUST be counted) · total archive delta · realized d_seg per
   class with Lane on its own row · realized d_pose · net ΔS.** The comparison bar is the shipped
   113,777 B stream + 13,515 B model, not the stream alone.
4. **Report the DECOMPOSITION at every rung: how much of the win (or loss) is the refit?** Each
   rung needs three numbers, not one: bytes(coarsened field @ shipped model) — that is LD1's
   measurement — bytes(coarsened field @ refit model), and their difference. **That difference IS
   the refit's contribution and it is the quantity this arm exists to produce.** A rung reporting
   only a final total has not measured the mechanism.
5. **Adjudicate honestly, including the empty outcome.** If every diagonal rung is net-positive, say
   so plainly: the joint local optimum is sharp in the diagonal too, the five-arm law becomes a
   six-arm law with no remaining untested direction, and the token axis on this body is closed
   unconditionally. **Build NO shipping candidate here**; the diagonal table is the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal
  fires (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every refit model, every coarsened field, every re-encoded stream and
  every per-class argmax field persists with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/` — BOTH
  SSD TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister arms
  at rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination
  per the disk rule while the tiers are full.** Do NOT write to `/Volumes/*` — a write there will
  kill you. Say which tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Every d_seg/d_pose states its GT lineage (DALI-GT where the tool family expects it; a PyAV-lineage
  GT on the pose axis is a measured wrong-objective defect).
- File ownership: LD1 owns the Lane rate curve · BL1 the per-position cost field · AD2/TO2 the
  ordering rows · EF1 the estimator race · OE1 the escape member · AR1B the residue census.
  CITE them; do not duplicate or touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` (`5e8d6011ba`) — **the direct parent.** Six lossy
  Lane rungs, ALL larger; the unedited control is the rate optimum **under the shipped model**. Its
  charter froze the model verbatim. This arm changes exactly that one thing. Do NOT re-run LD1's
  rungs at fixed model and call it new.
- `ddm_oe1_online_escape_member_*` — ADDING a member: all rungs larger, best +10,818 B, selectivity
  0.175524 vs a 1.5 bar; adaptation-zero control byte-identical (instrument trustworthy). With
  `ddm_ae1`'s two static kills the anti-predicted family is CLOSED whole. **Do NOT propose a new
  mixture member; refit the members that exist.**
- `ddm_to2` (orderings 196–687% worse) · `ddm_ef1` (generic estimators 3.21× worse) · `ddm_xs1_cross_section_joint_coding_20260818.md` (coder axis, all four sections) ·
  `ddm_mz2_frozen_section_representation_attack_20260815.md` (`5c073e915`, 38/38 semantic tensors receiver-required, exact re-encodings ALL
  +340 B) — **the coder-swap and storage-layout axes are CLOSED.** Do NOT swap the coder family.
- `ddm_ad2_addressing_cost_decomposition_20260822.md` — reorder-favourable +34.5% **only as a
  context-model substitute**. Read as a warning in this arm's direction: the model's members carry
  real, non-redundant information, so pruning-by-assumption is exactly the wrong move. This arm
  REFITS; it does not assume members are surplus (that question is `ddm_mp3`'s, concurrently).
- `ddm_ri1` + `ddm_ni1` — whole-body lossy re-representations DEAD on distortion (43.66× and 247.71×
> **[MAIN CORRECTION 2026-08-22, SUPERSEDES THE 08-22 ERRATUM: the `247.71x` figure is CONFIRMED, now MEASURED on contest-CUDA n600 (call fc-01M0PF62QK…, S 27.8, d_seg 0.07583781 = 247.69x NI1's own ceiling and 376.6x DX2, d_pose 40.53). NI1 is byte-feasible at 122,250 B and DISTORTION-DEAD. Its 98.6786% token agreement understated d_seg by 349x — do NOT use token agreement as an evaluator. RI1 43.66x also real+MEASURED. The whole-body lossy re-representation family is CLOSED on two authority rows. See `.omx/research/ddm_ni1_247x_erratum_20260822.md` (retraction section at the end).]**
  over ceiling), amplification exponent **16.69**. **Do NOT interpolate d_seg between rungs —
  measure each.** Note the distinction: those changed 1.5M+ tokens with no refit; this arm's rungs
  are the LD1-class Lane-scoped coarsenings WITH refit.
- SD1M (`ddm_mz2_frozen_section_representation_attack_20260815.md` lineage, memo §5 commit `c30f92fbc9`) — the weight-MSE proxy defect: "dead" tensors were dead in weight-space only, render
  amplification ~38,700×. **A modelled entropy is not a measured re-encode; a weight-space proxy is
  not a distortion measurement.**

## OPTIMAL FORM

- **REFERENCE FORM: the shipped DX2 receiver and the shipped 19-member HPAC/RC64 family, with the
  model REFIT (same family, same member count, same architecture — only the fitted parameters move)
  to each coarsened field.** A refit is legal SCOPE; changing the member count, the coder family, or
  the architecture is a MECHANISM change and belongs to a different arm (`ddm_mp3` owns member
  count). Declare any fitting-budget reduction as a SCOPE reduction with its own row.
- Family exemplar for conduct: `ddm_bl1_per_position_bit_allocation_20260822.md`, commit
  **`873947c665`** — it reconciled its instrument to the physical stream, explained its 56-bit
  residual instead of absorbing it, refused to call its allocation a bound, and reported its MS9 join
  in BOTH directions including the one that weakened its own story. SCOPE 2's positive control is
  this arm's version of that discipline: **report the refit deficit even when it damages the story.**
- SCOPE reductions declared per row (a strided pilot to order the rungs before full measurement is
  legal and must be labelled; the verdict is n600). MECHANISM reductions FORBIDDEN.
- VERIFIED ARITHMETIC (MAIN re-derived): archive 180,368 B · token stream 113,777 B · HPAC model
  13,515 B · residue 66,591 B. DX2 S 0.14821987563243377 · rate 25·180368/37545489 = 0.1200996 ·
  seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B
  → shed **42,382 B**; **6.658590e-07 S/B**; **1.2731082153 B/flip**.
- **PRIOR-LAW PREDICTION (falsifiable):** LD1's losses are an artifact of the frozen model, not a
  property of the field. **At least one diagonal rung lands BELOW its own fixed-model counterpart by
  >2,000 B** (i.e. the refit contribution in SCOPE 4 is negative and material), and at least one
  rung's TOTAL (stream + model) lands below the shipped 127,292 B combined baseline.
  **FALSIFIER:** every diagonal rung's refit contribution is ≥0, or every rung's total exceeds the
  shipped baseline. Then the joint optimum is sharp in the diagonal as well; the six-arm law closes
  the token axis unconditionally, and — composed with AR1B/AP1 on the residue — **no allocation
  change on this body can meet the demand**, routing the campaign to a body change (`ddm_nr1_taskcell_quotient_prebuild_20260822.md` (task-cell body rebase)).
  **Count it plainly if it lands; both outcomes route the campaign.**

## DELIVERABLE

`.omx/research/ddm_jf1_joint_field_model_refit_20260823.md` — the SCOPE-2 positive control (refit on
the unmodified field, with its deficit stated) + the ≥4-rung DIAGONAL table with, per rung: **stream
bytes (real re-encode) · model bytes · total vs the 127,292 B shipped baseline · the three-number
refit decomposition · realized d_seg per class (Lane on its own row) · realized d_pose · net ΔS** +
the located optimum or the honest all-positive verdict + its share of the 42,382 B demand +
verdict_scope at the NARROWEST level the evidence supports. Every figure carries its denominator and
its GT lineage. No shipping candidate. Commit via the serializer. End with the own-vehicle frontier line.
