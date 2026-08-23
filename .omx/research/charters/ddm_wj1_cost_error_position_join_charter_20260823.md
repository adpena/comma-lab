# ddm_wj1_cost_error_position_join — BL1 knows what every position COSTS and MST1 knows which positions the render gets WRONG ANYWAY; neither memo reports a join of the two, and their intersection is the exact set where we pay maximum rate for zero distortion return

## MANDATE

Routed finding, three memos, no operator verbatim. **This arm runs no scorer and needs no training:
it JOINS two already-measured per-position fields.**

**(1) The cost field exists.** `ddm_bl1_per_position_bit_allocation_20260822.md` (commit
`873947c665`, reconciled to the physical stream: 910,209.281 modeled vs 910,216 bits, 56-bit decoder
lookahead into defined zero fill) attributed the RC64 stream's **910,216 physical bits** over
**117,964,800 positions**. Concentration is extreme: **top 1% of positions = 96.323842% of bits,
Gini 0.995159.** BL1 published the CLASS-level join to seg error and found it ASYMMETRIC — seg
errors are 2.01% of the expensive set and carry 5.27% of bits (≤5,991 B) — concluding the shared
object is the CLASS, not position identity. **That join used FINAL seg errors. It is not the join
this charter asks for.**

**(2) The manufactured-error field exists, and it is a DIFFERENT, larger set.**
`ddm_mst1_manufactured_stage_split_20260822.md` (`1c33f278920b91bf922e9620deb9ce20615135e8`,
`verdict_scope: INSTANCE:DX2_T4_n600_WITH_MACOS_CPU_INTERMEDIATE_OBSERVATIONS`) split the
realization path by stage and measured the **native render + frozen SegNet head manufacturing
+22,321 errors** on top of the 9,182 transmitted-label errors — while R (−6,980) and uint8 (−771)
are net **REPAIRERS**. Final errors 23,757. **The 22,321 render-manufactured positions are ~2.4× the
transmitted-error set, and MST1's memo reports no join of them to any cost field.**

**(3) The question, and why it is well-posed.** A position whose rendered output the scorer reads
WRONG regardless is a position whose transmitted precision the realization path is not honoring. If
those positions are also the expensive ones, we are paying maximum rate for zero distortion return —
and that mass is the campaign's cheapest possible coarsening target. The prior is that they DO
coincide: expensive positions are surprising positions, surprising positions sit on boundaries, and
boundaries are exactly where the render manufactures error. **If the prior holds the waste set is
large; if it fails, cost and manufactured error are independent and the "pay-for-nothing" hypothesis
dies on measured evidence.** Either way this is a few hours of joining retained fields.

**(4) What this arm is NOT.** It is not an actuator. `ddm_ld1` measured that coarsening the field
under the SHIPPED model makes the archive LARGER at all six rungs — so a waste set cannot be
harvested by dropping bytes at fixed model. **This arm produces the TARGET SET; `ddm_jf1` (running
concurrently) owns the mechanism that could act on it (field coarsening WITH model refit).** Say so
plainly in the verdict; do not claim a byte win this arm cannot realize.

## SCOPE

1. **Verify pins; recover BOTH fields read-only; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token
   stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B · TO2
   decoded field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. Recover
   BL1's per-position cost field and MST1's per-stage error fields from their retained receipts
   (`.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local`, BL1's committed
   receipts). **Reproduce BL1's 910,216-bit total and MST1's 9,182 / 31,503 / 23,757 stage counts
   before joining; a disagreement IS the finding.** If either field was not retained per-position,
   say so precisely and REGENERATE it from its own committed tool rather than substituting an
   aggregate — **an aggregate cannot be joined and a join of aggregates is not this measurement.**
   Do NOT re-derive the exchange rate — `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived
   `25/37,545,489 = 6.658590e-07`; cite and use it.
2. **THE JOIN — the arm's whole job.** Build the per-position contingency over 117,964,800 positions:
   {expensive, cheap} × {render-manufactured-wrong, render-correct}, with "expensive" swept across
   ≥4 thresholds (top 0.1% / 1% / 5% / 10% by bits) rather than fixed at one. Per cell report:
   **position count · total bits · bytes · share of the 113,777 B stream · share of the 42,382 B
   demand.** Report the association strength with its denominator stated, and report the
   independence baseline (what the cell would hold if cost and manufactured error were independent)
   next to the observed value. **A correlation coefficient without the expected-under-independence
   mass is not a result; the BYTE MASS is the deliverable, not the statistic.**
3. **Decompose the join PER CLASS with Lane on its own row.** Lane is 0.5856% of area carrying
   33.56% of model bits (38,183 B, 57.31× the mean) and simultaneously the worst distortion class
   (IoU 0.263, ~19% of flips). **If the waste set is Lane-concentrated, that is a different campaign
   fact than if it is spread** — BL1's class-level asymmetry finding makes this split mandatory, not
   optional.
4. **Split the manufactured set by MST1's own stage attribution.** MST1 measured R and uint8 as net
   REPAIRERS (−6,980, −771). Positions the render breaks AND the realization path later REPAIRS are
   a distinct cell from positions that stay broken to the final argmax. **The repaired cell is where
   transmitted precision is most plausibly wasted** — it is spent, damaged, and then restored by a
   free downstream operation. Price that cell separately; it is the sharpest form of the hypothesis.
5. **Adjudicate honestly, including the empty outcome, and hand off.** State the waste-set byte mass
   and its share of the demand. If cost and manufactured error are independent-or-anticorrelated,
   say so plainly — that kills the pay-for-nothing hypothesis on measured evidence and tells JF1 to
   coarsen by cost alone rather than by cost×futility. **Emit the target set as a machine-readable
   position list with sha256 so JF1 can consume it directly. Build NO shipping candidate and claim
   NO byte win** — LD1 proved this set cannot be harvested at fixed model.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal
  fires (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; the recovered cost field, both error fields, the contingency tables and
  the emitted target-set position list all persist with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/` — BOTH
  SSD TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister arms
  at rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination
  per the disk rule while the tiers are full.** Do NOT write to `/Volumes/*` — a write there will
  kill you. Say which tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Any d_seg/d_pose that IS quoted states its GT lineage (DALI-GT where the tool family expects it;
  MST1's stage split carries macOS-CPU intermediate observations — **cite it at that strength, not
  stronger: the stage ORDERING is robust, the exact shares are advisory-lineage**).
- File ownership: BL1 owns the cost field · MST1 the stage split · MS9 the manufactured fraction ·
  LD1 the Lane rate curve · AR1B the residue census · **JF1 owns the refit mechanism and is running
  concurrently — this arm feeds it and does not duplicate it.** CITE them; do not touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` (`5e8d6011ba`) — six lossy Lane rungs, ALL larger
  (+196 … +1,528 B) under the shipped model. **This is why this arm claims no byte win.** The target
  set is an INPUT to JF1's refit, not a lever on its own.
- `ddm_bl1_per_position_bit_allocation_20260822.md` (`873947c665`) — its CLASS-level join found the
  seg×rate relation ASYMMETRIC and concluded the shared object is the CLASS, not position identity.
  **That is a prior AGAINST this arm's hypothesis at position granularity, drawn on the FINAL error
  set.** State it, and if this arm's manufactured-set join reproduces the same weak association,
  report the concordance as strengthening BL1 rather than dressing it as a new finding.
- `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` — 90.4702% of DX2's seg error manufactured
  downstream; only 2,264 of 23,757 are representation errors that survive. The 24.66% survival ratio
  is a BODY-WIDE baseline; **do not assume it holds per-cell — measure the per-cell survival in
  SCOPE 4 or report it as unmeasured.**
- `ddm_to2` (orderings 196–687% worse) · `ddm_ef1` (generic estimators 3.21× worse) · `ddm_xs1_cross_section_joint_coding_20260818.md` (coder axis, all four sections) ·
  `ddm_mz2_frozen_section_representation_attack_20260815.md` (`5c073e915`, section-coding) — the lossless coding axis is CLOSED. **Do NOT propose a coder change, a reordering, or a
  re-encoding.** This arm perturbs nothing.
- `ddm_ri1` + `ddm_ni1` — amplification exponent **16.69**; token disagreement ×1.0975 → d_seg
  ×4.7242. **Do NOT convert a position count into a predicted d_seg** — this arm reports MASS and
  MEMBERSHIP, and any distortion consequence belongs to the arm that measures it.
- SD1M (`ddm_mz2_frozen_section_representation_attack_20260815.md` lineage, memo §5 commit `c30f92fbc9`) — "dead in a proxy space" is not "dead in reality" (render amplification ~38,700×).
  **A high-cost position is not thereby a droppable position; the join produces a CANDIDATE set, and
  candidacy is not a verdict.**

## OPTIMAL FORM

- **REFERENCE FORM: both fields at their own native per-position granularity over all 117,964,800
  positions and all n600 pairs, joined without aggregation.** Aggregating either field before the
  join is a MECHANISM reduction that destroys the measurement (the whole question is a per-position
  coincidence) and is FORBIDDEN. Sweeping the "expensive" threshold is REQUIRED, not a reduction —
  a single fixed threshold would make the result an artifact of that choice.
- Family exemplar for conduct: `ddm_bl1_per_position_bit_allocation_20260822.md`, commit
  **`873947c665`** — it reconciled its instrument to the physical stream, explained its 56-bit
  residual instead of absorbing it, refused to call its allocation a bound, and reported its MS9
  join in BOTH directions **including the direction that weakened its own story.** This arm inherits
  BL1's field AND its prior-against; match that conduct exactly.
- SCOPE reductions declared per row (a strided pilot over a pair subset to shape the thresholds is
  legal and must be labelled; the verdict is n600 over all positions). MECHANISM reductions
  FORBIDDEN.
- VERIFIED ARITHMETIC (MAIN re-derived): archive 180,368 B · token stream 113,777 B = 910,216
  physical bits over 117,964,800 positions · HPAC model 13,515 B · residue 66,591 B. DX2
  S 0.14821987563243377 · rate 25·180368/37545489 = 0.1200996 · seg 100·(23,757/117,964,800) =
  0.020139 · pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B → shed **42,382 B**;
  **6.658590e-07 S/B**; **1.2731082153 B/flip**. MST1 stage deltas: render **+22,321** · R **−6,980**
  · uint8 **−771** · device **+5**.
- **PRIOR-LAW PREDICTION (falsifiable):** expensive positions and render-manufactured errors coincide
  because both localize to class boundaries. **The top-1% cost set (96.32% of all bits) contains
  render-manufactured errors at ≥2× the rate expected under independence, and the joint cell holds
  >5,000 B (>11.8% of the demand)** — with the SCOPE-4 repaired sub-cell nonempty and separately
  priced.
  **FALSIFIER:** the observed joint mass is within 1.25× of the independence baseline at every
  threshold, or the joint cell holds <1,000 B. Then cost and realization futility are effectively
  independent, the pay-for-nothing hypothesis is dead at position granularity, BL1's class-level
  conclusion is CONFIRMED at a finer grain on a different error set, and JF1 should coarsen by cost
  alone. **Count it plainly if it lands; both outcomes route JF1.**

## DELIVERABLE

`.omx/research/ddm_wj1_cost_error_position_join_20260823.md` — BL1's 910,216-bit total and MST1's
stage counts reproduced + the ≥4-threshold contingency with, per cell: **positions · bits · bytes ·
share of stream · share of the 42,382 B demand · observed-vs-independence mass** + the per-class
split with **Lane on its own row** + the SCOPE-4 repaired-vs-persistent split + the emitted
machine-readable target-set position list with sha256 for JF1, OR the honest independence verdict +
verdict_scope at the NARROWEST level the evidence supports. Every figure carries its denominator and
its lineage. No shipping candidate, no byte-win claim. Commit via the serializer. End with the
own-vehicle frontier line.
