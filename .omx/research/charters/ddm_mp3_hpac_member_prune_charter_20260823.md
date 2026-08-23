# ddm_mp3_hpac_member_prune — the 19-member model costs 13,515 B (31.89% of the demand) and buys 251,545 B at an 18.6× AVERAGE return; oe1 measured ADDING a member, no arm measured REMOVING one, and fs3's law says the margin is nothing like the average

## MANDATE

Routed finding, three memos, no operator verbatim.

**(1) The object and its price.** `ddm_ar1b_archive_residue_purchase_20260822.md`
(`verdict_scope: INSTANCE:DX2_PHYSICAL_CENSUS_ONLY`, every span SHA-custodied, zero remainder)
measured the HPAC probability model at **13,515 B = 20.2976% of the 66,591 B residue = 31.8885% of
the entire 42,382 B sub-0.12 demand.** It is the third-largest object in the archive.

**(2) What it buys, on AVERAGE.** `ddm_ef1_token_entropy_floor_20260822.md` measured the best
generic estimator at **365,322 B** against the shipped stream's **113,777 B** — the learned
19-member law is worth **251,545 B**, an **18.6× average return** on its 13,515 B. That number
justifies the model's existence. **It says nothing about the MARGINAL member, and marginal is what
a prune decision is priced on.**

**(3) The untested direction.** `ddm_oe1_online_escape_member_*` measured ADDING a zero-stored causal
member: all rungs LARGER, best +10,818 B, selectivity 0.175524 against a 1.5 bar, with a byte-identical
adaptation-zero positive control. That is a decisive result about **addition**. `ddm_ad2` measured
reordering as a context-model SUBSTITUTE (+34.5%), which says the members collectively carry real
information. **Neither measures what the WEAKEST member earns.** No arm in this wave removed a member
and re-priced model bytes against stream bytes jointly; the five-arm "you cannot subtract" reading is
an inference from the addition result, not a measurement of subtraction.

**(4) Why the marginal is the live question — and why the inference may invert.**
`ddm_fs3` measured, on a different admitted-set surface, that the **marginal member cost 2.24× the
set's average price at 3.98× less yield** — greedy-admitted sets degrade at the margin in BOTH
directions. If this model was fit greedily (its member count is a round 19), the last members
admitted plausibly earn far less than 251,545/19 ≈ 13,239 B each while still costing their share of
the 13,515 B. A member earning less than its own storage is pure loss and its removal is free bytes.
**This is an MDL question — is 19 the code-length-optimal member count? — and it is open.**

## SCOPE

1. **Verify pins; reproduce the incumbent; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token
   stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B.
   Reproduce AR1B's 13,515 B model span and the 113,777 B stream exactly before perturbing anything;
   a disagreement IS the finding. Do NOT re-derive the exchange rate —
   `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived `25/37,545,489 = 6.658590e-07`; cite and use it.
2. **Decompose the 13,515 B PER MEMBER.** First deliverable: the model's byte census by member —
   how many of the 13,515 B each of the 19 members owns, summing exactly with no unexplained
   remainder. **A remainder IS a finding; report it rather than absorbing it.** If members share
   parameters so the split is not clean, say so precisely and price the SEPARABLE portion.
3. **Leave-one-out, all 19, real re-encode.** For each member: remove it, **refit the remaining 18
   to the SAME unmodified field** (a prune without a refit prices a crippled model, not a smaller
   one), re-encode the real stream. Per member report: **model bytes saved · stream bytes added ·
   net archive delta · net ΔS at 6.658590e-07 S/B.** Distortion is unchanged BY CONSTRUCTION here —
   the decoded field is bit-identical — so **state that identity and PROVE it per row with a field
   sha comparison.** A row that cannot show field bit-identity is not a lossless prune and must be
   priced with a real d_seg/d_pose measurement instead.
4. **Then the MDL ladder: prune the k weakest jointly, k = 1…, refitting each time.** Greedy removal
   is not the same as removing the k individually-weakest — measure the ladder, do not extrapolate
   it. Stop when the net turns positive and report the code-length-optimal member count k*.
5. **Adjudicate honestly, including the empty outcome.** If every single-member removal is
   net-positive, say so plainly: 19 is at or below the MDL-optimal count, the model is fully earning
   its 13,515 B, and 31.89% of the demand is closed on measured evidence. That result also
   **inverts into a live question this arm must state but not answer: if every member earns its
   keep, would a TWENTIETH member of a different class earn its keep too?** — noting `ddm_oe1`
   already refuted the causal-escape class specifically. **Build NO shipping candidate here**; the
   per-member table is the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal
  fires (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every pruned model, every re-encoded stream, every decoded-field sha
  persists with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mp3_hpac_member_prune/` — BOTH SSD
  TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister arms at
  rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination per
  the disk rule while the tiers are full.** Do NOT write to `/Volumes/*` — a write there will kill
  you. Say which tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Any d_seg/d_pose that IS measured states its GT lineage (DALI-GT where the tool family expects it).
- File ownership: AR1B owns the residue census · EF1 the estimator floor · OE1 the escape member ·
  AD2 the ordering decomposition · BL1 the per-position cost field · **JF1 owns the model REFIT on a
  CHANGED field and is running concurrently — this arm refits on the UNCHANGED field only.** CITE
  them; do not duplicate or touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_oe1_online_escape_member_*` — ADDITION is closed: all rungs larger, best +10,818 B,
  selectivity 0.175524 vs a 1.5 bar; adaptation-zero control byte-identical. **Do NOT propose a new
  member.** Its instrument and its control design are the model to copy.
- `ddm_ef1_token_entropy_floor_20260822.md` — generic estimators 3.21× worse (365,322 B), FAMILY-
  scoped. This is the SOURCE of the 18.6× average-return figure and simultaneously the warning
  against replacing the learned law wholesale. **Do NOT swap the estimator family.**
- `ddm_ad2_addressing_cost_decomposition_20260822.md` — reorder-favourable +34.5% **only as a
  context-model substitute**: the members collectively carry non-redundant information. Read as
  evidence AGAINST an easy prune; the prediction below is deliberately stated against this grain and
  must survive it.
- `ddm_to2` (orderings 196–687% worse) · `ddm_xs1_cross_section_joint_coding_20260818.md` (coder axis, all four sections) · `ddm_mz2_frozen_section_representation_attack_20260815.md` (`5c073e915`, section-coding) — the coder-swap,
  reorder, and storage-layout axes are CLOSED. **Do NOT propose a re-encoding or a layout change.**
- SD1M (`ddm_mz2_frozen_section_representation_attack_20260815.md` lineage, memo §5 commit `c30f92fbc9`) — the analogous defect in the analogous shape: parameters that looked droppable in a
  PROXY space (weight-MSE) were 90× underwater in reality once render amplification (~38,700×) was
  paid. **A member's modelled log-loss contribution is a PROXY; the admissible number is the real
  re-encoded stream delta.** Report modelled contributions only alongside the measured one.
- `ddm_ld1` — six lossy Lane rungs all LARGER under the shipped model; the unedited field is the
  rate optimum at fixed model. Relevant here as the reason SCOPE 3 holds the field fixed: this arm
  isolates the MODEL axis, JF1 owns the joint move.

## OPTIMAL FORM

- **REFERENCE FORM: the shipped 19-member HPAC/RC64 law fit to the shipped dx2 field, with member
  count as the ONLY variable and the surviving members REFIT at each rung.** Same family, same
  architecture, same field, same coder. A prune without refit is a MECHANISM reduction (it prices a
  damaged model, not a smaller one) and is FORBIDDEN as a verdict row — it is legal only as a
  labelled diagnostic alongside the refit row.
- Fitting surfaces already in-tree: `tools/train_ddm_cl1_hpac_capacity.py`,
  `tools/fit_ddm_cl1_hpac_capacity.py`, `tools/pr86_hpac_codec.py`. **Positive control, mandatory:
  refit all 19 on the unmodified field and reproduce ≤113,777 B. If the refit of the incumbent's own
  configuration lands above the shipped stream, every prune row is bounded by that deficit — state
  the deficit as the arm's first number.** No prune row is admissible without this control.
- Family exemplar for conduct: `ddm_ar1b_archive_residue_purchase_20260822.md` — it closed its census
  to zero remainder, custodied every span, and REFUSED to call a byte ranking a purchase attribution
  when it lacked the lane. SCOPE 2 is that same census discipline one level down.
- SCOPE reductions declared per row (a reduced fitting budget for the 19 LOO rungs, with the top
  candidates refit at full budget, is legal and must be labelled). MECHANISM reductions FORBIDDEN.
- VERIFIED ARITHMETIC (MAIN re-derived): archive 180,368 B · token stream 113,777 B · HPAC model
  13,515 B · residue 66,591 B. DX2 S 0.14821987563243377 · rate 25·180368/37545489 = 0.1200996 ·
  seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B
  → shed **42,382 B**; **6.658590e-07 S/B**. Model average return 251,545/13,515 = **18.61×**;
  naive per-member share **711 B** stored / **13,239 B** earned.
- **PRIOR-LAW PREDICTION (falsifiable):** the fs3 margin law transfers — the weakest members earn
  far below the 13,239 B average. **At least one single-member leave-one-out is net-negative
  (model bytes saved exceed stream bytes added), and the MDL ladder's k\* is ≥2**, freeing >1,000 B
  at ZERO distortion by construction.
  **FALSIFIER:** every one of the 19 leave-one-out rungs is net-positive. Then 19 is at or below the
  MDL-optimal count, the model earns its 13,515 B in full, and — composed with the addition kill
  (`ddm_oe1`) — the member-count axis is closed in BOTH directions, which upgrades the five-arm
  local-optimum law from an inference to a measurement. **Count it plainly if it lands; both
  outcomes route the campaign.**

## DELIVERABLE

`.omx/research/ddm_mp3_hpac_member_prune_20260823.md` — the per-member byte census summing to
13,515 B with no unexplained remainder + the mandatory refit-all-19 positive control with its
deficit + the 19-row leave-one-out table with, per member: **model bytes saved · stream bytes added
(real re-encode) · decoded-field bit-identity proof · net archive delta · net ΔS** + the MDL ladder
to k\* + the freed-byte total against the 42,382 B demand, OR the honest all-positive verdict +
verdict_scope at the NARROWEST level the evidence supports. Every figure carries its denominator. No
shipping candidate. Commit via the serializer. End with the own-vehicle frontier line.
