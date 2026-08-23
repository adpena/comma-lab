# ddm_ap1_residue_purchase_scorer — AR1B mapped the 66,591 B residue exactly and was REFUSED the scorer lane; the lane is now GRANTED, so measure what each class BUYS

## MANDATE

Routed finding, three memos, no operator verbatim. **This charter GRANTS the exclusive n600 scorer
lane that `ddm_ar1b_archive_residue_purchase` was correctly refused.** Its census is the input; its
withheld SCOPE-2 measurement is this arm's whole job.

**(1) The residue is mapped exactly.** `ddm_ar1b_archive_residue_purchase_20260822.md`
(`verdict_scope: INSTANCE:DX2_PHYSICAL_CENSUS_ONLY`, every span SHA-256 custodied, **no
unexplained remainder**):

| residue class | bytes | share of residue | share of the 42,382 B demand |
|---|---:|---:|---:|
| semantic renderer | 30,856 | 46.3378% | **72.8045%** |
| carrier | 22,010 | 33.0526% | **51.9324%** |
| HPAC probability model | 13,515 | 20.2976% | **31.8885%** |
| fixed residual table | 96 | 0.1442% | 0.2265% |
| ZIP + RX1 structural framing | 114 | 0.1712% | 0.2690% |
| **total** | **66,591** | **100.0000%** | **157.1209%** |

Three objects each individually exceed 30% of the whole demand. AR1B refused to call a byte census
a purchase attribution, and was right to.

**(2) The CODING level of this body is closed — five arms, one measured law.** The shipped 19-member
HPAC/RC64 law and this field sit at a joint local optimum that is SHARP in every tested direction:
`ad2` reorder-favourable +34.5% but only as a context-model SUBSTITUTE · `ddm_to2` other orderings
**196–687% worse** · `ddm_ef1` generic estimators **3.21× worse** (365,322 B) · `ddm_ld1` lossy
coarsening **all six rungs larger** · `ddm_oe1` ADDING a mixture member **all rungs larger**
(+10,818 B best, selectivity 0.175524). You cannot add to the model, subtract from it, swap it,
reorder its input, or coarsen its input. **Do NOT propose any coding change. This arm measures
PURCHASE, which is a different question and the one that is still open.**

**(3) The ALLOCATION level is only HALF closed.** `ddm_ld1` + `ddm_lx2` closed the TOKEN allocation
(Lane's 38,183 B is defended on both sides — lossless cannot touch it, lossy cannot afford it). The
**66,591 B residue allocation has never been priced against what it BUYS.** `mz2` (`5c073e915`)
measured all 38/38 semantic tensors **receiver-required** with 0 derive-at-decode — that answers
*"can the decode run without it?"*, never *"what does it buy?"* **A parameter can be
receiver-REQUIRED (decode crashes without it) and simultaneously distortion-INERT (the decoded
frames score identically whether its value is exact or coarse).** The first is plumbing; the second
is economics, and only the second is what the score pays for.

**(4) Why now, and why it is the fork.** `ddm_mst1` measured **78.71% of manufactured seg error at
the NATIVE RENDER** with R and uint8 as net REPAIRERS. The semantic renderer is therefore the single
object that is BOTH the largest byte cost (72.80% of demand) AND the dominant distortion source. If
some of its 30,856 B buys little distortion, that is the campaign's cheapest remaining move. If
every class is fully load-bearing, then with the token allocation already closed **no allocation
change on this body can meet the demand**, and a body change is the only remaining representation
move. Either answer routes the campaign; there is no empty outcome.

## SCOPE

1. **Verify pins; REUSE AR1B's census read-only; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token
   stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B.
   Reproduce AR1B's five-class split summing to 66,591 with no remainder before perturbing anything;
   a disagreement IS the finding. Do NOT re-derive the exchange rate —
   `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived `25/37,545,489 = 6.658590e-07`; cite and use it.
2. **PER-GROUP PURCHASE MEASUREMENT — the arm's whole job.** For each parameter group in the
   residue, coarsen it along its own natural quantization axis, hold every other byte identical, and
   measure realized **d_seg and d_pose through the real render → R → uint8 → frozen scorer path,
   n600**, per class with **Lane on its own row**. Per group report: **bytes held · realized Δd_seg
   per class · realized Δd_pose · ΔS_distortion · the byte credit the coarsening returns · net ΔS**.
   ≥3 coarsening levels per group so the response shape is visible. **Do NOT interpolate distortion
   between levels** — `ri1`/`ni1` measured amplification exponent **16.69**; seg responds violently
   and non-linearly. Measure each level.
3. **Label every group on BOTH axes: {RECEIVER-REQUIRED, DISTORTION-LOAD-BEARING}.** mz2 settled the
   first for the semantic tensors; this arm settles the second. **A group that is receiver-required
   AND distortion-inert is the exact cell this arm exists to find**, and it is reachable by
   coarsening rather than removal.
4. **Rank by exchange rate; state the waterfill.** Sort by S-per-byte, name the net-negative set,
   total its byte credit against the 42,382 B demand, and state what fraction it clears.
5. **Adjudicate honestly, including the empty outcome.** If every group is net-positive at every
   level, say so plainly — that closes 36.92% of the archive on measured evidence and, composed with
   ld1+lx2, closes ALL allocation on this body. **Build NO shipping candidate here**; the
   attribution table is the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal
  fires (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every coarsened section, its re-encoded archive, and every per-class
  argmax field persist with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer/` — BOTH
  SSD TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister arms
  at rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination
  per the disk rule while the tiers are full.** Do NOT write to `/Volumes/*` — a write there will
  kill you. Say which tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Every d_seg/d_pose states its GT lineage (DALI-GT where the tool family expects it; a PyAV-lineage
  GT on the pose axis is a measured wrong-objective defect).
- File ownership: AR1B owns the census · BL1 the token cost field · MS9 the manufactured split ·
  MST1 the stage split · LD1 the Lane rate curve · AE1/OE1 the excess family. CITE them; do not
  duplicate or touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `#996` · `#1124` · `mz2` (`5c073e915`) — the coder and recoding axes on these sections are CLOSED.
  All 4 sections measured vs their memoryless bound; 38/38 semantic tensors receiver-required, 0
  derive-at-decode; exact dense/sparse/row-dict/hybrid ALL +340 B. **Propose no re-encoding, no coder
  swap, no storage-layout change.**
- `ddm_oe1_online_escape_member` — adding one causal member: all rungs larger, best +10,818 B,
  selectivity **0.175524** vs a 1.5 bar; the adaptation-zero control was byte-identical (instrument
  trustworthy). With `ddm_ae1`'s two static kills the anti-predicted family is CLOSED whole.
- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` (`5e8d6011ba`) — all six lossy Lane rungs LARGER
  (+196 … +1,528 B); the unedited control is the rate optimum.
- `#1127` SD1M — **the defect this arm must not repeat**: "dead" pointwise tensors were dead in
  weight-MSE ONLY; render amplification ~**38,700×** made them look droppable and they were 90×
  underwater. **A weight-space proxy is not a purchase measurement.**
- `ddm_ri1` + `ddm_ni1` — whole-body lossy re-representations DEAD on distortion (43.66× and 247.71×
  **[MAIN ERRATUM 2026-08-22: the `247.71×` NI1/NR1-K32 figure in this section is WITHDRAWN — fabricated, no receipt; NI1's d_seg is NOT MEASURED and its token-agreement proxy is 1.079× DX2, and at 122,250 B it is byte-feasible for sub-0.12. The RI1 `43.66×` is real and MEASURED. See `.omx/research/ddm_ni1_247x_erratum_20260822.md`.]**
  over ceiling), amplification exponent **16.69**.
- `ddm_mst1_manufactured_stage_split_20260822.md` (`1c33f278…`) — 78.71% of manufactured seg error at
  the native render; R and uint8 net REPAIRERS. Scope is
  `INSTANCE:DX2_T4_n600_WITH_MACOS_CPU_INTERMEDIATE_OBSERVATIONS` — the ordering is robust, the exact
  share is advisory-lineage. Cite it at that strength, not stronger.

## OPTIMAL FORM

- **REFERENCE FORM: the shipped DX2 archive and its shipped receiver, both UNCHANGED except the ONE
  parameter group under test, coarsened along its own natural quantization axis.** The coder, the
  token stream, the group map, and every other section stay FIXED so the attribution isolates
  purchase from coding.
- Family exemplar for conduct: `ddm_ar1b_archive_residue_purchase_20260822.md` — it closed its census
  to zero remainder, custodied every span, and REFUSED to call a byte ranking a purchase
  attribution when it lacked the lane. Match that refusal discipline; you have the lane, so you owe
  the measurement it was refused.
- SCOPE reductions declared per row (a strided pilot to order the groups before full measurement is
  legal and must be labelled; the verdict is n600). MECHANISM reductions FORBIDDEN — a weight-MSE
  proxy, a modelled-distortion estimate, or a coarsening not carried through the real render→R→uint8
  path is the exact SD1M defect above. **Measure through the real path or do not report the row.**
- VERIFIED ARITHMETIC (MAIN re-derived): archive 180,368 B · token stream 113,777 B · residue
  **66,591 B = 36.9195% of archive = 1.5713× the demand**. DX2 S 0.14821987563243377 · rate
  25·180368/37545489 = 0.1200996 · seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) =
  0.0079812. S<0.12 needs ≤137,986 B → shed **42,382 B**; **6.658590e-07 S/B**; **1.2731082153 B/flip**.
- **PRIOR-LAW PREDICTION (falsifiable):** receiver-required ≠ distortion-load-bearing, and the gap is
  material. **At least one parameter group totalling >5,000 B (>11.8% of the demand) coarsens to a
  net-negative ΔS** — its realized distortion cost stays below its byte credit at 6.658590e-07 S/B —
  because these sections were sized by the TRAINING objective, never by realized purchase, and
  nothing in the shipped stack ever optimized them against d_seg/d_pose per byte.
  **FALSIFIER:** every group's cheapest net-negative coarsening returns <1,000 B, or every group is
  net-positive at every level. Then the residue is fully load-bearing; composed with ld1+lx2 closing
  the token allocation and the five-arm law closing the coding level, **no allocation change on this
  body can meet the demand**, and a body change (`#1187`) becomes the only remaining representation
  move. **Count it plainly if it lands; both outcomes route the campaign.**

## DELIVERABLE

`.omx/research/ddm_ap1_residue_purchase_scorer_20260823.md` — AR1B's five-class census reproduced +
the per-group PURCHASE table with, per group per level: **bytes held · realized Δd_seg per class
(Lane on its own row) · realized Δd_pose · ΔS_distortion · byte credit · net ΔS · {RECEIVER-REQUIRED,
DISTORTION-LOAD-BEARING}** + the exchange-rate ranking + the waterfilled net-negative set and its
share of the 42,382 B demand, OR the honest all-positive verdict + verdict_scope at the NARROWEST
level the evidence supports. Every figure carries its denominator and its GT lineage. No shipping
candidate. Commit via the serializer. End with the own-vehicle frontier line.
