# ddm_ar1b_archive_residue_purchase — 66,591 B (36.92% of the archive, 1.57× the whole demand) has never been attributed; BL1 asked where the token BITS go, nobody asked what the OTHER THIRD BUYS

## MANDATE

Routed finding, arithmetic verified by MAIN, no operator verbatim.

**The gap.** The dx2 archive is **180,368 B**. `ddm_bl1_per_position_bit_allocation_20260822.md`
(commit `873947c665`) instrumented the shipped decoder and reconciled the **RC64 token stream at
113,777 B** (910,216 physical bits) to the exact bit. **Residue = 180,368 − 113,777 = 66,591 B =
36.92% of the archive = 1.57× the entire 42,382 B sub-0.12 demand.** Six arms this wave — bl1, ae1,
lx2, to2, ef1, ld1 — all aimed at the token stream. **Nobody asked where the other third goes or
what it buys.**

**This is NOT the coder question, and proposing a coder change is the defect that killed two arms
this wave.** The coding axis on these sections is CLOSED and you must not reopen it:
- `#996` measured all four sections against their own memoryless bound — coder axis closed on this base.
- `#1124` closed section-coding as an axis.
- `mz2` (commit `5c073e915`) measured **all 38/38 semantic tensors receiver-required, 0
  derive-at-decode**, and exact dense / sparse / row-dict / hybrid re-encodings **ALL +340 B**.

**mz2 answered "can the receiver run without this tensor?" — it did NOT answer "what does this
tensor BUY?"** Those are different questions and only the first has been measured. A parameter can be
receiver-REQUIRED (the decode crashes without it) and still be distortion-INERT (the decoded frames
score identically whether its value is exact or coarse). The first is a plumbing fact; the second is
an economic one, and it is the one the score cares about.

**The exact analogue that makes this well-posed.** BL1's contribution was attributing the token
stream's 910,216 bits to POSITIONS, producing a per-position cost field. The unbuilt analogue is
attributing the 66,591 B residue to PARAMETERS, and joining each parameter's byte cost to its
MEASURED distortion contribution. That yields a per-parameter S/B exchange rate in the SAME currency
as every other lever on this campaign, and it is the only form in which the residue can be waterfilled.

**Why it is worth an arm now.** Three arms just converged on the same direction. LD1
(`5e8d6011ba`) measured every lossy Lane rung making the archive LARGER; LX2 proved the lossless
control cannot buy distortion; MST1 (`1c33f278920b91bf922e9620deb9ce20615135e8`) measured 78.71% of
manufactured seg error at the native render with R and uint8 as net REPAIRERS. The token-stream
allocation is defended on both sides. If the demand is met at all on this body, the bytes plausibly
come from a section nobody has priced.

## SCOPE

1. **Verify pins; establish the residue's composition; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token stream
   sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B. **First
   deliverable: the exact section-by-section byte census of the 66,591 B residue** — every member,
   its byte count, its sha256, and its role, summing EXACTLY to 66,591 with no unexplained remainder.
   A remainder IS a finding; report it rather than absorbing it. Do NOT re-derive the exchange rate —
   `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived `25/37,545,489 = 6.658590e-07`; cite and use it.
2. **Per-parameter PURCHASE attribution — the arm's core.** For each parameter group in the residue,
   measure what its bytes BUY: perturb or coarsen it (its own natural quantization axis), hold
   everything else byte-identical, and measure realized **d_seg and d_pose through the real render →
   R → uint8 → frozen scorer path**, per class with Lane on its own row. Report per group: **bytes
   held · realized Δd_seg · realized Δd_pose · ΔS_distortion · the byte credit its removal or
   coarsening would return · the net ΔS.** A group reporting bytes without a measured distortion
   consequence has not been attributed.
3. **Rank by exchange rate and state the waterfill.** Sort groups by S-per-byte. Name the set whose
   removal or coarsening is net-negative and total its byte credit against the 42,382 B demand.
   **State plainly whether any subset clears a material fraction, and what fraction.**
4. **Distinguish RECEIVER-REQUIRED from DISTORTION-LOAD-BEARING explicitly.** Every group gets both
   labels. mz2 already established that all 38 semantic tensors are receiver-required; if a group is
   receiver-required AND distortion-inert, say so — that is the exact cell this arm exists to find,
   and it is reachable by coarsening rather than removal.
5. **Adjudicate honestly, including the empty outcome.** If every group's ablation costs more
   distortion than its bytes are worth, say so plainly: the residue is fully load-bearing, the demand
   cannot be met from the non-token third of this body, and 36.92% of the archive is closed on
   measured evidence. **Build NO shipping candidate here**; the attribution table is the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal
  fires (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every perturbed section, its re-encoded archive, and every per-class
  argmax field persist with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ar1b_archive_residue_purchase/` — BOTH
  SSD TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister arms at
  rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination per
  the disk rule while the tiers are full.** Do NOT write to `/Volumes/*` — a write there will kill you.
  Say which tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Every d_seg/d_pose states its GT lineage (DALI-GT where the tool family expects it; a PyAV-lineage
  GT on the pose axis is a measured wrong-objective defect).
- **If the exclusive n600 scorer lane is unavailable, say so and deliver the census + the ranked
  per-group byte table with distortion QUEUED — do NOT call a byte-only ranking a purchase
  attribution.** LD1 handled exactly this correctly; match it.
- File ownership: BL1 owns the token cost field · MS9 the manufactured split · MST1 the stage split ·
  LD1 the Lane rate curve · AE1 the excess · OE1 is concurrently measuring the online escape member.
  CITE them; do not duplicate or touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `#996` · `#1124` · `mz2` (`5c073e915`) — **the coder and recoding axes on these sections are
  CLOSED.** All 4 sections measured vs their memoryless bound; all 38/38 semantic tensors
  receiver-required with 0 derive-at-decode; exact dense/sparse/row-dict/hybrid ALL +340 B. Do NOT
  propose a re-encoding, a coder swap, or a storage-layout change. This arm measures PURCHASE.
- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` (`5e8d6011ba`) — all six lossy Lane rungs made the
  archive LARGER (+196 … +1,528 B); the unedited control is the rate optimum.
- `ddm_to2_token_ordering_race_20260822.md` (orderings 196–687% worse) ·
  `ddm_ef1_token_entropy_floor_20260822.md` (generic estimators 3.21× worse, FAMILY-scoped) —
  with LD1 these are **four independent arms measuring one law: the shipped 19-member HPAC model and
  this field sit jointly at a local optimum, and every perturbation of the CODING costs bytes.** That
  law is why this arm perturbs the PARAMETERS to measure what they buy, and never the coder.
- `ddm_mst1_manufactured_stage_split_20260822.md` (`1c33f278…`) — 78.71% of manufactured seg error at
  the native render, R and uint8 net REPAIRERS. Its scope is
  `INSTANCE:DX2_T4_n600_WITH_MACOS_CPU_INTERMEDIATE_OBSERVATIONS`; the ordering is robust, the exact
  share is advisory-lineage. Cite it at that strength, not stronger.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` —
  whole-body lossy re-representations DEAD on distortion (43.66× and 247.71× over ceiling),
**[MAIN ERRATUM 2026-08-22: the `247.71×` NI1/NR1-K32 figure in this section is WITHDRAWN — fabricated, no receipt; NI1's d_seg is NOT MEASURED and its token-agreement proxy is 1.079× DX2, and at 122,250 B it is byte-feasible for sub-0.12. The RI1 `43.66×` is real and MEASURED. See `.omx/research/ddm_ni1_247x_erratum_20260822.md`.]**
  amplification exponent **16.69**. **Do NOT interpolate distortion between perturbation levels —
  measure each.**

## OPTIMAL FORM

- **REFERENCE FORM: the shipped DX2 archive and its shipped receiver, both UNCHANGED except for the
  ONE parameter group under test, perturbed along its own natural quantization axis.** The coder, the
  token stream, the group map, and every other section are FIXED so the attribution isolates purchase.
- Family exemplar for conduct: `ddm_bl1_per_position_bit_allocation_20260822.md`, commit
  **`873947c665`** — it reconciled its instrument to the physical stream, explained its 56-bit
  residual instead of absorbing it, refused to call its allocation a bound, and reported its MS9 join
  in BOTH directions including the one that weakened its own story. This arm is BL1's method aimed at
  the sections BL1 did not cover. Match its conduct, especially the residual honesty in SCOPE 1.
- SCOPE reductions declared per row (a strided pilot to rank groups before full measurement is legal
  and must be labelled; the verdict is n600). MECHANISM reductions FORBIDDEN — a weight-MSE proxy
  standing in for realized distortion is the exact defect that made SD1M's "dead" pw tensors look
  droppable when render amplification was ~38,700× (`#1127`). **Measure through the real path or do
  not report the row.**
- VERIFIED ARITHMETIC (MAIN re-derived): archive 180,368 B · token stream 113,777 B · **residue
  66,591 B = 36.9195% of archive = 1.5713× the demand.** DX2 S 0.14821987563243377 · rate
  25·180368/37545489 = 0.1200996 · seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) =
  0.0079812. S<0.12 needs ≤137,986 B → shed **42,382 B**; **6.658590e-07 S/B**; **1.2731082153 B/flip**.
- **PRIOR-LAW PREDICTION (falsifiable):** receiver-required ≠ distortion-load-bearing, and the gap is
  material. **At least one parameter group totalling >5,000 B (>11.8% of the demand) coarsens to a
  net-negative ΔS** — its realized distortion cost stays below its byte credit at 6.658590e-07 S/B —
  because nothing in the shipped stack ever optimized these sections against realized distortion; they
  were sized by the training objective, not by purchase.
  **FALSIFIER:** every group's cheapest net-negative coarsening returns <1,000 B, or every group is
  net-positive at every perturbation level. Then the residue is fully load-bearing, 36.92% of the
  archive is closed, and — with LD1+LX2 closing the token allocation — the demand cannot be met from
  ANY allocation change on this body, which routes the campaign to a body change (`#1187` nr1) as the
  only remaining representation move. **Count it plainly if it lands; both outcomes are complete.**

## DELIVERABLE

`.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md` — the exact 66,591 B section census
summing with no unexplained remainder + the per-group PURCHASE table with, per group: **bytes held ·
realized Δd_seg per class (Lane on its own row) · realized Δd_pose · ΔS_distortion · byte credit ·
net ΔS · {RECEIVER-REQUIRED, DISTORTION-LOAD-BEARING} labels** + the exchange-rate ranking + the
waterfilled net-negative set and its share of the 42,382 B demand, OR the honest all-positive verdict
+ verdict_scope at the NARROWEST level the evidence supports. Every figure carries its denominator and
its GT lineage. No shipping candidate. Commit via the serializer. End with the own-vehicle frontier line.
