# ddm_ld1_lane_lossy_drop_exchange — Lane costs 38,183 B (90.1% of the demand) at 0.59% of area; LX2 proved the lossless control cannot buy distortion, so measure the LOSSY drop curve — and use MS9's measured 75.3% error-absorption as the buffer

## MANDATE

Routed finding, three memos, no operator verbatim:

**(1) The rate mass is located.** `ddm_bl1_per_position_bit_allocation_20260822.md` (commit `873947c665`,
reconciled to the physical stream: 910,209.281 modeled vs 910,216 bits, 56-bit decoder lookahead into
defined zero fill) measured **Lane = 0.5856% of area carrying 33.56% of the shipped HPAC/RC64 model
bits** — 305,463.969 bits = **38,183 B = 90.1% of the entire 42,382 B sub-0.12 demand**, at 0.442218
bits/position, **57.31× the mean**. Concentration overall: top 1% of positions = 96.323842% of bits,
Gini 0.995159.

**(2) The lossless control cannot reach it — my previous charter's mechanism was self-contradictory.**
`ddm_lx2_lane_bit_budget_exchange_20260822.md` REFUSED before sweep:
`REFUSED_BEFORE_SWEEP_NO_DISTORTION_BEARING_CONTROL`. Class-conditioned precision / context depth /
mixing weights inside the shipped 19-member HPAC law change only the RC64 coding PROBABILITIES; the
decoded token field is bit-identical by construction, so d_seg is INVARIANT. It refused to invent five
distortion-bearing rungs from a probability-only control. **The exchange curve requires changing the
DECODED TOKEN FIELD — a lossy act. That is this arm's mechanism and it must be lossy on its face.**

**(3) MS9 supplies a buffer nobody has priced.**
`ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` measured, n600, contest-CUDA DALI GT: of **9,182**
transmitted-label errors, only **2,264** survive to the final argmax — the realization path REPAIRS
6,918 of them. **24.66% survival.** If that ratio holds for newly-introduced Lane coarsening errors,
each transmitted error added costs ~0.247 final flips, a ~4× buffer against the naive count.
**Whether it holds for Lane-scoped, deliberately-introduced errors is UNMEASURED and is the crux
this arm decides.** Do not assume it; measure it as its own row.

**The arithmetic that makes this worth an arm.** At **1.2731082153 B per eliminated flip** (6.658e-7
S/B), freeing X bytes buys X/1.2731 flips of headroom. Halving Lane frees 19,091 B = **14,996 flips** —
a 63% increase over the body's ENTIRE current seg error of 23,757. Under a 24.66% survival ratio that
budget absorbs ~60,700 added transmitted errors. Both numbers are large; the curve decides.

## SCOPE

1. **Verify pins; REUSE BL1's cost field and MS9's split read-only; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token stream
   sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B · TO2 decoded field
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. Reproduce Lane's 33.56% row and
   MS9's 23,757/117,964,800 before sweeping; a disagreement IS the finding. Do NOT re-instrument the
   decoder and do NOT re-derive the exchange rate — `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived
   `25/37,545,489 = 6.658590e-07`; cite it.
2. **Sweep the LOSSY Lane-DROP curve, ≥5 rungs, DOWN only.** Each rung coarsens the DECODED token field
   at Lane positions (e.g. progressive coefficient/precision drop, run-merging, class-scoped
   quantization — the decoded symbols must actually change), then re-encodes with the SHIPPED 19-member
   HPAC law UNCHANGED. Per rung report: **bytes** (real re-encode of the actual stream, never an
   estimate) and **d_seg** through the real render → R → uint8 → frozen SegNet argmax path, **per class,
   Lane on its own row, collateral on the other four explicit.**
3. **Measure the ABSORPTION RATIO per rung — the crux.** For each rung report BOTH: transmitted-label
   errors ADDED, and final-argmax errors added after realization. Their ratio is the live survival
   fraction for deliberately-introduced Lane errors. State it next to MS9's 24.66% body-wide baseline
   and say plainly whether it holds, is better, or is worse. **A rung that reports only final d_seg has
   not measured the mechanism.**
4. **Price every rung in BOTH currencies.** `ΔS = 6.658e-7·Δbytes + 100·Δd_seg`. Report where the
   shipped allocation sits and the byte-and-flip coordinates of the best rung. A rung in bytes alone or
   flips alone is not a result.
5. **Adjudicate honestly, including the empty outcome.** If a rung is net-negative, state its ΔS and
   share of the 42,382 B demand. If every rung is net-positive, say so plainly — that closes a
   90.1%-of-demand question on measured evidence, which is a complete result. **Build NO shipping
   candidate here**; the curve is the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal fires
  (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every rung's re-encoded stream, per-class mask, per-rung transmitted-error
  field and final-argmax field persist with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/` — **BOTH SSD TIERS ARE AT 100% (measured 08-22; this killed the prior generation of this arm at rc=1 with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination per the disk rule while the tiers are full.** Do NOT write to /Volumes/* — a write there will kill you.
  Say which tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Every d_seg states its GT lineage (DALI-GT where the tool family expects it; a PyAV-lineage GT on the
  pose axis is a measured wrong-objective defect).
- File ownership: BL1 owns the cost field · MS9 the manufactured split · MST1 is concurrently splitting
  that loss BY STAGE · AE1 is concurrently measuring excess over log2(5) · XS1 cross-section
  conditioning. CITE them; do not duplicate or touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_lx2_lane_bit_budget_exchange_20260822.md` — **the direct parent.** Its refusal is why this
  charter's mechanism is lossy. `verdict_scope: FORMULATION`, explicitly NOT closing class-aware lossy
  token selection. Do not re-attempt a lossless control and call it an exchange.
- `ddm_lq1_lane_quotient_representability_20260822.md` — **the collateral warning.** A Lane-recall
  oracle recovered 417,267 Lane pixels for **+2,755,323 total mismatches (+194%)**. That pushed Lane
  fidelity UP. This arm pushes DOWN — a different direction, never measured — but the same collateral
  discipline binds: every rung carries its per-class collateral or it is not a rung.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` —
  whole-body lossy re-representations DEAD on distortion (**43.66×** and **247.71×** over their
**[MAIN ERRATUM 2026-08-22: the `247.71×` NI1/NR1-K32 figure in this section is WITHDRAWN — fabricated, no receipt; NI1's d_seg is NOT MEASURED and its token-agreement proxy is 1.079× DX2, and at 122,250 B it is byte-feasible for sub-0.12. The RI1 `43.66×` is real and MEASURED. See `.omx/research/ddm_ni1_247x_erratum_20260822.md`.]**
  ceilings), amplification exponent **16.69** (token disagreement ×1.0975 → d_seg ×4.7242). Seg responds
  violently and non-linearly. **Do NOT interpolate d_seg between rungs — measure each.** Note the
  distinction that justifies this arm: those changed 1.5M+ tokens body-wide; this is Lane-scoped with a
  known cost structure and a measured absorption ratio.
- `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` — **90.4702%** of DX2's seg error is MANUFACTURED
  downstream; only 2,264 of 23,757 are representation errors that survive. Coarsening ADDS
  representation error, so this arm moves INTO the 9.53% slice — small, but with 75.3% absorption. Both
  facts must appear in your adjudication.
- `ddm_cx3_context_axis_ceiling_20260822.md` (named-context ladder **0 B**) ·
  `ddm_to2_token_ordering_race_20260822.md` (orderings 196–687% worse) ·
  `ddm_ef1_token_entropy_floor_20260822.md` (best generic estimator **3.21× worse**) — the lossless rate
  axis is closed. Do not reopen it; the coder stays FIXED so the curve isolates the lossy variable.

## OPTIMAL FORM

- **REFERENCE FORM: the shipped DX2 receiver + the shipped 19-member HPAC/RC64 law, both UNCHANGED** —
  the only swept variable is the decoded Lane token content. Family exemplar for conduct:
  `ddm_bl1_per_position_bit_allocation_20260822.md`, commit **`873947c665`** — it reconciled its
  instrument to the physical stream, explained its 56-bit residual instead of hand-waving it, refused to
  call its allocation a bound, and reported the MS9 join in BOTH directions including the one that
  weakened its own story. Match that, especially the last part.
- SCOPE reductions declared per row (a strided pilot to shape the sweep is legal and must be labelled;
  the verdict is n600). MECHANISM reductions FORBIDDEN — a weaker coder, a proxy scorer, or a lossless
  stand-in is the exact defect that killed LX2.
- VERIFIED ARITHMETIC: DX2 S 0.14821987563243377 @ 180,368 B · rate 25·180368/37545489 = 0.1200996 ·
  seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B →
  shed **42,382 B**; **6.658e-7 S/B**; **1.2731082153 B/flip**. Lane = **38,183 B = 90.1% of demand**.
  MS9 survival **2,264/9,182 = 24.66%**.
- **PRIOR-LAW PREDICTION (falsifiable):** the absorption buffer makes an interior optimum exist. A Lane
  drop freeing **>8,000 B** adds **<60,000** transmitted errors whose realized survival stays within 2×
  of MS9's 24.66%, so realized added flips stay **<6,285** (the break-even at that byte count), for net
  **ΔS < −0.002**; collateral on Road/Undrivable stays under 25% of the added Lane flips.
  **FALSIFIER:** every rung is net-positive — either survival for introduced Lane errors runs far above
  24.66% (the absorption buffer does not extend to deliberate coarsening), or the 16.69 amplification
  dominates before 8,000 B is freed. Then the Lane rate mass is **defended on both sides** — lossless
  cannot touch it (LX2) and lossy cannot afford it (this arm) — and the seg axis needs a representation
  change, not an allocation change. **Count it plainly if it lands; both outcomes are complete.**

## DELIVERABLE

`.omx/research/ddm_ld1_lane_lossy_drop_exchange_20260822.md` — BL1's Lane row and MS9's split reproduced
+ the ≥5-rung DOWN curve with, per rung: REAL re-encoded bytes · real-path d_seg per class with **Lane on
its own row and collateral on the other four** · **transmitted-errors-added AND final-flips-added with
their ratio vs MS9's 24.66%** · ΔS in the joint currency + the located optimum or the honest all-positive
verdict + the share of the 42,382 B demand + verdict_scope at the NARROWEST level the evidence supports.
Every figure carries its GT lineage and denominator. No shipping candidate. Commit via the serializer.
End with the own-vehicle frontier line.
