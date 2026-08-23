# ddm_lx2_lane_bit_budget_exchange — Lane costs 38,183 B (90.1% of the whole demand) at 0.5856% of area; is that share ABOVE or BELOW its joint-S optimum?

## MANDATE

**The exchange rate is DERIVED and BANKED — what is missing is its use on Lane.**
`ddm_tx1_toolbox_crosswalk_20260819.md` §0 ("THE EXCHANGE RATES — what closing the gap actually costs")
derived `25/37,545,489 = 6.658590e-07` S per byte and priced label/position families in B/flip
(0.2627 · 0.9822 · 1.4610 B/flip at :190–:204). **Do NOT re-derive it — cite tx1 and use it.** What no
arm has done is apply it to the CLASS that dominates both axes. Every arm on 08-22 priced candidates in
ONE currency. The rate arms held distortion fixed and
attacked bytes (RB1 0 B · AD2 already free · TO2 196–687% worse · CX3 0 B · EF1 3.21× worse). The
distortion arms held bytes fixed (MS9 manufactured split · LQ1 representability · RI1/NI1 receiver
distortion). The exchange between them is **6.658e-7 S/B**, or equivalently **1.2731082153 B per
eliminated seg flip**. Applying it to Lane is the gap this arm fills.

**Lane is where the two axes collide.** BL1 (`ddm_bl1_per_position_bit_allocation_20260822.md`, commit
`873947c665`, reconciled to the physical stream) measured the shipped HPAC/RC64 per-position cost:

| class | area | share of model bits | bits/position | vs mean |
|---|---:|---:|---:|---:|
| **Lane** | **0.5856%** | **33.56%** (305,463.969 / 910,209.281 bits) | **0.442218** | **57.31×** |
| Movable | 1.2380% | 10.21% | 0.063655 | 8.25× |
| Road | 23.2335% | 39.67% | 0.013175 | 1.71× |
| Undrivable | 49.5174% | 11.37% | 0.001771 | 0.23× |
| MyCar | 25.4256% | 5.19% | 0.001574 | 0.20× |

**Lane's coding cost is 305,463.969 bits = 38,183 B = 90.1% of the entire 42,382 B demand**, spent on a
class occupying 0.5856% of the frame. The same class is the worst on the distortion axis: GT IoU 0.263,
and ~19% of all d_seg flips.

**The unpriced arithmetic.** At 1.2731082153 B/flip, freeing X bytes from Lane buys X/1.2731 flips of
headroom. Halving Lane's cost frees 19,091 B = **14,996 flips** — a **63% increase** over the body's
ENTIRE current seg error (23,757 / 117,964,800). That is a large budget nobody has measured against.

**And the sign is genuinely open.** Two live hypotheses point OPPOSITE ways:
- **Lane is OVER-represented** (this charter's rate reading): 57.31× the mean cost for 0.59% of area is
  a lot of bits to spend on a class the model predicts badly anyway.
- **Lane is UNDER-represented** (CB2's live hypothesis): the dictionary/model objective is AREA-weighted
  while the score is FLIP-paid, so Lane at 0.59% area but ~19% of flips is ~32× under-weighted — the
  named cause of its IoU 0.146 in that arm's object.
Both can be true of different objects. **Which side of the optimum the SHIPPED 33.56% sits on is one
measurement, and it is the deliverable.**

## SCOPE

1. **Verify pins, REUSE BL1's field, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token stream
   sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B · TO2 decoded field
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` · BL1's per-position cost field and
   per-class table. **Do NOT re-instrument the decoder** — BL1 did it and reconciled it (910,209.281
   modeled vs 910,216 physical, 56-bit lookahead into defined zero fill). Reproduce Lane's row and the
   96.323842% / 52.950688% / Gini 0.995159 concentration before sweeping; a disagreement IS the finding.
2. **Build the EXCHANGE CURVE, both directions, on the incumbent's own model class.** Sweep Lane's
   effective bit allocation across at least 5 rungs spanning BELOW and ABOVE the shipped 33.56% — via a
   class-conditioned rate allocation INSIDE the shipped 19-member HPAC mixing law (e.g. a per-class
   precision/context-depth or mixing-weight allocation), NOT by swapping in a different codec and NOT by
   deleting Lane positions outright. At each rung report: **bytes** (real re-encode, never a model
   estimate) and **d_seg** (through the real render → R → uint8 → frozen SegNet argmax path), per class
   with **Lane on its own row**, plus **collateral on the other four classes**.
3. **Price every rung in BOTH currencies and locate the optimum.** For each rung compute
   `ΔS = 6.658e-7·Δbytes + 100·Δd_seg`, and state where the shipped allocation sits: at, above, or below
   the joint-S optimum, with the byte-and-flip coordinates of the best rung found. **A rung reported in
   bytes alone, or in flips alone, is not a result** — that single-currency habit is what this arm exists
   to end.
4. **Report the COLLATERAL explicitly, per rung.** Seg mechanisms in this campaign die on collateral,
   not on targeting — measured three separate times on 08-22. A Lane rung that improves Lane and wrecks
   Road is a loss; the joint ΔS must carry it, and the per-class table must show it.
5. **Adjudicate against the 42,382 B demand, and be honest about what a curve is.** If an interior
   optimum exists, state its ΔS and what fraction of the demand it supplies. If the shipped allocation
   IS the optimum (flat or adverse in both directions), say so plainly — that closes a 90.1%-of-demand
   question and is a complete result. **Do NOT build a shipping candidate in this arm**; the curve and
   its optimum are the deliverable, and a candidate is a successor's job with its own gate.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY — R, uint8, the scorer and the argmax are frozen and are what we measure against.
- NO Modal fire. NO Metal fires (MAIN-fire-only). Local advisory launches ONLY via
  `tools/fire_local_advisory.py` — hand-assembled dispatch is the error factory.
- Every byte number is a REAL re-encode of the actual stream. Every d_seg is through the REAL path with
  its GT lineage stated (DALI-GT where the tool family expects it; a PyAV-lineage GT on the pose axis is
  a measured wrong-objective defect).
- Shipped receiver bytes are CUSTODY — never edit in place.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every rung's re-encoded stream, per-class mask, and per-position field
  persists with sha256 + bytes. A scalar-only artifact while the bytes exist in memory is forbidden AT
  THE TYPING MOMENT.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_lx2_lane_bit_budget_exchange/` — NOT APDataStore
  (~11 GiB free).** Say which tier you used.
- File ownership: BL1 owns the cost field (REUSE, cite) · MS9 the manufactured split · MST1 is
  concurrently splitting that loss by stage · AE1 is concurrently measuring excess over log2(5) · XS1
  cross-section conditioning. Do not duplicate them.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_lq1_lane_quotient_representability_20260822.md` — **the decisive collateral warning.** A
  Lane-recall oracle recovered 417,267 Lane pixels at the price of **+2,755,323 total mismatches
  (+194%)**. Pushing Lane's fidelity UP is measured to be catastrophic in that formulation. Your
  upward rungs must carry their collateral or they repeat it.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` — both
  byte-feasible re-representations DEAD on distortion (**43.66×** and **247.71×** over their d_seg
**[MAIN ERRATUM 2026-08-22: the `247.71×` NI1/NR1-K32 figure in this section is WITHDRAWN — fabricated, no receipt; NI1's d_seg is NOT MEASURED and its token-agreement proxy is 1.079× DX2, and at 122,250 B it is byte-feasible for sub-0.12. The RI1 `43.66×` is real and MEASURED. See `.omx/research/ddm_ni1_247x_erratum_20260822.md`.]**
  ceilings), with an amplification exponent of **16.69** between them: token disagreement ×1.0975 →
  d_seg ×4.7242. **Seg responds violently and non-linearly.** Do not interpolate d_seg between rungs;
  measure each one.
- `ddm_cx3_context_axis_ceiling_20260822.md` — the named-summary context ladder returns **0 B**; the best
  model-inclusive challenger is +11,433 B and its hindsight IDEAL data term is already +3,447 B worse than
  shipped. Richer models die on their own description length. Your reallocation must be BYTE-NEUTRAL in
  model size or pay for its growth in the same column.
- `ddm_to2_token_ordering_race_20260822.md` — reordering is a SUBSTITUTE for a context model, not a
  complement; nine generic orderings 196–687% worse. Do not reorder.
- `ddm_ef1_token_entropy_floor_20260822.md` — best generic estimator **365,322 B, 3.21× worse** than
  shipped; scope FAMILY twice, **UNKNOWN** for differently-trained HPAC. The incumbent is a strong
  domain model. Do not model it as naive, and do not replace it.
- `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` — **90.4702% of DX2's seg error is MANUFACTURED**
  downstream of a correct transmitted label; only 2,264 of 23,757 final errors are representation errors
  that survive. Consequence for you, and it is sharp: **coarsening Lane's REPRESENTATION attacks the
  9.53% slice, not the 90.47% one.** Report which slice each rung moves; a rung that adds representation
  error is fighting for a much smaller pool than the headline seg term suggests.
- `ddm_bl1_per_position_bit_allocation_20260822.md` §"Exact MS9 Seg-error join" — 90.7017% of seg errors
  lie inside the top-1% expensive set, but seg errors are only 2.013906% of that population and carry
  **5.265498%** of stream bits (5,990.882 B). The seg×rate join is ASYMMETRIC; **the shared object is
  the Lane CLASS, not seg-error position identity.** That is precisely why this arm is class-scoped.
- The self-audit filed against `ddm_ef1_token_entropy_floor_20260822.md`: its charter raced a weaker
  mechanism class against a tuned incumbent. Do not repeat it — reallocate WITHIN the incumbent, never replace it.

## OPTIMAL FORM

- **REFERENCE FORM (cited): the shipped 19-member HPAC context law over 190 groups
  `g=(x mod 64)+2*(y mod 64)`, with a CLASS-CONDITIONED rate allocation** — same model class, same coder,
  the allocation being the only thing swept. Substituting a different or weaker coder/model family is a
  MECHANISM reduction and is FORBIDDEN.
- Family exemplar for CONDUCT: `ddm_bl1_per_position_bit_allocation_20260822.md` — reconciled its
  instrument to the physical stream, explained its 56-bit residual rather than hand-waving it, refused to
  call its allocation a bound, and reported the MS9 join in BOTH directions including the direction that
  weakened its own story. Match that, especially the last part.
- **n600, all 600 pairs, on every rung that produces a verdict.** A strided pilot is legal to shape the
  sweep and must be labelled SCOPE-reduced; the verdict is n600. Prefix subsets are measured
  anti-conservative on some axes here (pose prefixes 2.54–4.21× harder; seg prefixes 0.95–0.97× easier).
- VERIFIED ARITHMETIC (re-derive once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.1200996 · seg 100·0.00020139 = 0.020139 (exactly 23,757/117,964,800) ·
  pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B → shed **42,382 B**; **6.658e-7 S/B**;
  **1.2731082153 B per eliminated flip**. Lane = 305,463.969 bits = **38,183 B = 90.1% of the demand**.
  Halving Lane frees 19,091 B = **14,996 flips** of headroom vs a current TOTAL seg error of 23,757.
- **PRIOR-LAW PREDICTION (falsifiable):** an interior optimum exists BELOW the shipped 33.56% — Lane is
  over-represented on the joint objective, and a byte-neutral-model reallocation that cuts Lane's share
  toward ~20% frees **>8,000 B** while adding **<6,285 flips** (the break-even at that byte count), for a
  net **ΔS < −0.002**. Collateral on Road/Undrivable stays under 25% of the added Lane flips.
  **FALSIFIER:** the curve is flat or adverse in BOTH directions within noise at every rung ⇒ the shipped
  Lane allocation IS at its joint optimum, the 33.56% share is EARNED, and a 90.1%-of-demand question is
  closed on measured evidence. Report either plainly with the optimum's coordinates in the FIRST line —
  both outcomes are complete and campaign-directing, and the falsifier would retire the largest single
  unexamined allocation in the archive.

## DELIVERABLE

`.omx/research/ddm_lx2_lane_bit_budget_exchange_20260822.md` — BL1's Lane row and concentration figures
reproduced + the ≥5-rung exchange curve spanning both sides of 33.56%, each rung carrying REAL bytes AND
real-path d_seg per class with **Lane on its own row** and **collateral on the other four** + each rung's
ΔS in the joint currency + the located optimum (or the honest flat/adverse verdict) with its byte-and-flip
coordinates + which MS9 slice (the 9.53% representational or the 90.47% manufactured) each rung moves +
the verdict on the prior-law prediction with verdict_scope at the NARROWEST level the evidence supports.
Every figure carries its GT lineage and its denominator. No shipping candidate. Commit via the
serializer. End with the own-vehicle frontier line.
