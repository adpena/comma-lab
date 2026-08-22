# ddm_ae1_anti_predicted_excess — the top 0.1% of positions average 1.76× the 5-symbol UNIFORM cost; measure how much of the stream is coded WORSE THAN NOTHING

## MANDATE

BL1 (`ddm_bl1_per_position_bit_allocation_20260822.md`, commit `873947c665`) instrumented the shipped
HPAC/RC64 decoder and reconciled per-position cost to the physical stream (910,209.281 modeled vs
910,216 bits; the 56-bit gap is decoder lookahead into defined zero fill). It measured extreme
concentration — top 1% of positions carry **96.323842%**, top 0.1% carry **52.950688%**, Gini
**0.995159** — and its per-class table:

| class | area | share of model bits | bits/position | vs mean |
|---|---:|---:|---:|---:|
| **Lane** | **0.5856%** | **33.56%** (305,463.969 / 910,209.281) | **0.442218** | **57.31×** |
| Movable | 1.2380% | 10.21% | 0.063655 | 8.25× |
| Road | 23.2335% | 39.67% | 0.013175 | 1.71× |
| Undrivable | 49.5174% | 11.37% | 0.001771 | 0.23× |
| MyCar | 25.4256% | 5.19% | 0.001574 | 0.20× |

**Two arithmetic facts BL1 did not draw from its own field.**

**(1) Lane's total coding cost is 305,463.969 bits = 38,183 B — 90.1% of the campaign's entire
42,382 B demand — in a class occupying 0.5856% of pixel area.**

**(2) The top 0.1% averages 4.0857 bits/position. The 5-symbol uniform cost is log2(5) = 2.3219
bits.** Any position costing more than log2(5) is coded **WORSE THAN A UNIFORM PRIOR** — the model is
not merely failing to predict it, it is assigning low probability to the symbol that actually occurs.
A uniform fallback would code such positions *cheaper*. BL1's memo does not raise this; a grep for
uniform / escape / log2(5) returns nothing on point.

**Why this is worth an arm and is not a free win.** If a material mass sits above log2(5), it is
recoverable in principle by the STANDARD cure in context modeling — mixing an escape/uniform member
into the model so no context can price a symbol above its own alphabet bound. But: (a) the mean over a
set is not a per-position count, and only the exact excess mass matters; (b) any escape must be
SIGNALLED or LEARNED, and signalling costs bits that must be netted; (c) the incumbent already mixes 19
members, so the honest prior is that some of this is already handled and what remains is the residual
the mixer cannot reach. **Report the NET, never the gross.**

## SCOPE

1. **Verify pins, REUSE BL1's retained per-position cost field, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 stream sha
   `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B · TO2 decoded field
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` · BL1's memo + its retained cost
   field. **Do NOT re-instrument the decoder — BL1 did it and reconciled it.** Reproduce
   96.323842% / 52.950688% / Gini 0.995159 and the Lane row before measuring anything new; a
   disagreement IS the finding.
2. **Measure the EXCESS-OVER-UNIFORM mass exactly.** Count positions with cost > log2(5) = 2.3219
   bits. Report: how many, what total bits they carry, and the **excess** `Σ max(0, cost_i − log2 5)`
   in bits and bytes. Break it down per class with **Lane on its own row**, per frame/time, and per the
   190 groups `g=(x mod 64)+2*(y mod 64)`. This exact excess is the GROSS ceiling of any
   alphabet-bound-respecting fix, and must be labelled GROSS.
3. **Price the SIGNALLING honestly — this is where the gross becomes a net or dies.** An escape that
   the receiver must be TOLD about costs bytes; an escape it can DERIVE costs none (rule-118: generic
   algorithm reading already-decoded content is free, stored side information is COUNTED). Price at
   least: (a) a learned uniform/escape MEMBER added to the incumbent's 19-member mix — needs no
   per-position signal, only model-blob growth, so price that growth; (b) an explicit per-position or
   per-run escape flag — price the flag stream with a real coder. **Net = gross excess − signalling.**
   A gross figure without its signalling column is not a result.
4. **Explain why the 19-member mixer does not already do this, or find that it partly does.** If the
   incumbent already caps some contexts near the alphabet bound, the residual is smaller than the raw
   excess. Characterize WHICH contexts overshoot (rare-context? high-entropy region? specific groups?
   Lane specifically?). A mechanism aimed at a context class the mixer already handles is empty.
5. **Adjudicate against the 42,382 B demand, and against Lane.** State the net recoverable in bytes and
   as a share of demand. Separately state Lane's share of the excess: if the anti-predicted mass
   concentrates in Lane, that is the campaign's first measured object that is simultaneously the worst
   rate class (57.31× mean) and the worst distortion class (GT IoU 0.263, ~19% of d_seg flips), and it
   should be named as such. **Do NOT build the mechanism in this arm** — measuring the excess and its
   net is the deliverable; building is a successor's job with its own gate.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO Metal fires (MAIN-fire-only). NO scorer runs — this is a
  LOSSLESS coding-side measurement; changing how symbols are CODED cannot alter d_seg or d_pose, and
  any candidate that does is a BUG.
- Any candidate re-coding admitted must decode back to TO2's exact 117,964,800-byte array
  byte-for-byte. That inversion check is what makes a byte number real.
- **rule-118 is the boundary:** a mixer member the receiver runs generically is FREE; a stored escape
  table or per-position hint is COUNTED. Price net, never gross.
- Shipped receiver bytes are CUSTODY — never edit in place.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): the excess field, per-class masks, and every priced coder output persist
  with sha256 + bytes. Scalar-only artifacts while the fields exist in memory are forbidden AT THE
  TYPING MOMENT.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_ae1_anti_predicted_excess/` — NOT APDataStore
  (~11 GiB free).** Say which tier you used.
- File ownership: BL1 owns the cost field (REUSE, cite, do not re-derive) · MS9 the seg split · MST1 is
  concurrently splitting the manufactured seg loss by stage · XS1 cross-section conditioning. Do not
  duplicate them.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- **`ddm_ma1_model_axis_miss_cost_20260819.md` — the MISS-COST RESERVOIR axis has been worked before** and reached the pointer via
  the built splice step recorded in `ddm_to1_tail_override_twelfth_move_20260819.md` — whose frontier line
  is SUPERSEDED (seven pointer moves have landed since; cite it as HISTORY, never as the live bar, which is
  the dx2 arithmetic in OPTIMAL FORM below). That was a DIFFERENT body and
  possibly a different formulation. **RECALL ma1's memo before designing anything**; if the
  worse-than-uniform question is already answered there, say so and re-scope this arm to the delta
  rather than re-running it. Re-discovering the ma1 result would be the cardinal sin here.
- `ddm_bl1_per_position_bit_allocation_20260822.md` — the parent. Its allocation is *"not a lower bound,
  a new representation, or a score."* Same discipline applies: an excess mass is an ALLOCATION FACT,
  not a saving, until a real coder produces fewer real bytes on the exact array.
- `ddm_cx3_context_axis_ceiling_20260822.md` — the named conditional-entropy ladder returns **0 B**;
  best model-inclusive challenger 125,210 B, hindsight ideal 117,224 B already worse than shipped
  BEFORE model cost. Adding a member has a COST; cx3 is the measured proof that richer-model claims die
  on their own description length. Your escape member must pay for itself in the same column.
- `ddm_ef1_token_entropy_floor_20260822.md` — best generic estimator **365,322 B, 3.21× worse** than
  shipped; `verdict_scope=FAMILY` twice, **UNKNOWN** for differently-trained HPAC networks. The
  incumbent is a strong domain model; do not model it as naive.
- `ddm_to2_token_ordering_race_20260822.md` — reordering is a SUBSTITUTE for a context model, not a
  complement; nine generic orderings 196–687% worse. Do not reorder.
- `ddm_lq1_lane_quotient_representability_20260822.md` — the Lane-recall oracle recovered 417,267 Lane
  pixels at the price of **+2,755,323 total mismatches (+194%)**. If Lane concentrates the excess, that
  is a LOCATION, not a licence — and this arm is LOSSLESS, so it cannot incur that collateral, but a
  successor tempted to change Lane's REPRESENTATION would.
- `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md` + BL1 §"Exact MS9 Seg-error join" — **the seg×rate
  join is ASYMMETRIC**: 90.7017% of seg errors sit inside the top 1%, but seg errors are only 2.013906%
  of that population and carry 5.265498% of stream bits (**5,990.882 B**). A seg-position-targeted rate
  mechanism captures ≤5,991 B. **Do not pitch this arm as the joint seg+rate lever** — the shared object
  is the Lane CLASS, not seg-error position identity.

## OPTIMAL FORM

- **REFERENCE FORM (cited): the shipped 19-member HPAC context law over 190 groups, EXTENDED by one
  member** — an escape/uniform mixture component, which is the standard cure in context modeling
  (PPM-style escape, Krichevsky–Trofimov smoothing, mixing against a uniform prior). Same model class,
  one member richer, priced with its own description cost. **Substituting a different or weaker model
  family is a MECHANISM reduction and is FORBIDDEN** — I made exactly that error today and filed it at
  the self-audit filed against `ddm_ef1_token_entropy_floor_20260822.md`, whose
  charter raced general-purpose context-mixing compressors against this same domain-tuned model; the cure
  is to extend the incumbent, never to replace it with something more convenient.
- Family exemplar for CONDUCT: `ddm_bl1_per_position_bit_allocation_20260822.md` — it reconciled its
  instrument to the physical stream and explained the 56-bit residual rather than hand-waving it,
  refused to call its allocation a bound, and reported the MS9 join in BOTH directions including the
  direction that weakened its own story. Match that, especially the last part.
- **n600, all 117,964,800 positions.** BL1's field already covers them; a subset here would be a
  self-inflicted wound.
- VERIFIED ARITHMETIC (re-derive once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 0.1200996 · seg 0.02013906 · pose 0.0079812. S<0.12 needs ≤137,986 B → shed **42,382 B**;
  6.658e-7 S/B. Token stream 113,777 B = 910,216 bits over 117,964,800 positions = 0.007715996636
  bits/position. **log2(5) = 2.321928094887362.** Top 0.1% = 117,964 positions carrying ~60,246 B at
  ~4.0857 bits/position ≈ **1.76× uniform**. Lane = 305,463.969 bits = **38,183 B = 90.1% of demand**.
- **PRIOR-LAW PREDICTION (falsifiable):** a material mass sits above log2(5) — the excess
  `Σ max(0, cost_i − log2 5)` exceeds **10,000 B gross**, and it concentrates in **Lane** (>50% of the
  excess in a class holding 0.5856% of area). A learned uniform/escape member added to the incumbent's
  mix recovers **>5,000 B NET** of its own description-length growth, decode-identical, at zero stored
  per-position side information.
  **FALSIFIER:** gross excess < 5,000 B, OR the net after signalling/model-growth is ≤0 ⇒ the incumbent
  mixer already respects its own alphabet bound wherever it matters, the worse-than-uniform reading is
  empty on this stream, and the anti-predicted-excess family is CLOSED on measured evidence. Report
  either plainly with the exact excess in the FIRST line — and if `ma1` already answered this, say THAT
  first and re-scope to the delta.

## DELIVERABLE

`.omx/research/ddm_ae1_anti_predicted_excess_20260822.md` — BL1's concentration figures and Lane row
reproduced + the ma1 recall verdict (already-answered / partially / open) + the exact count of
positions above log2(5) and their **gross excess** in bits and bytes, per class with **Lane on its own
row**, per frame, per group + the **signalling/model-growth price in its own column** + the **NET** and
its share of the 42,382 B demand + the characterization of WHICH contexts overshoot and why the
19-member mixer does not already cap them + the verdict on the prior-law prediction with verdict_scope
at the NARROWEST level the evidence supports. Every admitted re-coding inverts to TO2's exact array. No
mechanism build. Commit via the serializer. End with the own-vehicle frontier line.
