# ddm_tba1 — the token stream is the ONLY block bigger than the demand (268.5%) and six arms measured it in AGGREGATE: where do its 910,216 bits actually go? (owning memo: ddm_fb1_sub012_feasibility_bound_20260823.md)

## MANDATE

Operator 20260823: *"Orchestrate a swarm of opus subagents brother and continue with all"* —
routed from `ddm_fb1_sub012_feasibility_bound_20260823.md` (commit `9c137a91ed`).

fb1 derived the structural fact that reframes this block: **the token stream (113,777 B) is the
ONLY block on the dx2 object larger than the 42,382 B demand — 268.5% of it.** Renderer 72.8%,
carrier 51.9%, HPAC model 31.9% are each insufficient *even at zero bytes*. So either a route
composes ≥2 axes, or it cracks this one block.

Five concordant arms (oe1, ld1, ae1, ni1, wj1) measured this block at a **sharp local optimum in
every direction they tested** — and that qualifier is the opening. `ddm_fb1_sub012_feasibility_bound_20260823.md` §3 records the gap
plainly: **nobody asked where the stream's 910,216 bits go, and the arithmetic coder already
knows exactly** (per-symbol −log2 p is available at encode time for free). Aggregate byte totals
cannot distinguish "sharp everywhere" from "sharp in the five directions tested."

This arm produces the bit-level attribution. It does not propose a coder change.

## SCOPE

1. **Emit the per-symbol cost.** Instrument the LIVE encode path for the dx2 object (archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B; categorical
   field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`) and retain the
   per-symbol −log2 p vector. Verify the emitted bits SUM to the measured stream bytes within
   coder overhead, and state the residual. **If they do not sum, that is the finding — stop.**
2. **Attribute the bit mass.** Where do the bits go, by: SegNet class (canonical comma10k order
   Road/Lane/Undrivable/Movable/MyCar — SELF-DETECT by spatial/static signature, never hardcode
   the index) · spatial position · temporal pair · context/expert bucket · symbol value. Report
   concentration (Gini, top-1%/top-10% mass) per axis.
3. **Join cost to HARM.** ar1b's pattern-of-patterns law says what costs most is what hurts most.
   Test it AT BIT GRANULARITY on this stream: is the expensive bit mass the same mass that
   carries seg error / manufactured error? Report the enrichment with its independence baseline
   — wj1 (`ddm_wj1_*` cost↔error enrichment 90.96× count / 257.48× bit) measured 90.96× count / 257.48× bit enrichment on a related join; reproduce or
   contradict it at this granularity and say which.
4. **Test the sharpness qualifier.** From the attribution, name directions the five prior arms
   did NOT test. For each, state whether the attribution predicts it is also sharp, and why.
   This is a DERIVATION from the bit map, not a new coder race — do not nominate a coder.
5. **Price everything** via the exchange rate **6.658590e-07 S/B — CITE
   `ddm_tx1_toolbox_crosswalk_20260819.md` §0, do NOT re-derive** (the exchange-rate false-novelty catch: `ddm_tx1_toolbox_crosswalk_20260819.md` §0 already derived it). Report in BOTH
   currencies per `ddm_tl1_teacher_ledger_20260822.md`.
6. Any rate claim is a MEASURED re-encode, never a −log2 p estimate: `ddm_fs2_rc4_drop_carrier_resolve_20260820.md`
   measured that −log2 p is DIRECTION-DEPENDENT and ranker-based prices were 0.77–0.88× wrong
   away from argmax and 0.09× toward it. The bit map is a MAP, not a price.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- NO Metal slot: `ddm_jf1` holds it. Local CPU + derivation only.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD — the per-symbol bit vector is a PAYLOAD, not a scalar. Persist it to
  `/Volumes/APDataStore/pact/ddm_tba1_token_bit_attribution/` with sha256 + bytes. A run that
  reports only summary statistics while holding the vector in memory violates the P0 rule.
- File ownership vs the three parallel arms: `ddm_msr1` owns the manufactured-seg surfaces,
  `ddm_dg2` owns the jf1/diagonal surfaces, `ddm_tac1` owns the composition tables. jf1's
  receipts under `.omx/tmp/arm_receipts_local/ddm_jf1_*` are SACRED read-only.
- Every negative-existence claim states its SEARCH SCOPE or is not made (m53).
- Export `PYTHONDONTWRITEBYTECODE=1` (AppleDouble/`__pycache__` sidecars on ExFAT trip `src/tac/contest_compliance.py`; cure landed `ae0a8bb7b1`).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- **THE SHARP-OPTIMUM LAW** — `ddm_oe1_*`, `ddm_ld1_*`, `ddm_ae1_*`, `ddm_ni1_*`, `ddm_wj1_*`:
  five concordant arms, the HPAC model and field jointly at a local optimum, sharp in every
  measured direction. This arm's job is to MAP, not to assume an escape exists.
- `ddm_ad2_addressing_cost_decomposition_20260822.md` + `ddm_to2_token_ordering_race_20260822.md`
  — reordering is a SUBSTITUTE for a context model: ad2 +34.5%, to2 −196%. One law.
- `ddm_fs2_rc4_drop_carrier_resolve_20260820.md` — −log2 p prices are direction-dependent and
  were 0.77–0.88× / 0.09× wrong. Never price from the model; re-encode.
- `ddm_bl1_*` (Lane: 33.56% of model bits at 0.59% area) — Lane is 0.59% of area but 33.56% of MODEL bits, top-1% of positions carry
  96.32% of bits, Gini 0.995. Extreme concentration is the PRIOR here, not a discovery.
- `ddm_sv2_*` — SMEVR LOSES on the IX2TOK01 token bulk (+5,183 B): the live coder pays for
  LZ MATCH STRUCTURE, not symbol rank. A bit map built on symbol rank alone will mislead.
- `ddm_ni1_*` (247.69× over bar) + `ddm_nr1_*` (349× over bar) — whole-body lossy CLOSED on two authority rows.

## OPTIMAL FORM

- **PROVENANCE PINS (verify before reading; do not work from memory):**
  - routing memo: `.omx/research/ddm_fb1_sub012_feasibility_bound_20260823.md`, commit `9c137a91ed`
  - the residue map: `.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md`, commit `e864cb4ab4`
  - the composition law: `ddm_sy2_composition_synergy_deep_pass_20260823.md`, commit `fe2ba12dc2`
  - the renderer refusal: `.omx/research/ddm_w72_distortion_advisory_20260823.md`, commit `637af0c8c1`
- Family exemplar: the **reference** form for a concentration measurement on this object is bl1 — per-position bit attribution with Gini and top-k mass, joined to a distortion axis.
  Mechanism properties that may NOT be dropped: the ACTUAL live encoder's per-symbol costs (never
  a re-implemented model) · SUM-verified against measured bytes · concentration reported with its
  independence baseline. Dropping any is a MECHANISM reduction requiring an explicit TOY-BRACKET
  declaration, after which the row cannot produce a family verdict.
- SCOPE reductions LEGAL, declared per row: a pair subset (SEEDED RANDOM, never a prefix — m88/m96
  measured prefix bias 2.5–4.2× on pose, ~0.96× on seg, ~neutral on rate; and the prefix-advisory runtime guards measured that
  prefix ADVISORIES are jointly unsatisfiable on this lineage) · one attribution axis at a time.
- **PRIOR-LAW PREDICTION (falsifiable).** bl1's Gini 0.995 and ar1b's pattern-of-patterns both
  predict extreme concentration AND cost↔harm alignment. I predict **the bit mass is concentrated
  (top-10% of symbols carry >70% of bits) AND the expensive mass is enriched in seg-carrying /
  manufactured-error positions by >10× over independence.** FALSIFIER: near-uniform bit
  distribution (top-10% < 25% of mass), OR enrichment < 2× — either would mean the stream is
  structurally different from every other surface on this object and the sharp-optimum reads
  would need re-examination. Report the number plainly either way; a CONFIRMED concentration with
  no actionable direction is still the decision-relevant answer and must be said in those words.

## DELIVERABLE

`.omx/research/ddm_tba1_token_bit_attribution_20260823.md` — typed rows: the SUM verification with
its residual; bit mass by class/position/pair/context/value with Gini + top-k per axis; the
cost↔harm join with its independence baseline and enrichment factor; the untested-directions
table with per-direction sharpness prediction and reasoning; both currencies at the CITED exchange
rate; the prior-law prediction adjudicated CONFIRMED/REFUTED with its number; `verdict_scope` on
the narrowest rung the evidence supports; a `STORES CONSULTED:` line (the contract's literal key)
naming what was loaded, honestly including "none" where none. Commit via the serializer. End with
the own-vehicle frontier line.
