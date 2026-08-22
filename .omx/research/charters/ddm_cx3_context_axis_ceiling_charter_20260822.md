# ddm_cx3_context_axis_ceiling — TO2 proved reordering is a SUBSTITUTE for a context model, and the incumbent already has one tuned on someone else's data; measure what the context axis can actually supply before spending another arm on it

## MANDATE

TO2 (`ddm_to2_token_ordering_race_20260822.md`) refuted my ordering charter and, in doing so,
measured the incumbent's real anatomy for the first time:

> "The shipped token member is an RC64 arithmetic stream under a learned 19-member HPAC/context law.
> Decode traverses frame outermost, then groups `g=0..189` defined by `g=(x mod 64)+2*(y mod 64)`,
> then raster positions within each group."

Every generic reorder+generic-coder form measured **196.07% to 686.94% WORSE** than that incumbent.
TO2's own recall supplies the mechanism from a prior measurement (`ddm_gd1_undecided_defaults_audit_20260731.md`):
a matched split on a different object attributed **425 B to context versus 27 B to order** — ~16:1.

**The unifying law this session produced (state it, then test its consequence).** Reordering is a
SUBSTITUTE for a context model, not a complement to one. AD2's QPAIR win (−17,957 B, −34.5%) was on a
**Brotli** stream — a context-free LZ coder — where reordering was the only mechanism available to
expose temporal structure. DX2's token stream already carries a learned context law, so the structure
a reorder would expose is already modeled, and reordering only breaks the contexts. One mechanism
explains both results.

**The consequence, and why it needs measuring rather than assuming.** If order is spent and the model
is the live axis, the campaign's next question is not "which model refit" but **"how much is on this
axis at all?"** The scale gap is stark and must be stated up front: MA1's context work on this lineage
bought **−105 B**; GD1's context leg was **425 B**; the demand is **42,382 B**. A single refit is
~0.25% of the demand. So this arm measures the AXIS CEILING — the achievable conditional-entropy
floor of the token field under progressively richer contexts — and returns a number the campaign can
route on, whichever way it lands.

**The borrowed-constant suspicion (a hypothesis to test, not a premise to build on).** `g=(x mod 64)
+ 2*(y mod 64)` maps 4,096 tile-phase cells onto 190 groups — a heavily colliding hash, e.g. (x=2,y=0)
and (x=0,y=1) both give g=2. That collapse is a model-size-vs-sharpness tradeoff point. It appears
nowhere in our source tree; it is inherited with the PR130/135 receiver and was tuned on THEIR token
distribution. Our field has been through nineteen pointer moves of edits, drops, re-solves and
requantizations since. Whether 190-way collision is still near-optimal for the field it now codes is
UNMEASURED — but it is one candidate inside the ceiling question, not the headline.

## SCOPE

1. **Verify inherited state, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · TO2's decoded token
   field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` (117,964,800 B source
   array) · RC64 stream sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` ·
   TO2's checkpoint receipt sha `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9`.
   REUSE TO2's decoded array — do not re-decode unless the sha fails. Reproduce the incumbent
   113,777 B token-member size before measuring anything against it.
2. **Characterize the incumbent context law exactly.** The 19 members, the group map, what each member
   conditions on, and the model blob's own 13,515 B cost. State the incumbent's realized bits/symbol
   with its denominator (m50). This is the baseline every later row is measured against, and it is a
   MODEL, so its cost is part of the comparison — a richer model that saves stream bytes but grows the
   blob more has lost.
3. **Measure the CEILING, not a point.** Compute the token field's empirical conditional entropy under
   a LADDER of context orders, each with its exact conditioning set named: e.g. order-0 · previous
   symbol in decode order · spatial causal neighbourhood (left/up at several radii) · same-site
   previous-frame (temporal) · joint spatial×temporal · the incumbent's own 190-group conditioning ·
   and a deliberately over-rich context as an upper-bound probe. **Report each as an ideal data term
   with its model-description cost stated separately** — TO2's memo is explicit that an ideal entropy
   figure "is not a shippable size, and is not a universal lower bound." Any row that omits the
   model-cost column is not a measurement.
4. **Adjudicate the ceiling against the 42,382 B demand explicitly.** State the best achievable
   lossless token-stream size across the ladder, its model cost, and the NET vs 113,777 B. Then say
   plainly whether the context axis can supply 42,382 B, some fraction, or ~nothing. A number with its
   denominator is the deliverable; a route recommendation is secondary.
5. **Only if the ceiling is materially below the incumbent, test the borrowed constant.** Re-derive the
   group partition from OUR field's measured structure (context count and collision map chosen by
   measured conditional entropy, not inherited) and report bytes vs the 190-group incumbent, model cost
   included. If the ceiling says there is nothing to get, SAY SO and do not refit — a refit toward a
   ceiling that is already reached is the fake this charter refuses.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO Metal fires (MAIN-fire-only). NO scorer runs — LOSSLESS by
  construction: d_seg, d_pose and the rendered output are byte-identical under any re-coding, and any
  candidate that changes them is a BUG. Every admitted form must decode to TO2's exact source array.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every coded payload, every losing model variant, every context-count
  histogram persists with sha256 + bytes. Scalar-only artifacts while the arrays exist in memory are
  forbidden AT THE TYPING MOMENT.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_cx3_context_axis_ceiling/` — NOT APDataStore (~11 GiB
  free).** Say which tier you used.
- File ownership: TO2 owns the ordering race and the decoded-array checkpoint · AD2 owns the addressing
  decomposition · RB1 owns the coder-race negative. CITE their rows; do not touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_to2_token_ordering_race_20260822.md` — nine exact-invertible generic orderings × three generic
  coders, ALL 196.07%–686.94% worse than the incumbent. verdict_scope INSTANCE, and TO2 explicitly
  refused the larger claim. Do NOT re-run orderings under generic coders; that cell is measured. The
  live question is the MODEL, and TO2's own anatomy is the input.
- `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 —
  **0 B tested headroom across all seven DX2 streams at fixed distortion.** That is the CODER-SEARCH
  axis (which coder codes the existing sequence under the existing model). The standing campaign law
  distinguishes it: *coder-SEARCH closed ≠ probability-MODEL closed*. If your measurement reduces to
  swapping coders under the incumbent model, you have drifted onto RB1's axis and the answer is 0 B.
- `ddm_ad2_addressing_cost_decomposition_20260822.md` — DX2's tokens sit at IMPLICIT RASTER SITES:
  **0 B of addressing**. There is no positional overhead to reclaim on this stream. Also measured
  there: the HPAC MODEL BLOB's own 13,515 B codes 3,027 B (22.40%) above a first-order context bound —
  that is the blob, NOT the token member, and AD2 warns the reference is model-scoped, not universal.
- MA1 context work on this lineage produced **−105 B** (reached the pointer via the to1 splice);
  GD1's matched split attributed **425 B** to context. Both are ~0.25–1% of the 42,382 B demand.
  Treat "the model axis is live" as TRUE and "the model axis is large" as UNPROVEN — that gap is
  precisely what this arm measures.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** No token-level sensitivity corpus
  exists. Irrelevant to losslessness (which comes from the inversion check) but binding on any claim
  that some region "matters less."

## OPTIMAL FORM

- Family exemplar (reference): `ddm_to2_token_ordering_race_20260822.md` — it verified every pin before
  measuring, inverted all nine candidates to the exact 117,964,800-byte source array, decoded all 27
  coded payloads back, checked deterministic repeats, REFUTED its own charter's premise at source and
  said so in the verdict, and scoped its negative to INSTANCE instead of taking the free closure. Match
  that bar exactly, including the willingness to refute this charter.
- VERIFIED ARITHMETIC (check once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.1200996 · seg 100·0.00020139 = 0.020139 · pose √(10·6.37e-6) = 0.0079812.
  Distortion 0.028120 → S<0.12 needs archive ≤ **137,986 B** (STRICT ⇒ FLOOR of 137,986.8388) → shed
  **42,382 B**; 6.658e-7 S/B. Token member = 113,777 B, so the demand is **37.3% of this stream**.
  THE PHYSICS BOUND (jx1 §5.2): zeroing BOTH distortion axes still leaves rate above 0.12 — a lossless
  cut is the only kind of move that works without also solving distortion.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — reporting an ideal entropy figure
  as an achievable size, omitting the model-description cost from any comparison, or refitting the
  group partition after measuring that the ceiling is already reached, are the three fakes here.
- **PRIOR-LAW PREDICTION (falsifiable):** the incumbent RC64+HPAC stream already codes within ~15% of
  the achievable context-conditioned floor at context orders the receiver can afford inside the 30-min
  decode budget, so the context-model axis supplies **< 17,000 B** — under 40% of the demand — and the
  lossless axis on the token stream is CLOSED for sub-0.12 purposes. This follows from MA1's −105 B and
  GD1's 425 B being the only measured context wins on this lineage.
  **FALSIFIER:** a measured, model-cost-inclusive context-conditioned size **≤ 71,395 B** (i.e.
  ≥42,382 B below the incumbent) at an affordable context order ⇒ the lossless model axis ALONE
  supplies the entire sub-0.12 demand, and that is the campaign's answer — it must be the memo's FIRST
  line with the exact bytes and the recomputed-from-components S as a labelled PROJECTION.

## DELIVERABLE

`.omx/research/ddm_cx3_context_axis_ceiling_20260822.md` — the incumbent context law characterized
(19 members, group map, blob cost, realized bits/symbol with denominator) + the conditional-entropy
LADDER with each conditioning set named and its model-description cost in its own column + the
explicit ceiling-vs-42,382 B adjudication + the group-partition re-derivation ONLY if the ceiling
warrants it + the verdict on the prior-law prediction with verdict_scope at the NARROWEST level the
evidence supports. Every candidate inverted to TO2's exact source array before admission. Commit via
the serializer. End with the own-vehicle frontier line.
