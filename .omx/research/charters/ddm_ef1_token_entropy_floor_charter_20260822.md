# ddm_ef1_token_entropy_floor — four lossless axes now measure 0 B on the shipped token stream; measure whether ANY predictor could supply the demand, or close the route permanently

## MANDATE

Four independent arms have now closed four independent lossless axes on the SAME 113,777 B token
member of the DX2 archive:

| axis | verdict | arm |
|---|---|---|
| coder search | **0 B** across all seven archive streams | RB1 |
| addressing | **0 B — already free** (implicit raster sites) | AD2 |
| serialization order | 9 generic forms, **196.07%–686.94% WORSE** | TO2 |
| context model (named summaries) | **0 B**; best challenger 125,210 B model-inclusive, and its **hindsight IDEAL data term is 117,224 B — already 3,447 B worse than the shipped stream before any model cost** | CX3 |

**The reframe those four force.** The stream codes 117,964,800 token positions in 113,777 B =
**0.007716 bits/position**. Shedding the campaign's required 42,382 B means reaching
**0.004842 bits/position — 62.7% of current**. At sub-hundredth-bit density the coder is not leaving
bits on the table; the residual bits ARE the surprises, and cutting 37.3% means PREDICTING 37.3% more
of what already surprises a learned 19-member HPAC context law that beats every named alternative
including hindsight-ideal ones. This stopped being a coding problem and became a prediction problem.

**The one unmeasured question, and why it is decisive.** CX3 scoped its negative to FORMULATION and
named exactly what survives: *"a differently trained HPAC network, a new learned representation, or a
model that consumes the continuous five-class probability vector rather than the named summaries."*
Every one of those is a bet on a BETTER PREDICTOR. Nobody has measured whether a better predictor can
exist. **If this field's conditional entropy under a rich/unbounded-context estimator lies ABOVE
71,395 B, then no predictor of any kind supplies the demand and the lossless token route is closed at
the INFORMATION-THEORETIC level, not merely at the formulation level.** That verdict — in either
direction — redirects the campaign, and it costs no scorer, no GPU, and no bytes.

## SCOPE

1. **Verify pins, reuse the decoded field, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · TO2's decoded token
   field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` (117,964,800 B) · TO2's
   checkpoint receipt `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` · CX3's memo.
   **REUSE the decoded array** — TO2 and CX3 both did; a third decode is waste. Reproduce the
   incumbent 113,777 B and the 0.007716 bits/position figure before measuring anything.
2. **Estimate the field's conditional entropy under progressively RICHER estimators than CX3's named
   summaries.** CX3 raced hand-named conditioning sets; this arm races ESTIMATORS. At minimum:
   a strong general-purpose context-mixing compressor over the raw symbol array (real coder, real
   bytes) · a high-order adaptive model whose order is pushed until it stops improving · a
   compression-based entropy-rate estimate with its convergence curve shown. Each row: real coded
   bytes, the estimator named exactly, and its own model/description cost stated separately.
   **Report the CURVE, not one point** — the shape (still descending vs flattened) is the finding.
3. **State the tightest defensible LOWER bound and label its nature.** An achieved compressed size is
   an UPPER bound on the entropy, never a lower one. Say plainly which of your numbers are achieved
   sizes (upper bounds), which are estimates, and what — if anything — supports a lower bound. **Do
   not present an estimate as a floor.** If no defensible lower bound exists, say so; that is itself
   the honest answer to "is 71,395 B reachable in principle?"
4. **Adjudicate against 71,395 B explicitly.** Three outcomes, all complete results: (a) some estimator
   reaches ≤71,395 B ⇒ the demand IS reachable losslessly and the campaign's job is to build that
   predictor into a receiver — name it and price its decode cost against the 30-min budget;
   (b) estimators flatten well ABOVE 71,395 B with a converged curve ⇒ report the flattening value and
   the evidence for convergence, and state the lossless token route closed on measured evidence;
   (c) curves still descending at the compute you could afford ⇒ INCONCLUSIVE, and say what would
   settle it. Refusing to pick when the evidence does not support a pick is correct.
5. **Price any winner's decode side.** A predictor that supplies bytes but cannot run inside the
   30-minute decode budget on contest hardware supplies nothing. Any candidate you advance carries a
   measured or honestly-bounded decode-cost line.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO Metal fires (MAIN-fire-only). NO scorer runs — LOSSLESS
  by construction; d_seg/d_pose/rendered output are byte-identical under any re-coding of this field.
- Any candidate whose coded form is admitted MUST decode back to TO2's exact source array,
  byte-for-byte. That inversion check is what makes a byte number real.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every coded payload, every losing estimator, every convergence curve
  persists with sha256 + bytes. Scalar-only artifacts while the arrays exist in memory are forbidden
  AT THE TYPING MOMENT.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_ef1_token_entropy_floor/` — NOT APDataStore
  (~11 GiB free).** Say which tier you used.
- File ownership: TO2 owns the decoded-field checkpoint and the ordering race · CX3 owns the named-
  summary ladder · AD2 owns the addressing decomposition · RB1 owns the coder race. CITE their rows.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_cx3_context_axis_ceiling_20260822.md` — **0 B from the named conditional-entropy ladder**; best
  model-inclusive challenger 125,210 B (+11,433 B), hindsight ideal 117,224 B (+3,447 B before model
  cost). Do NOT re-race hand-named causal summaries; that cell is measured. This arm races ESTIMATORS,
  which is the axis CX3 explicitly left open.
- `ddm_to2_token_ordering_race_20260822.md` — nine exact-invertible orderings × three coders, all
  196.07%–686.94% worse. The incumbent is an RC64 arithmetic stream under a learned 19-member HPAC law
  (frame-outermost → 190 groups `g=(x mod 64)+2*(y mod 64)` → raster). Reordering is a SUBSTITUTE for
  a context model, not a complement — measured twice today (AD2's Brotli stream won 34.5% from
  reordering; this modeled stream lost 196%). Do not reorder.
- `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 —
  **0 B tested headroom, all seven streams, fixed distortion.** Swapping coders under the existing
  model is RB1's axis and its answer is already 0 B.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` —
  both byte-feasible LOSSY re-representations measured DEAD (d_seg 43.66× and 247.71× over ceiling).
  The measured amplification exponent between them is **16.69** (token disagreement ×1.0975 → d_seg
  ×4.7242): *which* tokens differ dominates *how many* by more than an order of magnitude. **Any lossy
  step in this arm inherits a 44×–248× prior against it — this arm is LOSSLESS ONLY.**
- `ddm_lq1_lane_quotient_representability_20260822.md` — a full-Hamming ORACLE assignment over RC1's
  codebook removes only 16.6% of mismatches ⇒ 83.4% of that family's error is REPRESENTATIONAL. A
  perfect encoder does not rescue a representation that lacks the content.
- `ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd —
  UNION ≠ SUM OF LEGS, measured 3.705×. No composed figure from summed legs is admissible.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_cx3_context_axis_ceiling_20260822.md` — it reproduced all four
  charter pins before measuring, reused TO2's decoded array instead of re-deriving it, kept model cost
  in its own column so no ideal figure could masquerade as a shippable size, reported that its best
  challenger's IDEAL already lost, and scoped the negative to FORMULATION while naming precisely what
  survives. Match that bar, including the naming-what-survives clause.
- VERIFIED ARITHMETIC (check once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.1200996 · seg 100·0.00020139 = 0.020139 · pose √(10·6.37e-6) = 0.0079812.
  Distortion 0.028120 → S<0.12 needs archive ≤ **137,986 B** (STRICT ⇒ FLOOR of 137,986.8388) → shed
  **42,382 B**. Token member 113,777 B ⇒ target **71,395 B** = 0.004842 bits/position. THE PHYSICS
  BOUND (jx1 §5.2): zeroing BOTH distortion axes still leaves rate above 0.12, so a lossless cut is the
  only kind of move that works without also solving distortion.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — presenting an achieved
  compressed size as a LOWER bound, or reporting an estimator's ideal without its model cost, are the
  two fakes here. An upper bound is not a floor and must never be labelled one.
- **PRIOR-LAW PREDICTION (falsifiable):** richer general-purpose estimators land BELOW CX3's named
  summaries but still ABOVE the shipped 113,777 B or within a few percent of it, and the order-vs-size
  curve FLATTENS well above 71,395 B — because the incumbent's learned 19-member context law is
  already a strong domain-specific predictor and CX3 measured its hindsight-ideal competitors losing.
  Consequence: the lossless token route closes on measured evidence and sub-0.12 requires a different
  token field, not a better code for this one.
  **FALSIFIER:** any real, invertible estimator reaching **≤71,395 B** ⇒ the demand IS losslessly
  reachable, the campaign's next job is building that predictor into a 30-min-budget receiver, and
  that must be the memo's FIRST line with the exact bytes, the estimator named, and its decode cost.

## DELIVERABLE

`.omx/research/ddm_ef1_token_entropy_floor_20260822.md` — the reproduced incumbent + bits/position +
the estimator race table (real coded bytes, estimator named exactly, model cost in its own column,
every admitted form inverted to TO2's exact array) + the order/complexity-vs-size CONVERGENCE CURVE
with its shape read honestly + the explicit upper-bound-vs-estimate-vs-lower-bound labelling + the
71,395 B adjudication under outcome (a)/(b)/(c) + decode-cost line for any advanced candidate + the
verdict on the prior-law prediction with verdict_scope at the NARROWEST level the evidence supports.
Commit via the serializer. End with the own-vehicle frontier line.
