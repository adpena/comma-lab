# ddm_ad2_addressing_cost_decomposition — three arms died or survived on ADDRESSING without naming it; measure it as its own object and test whether the receiver can derive it for free

## MANDATE

Three independent arms converged on one axis this session without coordinating, and none of them
named it:

| arm | addressing cost | payload cost | dominated by |
|---|---:|---:|---|
| DC1 sparse-grid Family A (CLOSED) | position field **227.4 KB** + per-block length table **101.8 KB** | — | **addressing** (it is why the family died) |
| RC1 K=2,048 (LIVE candidate) | spatial map **10,900 B** = 18.2% of the 59,884 B counted payload | codebook 48,920 B | **codebook** |
| NR1 K32 (LIVE candidate) | QPAIR **52,124 B** = 75.5% of the 69,004 B packet | QPARAM 239 + QCTX 152 B | **addressing** |

DC1's own closing words name it: *"An implicit shared task-cell certificate could avoid the
measured 227.4 KB position field; this is plausible because location metadata was the largest
debt"* and *"A quotient/certificate representation could amortize the 101.8 KB per-block length
table if widths are receiver-derived rather than stored."*

**Why this is worth an arm now.** Addressing is the one line item that rule-118 can make FREE:
positions a receiver DERIVES from a generic algorithm cost zero counted bytes; positions it must
be TOLD cost the largest single number we have measured. The two live candidates sit at opposite
ends of this axis (18.2% vs 75.5%), which means the cheapest next lever is DIFFERENT for each —
and nobody has computed either. This is also the prerequisite for RC1's own hypothesis 4 (the
RC1×NR1 hybrid: *"the two routes spend bytes on different objects"*).

## SCOPE

1. **Verify inherited state, refuse on drift.** RC1 payload sha `eab66bad…e61164` + result
   `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/RESULT.json` · NR1 K32 result
   `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1/RESULT.json` ·
   modules `src/tac/optimization/rc1_terminal_program_vq.py`,
   `src/tac/optimization/nr1_taskcell_quotient.py`. Reproduce the four section numbers above
   from the receipts; report any disagreement as the finding.
2. **Decompose ADDRESSING vs PAYLOAD per representation, with denominators.** For RC1, NR1-K32,
   and the DX2 incumbent, split every counted stream into (a) WHAT is stored and (b) WHERE/HOW-MUCH
   it goes — indices, positions, lengths, block sizes, per-cell assignment. State each as bytes and
   as a fraction of its own packet (m50: a count without its denominator is not a measurement).
   Some streams will not split cleanly; say which and why rather than forcing a number.
3. **Price the addressing against its own entropy floor.** An addressing stream that already
   codes near its empirical conditional entropy has no headroom regardless of how large it looks —
   size is not slack. Compute, per addressing stream, coded bytes vs a real memoryless AND a real
   context-conditioned bound. A large-but-incompressible field is a CLOSED row, and saying so is
   a complete result.
4. **Test receiver-derivability, and price it honestly.** For each addressing stream with measured
   headroom, determine whether the receiver can DERIVE it from already-decoded content by a
   GENERIC algorithm (raster order, run-structure, geometric prior, sorted-canonical order,
   implicit-from-payload). **rule-118 is the hard boundary: a generic algorithm in the receiver is
   FREE; a video-derived table moved into inflate.py to dodge the rate term is the hide-data-in-code
   FAKE and is forbidden.** Every "derivable" claim states the exact generic rule and what it reads.
   If a stream needs any video-derived side information, its cost is COUNTED, not free.
5. **Report per-candidate, not as one number.** RC1's cheapest addressing lever and NR1's are
   different by construction (18.2% vs 75.5%). Emit a per-candidate ranked table with measured
   byte deltas, and hand MAIN a sealed fire-order for any change large enough to re-close.
   Do NOT modify either live candidate's payload — RI1 and NI1 are measuring them right now.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO scorer runs. NO Metal fires (MAIN-fire-only). NO local
  advisory launches — this arm is scorer-free by construction; the single n600 lane is contended
  by two live arms.
- The jo1 r9 run directory is SACRED. r9 terminated by SELF-REFUSAL (`EXACT_DELTA_NONNEGATIVE`);
  there is no improved endpoint. Work from the current DX2 body and the two live payloads.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES and DESIGNS; it does
  not re-cut either live candidate's payload.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every decomposition table, entropy-bound computation, and losing
  derivation variant persists with sha256 + bytes. Scalar-only artifacts while bytes exist in
  memory are forbidden AT THE TYPING MOMENT.
- Receipts to `/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/`.
- File ownership: RI1 owns RC1's distortion measurement · NI1 owns NR1-K32's · CB2 owns RC1's
  dictionary re-weighting. Do not touch their memos or retained trees; CITE their landed rows.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_dc1s_sparse_grid_sweep_20260821.md` — sparse-grid Family A CLOSED at full n600 FX5 scope:
  **388,326 B actual vs the 113,777 B member, a 274,549 B loss, ALL 190 groups negative.** Also
  closed there: monolithic hash/free-run (26.2–27.9 reachable-rank-bit wall) · block-size retuning
  (variable selection gained only 1,434 B over fixed b=8) · independent adjacent-group
  factorization (later conditional rows depend on earlier decoded observations). Do NOT re-open
  explicit-position sparse grids; this arm asks the INVERSE question.
- `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 —
  **0 B tested headroom across all seven DX2 archive streams at fixed distortion.** That verdict
  is scoped to the DX2 body's existing streams; RC1 and NR1 are DIFFERENT representations with
  different stream anatomies. Do not treat rb1 as closing this question, and do not treat this
  arm as reopening rb1's.
- `ddm_rc1_rate_crush_20260822.md`=dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d —
  fixed-RC64 coder/context races are CLOSED (the composable 88 B ceiling is already shipped and
  cannot supply 42,382 B). This arm is about WHAT IS CODED, not which coder codes it. Also:
  *"Overall token agreement cannot be promoted as evaluator evidence"* — no agreement statistic
  is admissible as a distortion claim here either.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** There is NO retained
  token-level sensitivity corpus. Consequence: do not assume any addressing stream is inert;
  derive or measure.
- `ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd —
  UNION ≠ SUM OF LEGS, measured at 3.705× in this campaign. Addressing savings across streams do
  NOT add; any composed figure is an UPPER BOUND and must be labelled so on its face.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_rb1_rate_bound_decomposition_20260822.md`,
  sha fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 — the per-stream anatomy
  that turned "the archive is too big" into seven numbered streams with tested headroom each.
  This arm does the same decomposition on a DIFFERENT axis (addressing-vs-payload) across THREE
  representations. Match that bar: every row a number, every number a receipt.
- VERIFIED ARITHMETIC (check once, then use): pointer DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.120100 · seg 0.020139 · pose 0.007981. Distortion 0.028120 →
  S<0.12 needs archive ≤ **137,986 B** (STRICT inequality ⇒ FLOOR of 137,986.8388, not round-up)
  → shed **42,382 B**. Exchange 0.001 S distortion = 1,502 B; 6.658e-7 S/B. RC1 sits 24,980 B
  under the ceiling; NR1-K32 sits 2,391 B under.
- SCOPE reductions declared per row (scorer-free is a declared SCOPE reduction; retained-receipt
  derivation is legal — say so). MECHANISM reductions FORBIDDEN — pricing an addressing stream
  against a bound you did not actually compute, or calling a stream receiver-derivable when the
  derivation needs video-derived side information, are the two fakes this charter refuses.
- **PRIOR-LAW PREDICTION (falsifiable):** addressing is a MINORITY of RC1's packet (18.2%) and a
  MAJORITY of NR1-K32's (75.5%), so the cheapest remaining rate lever is candidate-specific —
  codebook-side for RC1, addressing-side for NR1 — and at least one of NR1's addressing streams
  carries ≥20% headroom against its context-conditioned entropy bound, because QPAIR is a
  per-pair assignment field with strong spatial and temporal structure.
  **FALSIFIER:** if every addressing stream in both candidates codes within ~5% of its
  context-conditioned bound, the addressing axis is CLOSED across the live representations and the
  campaign must find its 42,382 B in the payload half — report that plainly; it is a complete and
  campaign-directing result, and it would retire a whole axis honestly.

## DELIVERABLE

`.omx/research/ddm_ad2_addressing_cost_decomposition_20260822.md` — the addressing-vs-payload
split per representation (bytes + fraction + denominator, unsplittable streams named) + coded-vs-
entropy-bound per addressing stream + the receiver-derivability verdict per stream with its exact
generic rule and rule-118 adjudication + the per-candidate ranked lever table with measured byte
deltas + any sealed MAIN fire-order + the explicit verdict on the prior-law prediction with
verdict_scope at the NARROWEST level the evidence supports. Commit via the serializer. End with
the own-vehicle frontier line.
