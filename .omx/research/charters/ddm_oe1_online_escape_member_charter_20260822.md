# ddm_oe1_online_escape_member — AE1 measured 26,645 B of anti-predicted excess and killed both STATIC recovery routes on their signalling cost; measure the ZERO-STORED CAUSAL ONLINE member it named as unmeasured

## MANDATE

Routed finding, one memo, no operator verbatim.

**AE1 found the mass and closed only the routes that cost bytes to describe.**
`ddm_ae1_anti_predicted_excess_20260822.md` measured, exactly, on the dx2 pointer:

- **93,580 / 117,964,800 positions** (0.0793% of the field) cost MORE than `log2(5)` = 2.3219 bits.
  Their gross excess is **213,162.383261 bits = 26,645.297908 B = 62.8694% of the 42,382 B demand.**
  The model *anti-predicts* them: it codes them worse than a uniform 5-symbol prior would.
- **Route 1 — explicit stored flags: NET −103,582.702092 B.** The best real flag encoding of the
  overshoot mask costs 130,228 B, ~4.9× the gross it could recover. Serialized in inherited 190-group
  event order, coded with raw / Brotli q11 / zlib9 / raw LZMA1, every output with a byte-identical
  deterministic repeat that decodes back to the exact mask.
- **Route 2 — counted static uniform overlay: NET −14 B (global, selects alpha=0) and −34.468770 B
  (190-group: gains 10.531230 modelled bytes, costs 45 real descriptor bytes).**

**The mechanism that kills both routes is stated in ae1 §"Signalling and model-growth price": helping
93,580 rare misses perturbs the 117,871,220 positions that are ALREADY BELOW the uniform cost.** A
blanket blend pays everywhere to help almost nowhere. Any static description of *where* to blend costs
more than the blend recovers.

**AE1 explicitly recorded its own scope: `verdict_scope` is INSTANCE for the exact excess and
FORMULATION for "the two counted static uniform-overlay models" — and its own words are "The zero-stored
causal online-member family remains unmeasured."** That family is this arm's mechanism, and it is the
one shape that escapes the mechanism above: a member whose weight is DERIVED at decode time from
already-decoded history costs **zero stored bytes**, so it has no signalling term to lose to, and it can
be SELECTIVE in time — rising only where the model has recently been wrong, instead of blending
everywhere.

**Why the mass is plausibly reachable.** MA1's landed `free_corrector.py:251-288` reweights the four
non-argmax columns while preserving their total miss mass — it learns relative miss classes but, per
ae1, "still contains no hard `p_k >= 1/5` rule." Nothing in the shipped stack enforces the alphabet
bound. That is precisely the gap a causal escape/KT-style member fills, and it is the standard cure in
the context-modelling literature (PPM escape, Krichevsky–Trofimov smoothing, adaptive mixture weights).

**This is a LOSSLESS arm and must be lossy-free on its face.** Adding a mixture member changes only the
RC64 coding probabilities. The decoded token field stays bit-identical by construction, so **d_seg and
d_pose are INVARIANT and MUST NOT MOVE.** A candidate that changes distortion is a BUG, not a result.
(This is the exact law `ddm_lx2` refused on when it was asked to buy distortion from a lossless control;
here it is the arm's ally, because the measurement is purely a rate measurement.)

## SCOPE

1. **Verify pins; REUSE ae1's overshoot field read-only; refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · RC64 token stream
   sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` @ 113,777 B · TO2 decoded
   field `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. Reproduce ae1's
   **93,580** exceeding positions and **213,162.383261 bits** gross before sweeping; a disagreement IS
   the finding. Do NOT re-derive the exchange rate — `ddm_tx1_toolbox_crosswalk_20260819.md` §0 derived
   `25/37,545,489 = 6.658590e-07`; cite it and use it.
2. **Build the ZERO-STORED causal online member and prove it stores zero bytes.** One additional
   mixture member whose weight at each position is a deterministic function of ALREADY-DECODED history
   only (no lookahead, no stored parameters, no side channel). Encoder and decoder run the identical
   update. **Prove the zero-storage claim by exhibiting the byte-identical descriptor set: the archive's
   non-token sections must be unchanged, and any growth must appear only in the coded stream.**
3. **Sweep the adaptation rate, ≥4 rungs, and report NET at every rung.** The member's responsiveness
   (window length / learning rate / escape count — whichever the chosen form exposes) is the swept
   variable. Per rung report **REAL re-encoded stream bytes** vs the 113,777 B baseline, never an
   estimate or a modelled bound. Include the degenerate rung (adaptation → 0, which must reproduce the
   baseline within the coder's own determinism) as an executed positive control.
4. **Report the SELECTIVITY split — the crux.** For each rung, split the byte delta into (a) bytes
   recovered ON the 93,580 anti-predicted positions and (b) bytes SPENT on the 117,871,220 positions
   that were already below uniform. Their ratio is the live selectivity, and it is what decides the
   family. **A rung reporting only a net total has not measured the mechanism** — ae1's kill was
   mechanistic, and only this split can overturn or confirm it on the online form.
5. **Confirm distortion did not move.** State plainly that the decoded token field is bit-identical
   (sha vs the TO2 pin) at every rung. If any rung's field differs, STOP and report it as a bug.
6. **Adjudicate honestly, including the empty outcome.** Report the best rung's net bytes and its share
   of the 42,382 B demand. If every rung nets ≤0, say so plainly: combined with ae1's two static kills,
   the anti-predicted-excess family closes COMPLETELY on measured evidence across both its static and
   its zero-stored causal forms — a complete result about 62.87% of the demand. **Build NO shipping
   candidate here**; the curve and the selectivity split are the deliverable.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). NO Metal fires
  (MAIN-fire-only). Local advisory launches ONLY via `tools/fire_local_advisory.py`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; every rung's re-encoded stream, the per-position excess field, and the
  per-rung selectivity split persist with sha256 + bytes. **Receipts to
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_oe1_online_escape_member/` — BOTH SSD
  TIERS ARE AT 100% (measured 08-22; a write there killed a prior generation of two sister arms at rc=1
  with zero artifacts). Local disk has ~500 GiB free and is the EXPLICIT-OPT-IN destination per the disk
  rule while the tiers are full.** Do NOT write to `/Volumes/*` — a write there will kill you. Say which
  tier you used.
- Shipped receiver bytes are CUSTODY — never edit in place. The jo1 r9 run dir is SACRED
  (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Every distortion statement names its GT lineage (DALI-GT where the tool family expects it).
- File ownership: AE1 owns the excess field · BL1 the per-position cost field · MS9 the manufactured
  split · MST1 is concurrently splitting that loss BY STAGE · LD1 is concurrently measuring the Lane
  lossy drop curve. CITE them; do not duplicate or touch their trees.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ae1_anti_predicted_excess_20260822.md` — **the direct parent.** Both STATIC routes lose net
  (−103,582.70 B explicit flags; −14 B / −34.47 B counted overlays). Do NOT re-attempt a stored-flag or
  a counted static overlay; both are measured dead. This arm exists ONLY because ae1 scoped its own
  verdict to those two static formulations and named the zero-stored causal family as unmeasured.
- `ddm_lx2_lane_bit_budget_exchange_20260822.md` — `REFUSED_BEFORE_SWEEP_NO_DISTORTION_BEARING_CONTROL`.
  The lesson this arm inherits: a probability-only control cannot move distortion. Here that is the
  DESIGN, not a defect — but it also means **no d_seg claim of any kind may be made from this arm.**
- `ddm_ef1_token_entropy_floor_20260822.md` — generic estimators measured **3.21× worse** (365,322 B)
  than the shipped learned model. FAMILY-scoped. Do NOT propose replacing the 19-member law with a
  generic estimator; this arm ADDS one member to the shipped law and leaves the other 19 untouched.
- `ddm_cx3_context_axis_ceiling_20260822.md` (named-summary context ladder **0 B**) ·
  `ddm_to2_token_ordering_race_20260822.md` (orderings 196–687% worse) ·
  `ddm_ad2` (addressing already free) — the ordering, addressing, and named-context axes are closed.
  Do not reopen them; the coder, the order, and the addressing all stay FIXED so the sweep isolates the
  one new member.
- `ddm_bl1_per_position_bit_allocation_20260822.md` (commit `873947c665`) — the cost field this arm
  reads. Top 1% of positions carry 96.323842% of model bits, Gini 0.995159. Note the size relation
  honestly: the 93,580 anti-predicted positions are FAR fewer than the top 0.1% (117,964 positions), so
  "the top 0.1% averages 1.76× uniform" describes a mean over a set in which only a subset actually
  breaches the bound. AE1's exact count supersedes that framing; use 93,580.

## OPTIMAL FORM

- **REFERENCE FORM: the shipped 19-member HPAC context law over 190 groups, EXTENDED by one causal
  online mixture member — the standard cure in context modelling (PPM-style escape, Krichevsky–Trofimov
  smoothing, adaptive mixture weighting against a uniform prior), in its ZERO-STORED form.** Same model
  class, one member richer, nothing removed. The coder (RC64), the group map, the serialization order,
  and the other 19 members are FIXED.
- SCOPE reductions declared per row (a strided pilot to shape the adaptation-rate sweep is legal and
  must be labelled; the verdict is the full field). MECHANISM reductions FORBIDDEN — a generic
  estimator standing in for the shipped mixture, a modelled-bytes-only rung with no real re-encode, or
  a stored-parameter member wearing the "online" name are each the exact defect class that killed the
  static routes and, separately, that `ddm_ef1` and `ddm_to2` were caught by.
- Family exemplar for conduct: `ddm_ae1_anti_predicted_excess_20260822.md` — it reconciled its gross to
  the exact bit, priced BOTH recovery routes with REAL coders and REAL descriptor bytes, refused to
  report a modelled gain as a saving, and stated its own scope narrowly enough that this arm exists.
  Match that, especially the last part.
- VERIFIED ARITHMETIC: DX2 S 0.14821987563243377 @ 180,368 B · rate 25·180368/37545489 = 0.1200996 ·
  seg 100·(23,757/117,964,800) = 0.020139 · pose √(10·6.37e-6) = 0.0079812. S<0.12 needs ≤137,986 B →
  shed **42,382 B**; **6.658590e-07 S/B**. AE1 gross **26,645.297908 B = 62.8694% of demand**;
  exceeding positions **93,580 / 117,964,800**.
- **PRIOR-LAW PREDICTION (falsifiable):** zero signalling cost changes the sign. A causal online member
  recovers **>30% of the 26,645 B gross at zero stored bytes** — **>7,993 B ≈ 18.9% of the demand** —
  because its selectivity ratio (bytes recovered on the 93,580 vs bytes spent on the 117,871,220) comes
  out **above 1.5** at its best adaptation rate, where both static routes were structurally forced below
  1.0 by their descriptor cost.
  **FALSIFIER:** every rung nets ≤0, or the selectivity ratio stays below 1.0 at every adaptation rate.
  Then the perturbation cost on the already-cheap 99.92% of the field dominates regardless of whether
  the blend is described or derived, the anti-predicted mass is UNRECOVERABLE by the whole
  mixture-member family (static AND causal), and 62.87% of the demand is closed on measured evidence.
  **Count it plainly if it lands; both outcomes are complete.**

## DELIVERABLE

`.omx/research/ddm_oe1_online_escape_member_20260822.md` — ae1's 93,580 / 213,162.383261-bit gross
reproduced + the zero-storage proof (unchanged non-token descriptor bytes) + the ≥4-rung
adaptation-rate curve with, per rung: **REAL re-encoded stream bytes** vs 113,777 B · the
**selectivity split (recovered on the 93,580 / spent on the 117,871,220) and its ratio** · decoded-field
sha confirming bit-identity · net ΔS at 6.658590e-07 S/B + the executed degenerate positive control +
the best rung's share of the 42,382 B demand OR the honest all-negative verdict + verdict_scope at the
NARROWEST level the evidence supports. Every figure carries its denominator. No d_seg claim. No shipping
candidate. Commit via the serializer. End with the own-vehicle frontier line.
