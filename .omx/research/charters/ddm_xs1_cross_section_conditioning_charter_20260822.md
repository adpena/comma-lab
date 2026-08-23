# ddm_xs1_cross_section_conditioning — every rate measurement on this archive assumed the sections are independent, and RB1 wrote that assumption down as a property; measure the mutual information nobody could see

## MANDATE

Four arms closed four axes on the DX2 token stream today. **All four measured streams IN ISOLATION**,
and RB1 states the assumption in its own text (`ddm_rb1_rate_bound_decomposition_20260822.md:160`):

> "logical fields independently and exactly, so each physical stream can be optimized in isolation."

That is an ASSUMPTION written as a property. It is true for *decodability*. It is **not** established
for *optimality* — and if the sections share mutual information, then per-stream optimization is
provably not optimal and every 0 B verdict today is scoped to a design that could not see the gain.

**The concrete unexploited structure.** AD2's measured DX2 anatomy:

| region | bytes | archive fraction | what it is |
|---|---:|---:|---|
| semantic tokens at implicit raster sites | 113,777 | 63.0805% | the field CX3 modeled, conditioned ONLY on its own history |
| semantic renderer | 30,856 | 17.1072% | learned map tokens → pixels |
| carrier stream | 22,010 | 12.2028% | mixed payload + basis/coefficient metadata |
| learned HPAC probability model | 13,515 | 7.4930% | **the token model — 19 members over 190 groups** |
| ZIP framing + RX1 header + compact residual | 210 | 0.1164% | — |

The token model conditions on `g=(x mod 64)+2*(y mod 64)` groups and its own decode history. **It does
not read the carrier. It does not read the renderer.** Those are 52,866 B of content derived from the
SAME 600 pairs of the SAME video, describing the SAME scene, and the receiver has them in hand before
it needs the tokens. Reading already-decoded content is a GENERIC ALGORITHM — rule-118 FREE, zero
stored bytes, decode-identical.

**Why this is the right shape and not another generic race (the lesson from my own failure).** TO2
raced generic ORDERINGS × generic CODERS against this domain-tuned learned model and lost 196–687%.
A sister charter fired hours ago races generic ESTIMATORS against it — the same defect, mechanism-
reduced, and it is filed as such (`#1202`). **This arm does not substitute a weaker mechanism.** The
challenger is the incumbent's OWN class — a learned context model — given strictly MORE information.
That is the only honest way to ask whether the incumbent is at its ceiling.

**Honest scale statement, up front.** The demand is 42,382 B. The token stream is 113,777 B, so even a
20% cross-section gain is 22,755 B = 54% of the demand — real, and not sufficient alone. Do not let
the memo imply otherwise. Report what is there.

**The reframe this arm serves (verified arithmetic, use it).** The zero-distortion archive ceiling is
0.12·37,545,489/25 = **180,218.3 B**; we ship 180,368 B. The gap at zero distortion is **149.7 B**.
Equivalently: the 0.028120 of distortion we carry is worth **42,235 B** of archive budget — 99.65% of
the whole gap. So "shed 42,382 B" and "shed 150 B while eliminating all distortion" are the same
demand read two ways, and every byte this arm finds is interchangeable with distortion at
6.658e-7 S/B. State any win in BOTH currencies.

## SCOPE

1. **Verify pins, reuse decoded artifacts, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · TO2's decoded token
   field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` (117,964,800 B) · TO2's
   checkpoint receipt `c0c05971396ff066c16cc0a82a46c5fe3e99a9c0000b4a93933e4bb2a57359f9` · AD2 receipt
   `80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511` · RB1 memo
   `fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09` · CX3's memo. **REUSE TO2's
   decoded array** — a fourth decode is waste. Reproduce the 113,777 B token member and the incumbent's
   realized bits/symbol before measuring anything.
2. **Establish the DECODE ORDER exactly, because it bounds what is legal.** Determine, from the
   receiver, which sections are fully decoded BEFORE the token member is decoded. Only those may be
   conditioned on — conditioning on content that decodes AFTER is circular and inadmissible, and saying
   which is which is a required deliverable. If the order permits only a subset, that subset is the
   arm's scope and the exclusion is a finding, not a failure.
3. **Measure the MUTUAL INFORMATION between the token field and each legally-available decoded
   section.** Per section, with denominators (m50): how much of the token field's residual uncertainty
   — the uncertainty that SURVIVES the incumbent's own 19-member context law — is explained by
   features of the decoded carrier / renderer output / model blob? Report I(tokens ; section | incumbent
   context) in bits and in BYTES over the 117,964,800 positions. **A near-zero number here is the
   complete answer and closes the axis honestly.**
4. **If mutual information is non-trivial, EXTEND the incumbent — do not replace it.** Add cross-section
   context members to the SAME learned-context-model class the incumbent uses (its 19 members over 190
   groups is the reference form; cite it). Re-code the exact token array with the extended model and
   report REAL coded bytes vs 113,777 B, **with the extended model's own description cost in its own
   column** — a richer model that saves stream bytes but grows its blob more has lost, and CX3's best
   challenger lost exactly that way. Every admitted form inverts to TO2's exact source array.
5. **Adjudicate in BOTH currencies and price the decode side.** Net archive delta vs 42,382 B, AND the
   distortion-equivalent at 6.658e-7 S/B. Any candidate advanced carries a measured-or-bounded decode
   cost against the 30-min budget (current measured decode wall: 498 s PASS, so there is real headroom
   — say how much the extension consumes). Emit a sealed MAIN fire-order if the win is real; do NOT
   fire.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO Metal fires (MAIN-fire-only). NO scorer runs — LOSSLESS by
  construction: re-coding the same token array cannot move d_seg or d_pose, and any candidate that
  does is a BUG. Every admitted form decodes back to TO2's exact array byte-for-byte.
- **rule-118 is the hard boundary.** Conditioning on ALREADY-DECODED content via a generic algorithm is
  FREE. Storing any new table, any side information, or any per-position hint is COUNTED — price it
  net, never gross. If an extension needs a stored component, its bytes go in the ledger.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES and DESIGNS.
- The jo1 r9 run directory is SACRED (terminal by SELF-REFUSAL, `EXACT_DELTA_NONNEGATIVE`).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every coded payload, every losing context set, every MI estimate and
  its inputs persist with sha256 + bytes. Scalar-only artifacts while the arrays exist in memory are
  forbidden AT THE TYPING MOMENT.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_xs1_cross_section_conditioning/` — NOT APDataStore
  (~11 GiB free).** Say which tier you used.
- File ownership: TO2 owns the decoded-field checkpoint · CX3 owns the named-summary ladder · AD2 owns
  the addressing/anatomy decomposition · RB1 owns the per-stream coder race. CITE their rows.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 —
  **0 B tested headroom, all seven streams.** Scoped to the CODER axis under per-stream isolation; its
  line 160 states that isolation as the operating assumption. This arm tests the ASSUMPTION, not the
  coder. If your measurement reduces to a different coder on an isolated stream, you have drifted onto
  RB1's axis and the answer is already 0 B.
- `ddm_cx3_context_axis_ceiling_20260822.md` — the named conditional-entropy ladder returns **0 B**;
  best model-inclusive challenger 125,210 B, and its hindsight IDEAL data term 117,224 B is already
  3,447 B WORSE than the shipped stream before model cost. Every conditioning set there was built from
  the token field's OWN history. CX3 explicitly left open "a model that consumes... rather than the
  named summaries" — this arm takes the cross-section reading of that opening. **Do not re-race
  self-history summaries.**
- `ddm_to2_token_ordering_race_20260822.md` — nine generic orderings × three coders, **196.07%–686.94%
  worse**. Reordering is a SUBSTITUTE for a context model, not a complement (measured twice on 08-22:
  AD2's Brotli stream won 34.5% from reordering; this modeled stream lost 196%). Do not reorder.
- `ddm_ad2_addressing_cost_decomposition_20260822.md` — DX2's tokens sit at IMPLICIT RASTER SITES:
  **0 B of addressing**. Nothing positional to reclaim. Its one real win (−17,957 B, QPAIR tile-major
  time) was on a DIFFERENT candidate's Brotli stream and does not transfer here.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` —
  both byte-feasible LOSSY re-representations measured DEAD (**43.66×** and **247.71×** over their
**[MAIN ERRATUM 2026-08-22: the `247.71×` NI1/NR1-K32 figure in this section is WITHDRAWN — fabricated, no receipt; NI1's d_seg is NOT MEASURED and its token-agreement proxy is 1.079× DX2, and at 122,250 B it is byte-feasible for sub-0.12. The RI1 `43.66×` is real and MEASURED. See `.omx/research/ddm_ni1_247x_erratum_20260822.md`.]**
  d_seg ceilings). Measured amplification exponent **16.69**. **Any lossy step here inherits a 44–248×
  prior against it — this arm is LOSSLESS ONLY.**
- `ddm_lq1_lane_quotient_representability_20260822.md` — a full-Hamming ORACLE assignment removes only
  **16.6%** of RC1's mismatches ⇒ 83.4% representational. Perfect encoding does not rescue a
  representation lacking the content.
- `ddm_jx1_joint_exchange_envelope_20260822.md`=9a6a6adcd06cd4faf454c28b5f0175a691a7da07112457535b2a1521ed92f6fd —
  UNION ≠ SUM OF LEGS, measured **3.705×**. If you measure MI against several sections, the joint gain
  is NOT their sum; measure the joint directly or label the sum an UPPER BOUND on its face.

## OPTIMAL FORM

- **REFERENCE FORM (cited, with receipt): the incumbent itself** — the shipped RC64 arithmetic stream
  under its learned 19-member HPAC context law over 190 groups `g=(x mod 64)+2*(y mod 64)`, traversing
  frame-outermost then group then raster, coding 117,964,800 positions in 113,777 B = **0.007716
  bits/position** (TO2's anatomy, AD2 receipt `80124acd…b73511`). The challenger MUST be this class
  with MORE conditioning — same model family, extra context members. **Substituting a general-purpose
  or domain-agnostic compressor is a MECHANISM reduction and is FORBIDDEN here**; a sister charter made
  exactly that error today and it is filed at `#1202` as a self-audit. If you cannot extend the
  incumbent's class, say so and declare the TOY BRACKET explicitly rather than racing a weaker family.
- Family exemplar for CONDUCT: `ddm_cx3_context_axis_ceiling_20260822.md` — reproduced every pin,
  reused the decoded array, kept model cost in its own column, reported that its best challenger's
  IDEAL already lost, and scoped its negative to FORMULATION while naming what survives. Match that.
- VERIFIED ARITHMETIC (check once, then use): DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.1200996 · seg 100·0.00020139 = 0.020139 · pose √(10·6.37e-6) = 0.0079812.
  Distortion 0.028120 → S<0.12 needs ≤ **137,986 B** (STRICT ⇒ FLOOR of 137,986.8388) → shed **42,382 B**;
  6.658e-7 S/B. Zero-distortion ceiling **180,218.3 B**, so the distortion we carry is worth
  **42,235 B** and the zero-distortion gap is **149.7 B**.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — substituting a weaker model
  family for the incumbent's class, reporting a gross stream cut whose extension needs stored side
  information, or summing per-section MI into a joint claim, are the three fakes here.
- **PRIOR-LAW PREDICTION (falsifiable):** the token field carries NON-TRIVIAL mutual information with
  the decoded carrier — both describe the same 600-pair scene and the token model has never seen it —
  and extending the incumbent's context law with cross-section members cuts the token stream by
  **≥5% (≥5,689 B)** net of the extended model's own description cost, decode-identical, at zero stored
  bytes. This is the first measurement in the campaign that can see between sections.
  **FALSIFIER:** measured I(tokens ; legally-available sections | incumbent context) is negligible, or
  the extended model's real coded bytes plus its blob growth land within ~2% of 113,777 B ⇒ the token
  field is conditionally independent of the rest of the archive given its own history, RB1's isolation
  assumption is VINDICATED as an optimality property and not merely a decodability one, and the
  lossless axis on this body is closed across coder, addressing, ordering, self-context AND
  cross-section. Report that plainly — it is a complete, campaign-directing result and it would retire
  the last unexamined assumption in the rate stack.

## DELIVERABLE

`.omx/research/ddm_xs1_cross_section_conditioning_20260822.md` — the verified decode ORDER with the
legally-conditionable set named + per-section mutual information vs the token field's post-incumbent
residual (bits AND bytes, with denominators) + the extended-context-model race with real coded bytes
and model cost in its own column, every form inverted to TO2's exact array + the rule-118 adjudication
per admitted extension (what generic rule, what already-decoded content it reads, zero stored bytes or
the counted ledger) + net delta stated in BOTH currencies (archive bytes AND distortion-equivalent at
6.658e-7 S/B) + decode-cost line against the 498 s / 1800 s budget + the verdict on the prior-law
prediction with verdict_scope at the NARROWEST level the evidence supports. Commit via the serializer.
End with the own-vehicle frontier line.
