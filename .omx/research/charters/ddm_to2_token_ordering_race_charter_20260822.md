# ddm_to2_token_ordering_race — AD2 just proved a pure serialization ORDER change cuts a Brotli stream 34.5% with zero information loss; the largest object in the live archive has never been ordering-raced

## MANDATE

AD2 (`ddm_ad2_addressing_cost_decomposition_20260822.md`, commit `4a49821f8f`) measured a real,
lossless, decode-identical win on NR1's QPAIR stream by changing **nothing but the order symbols are
written in**:

| NR1 QPAIR form | real Brotli-q11 bytes | delta |
|---|---:|---:|
| incumbent pair-major raster u8 | 52,040 | baseline |
| **tile-major time u8** | **34,083** | **−17,957 (−34.5%)** |

Same symbols, same count, same information — the receiver reconstructs identically. Brotli simply
finds longer matches when the same cell's values across time sit adjacent. AD2's own note: this real
stream **beat the listed 44,995 B first-order entropy reference**, which is direct evidence that the
reference is model-scoped rather than a universal bound.

**The gap this arm exists to close.** AD2 raced orderings on RC1-assignment, NR1-QCTX, NR1-QPAIR and
NR1-QEVENT. It did **not** race orderings on the DX2 incumbent's token stream. AD2's own DX2 anatomy:

| DX2 physical region | class | bytes | archive fraction |
|---|---|---:|---:|
| **semantic tokens at implicit raster sites** | payload / what | **113,777** | **63.0805%** |
| semantic renderer | payload / what | 30,856 | 17.1072% |
| carrier stream | mixed payload + basis/coefficient metadata | 22,010 | 12.2028% |
| learned HPAC probability model | addressing / how-to | 13,515 | 7.4930% |
| ZIP framing + RX1 header + compact residual | framing / addressing | 210 | 0.1164% |

The token stream is the single largest object in the archive, and **two orthogonal axes on it are
already closed**: RB1 closed the CODER axis (0 B tested headroom across all seven DX2 streams at
fixed distortion) and AD2 closed the ADDRESSING axis (implicit raster sites cost 0 B — DX2 already
pays nothing to say *where*, which is rule-118 played correctly and fully exploited). The ORDERING
axis is a THIRD, distinct question and it is unmeasured on this stream.

**Why this arm and not another quotient.** Today measured both byte-feasible lossy re-representations
DEAD on distortion: RC1 at 43.66× its ceiling and NR1-K32 at 247.71× (see PRIOR NEGATIVE SIGNAL).
> **[MAIN CORRECTION 2026-08-22, SUPERSEDES THE 08-22 ERRATUM: the `247.71x` figure is CONFIRMED, now MEASURED on contest-CUDA n600 (call fc-01M0PF62QK…, S 27.8, d_seg 0.07583781 = 247.69x NI1's own ceiling and 376.6x DX2, d_pose 40.53). NI1 is byte-feasible at 122,250 B and DISTORTION-DEAD. Its 98.6786% token agreement understated d_seg by 349x — do NOT use token agreement as an evaluator. RI1 43.66x also real+MEASURED. The whole-body lossy re-representation family is CLOSED on two authority rows. See `.omx/research/ddm_ni1_247x_erratum_20260822.md` (retraction section at the end).]**
Their shared mechanism is that they change WHICH tokens the receiver sees. **This arm changes none.**
It is lossless by construction, so every distortion negative measured today is irrelevant to it —
d_seg, d_pose, and the rendered output are byte-identical, and only the rate term can move.

## SCOPE

1. **Verify inherited state, refuse on drift.** DX2 archive sha
   `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` @ 180,368 B · AD2's memo +
   its receipt `/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/measurement_v6/RESULT.json`
   sha `80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511` · RB1's memo
   `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09.
   Reproduce AD2's DX2 anatomy rows from the archive itself; any disagreement IS the finding, report
   it first.
2. **Extract the exact token symbol array.** Recover the semantic token field the 113,777 B stream
   codes — as symbols, with its true shape (pairs × sites × whatever the field's real rank is).
   State the symbol count and alphabet with denominators (m50: a count without its denominator is not
   a measurement). **Every candidate form must be INVERTED back to this exact array and compared
   byte-for-byte before its coded size is admitted** — that is AD2's discipline and it is what makes
   an ordering result real rather than a hash collision of convenience.
3. **Race GENERIC orderings with real coders.** At minimum: incumbent (whatever DX2 actually uses) ·
   tile-major time (the AD2 winner's shape, adapted to this field's real geometry) · 8×8 block ·
   a space-filling curve (Morton and/or Hilbert) · class-sorted canonical order · any ordering the
   field's measured structure suggests. Code each with real Brotli-q11 AND at least one non-Brotli
   coder (LZMA1, zlib-9) — AD2 measured that the ordering win is coder-structure-dependent (it buys
   LZ match length, not symbol-rank cost), so a win on one coder is not a win on the shipped one.
   Retain every losing variant's bytes.
4. **Adjudicate rule-118 explicitly per candidate.** An ordering is FREE only if the receiver computes
   the permutation from the field's SHAPE alone by a generic algorithm it already knows. **If a
   candidate requires storing WHICH permutation, or a per-frame/per-cell ordering table, that table is
   VIDEO-DERIVED and its bytes are COUNTED** — price it net, do not report the gross stream cut. This
   is the hide-data-in-code fake in its exact local form and it is the one way this arm can produce a
   false win. State, per admitted candidate, the generic rule and what it reads.
5. **Report the net archive delta, and say what it means for 42,382 B.** A stream cut is not an
   archive until the container is rebuilt. Give the projected archive bytes and the recomputed S from
   components, labelled PROJECTION until a byte-closed archive exists. If the win is real and large,
   emit a sealed fire-order for MAIN (byte-close → decode identity → seal → T4); do NOT fire.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO Metal fires (MAIN-fire-only). NO scorer runs — this arm is
  SCORER-FREE BY CONSTRUCTION, and that is not a scope reduction here but a property of the question:
  a lossless reordering cannot move d_seg or d_pose, and any candidate that does is a BUG, not a
  result. If you cannot prove byte-identity of the inverted array, the candidate is refused.
- Shipped receiver bytes are CUSTODY — never edit in place. This arm MEASURES and DESIGNS.
- The jo1 r9 run directory is SACRED. r9 is terminal by SELF-REFUSAL (`EXACT_DELTA_NONNEGATIVE`);
  nothing to wait on.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0): every candidate's coded bytes, every losing variant, every inverted
  array check, and the extracted symbol array itself persist with sha256 + bytes. Scalar-only
  artifacts while the arrays exist in memory are forbidden AT THE TYPING MOMENT.
- **Receipts to `/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/` — NOT APDataStore, which
  is at ~11 GiB free.** Say which tier you used.
- File ownership: AD2 owns the addressing decomposition · RI1 owns RC1's distortion · NI1 owns
  NR1-K32's · LQ1 owns the Lane representability question. Do not touch their memos or retained
  trees; CITE their rows.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_rb1_rate_bound_decomposition_20260822.md`=fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09 —
  **0 B tested headroom across all seven DX2 archive streams at fixed distortion.** That verdict is
  scoped to the CODER axis on the fixed representation. Reordering is not a recode: it changes the
  symbol SEQUENCE the coder sees, which is why AD2's QPAIR result beat a first-order entropy reference
  that a coder race could never have beaten. Do NOT treat RB1 as closing this question, and do NOT
  treat this arm as reopening RB1's — if your measurement reduces to "a different coder on the same
  sequence", you have drifted onto RB1's axis and the result is already known to be 0 B.
- `ddm_ad2_addressing_cost_decomposition_20260822.md` — **RC1's tested assignment layouts save 0 B**
  (8×8-block and fixed-width forms all LOSE: −3,453 to −8,946 B) and **NR1 QCTX's alternatives all
  lose** (−16 to −44 B). Reordering is NOT a universal win; it won on exactly one stream, the one with
  strong per-cell temporal structure. Predict which shape wins from the field's MEASURED structure
  before racing, and report the prediction against the outcome.
- `ddm_ri1_rc1_full_rgb_receiver_20260822.md` + `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` —
  both byte-feasible LOSSY re-representations measured DEAD on distortion (RC1 d_seg 0.01605413 vs
  ceiling 0.000367727 = 43.66× · NR1-K32 d_seg 0.07584291 vs ceiling 0.000306175 = 247.71×). Do NOT
  propose a lossy variant of any ordering to buy extra bytes — the amplification measured between
  those two points is an exponent of **16.69** (token disagreement ×1.0975 → d_seg ×4.7242), so
  *which* tokens differ dominates *how many* by more than an order of magnitude. Any lossy step here
  inherits a 43×–248× prior against it.
- `ddm_dc1s_sparse_grid_sweep_20260821.md` — explicit-position sparse grids CLOSED at full n600 FX5
  scope (388,326 B actual vs a 113,777 B member, ALL 190 groups negative). Do not re-open explicit
  position fields; this arm stores no positions at all.
- `ddm_vf1_evaluator_visible_floor_20260822.md`=f65e641edfc987a127dd2813d4136bbb01ad1c46ef4b211c80176416afcb87b4 —
  **0 of 117,964,800 token positions carry qualifying DX2 evidence.** No retained token-level
  sensitivity corpus exists. Consequence here: do not assume any token region is inert or reorderable
  "because it does not matter" — losslessness must come from the INVERSION CHECK, never from an
  argument about which tokens are important.

## OPTIMAL FORM

- Family exemplar (reference): `ddm_ad2_addressing_cost_decomposition_20260822.md`, receipt sha
  `80124acd71ff63d4d9379b87674d1a976e1aa73857b4062a1c9ea2afb1b73511` — it raced real coders on real
  streams, inverted every form to the exact source symbol array before admitting a byte result,
  retained every losing variant and deterministic repeat, distinguished a model-scoped entropy
  reference from a universal bound when its own winner beat the reference, and refused to promote a
  stream delta into an archive claim. Match that bar exactly; this arm applies its proven mechanism to
  the one stream it did not reach.
- VERIFIED ARITHMETIC (check once, then use): pointer DX2 S 0.14821987563243377 @ 180,368 B.
  rate 25·180368/37545489 = 0.1200996 · seg 100·0.00020139 = 0.020139 · pose √(10·6.37e-6) =
  0.0079812 → 0.1482198. Distortion 0.028120 → S<0.12 needs archive ≤ **137,986 B** (STRICT
  inequality ⇒ FLOOR of 137,986.8388) → shed **42,382 B**; 6.658e-7 S/B. The token stream is
  113,777 B, so the demand is **37.3% of this one stream** — and AD2's proven mechanism delivered
  34.5% on a comparable one. THE PHYSICS BOUND (jx1 §5.2): zeroing BOTH distortion axes still leaves
  rate above 0.12, so a LOSSLESS rate cut is not merely helpful here, it is the only kind of move that
  can work without also solving distortion.
- SCOPE reductions declared per row. MECHANISM reductions FORBIDDEN — admitting a coded size without
  inverting the form back to the exact symbol array, or reporting a gross stream cut whose ordering
  requires a stored (video-derived, therefore COUNTED) permutation table, are the two fakes this
  charter refuses.
- **PRIOR-LAW PREDICTION (falsifiable):** DX2's semantic token field carries strong per-cell temporal
  structure across the 600 pairs — the same property that made QPAIR's tile-major-time form win — and
  its incumbent serialization is pair-major raster, which separates each cell's temporal neighbours by
  a full frame of symbols. A generic tile-major-time (or space-filling-curve) reordering therefore
  beats the incumbent by **≥10% of the stream (≥11,378 B)** at Brotli-q11, decode-identical, with a
  receiver-derived permutation and zero counted side information.
  **FALSIFIER:** every generic ordering lands within ~2% of the incumbent ⇒ the token stream is
  already at its ordering optimum, the LOSSLESS axis is CLOSED across the entire live body (coder by
  RB1, addressing by AD2, ordering here), and the campaign's 42,382 B cannot come from re-serializing
  what we already ship. Report that outcome plainly and put the decisive number in the FIRST line —
  it retires a whole axis honestly and redirects the campaign, which is a complete result.

## DELIVERABLE

`.omx/research/ddm_to2_token_ordering_race_20260822.md` — the extracted token field (shape, symbol
count, alphabet, sha256, bytes) + the incumbent ordering identified from the archive + the per-form
race table (real Brotli-q11 AND ≥1 non-Brotli coder, every form inverted-and-verified, every loser
retained) + the rule-118 adjudication per admitted candidate naming its generic rule and what it
reads + the net projected archive bytes and recomputed-from-components S labelled PROJECTION + a
sealed MAIN fire-order if the win is real + the explicit verdict on the prior-law prediction with
verdict_scope at the NARROWEST level the evidence supports. Commit via the serializer. End with the
own-vehicle frontier line.
