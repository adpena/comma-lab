# ddm_sm4 — Recover the refuted -6,272 B: is the low-rank failure factor-quantization or rank?

**Owner:** codex arm · **Base:** PR130 CPR1 · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: a low-rank semantic codec measured at its OWN optimum for the byte budget, not at
one arbitrary (rank, precision) point. Reference pin: `sm3r_receiver.py::_decode_lowrank` in
`/Volumes/VertigoDataTier/pact/ddm_cp2_20260810/pointwise_lowrank_r32__temporal_reversion/submission/`.
Declared reductions: SCOPE only — screen candidates by the $0 frame-parity method before any
scorer pass. MECHANISM reductions are TOY-BRACKET: testing one rank; using a synthetic matrix
instead of the real weights; declaring a family verdict from a single (rank, precision) cell.

## WHY THIS ARM EXISTS

cp2's composed candidate was REFUTED by measurement: S 0.2098374 -> 7.4924. Attribution is exact
(frame-parity: even frames byte-identical, odd frames 99.26% changed) — ALL damage is the
semantic low-rank leg. **That withdrew -6,272 B, which is 2.6x our entire remaining bankable
saving.** Recovering it is the single largest rate number on the board.

READ AT SOURCE (confirmed, but a HYPOTHESIS for the mechanism, not a measurement of it):
```python
left,  remaining = _decode_standard_q4("factor.left",  left_template,  remaining)
right, remaining = _decode_standard_q4("factor.right", right_template, remaining)
restored[name] = (left @ right).reshape(value.shape)
```
Both FACTORS are int4-quantized then multiplied — error compounds across 32 rank terms, where the
standard path quantizes W directly with bounded per-element error. And there is NO mean/centering
term. Pixel signature is consistent: corr ~0.31, DC 130->78, sd 68.5->50 on every frame sampled.

## WHAT TO MEASURE

1. **The discriminator, at EQUAL BYTES.** Receiver byte law: std q4 = `rows*cols/2 + rows*2`;
   lowrank = `(rows*r)/2 + (r*cols)/2 + (rows+r)*2`. Break-even r* = rows*cols/(rows+cols).
   At 128x128: r32-int4 ~4,416 B vs **r16-int8 ~4,384 B — the same budget, 16x finer factors.**
   - FACTOR-QUANTIZATION COMPOUNDING => r16-int8 beats r32-int4 decisively.
   - RANK INSUFFICIENCY => r16-int8 loses.
   Run the (rank x factor-precision) grid at matched bytes on the REAL weights.
2. **Centering arm.** Store a per-row mean (tiny) and factor the CENTERED matrix. Rank then
   spends itself on structure, not DC. Measure separately AND composed with arm 1.
3. **Weight-level error first, pixels second.** Compare reconstructed W vs base W per tensor
   (`coord_mix.weight`, `blocks.{0..3}.pw.weight`) before spending any decode. Cheapest
   possible discrimination.
4. **$0 SCREEN BEFORE SCORER.** Any candidate you decode: compare its raw against the base raw
   by FRAME PARITY (even = pose carrier, odd = semantic). MAIN proved this isolates damage for
   zero cost. Only escalate a candidate to a scorer request if it survives the screen.

## DELIVERABLE

The (rank x precision x centering) table at matched bytes with per-tensor weight error · the
mechanism verdict with its falsifier stated · IF a candidate survives, ONE byte-closed archive +
a frame-parity screen + a scorer queue row for MAIN. If low-rank is genuinely dead at every cell
of the matched-byte grid, say so — that closes 19.15% of the archive honestly and routes the
semantic section elsewhere.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Persist EVERY candidate payload with sha256 + byte
count, not just the winner. Run `tac.payload_retention_gate` on anything you write. Known
limitation: it tests PERSISTENCE, not reachability-to-a-writer.

## HARD RULES

- Bulk artifacts → your own `/Volumes/VertigoDataTier/pact/<arm>/` dir. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per
  file, tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 x `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- IF THE SERIALIZER CANNOT RUN AT ALL: emit an UNCOMMITTED-WORK MANIFEST (every changed path +
  post-edit sha256) in your final message. Measured work you cannot commit is signal loss.
- Write ONLY under your own arm paths. Sister arms landed artifacts are APPEND-ONLY inputs.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY.
- Every number carries its axis. macOS = `[macOS-CPU advisory]`, never `[contest-CPU]`.
  No Modal without operator GO. Do NOT claim the scorer slot without saying so explicitly.
- Decode wall-clock is NEVER an admissibility criterion (standing operator directive). Time is
  report-only.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## PROVENANCE PINS (verify each at source; a pin that does not reproduce is a STOP)

- base archive 191,052 B sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
- base S = 0.172141297491896447 `[contest-CUDA, DALI GT, n600]` = seg 0.028609 + pose ~0.0152 + rate 0.127214
- base raw sha256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353`
- exchange rate: 150.18 bytes = 1e-6 of d_seg  ·  1,000 B = 0.000666 S
- CORRECTED LEDGER: bankable = -2,424 B (ai1 -2,416 + hp3 -8, both LOSSLESS).
  sm3 low-rank -6,272 B is **REFUTED** (S 0.2098 -> 7.4924). See
  `.omx/research/ddm_main_paired_eval_20260810/PAIRED_EVAL_FINDINGS.md` (e8b32e961b).

