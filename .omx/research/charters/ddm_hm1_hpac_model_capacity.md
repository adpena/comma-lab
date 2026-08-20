# ddm_hm1 — The token axis is 61.23% of the archive and the CODER is closed. Attack the MODEL.

**Owner:** codex arm · **Base:** PR130 CPR1 · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: the HPAC model measured as a CAPACITY-ALLOCATION problem — bits spent on model
vs bits saved on tokens, traced as a curve, not a point. Reference pins: `ddm_rc2_20260810/RC2_FINAL_REPORT.md`
(52cb73adc0), `ddm_hp3_20260810/FINAL_REPORT.md`. Declared reductions: SCOPE only (n120 seeded
stratified-random for iteration, NEVER a prefix per m88/m96; winner re-run at n600). MECHANISM
reductions are TOY-BRACKET: pricing the model change WITHOUT re-coding the tokens it conditions;
citing a token saving that a model-size increase already paid for.

## WHY THIS ARM EXISTS

Two measured facts now converge on one axis.

**(1) The coder axis is CLOSED — measured, not asserted.** `ddm_rc2` (52cb73adc0) tested both
requested reference families and both LOSE on every unchanged PR130 section: PPMd adaptive
arithmetic +4,618 (tokens) / +2,263 (semantic) / +441 (pose) / +600 (HPAC); all six exact
LDPC/BP variants lose, best +540,909 B vs the 114,860 B ANS incumbent. rc2's own live hypothesis
names the survivor: *"a changed HPAC model could reduce token bytes because this race closed
coder SUBSTITUTIONS, not model-capacity allocation."*

**(2) The coupling is already measured.** `hp3` found its model-side -548 B partly re-emerged
as +516 B token-side. So model and tokens are ONE joint budget, and any model change must be
priced through the tokens it conditions — never on the model section alone.

tokens = 116,980 B = **61.23%** of the archive. It is the largest section by far and its coder
is exhausted. This is where the remaining rate lives.

## WHAT TO MEASURE

1. **The joint curve, not a point.** Sweep HPAC model capacity (depth/width/quantization/frame-
   embedding dimension) and for EACH cell re-code the tokens through the real receiver. Report
   `model_bytes + token_bytes` as ONE number. A model change that shrinks its own section while
   growing tokens more is a LOSS — hp3 already caught that shape once.
2. **Where is the model underfit vs overfit?** If tokens are large because the model predicts
   them badly, more capacity pays. If the model is already saturated, capacity is dead and the
   token entropy is the floor. Measure which regime we are in — that is the fork.
3. **The free-side check.** rule-118: GENERIC algorithm in inflate.py is FREE; VIDEO-DERIVED /
   LEARNED content is COUNTED. Anything the receiver can DERIVE rather than store is a pure win.
   Audit what the current HPAC stores that a decoder could regenerate deterministically.
4. **PR130 code is off-the-shelf authorized** for this lineage. Reuse their trainer/HPAC directly
   rather than reimplementing; cite path + commit.

## DELIVERABLE

The joint (model bytes + token bytes) curve over the capacity sweep · the underfit-vs-saturated
verdict with its falsifier · any derive-instead-of-store finding priced · ONE byte-closed
candidate + $0 frame-parity screen (even = pose carrier, odd = semantic; MAIN's method) + a
scorer queue row IF a cell wins. An honest "the model is saturated, tokens are at their floor"
is a valuable close — it would retire 61% of the archive as a rate target.

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

