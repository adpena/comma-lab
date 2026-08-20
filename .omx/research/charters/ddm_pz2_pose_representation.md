# ddm_pz2 — The pose section is 23,384 B and 0.0155704 S, and its CODER is closed. What about its REPRESENTATION?

**Owner:** codex arm · **Base:** PR130 CPR1 · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: the pose section priced as a REPRESENTATION problem at its own optimum — what the
6 PoseNet scalars per pair actually require, versus what PR130 currently spends. Reference pins:
`ddm_pk2` (pose-carrier representation attack; baseline WON, basis symbols already signed int5
-15..15), `ddm_rc2_20260810/RC2_FINAL_REPORT.md` (52cb73adc0, coder axis closed: PPMd +441 B on
this section, all six LDPC variants lose). Declared reductions: SCOPE only (n120 seeded
stratified-random, NEVER a prefix). MECHANISM reductions are TOY-BRACKET: pricing a pose
representation without measuring realized d_pose through the real decode; transferring a d_pose
number across GT decoders (see the hard fact below).

## WHY THIS ARM EXISTS

pose = 23,384 B = 12.24% of the archive = 0.0155704 S of marginal rate, PLUS the ~0.0152 S pose
distortion term. Two independent closures point the same direction:

- **Coder substitution is exhausted** on this section (rc2: PPMd +441 B, LDPC catastrophic).
- **pk2 attacked the carrier representation and the BASELINE WON** — the symbols are already
  signed int5. So the cheap re-encodings are gone.

What has NOT been done is asking what the section must contain AT ALL. The scorer computes
`d_pose = MSE(PoseNet(generated)[:6], PoseNet(original)[:6])` — SIX scalars per pair, 600 pairs.
That is 3,600 numbers. We are spending 23,384 B on them: **~39 B per pair, ~6.5 B per scalar.**
Whether that is near a floor or far above it has never been measured on this base.

**HARD FACT, measured 2026-08-10, do not violate it:** the AV-vs-DALI GT gap on this base is
**d_pose 6.83x** (vs d_seg 1.44x) — the ground-truth decoder gap is overwhelmingly a POSE
phenomenon. Never transfer a pose result across GT axes. State the GT decoder on every pose number
you report, and compare only within one axis.

## WHAT TO MEASURE

1. **The floor question.** What is the entropy of the 3,600 target scalars themselves, at the
   precision d_pose actually needs? Derive the precision from the score: how much does d_pose move
   per unit of quantization error in each of the 6 dimensions? Some dimensions may be nearly free.
2. **Per-dimension rate allocation.** The 6 dimensions are almost certainly not equally
   score-sensitive. Measure the sensitivity, then waterfill bits across dimensions instead of
   spending uniformly. This is a pure representation change with no new coder.
3. **Temporal structure.** 600 pairs from ONE continuous drive. Ego-motion is smooth. Is the
   current representation exploiting that, or coding pairs independently? Measure the residual
   after a smooth predictor before proposing one.
4. **Price everything jointly.** Any pose change that shrinks bytes but raises d_pose must be
   priced through `sqrt(10*d_pose)` — the square root means the marginal cost of d_pose RISES
   steeply as d_pose falls. At the base's small d_pose, a little damage is expensive. Compute the
   exchange rate at the operating point; do not assume it.
5. **$0 screen before any scorer request** — decode and compare against the base raw by frame
   parity (EVEN frames = the pose carrier on this vehicle). MAIN's method, proven 2026-08-10.

## DELIVERABLE

The measured per-dimension score-sensitivity of the 6 PoseNet outputs · the entropy floor of the
target scalars at score-adequate precision vs the 23,384 B actually spent · any representation
change priced jointly in S · a frame-parity screen + scorer queue row for anything that survives.
If 23,384 B turns out to be near the floor for this representation, that closes 12.24% of the
archive honestly and concentrates everything on tokens.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000)

CLAUDE.md `## ⛔ ALWAYS KEEP THE PAYLOAD`. Persist EVERY candidate payload with sha256 + byte
count, not just the winner. Run `tac.payload_retention_gate` on anything you write.

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
- Decode wall-clock is NEVER an admissibility criterion (standing operator directive).

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. PR130 repro code is
off-the-shelf authorized. Cite path + commit for anything reused.

## PROVENANCE PINS (verify each at source; a pin that does not reproduce is a STOP)

- base archive 191,052 B sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
- base S = 0.172141297491896447 `[contest-CUDA, DALI GT, n600]`
- base raw sha256 `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` (3,662,409,600 B)
- exchange rate: 150.18 bytes = 1e-6 of d_seg  ·  1,000 B = 0.000666 S
- CORRECTED LEDGER: bankable = -2,424 B. sm3 low-rank -6,272 B is **REFUTED**
  (paired eval S 0.2098374 -> 7.4924). `.omx/research/ddm_main_paired_eval_20260810/PAIRED_EVAL_FINDINGS.md`

