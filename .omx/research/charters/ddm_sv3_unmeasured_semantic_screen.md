# ddm_sv3 — Two byte wins are still UNMEASURED for distortion, and they are siblings of one that just failed catastrophically

**Owner:** codex arm · **Base:** PR130 CPR1 · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: every banked byte win screened for distortion damage BEFORE it consumes a scorer
pass, using the cheapest instrument that can see the damage. Reference pin: the frame-parity screen
in `.omx/research/ddm_main_paired_eval_20260810/PAIRED_EVAL_FINDINGS.md` (b16244dea8, e8b32e961b),
which attributed a catastrophic failure to its exact cause at zero compute. Declared reductions:
SCOPE only. MECHANISM reductions are TOY-BRACKET: declaring a candidate "safe" from byte counts;
screening on a frame subset without saying which; treating a raw-diff of zero on ONE frame class
as proof for the other.

## WHY THIS ARM EXISTS

The paired eval on 2026-08-10 measured cp2's composed candidate at **S 7.4924 vs base 0.2098374**
`[macOS-CPU advisory, AV GT, n600]` — the -8,688 B bought 0.005785 S of rate and cost ~7.28 S of
distortion. Frame-parity attribution put **100% of the damage** on the odd/semantic frames
(mean|delta| 67.71, 99.26% of pixels changed) with the even/pose-carrier frames byte-identical.
The failing leg was sm3's pointwise low-rank r32.

**Two siblings from the same arm and the same family remain UNMEASURED for distortion:**

| candidate | section | delta bytes | distortion state |
|---|---|---:|---|
| `sm3` joint vector/scale VQ32 | semantic | -4,648 | **NEVER MEASURED** |
| prior SD1 mixed q3/q4 | semantic | -848 | semantic-leg improvement measured; **pose UNMEASURED** |

Both are lossy re-representations of the same pointwise-conv capacity that the low-rank leg
destroyed. Neither is presumed guilty — but neither may be banked on byte count alone, and that is
exactly the error the corrected ledger just removed.

## WHAT TO MEASURE

1. **Screen both, cheapest instrument first.** Order of escalation, stop at the first decisive step:
   (a) reconstructed-WEIGHT error vs base per tensor — no decode at all;
   (b) decoded RAW vs the base raw `a18eb42a...` by FRAME PARITY (even = pose carrier,
       odd = semantic) — no scorer;
   (c) only if (a)+(b) are clean, queue a scorer row for MAIN with the evidence attached.
2. **Report the screen's own resolution.** State what damage magnitude each step CAN and CANNOT
   see. A clean screen is only as strong as its sensitivity — say what it would have missed.
3. **Re-screen the refuted low-rank as a POSITIVE CONTROL.** Your instrument must flag it. An
   instrument that passes a known-catastrophic candidate is broken, and finding that out here is
   cheap. Report the control result explicitly; if it does not fire, STOP and fix the instrument.
4. **Generalize the screen into a reusable tool** under your own path, so the next byte win is
   screened by default instead of by remembering.

## DELIVERABLE

Per-candidate screen results with the positive control · the corrected bankable ledger (what may be
banked, what is withdrawn, what is queued) · the reusable screening tool · a scorer queue ONLY for
candidates that survive. An honest "both siblings are also damaged" is a full-value result — it
would close the lossy-semantic-re-representation family and route the 19.15% semantic section
somewhere else.

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

