# ddm_lc2 — the LOSSLESS-ONLY coder stack on the live candidate

## THE ONE THING
Compose CX2's **lossless** coder steps onto the AI1 ANS archive on the PR130 base, byte-close,
and prove the decoded raw output is **byte-identical to PR130's**. If it is, the score follows
from bytes alone — seg and pose are PR130's own, unchanged by construction.

Predicted (ARITHMETIC, non-additive — this is what you MEASURE, not what you claim):
`191,052 − 2,120 − 296 − 903 = 187,733 B`, i.e. **−903 B below the live candidate**.
Falsifier: composed archive ≥ 188,636 B ⇒ the steps do not stack; report the measured
interaction sign and STOP (do not invent a fourth step to rescue the number).

> ⚠ **CORRECTION (MAIN, at spawn+3min) — the first version of this line was a CROSS-REGIME
> CONSTANT TRANSFER and is withdrawn.** It read `191,052 − 963 − 423 − 2,416 = 187,250 B`,
> taking CX2's split-Brotli `−963` and xcodec `−423` step deltas from a ladder where **SD1M was
> already in the stack**. Those deltas are SD1M-conditioned; they do not transfer to the clean
> PR130 base. The corrected figures come from **VP1** (`.omx/research/ddm_vp1_20260810/VP1_RESCORING_REPORT.md`),
> which measured the lossless family NATIVELY on this base and decomposes it as
> ANS `−2,120 B` · split-model streams `−903 B` · temporal-reversion `−296 B` incremental,
> all tied at `0.000665858953 S` per 1,000 bytes.
> **Corroboration:** VP1's ANS + temporal = 2,120 + 296 = **2,416 B** — EXACTLY AI1's measured
> total. Two independent arms agreeing to the byte is why the corrected decomposition is trusted
> and the SD1M-conditioned one is not. The NEW mechanism this arm adds to AI1's stack is the
> **split-model streams (−903 B)**, not CX2's two coder steps.
> Measure the composition anyway — additivity is the hypothesis, not the finding.

## WHY THIS IS THE CRITICAL PATH
The bar is PR130 CPR1 **S = 0.172141297491896447** @ **191,052 B**
sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` [contest-CUDA, DALI GT, n600].
The live best candidate is AI1 ANS+temporal_reversion @ **188,636 B**
sha `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84`,
measured S = **0.170536856816211** on Modal T4 CUDA n600 (`score_claim=False`, env-mismatch axis).
Seg is ~100% solvable on this base; the remaining gap is CARRIAGE — **rate is the axis**.
Remaining to sub-0.15 from the candidate: **0.020537 S = 30,842 B**. This arm attacks 4.5% of it
at essentially zero distortion risk, which is why it is ranked ahead of any new-mechanism work.

## THE ARITHMETIC THAT LICENSES "SCORE FROM BYTES ALONE"
AI1's row is a PURE RATE MOVE, and its own measurement proves it:
`0.1721413 − 25·2,416/37,545,489 = 0.17053255`; AI1 MEASURED `0.170536856816211`.
Agreement to **4.3e-6** — inside the bar's own ~4e-6 display rounding (the bar is a bot comment
displayed to 8 decimals). A content-changing step could not land that close.
So: for any archive whose decode is raw-byte-identical to PR130's,
`S(B) = 0.172141297491896447 − 25·(191,052 − B)/37,545,489`.
**You still owe ONE exact confirming row** — the identity above is the PREDICTION, never the receipt.

## STEP PROVENANCE (CX2 ladder, `.omx/research/ddm_cx2_20260809/CX2_FINDINGS.md:25-28`)
| step | Δbytes | admit? | reason |
|---|---|---|---|
| SD1 mixed semantic + joint XZ + Range | −848 | **EXCLUDE** | SD1M is "the only content-changing component" (CX2 §129). Lossy → changes seg/pose. |
| split Brotli + Range | −963 | ADMIT | coder-only |
| CX2 reversible xcodec split Brotli | −423 | ADMIT | "reversible" by its own name; VERIFY, don't trust |
| retained ANS | −2,120 (CX2) / −2,416 (AI1) | ADMIT | AI1 row confirms losslessness |

CX2 measured a **favorable −60 B interaction** (189,241 actual vs 189,301 additive), so the
composed total is NOT the sum. Measure it.

Artifacts: CX2 archive `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/composed/archive.zip`
(186,698 B, sha `2acd09e7a585c12403936d1e8a6dc70a9b35d826fe61ead7dea49ad470c4a996`) — reference for
the coder implementations ONLY; it contains SD1M and must not be shipped as-is.
AI1 handoff: `.omx/research/ddm_ai1_ans_receiver_integration_20260809.md`
(landed `01d801c3cf`; pure-ANS predecessors `46c7b85219`, `caa8eef4d8`).

## OPTIMAL FORM
- REFERENCE form: the coder steps at CX2's own measured parameterizations, on the FULL n600 object.
  Provenance pins for the reference form:
  - CX2 implementation commit `cf53216e3e856c15f849bcfe96a5dd4717da2d04`
    (terminal custody correction `442e0d593c7635da77963c4d2d50719d0838768a`);
    reference archive sha256 `2acd09e7a585c12403936d1e8a6dc70a9b35d826fe61ead7dea49ad470c4a996` @ 186,698 B.
  - AI1 ANS commit `01d801c3cf` (pure-ANS predecessors `46c7b85219`, `caa8eef4d8`);
    candidate archive sha256 `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84` @ 188,636 B.
  - PR130 CPR1 base archive sha256
    `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` @ 191,052 B.
- SCOPE reductions (legal): none — the archive is one object; compose and measure it whole.
- MECHANISM reductions: **none permitted**. Do not substitute a cheaper coder, do not sample
  sections, do not estimate a section's contribution from a prefix. If a step cannot be lifted
  cleanly off SD1M, say so and report which one — that is a finding, not a failure.
- CX2 notes six parameter rows TIE at its minimum and carrier Brotli q9/q10/q11 are byte-identical.
  Ties are information: report them; do not silently pick one and call it the optimum.

## VERIFICATION (fail-closed; these are the deliverable, not the byte count)
1. **Raw-decode identity**: decode the composed archive through the real receiver and prove the
   raw output is byte-identical (sha256) to PR130 CPR1's raw decode. This is the whole claim.
   If it differs by one byte, the "score from bytes alone" licence is VOID — report and stop.
2. Exact parse-back: every section consumed, restored exactly (CX2's own discipline, §67).
3. Decode wall-clock recorded against the 30-min budget (CX2 measured 1,010.81 s with 789.19 s
   headroom on its object — you are adding coder work; re-measure, do not inherit).
4. ONE exact n600 row on the composed archive to confirm the predicted S. Do NOT self-dispatch a
   paid run: emit the READY receipt and hand the fire to MAIN.

## ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000)
Persist EVERY composed archive — per-step, not only the winner — under
`/Volumes/VertigoDataTier/pact/ddm_lc2_<date>/retained/`, each with sha256 + byte count in the
result JSON. A scalar-only artifact when bytes existed in memory is a FORBIDDEN run, not a
reviewable choice. Retain the tie-set archives too (CX2's six-row tie says they are not redundant).

## HONESTY BOUNDARIES
- `score_claim=false` on every row you emit. The bar is contest-CUDA/DALI; anything you run
  locally is a different axis and is NOT a promotion comparison (CX2 §18 is the model to copy).
- Do not compare a local CPU/AV number to 0.172141 as if it were a gap.
- The predicted S is DERIVED. Label it DERIVED until an exact row lands.
- If the stack does not compose, the honest negative IS the deliverable: name the interaction,
  its sign, and which step blocked. Do not reach for SD1M to make the headline.

## STORES TO CONSULT FIRST (recall-before-decide)
`.omx/research/ddm_cx2_20260809/CX2_FINDINGS.md` (the ladder + its dead-ends: outer ZIP DEFLATE
+60 B, 2-byte SD1M header +86 B, six-row tie) · `.omx/research/ddm_ai1_ans_receiver_integration_20260809.md`
· task #996 (coder axis measured vs memoryless bound on this base) · #940 (SMEVR/LOTTO are
PER-SURFACE races, not reputations) · #859 (a coder win on one field can INVERT on another —
IX2TOK01 moved the win from symbol rank to LZ match structure). Check whether #996 already
measured one of these steps against its bound before you re-measure it.

## NEXT_IF_RESUMED
Emit the standard block. If the composition lands below 188,636 B with verified raw identity,
the named successor is the exact confirming row (MAIN fires), then the SD1M question as a
SEPARATE distortion-priced arm — never folded into a rate headline.
