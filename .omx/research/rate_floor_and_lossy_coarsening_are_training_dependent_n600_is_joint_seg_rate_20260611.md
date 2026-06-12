# CORRECTION: the rate entropy-floor + lossy-coarsening are TRAINING-DEPENDENT, not recipe-independent — the n600 is a JOINT seg+rate lever (2026-06-11)

**Authority:** design/analysis, advisory. Frontier UNMOVED 0.19109982 [contest-CPU], 177,169 B. Trigger:
operator "the rate entropy floor and lossy-coarsening may be more nuanced than we have stated thus far."
This corrects the re-audit's TIER-3 "STILL-HOLDS / do-not-resurrect" classification of these two
(`recipe_bug_lens_findings_reaudit_ledger_20260611.md` #10/#12) AND overrides the resurrection sweep's
do-not-resurrect instruction for them.

## The nuance (why these two are NOT recipe-independent)

Both negatives were measured on the FROZEN / suboptimally-trained frontier decoder. But two PR95 curriculum
ingredients are explicitly RATE/robustness levers that shape the weight distribution the floor/wall are
properties OF:
- **C1a coder-aware regularization (PR95 L16, lambda 0.01→0.02):** "a structural prior that biases decoder
  weights toward brotli-friendly distributions." A RATE lever. If applied, the trained decoder's weight
  ENTROPY is lower → the "7.999 bits/byte floor" moves DOWN.
- **Sigma noise injection (L17) + QAT (stage 4):** "simulates uint8 quantization roundtrip during training"
  → makes the decoder QUANTIZATION-ROBUST. If applied, the decoder tolerates COARSER quantization → the
  lossy-coarsening "9.8:1 R-D wall" moves.

**The recipe-bug connection:** our buggy curriculum (`c1prime`) ran exactly those stages throttled (the
dropped muon_lr + LR floor crawled through C1a/sigma/QAT). The re-audit even classified C1a/sigma as
"RATE/robustness levers, not d_seg levers" — and our decoder was NEVER properly coder-aware / robustness
trained. So:
- the **rate entropy-floor (7.999 b/byte)** was measured on a decoder NOT coder-aware-trained → it is the
  floor for a coding-SUBOPTIMAL decoder, not a hard information limit;
- the **lossy-coarsening 9.8:1 R-D wall + the 0.3517 CUDA-negative** were measured on a decoder NOT
  quantization-robust-trained → for the wrong decoder.

## The consequence: the n600 is a JOINT seg + rate lever (raised value proposition)

The same fixed PR95 curriculum that fixes seg-convergence (the stage schedule + un-throttled muon) ALSO
carries the rate levers (C1a coder-aware reg + sigma + QAT). So a correctly-trained n600 decoder is expected
to be SIMULTANEOUSLY: (a) seg-basin-reaching (the recipe fix), (b) LOWER-entropy / more compressible (C1a →
lower rate floor), (c) coarsening-robust (sigma + QAT → coarser quant at same distortion → fewer bytes).
**The n600 attacks the seg term AND the 62%-of-score rate term in one run** — not just seg-convergence. This
is a materially stronger pointer-mover thesis than "reach the seg basin."

## The distinction that STILL holds (honest, NO-FAKE)

The FROZEN-frontier byte-transform on the CURRENT bytes IS genuinely exhausted — you cannot re-code today's
177,169 B lower (agent B measured every coder loses; that stands). What moves is the floor/wall for a
DIFFERENTLY-TRAINED decoder. So the rate win is a CONSEQUENCE of the n600's coder-aware training, NOT a new
post-hoc lever on the existing archive. Both statements are true and not in tension:
- frozen-frontier post-hoc rate re-coding: EXHAUSTED (stays closed);
- trained-representation rate floor (entropy + coarsening): TRAINING-DEPENDENT, moves with the n600 (re-open).

## Routing (folds into the in-flight pre-dispatch rigor)

- **Config agent (a97c23f):** must treat C1a (L16) + sigma (L17) + QAT as FIRST-CLASS RATE LEVERS in the
  n600 config (not just robustness afterthoughts), with their lambda/sigma/bit schedules justified against
  PR95 source — AND must MEASURE the trained decoder's weight-entropy + coarsening R-D curve as a byte
  budget input (the rate win is a config deliverable, not assumed).
- **Symposium:** the n600 PROCEED thesis is now joint seg+rate; the rate-win-via-coder-aware-training claim
  must be in the 9-dim checklist + the pointer-mover thesis (and post-training-validated per Catalog #324,
  measured on the byte-closed archive's actual entropy, not assumed).
- **Resurrection sweep (ac792687):** its "do-not-resurrect rate-floor/lossy-coarsening" is OVERRIDDEN — they
  reclassify to MEASUREMENT/WRONG-OP-POINT (measured on a non-coder-aware-trained decoder), re-open AS the
  n600's joint rate lever (NOT as a frozen-byte re-code).

## Bottom line

I over-stated the rate floor + lossy-coarsening as recipe-independent genuine negatives; the operator is
right that they are more nuanced. They are properties of the trained representation, our training was buggy,
and the fixed PR95 curriculum's own C1a/sigma/QAT stages are the levers that move them — making the n600 a
joint seg+rate pointer-mover. The frozen-byte re-code stays closed; the trained-floor re-opens.
