# CORRECTION: clean60 LOST on both axes — and the real lever is PR130's coder, not its model

`score_claim=false` · scorer-free byte measurements · supersedes the headline in
`CLEAN60_PACKED_ROW.md` (commit e39f3a31d8-lineage) **in full**.

## 1. The correction — my "−1,871 B win" was two stacked quantity mismatches

`train_hpac_self_compress.py:143-152` computes
`nats += F.cross_entropy(...); bpp = nats/log(2)/pixels` — the trainer's `best_bpp` is the model's
**CROSS-ENTROPY (ideal)**, not a coded stream. I compared our IDEAL against PR130's REALIZED, and
separately compared a standalone-lzma model pack against a leave-one-out packed marginal.

**Apples-to-apples (same quantity, same tool, both packs `verified_exact` w/ `max_logit_diff 0.0`):**

| quantity | PR130 | clean60 | delta |
|---|---:|---:|---:|
| token IDEAL (cross-entropy) | **114,852** | 115,013 | **+161 ours WORSE** |
| model packed (same packer) | **15,164** | 15,188 | **+24 ours WORSE** |
| **joint** | **130,016** | 130,201 | **+185 ours WORSE** |

**clean60 LOST.** 60 epochs from the pinned init on our own labels did not beat their model. The
earlier −1,871 B is WITHDRAWN. (Field caveat: their ideal is on the DALI label field, ours on the AV
field; the two differ by 20,671 px = 1.7523023e-04. Both are IDEALs, so the quantity mismatch is
cured, but a residual field confound remains — the clean control is coding OUR field with THEIR
model, UNRUN.)

## 2. What survives, and it is bigger than what I lost

The prior Metal encode job (`ddm_pr130_encode_tokens_metal_20260809`, tokens.bin **116,980 B** =
PR130's shipped count exactly) reports BOTH quantities:

```
token_bpp 0.007933213975694445   (realized)
ideal_bpp 0.007788886871760884   (their model's own entropy)
```

**PR130's arithmetic coder runs 2,128 B = 1.8530% above its own model's entropy.** Closing it is
worth **ΔS −0.0014171** at zero model change, zero distortion change, on the largest section.

### Two hypotheses tested, both REFUTED

1. **Per-frame flush** — REFUTED at source: `codec_hpac_integer.py:63` creates ONE `RangeEncoder()`
   *outside* the frame loop (`:69`), one `get_compressed()` (`:93`). Single stream.
2. **`Categorical(perfect=False)`** (`:64`, `:102`) — REFUTED by measurement: `perfect=True` saves
   **0 B** on a 1M-symbol synthetic at the matched skew.

### The mechanism, MEASURED

Synthetic control, 1,000,000 symbols, 5 classes, p=[0.99946,3e-4,1.6e-4,6e-5,2e-5] (matched to the
0.0078 bits/symbol regime), seed 20260716, ideal 909.5 B:

| coder | bytes | overhead vs entropy |
|---|---:|---:|
| `constriction.stream.queue.RangeEncoder` (PR130's) | 924 | **+1.5990%** |
| `constriction.stream.stack.AnsCoder` | **912** | **+0.2795%** |

The synthetic REPRODUCES PR130's overhead (+1.599% vs their +1.853%) on an exactly-known
distribution, so the overhead is a property of the coder at this skew, not of their data.

**ANS saves 12 B / 1M symbols → ≈ −1,416 B projected at n600 → ΔS ≈ −0.000943.**

### Why ANS is admissible despite being LIFO

`encode_reverse` encodes backwards so the stack POPS FORWARD; decode emerges in causal order with
AR context intact. Encode-side holds all symbols, so all conditional tables are computable forward
first. Named cost: encoding backwards needs the tables materialized (117.9M × 5 fp32 ≈ 2.4 GB) where
the current encoder computes them on the fly — an engineering cost, not a correctness blocker.
Decoder-side change is rule-118 FREE (generic algorithm, zero counted bytes).

## 3. What is NOT measured

- **The −1,416 B is DERIVED**, by linear extrapolation of a per-symbol difference from a 1M-symbol
  IID synthetic. PR130's real stream is AR-CONDITIONAL with a per-symbol varying table; the overhead
  structure may differ. The measured-at-scale number requires re-encoding the real stream — a full
  n600 model forward, not yet run.
- The remaining 2,128 − 1,416 = 712 B of their coder gap is unexplained by the ANS swap.
- No archive was built; no evaluate.py row. No score claim. `score_claim=false`.
