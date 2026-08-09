# The coder gap MEASURED on PR130's real conditional tables — ANS recovers 97% of it

`score_claim=false` · scorer-free · MEASURED on the real model + real AR-conditional probability
tables, not a synthetic. Supersedes the synthetic projection in
`CORRECTION_clean60_lost_and_the_coder_gap.md` §2 (which UNDER-predicted).

## Measurement

PR130's shipped HPAC model + the recorded encode argv (reused VERBATIM from
`ddm_pr130_encode_tokens_metal_20260809/run/launch_manifest.json`, `--frames 60 --device mps`),
`codec_hpac_integer.encode` monkeypatched to collect every `(symbols, table)` pair in order and then
race both coders over the identical sequence:

| coder | bytes | overhead vs the model's own entropy |
|---|---:|---:|
| ideal (cross-entropy of the real tables) | 11,397.6 | — |
| `queue.RangeEncoder` (PR130's) | 11,612 | **+1.8809%** |
| **`stack.AnsCoder`** | **11,404** | **+0.0559%** |

**Δ = 208 B on 60 frames → ≈2,080 B projected at n600 → ΔS ≈ −0.0013850.**

### Why the window is trustworthy

The 60-frame range overhead (**+1.8809%**) reproduces the SHIPPED n600 overhead
(**+1.8530%**, from `tokens.codec.json`: realized 116,980 vs ideal 114,852) to within 0.03
percentage points. The window is representative of the full stream, and the shipped gap is 2,128 B —
so ANS at 2,080 B recovers **97.7%** of the entire coder gap.

**The earlier synthetic said −1,416 B. The real conditional distribution says −2,080 B. The IID
synthetic UNDER-predicted by 32%** — real AR tables are more skewed per-symbol than the fixed
distribution I approximated them with, and the range coder's quantization penalty grows with skew.

## Composed arithmetic vs the PR130 base

| step | Δ bytes | archive | S |
|---|---:|---:|---:|
| PR130 base | — | 191,052 | 0.172141297 |
| + split-stream brotli pack (MEASURED, parse-back exact) | −903 | 190,149 | 0.171540027 |
| + ANS token coder (measured mechanism, n600 PROJECTED) | −2,080 | **188,069** | **≈0.170154897** |

The two compose exactly — disjoint sections (model bundle vs token stream), no interaction term.
Distortion unchanged on both: the model bundle reconstructs bit-identically, and ANS is a lossless
re-coding of the identical symbol sequence under the identical model.

## OWED / NOT MEASURED

1. **n600 ANS re-encode RUNNING** (pid 77736, log `ans_vs_range_n600.log`) — converts the −2,080
   projection into a measurement. Until it lands, −2,080 is PROJECTED from a 60-frame window whose
   overhead matches the full stream to 0.03pp, not measured at n600.
2. **Receiver change OWED, both legs**: the 3-stream model parser AND an ANS (LIFO) token decoder.
   Both are rule-118 FREE (generic decoder algorithm, zero counted bytes) but neither is built. The
   intake copy is read-only; the change belongs in our own receiver.
3. **ANS encode-side memory**: encoding backwards requires the conditional tables materialized
   (117.9M × 5 ≈ 2.4 GB fp32) where the current encoder streams them. Engineering cost, not a
   correctness blocker; the n600 run is testing exactly this.
4. No archive built with ANS tokens; no `evaluate.py` row. `score_claim=false`.
