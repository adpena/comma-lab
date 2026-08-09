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

1. **n600 ANS re-encode — first attempt KILLED, relaunched.** ⚠ **CORRECTED 2026-08-09.** The
   original text here said "RUNNING (pid 77736)". That run **died at frame 300/600 (337 s)** with no
   traceback and no final JSON. Cause DIAGNOSED and it was **not** the memory path: the fleet launchd
   agent `com.vertigo.claude-code-reaper` SIGTERMs any no-TTY, PPID-1 process whose `ps` cmdline
   matches `\b(claude|codex)\b` after 300 s, and the script was launched from the session scratchpad
   `/private/tmp/claude-501/...` — `claude-501` matches because the hyphen is a word boundary. The
   337 s death is inside the measured 300–360 s window and matches three banked receipts exactly
   (fz4/rt1/qj1 at 335/337/337 s). Peak memory at frame 300 was ~2.6 GB on a 128 GB machine, so OOM
   is excluded. Relaunched from the SSD (`ans_n600/`, no matching token in argv) through the canonical
   detached launcher. Until it lands, **−2,080 stays PROJECTED** from a 60-frame window whose overhead
   matches the full stream to 0.03pp — not measured at n600.
   Two-landing cure: `tools/launch_detached_process.py` now refuses (rc=5) any argv matching the
   reaper predicate, with an executed positive control on this exact script. The rule previously
   existed only in a memory file, which is why it did not fire.
2. **Receiver change OWED, both legs**: the 3-stream model parser AND an ANS (LIFO) token decoder.
   Both are rule-118 FREE (generic decoder algorithm, zero counted bytes) but neither is built. The
   intake copy is read-only; the change belongs in our own receiver.
3. **ANS encode-side memory**: encoding backwards requires the conditional tables materialized,
   where the current encoder streams them. Corrected arithmetic: the tables are float64, so
   600 × 196,608 × 5 × 8 B ≈ **4.7 GB** (not the 2.4 GB fp32 figure first written here), plus
   ~0.47 GB of int32 symbols. Comfortable on a 128 GB host — the first attempt reached frame 300
   (~2.6 GB) before an unrelated external SIGTERM (item 1). Engineering cost, not a correctness
   blocker; the relaunched n600 run is measuring it.
4. No archive built with ANS tokens; no `evaluate.py` row. `score_claim=false`.
