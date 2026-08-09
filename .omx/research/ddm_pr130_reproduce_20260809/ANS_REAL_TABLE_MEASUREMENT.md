# The coder gap MEASURED on PR130's real conditional tables — n600 ANS recovers 99.61%

`score_claim=false` · scorer-free · MEASURED on the real model + real AR-conditional probability
tables, not a synthetic. Supersedes the synthetic projection in
`CORRECTION_clean60_lost_and_the_coder_gap.md` §2 (which UNDER-predicted).

Axis: `[macOS-MPS table materialization + macOS-CPU entropy coding, scorer-free]`.
This is not contest score authority.

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

| step | Δ bytes | derived archive bytes | derived S |
|---|---:|---:|---:|
| PR130 base | — | 191,052 | 0.172141297491896447 |
| + split-stream brotli pack (MEASURED, parse-back exact) | −903 | 190,149 | 0.171540027 |
| + ANS token coder (**MEASURED n600**) | **−2,120** | **188,029** | **0.170128405876608123** |

**n600 UPGRADE 2026-08-09 — PROJECTED → MEASURED.** On the axis above, the full
600-frame re-encode landed
(rc=0, 681 s, receipt `ans_n600/ans_vs_range_n600_result.json`):

```
frames 600 · ideal_B 114,851.8 · range_B 116,980 · ans_B 114,860
range overhead vs ideal +1.8530%  ·  ANS overhead vs ideal +0.0071%
```

**LENGTH CONTROL PASSED EXACTLY:** the re-encoded range-coder stream is **116,980 B**, the same
length as the shipped token stream. The harness did not persist or hash either newly encoded word
stream, so this is not byte-for-byte equality. Both arms nevertheless used the same real symbols,
tables, and coder process; the measured claim here is their serialized lengths only.

**Coder inefficiency recovered: 99.61%.** ANS sits 8.2 B (+0.0071%) above the model's own
cross-entropy. This near-entropy length is what RATE_AXIS §4's "no generic coder win exists" was
measuring, and confirms the lever was the CODER, not the packing.

The 60-frame projection (−2,080) was **1.9% low** vs the measured −2,120. Direction and magnitude
both held; the small window was not misleading here.

The two byte deltas are disjoint-section arithmetic. The derived 188,029 B and score are not a
materialized archive or evaluator row. Model reconstruction is bit-identical, and a real n2 ANS
round-trip proves the causal lossless mechanism; n600 ANS words were not retained or decoded.

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
   detached launcher. ✅ **CLOSED: the relaunch completed rc=0 in 681 s and the row above is now
   MEASURED at n600 (−2,120 B), with the range-coder arm reproducing the shipped 116,980 B exactly.**
   Two-landing cure: `tools/launch_detached_process.py` now refuses (rc=5) any argv matching the
   reaper predicate, with an executed positive control on this exact script. The rule previously
   existed only in a memory file, which is why it did not fire.
2. **Receiver change — CLOSED at TOY-BRACKET n2, not n600.** The owned receiver now has explicit
   legacy/split-Brotli/split-raw-LZMA2 model dispatch plus Range/ANS dispatch. Durable tagged Range
   archives preserve the measured ZIP sizes, and their model bundles reconstruct to SHA-256
   `62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517`. A pinned
   constriction-0.5.0 real n2 run decoded 393,216/393,216 tokens exactly and exhausted the ANS
   state in 2.683 s. This is explicitly not n600 receiver authority.
3. **ANS encode-side memory**: encoding backwards requires the conditional tables materialized,
   where the current encoder streams them. `probability_table` returns float32, so the raw table
   field is 2,359,296,000 B (2.197 GiB), plus 471,859,200 B of int32 symbols. The model first
   quantizes logits to int16; spilling those exact codes would reduce the table field to
   1,179,648,000 B. Reverse-chunk helpers are built and tested, but the resumable n600 materializer
   and interrupted/resumed real-table proof remain owed.
4. No archive built with ANS tokens; no `evaluate.py` row. `score_claim=false`.
5. ⚠ **THE −2,120 B IS A LENGTH, NOT A PAYLOAD — my defect, found by `ddm_rc1_receiver`
   (`5de03569ad`).** `ans_n600/ans_real_n600.py` computed `len(enc.get_compressed().tobytes())`
   and DISCARDED the bytes (line 37 literally `del enc`; line 41 same shape for ANS). The
   measurement is real — the faithfulness control passed and the range arm reproduced 116,980 B
   exactly — but **no ANS words were retained, so no archive can be assembled from this run.**
   The composed 188,029 B / `S=0.170128405876608123` is arithmetic over a measured length, not a
   built object. Re-run retaining the words (SSD atomic int16 chunks + resume receipt; the tables
   are ~4.7 GB and the run is 681 s) before any archive claim.
   **Correction to how MAIN framed this:** the blocker was NOT "no receiver" — the receiver IS
   built and selector-explicit as of `5de03569ad`. The blocker is the discarded payload.
6. ⚠ **DECODE WALL-CLOCK IS AN UNCLOSED GATE, newly named.** RC1 measured n2 ANS decode at
   2.683 s; a **linear** extrapolation to n600 is **~805 s (13.4 min) of ANS decode alone**, with
   rendering SEPARATE and on top, against the contest's **30-min total** budget
   (`upstream/README.md:114`). Extrapolated from n=2 — very weak evidence, promotable only by a
   real n600 decode. The quantity that actually matters is **ANS-decode-time MINUS range-decode-time**
   (range decode is already inside the shipped budget), and that delta is UNMEASURED. It is
   possible that PR130 chose the range coder partly for decode speed; we have not checked.
   Until measured, treat −2,120 B as a rate win with an unpriced wall-clock cost.
