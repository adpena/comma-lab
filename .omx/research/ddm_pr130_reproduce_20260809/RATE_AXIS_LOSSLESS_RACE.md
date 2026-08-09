# The PR130 base, opened up: section anatomy and the free lossless win

**Why this measurement:** the operator's steer — review work is instrumental; pivot to frontier score
lowering. On the reproduced PR130 base, the *rate* axis is the one axis that can be moved and priced
**exactly on this laptop with no scorer and no CUDA**: if a re-encoded archive decodes to bit-identical
output, d_seg and d_pose are unchanged **by construction**, so the whole score delta is `25·Δbytes/W`.

Base: `archive.zip` 191,052 B, sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`,
S = 0.172141297491896447 `[contest-CUDA, DALI GT, n600]`. `score_claim=false` for everything below —
these are byte measurements, not eval rows.

## 1. Anatomy (MEASURED, from the receiver's own parser)

The ZIP holds **one stored member** `p` (method 0, no ZIP compression), 190,952 B + 100 B ZIP overhead.

```
p          = [u32 models_bytes][ LZMA(models_raw) ][ HPAC arithmetic-coded token stream ]
models_raw = [u32 sem][u32 car][ semantic ][ pose carrier ][ hpac weights ]
```

| section | bytes | share of p | S contribution |
|---|---:|---:|---:|
| header u32 | 4 | — | — |
| models, LZMA'd | 73,968 | 38.74% | 0.0492523 |
| token stream | 116,980 | 61.26% | 0.0778922 |

`models_raw` uncompressed is 83,493 B — so the shipped LZMA ratio is only **0.8859**. That weak ratio is
diagnostic: the bundle is already int4/int12-packed, so little *statistical* redundancy remains for a
byte-oriented coder. Sub-sections: semantic 40,252 · pose carrier 23,054 · hpac weights 20,179.

(Note: these are SECTION SIZES. The leave-one-out *marginals* recorded in hot state are a different
quantity and do not sum to the total; both are legitimate, they answer different questions.)

## 2. Coder race — models bundle (lossless, decode-identical)

| coder | bytes | Δ vs shipped | ΔS |
|---|---:|---:|---:|
| **LZMA (as shipped)** | **73,968** | — | — |
| lzma preset=9\|EXTREME | 75,036 | +1,068 | +0.0007111 |
| lzma2 xtreme pb=0 lc=0 | 73,961 | −7 | −0.0000047 |
| lzma1 RAW xtreme pb=0 lc=0 | 73,950 | −18 | −0.0000120 |
| zlib −9 | 76,829 | +2,861 | +0.0019050 |
| bz2 −9 | 79,310 | +5,342 | +0.0035570 |
| **brotli q11 lgwin24** | **73,371** | **−597** | **−0.0003975** |

Note the shipped LZMA already beats `preset=9|EXTREME` by 1,068 B — the pb/lc settings matter more than
the preset, which is the signature of a byte-aligned packed payload.

## 3. Split-stream (L23) — each section with its own best coder

| section | raw | best coder | bytes |
|---|---:|---|---:|
| semantic | 40,252 | brotli q11 | 35,033 |
| pose carrier | 23,054 | brotli q11 | 23,058 |
| hpac weights | 20,179 | brotli q11 | 14,962 |
| **total + 12 B framing** | | | **73,065** |

- vs shipped joint LZMA: **−903 B → ΔS −0.0006013**
- brotli-free variant (lzma2 xtreme pb=0 lc=0 per section): 73,734 B = **−234 B → ΔS −0.0001559**

**The pose carrier is incompressible**: 23,054 → 23,058 under brotli, 23,058 under lzma2. Every coder
loses. Those bytes are at their entropy *as represented*; only a different representation shrinks them,
never a better coder.

## 4. Token stream — no generic-coder win exists

| attempt | bytes | Δ |
|---|---:|---:|
| shipped (HPAC arithmetic) | 116,980 | — |
| brotli q11 | 116,985 | +5 |
| lzma2 xtreme | 117,044 | +64 |

+5 B from brotli means the arithmetic stream is essentially incompressible — i.e. **HPAC has already
driven the tokens to its model's entropy.** The lever on 61% of the archive is therefore the *model* or
the *token content*, never the coder. This is a clean negative and it is worth more than a small win:
it forecloses an entire family.

## 5. Honest arithmetic against the goal

To reach S < 0.15 from 0.172141 by rate alone: ΔS = −0.02214 → **Δbytes = −33,254 B (17.4% of the
archive)**.

- free lossless win, brotli: −903 B = **2.7%** of that
- free lossless win, brotli-free: −234 B = **0.7%**

So the lossless recode is **real, free, and small**. It should be banked because it costs nothing and
compounds with everything else, but it is **not** the sub-0.15 path and must not be narrated as one.

The 33 KB has to come from one of:
1. **tokens (116,980 B)** — the only section large enough. Since the stream is at its model's entropy,
   the levers are (a) a better/larger AR prior (spend model bytes to buy token bytes), or (b) fewer or
   coarser tokens (lossy — moves d_seg, needs the scorer).
2. **pose carrier (23,054 B)** — representation change only; the coder axis is measured shut.
3. **joint rate+distortion** — trade seg/pose headroom for bytes, which needs the scorer and therefore
   CUDA.

**The model-vs-code exchange rate is the unmeasured, gap-sized question:** the HPAC weights cost
14,962 B (recoded) and buy the compression of a 116,980 B stream. If d(tokens)/d(model) < −1, growing
the prior is a net win. Nothing in our receipts shows PR130 optimized that point.

## 6. Cost of the brotli increment

The brotli-only gain over the brotli-free split is 669 B. The runtime tree currently self-installs
`constriction` and asserts numpy/torch as host-provided. brotli would be a **second self-install, and a
required one** — you cannot fall back to LZMA at decode time, because the archive is already
brotli-encoded. That is a real availability risk against a −0.0004 S gain, and it should be decided on
that trade, not adopted by default.

## 7. What was NOT checked

- No decode was run (the receiver requires CUDA at `inflate.py:665`); bit-identity is argued from
  section reconstruction, and would need an actual decode to be *demonstrated* rather than derived.
- Per-tensor byte-maps / storage permutations (L21/L22) were **not** raced — only whole-section coders.
  Those are the natural next lossless increment and are unmeasured.
- No score claim. `score_claim=false`.
