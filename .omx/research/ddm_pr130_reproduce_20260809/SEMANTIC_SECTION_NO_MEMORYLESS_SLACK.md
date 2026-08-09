# THE CODER AXIS IS CLOSED on all four sections — three are already BELOW their memoryless bound

> **Scope note on the filename:** this doc opened as the semantic-section measurement and grew to
> cover all four sections. The axis-complete table is §"All four sections" below; the semantic
> detail is retained beneath it.

`score_claim=false` · scorer-free · MEASURED 2026-08-09 on the reproduced PR130 archive
(sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B).
`[macOS-CPU advisory]`.

## All four sections — MEASURED, the axis is complete

Every section of the archive, against its own order-0 (memoryless) entropy bound. The bound is
assumption-free: it is the byte-histogram entropy of the exact bytes being coded.

| section | achieved | memoryless floor | achieved − floor | coder-swap verdict |
|---|---:|---:|---:|---|
| **tokens** 116,980 → **114,860** (ANS) | 114,860 | 114,852 (model cross-entropy) | **+8 B (+0.0071%)** | **HARVESTED −2,120 B** |
| semantic | 35,033 (brotli-q11) | 36,805 | **−1,772 B** | DEAD (loses ≥1,772) |
| pose | 23,054 (canonical Huffman, shipped) | 22,989 | +65 B (+0.28%) | DEAD (re-code measured **+4 B WORSE**) |
| hpac | 14,962 (brotli-q11) | 16,567 | **−1,605 B** | DEAD (loses ≥1,605) |

**Read this table as the closure of an axis.** Three sections are already *below* what any memoryless
coder can achieve — they buy that with inter-byte/LZ structure. The fourth (pose) sits 0.28% above its
own floor, i.e. its Huffman is essentially optimal for its own symbol statistics. The ONE section that
carried real coder slack was tokens, because it alone has an **explicit conditional model** (HPAC AR)
that the range coder was coding 1.85% worse than; ANS took that to +0.0071%.

**Consequence, binding on every future rate arm:** on this archive there is no coding win left. Every
remaining rate byte must come from a **better REPRESENTATION or a better MODEL**. An arm that proposes
a coder race on any section of this base is proposing a measured-dead cell — point it at this table.

## Why this was asked

The token section just gave up **−2,120 B** to an entropy-coder swap (ANS for range, same model,
`ANS_REAL_TABLE_MEASUREMENT.md`). The obvious next question is whether the same lever exists on the
semantic section (36,580 B marginal = 19.15% of archive = 0.0243571 S). It does **not**, and the
reason is mechanically clear.

## Measurement

Raw semantic section: **40,252 B** (`models_raw[8 : 8+sem_len]`).

| bound / achievement | bytes | note |
|---|---:|---|
| order-0 **byte** entropy floor | **36,805 B** | 7.3150 bits/B over the raw section — assumption-free |
| order-0 nibble floor | 37,127 B | 3.6895 bits/nibble; assumes 4-bit codes packed 2/byte |
| order-1 nibble conditional floor | 36,764 B | previous nibble as context |
| **brotli-q11 ACHIEVED** (split-stream receipt) | **35,033 B** | **1,772 B BELOW the memoryless floor** |

The byte-entropy row needs no assumption about the payload layout, and it alone settles the question:
**brotli is 4.8% below what any memoryless coder can reach.** It is buying that with inter-byte
structure (LZ matches over repeated quantized-weight patterns), which an order-0 or order-1 arithmetic
coder cannot see.

## Verdict

**A memoryless / order-0 / order-1 entropy coder on this section LOSES by ≥1,772 B.** Do not run that
race. Scope: INSTANCE — this section, this quantization (quant_bits=4, 66,339 params), this coder.

## The mechanism, stated generally (the transferable part)

A coder swap harvests slack only where an **explicit conditional model already exists** and the coder
codes worse than that model knows:

| section | explicit model? | coder vs its own model | coder-swap verdict |
|---|---|---|---|
| tokens 116,980 B | yes — HPAC AR conditionals | range coder **+1.8530% OVER** the model cross-entropy | **WON −2,120 B (ANS)** |
| semantic 36,580 B | no — raw packed weights; the coder IS the model | brotli **−4.7% BELOW** the order-1 bound | **LOSES ≥1,772 B** |
| pose 23,384 B | canonical-Huffman, at its own byte entropy (7.9817 b/B) | split-stream re-code **+4 B WORSE** | **LOSES (closed)** |

Where the coder is the model, you must beat it with a **better model**, not a better coder — and
brotli's LZ structure is a strong incumbent. Same mechanism `sv2` measured on IX2TOK01
(commit `fec6dae38b`: the win moved from symbol-rank cost to LZ MATCH STRUCTURE, and every symbol-rank
arm lost); this is an independent second instance on a different object.

## What WOULD be a live lever here (not run, named honestly)

A better MODEL, not a better coder — and it must beat brotli's LZ exploitation, which is the bar:
- per-tensor / per-layer conditioning (the section is a concatenation of structurally distinct
  tensors; one global model over the concatenation is a modelling choice nobody derived)
- positional context within conv kernels (spatial position in a k×k kernel is a real context variable)
- cross-tensor structure (permutation / shared codebook — L21/L22-class, whose L21 arm already lost
  on our own vehicle, so this is RACE-not-adopt)

Each is a *model* change and must be raced against 35,033 B, not against the 36,805 B floor. Nothing
here is measured; it is a named next step with the correct bar attached.

## Reproduce

```
.venv/bin/python - <<'PY'
import struct, lzma, zipfile, collections, math
z = zipfile.ZipFile("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip")
blob = z.read("p"); n = struct.unpack("<I", blob[:4])[0]
raw = lzma.decompress(blob[4:4+n])
sem_len, pose_len = struct.unpack("<II", raw[:8]); sem = raw[8:8+sem_len]
h = collections.Counter(sem); t = sum(h.values())
H = -sum(c/t*math.log2(c/t) for c in h.values())
print(len(sem), f"{H:.4f} bits/B", f"floor {t*H/8:.0f} B")
PY
```
