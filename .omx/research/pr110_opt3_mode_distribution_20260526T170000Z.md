# PR110 OPT-3 — Empirical Mode-Distribution Analysis of FEC6 K=16 Selector Stream

**Date:** 2026-05-26T17:00:00Z
**Lane:** lane_pr110_opt3_adaptive_arith_selector_index_stream_20260526
**Source archive:** experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip
**Archive sha256 (member `x`):** f174192aeadfccf4... (selector lives in FP11 wrapper at byte offset 8+source_len+2; selector_payload sha256 of first 16 bytes = 4645433658023f4bfc3bf7b7c5e3339f)
**TaskCreate:** #1315
**Sister non-interference:** disjoint with `pr110-opt-frame0-bundle-20260526` (catalog generation) and `hinton-mlx-local-pivot-20260526` (MLX trainer).

## Method

Decoded the live FEC6 selector_payload from the canonical PR110 archive using the unmutated `unpack_fec6_fixed_huffman_codes` decoder copied verbatim from `submissions/hnerv_fec6_fixed_huffman_k16/inflate.py`. Computed 0-order Shannon entropy + 1st-order Markov conditional entropy on the 600-pair index stream.

## Empirical mode histogram (`[macOS-CPU advisory]` not score — pure encoding statistics)

| code | mode_id | count | freq | huff_bits | huff_len |
|------|---------|------:|-----:|----------:|---------:|
|  0 | `none`                            | 134 | 0.2233 | `00`       | 2 |
|  1 | `frame0_blue_chroma_amp_1`        |  35 | 0.0583 | `1100`     | 4 |
|  2 | `frame0_blue_chroma_amp_3`        | 129 | 0.2150 | `01`       | 2 |
|  3 | `frame0_luma_bias_+1`             |   9 | 0.0150 | `111010`   | 6 |
|  4 | `frame0_luma_bias_-1`             |  25 | 0.0417 | `11010`    | 5 |
|  5 | `frame0_luma_bias_-2`             |  13 | 0.0217 | `111011`   | 6 |
|  6 | `frame0_luma_bias_-4`             |  11 | 0.0183 | `111100`   | 6 |
|  7 | `frame0_rgb_bias_m2_p1_p1`        |  71 | 0.1183 | `100`      | 3 |
|  8 | `frame0_rgb_bias_m4_p2_p2`        |  10 | 0.0167 | `111101`   | 6 |
|  9 | `frame0_rgb_bias_p0_m1_p1`        |  24 | 0.0400 | `11011`    | 5 |
| 10 | `frame0_rgb_bias_p0_m2_p2`        |   7 | 0.0117 | `1111110`  | 7 |
| 11 | `frame0_rgb_bias_p0_p1_m1`        |  16 | 0.0267 | `111110`   | 6 |
| 12 | `frame0_rgb_bias_p0_p2_m2`        |   6 | 0.0100 | `11111110` | 8 |
| 13 | `frame0_rgb_bias_p2_m1_m1`        |  92 | 0.1533 | `101`      | 3 |
| 14 | `frame0_rgb_bias_p4_m2_m2`        |  17 | 0.0283 | `11100`    | 5 |
| 15 | `frame0_roll_dx+0_dy+1`           |   1 | 0.0017 | `11111111` | 8 |
| **TOTAL** | | **600** | 1.0000 | | |

## Information-theoretic bounds

- **0-order Shannon entropy:** H(X) = **3.211618 bits/pair**
- **Shannon lower-bound bytes:** ceil(H * 600 / 8) = **241 bytes**
- **Fixed-Huffman achieved bits/pair:** 1944 / 600 = **3.2400 bits/pair**
- **Fixed-Huffman achieved bytes:** 243 (bitstream) + 6 (header) = **249 bytes (selector_payload total)**
- **0-order slack vs Shannon:** 3.24 - 3.21 = **0.0284 bits/pair**
- **Max bytes savable by 0-order arithmetic coder vs fixed-Huff:** (1944 - ceil(1926.97)) / 8 = **~2.12 bytes (encoder side); -0 to -2 bytes wire selector_payload after header overhead**

- **1st-order Markov conditional entropy:** H(next | prev) = H(prev, next) - H(prev) = 6.1535 - 3.2133 = **2.9402 bits/pair**
- **1st-order Markov lower-bound bits:** H(X_1) + (n-1) * H(next|prev) = 3.21 + 599 * 2.94 = **1764 bits = 221 bytes**
- **Max bytes savable by 1st-order context coder vs fixed-Huff:** (1944 - 1764) / 8 = **~22 bytes (encoder side); after context-side-info overhead estimate -10..-18 bytes wire**

## ΔS contribution (predicted)

Per the canonical contest rate formula `25 × archive_bytes / 37_545_489`:

| Variant | Encoder savings | Predicted wire savings (after header) | Predicted ΔS |
|---------|----------------:|--------------------------------------:|-------------:|
| A. Adaptive 0-order range coder       |  ~2 bytes |   0 to -2 bytes | 0 to -0.0000013 |
| B. 1st-order Markov context coder     | ~22 bytes | -10 to -18 bytes | -0.0000067 to -0.000012 |
| C. Range coder + static prior table   |  ~2 bytes |  -16B prior overhead → +14 bytes WORSE | +0.0000093 (worse) |

**Threshold per task spec:** >10 bytes savings = -0.000007 ΔS = "fold into next iteration".

## Variant analysis

**Variant A (adaptive 0-order).** Shannon-floor-asymptotic but slack is already 0.028 bits/pair on this 600-symbol stream → measurement noise dominates. Predicted savings ≤ 2 bytes encoder side; after the 6-byte header (which must persist for n_pairs + magic + n_specs disambiguation per FEC2/FEC3/FEC5/FEC6/FEC7 multiplexer), wire savings are 0 to -2 bytes. **Below noise floor.** Research-only.

**Variant B (1st-order Markov / context coder).** Saves ~22 encoder bytes but requires shipping the 16×16 transition table (256 cells × ~4 bits each ≈ 128 bytes) OR a static prior table OR online adaptation. Online adaptation (adaptive Markov) ships zero side info but converges slowly on 600 symbols (16 contexts × few-symbols-per-context). Net wire savings: -10 to -18 bytes after side info amortizes. **Above noise floor.** Operator-routable for follow-up if PR110 next iteration genuinely wants more squeeze.

**Variant C (range coder + static prior).** Equivalent asymptotically to Variant A (0-order Shannon). Adding the 16-byte prior table makes wire payload LARGER, not smaller. **Eliminated by analysis.**

## Decision

Implement **Variant A (adaptive 0-order range coder)** as the cleanest reference implementation matching the operator's "adaptive arithmetic coding" wording. Verify empirically (predict 0 to -2 bytes wire). Document Variant B as operator-routable follow-up (the actual high-EV path) with the empirical 22-byte 1st-order Markov upper bound as the design anchor.

**The honest answer:** The fixed-Huffman K=16 codebook is already within 0.028 bits/symbol of Shannon-floor on this empirical distribution; further 0-order squeeze is below the measurement-noise floor. The 1st-order Markov structure (H drops from 3.21 to 2.94 bits/pair) is where the real ~22-byte EV lives — but that requires Variant B + careful side-info amortization vs ship-prior tradeoff. The PR110 body's mention of "adaptive arithmetic coding" likely meant the 0-order variant, which empirically saturates here.

## Sister-discipline cross-references

- Catalog #287 evidence-tag: every numeric claim above is encoding-statistics not score (no `[contest-CPU]` / `[contest-CUDA]` claims).
- Catalog #309 HORIZON-CLASS: `plateau_adjacent` (sub-noise-floor 0-order savings; 1st-order would be `plateau_adjacent` to `frontier_pursuit_lower` boundary at ~22 bytes saved).
- Catalog #303 cargo-cult audit (per-assumption): the "Huffman is suboptimal" assumption surfaces empirically as CARGO-CULTED on this stream — the K=16 codebook was hand-fit and lands within 1% of Shannon-floor (HARD-EARNED that 0-order is saturated).
- Catalog #244 dispatch protocol: N/A (no paid dispatch).
- Catalog #110/#113 APPEND-ONLY: this memo is the canonical empirical anchor; supersession requires a new dated memo.

## Reproducer

```python
from collections import Counter
import math, struct, zipfile, sys

FEC6_CB = ("00","1100","01","111010","11010","111011","111100","100",
           "111101","11011","1111110","111110","11111110","101","11100","11111111")
FEC6_DEC = {b: c for c, b in enumerate(FEC6_CB)}

with zipfile.ZipFile("experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip") as zf:
    data = zf.read("x")
assert data[:4] == b"FP11"
(src_len,) = struct.unpack_from("<I", data, 4)
pos = 8 + src_len
(sel_len,) = struct.unpack_from("<H", data, pos)
sel = data[pos+2:pos+2+sel_len]
assert sel[:4] == b"FEC6"
(n_pairs,) = struct.unpack_from("<H", sel, 4)
bits = sel[6:]

codes, prefix, bp = [], "", 0
while len(codes) < n_pairs:
    bit = (bits[bp//8] >> (7-(bp%8))) & 1
    bp += 1
    prefix += "1" if bit else "0"
    c = FEC6_DEC.get(prefix)
    if c is not None:
        codes.append(c); prefix = ""

H = -sum((v/n_pairs) * math.log2(v/n_pairs) for v in Counter(codes).values())
print(f"H = {H:.4f} bits/pair; fixed-Huff = {len(bits)*8/n_pairs:.4f}; slack = {(len(bits)*8/n_pairs)-H:.4f}")
```
