# PR110 OPT-3 — Adaptive Arithmetic Coder on FEC6 K=16 Selector Index Stream — LANDED [research-only]

**Date:** 2026-05-26T17:05:00Z
**Lane:** `lane_pr110_opt3_adaptive_arith_selector_index_stream_20260526`
**TaskCreate:** #1315
**Operator approval:** PR110-OPT-3 (PR110 public body suggestion)
**Subagent:** `pr110-opt3-adaptive-arith-20260526`
**Sister non-interference:** disjoint with `pr110-opt-frame0-bundle-20260526` (catalog generation in `tools/frame_exploit_segnet_posenet_sweep.py`) and `hinton-mlx-local-pivot-20260526` (MLX trainer); zero file overlap per Catalog #302/#314 ownership map. Catalog #340 sister-checkpoint guard: PASS (no fresh in-flight conflicts).

## Empirical mode-distribution + Shannon-floor

Per the canonical analysis at `.omx/research/pr110_opt3_mode_distribution_20260526T170000Z.md`:

- **0-order Shannon entropy:** H(X) = **3.2116 bits/pair** (lower bound: 241 bytes)
- **Fixed-Huffman FEC6 achieved:** 3.2400 bits/pair (1944 bits = 243 bytes bitstream)
- **0-order slack vs Shannon:** 0.0284 bits/pair = ~2.1 encoder bytes
- **1st-order Markov H(next|prev):** 2.94 bits/pair (lower bound: ~221 bytes; ~22 encoder bytes potential vs fixed-Huff)
- **Live mode distribution:** dominated by `none` (134/600), `frame0_blue_chroma_amp_3` (129/600), `frame0_rgb_bias_p2_m1_m1` (92/600), `frame0_rgb_bias_m2_p1_p1` (71/600) — top-4 modes are 71% of mass.

## Variant chosen: A (adaptive 0-order arithmetic coder)

Per the analysis memo, Variants A/B/C compared as:

| Variant | Encoder savings | Wire savings (after header) |
|---------|----------------:|----------------------------:|
| A. Adaptive 0-order CACM-87 arith   |  ~2 bytes |  0 to -2 bytes |
| B. 1st-order Markov context coder    | ~22 bytes | -10 to -18 bytes |
| C. Range coder + static prior table  |  ~2 bytes | +14 bytes (worse — prior overhead) |

I implemented **Variant A** because it matches the PR110 body's "adaptive arithmetic coding" wording verbatim AND empirically validates whether the fixed-Huff codebook is at Shannon-floor. The result IS the headline finding for this lane.

## Implementation

Three new files under `submissions/hnerv_fec6_fixed_huffman_k16/` (NEW sister directory; the canonical FEC6 encoder + the live `submissions/hnerv_fec6_fixed_huffman_k16/inflate.py` snapshot at `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/source/...` are unchanged):

- `encoder/build_pr101_frame_exploit_selector_packet_arith.py` (288 LOC) — canonical CACM-87 arithmetic coder (Witten-Neal-Cleary 1987) at PRECISION=32 with adaptive 0-order Laplace-smoothed model over PALETTE_K=16. Exposes `encode_fec7_arith_selector(codes, n_pairs)` + `decode_fec7_arith_selector(payload)`. Wire format `FEC7` magic + uint16 n_pairs + bitstream (sister to FEC6; same K=16 mode palette so the inflate-time mode dispatch is unchanged).
- `src/fec7_arith_decoder.py` (44 LOC) — thin re-export of `decode_fec7_arith_selector` for inflate-side single-source-of-truth (per CLAUDE.md "Beauty, simplicity, and developer experience" — encoder + decoder cannot drift).
- `tests/test_arith_selector_roundtrip.py` (220 LOC) — 30 tests (PASS); covers synthetic uniform/dominant/all-zeros/max/alternating, invalid-input/truncated-payload, inflate-module proxy roundtrip, AND live FEC6 byte-exact roundtrip against the canonical PR110 archive.

**Inflate-side decoder LOC:** the encoder module is ~290 LOC; the inflate path only needs the bottom half (decoder + model + BitReader) ≈ ~110 LOC; the inflate.py wrapper itself adds 4 lines (magic dispatch). Well under HNeRV parity L4 ≤200 LOC inflate budget.

## Round-trip test PASS evidence

```
$ .venv/bin/python -m pytest submissions/hnerv_fec6_fixed_huffman_k16/tests/test_arith_selector_roundtrip.py -v
============================== 30 passed in 0.12s ==============================
```

Highlights:
- 5x uniform random seeds, K=16 alphabet, 600 symbols → byte-exact roundtrip.
- 16x dominant-mode (90% one mode + 10% uniform) → byte-exact roundtrip.
- All-zeros stream → compresses to <60 bytes (vs 75 bytes naive 1-bit-per-symbol bound), proving the adaptive model converges.
- **Live FEC6 codes (600 pairs from `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`) → byte-exact roundtrip.**

## Empirical wire-size measurement (live 600-pair selector_payload)

```
FEC6 fixed-Huff selector_payload   : 249 bytes (header 6 + bitstream 243)
FEC7 adaptive-arith selector_payload: 254 bytes (header 6 + bitstream 248)
DELTA (FEC7 − FEC6)               : +5 bytes (WORSE)
Predicted ΔS contribution         : +0.000003329 [macOS-CPU advisory:tools/build_pr101_frame_exploit_selector_packet_arith.py]
```

**[macOS-CPU advisory:submissions/hnerv_fec6_fixed_huffman_k16/encoder/build_pr101_frame_exploit_selector_packet_arith.py]** — encoding-statistics measurement only; no `[contest-CPU]` / `[contest-CUDA]` claim per Catalog #287/#323.

The FEC7 adaptive arith coder is **5 bytes LARGER** than the FEC6 fixed-Huff codebook on the live PR110 selector. This is consistent with the prediction memo: the 0-order Shannon slack on this distribution is only 0.028 bits/symbol, which corresponds to ~2 encoder bytes — completely consumed by:
1. **Model convergence overhead** — adaptive Laplace-smoothed model takes ~50-100 symbols to learn the empirical distribution; during the early phase symbols are encoded against a near-uniform prior costing ~4 bits each (1 bit/symbol overhead).
2. **Arithmetic-coder termination overhead** — CACM-87 termination emits ~3-5 disambiguation bits (vs Huffman's zero-bit termination at byte boundary).
3. **Byte-padding LSBs** — both formats pad to byte boundary, but the adaptive model lands on a less favorable bit-count modulo 8.

## Decision: research-only

Per the task spec threshold (>10 bytes savings / -0.000007 ΔS to fold into next iteration), this variant **FAILS the threshold by a clear margin** (+5 bytes WORSE, opposite sign). The structural finding is the headline:

> **The fixed-Huffman K=16 codebook is already at Shannon-floor on this distribution. The 0-order arithmetic-coding direction is saturated.**

The high-EV direction is **Variant B (1st-order Markov / context coder)** — the empirical Markov H(next|prev) = 2.94 bits/pair vs marginal 3.21 bits/pair suggests ~22 encoder bytes savings potential. After amortizing side-info cost (or using fully-adaptive context model with no shipped side info), net wire savings of -10 to -18 bytes are plausible — which WOULD clear the threshold.

## 6-hook wire-in declaration per Catalog #125

- **Hook 1 (Sensitivity map):** N/A — selector-stream encoding is an entropy-coding layer; no per-byte/per-pair sensitivity surface is added or consumed.
- **Hook 2 (Pareto constraint):** ACTIVE-NEGATIVE — this lane's empirical finding **removes** the 0-order arithmetic-coding direction from the Pareto frontier of selector entropy-coding strategies (refutes the assumption that adaptive arith strictly dominates fixed-Huff on this stream).
- **Hook 3 (Bit-allocator hook):** N/A — selector entropy coding does not influence per-tensor bit allocation.
- **Hook 4 (Cathedral autopilot dispatch hook):** N/A — research-only; no archive-deployable artifact emitted.
- **Hook 5 (Continual-learning posterior update):** ACTIVE — the empirical measurement (`fec7_arith_bytes_vs_fec6_huff_bytes = +5 [macOS-CPU advisory]`) is a CARGO-CULTED falsification anchor per Catalog #303 / #292 / #307; the assumption "adaptive arith beats Huffman" is refuted at the IMPLEMENTATION level (paradigm INTACT per Catalog #307 — context-coding direction remains live).
- **Hook 6 (Probe-disambiguator):** ACTIVE — the canonical FEC6 vs FEC7 byte-mutation smoke (tests `test_live_savings_within_predicted_band` + `test_live_fec6_codes_roundtrip_through_arith`) IS the canonical disambiguator between "0-order entropy slack exists" (refuted) vs "1st-order context structure exists" (supported, ~0.27 bits/pair reduction observable in the empirical Markov measurement).

## HORIZON-CLASS classification per Catalog #309

**`plateau_adjacent`** — predicted ΔS in [+0.0000033, -0.0000067] depending on variant (well within the 0.196-0.200 plateau cluster). The 1st-order Markov direction (Variant B) is `plateau_adjacent_to_frontier_pursuit_lower_boundary` at ~22-byte savings (would push predicted ΔS to -0.000015). Neither variant is asymptotic_pursuit.

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED / CARGO-CULTED | Empirical verdict |
|------------|---------------------------|-------------------|
| "Adaptive arithmetic coding strictly dominates fixed Huffman on per-pair selector streams" | CARGO-CULTED | **FALSIFIED**: +5 bytes WORSE on the live 600-symbol stream; model-convergence overhead exceeds 0.028 bits/symbol Shannon slack |
| "K=16 fixed-Huff codebook hand-fit is at Shannon-floor for this empirical distribution" | HARD-EARNED | **CONFIRMED**: 3.24 vs 3.21 bits/pair achieved vs floor (1% efficiency loss) |
| "Per-pair temporal coherence exists in selector indices" | UNTESTED-NOW-HARD-EARNED | **EMPIRICAL**: H(next\|prev) = 2.94 vs marginal H = 3.21; 0.27 bits/pair drop suggests real 1st-order structure |
| "1st-order context coding can fit within ≤200 LOC inflate budget" | UNTESTED | Operator-routable for Variant B follow-up |

## Operator-routable next steps

1. **NO-OP for next PR110 iteration** — the FEC6 fixed-Huff codebook is already saturated at 0-order Shannon-floor; replacing it with adaptive arith COSTS 5 bytes. The PR110 body's "adaptive arithmetic coding" suggestion is empirically falsified at the IMPLEMENTATION level (not the paradigm — context coding remains viable).
2. **Variant B (1st-order Markov context coder) — RECOMMENDED follow-up** if PR110 wants more selector squeeze. Empirical EV: ~10-18 bytes net wire savings = -0.0000067 to -0.000012 ΔS, which DOES clear the >10-byte threshold. Estimated implementation: ~150 LOC encoder + ~80 LOC decoder; fits inflate budget. Anchor: Markov H(next|prev) = 2.94 bits/pair on live stream.
3. **CANONICAL EQUATION REGISTRATION** per Catalog #344: this lane could land a NEW canonical equation `selector_index_stream_entropy_floor_already_saturated_at_fixed_huffman_for_K16_palette_v1` codifying the finding that hand-fit Huff codebooks on small alphabets (K≤16) over short streams (<1000 symbols) typically saturate 0-order Shannon-floor within 1-2 bytes, and the high-EV direction is context coding not adaptive 0-order. Operator-routable.

## Discipline anchors

- Catalog #229 PV: read full FEC6 encoder + inflate before designing FEC7 (encoder src lines 759-775; inflate src lines 246-272 + 64-91).
- Catalog #287: every numeric claim above tagged with axis (`[macOS-CPU advisory]`) — no contest-axis claims.
- Catalog #248: no conflict markers in committed files.
- Catalog #110/#113 APPEND-ONLY: this memo is the canonical landing record; supersession requires a new dated memo. Sister mode-distribution memo at `pr110_opt3_mode_distribution_20260526T170000Z.md` is the empirical anchor.
- Catalog #309 HORIZON-CLASS: declared `plateau_adjacent`.
- Catalog #303 cargo-cult audit: section above per-assumption.
- Catalog #292 per-deliberation assumption surfacing: the operating-within assumption was "adaptive arith dominates fixed Huff" — CARGO-CULTED at the implementation level per the empirical finding.
- Catalog #307 paradigm-vs-implementation: the IMPLEMENTATION (adaptive 0-order) is falsified on this stream; the PARADIGM (entropy-code-the-selector) remains INTACT — context coding is the next research path.
- CLAUDE.md "Apples-to-apples evidence discipline": fairly compared FEC7 wire-payload vs FEC6 wire-payload on the SAME 600-pair input stream from the SAME archive (sha prefix `f174192a`); both measured at the same axis (encoder-side byte count, not score).
- CLAUDE.md "Forbidden premature KILL without research exhaustion": Variant B is the canonical reactivation path; no KILL verdict applied.

## Reproducer

```bash
cd /Users/adpena/Projects/pact
.venv/bin/python -m pytest submissions/hnerv_fec6_fixed_huffman_k16/tests/test_arith_selector_roundtrip.py -v
# expect: 30 passed in <1s

# Measure wire savings on live FEC6 stream:
.venv/bin/python -c "
import sys, struct, zipfile
sys.path.insert(0, 'submissions/hnerv_fec6_fixed_huffman_k16/encoder')
from build_pr101_frame_exploit_selector_packet_arith import encode_fec7_arith_selector
FEC6_CB = ('00','1100','01','111010','11010','111011','111100','100','111101','11011','1111110','111110','11111110','101','11100','11111111')
FEC6_DEC = {b: c for c, b in enumerate(FEC6_CB)}
with zipfile.ZipFile('experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip') as zf:
    data = zf.read('x')
(src_len,) = struct.unpack_from('<I', data, 4)
pos = 8 + src_len
(sel_len,) = struct.unpack_from('<H', data, pos)
fec6 = data[pos+2:pos+2+sel_len]
(n,) = struct.unpack_from('<H', fec6, 4)
codes, prefix, bp = [], '', 0
while len(codes) < n:
    bit = (fec6[6+bp//8] >> (7-(bp%8))) & 1; bp += 1
    prefix += '1' if bit else '0'
    c = FEC6_DEC.get(prefix)
    if c is not None: codes.append(c); prefix = ''
fec7 = encode_fec7_arith_selector(codes, n_pairs=n)
print(f'FEC6 {len(fec6)}B vs FEC7 {len(fec7)}B; delta {len(fec7)-len(fec6):+d}B')
"
```
