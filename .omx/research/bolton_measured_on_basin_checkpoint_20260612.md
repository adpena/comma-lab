# Bolt-on rate-attack measurement on the LIVE base_ch=20 HNeRV basin checkpoint

**Date:** 2026-06-12
**Subagent:** `bolton-measure-20260612b` (resumed a dead predecessor; nothing had landed)
**Authority:** `[contest-CPU advisory] NON-PROMOTABLE` (lossless byte-recode measurement; no exact-eval dispatch)
**Frontier:** UNMOVED — 0.19109982. This measures the **bolt-on contribution** to a substrate basin, NOT a new frontier row.
**Mission contribution:** `frontier_breaking_enabler` (negative/null result that retires a presumed lever)

## TL;DR — the bolt-on stack contributes ESSENTIALLY ZERO on this checkpoint

| Bolt-on | What it recodes | Byte Δ | Parse-back | Verdict |
|---|---|---|---|---|
| **A — adaptive brotli `lgwin`** | both brotli streams, window-size search | **−15 B** | lossless ✓ | tiny positive |
| **B — raw-LZMA latents** (PR95 L24) | latents stream brotli→LZMA1 RAW | **+82 B (REGRESS)** | bit-identical ✓ | honest negative |
| **C — per-tensor categorical AC** (PR103 / PR95 L30) | decoder INT8 stream, AC vs brotli per-tensor | **+188 B (REGRESS)** | bit-identical ✓ | honest negative |

**Best achievable: 90427 → 90412 bytes (−15 B, bolt-on A only).**
Implied score delta: `25 · 15 / 37,545,489 = 0.0000100` — i.e. **~+1.0e-5 score improvement**. Negligible.

This is a REAL, parse-back-verified measurement (every recode decodes back to the bit-identical
INT8 decoder state-dict + latents tensor the vendored codec parses, so d_seg/d_pose are unchanged
by construction — no lossy claim, no fake reduction).

## The checkpoint measured

`experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/` (LIVE basin — not disturbed):
- `best_ema_decoder.pt` (28 tensors, 83,356 params), `best_ema_latents.pt` (600×28)
- best_meta.json: epoch 100, stage1_v328_ce, score 0.7485, d_seg 0.005715, d_pose 0.001363, rate 0.002408, archive_bytes 90427

### Baseline archive byte breakdown (vendored PR95 codec, `codec.build_archive`)
- **total 90,427 B**  (build is deterministic — verified)
- meta_brotli: 81 B (0.09%)
- **decoder_brotli: 74,715 B (82.62%)** ← the dominant rate term
- latents_brotli: 15,619 B (17.27%)
- 4-byte section prefixes: 12 B

## Why every entropy-coding bolt-on is at-or-below break-even (the decisive measurement)

The decoder INT8 weights are **near-uniformly distributed** — they sit at the order-0 Shannon
limit and brotli already achieves it:

| tensor | n | order-0 entropy | order-0 bytes | brotli bytes | brotli / order-0 |
|---|---|---|---|---|---|
| stem.weight | 26,880 | 7.053 b/sym | 23,698 | 23,908 | **1.009** |
| blocks.0.weight | 14,400 | 6.929 b/sym | 12,472 | 12,602 | **1.010** |
| stem.bias | 960 | 7.305 b/sym | 877 | 933 | 1.064 |

**Brotli is within ~0.9–1.0% of the per-tensor order-0 entropy bound.** A memoryless
Categorical arithmetic coder (PR103) cannot beat that bound, and its 256-byte-per-tensor
histogram sidecar pushes it strictly over (gross AC-vs-brotli win = 0 on all 28 tensors;
net −188 B with headers). There is no high-order structure left for AC to exploit and no
order-0 slack for brotli to give back. The rate is **floored by the INT8 representation itself**,
not by codec slack.

### On the PR103 "~217 B better" docstring claim
The vendored codec docstring says it dropped per-tensor categorical AC because it was "only ~217
bytes worse" than AC on *PR95's own fully-trained checkpoint*. That does NOT transfer here:
- PR95's claim was on a converged decoder where weights resemble peaked-categorical IID draws (AC-friendly).
- This basin checkpoint is EARLY (epoch 100 of the 8-stage curriculum) with near-uniform 7.05 b/sym weights.
- Substrate/training-stage-dependent — a real finding, not a tool failure. Per Catalog #307 this is
  IMPLEMENTATION/INPUT-LEVEL: the AC paradigm is intact; it simply has no headroom on THIS input.

## Integration gaps (honest)
- **PR101 split-brotli / PR103 `encode_decoder_ac`** are hard-wired to PR103's own tensor ordering +
  `AC_TENSOR_INDICES=(0,2,4,6,8,10,12,21)` and a multi-stream blob shape. They do NOT ingest the
  vendored single-brotli grammar directly. Rather than force-fit them (fake "applied"), I reused
  their **canonical AC primitives** (`pack_ac_stream`/`unpack_ac_stream`/`_build_q8_histogram`/
  `_make_categorical`) directly on the basin's per-tensor zigzag bytes — the same math, applied
  honestly to this substrate. Result: no win (see above).
- **`tools/build_pr110_payload_entropy_recode_candidate.py`** and `hnerv_decoder_recode.py` target
  the PR110/HNeRV archive shapes, not the vendored grammar; not run to avoid a fake-apply.
- **`score_aware_weight_requant_sweep.py`** is a LOSSY requant lever (would change d_seg/d_pose and
  require a torch-CPU re-measure). Out of scope for this lossless bolt-on measurement; flagged as the
  one remaining lever with potential headroom (see below).

## What this means for the campaign
1. **The vendored PR95 codec is already at the entropy floor for this representation.** Lossless
   byte-recode bolt-ons (brotli tuning / LZMA / arithmetic coding) are exhausted: max −15 B (~+1e-5 score).
   Do NOT spend more cycles on lossless entropy-coding bolt-ons for the base_ch=20 basin.
2. **Rate progress on this substrate must come from REPRESENTATION, not coding:** fewer/smaller
   tensors (base_ch, latent_dim), lower-bit quantization (sub-INT8 / mixed precision via the
   score-aware requant sweep — LOSSY, needs re-measure), or latent-count reduction. The 74.7 KB
   decoder is 82.6% of the archive and is incompressible past INT8.
3. The bolt-on stack's contribution to the FINAL (better-trained) substrate will be similar-or-smaller
   in *byte* terms (a more converged decoder is more AC-friendly but the absolute headroom on an
   already-floor-bound brotli stream stays ≲ a few hundred bytes ≈ ≲ 2e-4 score). Bolt-ons are not
   the path to sub-0.15 on this basin; the substrate quality (d_seg/d_pose) is.

## Reproducibility
- Harness: `experiments/results/bolton_measure_basin_bc20_20260612/measure.py` (A + B),
  `measure_ac.py` (C), results: `results.json`.
- Read-only on the LIVE basin (`best_*.pt`); no GPU; pure-CPU; vendored codec via
  `tac.torch_vehicle.vendored_imports.import_vendored("codec")`; AC via `tac.pr103_arithmetic_codec`.
- All recodes parse-back-verified bit-identical (max_abs_err = 0.0 for C; brotli/LZMA losslessness for A/B).

## Wire-in (6-hook per Catalog #125)
1. sensitivity-map: N/A (no score-axis change; lossless byte recode).
2. Pareto constraint: ACTIVE — adds the empirical point "PR95 vendored codec is at the order-0 entropy
   floor for INT8 HNeRV decoder weights; lossless bolt-on headroom ≤ 15 B on base_ch=20."
3. bit-allocator hook: N/A (no per-tensor importance change).
4. cathedral autopilot: N/A (advisory, non-promotable, no archive deployed).
5. continual-learning posterior: this memo IS the anchor — retires the "apply orthogonal lossless
   rate-attack bolt-ons to the basin" lever as null (≤ +1e-5 score).
6. probe-disambiguator: ACTIVE — the order-0-entropy-vs-brotli table is the disambiguator between
   "codec has slack" (FALSE) vs "representation is floor-bound" (TRUE).
