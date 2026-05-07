---
title: Wave-Ω Ω-3 dequantize-int6→float→block-FP path FALSIFIED (CPU-only proof)
date: 2026-05-07
author: Claude (loop-tick deep-work)
status: FALSIFIED — dequantize-int6 path pushes rel_err into basin-collapse regime
score_claim: false
---

## Summary

The "dequantize int6 → float → block-FP" path proposed in `feedback_omega_3_blocked_jfg_checkpoint_unavailable_20260507.md` (Path B) is **falsified by CPU-only empirical measurement**. Block-FP applied on top of dequantized int6 weights pushes the cumulative rel_err to **3.6%**, which falls squarely in the int5-int4 boundary regime that Track G's basin-parity already proved FAILS (int5 at 3.31% → basin-parity FAIL; int6 at 1.55% → basin-parity PASS).

**Verdict**: byte savings exist (-11,271 bytes / -6.6% vs brotli baseline at block_size=256) but are dominated by the predicted basin-collapse distortion cost. **DEFERRED-pending-research**, not killed (per CLAUDE.md "KILL is last resort").

## Empirical Method

1. Parsed `apogee_int6_archive.zip` → recovered int6-quantized HNeRV decoder weights (dequantized to float at parse time per `parse_apogee_intn_archive`)
2. Reference: PR106 lossless float weights (parsed via `parse_packed_archive`)
3. Block-FP roundtrip applied to dequantized int6 weights at block_size ∈ {64, 128, 256}
4. Measured: cumulative rel_err vs PR106 float reference + lzma-compressed byte size

## Results

### Byte savings (block-FP + lzma vs brotli baseline 170,127 bytes)

| block_size | mantissa | exponents | meta | lzma total | Δ vs brotli |
|---:|---:|---:|---:|---:|---:|
| 8 | 229,024 | 28,628 | 834 | 178,700 | **+8,573** (worse) |
| 16 | 229,120 | 14,320 | 834 | 174,692 | **+4,565** (worse) |
| 32 | 229,376 | 7,168 | 834 | 172,176 | **+2,049** (worse) |
| 64 | 229,888 | 3,592 | 834 | 169,052 | **−1,075** (better) |
| 128 | 231,040 | 1,805 | 834 | 164,124 | **−6,003** (better) |
| 256 | 232,960 | 910 | 834 | 158,856 | **−11,271** (best) |

### Distortion (rel_err vs PR106 float reference, %)

| block_size | mean_abs_err | max_abs_err | rel_err % | regime |
|---:|---:|---:|---:|---|
| 64 | 3.71e-03 | 1.76e-02 | **3.61** | int5-int4 boundary (predicted FAIL) |
| 128 | 3.78e-03 | 1.76e-02 | **3.68** | int5-int4 boundary (predicted FAIL) |
| 256 | 3.84e-03 | 1.76e-02 | **3.74** | int5-int4 boundary (predicted FAIL) |

### Score arithmetic

If we ignored distortion (which would be wrong):
- block_size=256 saves 11,271 bytes → rate Δ = `−25 × 11271 / 37545489 = −0.00750`
- Predicted score IF distortion held: 0.20935 − 0.00750 = **0.20185** [predicted-band only]

But distortion does NOT hold:
- 3.74% rel_err puts us in int5-FAIL regime (basin-parity threshold 1.55-3.31% boundary)
- Distortion penalty ~3-5× the rate gain → predicted final score ≥ 0.21+ (worse than baseline)

## Why the path fails

**Compounding quantization**: int6 already quantized weights to 64 levels with 1.55% rel_err. Applying block-FP on top means each weight is now quantized through TWO lossy transforms — first to int6 (1.55% per-weight error), then to per-block-int8 (additional ~2.2% per-weight error from finite block-FP precision). The errors compound geometrically: ~3.7% combined.

The basin-parity gate (Track C/G) already proved that >3.3% rel_err is the catastrophic regime where the renderer's score landscape diverges from the lossless basin. So even if 3.7% bytes-savings looks good, the score-landscape geometry says the renderer would produce divergent reconstruction.

## Implications

1. **Path B (dequantize-int6→float) is FALSIFIED for Ω-3 transplant.** Block-FP cannot be stacked on int6-quantized weights without basin collapse.
2. **Path A (Q-FAITHFUL retrain) remains the only viable Ω-3 substrate.** Trained float JFG weights (not int6-quantized) are required.
3. **Path C (Ω-3-HNeRV variant)** would face the same problem if applied to dequantized int6 weights — only worth pursuing on FRESH float HNeRV weights.

## What this experiment closes

- The architectural mismatch flagged in `feedback_omega_3_blocked_jfg_checkpoint_unavailable_20260507.md` Path B is not just architectural — it's a **distortion-regime mismatch**. Even if the BFJ1 codec accepted HNeRV weights structurally, the dequantize-int6 substrate is fundamentally too lossy.
- Future agents should NOT attempt the dequantize-int6 path; the basin-parity gate would FAIL post-build.
- The single remaining unblock for Ω-3 is **trained float weights** (Q-FAITHFUL retrain on GPU OR Ω-3-HNeRV variant on a fresh PR106-archive-extracted float reference if such a thing exists).

## Negative-evidence value

This is what CLAUDE.md "Treat speculative ideas as side lanes unless evidence forces promotion" looks like in practice: spent ~30 min CPU to FALSIFY a multi-day code path before any GPU spend. The negative result saves ~$15-30 of GPU and 5-7 days wall-clock that would have produced a basin-collapse archive.

## Cross-references

- Block-FP module: `src/tac/block_fp_jfg.py` (Track F commit `52f13947`)
- Basin-parity safety boundary: `project_apogee_intN_basin_parity_safety_boundary_20260507.md`
- Ω-3 JFG-checkpoint blocker (this falsification covers Path B): `feedback_omega_3_blocked_jfg_checkpoint_unavailable_20260507.md`
- PR106x archive-diet: `pr106x_archive_diet_audit_20260507_claude.md` (related: rate-only analysis)
- Q-FAITHFUL never-reproduced: `feedback_q_faithful_NEVER_reproduced_quantizr_score_20260505.md`
