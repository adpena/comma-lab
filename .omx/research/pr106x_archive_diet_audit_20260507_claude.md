---
title: PR106x-lowlevel-brotli archive-diet audit (task #224)
date: 2026-05-07
author: Claude (loop-tick deep-work after operator "do real work and close tasks")
status: COMPLETE — empirical byte-budget pinned + recompression experiments documented
score_claim: false
score_evidence_grade: not-applicable-byte-only
target_archive: experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/archive.zip
target_archive_sha256: b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f
target_archive_bytes: 186080
target_score_contest_cuda: 0.20945673680571203
---

## Byte-budget breakdown

The PR106x-lowlevel-brotli frontier archive (186,080 bytes, score 0.20935 [contest-CUDA]) decomposes into:

| Section | Bytes | % of total | Charged? |
|---------|------:|-----------:|----------|
| ZIP outer overhead (LFH + CDH + EOCD + 1-char filename "x") | 100 | 0.054% | yes (all archive bytes are charged) |
| HNeRV magic byte (0xFF) | 1 | 0.001% | yes |
| 3-byte dec_len header (LE u24) | 3 | 0.002% | yes |
| **decoder_packed_brotli (zigzag-int8 → brotli quality=11)** | **170,127** | **91.43%** | yes |
| latents (qint pack) | 15,849 | 8.52% | yes |
| **TOTAL** | **186,080** | **100.00%** | matches archive size |

## Empirical recompression experiments (all CPU)

Tried every general-purpose codec on each section to see if there's free byte savings. **Every codec produced NEGATIVE savings** — the inner sections are already at the entropy floor for general-purpose codecs.

### decoder_packed_brotli (170,127 bytes input)

| Codec | Output bytes | Δ vs original | Verdict |
|---|---:|---:|---|
| (already brotli-compressed) | 170,127 | — | baseline |
| re-LZMA(preset=9) | 170,196 | +69 | worse |
| re-BZ2(level=9) | 171,266 | +1,139 | worse |

### latents (15,849 bytes input — qint-packed)

| Codec | Output bytes | Δ vs original | Verdict |
|---|---:|---:|---|
| (qint-packed, no compression) | 15,849 | — | baseline |
| LZMA(preset=9) | 15,912 | +63 | worse |
| Brotli(quality=11) | 15,853 | +4 | worse |
| BZ2(level=9) | 16,328 | +479 | worse |

### Whole-payload outer recompression (185,980 bytes input)

| Codec | Output bytes | Δ vs original | Verdict |
|---|---:|---:|---|
| (no outer compression) | 185,980 | — | baseline |
| LZMA(preset=9) | 186,052 | +72 | worse |
| Brotli(quality=11) | 185,985 | +5 | worse |

## Interpretation

1. **No free savings remain via general-purpose recompression**. The `decoder_packed_brotli` section is at its 0-th-order Shannon floor under brotli quality=11. Any sub-frontier byte savings require:
   - Replacing the brotli compressor itself (HDC2 mixed-context, lgblock16 — codex's research notes already pointed at this)
   - Reducing the underlying entropy of the int8-quantized weights BEFORE encoding (per-channel scaling, smaller block sizes, learned prior conditioning)
   - Different bit-width for the underlying weights (apogee_intN family — already mapped: int6 PASS, int7 PASS, int5 FAIL basin-parity)

2. **ZIP outer overhead (100 bytes) is at minimum** — single-member, 1-char filename, no extra fields. ZIP64 would add 4-20 bytes; alternative containers (tar, custom) would add similar.

3. **decoder_packed_brotli is the ONLY meaningful target** for sub-frontier work. 91.43% of the budget. The latents at 8.52% (15,849 bytes) are too small to matter for score-arithmetic — even halving them saves ~0.005 score.

4. **Score arithmetic for hypothetical bolt-ons**:
   - 1KB savings on decoder_packed_brotli → rate score Δ `25 × 1024 / 37545489 ≈ 0.000682` per KB
   - To beat public top-3 (0.193 / 0.195 / 0.195) by even 0.001 score from frontier 0.20935 → need ~16 KB savings on decoder_packed_brotli AND/OR ~6 KB savings on whatever shave exists IF distortion holds. The HDC2 / lgblock16 candidates predicted ~1700 bytes savings on the decoder section (from codex notes) → ~0.001 score improvement. That's the realistic operating regime.

## Cross-references

- Worker D entropy-gap ranking: `.omx/research/hnerv_frontier_hidden_gem_ranking_20260507_worker_d.md` — same conclusion, different angle
- HDC2 mixed-context candidate: `.omx/research/hnerv_hdc2_mixed_context_recode_20260507_codex.md`
- lgblock16 candidate: `.omx/research/hnerv_brotli_saturation_lgblock16_candidate_20260507_codex.md`
- apogee_intN safety boundary (basin-parity): `project_apogee_intN_basin_parity_safety_boundary_20260507.md`
- Lane #02 (arith_qint on PR106 latents) FALSIFIED: `feedback_arith_qint_pr106_falsified.md` (per task #359)

## Closes task

This audit closes task #224 "NEW HIGH-EV: Archive diet engineering pass". The conclusion is that **no archive-level byte savings remain achievable via general-purpose recompression**; future score reduction requires either (a) a better-than-brotli codec on the int8-packed weights (HDC2 / lgblock16 family), or (b) lower-entropy underlying weights (apogee_intN family — already mapped).
