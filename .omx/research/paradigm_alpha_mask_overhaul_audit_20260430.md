# PARADIGM SHIFT α — Mask Payload Overhaul: Audit (Phase 1)

**Date**: 2026-04-30
**Mandate**: Replace 421KB AV1 mask payload with sub-80KB neural codec.
**Predicted score impact**: −0.20 to −0.25 (largest single shift in Grand Council #294 battleplan).
**Approach**: 4 parallel candidates → empirical bytes → contest-CUDA → synthesis winner.

---

## Existing scaffold inventory

### Candidate α1 — Lane 12 NeRV coordinate-MLP

| Asset | Path | State |
|---|---|---|
| Codec module | `src/tac/nerv_mask_codec.py` | 846 LOC, NRV1/NRV2 magic, fp16 + int8 paths |
| Trainer | `experiments/train_nerv_mask.py` | 16.4 KB, full training loop |
| Remote dispatch | `scripts/remote_lane_nerv.sh` | canonical 4-stage |
| Tests | `src/tac/tests/test_nerv_mask_codec.py` (+ `test_lane12_nerv_dependency_closure.py`, `test_preflight_nerv_codec_discipline.py`) | 3 test files |
| Inflate dispatch | `submissions/robust_current/inflate_renderer.py:962-996,1118-1130` | NRV1 magic-byte sniff + `decode_nerv_codec` + `render_mask_argmax` |
| Empirical | `reports/lane_12_nerv_real_archive.json` | **94.4% byte savings (421,483B → 23,594B fp16; 11,845B int8)**; argmax disagreement 2.0% vs AV1 source [empirical] |
| Contest-CUDA | `lane_12_nerv_2026-04-30_b` | **In flight per active_dispatches.md** (PID 4889, predicted [1.00, 1.10]) |
| Council reviews | `council_lane_12_nerv_round{1,2,3}_20260430.md` | 3 rounds |

**Verdict**: Level 2 complete, awaiting Level 3 contest-CUDA. NO new dev work needed; only verify dispatch outcome + tag.

### Candidate α2 — Wavelet mask codec (Mallat) — NOT YET BUILT

| Asset | Path | State |
|---|---|---|
| Codec module | `src/tac/wavelet_mask_codec.py` | **MISSING — must create** |
| Renderer (related) | `src/tac/contrib/wavelet_renderer.py` | 537 LOC; this is a mask→RGB renderer using wavelet basis, NOT a mask codec |
| Wavelet primitives | `src/tac/wavelet_variance.py` (203 LOC), `haar_dwt2d` / `haar_idwt2d` in `contrib/wavelet_renderer.py` | Reusable Haar DWT primitives |
| Tests | none | **MISSING — must create** |
| Inflate dispatch | none | **MISSING — must wire** |
| Stack builder | none | **MISSING — must create** |
| Remote dispatch | none | **MISSING — must create** |

**Math foundation (to be detailed)**: 5-class mask sequence per-frame argmax → Haar DWT (2-3 levels) per-class one-hot → quantize coefficients (LL keep, LH/HL/HH threshold + uniform quantize) → arithmetic code on indices → bitstream. Inverse: ID→idx-table→iDWT→argmax.

**Sparsity assumption**: 5-class masks have large flat regions (sky/road) → LL dominates; LH/HL/HH sparse outside class boundaries. Fridrich UNIWARD lineage: detail subbands compress to near-zero, only boundary coefficients carry meaningful bits.

### Candidate α3 — VQ-VAE mask codec (van den Oord) — NOT YET BUILT

| Asset | Path | State |
|---|---|---|
| Codec module | `src/tac/vqvae_mask_codec.py` | **MISSING — must create** |
| VQ-VAE renderer (related) | `src/tac/contrib/vqvae_codec.py` | 881 LOC; this is a frame→RGB VQ-VAE replacing the mask path entirely (different paradigm), NOT a mask-stream codec |
| VQ primitives | `VectorQuantizer` in `contrib/vqvae_codec.py` | Reusable codebook + STE |
| Tests | none | **MISSING — must create** |
| Inflate dispatch | none | **MISSING — must wire** |
| Stack builder | none | **MISSING — must create** |
| Remote dispatch | none | **MISSING — must create** |

**Math foundation (to be detailed)**: Mask sequence → tokenize 4×4 patches per frame → look up nearest codebook entry K=256 → arithmetic code per-token stream over time + spatial prior. Codebook (K=256, D=16) ~16KB; token stream 600×6×8 bits ≈ 28KB; arithmetic-coded ≈ 12-20KB. Total <40KB target.

### Candidate α4 — Selfcomp grayscale-LUT (already partially built)

| Asset | Path | State |
|---|---|---|
| Codec module | `src/tac/mask_grayscale_lut.py` | 178 LOC: `CLASS_TO_GRAY`, `encode_masks_grayscale`, `decode_grayscale_to_classes`, `create_gaussian_softmax_lut` |
| Stack builder | `experiments/build_lane_mm_archive.py` | 242 LOC: full re-encode of Lane A archive's masks.mkv → grayscale.mkv (CRF 50, libsvtav1, monochrome, deterministic ZIP) |
| Tests | `src/tac/tests/test_lane_mm_archive_*` (TBD verify) | partial |
| Inflate dispatch | `submissions/robust_current/inflate_renderer_grayscale.py` | EXISTS — separate inflate variant |
| Remote dispatch | none specific (Lane MM batch dispatch) | partial |
| Empirical | none yet — Lane MM v2 score 2.63 was FALSIFIED (encoder-only with 3ch-trained renderer) | **Need re-run with retrained renderer OR decoder-side LUT** |

**Math foundation (already in module docstring)**: 5-class → grayscale gray ∈ [0,255] via `CLASS_TARGETS = [0, 255, 64, 192, 128]`; AV1 monochrome encode (1 chroma plane skipped); inflate: AV1 decode → Gaussian-softmax LUT (σ=15) → 5-class probability → argmax. Predicted ~50% smaller than 5ch-AV1 because (a) chroma planes skipped and (b) 64-pixel quantizer gaps tolerate decode noise.

**Verdict**: Level 1.5 (encoder works, but 2026-04-29 falsification showed PoseNet 51× regression on 3ch-trained renderer). Need retrained renderer OR pure decoder-side LUT (drop-in mask path).

---

## Gaps to fill (this session)

1. **α2 Wavelet**: build `src/tac/wavelet_mask_codec.py` + tests + stack builder + remote dispatch + STRICT preflight check.
2. **α3 VQ-VAE**: build `src/tac/vqvae_mask_codec.py` + tests + stack builder + remote dispatch + STRICT preflight check.
3. **α4 Grayscale-LUT**: build a **decoder-side-only** stack builder that drops `grayscale.mkv` into the canonical Lane G v3 archive WITHOUT touching the renderer (skip retraining, use Gaussian-softmax LUT to convert grayscale → 5ch logits at inflate time).
4. **α1 NeRV**: monitor `lane_12_nerv_2026-04-30_b` dispatch, log Level 3 result.

---

## Score-arithmetic priority (Shannon LEAD's ranking)

`dScore/dByte = 0.00067` for any byte saved on the archive. Mask stream is currently 421,483B / 694,074B = 60.7% of archive. Per-candidate predicted score impact:

| Candidate | Predicted bytes | Bytes saved vs 421KB | Predicted Δscore | Tag |
|---|---|---|---|---|
| α1 NeRV | 23,594 (fp16) / 11,845 (int8) | 397,889 / 409,638 | −0.265 / −0.273 | [empirical] verified, [contest-CUDA] in flight |
| α2 Wavelet | 50,000-90,000 (predicted) | 331,483 - 371,483 | −0.221 to −0.247 | [prediction] |
| α3 VQ-VAE | 30,000-50,000 (predicted) | 371,483 - 391,483 | −0.247 to −0.260 | [prediction] |
| α4 Grayscale-LUT | 200,000-280,000 (predicted, AV1 monochrome) | 141,483 - 221,483 | −0.094 to −0.147 | [prediction] |

**Caveat**: distortion impact must hold within Lane G v3's PoseNet/SegNet bands. NeRV's 2.0% argmax disagreement is within tolerance (Quantizr leader does ~3-5% with frame1-warped duality). Wavelet/VQ-VAE distortion TBD empirically.

**Composability**: All four are MASK STREAM replacements; only ONE may ship in the final archive. The synthesis step picks the lowest [contest-CUDA] score with archive < 100KB.

---

## Cross-references

- Grand Council strategy: `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- Score arithmetic: Section 1.3
- Prior NeRV scaffold audit: `.omx/research/lane_12_nerv_scaffold_audit_20260430.md`
- Selfcomp reverse engineering: `memory/project_selfcomp_reverse_engineered_20260429.md`
- Lane MM v2 falsification: `memory/project_lane_mm_v2_landed_2_63_falsified_20260429.md`
- Phase 2/3/4 detailed designs: `memory/project_phases_2_3_4_design_implementation_math_provenance_20260429.md`
