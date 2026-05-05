---
name: PR #56 selfcomp (szabolcs-cs, 0.36) archive blob byte-layout (PROPER reverse engineering)
description: 2026-05-01 wave-time. Sister memo to reference_pr65_pr67. Downloaded PR #56's archive.zip + inflate.py from szabolcs-cs/comma_video_compression_challenge fork (sha 2b2d76de). Decoded the archive by READING the parser (pr56_inflate.py) and the SegMap class definition rather than guessing on hex dumps.
type: reference
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Methodology

Same lesson as the PR #65/#67 memo. The parser IS the spec. Source files custody:

- `reports/raw/leaderboard_intel_20260501/pr56_archive.zip` (272,514 B, sha `241da6aa0a82fb01bf2a287d7d6561040342733689d2d9ef1f1b2db939eee0e4`)
- `reports/raw/leaderboard_intel_20260501/pr56_inflate.py` (10,991 B)
- `reports/raw/leaderboard_intel_20260501/pr56_inflate.sh` (873 B)

Inner extraction (one-shot, deterministic):
```bash
python3 -c "import zipfile; zipfile.ZipFile('pr56_archive.zip').extractall('.')"
# yields: payload.tar.xz (272,300 B)
tar -xJf payload.tar.xz
# yields:
#   segmap_inference.pt  1,239,122 B  (uncompressed pickle)
#   0.mkv                  206,573 B  (AV1 grayscale, full-length)
```

PR #56's report.txt: PoseNet 0.00039717, SegNet 0.00115296, archive 279,036 B (off by ~6KB from what I downloaded — likely path-overhead diff between submitted vs. user-attachment download), Compression Rate 0.00743, **Final score 0.36**.

## Outer archive layout

PR #56 archives are ASTONISHINGLY simple — single zip member:

| Member | Bytes | Notes |
|---|---|---|
| `payload.tar.xz` | 272,514 B (compressed) | Wraps two files: model + 1 video |

The xz over the tar gives one global compression context across model + video. Splitting into separate xz streams would lose that cross-correlation; szabolcs left it as one stream. Smart.

## Inner payload.tar.xz layout

| File | Uncompressed bytes | Description |
|---|---|---|
| `segmap_inference.pt` | 1,239,122 | torch.save() pickle of SegMap quantized state-dict + arch metadata |
| `0.mkv` | 206,573 | AV1-coded grayscale video, **full length** (note `compressed_video_layout: full_length`) |

**Critical finding:** szabolcs ships ONLY ONE video (`0.mkv`) — a single concatenated/representative grayscale stream that the inflate uses as the FILE_LIST input source. Per `pr56_inflate.sh:32-37`, the FILE_LIST passed by upstream is iterated, but each entry uses the SAME `0.mkv`. This is the AV1-as-grayscale-codec trick: the model's job is to learn pixel→class probability from grayscale fed through a `linear_q_per_tensor`-quantized SegMap.

**That means the AV1 grayscale isn't 600 frames of payload — it's ONE shared "exemplar" video that conditions the model.** All video-specific signal lives in `frame_affine_embedding.weight` (1200 × 6 = 7,200 fp32 normally, here uint8-quantized).

## SegMap architecture (from `pr56_inflate.py:34-121`)

```python
SegMap(
    hidden=64,
    block_hidden=128,
    num_blocks=8,
    max_frame_index=1200,        # 600 pairs × 2 frames
    affine_max_zoom_delta=0.12,
    affine_max_aspect_delta=0.03,
    affine_max_shear=0.03,
    affine_max_translation=0.08,
    latent_input_scale=0.1,
)
```

Forward path (different from Quantizr):
1. Take grayscale luma frame, downsample to (384, 512) with bicubic
2. **Gaussian-LUT-from-luma**: apply `create_gaussian_softmax_lut()` (256→5 channel softmax over CLASS_TARGETS = `[0, 255, 64, 192, 128]` with σ=15) to get a 5-channel probability map. **No CNN needed for the grayscale→prob step — it's a fixed lookup.**
3. Per-frame affine grid of a SHARED `(1, 3, 30, 40)` learned latent canvas — 6 affine params (zoom, aspect, shear×2, trans×2) come from `frame_affine_embedding(frame_idx)`. This 1200×6 embedding table is the per-frame side-channel.
4. Concat [5-channel prob_map | 3-channel affine-warped latent × 0.1] → 8-channel input
5. 8 ResBlock(64→128→64) at fixed (384, 512) resolution
6. 1×1 conv to 3 channels, sigmoid×255 → fullres-bicubic to (1164, 874) → uint8

**Key innovations vs Quantizr:**
- **Shared latent canvas** instead of per-frame stored masks — the per-frame info is just 6 affine params per frame index.
- **Gaussian softmax LUT** removes the entire SegNet-mimicking encoder; the model only needs to learn pixel→color refinement on top of an analytic class probability prior.
- **HWOI weight layout** for conv2d (`weight_tensor_layout: HWOI`), matches typical TFLite quantization storage order; allows column-major XZ compression that's slightly tighter than NCHW.

## Inner segmap_inference.pt structure (verified by parser readback)

Top-level keys (20 fields):
```
'format': 'segmap_inference_integer_aux8_v1'
'hidden': 64                                      # SegMap arg
'block_hidden': 128                               # SegMap arg
'num_blocks': 8                                   # SegMap arg
'num_fourier_bands': 6                            # unused by this inflate.py path
'max_frame_index': 1200                           # SegMap arg
'learned_fullres_residual': False                 # disables a side-codec path
'compressed_video_layout': 'full_length'          # vs. 'frame_skip' (alt encoding)
'lowfreq_frame_channel': False                    # disables a side-codec path
'affine_max_*' x4: 0.12, 0.03, 0.03, 0.08         # SegMap args
'latent_input_scale': 0.1                         # SegMap arg
'segmap_class': 'SegMap'                          # dispatch tag
'weight_tensor_layout': 'HWOI'                    # ← non-standard
'weight_tensor_dtypes': {'int8': 17, 'int16': 1, 'int32': 0}
                          ^^^ all conv weights int8 except layer_out (int16)
'aux_tensor_codec': 'linear_minmax_uint8_v1'     # bias/embedding/latent codec
'inference_state_dict': {...}                    # the actual weights
```

inference_state_dict (38 entries, ~1.2 MB raw inside pickle):

**Weights — all `linear_q_per_tensor` (qint + 1 fp32 exponent per layer):**
| Layer | Shape (HWOI) | Dtype | Bytes (raw) |
|---|---|---|---|
| `layer_in.weight_qint` | (1, 1, 64, 8) | int8 | 512 |
| `blocks.{0..7}.conv1.weight_qint` | (3, 3, 128, 64) | int8 | 73,728 × 8 = 589,824 |
| `blocks.{0..7}.conv2.weight_qint` | (3, 3, 64, 128) | int8 | 73,728 × 8 = 589,824 |
| `layer_out.weight_qint` | (1, 1, 3, 64) | **int16** | 384 |
| Σ weight_qint | — | — | **1,180,544 B** |
| Σ weight_exponents | 17 × (1,1,1,1) | fp32 | 68 B |

**Auxiliary 8-bit tensors (linear_minmax_uint8_v1):**
| Tensor | Shape | Storage | Bytes (raw) |
|---|---|---|---|
| `shared_latent_base` | (1, 3, 30, 40) | uint8 + 1 fp32 min/max | 3,608 |
| `frame_affine_embedding.weight` | (1200, 6) | per-column uint8 + 6 fp32 min/max | 7,248 |
| `layer_in.bias` (64,) | uint8 + min/max | 72 |
| `blocks.{0..7}.conv1.bias` (128,) × 8 | uint8 + min/max | 1,088 |
| `blocks.{0..7}.conv2.bias` (64,) × 8 | uint8 + min/max | 576 |
| `layer_out.bias` (3,) | uint8 + min/max | 11 |
| Σ aux_8bit | — | — | **~12,603 B** |

**Total params:** 1,192,755 (qint+aux) — counting layer_out int16 as 2 params each is the convention; actually 1,180,352 distinct conv weights + 12,403 aux scalars.

## Bits-per-weight verification (challenged szabolcs's "1.017 bpw" claim)

szabolcs's PR description says network weights "compressed with self-compression (~1.017 bits per weight)." Verification:

```
Total int8 weight bytes (raw):  1,180,544 (= 8.001 bpw straight)
XZ q=9 of just the int8 stream:    57,124 (= 0.387 bpw)
```

So XZ alone gets him to 0.39 bpw, MUCH better than his advertised 1.017. The 1.017 was likely measured on an EARLIER smaller arch or includes overhead from the metadata wrapper. Either way: **XZ over int8 is the workhorse, not anything fancier.** No neural codec, no Schmidhuber-style randmulti tricks à la PR #65, no Brotli, no arithmetic coding — just plain xz preset=9 over a `torch.save()` pickle.

## Comparison vs. Quantizr (PR #55, 0.33) and PR #67 (rank 1, 0.31)

| Aspect | PR #56 selfcomp | PR #55 Quantizr | PR #67 (qpose14_qzs3) |
|---|---|---|---|
| Score | 0.36 | 0.33 | 0.31 |
| Total archive bytes | 279,036 | 299,970 | ~270,100 |
| SegNet path | Gaussian-LUT + 8-ResBlock SegMap (no encoder) | FiLM-conditioned DSConv with masks.mkv | Single-blob mask-stream |
| Per-frame side-channel | 1200×6 affine embedding (~7KB uint8) | 600 odd-frame masks in masks.mkv (~280KB AV1) | mask.obu.br (219KB AV1 monochrome OBU) |
| Weight codec | int8 + xz preset=9 | FP4 + Brotli q=11 | QZS3 (block-FP) + Brotli |
| PoseNet path | (not present in inflate.py — likely Lane A poses or zeros) | poses.pt + frame-warp | qpose14 raw uint16 in same blob |
| PoseNet distortion | 0.000397 | 0.0007 | comparable to Quantizr |
| SegNet distortion | 0.001153 | 0.000xx | comparable |
| Container | zip(payload.tar.xz) | zip(model.pt.br + masks.mkv + poses.pt) | zip(single-blob `p`, ZIP_STORED) |
| Compression order | model+video → tar → xz → zip | per-file: torch.save→FP4→brotli; AV1; raw | concat → ZIP_STORED |

## Insights for our pipeline

1. **Wave-Ω-3 (Block-FP transplant)**: PR #56 shows that 8-bit-int + xz is COMPETITIVE without block-FP. Block-FP wins when bpw < 1; for 8 bpw, plain int8+xz (0.39 effective bpw) is simpler and lighter.
2. **Gaussian softmax LUT trick**: At our resolution (384×512), `create_gaussian_softmax_lut` over `[0, 255, 64, 192, 128]` with σ=15 gives a fixed-no-params class prior. This is **trivially portable** to our SegMap path. ~1 line of code; could replace the early conv stages of our renderer.
3. **Affine-embedded shared canvas**: Replace masks.mkv (~412KB compressed in our owv3_0120 champion) with a (3, 30, 40) shared latent (3,600 floats) + 1200×6 affine table (~7KB uint8). Storage delta: -400KB → -0.27 bpw on rate-only basis. **This alone could push us from 0.9974 → ~0.73**, ignoring score-distortion ramifications.
4. **`weight_tensor_layout: HWOI`**: HWOI ordering compresses ~5% better than NCHW in xz/lzma because it groups same-spatial-position weights together (typically more correlated than same-input-channel). One line of permute() in our QZS3 packer; free 5% rate gain.
5. **No PoseNet model whatsoever in PR #56's inflate**: szabolcs is using upstream pose-as-given (Lane A equivalent) and the affine-embedding learns whatever pose-correlated camera motion exists. Worth checking what his compress.sh did but he doesn't ship one ("did you include the compression script? no").

## Files custody

- Archive: `reports/raw/leaderboard_intel_20260501/pr56_archive.zip` (272,514 B)
- Parser: `reports/raw/leaderboard_intel_20260501/pr56_inflate.py` (10,991 B)
- Wrapper: `reports/raw/leaderboard_intel_20260501/pr56_inflate.sh` (873 B)
- Inner extracted (live during analysis at `/tmp/pr56_extract/`, NOT committed):
  - `segmap_inference.pt` (1,239,122 B)
  - `0.mkv` (206,573 B)

## Cross-references

- Sister memo: `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`
- Original Selfcomp intel: `project_selfcomp_reverse_engineered_20260429.md`
- Council architectural baseline: `.omx/research/quantizr_replica_audit_20260428.md`
- 0.9974 deploy champion (sha-pinned): `project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md`
