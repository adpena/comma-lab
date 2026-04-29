# Lane STC boundary-mask coding design (2026-04-29)

Author: Codex design pass. Status: Stage 1 design only. No implementation in this stage.

## TL;DR

Lane STC replaces the current AV1 monochrome mask payload with a deterministic
boundary-focused mask-class codec. The encoder spends bits where SegNet class
boundaries matter, using a stored boundary bitmap plus a mask-class syndrome
stream for boundary pixels. Non-boundary interiors are encoded as majority-class
runs or majority deltas.

The current `src/tac/fridrich.py::optimal_quantization_stc` is not reusable as
the production codec. It works on continuous RGB frame quantization by choosing
ceil/floor rounding directions, has no mask-class bitstream, and is not a true
Filler/Pevny/Fridrich syndrome-trellis implementation. Lane STC needs a new
`src/tac/stc_boundary_codec.py`. It can reuse the local cost-map design pattern
from `fridrich.py`, the mask routing pattern from `mask_codec.py`, and possibly
the deterministic arithmetic-coder machinery from `arithmetic_qint_codec.py`.

Expected target: current AV1 monochrome mask payload is around 200 KB for 1200
mask frames. Lane STC target is 120-140 KB, saving 60-80 KB. The rate term for
60 KB is:

```text
Delta_score_rate = -25 * 60000 / 37545489 = -0.03995
```

Standalone expected score delta is conservatively -0.03 to -0.04 versus the
Lane G v3 1.05 baseline after bitmap overhead and implementation risk.

## 1. Mathematical formulation

Let `M[t, y, x] in {0, 1, 2, 3, 4}` be the SegNet argmax class-id mask for
frame `t`, and let `Z[t, c, y, x]` be the corresponding SegNet logit tensor
available at encode time only.

### 1.1 Pixel-cost map

The encoder computes a cheap deterministic boundary cost map from the SegNet
logit field:

```text
gx[t, c] = SobelX * Z[t, c]
gy[t, c] = SobelY * Z[t, c]
rho[t, y, x] = max_c sqrt(gx[t, c, y, x]^2 + gy[t, c, y, x]^2)
```

`rho` is encoder-time side information. It must not be recomputed from SegNet at
inflate time, because the strict-scorer rule forbids scorer/model loads in the
decoder. For the first implementation, prefer the integer-safe variant:

```text
A[t, y, x] = argmax_c Z[t, c, y, x]
rho_id[t, y, x] = sqrt((SobelX * A)^2 + (SobelY * A)^2)
```

This uses logits only to obtain the class IDs, then detects boundaries on the
class map. It is less nuanced than logit-gradient `rho`, but it is deterministic
and aligns with what the decoder can reconstruct after reading the boundary
bitmap.

### 1.2 Boundary selection

Default boundary fraction:

```text
p_boundary = 0.05
tau_t = quantile(rho[t].flatten(), 1 - p_boundary)
B[t, y, x] = 1[rho[t, y, x] >= tau_t]
```

Use a per-frame threshold by default so motion-heavy frames do not consume the
entire video-level budget. The threshold must be configurable, with a future
adaptive mode that raises or lowers `p_boundary` when the compressed bitmap or
boundary stream misses the byte budget.

### 1.3 Boundary class coding

Boundary pixels are:

```text
Omega_B = {(t, y, x) : B[t, y, x] = 1}
C_B = M restricted to Omega_B
```

The codec should encode `C_B` near its empirical conditional entropy:

```text
H(C_B | context) = -sum_r p(r) log2 p(r)
```

where `r` is a class residual relative to a deterministic context predictor.
Useful predictors are previous-frame same pixel, nearest non-boundary neighbor,
and local majority class in a small window. The Stage 2 implementation should
include an arithmetic-coded residual baseline to measure the Shannon bound and
a true syndrome-trellis path for the final STC test.

For STC proper, represent the 5-class residual stream as bitplanes or as a
small alphabet with a fixed symbol mapping. For each binary bitplane `b`, choose
a parity-check matrix `H_stc` with trellis height `h` and transmit:

```text
s = H_stc * b mod 2
```

The decoder uses the same deterministic context, the stored boundary bitmap,
and the syndrome `s` to recover the minimum-cost bitplane. This is the
Filler/Pevny/Fridrich 2011 STC role: near-optimal coding against per-position
costs with overhead expected in the 5-15% range relative to entropy. If the
first implementation cannot deliver this cleanly, keep the arithmetic-coded
residual as the oracle baseline and gate promotion on the 15% overhead test.

### 1.4 Non-boundary coding

For `Omega_N = not B`, the default reconstruction is per-frame majority class:

```text
m_t = mode(M[t] restricted to Omega_N)
M_hat[t, y, x] = m_t for non-boundary pixels
```

Lossless recovery still requires exceptions. Encode non-boundary pixels as the
smaller of:

1. constant-class run-lengths over raster order, or
2. `(gap, class_delta)` exceptions from the per-frame majority class.

Large interiors make this stream near-zero cost in practice. It is not allowed
to be lossy by default; `decode_mask_video_stc(encode_mask_video_stc(M))` must
recover the exact argmax class IDs unless a future experiment explicitly adds a
lossy mode and evaluates it with `[contest-CUDA]`.

### 1.5 Total byte model

The target total is:

```text
bytes_total =
    bytes_header
  + bytes_boundary_bitmap
  + bytes_boundary_class_syndrome
  + bytes_nonboundary_runs

bytes_boundary_class_syndrome ~= |Omega_B| * H(C_B | context) / 8 * (1 + eps_stc)
eps_stc target: 0.05 to 0.15
```

The phrase "H(boundary)" must include both the boundary-position side
information and the boundary class IDs. A raw bitmap is impossible: 1200 * 384 *
512 bits is about 29.5 MB before compression. The boundary bitmap must be
itself compressed, using frame-0 RLE plus inter-frame XOR sparse deltas or an
arithmetic/RLE hybrid. If bitmap side information exceeds roughly 15-25 KB, the
60-80 KB savings target is at risk.

## 2. Audit verdict: existing modules

### `src/tac/fridrich.py`

`compute_pixel_cost_map` at line 578 is a scorer-aware RGB-frame cost-map API.
It accepts frames plus PoseNet and SegNet modules, offers `jacobian`,
`uniward`, and `hybrid` modes, and normalizes the output to `[0, 1]`. This is
not inflate-safe because it imports and runs scorers. For Lane STC, only the
pattern is reusable: normalize a deterministic cost map, threshold it, and keep
the scoring models out of the decoder.

`optimal_quantization_stc` at line 907 is not production-ready for this lane.
Reasons:

- input is continuous RGB frames `(N, 3, H, W)`, not class IDs `(N, H, W)`;
- decision variable is ceil/floor rounding, not 5-class symbol recovery;
- implementation is greedy cost sorting, not a real syndrome-trellis code;
- it emits quantized frame tensors, not a self-describing bitstream;
- it has no decoder, no boundary bitmap, no header, and no roundtrip contract;
- it uses floating-point Torch operations, which are unnecessary risk for a
  bitexact mask decoder.

Verdict: do not wrap this function for Lane STC. Write a new mask-class codec.
At most, share naming conventions and the cost-map normalization pattern.

### `src/tac/mask_grayscale_lut.py`

This is the Selfcomp 5-class grayscale-LUT path. It maps class IDs to gray
targets `[0, 255, 64, 192, 128]` and optionally reconstructs soft class
probabilities through a Gaussian-softmax LUT. Lane STC should not depend on the
Gaussian LUT for its default mode. STC should reconstruct exact class IDs and
feed the existing renderer/mask path. The class-target ordering is still useful
as a stable 5-symbol ordering if residual symbols need a deterministic mapping.

### `src/tac/mask_codec.py`

The current canonical mask route is AV1 monochrome / grayscale video through
`encode_masks`, `decode_masks`, and `encode_masks_auto` / `decode_masks_auto`.
The new codec should be registered as a new format, for example
`stc_boundary`, with content-based magic detection in `detect_mask_codec`.

For Lane STC archives, this replaces the AV1 monochrome `masks.mkv` payload
entirely. The old AV1 path can remain for other lanes, but STC archives should
ship an STC mask payload and route decode through `decode_mask_video_stc`.

### `src/tac/arithmetic_qint_codec.py`

This module already has deterministic integer arithmetic coding with a
self-describing header and roundtrip tests. It is designed for qint streams, not
mask videos, but its static-model arithmetic coder is a good implementation
reference and likely reusable for:

- boundary bitmap entropy coding,
- non-boundary run/exception streams,
- the arithmetic residual baseline for the boundary class stream.

It is not a syndrome-trellis codec by itself.

### `src/tac/fridrich_losses.py`

This file provides the Fridrich/UNIWARD lineage for training losses and local
texture complexity. It supports the conceptual framing but should not be
imported into the STC inflate path. The STC decoder must stay pure integer /
byte parsing.

## 3. Stack composition

### Lane STC and Lane MM

Lane MM is the Selfcomp grayscale-LUT mask representation. Lane STC is a
class-ID entropy codec. They attack related mask-rate bytes but at different
levels:

- Lane MM changes the mask representation and can introduce soft/LUT semantics.
- Lane STC compresses exact argmax class IDs, especially boundary IDs.

They are not independent full savings. The additive version is: MM or
CLASS_TARGETS changes the residual statistics, then STC codes the remaining
class-id or residual stream. In that sense, Lane MM stacks only on the residual
left after the representation choice. It should not be counted as another full
60-80 KB after STC without measuring overlap.

### Lane STC and Lane SH

This document uses "Lane SH" in the user-provided sense: Selfcomp half-frame /
frame-count reduction. That is distinct from the existing
`arithmetic_qint_codec.py` Lane SH naming for qint arithmetic coding.

The half-frame lane is orthogonal to STC because it changes how many mask frames
need to be stored. Apply half-frame first, then run STC on the remaining 600
stored masks:

```text
1200 full masks --SH/half-frame--> 600 stored masks --STC--> boundary-coded payload
```

This is only valid for renderers trained for the half-frame distribution.
Memory notes show half-frame retrofits can destroy PoseNet when train/inflate
distributions diverge. STC must not be used as an excuse to ship half-frame
against an incompatible renderer.

### Replacement boundary

For a Lane STC archive, `mask_codec.py` should treat STC as the mask payload
format. It is not an AV1 postfilter, not a wrapper around `masks.mkv`, and not a
ZIP-level diet. It replaces the mask video payload.

## 4. Boundary detection and decode protocol

### Encode side

Inputs:

```text
logits: optional Tensor[N, 5, H, W]
masks:  required Tensor[N, H, W] class IDs
```

Algorithm:

1. If logits are supplied, compute `argmax(logits)` and optionally
   logit-gradient `rho`.
2. For Stage 2 default, compute integer Sobel magnitude on the argmax class-ID
   map. This avoids float roundoff and makes the boundary semantics match the
   stored class map.
3. Select the top `p_boundary` pixels per frame.
4. Encode the boundary bitmap into the STC container preamble using deterministic
   RLE/XOR/sparse-delta coding.
5. Encode boundary class residuals through `encode_stc_syndrome`.
6. Encode non-boundary majority/runs/exceptions.
7. Decode immediately and assert exact class-ID roundtrip before writing the
   final payload.

### Decode side

Inputs:

```text
stc payload bytes only
```

Algorithm:

1. Read header: magic, version, shape, class count, boundary mode, and stream
   lengths.
2. Decode boundary bitmap. No SegNet, no logits, no scorer imports.
3. Reconstruct deterministic context from decoded non-boundary runs and
   previously decoded frames.
4. Decode STC syndrome into boundary class IDs.
5. Fill non-boundary classes, apply exceptions, and return
   `Tensor[N, H, W]` class IDs.

This preserves strict-scorer-rule compliance. The encoder may spend unlimited
time computing logits and boundary maps; inflate only parses deterministic
bytes.

## 5. Predicted byte savings

Current mask payload target:

```text
AV1 monochrome masks: ~200 KB for 1200 frames
STC target payload:   120-140 KB
Savings:              60-80 KB
```

Rate-only score math:

```text
60 KB:  -25 * 60000 / 37545489 = -0.03995
80 KB:  -25 * 80000 / 37545489 = -0.05327
```

Conservative standalone score delta versus Lane G v3 1.05:

```text
expected: -0.03 to -0.04
```

The conservative delta accounts for boundary-bitmap side information, archive
headers, STC overhead above Shannon entropy, and the chance that the first
exact-lossless implementation lands closer to 140 KB than 120 KB.

## 6. Stage 2 and Stage 3 task list

### Stage 2: local codec and tests

Add `src/tac/stc_boundary_codec.py` with:

- `detect_boundary_pixels(masks, logits=None, boundary_fraction=0.05, mode="argmax_sobel", per_frame=True) -> torch.Tensor`
- `encode_stc_syndrome(boundary_classes, context_symbols, costs, *, num_classes=5, trellis_height=10) -> bytes`
- `decode_stc_syndrome(blob, context_symbols, costs, *, num_classes=5, expected_symbols: int) -> torch.Tensor`
- `encode_mask_video_stc(masks, output_path, logits=None, boundary_fraction=0.05, verify_roundtrip=True, **cfg) -> int`
- `decode_mask_video_stc(input_path) -> torch.Tensor`

Internal helpers expected:

- `encode_boundary_bitmap` / `decode_boundary_bitmap`
- `encode_nonboundary_runs` / `decode_nonboundary_runs`
- `estimate_symbol_entropy_bits`
- `measure_stc_overhead`

Add `src/tac/tests/test_stc_boundary_codec.py` with approximately these tests:

- `test_detect_boundary_pixels_density_near_configured_5_percent`
- `test_encode_decode_roundtrip_exact_class_ids`
- `test_encoded_payload_respects_byte_budget_on_synthetic_masks`
- `test_nonboundary_majority_delta_recovers_pixels_exactly`
- `test_stc_syndrome_meets_shannon_bound_within_15_percent`
- `test_lossless_argmax_recovery_from_logits_input`
- `test_codec_is_deterministic_byte_for_byte`
- `test_empty_boundary_edge_case_roundtrips`
- `test_all_boundary_edge_case_roundtrips`
- `test_decode_rejects_bad_magic_or_truncated_stream`

Wire `src/tac/mask_codec.py` in Stage 2 only after tests pass:

- add `_STC_BOUNDARY = "stc_boundary"`;
- add it to `SUPPORTED_MASK_FORMATS`;
- route `encode_masks_auto(..., codec="stc_boundary")`;
- route `decode_masks_auto(..., codec="stc_boundary")`;
- update `detect_mask_codec` for the STC magic.

### Stage 3: archive builder and remote proof

Add `experiments/build_lane_stc_archive.py`:

- CLI reads a Lane A or Lane G anchor archive;
- extracts or decodes the existing mask payload;
- optionally loads precomputed logits if present;
- writes `masks.stcb` through `encode_mask_video_stc`;
- repackages a deterministic zip with renderer and pose payloads unchanged;
- verifies decode roundtrip before final archive write;
- prints original mask bytes, STC bytes, total archive bytes, and predicted
  rate delta.

Add `scripts/remote_lane_stc_boundary_coding.sh` with five stages:

1. clone/extract the tarball and verify code parity;
2. run the canonical remote setup (`scripts/remote_setup_full.sh`);
3. build the STC archive on the Lane A anchor;
4. run `inflate.sh` on the exact archive bytes;
5. run upstream `evaluate.py` on CUDA and print the result tagged
   `[contest-CUDA]`.

Remote script constraints:

- no `git reset --hard`;
- no invented flags in subprocess calls;
- no scorer import in the inflate-side decoder;
- archive bytes evaluated must be the exact bytes reported.

## 7. Risks and known unknowns

- **STC rate versus Shannon bound:** literature expectations are 5-15% overhead,
  but our boundary residual source is structured and empirical validation is
  required. The arithmetic-coded residual baseline is mandatory for comparison.
- **Boundary bitmap can dominate:** a raw bitmap is impossible. Bitmap RLE/XOR
  compression must be measured early. If it exceeds 15-25 KB, adapt the
  threshold or switch to deterministic decode-side boundary reconstruction from
  class IDs plus sparse corrections.
- **Boundary fraction drift:** motion-heavy frames may need more than 5%
  boundary pixels. Use configurable per-frame thresholds and add an adaptive
  threshold mode driven by byte budget.
- **Bitexact determinism:** AV1 was deterministic at the file level. STC must be
  deterministic at the bitstream level across platforms. Keep decode integer
  only; avoid float thresholding on inflate.
- **Lossless default:** a lossy boundary/non-boundary approximation might save
  more bytes, but it changes SegNet/PoseNet behavior and requires full
  `[contest-CUDA]` validation. Stage 2 default is exact class-ID recovery.
- **eval_roundtrip impact:** none. This is encoder-time mask packaging plus a
  deterministic decoder. No training path changes and no scorer at inflate.
- **Naming collision:** existing Lane SH arithmetic-qint work and the
  user-described Selfcomp half-frame lane both use "SH" in nearby notes. Stage 2
  implementation should spell out `stc_boundary` and avoid overloading "SH" in
  code names.

## 8. Promotion gate

Promote Lane STC only if all are true:

1. `decode_mask_video_stc(encode_mask_video_stc(masks))` is exact on the anchor
   masks.
2. STC boundary class stream is within 15% of the measured Shannon entropy.
3. Full mask payload is 120-140 KB or better on the target 1200-frame archive.
4. The final archive is smaller by at least 60 KB versus the AV1 monochrome
   anchor.
5. The exact archive bytes run through `inflate.sh` and upstream `evaluate.py`
   on CUDA with a `[contest-CUDA]` result.
