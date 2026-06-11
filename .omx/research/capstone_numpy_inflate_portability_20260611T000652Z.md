# Capstone VQ-NeRV — numpy-reference portability port + contest inflate runtime

**UTC:** 2026-06-11T00:06:52Z · **Authority:** `[macOS-MLX research-signal]` / `[macOS-CPU advisory]`
(NON-PROMOTABLE; a contest score needs `upstream/evaluate.py` paired CUDA + Linux-x86_64 CPU).
**Lane:** capstone VQ-NeRV (Task #78). **Mission:** `frontier_breaking_enabler` (unblocks the capstone's
contest exact eval — the only thing that moves the frontier pointer).

## What was built (the MLX-FIRST portability contract: MLX → numpy reference → torch)

Three files, pure-numpy (no MLX, no torch in the decode path), reproducing
`CapstoneVqNervBundle._decode_with_film` (the per-frame-FiLM crux fix, 2026-06-10):

1. **`src/tac/capstone_vq_nerv/numpy_reference.py`** — op-for-op numpy port of the decoder forward,
   parameterized over `base_channels` (reads the PR95 channel taper + weights from the parsed archive;
   works for base_ch 16/20/24/36 unchanged). Ports: stem (Linear `x@W.T+b`) → reshape → NHWC → `sin` →
   6 upsample blocks (canonical channel-FIRST PixelShuffle(2) + closed-form 2x bilinear align_corners=False
   skip + 1x1 skip_conv on channel change + `sin(decoded+identity)`) → refine0 (3x3 pad2 dil2) → refine1
   (3x3 pad1) → `feat = x + 0.1*sin(refined)` → **PER-FRAME FiLM** (`pose_film0`/`pose_film1`,
   `gamma=1+tanh(fc2(sin(fc1(norm_pose)))[:, :C])`, `beta=...[:, C:]`, modulating `feat` DIFFERENTLY before
   each rgb head) → `sigmoid(rgb_k(feat_k))*255` → N2CHW. Plus a general bilinear resize for the camera
   upsample, and helpers to extract the FULL render basis (decoder + FiLM) from a live bundle.

2. **`src/tac/capstone_vq_nerv/inflate.py`** (core decode ≤ ~70 LOC, 2 deps: numpy + brotli) — parses the
   archive (`parse_capstone_archive_bytes` + `_decode_int8_brotli` reused, not reimplemented), decodes the
   render basis + codebook + bit-packed VQ indices (`z_q = codebook[index]`) + stored fp16 pose + config
   sidecar, renders every pair via `numpy_decode_pair`, bilinear-upsamples each 384×512 frame to camera
   1164×874, round/clamp/uint8, and writes the flat raw tensor `TensorVideoDataset` reads:
   `(N, 874, 1164, 3)` uint8, C-contiguous, pair k → frames 2k (frame0/rgb_0), 2k+1 (frame1/rgb_1).
   CLI `inflate.py <archive> <dst_raw>` (anr/PR95 convention). NO scorer loaded (Strict scorer rule).

3. **`src/tac/capstone_vq_nerv/runtime/inflate.sh`** — contest runtime stub
   (`inflate.sh DATA_DIR OUTPUT_DIR FILE_LIST` → `OUTPUT_DIR/<base>.raw`), pure-numpy, CPU/CUDA-agnostic.
   Verified end-to-end: `archive.zip` (member `x` + `capstone_config_v1` sidecar) → `0.raw` =
   `(4, 874, 1164, 3)` uint8 — exactly the `<submission_dir>/inflated/0.raw` shape evaluate.py reads.

4. **`src/tac/capstone_vq_nerv/tests/test_numpy_reference_parity.py`** — the GATE (7 tests, all pass):
   render-closeness sweep (base_ch 16/20/24), the score-parity gate (frozen proto DistortionNet d_seg/d_pose
   on numpy-inflate vs MLX-render), identity-FiLM control, stub-fails-pixel-parity NO-FAKE control, and the
   zip+config-sidecar packaging round-trip.

## Parity result (the numbers — did the numpy port achieve score-parity? YES)

Frozen proto DistortionNet (SegNet argmax + PoseNet), numpy inflate vs MLX render, 5 pairs:

| metric | MLX render | numpy inflate | |Δ| | note |
|---|---|---|---|---|
| **d_seg** | 0.80000001 | 0.80000001 | **0.0 (EXACT)** | bit-identical SegNet argmax |
| **d_pose** | 3293.0972 | 3293.2603 | 0.163 (rel 5e-5) | dominated by fp16 pose-storage |

Direct render drift (max\|Δpixel\| on [0,255]): base_ch 16→0.022, 20→0.013, 24→0.009, 36→0.012.
Camera uint8 frames (faithful inflate vs MLX-render torch-reference camera): **mean |Δ| = 0.005, max = 1**
(essentially bit-exact uint8 camera frames).

## The one documented residual (honest, NOT score-affecting)

The d_pose |Δ| of 0.163 is **the fp16 pose-storage roundtrip**, not a port defect. Apples-to-apples
control (MLX render fed the SAME fp16-roundtripped pose the archive stores): d_seg |Δ| = 0.0, d_pose
|Δ| = 0.026 on ~3193 (rel 8e-6). The residual decomposes as:
- **fp16 pose storage** (~0.14 of the d_pose gap): the archive carries pose as fp16 — this is a real,
  intentional part of the inflate (the bytes the contest decodes), so the inflate is FAITHFUL to the
  archive, and the MLX-render-with-fp32-pose is the one that's "ahead" of the bytes.
- **fp16 weight storage + conv accumulation order** (~0.026 residual): sub-uint8, never flips a SegNet
  argmax (d_seg parity is EXACT), pose-relative 8e-6. Not score-affecting.

Verdict (Catalog #307 IMPLEMENTATION level): the numpy port is the faithful numeric reference; the MLX
path drifts from it only by the small fp32 accumulation-order delta of `mx.conv2d`/MLX matmul, which is
argmax-invariant and pose-negligible. The portability contract HOLDS.

## Inflate runtime: contest-ready or blocked?

**Contest-ready for the archive grammar.** The runtime parses `archive.zip` → decodes → renders →
writes evaluate.py-readable `(N,874,1164,3)` uint8 frames; verified end-to-end via both the CLI and
`inflate.sh`. NO scorer at inflate (Strict scorer rule). Pure-numpy, CPU/CUDA-agnostic, deterministic.

**One archive-grammar gap surfaced + closed by this work (NO-FAKE).** The existing
`build_capstone_archive_bytes` test path archived only `bundle.decoder.parameters()` — it DROPPED the
per-frame FiLM weights and the pose-normalization stats, which the render depends on. A decoder-only
archive is insufficient for a FiLM-enabled bundle. This port closes that by:
(a) `full_render_weights_from_bundle` packing decoder + `pose_film0.*`/`pose_film1.*` into the (name-keyed)
decoder section; (b) a `capstone_config_v1` sidecar carrying `base_channels`/pose-stats/`film_enabled`/
`decoder_dtype`. The remaining work to reach an EXACT-EVAL row is upstream of this port: a real
600-pair/384×512 trained bundle (the MLX viability daemon, pid 26696) + the byte-closed archive built with
the FULL render basis + paired CPU+CUDA `upstream/evaluate.py`. The inflate runtime itself is not the
blocker.

## 6-hook wire-in (Catalog #125)

- #1 sensitivity-map: N/A (portability port; no per-axis byte allocation).
- #2 Pareto: N/A (no new byte/score tradeoff; the archive grammar is unchanged in section count).
- #3 bit-allocator: N/A.
- #4 cathedral autopilot: N/A (research-signal port; not archive-deployable as a score-mover by itself).
- #5 continual-learning posterior: N/A (no empirical contest anchor; advisory parity only).
- #6 probe-disambiguator: **ACTIVE** — the parity test IS the MLX↔numpy disambiguator (decides whether
  the two render paths agree; the documented fp16-pose residual is the regime-conditional verdict).

## Files + verification

- `src/tac/capstone_vq_nerv/numpy_reference.py` (new) · `inflate.py` (new) · `runtime/inflate.sh` (new) ·
  `tests/test_numpy_reference_parity.py` (new, 7 tests).
- `ruff check` clean on all 3 .py files. `pytest tests/test_numpy_reference_parity.py` 7 passed.
  Full capstone suite: 29 passed (1 pre-existing `@pytest.mark.slow` real-scorer test times out at the
  60s pytest-timeout — unrelated to this port; it trains against the real contest scorer).
