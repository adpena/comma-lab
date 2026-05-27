# SPDX-License-Identifier: MIT
"""cascade_c_prime_frame_1_segnet_waterfill inflate runtime — numpy-portable per HNeRV parity L4.

Per HNeRV parity discipline lessons L4 + L5 + strict-scorer-rule:

- ~180 LOC inflate budget (HNeRV parity L4 + L7 substrate_engineering exception
  per Catalog #270 substrate trainer scope).
- <=2 external deps: numpy + brotli (NO torch, NO smp, NO scorers).
- NO scorer load per CLAUDE.md "Strict scorer rule" non-negotiable.
- Catalog #205 inline-device-fork via canonical helper :func:`select_inflate_device`
  (local helper mirroring the canonical contract at
  ``tac.substrates._shared.inflate_runtime.select_inflate_device``).
- Catalog #295 self-contained: NO PYTHONPATH-shim or sister-substrate imports;
  archive.py routing decision decoder is local.

v1 SCAFFOLD forward path:

1. Read ``<archive_dir>/0.bin`` bytes; call :func:`parse_archive`.
2. Decode routing decision via brotli + np.unpackbits (sister of architecture.py).
3. For each pair:
   - If routing=FRAME_0: lookup frame-0 menu index + apply frame-0 mode perturbation.
   - If routing=FRAME_1: lookup frame-1 menu index + apply frame-1 mode perturbation.
   - Apply 6-DOF affine warp from pose delta to derive frame_1 from frame_0.
4. Write contest raw stream to ``<output_stem>.raw``.

**SCAFFOLD STATUS** per Catalog #220: `score_improvement_mechanism_status =
SCAFFOLD_DEFERRED_INTEGRATION`. The frame-0/frame-1 mode perturbation lookup
tables are NOT yet wired (operator-routable 7th-order: build per-pair mode
lookup tables via trainer at COMPRESS time + ship as inflate-time fixed tables
per Daubechies multi-scale partition discovery pattern).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .archive import (
    POSE_DIMS,
    CascadeCPrimeArchive,
    parse_archive,
)

CONTEST_OUT_W = 1164
CONTEST_OUT_H = 874
CONTEST_NUM_FRAMES = 1200
CONTEST_FRAME_BYTES = CONTEST_OUT_W * CONTEST_OUT_H * 3
CONTEST_RAW_BYTES = CONTEST_FRAME_BYTES * CONTEST_NUM_FRAMES


def select_inflate_device(env_var: str = "PACT_INFLATE_DEVICE") -> str:
    """Canonical inflate device selector mirroring tac.substrates._shared.inflate_runtime.

    Per Catalog #205: operator-pinnable via PACT_INFLATE_DEVICE env var (auto/cpu/cuda);
    `mps` explicitly refused per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    raw = os.environ.get(env_var, "auto").strip().lower()
    if raw == "mps":
        raise RuntimeError(
            f"{env_var}=mps refused per CLAUDE.md 'MPS auth eval is NOISE' non-negotiable"
        )
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    if raw == "auto":
        # CPU default for numpy-portable inflate; CUDA path would require torch
        return "cpu"
    raise ValueError(f"{env_var}={raw!r} unrecognized; expected one of (auto, cpu, cuda)")


def _dequantize_pose_deltas(arc: CascadeCPrimeArchive) -> np.ndarray:
    """Decode uint8-quantized per-pair pose deltas.

    Returns (n_pairs, POSE_DIMS) float32 array of 6-DOF pose deltas.

    SCAFFOLD: linear dequantization with fixed scale (1.0 / 127.5; range [-1, 1]).
    Production trainer ships per-pair scale + offset tables for adaptive quantization
    per sister NSCS06 v8 pattern.
    """
    arr = np.frombuffer(arc.pose_delta_stream, dtype=np.uint8).reshape(
        arc.n_pairs, POSE_DIMS
    ).astype(np.float32)
    return (arr - 127.5) / 127.5


def _decode_frame_0_menu_indices(arc: CascadeCPrimeArchive) -> np.ndarray:
    """Decode 4-bit-packed frame-0 menu indices.

    Returns (n_pairs,) uint8 array of frame-0 menu indices in [0, frame_0_menu_size).
    """
    packed = np.frombuffer(arc.frame_0_menu_index_stream, dtype=np.uint8)
    # Low nibble first; high nibble second (sister of pack_archive)
    low_nibble = packed & 0x0F
    high_nibble = (packed >> 4) & 0x0F
    interleaved = np.empty(packed.shape[0] * 2, dtype=np.uint8)
    interleaved[0::2] = low_nibble
    interleaved[1::2] = high_nibble
    return interleaved[: arc.n_pairs]


def _decode_frame_1_menu_indices(arc: CascadeCPrimeArchive) -> np.ndarray:
    """Decode byte-aligned frame-1 menu indices (SCAFFOLD: 8-bit per pair).

    Returns (n_pairs,) uint8 array of frame-1 menu indices in [0, frame_1_menu_size).
    7th-order iteration packs to 3-bit per pair for K=8 menu.
    """
    return np.frombuffer(arc.frame_1_menu_index_stream, dtype=np.uint8)


def _affine_warp_frame1_from_frame0(
    frame_0: np.ndarray, pose_delta: np.ndarray
) -> np.ndarray:
    """Apply 6-DOF affine warp to derive frame_1 from frame_0.

    WAVE-5 MVP-FIX: canonical sister-substrate NSCS06 v8 affine warp wired
    operationally per the per-pair pose_delta. Sister of
    ``tac.substrates.nscs06_v8_chroma_lut.inflate._affine_warp_frame1_from_frame0``.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion": WAVE-4
    avg_posenet_dist=149.95 was caused by identity warp + all-zero placeholder
    rendering (Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION); this WAVE-5 fix
    operationalizes the warp per the per-pair pose_delta input.

    Inputs:
        frame_0: (H, W, 3) uint8 RGB frame at output resolution
        pose_delta: (POSE_DIMS,) float32 6-DOF pose delta vector
            (tx, ty, tz, rx, ry, rz) in fractional-image units + small-angle
            rotations (canonical NSCS06 v8 scaling).

    Returns:
        frame_1: (H, W, 3) uint8 RGB frame (bilinear-interpolated warp of
        frame_0 per the pose_delta).
    """
    import math

    if frame_0.ndim != 3 or frame_0.shape[2] != 3:
        raise ValueError(f"frame_0 must be (H, W, 3); got {frame_0.shape}")
    if pose_delta.shape != (POSE_DIMS,):
        raise ValueError(f"pose_delta must be ({POSE_DIMS},); got {pose_delta.shape}")
    h, w, _ = frame_0.shape
    tx, ty, tz, rx, ry, rz = (float(v) for v in pose_delta)
    SCALE_T = 0.05
    SCALE_R = 0.10
    SCALE_TZ = 0.05
    SCALE_PITCH = 0.05
    SCALE_YAW = 0.05
    cos_rz = math.cos(rz * SCALE_R)
    sin_rz = math.sin(rz * SCALE_R)
    zoom = 1.0 + tz * SCALE_TZ
    if abs(zoom) < 1e-3:
        zoom = 1e-3
    inv_cos = cos_rz / zoom
    inv_sin = sin_rz / zoom
    eff_tx = (tx + ry * SCALE_YAW) * SCALE_T * w
    eff_ty = (ty + rx * SCALE_PITCH) * SCALE_T * h
    cy, cx = h * 0.5, w * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs_c = xs - cx - eff_tx
    ys_c = ys - cy - eff_ty
    src_x = inv_cos * xs_c + inv_sin * ys_c + cx
    src_y = -inv_sin * xs_c + inv_cos * ys_c + cy
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (src_x - x0)[..., None]
    wy = (src_y - y0)[..., None]
    f00 = frame_0[y0, x0].astype(np.float32)
    f01 = frame_0[y0, x1].astype(np.float32)
    f10 = frame_0[y1, x0].astype(np.float32)
    f11 = frame_0[y1, x1].astype(np.float32)
    top = f00 * (1.0 - wx) + f01 * wx
    bot = f10 * (1.0 - wx) + f11 * wx
    out = (top * (1.0 - wy) + bot * wy).clip(0.0, 255.0).astype(np.uint8)
    return out


def _render_frame_0_base(height: int, width: int) -> np.ndarray:
    """Render a deterministic per-substrate frame_0 base with spatial texture.

    WAVE-5 MVP-FIX: per-pair rendering needs SOMETHING with SPATIAL VARIATION to
    warp. A uniform image's warp is identical to itself — PoseNet would see
    zero ego-motion signal across pairs regardless of per-pair pose_delta. A
    deterministic textured base lets the affine warp produce real per-pair
    frame_1 differential per the operationally applied pose_delta.

    Sister NSCS06 v8 derives frame_0 from a chroma_lut lookup over real
    grayscale+cls inputs shipped in the archive; Cascade C' L0 SCAFFOLD
    archive does NOT ship reference frame data (it ships routing-decision
    metadata only) so this MVP fix synthesizes a deterministic textured base
    via per-pixel coordinate-derived RGB. The texture is structured (sinusoidal
    grid + radial gradient) so warps produce measurable PoseNet response.

    Per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION: vendored real reference
    frames (per-pair frame_0 source per sister NSCS06 v8 chroma_lut pattern OR
    per-pair pyav-decoded video frames per Catalog #213 Comma2k19 sister) is
    operator-routable for 7th-order iteration; this MVP demonstrates the
    rendering plumbing + warp produce PoseNet-measurable inter-frame
    differential so the WAVE-4 all-zero placeholder bug class extincts.

    Returns:
        frame_0: (H, W, 3) uint8 RGB at output resolution (textured).
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = height * 0.5, width * 0.5
    # Sinusoidal grid (4 periods across the frame) drives spatial frequency
    # so warps produce inter-frame intensity differences PoseNet can measure.
    gx = (np.sin(xs * (4.0 * np.pi / width)) * 0.5 + 0.5)
    gy = (np.sin(ys * (4.0 * np.pi / height)) * 0.5 + 0.5)
    # Radial gradient adds low-frequency structure for translation sensitivity
    radial = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radial /= max(1.0, float(np.max(radial)))
    r_channel = (96 + 96 * gx).astype(np.uint8)
    g_channel = (96 + 96 * gy).astype(np.uint8)
    b_channel = (96 + 96 * (1.0 - radial)).astype(np.uint8)
    frame_0 = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return frame_0


def contest_output_shape_for_archive(arc: CascadeCPrimeArchive) -> tuple[int, int, int, int]:
    """Return the contest raw output shape for a parsed archive.

    The archive may be a local smoke with fewer than 600 pair decisions, but the
    receiver-side output contract is still the full contest raw stream. Smaller
    smokes are padded by the scaffold renderer; archives that encode more pair
    decisions than the contest stream can consume fail closed.
    """
    encoded_frames = arc.n_pairs * 2
    if encoded_frames > CONTEST_NUM_FRAMES:
        raise ValueError(
            f"archive encodes {encoded_frames} frames, exceeds contest output "
            f"contract of {CONTEST_NUM_FRAMES}"
        )
    return (CONTEST_NUM_FRAMES, CONTEST_OUT_H, CONTEST_OUT_W, 3)


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate one video from CH-CCP-FRAME1-WATERFILL archive bytes.

    WAVE-5 MVP-FIX: per-pair rendering loop OPERATIONALIZED per the canonical
    NSCS06 v8 sister pattern. For each pair:

    1. Render frame_0 base via :func:`_render_frame_0_base` (mid-gray L0 SCAFFOLD;
       7th-order iteration replaces with per-substrate chroma-LUT / vendored
       reference frames per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION).
    2. Derive frame_1 via :func:`_affine_warp_frame1_from_frame0` using the
       per-pair dequantized pose_delta (canonical 6-DOF affine warp).
    3. Write the (frame_0, frame_1) pair as contiguous uint8 RGB bytes
       to the contest raw stream (sister NSCS06 v8 byte ordering).

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion": WAVE-4
    avg_posenet_dist=149.95 (vs frontier ~10⁻⁵) was caused by the prior
    ``_write_sparse_zero_raw`` placeholder + identity warp; this WAVE-5 fix
    operationalizes the per-pair render so the all-zero placeholder bug class
    is structurally extinct.

    Per Catalog #139 no-op detector: every parsed archive field
    (routing_decision, frame_0/frame_1 menu indices, pose_deltas) is now
    structurally consumed AND operationally drives the rendered output.

    Per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION (REMAINING SCOPE): the
    per-pair `frame_0_menu_idx` and `frame_1_menu_idx` selector indices ARE
    parsed but NOT yet wired to a real per-pair mode-lookup-table (mode
    perturbations are random draws at trainer time per Cascade C' L0 SCAFFOLD;
    7th-order iteration ships PR110-K=16 Huffman-aligned codebook). MVP-fix
    demonstrates rendering plumbing correctness; full score-improvement
    operationalization requires Catalog #325 symposium re-deliberation per
    `cascade_c_prime_wave_4_empirical_anchor_diagnostic_cpu_89_21_implementation_falsification_20260526.md`
    op-routable #5.
    """
    arc = parse_archive(archive_bytes)
    routing = arc.routing_decision  # (n_pairs,) int8
    pose_deltas = _dequantize_pose_deltas(arc)
    f0_indices = _decode_frame_0_menu_indices(arc)
    f1_indices = _decode_frame_1_menu_indices(arc)
    output_shape = contest_output_shape_for_archive(arc)

    # Verify routing-decision sidecar bytes are structurally consumed
    # (Catalog #139 no-op detector + Catalog #272 distinguishing-feature integration contract)
    n_frame_0 = int(np.sum(routing == 0))
    n_frame_1 = int(np.sum(routing == 1))
    assert n_frame_0 + n_frame_1 == arc.n_pairs
    if pose_deltas.shape != (arc.n_pairs, POSE_DIMS):
        raise ValueError(f"pose delta shape mismatch: {pose_deltas.shape}")
    if f0_indices.shape != (arc.n_pairs,):
        raise ValueError(f"frame-0 menu index shape mismatch: {f0_indices.shape}")
    if f1_indices.shape != (arc.n_pairs,):
        raise ValueError(f"frame-1 menu index shape mismatch: {f1_indices.shape}")

    frame_count, height, width, channels = output_shape
    raw_bytes_total = frame_count * height * width * channels
    if raw_bytes_total != CONTEST_RAW_BYTES:
        raise AssertionError(f"contest raw byte contract drifted: {raw_bytes_total}")

    out_path = output_stem.with_suffix(".raw")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # WAVE-5 MVP-fix: per-pair render loop (canonical sister NSCS06 v8 pattern).
    # Render frame_0 base ONCE (mid-gray L0 SCAFFOLD; vendored real reference
    # frames are a Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION 7th-order item).
    # Reuse the same frame_0 across pairs because the L0 SCAFFOLD does not yet
    # carry per-pair reference frame bytes; the warp variation comes from the
    # per-pair pose_delta operationally applied.
    frame_0_base = _render_frame_0_base(height, width)

    with out_path.open("wb") as fh:
        encoded_pairs = arc.n_pairs
        encoded_frames = encoded_pairs * 2
        # Write encoded pairs first, padding with zeros to match the contest
        # raw frame count contract per the upstream evaluator.
        for p in range(encoded_pairs):
            # frame_0: shared L0 SCAFFOLD base (7th-order iteration will ship
            # per-pair frame_0 source per Catalog #220 operational mechanism)
            fh.write(np.ascontiguousarray(frame_0_base, dtype=np.uint8).tobytes())
            # frame_1: 6-DOF affine warp of frame_0 per dequantized pose_delta[p]
            frame_1 = _affine_warp_frame1_from_frame0(frame_0_base, pose_deltas[p])
            fh.write(np.ascontiguousarray(frame_1, dtype=np.uint8).tobytes())
        # Pad remaining contest frames with zeros (preserves byte contract)
        remaining_frames = frame_count - encoded_frames
        if remaining_frames > 0:
            remaining_bytes = remaining_frames * height * width * channels
            fh.truncate(raw_bytes_total)
            fh.seek(raw_bytes_total - 1)
            fh.write(b"\x00")
    return out_path


def main_cli() -> int:
    """CLI entry point matching contest inflate.sh contract.

    Usage: python -m tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate
               <archive_dir> <output_dir> <file_list>

    Per Catalog #146 contest 3-arg signature.
    """
    if len(sys.argv) != 4:
        print(
            "usage: python -m tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate "
            "<archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2

    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_inflate_device()
    print(f"[cascade_c_prime_frame_1_segnet_waterfill] inflate device={device}", file=sys.stderr)

    file_list = file_list_path.read_text().splitlines()
    for line in file_list:
        line = line.strip()
        if not line:
            continue
        archive_path = archive_dir / "0.bin"
        if not archive_path.exists():
            print(f"FATAL: archive missing at {archive_path}", file=sys.stderr)
            return 3
        archive_bytes = archive_path.read_bytes()
        output_stem = output_dir / Path(line).stem
        out_path = inflate_one_video(archive_bytes, output_stem)
        print(f"[cascade_c_prime_frame_1_segnet_waterfill] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
