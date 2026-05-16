# SPDX-License-Identifier: MIT
"""Contest-compliant inflate runtime for v8 Path B (numpy + Pillow + pywavelets).

NO torch, NO scorer, NO learned weights at inflate time per CLAUDE.md
"Strict scorer rule" non-negotiable + Catalog #6. Per HNeRV parity L7
substrate_engineering exception: ~310 LOC budget (vs L4's default 100 LOC).

Design memo Section 10 specifies the inflate pipeline:
    1. Per-pair per-stream arith-decode (gray, cb, cr) × (frame_0, frame_1_residual)
       + cls subbands
    2. Inverse Wyner-Ziv: reconstruct frame_1 subbands from frame_0 + residual
    3. Inverse DB4 depth-2 DWT per channel per frame
    4. YUV->RGB (BT.601 inverse)
    5. Bilinear upscale to camera resolution (874, 1164)

Per Catalog #146 the contract is ``inflate.py <archive_dir> <output_dir> <file_list>``.
Per Catalog #205 we expose the canonical ``select_inflate_device()`` helper
even though we have no torch (it's a no-op that honors PACT_INFLATE_DEVICE).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .archive import WaveletResidualArchive, parse_archive
from .wavelet_codec import (
    NUM_SUBBANDS,
    QUANT_LEVELS,
    QUANT_ZERO_INDEX,
    SUBBAND_LABELS,
    decode_subband_arith,
    dequantize_subband,
    idwt2_db4_depth2,
)
from .wyner_ziv_temporal import reconstruct_frame1_from_frame0_and_residual


def select_inflate_device() -> str:
    """Catalog #205 canonical helper; v8 needs no torch.

    The substrate has no neural primitives, so cuda vs cpu is a no-op.
    Honor PACT_INFLATE_DEVICE env var per the canonical contract; refuse
    "mps" per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    # INLINE_DEVICE_FORK_OK:nscs06-v8-substrate-has-no-torch-no-cuda-cpu-distinction
    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()
    if pinned not in {"auto", "cpu", "cuda"}:
        raise SystemExit(
            f"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}"
        )
    return "cpu"


def _depth2_subband_shapes(h: int, w: int) -> list[tuple[int, int]]:
    """Per :data:`SUBBAND_LABELS` order, the canonical periodization-mode
    DB4 depth-2 subband shapes for an input of (h, w).

    Per pywt periodization convention: at each level, output size is
    ``ceil(input_size / 2)``. So depth-2 LL2 is at (h/4, w/4), depth-1
    detail bands at (h/2, w/2).
    """
    h2 = h // 4
    w2 = w // 4
    h1 = h // 2
    w1 = w // 2
    return [
        (h2, w2),  # LL2
        (h2, w2),  # LH2
        (h2, w2),  # HL2
        (h2, w2),  # HH2
        (h1, w1),  # LH1
        (h1, w1),  # HL1
        (h1, w1),  # HH1
    ]


def _yuv601_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """BT.601 inverse: Y,Cb,Cr (uint8) -> RGB (uint8).

    Per ITU-R BT.601 standard (the contest's reference per
    ``upstream/rgb_to_yuv6.py`` source convention).
    """
    y_f = y.astype(np.float64)
    cb_f = cb.astype(np.float64) - 128.0
    cr_f = cr.astype(np.float64) - 128.0
    r = y_f + 1.402 * cr_f
    g = y_f - 0.344136 * cb_f - 0.714136 * cr_f
    b = y_f + 1.772 * cb_f
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _upscale_to_camera(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear upscale via Pillow (canonical per v7 pattern)."""
    th, tw = target_hw
    img = Image.fromarray(rgb)
    img_up = img.resize((tw, th), resample=Image.BILINEAR)
    return np.array(img_up, dtype=np.uint8)


def _decode_one_channel_subbands(
    stream_bytes: bytes,
    cls_subbands: list[np.ndarray],
    priors,
    eval_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Decode all 7 subbands for one channel (gray or cb or cr) from one stream.

    The stream is a single arith-coded byte sequence with ALL subbands
    concatenated in :data:`SUBBAND_LABELS` order. We pass ``cls_subbands`` —
    per-subband class label arrays — to feed per-cell class context to the
    arith decoder.

    Per Catalog #220 operational mechanism: this function consumes the
    stream bytes to produce reconstructed coefficient arrays. Byte mutation
    here changes output frames.
    """
    if len(cls_subbands) != NUM_SUBBANDS:
        raise ValueError(f"cls_subbands length {len(cls_subbands)} != {NUM_SUBBANDS}")
    h, w = eval_hw
    expected_shapes = _depth2_subband_shapes(h, w)
    # The full stream encodes all 7 subbands' symbols sequentially. We decode
    # ALL symbols at once using the canonical ArithmeticCoder, but we feed
    # the per-subband per-cell class label so the decoder can pick the right
    # per-class CDF.
    out: list[np.ndarray] = []
    # Flatten class labels in the same order as encode: subband-by-subband,
    # raveled within each subband.
    flat_cls_list: list[np.ndarray] = [s.ravel() for s in cls_subbands]
    concat_cls = np.concatenate(flat_cls_list).astype(np.uint8)
    # Now decode with per-(global-symbol) CDF lookup keyed by subband_index.
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        ArithmeticCoder,
    )

    from .wavelet_codec import laplacian_cdf_uint16

    cdfs_per_subband_per_class: list[list[np.ndarray]] = []
    for s in range(NUM_SUBBANDS):
        cdfs_per_class: list[np.ndarray] = []
        from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
            NUM_SEGNET_CLASSES,
        )

        for c in range(NUM_SEGNET_CLASSES):
            scale = float(priors.scales[s, c])
            cdfs_per_class.append(laplacian_cdf_uint16(scale))
        cdfs_per_subband_per_class.append(cdfs_per_class)

    coder = ArithmeticCoder.from_bytes(stream_bytes)
    # Walk through each subband in order, decoding its full set of symbols.
    sym_idx = 0
    for s in range(NUM_SUBBANDS):
        sb_h, sb_w = expected_shapes[s]
        size_s = sb_h * sb_w
        out_flat = np.zeros(size_s, dtype=np.int8)
        cdfs = cdfs_per_subband_per_class[s]
        for i in range(size_s):
            c = int(concat_cls[sym_idx])
            symbol = coder.decode_symbol(cdfs[c])
            out_flat[i] = symbol - QUANT_ZERO_INDEX
            sym_idx += 1
        out.append(out_flat.reshape(sb_h, sb_w))
    return out


def _decode_class_label_subbands(
    stream_bytes: bytes,
    eval_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Decode per-subband class label arrays from a single arith stream.

    Class labels are coded with a UNIFORM per-position CDF (no class context
    yet — bootstrap). Sizes per subband come from periodization shape rule.
    """
    from tac.substrates.nscs06_carmack_hotz_strip_everything.codec import (
        ArithmeticCoder,
        CDF_MAX,
        NUM_SEGNET_CLASSES,
    )

    # Uniform CDF over NUM_SEGNET_CLASSES symbols (length NUM_SEGNET_CLASSES + 1).
    uniform_cdf = np.linspace(0, CDF_MAX, NUM_SEGNET_CLASSES + 1, dtype=np.int64)
    uniform_cdf[-1] = CDF_MAX
    uniform_cdf = uniform_cdf.astype(np.uint16)

    h, w = eval_hw
    expected_shapes = _depth2_subband_shapes(h, w)
    coder = ArithmeticCoder.from_bytes(stream_bytes)
    out: list[np.ndarray] = []
    for s in range(NUM_SUBBANDS):
        sb_h, sb_w = expected_shapes[s]
        flat = np.zeros(sb_h * sb_w, dtype=np.uint8)
        for i in range(flat.size):
            flat[i] = coder.decode_symbol(uniform_cdf)
        out.append(flat.reshape(sb_h, sb_w))
    return out


def inflate_one_video(archive_bytes: bytes, output_stem: Path) -> Path:
    """Inflate v8 Path B archive to a raw RGB byte stream.

    Args:
        archive_bytes: bytes of ``0.bin`` (WLV2 grammar).
        output_stem: target path; ``.raw`` suffix is appended.

    Returns:
        Path to the written ``.raw`` file.

    The raw file format is the contest convention: row-major RGB bytes at
    ``OUTPUT_HEIGHT × OUTPUT_WIDTH × 3`` per frame, with frame_0 then frame_1
    for each of NUM_PAIRS pairs.
    """
    arc: WaveletResidualArchive = parse_archive(archive_bytes)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_stem.with_suffix(".raw")
    eval_h, eval_w = arc.eval_height, arc.eval_width
    out_h, out_w = arc.output_height, arc.output_width

    # 0. Decode class label subbands (shared across the 6 wavelet streams)
    cls_subbands = _decode_class_label_subbands(arc.cls_bytes, (eval_h, eval_w))

    # 1. Decode 6 wavelet streams: (gray, cb, cr) × (f0, f1_residual)
    def _dec(stream: bytes) -> list[np.ndarray]:
        return _decode_one_channel_subbands(
            stream, cls_subbands, arc.priors, (eval_h, eval_w)
        )

    # For v1 SCAFFOLD, all 6 channel streams may be empty (smoke / dry-run)
    # in which case we synthesize ZERO subbands at expected sizes. This gives
    # the inflate path a non-throwing degenerate output for the smoke gate.
    expected_shapes = _depth2_subband_shapes(eval_h, eval_w)

    def _zero_subbands() -> list[np.ndarray]:
        return [np.zeros(sh, dtype=np.int8) for sh in expected_shapes]

    def _dec_or_zero(stream: bytes) -> list[np.ndarray]:
        if len(stream) == 0:
            return _zero_subbands()
        return _dec(stream)

    gray_f0_subbands_q = _dec_or_zero(arc.gray_f0_bytes)
    gray_f1res_subbands_q = _dec_or_zero(arc.gray_f1res_bytes)
    cb_f0_subbands_q = _dec_or_zero(arc.cb_f0_bytes)
    cb_f1res_subbands_q = _dec_or_zero(arc.cb_f1res_bytes)
    cr_f0_subbands_q = _dec_or_zero(arc.cr_f0_bytes)
    cr_f1res_subbands_q = _dec_or_zero(arc.cr_f1res_bytes)

    # 2. Inverse Wyner-Ziv: reconstruct frame_1 subbands
    gray_f1_subbands_q = reconstruct_frame1_from_frame0_and_residual(
        gray_f0_subbands_q, gray_f1res_subbands_q
    )
    cb_f1_subbands_q = reconstruct_frame1_from_frame0_and_residual(
        cb_f0_subbands_q, cb_f1res_subbands_q
    )
    cr_f1_subbands_q = reconstruct_frame1_from_frame0_and_residual(
        cr_f0_subbands_q, cr_f1res_subbands_q
    )

    # 3. Inverse DB4 depth-2 DWT per channel per frame
    def _idwt_channel(qs: list[np.ndarray]) -> np.ndarray:
        # Dequantize per-subband, then idwt
        dq = [dequantize_subband(qs[s], arc.quant_steps[s]) for s in range(NUM_SUBBANDS)]
        rec = idwt2_db4_depth2(dq, (eval_h, eval_w))
        return np.clip(rec, 0, 255).astype(np.uint8)

    gray_f0 = _idwt_channel(gray_f0_subbands_q)
    gray_f1 = _idwt_channel(gray_f1_subbands_q)
    cb_f0 = _idwt_channel(cb_f0_subbands_q)
    cb_f1 = _idwt_channel(cb_f1_subbands_q)
    cr_f0 = _idwt_channel(cr_f0_subbands_q)
    cr_f1 = _idwt_channel(cr_f1_subbands_q)

    # NOTE: the inflate path above produces FRAME 0 and FRAME 1 reconstruction
    # that is SHARED across all pairs in this L1 SCAFFOLD; the per-pair offset
    # logic + per-pair stream slicing is part of the full integration path
    # (deferred to L2 per the design memo Section 14 reactivation criteria).
    # For the L1 smoke + Catalog #220 byte-mutation proof, the shared
    # reconstruction is sufficient — every wavelet stream byte IS consumed
    # and DOES affect the rendered frame pixels.

    rgb_f0 = _yuv601_to_rgb(gray_f0, cb_f0, cr_f0)
    rgb_f1 = _yuv601_to_rgb(gray_f1, cb_f1, cr_f1)

    rgb_f0_full = _upscale_to_camera(rgb_f0, (out_h, out_w))
    rgb_f1_full = _upscale_to_camera(rgb_f1, (out_h, out_w))

    with raw_path.open("wb") as fh:
        for _ in range(arc.num_pairs):
            fh.write(rgb_f0_full.tobytes())
            fh.write(rgb_f1_full.tobytes())
    return raw_path


def main_cli() -> int:
    """CLI per Catalog #146: ``inflate.py <archive_dir> <output_dir> <file_list>``."""
    if len(sys.argv) != 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    select_inflate_device()
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0]
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
