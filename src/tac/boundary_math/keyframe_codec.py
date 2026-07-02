# SPDX-License-Identifier: MIT
"""KEYFRAME CODEC PRIMITIVES — rate minimization for the warp-real-luma pose carrier.

The warp-real-luma frame0 pose carrier (``tac.boundary_math.warp_real_luma_frame0``)
warps a STORED REAL keyframe by the per-pair ego-twist to synthesize frame0 for
PoseNet. Those keyframe bytes are VIDEO-DERIVED -> COUNTED in ``archive.zip`` (the
decoder libs are FREE per rule-118, like brotli). The keyframe payload DWARFS the
twist payload (rate ~0.03-0.07 for 13 native-ish keyframes vs ~0.0016 for the
twist) and is the #202 byte-close rate crux this module attacks.

This module is the PURE (numpy / cv2 / PIL / scipy / codec-lib) primitive layer
shared by the measurement tools. It carries NO torch / NO mlx (the d_pose authority
lives in the tools that import ``cpu_verdict_d_pose_batch``), so it is cleanly
unit-testable and reusable by the byte-close path.

Primitive families
------------------
* DEGRADATION ops (native uint8 -> native uint8, "degrade then restore to native"
  so the warp+scorer see native geometry at reduced INFORMATION): resolution
  roundtrip, Gaussian blur, block/global DCT truncation, bit-depth quantize.
* ENTROPY / CODEC BYTES: order-0 Shannon entropy (the ideal arithmetic-coder rate)
  + real compressor bytes (zlib / brotli / PNG / WebP / JPEG2000) + temporal
  HEVC/VP9/FFV1 mini-video bytes (ffmpeg).
* EGO-WARP PREDICTOR (rule-118 lever R1): fit the inter-keyframe homography
  (dense ECC or translation/affine ladder), warp the previous keyframe forward,
  and code only the RESIDUAL. The motion is exact/free at decode (stored ego-twist);
  the fitted homography params here upper-bound the achievable residual and cost
  ~64 B/keyframe (counted, negligible).

Authority: every BYTE number here is a real codec on real frames; the d_pose
COUPLING (does a residual/degradation preserve PoseNet?) is measured by the
importing tool through the frozen CPU-torch PoseNet (NEVER MPS). ``[macOS-CPU
advisory / research-signal]`` only; pointer 0.19110 UNMOVED; score_claim=False.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
_SCRATCH = REPO / ".omx" / "tmp" / "keyframe_codec"


class KeyframeCodecError(RuntimeError):
    """A keyframe-codec contract violation (shape / dtype / codec unavailable)."""


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def _as_u8_hwc(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(np.round(a), 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise KeyframeCodecError(f"expected (H,W,3) uint8, got {a.shape} {a.dtype}")
    return a


def rgb_to_luma(img_u8: np.ndarray) -> np.ndarray:
    """BT.601 luma (uint8 HxW). PoseNet's YUV6 luma is the pose-relevant channel."""
    a = _as_u8_hwc(img_u8).astype(np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# DEGRADATION ops (native -> native; reduce INFORMATION, keep geometry).
# --------------------------------------------------------------------------- #
def resize_roundtrip(img_u8: np.ndarray, out_w: int, out_h: int, *, interp: str = "area") -> np.ndarray:
    """Downsample to ``(out_h,out_w)`` then upsample back to native (info bottleneck)."""
    import cv2

    a = _as_u8_hwc(img_u8)
    H, W = a.shape[:2]
    down_i = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}[interp]
    small = cv2.resize(a, (int(out_w), int(out_h)), interpolation=down_i)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def downsample_only(img_u8: np.ndarray, out_w: int, out_h: int, *, interp: str = "area") -> np.ndarray:
    """Downsample to ``(out_h,out_w)`` uint8 (the actually-stored low-res keyframe)."""
    import cv2

    a = _as_u8_hwc(img_u8)
    down_i = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR}[interp]
    return cv2.resize(a, (int(out_w), int(out_h)), interpolation=down_i)


def gaussian_blur(img_u8: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass (native res, reduced spatial-frequency content)."""
    import cv2

    a = _as_u8_hwc(img_u8)
    if sigma <= 0:
        return a.copy()
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(a, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def bitdepth_quantize(img_u8: np.ndarray, bits: int) -> np.ndarray:
    """Uniform per-channel requantization to ``bits`` bits (8 = identity)."""
    a = _as_u8_hwc(img_u8)
    b = int(bits)
    if b >= 8:
        return a.copy()
    levels = 1 << b
    step = 256.0 / levels
    q = np.floor(a.astype(np.float32) / step)
    recon = np.clip((q + 0.5) * step, 0, 255)
    return recon.astype(np.uint8)


def dct_truncate(img_u8: np.ndarray, keep_h: int, keep_w: int) -> np.ndarray:
    """Global 2D DCT per channel; keep only the top-left ``keep_h x keep_w`` low-freq
    coefficients (zero the rest); inverse. Measures the low-frequency SUFFICIENCY:
    ``keep_h*keep_w`` coefficients/channel is the compressed statistic dimensionality.
    """
    from scipy.fft import dctn, idctn

    a = _as_u8_hwc(img_u8).astype(np.float32)
    H, W, C = a.shape
    kh = max(1, min(int(keep_h), H))
    kw = max(1, min(int(keep_w), W))
    out = np.empty_like(a)
    for c in range(C):
        coeff = dctn(a[..., c], norm="ortho")
        mask = np.zeros_like(coeff)
        mask[:kh, :kw] = coeff[:kh, :kw]
        out[..., c] = idctn(mask, norm="ortho")
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# ENTROPY / codec bytes.
# --------------------------------------------------------------------------- #
def order0_entropy_bits_per_symbol(arr: np.ndarray) -> float:
    """Order-0 Shannon entropy H = -sum p log2 p (bits/symbol) over flattened values.

    The ideal (memoryless) arithmetic-coder rate. Real coders (zlib/brotli) beat or
    trail this depending on higher-order structure; both are reported by callers.
    """
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return 0.0
    _, counts = np.unique(a, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def order0_entropy_bytes(arr: np.ndarray) -> float:
    """Ideal order-0 arithmetic-coder byte count for ``arr`` (H * N / 8)."""
    return order0_entropy_bits_per_symbol(arr) * float(np.asarray(arr).size) / 8.0


def residual_wraparound_u8(target_u8: np.ndarray, pred_u8: np.ndarray) -> np.ndarray:
    """Exact-invertible P-frame residual: ``(target - pred) mod 256`` as uint8.

    Reconstruction ``(pred + residual) mod 256 == target`` is exact (the standard
    lossless predictive-coding trick). Concentrated near 0/255 -> low entropy.
    """
    t = _as_u8_hwc(target_u8).astype(np.int16)
    p = _as_u8_hwc(pred_u8).astype(np.int16)
    return ((t - p) % 256).astype(np.uint8)


def reconstruct_from_residual(pred_u8: np.ndarray, residual_u8: np.ndarray) -> np.ndarray:
    p = _as_u8_hwc(pred_u8).astype(np.int16)
    r = np.asarray(residual_u8).astype(np.int16)
    return ((p + r) % 256).astype(np.uint8)


def zlib_bytes(buf: np.ndarray, level: int = 9) -> int:
    import zlib

    return len(zlib.compress(np.ascontiguousarray(buf).tobytes(), int(level)))


def brotli_bytes(buf: np.ndarray, quality: int = 11) -> int:
    import brotli

    return len(brotli.compress(np.ascontiguousarray(buf).tobytes(), quality=int(quality)))


def png_bytes(img_u8: np.ndarray) -> int:
    from PIL import Image

    a = _as_u8_hwc(img_u8)
    bio = io.BytesIO()
    Image.fromarray(a).save(bio, format="PNG", optimize=True)
    return bio.tell()


def png_bytes_single(plane_u8: np.ndarray) -> int:
    """PNG bytes of a single-channel (H,W) uint8 plane (for residual planes)."""
    from PIL import Image

    a = np.asarray(plane_u8)
    if a.ndim == 3 and a.shape[2] == 3:
        return png_bytes(a)
    bio = io.BytesIO()
    Image.fromarray(a.astype(np.uint8), mode="L").save(bio, format="PNG", optimize=True)
    return bio.tell()


def webp_bytes(img_u8: np.ndarray, quality: int = 75, *, lossless: bool = False) -> int:
    from PIL import Image

    a = _as_u8_hwc(img_u8)
    bio = io.BytesIO()
    Image.fromarray(a).save(bio, format="WEBP", quality=int(quality), lossless=bool(lossless), method=6)
    return bio.tell()


def jpeg2000_bytes(img_u8: np.ndarray, *, quality_layers: int | None = None) -> int:
    from PIL import Image

    a = _as_u8_hwc(img_u8)
    bio = io.BytesIO()
    Image.fromarray(a).save(bio, format="JPEG2000", irreversible=True)
    return bio.tell()


# --------------------------------------------------------------------------- #
# Temporal (mini-video) codecs via ffmpeg — the strong measured lever (R2).
# --------------------------------------------------------------------------- #
def _pad_even(frames_u8: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """Edge-replicate pad each frame to even H,W (yuv420p requires even); return
    (padded_frames, orig_h, orig_w) so callers can crop decoded frames back."""
    a0 = _as_u8_hwc(frames_u8[0])
    H, W = a0.shape[:2]
    ph, pw = H + (H & 1), W + (W & 1)
    if (ph, pw) == (H, W):
        return [_as_u8_hwc(f) for f in frames_u8], H, W
    out = [np.pad(_as_u8_hwc(f), ((0, ph - H), (0, pw - W), (0, 0)), mode="edge") for f in frames_u8]
    return out, H, W


def _ffmpeg_encode_bytes(frames_u8: list[np.ndarray], *, vcodec: str, crf: int,
                         extra: list[str] | None = None, container: str = "mp4") -> int:
    """Encode a list of native (H,W,3) uint8 RGB frames as a mini-video; return bytes.

    Raw rgb24 -> ffmpeg (libx265 / libvpx-vp9 / ffv1). One temp file under
    ``.omx/tmp/`` (transient scratch, immediately stat'd + deleted; NOT an evidence
    path per CLAUDE.md /tmp discipline).
    """
    if not frames_u8:
        return 0
    frames_u8, _oh, _ow = _pad_even(frames_u8)
    a0 = _as_u8_hwc(frames_u8[0])
    H, W = a0.shape[:2]
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=f".{container}", dir=str(_SCRATCH))
    os.close(fd)
    tmp_p = Path(tmp)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "20",
            "-i", "pipe:0", "-an", "-c:v", vcodec,
        ]
        if extra:
            cmd += list(extra)
        cmd += [str(tmp_p)]
        raw = b"".join(_as_u8_hwc(f).tobytes() for f in frames_u8)
        proc = subprocess.run(cmd, input=raw, capture_output=True)
        if proc.returncode != 0:
            raise KeyframeCodecError(
                f"ffmpeg {vcodec} failed rc={proc.returncode}: {proc.stderr.decode('utf-8','replace')[:400]}"
            )
        return tmp_p.stat().st_size
    finally:
        tmp_p.unlink(missing_ok=True)


def hevc_video_bytes(frames_u8: list[np.ndarray], crf: int = 34) -> int:
    return _ffmpeg_encode_bytes(
        frames_u8, vcodec="libx265", crf=crf,
        extra=["-pix_fmt", "yuv420p", "-x265-params", f"crf={crf}:log-level=none", "-preset", "medium"],
    )


def vp9_video_bytes(frames_u8: list[np.ndarray], crf: int = 40) -> int:
    return _ffmpeg_encode_bytes(
        frames_u8, vcodec="libvpx-vp9", crf=crf,
        extra=["-pix_fmt", "yuv420p", "-b:v", "0", "-crf", str(crf)], container="webm",
    )


def ffv1_video_bytes(frames_u8: list[np.ndarray]) -> int:
    """Lossless intra+inter FFV1 (the lossless temporal reference)."""
    return _ffmpeg_encode_bytes(
        frames_u8, vcodec="ffv1", crf=0, extra=["-pix_fmt", "gbrp"], container="mkv",
    )


# Proper VCM temporal-stream coding: codec x quality x GOP x B-frames, with DECODE
# roundtrip so d_pose can be measured on the ACTUALLY-decoded keyframes (not a still
# proxy). This is the non-toy MPEG/HEVC/AV1 keyframe-STREAM coding the rate track's
# ~4x-vs-WebP finding pointed at (adversarial overturn of the anti-MPEG negative).
_VIDEO_CODECS: dict[str, dict[str, Any]] = {
    "x265": {"vcodec": "libx265", "container": "mp4", "pix_fmt": "yuv420p"},
    "x265_444": {"vcodec": "libx265", "container": "mp4", "pix_fmt": "yuv444p"},
    "svtav1": {"vcodec": "libsvtav1", "container": "mp4", "pix_fmt": "yuv420p"},
    "vp9": {"vcodec": "libvpx-vp9", "container": "webm", "pix_fmt": "yuv420p"},
}


def _video_codec_args(codec: str, crf: int, gop: int, bframes: int, preset: str) -> tuple[str, list[str], str, str]:
    spec = _VIDEO_CODECS.get(codec)
    if spec is None:
        raise KeyframeCodecError(f"unknown video codec {codec!r}; known {list(_VIDEO_CODECS)}")
    vcodec, container, pix_fmt = spec["vcodec"], spec["container"], spec["pix_fmt"]
    extra = ["-pix_fmt", pix_fmt, "-g", str(int(gop))]
    if vcodec == "libx265":
        xp = f"crf={crf}:keyint={gop}:bframes={bframes}:log-level=none"
        extra += ["-preset", preset, "-x265-params", xp]
    elif vcodec == "libsvtav1":
        # SVT-AV1: preset is numeric 0..13 (lower=slower/better); default 6.
        sp = preset if preset.isdigit() else "6"
        extra += ["-preset", sp, "-crf", str(crf), "-svtav1-params", f"keyint={gop}:tune=0"]
    elif vcodec == "libvpx-vp9":
        extra += ["-b:v", "0", "-crf", str(crf), "-bf", str(bframes)]
    return vcodec, extra, container, pix_fmt


def encode_video_stream_bytes(frames_u8: list[np.ndarray], *, codec: str = "x265", crf: int = 34,
                              gop: int = 9999, bframes: int = 0, preset: str = "medium") -> int:
    """Temporal-stream bytes for a keyframe stream (I+P[+B]). ``gop`` large => 1 GOP
    (single I-frame, all others inter) — the correlated-keyframe regime."""
    vcodec, extra, container, _ = _video_codec_args(codec, crf, gop, bframes, preset)
    return _ffmpeg_encode_bytes(frames_u8, vcodec=vcodec, crf=crf, extra=extra, container=container)


def encode_decode_video_stream(frames_u8: list[np.ndarray], *, codec: str = "x265", crf: int = 34,
                               gop: int = 9999, bframes: int = 0, preset: str = "medium",
                               ) -> tuple[int, list[np.ndarray]]:
    """Encode the stream, then DECODE it back to native RGB frames. Returns
    ``(bytes, decoded_frames)`` so d_pose can be measured on the real decoded keyframes.
    """
    if not frames_u8:
        return 0, []
    import imageio.v3 as iio

    frames_u8, orig_h, orig_w = _pad_even(frames_u8)
    a0 = _as_u8_hwc(frames_u8[0])
    H, W = a0.shape[:2]
    vcodec, extra, container, _ = _video_codec_args(codec, crf, gop, bframes, preset)
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=f".{container}", dir=str(_SCRATCH))
    os.close(fd)
    tmp_p = Path(tmp)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "20",
            "-i", "pipe:0", "-an", "-c:v", vcodec, *extra, str(tmp_p),
        ]
        raw = b"".join(_as_u8_hwc(f).tobytes() for f in frames_u8)
        proc = subprocess.run(cmd, input=raw, capture_output=True)
        if proc.returncode != 0:
            raise KeyframeCodecError(
                f"ffmpeg {vcodec} failed rc={proc.returncode}: {proc.stderr.decode('utf-8','replace')[:400]}"
            )
        nbytes = tmp_p.stat().st_size
        decoded = [np.asarray(fr)[:orig_h, :orig_w, :3].astype(np.uint8) for fr in iio.imiter(tmp_p)]
        return nbytes, decoded
    finally:
        tmp_p.unlink(missing_ok=True)


def avif_bytes(img_u8: np.ndarray, quality: int = 50) -> int | None:
    """Per-frame AVIF (still) bytes via PIL; ``None`` if AVIF plugin unavailable."""
    from PIL import Image

    a = _as_u8_hwc(img_u8)
    bio = io.BytesIO()
    try:
        Image.fromarray(a).save(bio, format="AVIF", quality=int(quality))
    except (KeyError, OSError, ValueError):
        return None
    return bio.tell()


# --------------------------------------------------------------------------- #
# EGO-WARP PREDICTOR (rule-118 lever R1).
# --------------------------------------------------------------------------- #
def warp_by_homography(img_u8: np.ndarray, H: np.ndarray, *, out_hw: tuple[int, int] | None = None) -> np.ndarray:
    import cv2

    a = _as_u8_hwc(img_u8)
    hh, ww = (out_hw if out_hw is not None else a.shape[:2])
    return cv2.warpPerspective(
        a, np.asarray(H, dtype=np.float64), (int(ww), int(hh)),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


@dataclass(frozen=True)
class EgoFit:
    H: np.ndarray
    converged: bool
    mode: str


def fit_ego_homography(src_u8: np.ndarray, dst_u8: np.ndarray, *, mode: str = "ecc",
                       ecc_width: int = 320, iters: int = 60, eps: float = 1e-4) -> EgoFit:
    """Fit the homography mapping ``src -> dst`` (dense ECC on luma; no features).

    ``mode`` ladder: ``translation`` < ``affine`` < ``ecc`` (homography). ECC runs on
    luma downscaled to ``ecc_width`` for speed, then the warp matrix is rescaled to
    full resolution. Falls back to identity (== prev-copy predictor) on non-convergence.
    """
    import cv2

    src = rgb_to_luma(src_u8).astype(np.float32) / 255.0
    dst = rgb_to_luma(dst_u8).astype(np.float32) / 255.0
    H, W = src.shape
    scale = float(ecc_width) / float(W)
    sh, sw = max(8, int(round(H * scale))), int(ecc_width)
    src_s = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
    dst_s = cv2.resize(dst, (sw, sh), interpolation=cv2.INTER_AREA)
    if mode == "translation":
        motion, warp = cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)
    elif mode == "affine":
        motion, warp = cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32)
    elif mode == "ecc":
        motion, warp = cv2.MOTION_HOMOGRAPHY, np.eye(3, 3, dtype=np.float32)
    else:
        raise KeyframeCodecError(f"unknown ego-fit mode {mode!r}")
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps))
    converged = True
    try:
        _cc, warp = cv2.findTransformECC(src_s, dst_s, warp, motion, crit, None, 1)
    except cv2.error:
        converged = False
    # rescale warp (estimated at ecc_width) to full resolution: S = diag(1/scale,1/scale,1)
    S = np.diag([1.0 / scale, 1.0 / scale, 1.0]).astype(np.float64)
    Sinv = np.diag([scale, scale, 1.0]).astype(np.float64)
    if motion == cv2.MOTION_HOMOGRAPHY:
        Wm = warp.astype(np.float64)
    else:
        Wm = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
    Hfull = S @ Wm @ Sinv
    if not converged:
        Hfull = np.eye(3, dtype=np.float64)
    return EgoFit(H=Hfull, converged=converged, mode=mode)


def ego_warp_residual_u8(prev_kf_u8: np.ndarray, cur_kf_u8: np.ndarray, fit: EgoFit) -> np.ndarray:
    """Exact-invertible residual ``(cur - warp(prev, H)) mod 256`` (uint8)."""
    a = _as_u8_hwc(prev_kf_u8)
    pred = warp_by_homography(a, fit.H, out_hw=a.shape[:2])
    return residual_wraparound_u8(cur_kf_u8, pred)


# --------------------------------------------------------------------------- #
# TEXTURE-FREE SYNTHETIC RENDER (the store-nothing-but-xi endpoint proxy, ladder (c)).
#
# The SDF witness renders a piecewise-constant class-mean image on the SegNet argmax
# partition (no real texture). The "store nothing, just FiLM-condition by xi" endpoint
# stores only xi and renders frame0 synthetically. We proxy that render NOW (no trained
# checkpoint) by CLASS-MEAN FLATTENING the real frame: keep the partition structure +
# per-class mean colour, REMOVE all within-class texture. If PoseNet reads the ego-pose
# from THAT (paired with its xi-warp), then real texture is unnecessary -> keyframe rate
# collapses to ~xi. This is the honest, adversarial test of "PoseNet needs real texture".
# --------------------------------------------------------------------------- #
def upsample_argmax_nearest(argmax_hw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour upsample an integer argmax map to ``(out_h,out_w)`` (label-safe)."""
    import cv2

    a = np.asarray(argmax_hw).astype(np.int32)
    return cv2.resize(a, (int(out_w), int(out_h)), interpolation=cv2.INTER_NEAREST).astype(np.int64)


def class_mean_render(src_u8: np.ndarray, argmax_hw: np.ndarray, *, n_classes: int = 5,
                      residual_lowfreq: int = 0) -> np.ndarray:
    """Paint each argmax region with that region's MEAN colour from ``src`` (texture-free).

    ``argmax_hw`` is upsampled (nearest) to the source resolution if needed. Optionally add
    back the lowest ``residual_lowfreq x residual_lowfreq`` DCT coefficients of the
    per-class residual (a knob between pure-flat (c) and (b): 0 = pure class-mean).
    """
    a = _as_u8_hwc(src_u8).astype(np.float32)
    H, W, C = a.shape
    am = np.asarray(argmax_hw)
    if am.shape != (H, W):
        am = upsample_argmax_nearest(am, H, W)
    out = np.zeros_like(a)
    for k in range(int(n_classes)):
        mask = am == k
        if not mask.any():
            continue
        out[mask] = a[mask].mean(axis=0)
    render = np.clip(np.round(out), 0, 255).astype(np.uint8)
    if int(residual_lowfreq) > 0:
        resid = a - out
        lf = dct_truncate(np.clip(resid + 128, 0, 255).astype(np.uint8),
                          int(residual_lowfreq), int(residual_lowfreq)).astype(np.float32) - 128.0
        render = np.clip(np.round(out + lf), 0, 255).astype(np.uint8)
    return render


# --------------------------------------------------------------------------- #
# Rate accounting.
# --------------------------------------------------------------------------- #
_ARCHIVE_DENOM = 37_545_489  # upstream/evaluate.py rate denominator


def rate_from_bytes(total_bytes: float) -> float:
    """Contest rate-term contribution ``25 * bytes / 37_545_489``."""
    return 25.0 * float(total_bytes) / _ARCHIVE_DENOM


__all__ = [
    "KeyframeCodecError",
    "rgb_to_luma",
    "resize_roundtrip",
    "downsample_only",
    "gaussian_blur",
    "bitdepth_quantize",
    "dct_truncate",
    "order0_entropy_bits_per_symbol",
    "order0_entropy_bytes",
    "residual_wraparound_u8",
    "reconstruct_from_residual",
    "zlib_bytes",
    "brotli_bytes",
    "png_bytes",
    "png_bytes_single",
    "webp_bytes",
    "jpeg2000_bytes",
    "hevc_video_bytes",
    "vp9_video_bytes",
    "ffv1_video_bytes",
    "encode_video_stream_bytes",
    "encode_decode_video_stream",
    "avif_bytes",
    "upsample_argmax_nearest",
    "class_mean_render",
    "warp_by_homography",
    "EgoFit",
    "fit_ego_homography",
    "ego_warp_residual_u8",
    "rate_from_bytes",
]
