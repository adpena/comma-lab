# SPDX-License-Identifier: MIT
"""Distillation to a SMALLER learned student basis (task #74) — the architecture + numpy-portable
decode + byte accounting + KD-loss math.

THE KEY INSIGHT (why distillation breaks the #57/#61/#62 wall):

  Task #62 (``lever_c_viability_smoke``) proved a small fresh-init conv decoder CANNOT learn d_seg
  from argmax-CE-against-GT — the RGB->frozen-SegNet composition is deep + ill-conditioned, so the
  trained student's exact d_seg pinned at the *constant-frame floor* (0.507) while pose moved 114x.
  The seg objective and the pose objective are antagonistic at small-conv capacity when the target
  is the GROUND TRUTH (the student must simultaneously discover a frame whose frozen-SegNet argmax
  matches GT AND whose luma gives the right pose — two halves of the #61 wall).

  The frontier TEACHER (the 177KB HNeRV decoder) ALREADY decodes 1200 frames that ARE
  d_seg-correct (mean d_seg ~5.4e-4) AND pose-in-tube (mean d_pose ~2.3e-5). So a SMALLER student
  trained to MATCH THE TEACHER'S OUTPUT FRAMES learns from targets that are already argmax-correct +
  pose-in-tube. It inherits score-correctness via simple RGB recon (a well-conditioned objective)
  instead of fighting the RGB->SegNet conditioning. **The teacher IS the loss.** This is the
  Hinton-Vinyals-Dean 2014 dark-knowledge transfer applied to a video-codec basis: the student need
  not rediscover the scorer manifold, only mimic a frame already on it.

DISTINCT FROM #71 (structural weight compression): #71 compresses the TEACHER's existing weights
post-hoc (factor / prune / quant). #74 TRAINS A NEW SMALLER ARCHITECTURE (fewer blocks / channels /
per-pair-latent) via knowledge distillation. They COMPOSE: distill (#74) -> then #71-compress +
#69-requant the student.

THE STUDENT ARCHITECTURE: a shared conv decoder (PR95 L18 block: Conv(in,out*4,3x3) +
PixelShuffle(2) + bilinear-skip + sin) amortized across pairs, with a per-pair latent ``z_p`` that
decodes BOTH frames of the pair (PoseNet reads both frames; SegNet reads only frame1). The size
knob is ``latent_dim`` / ``seed_ch`` / ``stage_channels`` (PR95 L19: ~94% weights / ~6% latents).
Byte cost = int8+brotli(shared weights) + int8+brotli(per-pair latents) + fp16 per-tensor scales.

THE KD LOSS (the teacher is the target, NOT GT):
  1. **Teacher-frame recon** — student frame0/frame1 recon-MSE against the TEACHER'S decoded frame0/
     frame1 (the well-conditioned objective the #62 finding says works). Frame1 carries d_seg+pose;
     frame0 carries pose only.
  2. **PR95 KL-T=2.0 SegNet-logit distill** — student SegNet logits (gradient flows) match the
     TEACHER'S SegNet logits on the teacher's frame1. Hinton T^2 normalization. The full soft 5-class
     distribution (boundary-aware dark knowledge) — exactly the teacher's argmax partition.
  3. **Pose-MSE distill** — student 6-dim PoseNet pose matches the TEACHER's 6-dim pose (the
     teacher's pose is in-tube; matching it inherits the in-tube pose).

COMPUTE-SUBSTRATE LAW (CLAUDE.md): the numpy forward here is the PORTABILITY reference (inflate-time
decode is pure numpy, scorer-free, deterministic). torch is the training fast path. NO MPS. d_seg/
d_pose RE-MEASURED on the exact frozen CPU-torch scorer; GT via ``frame_utils.yuv420_to_rgb`` ONLY.

NO-FAKE (class 2 + class 6 + class 8): the numpy forward ACTUALLY runs the conv stack per pair (not a
stored frame table); the student frames ACTUALLY depend on (latent, pixel); the byte cost is the
brotli of ACTUAL quantized weights+latents (not a constant); a CONSTANT-frame student CANNOT reduce
d_seg (its argmax is a blank partition) and the tests assert this. The d_seg/d_pose are the EXACT
frozen-scorer argmax-flip / 6-dim-MSE measurements (not a proxy). This is a candidate generator; the
contest exact ``evaluate.py`` (CPU+CUDA) is the only authority that can move the pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tac.boundary_math.conv_pair_decoder import (
    _bilinear_resize,
    _conv3x3,
    _pixel_shuffle,
    _sigmoid,
    _sin,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")

# Camera-native resolution the scored frame lives at. The student decodes at a block-stack resolution
# then bilinearly resizes to camera res (same contract as the teacher's bicubic up).
CAMERA_H, CAMERA_W = 874, 1164
# SegNet input resolution (the d_seg metric reads SegNet on the last frame resized here).
SEG_H, SEG_W = 384, 512

KL_TEMPERATURE = 2.0  # PR95 / Quantizr canon (Hinton-Vinyals-Dean 2014, T=2.0).


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Student config — the SIZE knob (the rate-vs-distortion sweep axis).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudentDecoderConfig:
    """Architecture hyperparameters that fully determine the student shape + byte cost.

    The student maps a per-pair latent ``z_p`` (``latent_dim``) -> a spatial seed (``seed_ch`` x
    ``seed_h`` x ``seed_w``) -> ``len(stage_channels)`` PixelShuffle(2) conv-blocks -> ``2 *
    n_channels`` output channels (BOTH frames of the pair: frame0 channels then frame1 channels).
    The size knob is ``latent_dim`` / ``seed_ch`` / ``stage_channels``.

    ``num_pairs`` is the data shape. ``size_label`` is an operator tag (e.g. ``"80kb"``) carried in
    the result rows so the sweep is self-describing; it does NOT affect the math.
    """

    num_pairs: int
    latent_dim: int = 24
    seed_ch: int = 32
    seed_h: int = 6
    seed_w: int = 8
    stage_channels: tuple[int, ...] = (32, 24, 16, 12)  # per PixelShuffle(2) stage; 4 stages = 16x up
    n_channels: int = 3
    quant_bits: int = 8
    size_label: str = ""

    def to_dict(self) -> dict:
        return {
            "num_pairs": self.num_pairs,
            "latent_dim": self.latent_dim,
            "seed_ch": self.seed_ch,
            "seed_h": self.seed_h,
            "seed_w": self.seed_w,
            "stage_channels": list(self.stage_channels),
            "n_channels": self.n_channels,
            "quant_bits": self.quant_bits,
            "size_label": self.size_label,
        }

    def final_hw(self) -> tuple[int, int]:
        """Block-stack output resolution before the final bilinear resize to camera res."""

        return (
            self.seed_h * (2 ** len(self.stage_channels)),
            self.seed_w * (2 ** len(self.stage_channels)),
        )

    @property
    def out_channels(self) -> int:
        """The student decodes BOTH frames of the pair: 2 * n_channels output channels."""

        return 2 * self.n_channels


# ---------------------------------------------------------------------------
# Pure-numpy forward (the portable inflate-time reference; scorer-free, deterministic).
# ---------------------------------------------------------------------------
def numpy_reference_forward(
    params: dict[str, np.ndarray],
    cfg: StudentDecoderConfig,
    latent: np.ndarray,
) -> np.ndarray:
    """Pure-numpy mirror of :class:`TorchStudentDecoder.forward` for one pair latent.

    Returns ``(2, n_channels, final_h, final_w)`` RGB in [0,255] at the block-stack resolution
    (BEFORE the final camera-res resize; :func:`student_pair_frames` does the resize). float64.
    Index 0 = frame0, index 1 = frame1.

    Params (the shared decoder weights, NOT the per-pair latents):
      * ``seed.weight`` (seed_ch*seed_h*seed_w, latent_dim), ``seed.bias``
      * ``stage{i}.weight`` (out_ch*4, in_ch, 3, 3), ``stage{i}.bias`` for each PixelShuffle stage
      * ``stage{i}.skip`` (out_ch, in_ch, 1, 1), ``stage{i}.skip_bias`` — bilinear-skip 1x1 conv
      * ``out.weight`` (2*n_channels, last_ch, 3, 3), ``out.bias`` — BOTH frames
    """

    p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    z = np.asarray(latent, dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        seed = p["seed.weight"] @ z + p["seed.bias"]  # (seed_ch*seed_h*seed_w,)
        h = seed.reshape(cfg.seed_ch, cfg.seed_h, cfg.seed_w)

        for i, _out_ch in enumerate(cfg.stage_channels):
            conv = _conv3x3(h, p[f"stage{i}.weight"], p[f"stage{i}.bias"])  # (out_ch*4, H, W)
            up = _pixel_shuffle(conv, 2)  # (out_ch, H*2, W*2)
            skip_in = _bilinear_resize(h, up.shape[1], up.shape[2])  # (in_ch, H*2, W*2)
            sk_w = p[f"stage{i}.skip"]  # (out_ch, in_ch, 1, 1)
            skip = np.einsum("oixy,ihw->ohw", sk_w, skip_in) + p[f"stage{i}.skip_bias"][:, None, None]
            h = _sin(up + skip)  # sin activation (PR95 L18)

        rgb01 = _sigmoid(_conv3x3(h, p["out.weight"], p["out.bias"]))  # (2*n_channels, fh, fw)
    fh, fw = rgb01.shape[1], rgb01.shape[2]
    rgb01 = rgb01.reshape(2, cfg.n_channels, fh, fw)
    return rgb01 * 255.0  # (2, n_channels, fh, fw)


def student_pair_frames(
    params: dict[str, np.ndarray],
    cfg: StudentDecoderConfig,
    latents: np.ndarray,
    pair_idx: int,
    out_h: int = CAMERA_H,
    out_w: int = CAMERA_W,
) -> np.ndarray:
    """Per-pair student decode -> ``(2, H, W, 3)`` uint8 (frame0, frame1) — numpy-portable inflate.

    ``latents`` is (num_pairs, latent_dim). Runs the conv stack for ``pair_idx`` then bilinearly
    resizes each frame to camera resolution. SCORER-FREE.
    """

    z = np.asarray(latents)[pair_idx]
    pair_chw = numpy_reference_forward(params, cfg, z)  # (2, 3, fh, fw)
    out = np.empty((2, out_h, out_w, cfg.n_channels), dtype=np.uint8)
    for t in range(2):
        rgb_chw = _bilinear_resize(pair_chw[t], out_h, out_w)  # (3, H, W)
        rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
        out[t] = np.clip(np.round(rgb_hwc), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Byte accounting — quantized shared weights + per-pair latents (brotli q11).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudentByteAccount:
    weight_bytes: int   # quantized shared decoder weights, brotli
    latent_bytes: int   # quantized per-pair latents, brotli
    scale_bytes: int    # fp16 per-tensor dequant scales
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "weight_bytes": self.weight_bytes,
            "latent_bytes": self.latent_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.total_bytes,
        }


def _quantize_symmetric(arr: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    a = np.asarray(arr, dtype=np.float64)
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(a))) if a.size else 0.0
    scale = (amax / qmax) if amax > 0 else 1.0
    codes = np.clip(np.round(a / scale), -qmax, qmax).astype(np.int16)
    return codes, scale


def quantize_params(
    params: dict[str, np.ndarray], bits: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    codes: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for k, v in params.items():
        c, s = _quantize_symmetric(v, bits)
        codes[k] = c
        scales[k] = s
    return codes, scales


def dequantize_params(
    codes: dict[str, np.ndarray], scales: dict[str, float]
) -> dict[str, np.ndarray]:
    return {k: (codes[k].astype(np.float32) * np.float32(scales[k])) for k in codes}


def measure_student_bytes(
    weights: dict[str, np.ndarray],
    latents: np.ndarray,
    cfg: StudentDecoderConfig,
) -> StudentByteAccount:
    """Honest brotli-q11 byte cost: quantized shared weights + per-pair latents + fp16 scales.

    Mirrors the PR95 L19/L20 split (decoder weights dominate; latents are the small per-pair tail).
    NO-FAKE: this is the ACTUAL brotli of the ACTUAL quantized arrays, not a formula estimate.
    """

    import brotli

    wcodes, wscales = quantize_params(weights, cfg.quant_bits)
    lcodes, lscale = _quantize_symmetric(latents, cfg.quant_bits)

    wchunks = [
        (c.astype(np.int8) if cfg.quant_bits <= 8 else c.astype(np.int16)).tobytes()
        for c in wcodes.values()
    ]
    weight_blob = brotli.compress(b"".join(wchunks), quality=11) if wchunks else b""
    lblob = brotli.compress(
        (lcodes.astype(np.int8) if cfg.quant_bits <= 8 else lcodes.astype(np.int16)).tobytes(),
        quality=11,
    )
    scale_vals = [*list(wscales.values()), lscale]
    scale_bytes = len(np.asarray(scale_vals, dtype=np.float16).tobytes())
    return StudentByteAccount(
        weight_bytes=len(weight_blob),
        latent_bytes=len(lblob),
        scale_bytes=scale_bytes,
        total_bytes=len(weight_blob) + len(lblob) + scale_bytes,
    )


def student_param_count(cfg: StudentDecoderConfig) -> int:
    """Total trainable param count (shared decoder weights + per-pair latents)."""

    n = 0
    seed_out = cfg.seed_ch * cfg.seed_h * cfg.seed_w
    n += seed_out * cfg.latent_dim + seed_out
    in_ch = cfg.seed_ch
    for out_ch in cfg.stage_channels:
        n += (out_ch * 4) * in_ch * 9 + (out_ch * 4)  # 3x3 conv to out*4
        n += out_ch * in_ch * 1 + out_ch              # 1x1 skip
        in_ch = out_ch
    n += cfg.out_channels * in_ch * 9 + cfg.out_channels  # out 3x3 -> BOTH frames
    n += cfg.num_pairs * cfg.latent_dim                   # per-pair latents
    return int(n)


# ---------------------------------------------------------------------------
# A canonical size ladder (the rate-vs-distortion sweep axis). Each entry is a
# StudentDecoderConfig factory parameterized by num_pairs. Byte cost is MEASURED,
# not predicted (the labels are nominal targets, the actual bytes are reported).
# ---------------------------------------------------------------------------
def size_ladder(num_pairs: int) -> dict[str, StudentDecoderConfig]:
    """Return the canonical {nominal_label -> StudentDecoderConfig} sweep ladder.

    The labels are nominal KB targets; the ACTUAL byte cost is MEASURED per config (the rate-vs-
    distortion curve x-axis is the measured total_bytes, not the label). Sizes are achieved by
    scaling latent_dim + stage channel widths (the two PR95-L19 capacity knobs).

    RESOLUTION CONTRACT (the task-#74 correction): the seg-bearing student frame MUST decode at the
    teacher's eval resolution (384x512) so it can represent the teacher's argmax boundaries the
    d_seg metric reads — a coarser block-stack (e.g. 96x128) physically cannot inherit the teacher's
    d_seg. The ladder uses seed (6,8) with SIX PixelShuffle(2) stages: 6*2^6=384, 8*2^6=512 = the
    teacher's eval_size exactly. (Task #62's 4-stage 96x128 decoder was 16x too coarse for d_seg.)
    """

    def _c(lat, sc, stages, label):
        return StudentDecoderConfig(num_pairs, latent_dim=lat, seed_ch=sc, seed_h=6, seed_w=8,
                                    stage_channels=stages, size_label=label)

    return {
        "40kb": _c(12, 16, (16, 16, 14, 12, 10, 8), "40kb"),
        "60kb": _c(16, 20, (20, 20, 18, 14, 12, 10), "60kb"),
        "80kb": _c(20, 24, (24, 24, 20, 16, 14, 12), "80kb"),
        "100kb": _c(24, 28, (28, 28, 24, 20, 16, 14), "100kb"),
        "120kb": _c(28, 32, (32, 32, 28, 22, 18, 16), "120kb"),
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O — portable .npz (NO /tmp).
# ---------------------------------------------------------------------------
def save_student_npz(
    path: Path,
    weights: dict[str, np.ndarray],
    latents: np.ndarray,
    cfg: StudentDecoderConfig,
) -> Path:
    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"w::{k}": np.asarray(v).astype(np.float32) for k, v in weights.items()}
    flat["latents"] = np.asarray(latents).astype(np.float32)
    for k, v in cfg.to_dict().items():
        flat[f"cfg::{k}"] = np.asarray(v)
    np.savez(path, **flat)
    return path


def load_student_npz(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, StudentDecoderConfig]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        weights = {k[len("w::"):]: z[k] for k in z.files if k.startswith("w::")}
        latents = z["latents"]
        cfg_kv = {k[len("cfg::"):]: z[k] for k in z.files if k.startswith("cfg::")}
    cfg = StudentDecoderConfig(
        num_pairs=int(cfg_kv["num_pairs"]),
        latent_dim=int(cfg_kv["latent_dim"]),
        seed_ch=int(cfg_kv["seed_ch"]),
        seed_h=int(cfg_kv["seed_h"]),
        seed_w=int(cfg_kv["seed_w"]),
        stage_channels=tuple(int(x) for x in cfg_kv["stage_channels"]),
        n_channels=int(cfg_kv["n_channels"]),
        quant_bits=int(cfg_kv["quant_bits"]),
        size_label=str(cfg_kv["size_label"]) if "size_label" in cfg_kv else "",
    )
    return weights, latents, cfg


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "KL_TEMPERATURE",
    "SEG_H",
    "SEG_W",
    "StudentByteAccount",
    "StudentDecoderConfig",
    "dequantize_params",
    "load_student_npz",
    "measure_student_bytes",
    "numpy_reference_forward",
    "quantize_params",
    "save_student_npz",
    "size_ladder",
    "student_pair_frames",
    "student_param_count",
]
