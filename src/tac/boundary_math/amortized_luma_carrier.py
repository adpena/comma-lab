# SPDX-License-Identifier: MIT
"""Amortized luma/RGB carrier — the score-native POSE-carrying appearance section.

THE #56 BLOCKER this closes: the score-native carrier amortized the SEG argmax into a 65KB INR
(``lever_b_generator``) and won rate (−59%) but had NO pose-carrying appearance — the palette
frame1 collapses PoseNet (d_pose 12.66). The reactivation path (#57) is an AMORTIZED appearance
section: instead of storing a per-pair low-res RGB frame (17 MB at factor-8, the #56 dead end), an
INR conditioned on ``(pair_idx, x, y)`` GENERATES the frame the pose-axis needs, amortized like the
seg generator.

THE SCORER FACTS (upstream/modules.py, verified):
  * PoseNet reads BOTH frames: resize 384×512 (bilinear) → ``rgb_to_yuv6`` (HALF-res luma 4ch +
    2×2-box-subsampled chroma 2ch) → 12ch → MSE on the first 6 of 12 pose-head dims. The pose
    signal is the 2-frame YUV6 motion — low-dimensional, hence amortizable.
  * SegNet reads ONLY the last frame (``x[:, -1, ...]``). So frame0 is SegNet-INVISIBLE → free to
    carry pose motion while frame1 carries d_seg (seg generator + boundary solve).

ARCHITECTURE (``AmortizedLumaCarrier``): a coordinate-INR mirroring ``ScoreNativeSegGenerator``:
Fourier(x, y) features → in_proj → per-pair FiLM-modulated hidden stack → 3-channel RGB head
(through a sigmoid → [0, 255]). One tiny per-pair ``mod`` code carries pair identity. The whole
carrier is the trained net weights + per-pair mod codes, quantized + brotli'd = the appearance
section bytes.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): MLX is the fast path (training); the numpy forward is the
PORTABILITY reference (the inflate-time decoder is pure numpy, scorer-free, deterministic). NO MPS.
d_pose / d_seg read on the exact frozen CPU-torch scorer; GT via ``frame_utils.yuv420_to_rgb`` ONLY.
Evidence ``[macOS-MLX research-signal]`` for the carrier forward; ``[local CPU-torch advisory]`` once
the frame is scored. Non-promotable.

NO-FAKE (class 2): the byte cost is the brotli of the ACTUAL quantized weights+mod (not a constant);
the numpy forward ACTUALLY runs the INR (not a stored per-pair table); the reconstructed frame
ACTUALLY depends on (pair, x, y) — a stub returning a constant frame FAILS the tests (which assert
the carrier output varies across pairs and across pixels, and reduces d_pose on the real PoseNet vs
a constant frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── deterministic free Fourier table seed (reconstructed at inflate, NOT stored) ──
_LUMA_FOURIER_SEED = 7

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


@dataclass(frozen=True)
class LumaCarrierConfig:
    """Architecture hyperparameters that fully determine the carrier shape + byte cost.

    The capacity knobs swept in the RD curve are ``hidden_dim`` / ``mod_dim`` / ``n_hidden`` /
    ``n_fourier`` / ``quant_bits``. ``num_pairs`` is the data shape (600). The carrier outputs a
    full-resolution camera RGB frame (the score-native quantity is the SCORED frame, so the INR
    decodes directly to camera resolution at inflate time — a free coordinate grid).
    """

    num_pairs: int
    n_fourier: int = 24
    hidden_dim: int = 64
    n_hidden: int = 3
    mod_dim: int = 24
    n_channels: int = 3
    fourier_sigma: float = 6.0
    quant_bits: int = 8  # weight quantization for the byte cost (per-tensor symmetric)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "num_pairs": self.num_pairs,
            "n_fourier": self.n_fourier,
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "mod_dim": self.mod_dim,
            "n_channels": self.n_channels,
            "fourier_sigma": self.fourier_sigma,
            "quant_bits": self.quant_bits,
        }


def build_coords(h: int, w: int) -> np.ndarray:
    """(H*W, 2) pixel coordinate grid in [-1, 1] (row-major, gx, gy) — same convention as seg gen."""

    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def deterministic_fourier_B(n_fourier: int, fourier_sigma: float) -> np.ndarray:
    """Deterministic (seeded) Fourier feature matrix (2, n_fourier) — a FREE table at inflate."""

    rng = np.random.default_rng(_LUMA_FOURIER_SEED)
    return (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Pure-numpy forward (the PORTABLE inflate-time reference; scorer-free).
# ---------------------------------------------------------------------------
def numpy_reference_forward(
    params: dict[str, np.ndarray],
    fourier_B: np.ndarray,
    coords: np.ndarray,
    mod_vec: np.ndarray,
    n_hidden: int,
    hidden_dim: int,
) -> np.ndarray:
    """Pure-numpy mirror of ``AmortizedLumaCarrier.__call__`` (float64 accumulation).

    Returns (P, n_channels) RGB in [0, 255] (sigmoid head × 255). The portability contract is
    near-exact RGB (NOT argmax like the seg gen): a small fp32-vs-fp64 drift is tolerated by the
    parity gate (max abs RGB delta ≤ ~1 LSB after rounding), so the inflate-time numpy decode and
    the MLX training forward produce the same scored frame.
    """

    coords = coords.astype(np.float64)
    fourier_B = fourier_B.astype(np.float64)
    mod_vec = mod_vec.astype(np.float64)
    p = {k: v.astype(np.float64) for k, v in params.items()}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = coords @ fourier_B
        feat = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
        h = np.maximum(feat @ p["in_proj.weight"].T + p["in_proj.bias"], 0.0)
        film = mod_vec @ p["film.weight"].T + p["film.bias"]
        film = film.reshape(n_hidden, 2, hidden_dim)
        for li in range(n_hidden):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            h = np.maximum((h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * scale + shift, 0.0)
        rgb01 = _sigmoid(h @ p["out.weight"].T + p["out.bias"])
    return (rgb01 * 255.0).astype(np.float64)


def carrier_frame(
    params: dict[str, np.ndarray],
    cfg: LumaCarrierConfig,
    coords: np.ndarray,
    pair_idx: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Per-pair carrier RGB frame (H, W, 3) uint8 — the numpy-portable inflate-time decode.

    ``params`` must contain ``mod`` (num_pairs, mod_dim) + ``in_proj/film/hidden.*/out``.
    Reconstructs the deterministic Fourier table from cfg (free table). SCORER-FREE.
    """

    fourier_B = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    mod_vec = np.asarray(params["mod"])[pair_idx]
    rgb = numpy_reference_forward(params, fourier_B, coords, mod_vec, cfg.n_hidden, cfg.hidden_dim)
    return np.clip(np.round(rgb.reshape(h, w, cfg.n_channels)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Byte accounting — the appearance-section rate (quantized weights + mod, brotli).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CarrierByteAccount:
    """The carrier's appearance-section byte cost (honest, brotli-q11 measured)."""

    weight_bytes: int       # quantized net weights (everything except per-pair mod), brotli
    mod_bytes: int          # quantized per-pair mod codes, brotli
    scale_bytes: int        # the fp16 dequant scales (one per quantized tensor)
    total_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "weight_bytes": self.weight_bytes,
            "mod_bytes": self.mod_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.total_bytes,
        }


def _quantize_symmetric(arr: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric quantization → (int codes, fp scale). Round-trip = codes*scale."""

    a = np.asarray(arr, dtype=np.float64)
    qmax = float(2 ** (bits - 1) - 1)
    amax = float(np.max(np.abs(a))) if a.size else 0.0
    scale = (amax / qmax) if amax > 0 else 1.0
    codes = np.clip(np.round(a / scale), -qmax, qmax).astype(np.int16)
    return codes, scale


def quantize_params(params: dict[str, np.ndarray], bits: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Quantize every param tensor (per-tensor symmetric). Returns (codes, scales)."""

    codes: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for k, v in params.items():
        c, s = _quantize_symmetric(v, bits)
        codes[k] = c
        scales[k] = s
    return codes, scales


def dequantize_params(codes: dict[str, np.ndarray], scales: dict[str, float]) -> dict[str, np.ndarray]:
    """Inverse of :func:`quantize_params` → fp32 params (what the inflate decode actually uses)."""

    return {k: (codes[k].astype(np.float32) * np.float32(scales[k])) for k in codes}


def measure_carrier_bytes(params: dict[str, np.ndarray], cfg: LumaCarrierConfig) -> CarrierByteAccount:
    """Honest brotli-q11 byte cost of the quantized carrier (weights + mod + scales).

    The dequant scales (one fp16 per tensor) are counted explicitly. The codes are packed to the
    minimal byte width for the quant_bits, then brotli'd (the per-tensor distributions are smooth →
    brotli exploits them). This is the appearance-section rate the score adds.
    """

    import brotli

    codes, scales = quantize_params(params, cfg.quant_bits)
    weight_chunks = []
    mod_chunks = []
    for k, c in codes.items():
        # int16 codes within +-127 for 8-bit; store as int8 when it fits, else int16.
        packed = c.astype(np.int8).tobytes() if cfg.quant_bits <= 8 else c.astype(np.int16).tobytes()
        (mod_chunks if k == "mod" else weight_chunks).append(packed)
    weight_blob = brotli.compress(b"".join(weight_chunks), quality=11) if weight_chunks else b""
    mod_blob = brotli.compress(b"".join(mod_chunks), quality=11) if mod_chunks else b""
    scale_bytes = len(np.asarray(list(scales.values()), dtype=np.float16).tobytes())
    return CarrierByteAccount(
        weight_bytes=len(weight_blob),
        mod_bytes=len(mod_blob),
        scale_bytes=scale_bytes,
        total_bytes=len(weight_blob) + len(mod_blob) + scale_bytes,
    )


# ---------------------------------------------------------------------------
# Checkpoint I/O — portable .npz (NO /tmp).
# ---------------------------------------------------------------------------
def save_carrier_npz(path: Path, params: dict[str, np.ndarray], cfg: LumaCarrierConfig) -> Path:
    """Persist trained weights + per-pair mod + cfg to a portable .npz."""

    path = Path(path)
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"param::{k}": np.asarray(v).astype(np.float32) for k, v in params.items()}
    for k, v in cfg.to_dict().items():
        flat[f"cfg::{k}"] = np.asarray(v)
    np.savez(path, **flat)
    return path


def load_carrier_npz(path: Path) -> tuple[dict[str, np.ndarray], LumaCarrierConfig]:
    """Load a carrier checkpoint saved by :func:`save_carrier_npz`."""

    path = Path(path)
    with np.load(path) as z:
        params = {k[len("param::"):]: z[k] for k in z.files if k.startswith("param::")}
        cfg_kv = {k[len("cfg::"):]: z[k] for k in z.files if k.startswith("cfg::")}
    cfg = LumaCarrierConfig(
        num_pairs=int(cfg_kv["num_pairs"]),
        n_fourier=int(cfg_kv["n_fourier"]),
        hidden_dim=int(cfg_kv["hidden_dim"]),
        n_hidden=int(cfg_kv["n_hidden"]),
        mod_dim=int(cfg_kv["mod_dim"]),
        n_channels=int(cfg_kv["n_channels"]),
        fourier_sigma=float(cfg_kv["fourier_sigma"]),
        quant_bits=int(cfg_kv["quant_bits"]),
    )
    return params, cfg


def carrier_param_count(cfg: LumaCarrierConfig) -> int:
    """Total trainable param count for a given config (the capacity scalar for the RD sweep)."""

    coord_feat = 2 * cfg.n_fourier
    n = 0
    n += coord_feat * cfg.hidden_dim + cfg.hidden_dim                       # in_proj
    n += cfg.mod_dim * (2 * cfg.hidden_dim * cfg.n_hidden) + (2 * cfg.hidden_dim * cfg.n_hidden)  # film
    n += cfg.n_hidden * (cfg.hidden_dim * cfg.hidden_dim + cfg.hidden_dim)  # hidden stack
    n += cfg.hidden_dim * cfg.n_channels + cfg.n_channels                   # out
    n += cfg.num_pairs * cfg.mod_dim                                        # per-pair mod
    return int(n)


__all__ = [
    "CarrierByteAccount",
    "LumaCarrierConfig",
    "build_coords",
    "carrier_frame",
    "carrier_param_count",
    "dequantize_params",
    "deterministic_fourier_B",
    "load_carrier_npz",
    "measure_carrier_bytes",
    "numpy_reference_forward",
    "quantize_params",
    "save_carrier_npz",
]
