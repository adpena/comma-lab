# SPDX-License-Identifier: MIT
"""S2SBS architecture — HF Hermitian-FFT byte channel + base renderer.

The substrate exploits the empirically-measured HF blindspot of the
stride-2-stem SegNet / PoseNet pathway. Useful side-information bytes
(per-pair pose deltas, boundary refinements) are encoded into the high
spatial-frequency components of the rendered RGB frames via
Hermitian-symmetric FFT coefficient pairs at amplitude delta_amp_uint8,
then quantized to uint8 by the contest raw-file emit path.

The base renderer is a deterministic placeholder; production training
replaces it with a learned head. Per HNeRV parity lesson 5 the
architecture is the FULL renderer (RGB out), NOT a single-component slot.

Key reference numbers (audit memo, macOS-CPU advisory, single-pair):

* Joint-safe capacity bytes per frame ≈ 97_460 at delta_amp=0.75
* Joint-safe capacity bytes per pair ≈ 194_920 (frame0 + frame1)
* PRBS-31 BER at joint-safe delta ≈ 0.4231 (uint8 quant noise)
* BSC capacity ≈ 0.0171 bits/encoded-bit before ECC
* Effective payload pre-ECC upper bound ≈ 140 bytes per (single pair, single
  channel) demo; substrate codecs must wrap their own ECC.

All capacity numbers stay ``score_claim=false`` and
``ready_for_exact_eval_dispatch=false`` until a byte-closed archive lands
through a contest exact eval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

CONTEST_NUM_PAIRS = 600
SCORER_HW = (384, 512)
CAMERA_HW = (874, 1164)

# Default Hermitian-FFT amplitude (in uint8 units). Audit demo used 0.75
# delta at single-channel R; 1.0 delta moves PoseNet MSE > 1e-5 joint-safe
# threshold. Substrate codec defaults to 0.75; ECC must absorb the ~42%
# raw BER. NEVER set above 1.0 without re-running joint-safety audit.
DEFAULT_DELTA_AMP_UINT8 = 0.75
MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT = 1.0


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class S2sbsConfig:
    """Static design-time config for the S2SBS substrate."""

    num_pairs: int = CONTEST_NUM_PAIRS
    output_height: int = SCORER_HW[0]
    output_width: int = SCORER_HW[1]
    hf_blindspot_lf_cutoff_h: int = 96  # half SegNet stem Nyquist
    hf_blindspot_lf_cutoff_w: int = 128
    delta_amp_uint8: float = DEFAULT_DELTA_AMP_UINT8
    payload_channel: str = "R"  # one of R / G / B
    base_seed: int = 124
    payload_bytes_per_pair: int = 32  # post-ECC budget per pair
    ecc_rate: float = 0.25  # 4:1 ECC overhead per audit BER

    def __post_init__(self) -> None:
        for name in ("num_pairs", "output_height", "output_width"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_pairs > 0xFFFF:
            raise ValueError("num_pairs must fit in uint16")
        if self.output_height > 0xFFFF or self.output_width > 0xFFFF:
            raise ValueError("output dimensions must fit in uint16")
        if not 1 <= int(self.hf_blindspot_lf_cutoff_h) < int(self.output_height):
            raise ValueError("hf_blindspot_lf_cutoff_h must be in [1, output_height)")
        if not 1 <= int(self.hf_blindspot_lf_cutoff_w) < int(self.output_width):
            raise ValueError("hf_blindspot_lf_cutoff_w must be in [1, output_width)")
        _require_finite("delta_amp_uint8", float(self.delta_amp_uint8))
        if not 0.0 < float(self.delta_amp_uint8) <= MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT:
            raise ValueError(
                "delta_amp_uint8 must be in (0, "
                f"{MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT}] per audit joint-safety"
            )
        if self.payload_channel not in ("R", "G", "B"):
            raise ValueError("payload_channel must be one of R/G/B")
        if not 0 <= int(self.base_seed) <= 0xFFFFFFFF:
            raise ValueError("base_seed must fit in uint32")
        if int(self.payload_bytes_per_pair) < 0:
            raise ValueError("payload_bytes_per_pair must be non-negative")
        _require_finite("ecc_rate", float(self.ecc_rate))
        if not 0.0 < float(self.ecc_rate) <= 1.0:
            raise ValueError("ecc_rate must be in (0, 1]")

    @property
    def channel_index(self) -> int:
        return {"R": 0, "G": 1, "B": 2}[self.payload_channel]

    @property
    def raw_payload_bytes_per_pair(self) -> int:
        """Pre-ECC raw bytes per pair after the ECC overhead is reserved."""
        return max(0, round(self.payload_bytes_per_pair / float(self.ecc_rate)))


@dataclass(frozen=True)
class PayloadChannel:
    """Per-pair payload byte tuple. Empty payload renders the base frame."""

    pair_index: int
    payload: bytes = b""

    def __post_init__(self) -> None:
        if not 0 <= int(self.pair_index) < CONTEST_NUM_PAIRS:
            raise ValueError(
                f"pair_index must be in [0, {CONTEST_NUM_PAIRS}); got {self.pair_index}"
            )
        if not isinstance(self.payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes")
        if len(self.payload) > 0xFFFF:
            raise ValueError("payload per pair must fit in uint16 byte length")


class HfBlindspotMask(nn.Module):
    """Build a deterministic HF blindspot mask in the FFT domain.

    The mask is True in the high-frequency annulus that lies OUTSIDE the
    bilinear-resize Nyquist band of the SegNet stem after rgb_to_yuv6 and
    half-resolution downsampling. Per audit memo §"Architectural
    derivation" + §Caveats: bilinear leakage is ~5-15%; the mask is the
    safe band, not a Shannon-tight bound.
    """

    def __init__(self, cfg: S2sbsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, w = cfg.output_height, cfg.output_width
        # FFT uses rfft2 -> shape (h, w//2 + 1) for real input.
        # Coordinates are wrapped: |kh| ∈ [0, h/2], |kw| ∈ [0, w/2].
        kh = torch.arange(h)
        kh = torch.where(kh > h // 2, kh - h, kh).abs()
        kw = torch.arange(w // 2 + 1)
        kh_grid, kw_grid = torch.meshgrid(kh, kw, indexing="ij")
        lf_cutoff_h = int(cfg.hf_blindspot_lf_cutoff_h)
        lf_cutoff_w = int(cfg.hf_blindspot_lf_cutoff_w)
        mask = (kh_grid >= lf_cutoff_h) | (kw_grid >= lf_cutoff_w)
        # Exclude the DC bin and the Nyquist column / row to avoid the
        # self-conjugate self-energy lines (which can leak into LF after
        # downsampling).
        mask[0, 0] = False
        mask[h // 2, :] = False
        mask[:, w // 2] = False
        self.register_buffer("mask", mask)

    def coordinate_count(self) -> int:
        return int(self.mask.sum().item())  # type: ignore[attr-defined]


class HfFftByteCodec(nn.Module):
    """Hermitian-symmetric FFT byte codec.

    Encode: byte stream -> sign bits at HF mask coordinates -> Hermitian
    inverse FFT -> add to base frame channel.

    Decode: subtract base frame channel from observed frame -> forward FFT
    -> read sign of HF coefficients -> bit stream -> bytes.

    Per audit caveat: BER under uint8 quantization at delta_amp=0.75
    is ~0.42; substrate codec consumers must wrap their own ECC.

    No scorer imports; safe for inflate-time use.
    """

    def __init__(self, cfg: S2sbsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_mod = HfBlindspotMask(cfg)
        # Flat coordinate index for the True positions in the rfft2 mask.
        coords = torch.nonzero(self.mask_mod.mask, as_tuple=False)
        self.register_buffer("coord_kh", coords[:, 0].long())
        self.register_buffer("coord_kw", coords[:, 1].long())

    @property
    def capacity_bits(self) -> int:
        """One bit per Hermitian-symmetric HF coordinate pair (sign-of-real)."""
        return int(self.coord_kh.numel())

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_bits // 8

    def encode(self, base_frame: torch.Tensor, payload: bytes) -> torch.Tensor:
        """Encode payload bytes into the HF band of one channel.

        Args:
            base_frame: ``(H, W)`` float tensor (single channel, ``[0, 1]``)
                or ``(C, H, W)``; the configured channel index is updated.
            payload: bytes to encode. Must satisfy
                ``len(payload) * 8 <= capacity_bits``.

        Returns:
            A new tensor with the same shape as ``base_frame``.
        """
        if base_frame.dim() not in (2, 3):
            raise ValueError("base_frame must be (H, W) or (C, H, W)")
        h = int(self.cfg.output_height)
        w = int(self.cfg.output_width)
        if base_frame.shape[-2:] != (h, w):
            raise ValueError(
                f"base_frame spatial dims must match config ({h}, {w}); got "
                f"{tuple(base_frame.shape[-2:])}"
            )
        capacity_bits = self.capacity_bits
        if len(payload) * 8 > capacity_bits:
            raise ValueError(
                f"payload {len(payload)} bytes exceeds capacity {capacity_bits // 8} bytes"
            )
        device = base_frame.device
        dtype = base_frame.dtype
        # Bit stream: pad with zeros up to capacity.
        bits = torch.zeros(capacity_bits, dtype=torch.float32, device=device)
        for i, byte in enumerate(payload):
            for j in range(8):
                bit = (byte >> (7 - j)) & 1
                bits[i * 8 + j] = 1.0 if bit else 0.0
        # Sign-of-real: bit=1 -> +1, bit=0 -> -1.
        signs = (bits * 2.0 - 1.0).to(dtype)
        delta = float(self.cfg.delta_amp_uint8) / 255.0
        # Place signs at HF coordinates of an empty rfft2 spectrum.
        spec = torch.zeros(h, w // 2 + 1, dtype=torch.complex64, device=device)
        kh = self.coord_kh.to(device)
        kw = self.coord_kw.to(device)
        spec[kh, kw] = (signs * delta).to(torch.complex64)
        # The rfft2 inverse preserves Hermitian symmetry automatically.
        hf_image = torch.fft.irfft2(spec, s=(h, w)).to(dtype)
        if base_frame.dim() == 2:
            return (base_frame + hf_image).clamp(0.0, 1.0)
        out = base_frame.clone()
        out[self.cfg.channel_index] = (out[self.cfg.channel_index] + hf_image).clamp(0.0, 1.0)
        return out

    def decode(self, encoded_frame: torch.Tensor, base_frame: torch.Tensor) -> bytes:
        """Recover payload bytes from an encoded frame given the base.

        Inflate-time consumers WITHOUT the base frame must rely on the
        substrate's archive header to seed a deterministic regeneration of
        the base, then subtract it before decoding.
        """
        if encoded_frame.shape != base_frame.shape:
            raise ValueError("encoded_frame and base_frame shapes must match")
        if encoded_frame.dim() == 3:
            channel = self.cfg.channel_index
            residual = encoded_frame[channel] - base_frame[channel]
        else:
            residual = encoded_frame - base_frame
        spec = torch.fft.rfft2(residual)
        kh = self.coord_kh.to(spec.device)
        kw = self.coord_kw.to(spec.device)
        real_vals = spec[kh, kw].real
        bits = (real_vals > 0.0).to(torch.uint8)
        capacity_bits = int(bits.numel())
        bytes_out = bytearray()
        for i in range(0, capacity_bits // 8 * 8, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(bits[i + j].item())
            bytes_out.append(byte)
        return bytes(bytes_out)


class S2sbsRenderer(nn.Module):
    """Deterministic base renderer + HF byte-stuffing head.

    This is an L0/L1 scaffold: ``base_decoder`` is a tiny deterministic
    placeholder so the archive grammar + inflate contract can be tested
    BEFORE training a real head. Production replaces ``base_decoder``
    with a learned head (e.g. inherit from a frozen base substrate
    archive).
    """

    def __init__(
        self,
        cfg: S2sbsConfig,
        base_decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.codec = HfFftByteCodec(cfg)
        self.base_decoder = base_decoder or _DeterministicBaseDecoder(cfg)

    def base_pair(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_decoder(pair_indices)

    def forward(
        self,
        pair_indices: torch.Tensor,
        payload_by_pair: Sequence[PayloadChannel] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render frame pairs with HF payload stuffing applied."""
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.numel() == 0:
            raise ValueError("pair_indices must be non-empty")
        rgb0, rgb1 = self.base_pair(pair_indices)
        by_idx = {row.pair_index: row.payload for row in payload_by_pair}
        out0, out1 = [], []
        for i, idx in enumerate(pair_indices.tolist()):
            payload = by_idx.get(int(idx), b"")
            if payload:
                out0.append(self.codec.encode(rgb0[i], payload))
                out1.append(self.codec.encode(rgb1[i], payload))
            else:
                out0.append(rgb0[i])
                out1.append(rgb1[i])
        return torch.stack(out0, dim=0), torch.stack(out1, dim=0)


class _DeterministicBaseDecoder(nn.Module):
    """Tiny deterministic base RGB decoder for L0/L1 scaffold tests."""

    def __init__(self, cfg: S2sbsConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = self.cfg.output_height, self.cfg.output_width
        device = pair_indices.device
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        frames0, frames1 = [], []
        seed = float(self.cfg.base_seed)
        for idx in pair_indices.tolist():
            t = float(idx) / max(1.0, float(self.cfg.num_pairs - 1))
            phase = torch.tensor(t * 6.28 + seed * 1e-4, device=device)
            r = (0.40 + 0.20 * gx).clamp(0.0, 1.0)
            g = (0.35 + 0.25 * gy + 0.10 * torch.sin(2.0 * gx + phase)).clamp(0.0, 1.0)
            b = (0.45 + 0.10 * gy * gx).clamp(0.0, 1.0)
            rgb0 = torch.stack([r, g, b], dim=0)
            r2 = (r + 0.02 * torch.cos(phase)).clamp(0.0, 1.0)
            g2 = (g + 0.02 * torch.sin(phase + gx)).clamp(0.0, 1.0)
            b2 = (b - 0.01 * torch.cos(phase + gy)).clamp(0.0, 1.0)
            rgb1 = torch.stack([r2, g2, b2], dim=0)
            frames0.append(rgb0)
            frames1.append(rgb1)
        return torch.stack(frames0, dim=0), torch.stack(frames1, dim=0)


__all__ = [
    "CAMERA_HW",
    "CONTEST_NUM_PAIRS",
    "DEFAULT_DELTA_AMP_UINT8",
    "MAX_DELTA_AMP_UINT8_BEFORE_POSE_DRIFT",
    "SCORER_HW",
    "HfBlindspotMask",
    "HfFftByteCodec",
    "PayloadChannel",
    "S2sbsConfig",
    "S2sbsRenderer",
]
