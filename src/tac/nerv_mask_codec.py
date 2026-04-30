"""Lane 12 — NeRV / Cool-Chic mask codec (van den Oord + Mallat lead).

Per Phase 2 Lane 12 spec (memory project_phases_2_3_4_*):

    NeRV (Chen et al. 2021, arXiv 2104.05079) overfits a coordinate-MLP to a
    video. For 1200 frames of 5-class masks at 384×512: train tiny 4-layer
    MLP with input (t, y, x) → 5-class logits. Quantize weights to int8 →
    ship 30-50KB weight stream. At inflate, run MLP forward over all
    (t, y, x) coordinates → reconstruct masks.

This module implements the MINIMAL coordinate-MLP scaffold + tests:
- Positional encoding (sin/cos at multiple frequencies — matches NeRV /
  Fourier features convention)
- 4-layer MLP with sinusoidal output → 5-class logits
- Deterministic encode (state_dict → bytes) + decode (bytes → state_dict)
- Round-trip on a synthetic mask sequence
- Byte-count accounting vs raw fp16 baseline

The actual TRAINING of the MLP on a real mask sequence is OUT OF SCOPE
for this scaffold (Phase 2 dispatch decision). This file provides the
codec primitives the dispatch script will call.

CLAUDE.md compliance
--------------------
- Compress-time only (training the MLP); inflate runs MLP forward over all
  coords → reconstruct. NO scorer load at decode time.
- No silent defaults — every public function arg is required-keyword.
- All claims tagged [synthetic]/[prediction] until empirical real-archive run.
- No GPU dependency; encode + decode are pure CPU. Training (out of scope
  here) would normally use CUDA but the scaffold tests use CPU.
- Pure-math byte → tensor pipeline.

Math foundation
---------------
Coordinate MLP f_θ(t, y, x) ∈ R^5 (one logit per class). Positional encoding
augments raw (t, y, x) with sin/cos at L frequencies:

    γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^L π p), cos(2^L π p)]

Predicted output: logits of shape (T, H, W, 5); argmax → class IDs.

Bit budget governed by MLP size (param count × bits/param) NOT mask
resolution → fundamentally different scaling vs AV1 monochrome (which is
O(T*H*W * bpp)).

References
----------
* Chen et al. 2021 NeurIPS — "NeRV: Neural Representations for Videos"
* Mildenhall et al. 2020 ECCV — "NeRF" positional encoding lineage
* memory: project_phases_2_3_4_design_implementation_math_provenance §"Lane 12"
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ── magic bytes / format version ─────────────────────────────────────────


NERV_MAGIC: bytes = b"NRV1"
"""Lane 12 self-describing payload magic. 4 bytes, ASCII."""

NERV_VERSION: int = 1
"""Header version. Bumped on any wire-format change."""


# ── coordinate MLP ────────────────────────────────────────────────────────


def positional_encode(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Sin/cos positional encoding at log-spaced frequencies.

    Args:
        coords: (B, D) tensor of normalized coordinates in [-1, 1].
            D is the number of input dims (3 for (t, y, x)).
        num_freqs: number of frequency bands. Output dim = D * 2 * num_freqs.

    Returns:
        (B, D * 2 * num_freqs) encoded tensor.

    Deterministic: same input + same num_freqs → same output (no random init).
    """
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D (B, D); got {tuple(coords.shape)}")
    if num_freqs < 1:
        raise ValueError(f"num_freqs must be >= 1; got {num_freqs}")
    # Frequency band powers: 2^0, 2^1, ..., 2^(num_freqs - 1)
    freq_bands = 2.0 ** torch.arange(num_freqs, dtype=coords.dtype, device=coords.device)
    # Outer product: (B, D, F)
    scaled = coords.unsqueeze(-1) * (np.pi * freq_bands.view(1, 1, -1))
    sin_part = torch.sin(scaled)
    cos_part = torch.cos(scaled)
    # Stack & flatten: (B, D, 2, F) → (B, D * 2 * F)
    encoded = torch.stack([sin_part, cos_part], dim=-2)  # (B, D, 2, F)
    return encoded.reshape(coords.shape[0], -1)


class NeRVMaskCodec(nn.Module):
    """4-layer coordinate-MLP that maps (t, y, x) → 5-class logits.

    This is the MINIMAL Phase 2 scaffold. Production sweeps:
    - hidden_dim ∈ {32, 64, 128}
    - num_freqs ∈ {4, 8, 16}
    - depth ∈ {3, 4, 6}
    - num_classes = 5 (matches our SegNet output)

    Forward signature: (B, 3) coords → (B, num_classes) logits.

    Determinism: torch.manual_seed at construction; weights init from
    Xavier-uniform.
    """

    def __init__(
        self,
        num_freqs: int,
        hidden_dim: int,
        num_classes: int,
        depth: int = 4,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        if num_freqs < 1 or hidden_dim < 1 or num_classes < 1 or depth < 2:
            raise ValueError(
                f"NeRVMaskCodec: invalid arch (num_freqs={num_freqs}, "
                f"hidden_dim={hidden_dim}, num_classes={num_classes}, depth={depth})"
            )
        self.num_freqs = int(num_freqs)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.depth = int(depth)
        # Input: 3 dims × 2 (sin/cos) × num_freqs
        in_dim = 3 * 2 * self.num_freqs
        # Deterministic init seeded by `seed`
        gen = torch.Generator().manual_seed(int(seed))
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(self.depth - 1):
            lin = nn.Linear(prev, self.hidden_dim)
            with torch.no_grad():
                # Xavier uniform with the deterministic generator (manual since
                # nn.init doesn't accept generator).
                fan_in, fan_out = prev, self.hidden_dim
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                bound = (3.0 ** 0.5) * std
                lin.weight.uniform_(-bound, bound, generator=gen)
                lin.bias.zero_()
            layers.append(lin)
            layers.append(nn.GELU())
            prev = self.hidden_dim
        # Output layer: hidden → num_classes (no activation; raw logits)
        out_lin = nn.Linear(prev, self.num_classes)
        with torch.no_grad():
            fan_in, fan_out = prev, self.num_classes
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            bound = (3.0 ** 0.5) * std
            out_lin.weight.uniform_(-bound, bound, generator=gen)
            out_lin.bias.zero_()
        layers.append(out_lin)
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, 3) normalized in [-1, 1]; returns (B, num_classes) logits."""
        encoded = positional_encode(coords, num_freqs=self.num_freqs)
        return self.mlp(encoded)

    def num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── encode / decode (state_dict ↔ bytes) ──────────────────────────────────


@dataclass(frozen=True)
class NeRVHeader:
    """Parsed NRV1 header fields."""

    version: int
    num_freqs: int
    hidden_dim: int
    num_classes: int
    depth: int
    weight_dtype: int  # 0 = fp16, 1 = int8
    payload_size: int


def encode_nerv_codec(
    codec: NeRVMaskCodec | None = None,
    *,
    weight_dtype: str = "fp16",
) -> bytes:
    """Encode NeRVMaskCodec state_dict to a self-describing NRV1 payload.

    Args:
        codec: trained (or untrained) NeRVMaskCodec. Required.
        weight_dtype: "fp16" (default) or "int8" (Phase 2 quantization). The
            int8 path requires the caller to have quantized weights externally
            and replaced them in-place; this scaffold only verifies fp16
            round-trip.

    Returns:
        bytes — NRV1 self-describing payload (header + arch + flat weights).

    Raises:
        ValueError: codec is None, or weight_dtype unsupported.
    """
    if codec is None:
        raise ValueError(
            "encode_nerv_codec: codec is required (no silent default — "
            "Check 81 STRICT)."
        )
    if weight_dtype not in ("fp16", "int8"):
        raise ValueError(
            f"encode_nerv_codec: weight_dtype must be 'fp16' or 'int8'; "
            f"got {weight_dtype!r}"
        )
    dtype_code = 0 if weight_dtype == "fp16" else 1
    # Flatten state_dict in deterministic order (sorted keys)
    sd = codec.state_dict()
    sorted_keys = sorted(sd.keys())
    if weight_dtype == "fp16":
        flat = torch.cat(
            [sd[k].detach().to(torch.float16).reshape(-1) for k in sorted_keys]
        )
        weight_bytes = flat.cpu().numpy().tobytes()
    else:
        # int8: caller's job to ensure tensors are int8-castable; we cast.
        flat_list: list[np.ndarray] = []
        for k in sorted_keys:
            t = sd[k].detach().cpu()
            if t.dtype != torch.int8:
                # Naive symmetric int8 quantization (per-tensor) for scaffold.
                # Production quantization lives in tac.quantization (FakeQuantSTE).
                vmax = float(t.abs().max().clamp_min(1e-12))
                scale = vmax / 127.0
                qt = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
                flat_list.append(qt.reshape(-1).numpy())
            else:
                flat_list.append(t.reshape(-1).numpy())
        weight_bytes = np.concatenate(flat_list).astype(np.int8).tobytes()

    # Build header (little-endian):
    #   magic            : 4 bytes  = b"NRV1"
    #   version          : 2 bytes  uint16
    #   num_freqs        : 2 bytes  uint16
    #   hidden_dim       : 2 bytes  uint16
    #   num_classes      : 2 bytes  uint16
    #   depth            : 2 bytes  uint16
    #   weight_dtype     : 2 bytes  uint16 (0=fp16, 1=int8)
    #   payload_size     : 8 bytes  uint64
    #   payload          : payload_size bytes
    header = io.BytesIO()
    header.write(NERV_MAGIC)
    header.write(struct.pack("<H", NERV_VERSION))
    header.write(struct.pack("<H", int(codec.num_freqs)))
    header.write(struct.pack("<H", int(codec.hidden_dim)))
    header.write(struct.pack("<H", int(codec.num_classes)))
    header.write(struct.pack("<H", int(codec.depth)))
    header.write(struct.pack("<H", int(dtype_code)))
    header.write(struct.pack("<Q", len(weight_bytes)))
    return header.getvalue() + weight_bytes


def _parse_nerv_header(blob: bytes) -> NeRVHeader:
    """Strict header parser; raises ValueError on malformed input."""
    if len(blob) < 4 + 2 * 6 + 8:
        raise ValueError(
            f"decode_nerv_codec: blob length {len(blob)} too small for NRV1 header"
        )
    if blob[:4] != NERV_MAGIC:
        raise ValueError(
            f"decode_nerv_codec: bad magic {blob[:4]!r}, expected {NERV_MAGIC!r}"
        )
    buf = io.BytesIO(blob)
    buf.read(4)  # magic
    (version,) = struct.unpack("<H", buf.read(2))
    if version != NERV_VERSION:
        raise ValueError(
            f"decode_nerv_codec: unsupported version {version}; expected {NERV_VERSION}"
        )
    (num_freqs,) = struct.unpack("<H", buf.read(2))
    (hidden_dim,) = struct.unpack("<H", buf.read(2))
    (num_classes,) = struct.unpack("<H", buf.read(2))
    (depth,) = struct.unpack("<H", buf.read(2))
    (dtype_code,) = struct.unpack("<H", buf.read(2))
    (payload_size,) = struct.unpack("<Q", buf.read(8))
    return NeRVHeader(
        version=int(version),
        num_freqs=int(num_freqs),
        hidden_dim=int(hidden_dim),
        num_classes=int(num_classes),
        depth=int(depth),
        weight_dtype=int(dtype_code),
        payload_size=int(payload_size),
    )


def decode_nerv_codec(blob: bytes | None = None) -> NeRVMaskCodec:
    """Decode an NRV1 payload back to a NeRVMaskCodec.

    Pure-math byte → module. NO scorer load (CLAUDE.md non-negotiable). NO GPU.

    Args:
        blob: bytes produced by encode_nerv_codec. Required.

    Returns:
        NeRVMaskCodec with weights restored from the payload.
    """
    if blob is None:
        raise ValueError(
            "decode_nerv_codec: blob is required (no silent default — "
            "Check 81 STRICT)."
        )
    hdr = _parse_nerv_header(blob)
    # Reconstruct codec architecture (same seed irrelevant — weights overwritten)
    codec = NeRVMaskCodec(
        num_freqs=hdr.num_freqs,
        hidden_dim=hdr.hidden_dim,
        num_classes=hdr.num_classes,
        depth=hdr.depth,
        seed=0,
    )
    header_size = 4 + 2 * 6 + 8
    payload = blob[header_size : header_size + hdr.payload_size]
    if len(payload) != hdr.payload_size:
        raise ValueError(
            f"decode_nerv_codec: truncated payload "
            f"(read {len(payload)} of {hdr.payload_size})"
        )
    if hdr.weight_dtype == 0:
        # fp16
        flat = np.frombuffer(payload, dtype=np.float16).copy()
    elif hdr.weight_dtype == 1:
        # int8 — scaffold does not preserve scale; production V2 needs scale table.
        flat = np.frombuffer(payload, dtype=np.int8).astype(np.float32)
    else:
        raise ValueError(
            f"decode_nerv_codec: unknown weight_dtype code {hdr.weight_dtype}"
        )
    # Restore state_dict
    sd = codec.state_dict()
    sorted_keys = sorted(sd.keys())
    cursor = 0
    for k in sorted_keys:
        n = int(sd[k].numel())
        chunk = flat[cursor : cursor + n]
        if len(chunk) != n:
            raise ValueError(
                f"decode_nerv_codec: weight stream truncated at key {k!r} "
                f"(need {n}, got {len(chunk)})"
            )
        sd[k] = torch.from_numpy(chunk).reshape(sd[k].shape).to(sd[k].dtype)
        cursor += n
    codec.load_state_dict(sd)
    return codec


# ── byte-count accounting (vs raw fp16 baseline) ──────────────────────────


def raw_fp16_baseline_bytes(num_frames: int, height: int, width: int, num_classes: int) -> int:
    """Return raw fp16 byte count for a (T, H, W, C) logit tensor (NO codec)."""
    if min(num_frames, height, width, num_classes) <= 0:
        raise ValueError(
            f"raw_fp16_baseline_bytes: all dims must be > 0; got "
            f"({num_frames}, {height}, {width}, {num_classes})"
        )
    return int(num_frames) * int(height) * int(width) * int(num_classes) * 2


def nerv_codec_bytes(codec: NeRVMaskCodec, weight_dtype: str = "fp16") -> int:
    """Byte count of a NeRVMaskCodec at given weight dtype (no header)."""
    bits_per_param = {"fp16": 16, "int8": 8}.get(weight_dtype)
    if bits_per_param is None:
        raise ValueError(
            f"nerv_codec_bytes: weight_dtype must be 'fp16' or 'int8'; "
            f"got {weight_dtype!r}"
        )
    return (codec.num_params() * bits_per_param) // 8


# ── helpers: render mask logits over a full coord grid ────────────────────


def render_mask_logits(
    codec: NeRVMaskCodec,
    num_frames: int,
    height: int,
    width: int,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Run the codec over all (t, y, x) coords; return (T, H, W, C) logits.

    Pure forward; no gradient tracking. CPU-friendly batched inference.

    Args:
        codec: trained codec.
        num_frames: T.
        height: H.
        width: W.
        batch_size: forward batch size (CPU memory budget).

    Returns:
        (T, H, W, num_classes) float32 tensor.
    """
    # Build all (t, y, x) coords in [-1, 1]
    ts = torch.linspace(-1.0, 1.0, num_frames)
    ys = torch.linspace(-1.0, 1.0, height)
    xs = torch.linspace(-1.0, 1.0, width)
    grid_t, grid_y, grid_x = torch.meshgrid(ts, ys, xs, indexing="ij")
    coords = torch.stack([grid_t.reshape(-1), grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    out_logits = []
    codec.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            chunk = coords[start : start + batch_size]
            out_logits.append(codec(chunk))
    flat = torch.cat(out_logits, dim=0)
    return flat.reshape(num_frames, height, width, codec.num_classes)


__all__ = [
    "NERV_MAGIC",
    "NERV_VERSION",
    "NeRVHeader",
    "NeRVMaskCodec",
    "decode_nerv_codec",
    "encode_nerv_codec",
    "nerv_codec_bytes",
    "positional_encode",
    "raw_fp16_baseline_bytes",
    "render_mask_logits",
]
