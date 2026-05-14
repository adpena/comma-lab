# SPDX-License-Identifier: MIT
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
"""Lane 12 self-describing payload magic. 4 bytes, ASCII.

Both NRV1 (v1) and NRV2 (v2) payloads start with the SAME 4 magic bytes;
the version u16 immediately after disambiguates. Inflate magic-byte sniff
matches NERV_MAGIC and then dispatches on the version field.
"""

NERV_VERSION: int = 2
"""Header version.

- v1 (legacy): fp16 OR int8 weights with NO per-tensor scale table. The
  int8 v1 path was scaffold-only — decoder cannot reproduce float numerics
  because the scale was discarded at encode time. v1 fp16 round-trip works.
- v2 (current, Lane 12 production): fp16 OR int8 weights, where int8 ALSO
  ships a per-tensor float32 scale table immediately AFTER the weights,
  one float32 per state-dict key in sorted order. Decoder dequantizes
  ``q * scale`` per key. v2 fp16 ⇄ v1 fp16 are wire-equivalent
  (the scale-table block is empty when ``weight_dtype=fp16``).

Bumped 2026-04-30 (Phase C) when the trainer + integration landed; the
broken-int8 v1 path is preserved for backwards compatibility but new
encoders default to v2.
"""


# ── coordinate MLP ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NeRVSamplingComponent:
    """One deterministic coordinate-sampling component for NeRV training."""

    name: str
    weight: float
    flat_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class NeRVSamplingPool:
    """Weighted coordinate-sampling pool.

    ``flat_indices=None`` means sample uniformly from the full mask tensor.
    Non-empty index tensors sample uniformly within that component. Component
    selection is weighted and driven by the trainer's CPU RNG, so repeated runs
    with the same seed are deterministic.
    """

    components: tuple[NeRVSamplingComponent, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("NeRVSamplingPool requires at least one component")
        for component in self.components:
            if float(component.weight) <= 0.0:
                raise ValueError(
                    f"sampling component {component.name!r} weight must be > 0"
                )
            if component.flat_indices is not None:
                if component.flat_indices.ndim != 1:
                    raise ValueError(
                        f"sampling component {component.name!r} indices must be 1-D"
                    )
                if component.flat_indices.numel() == 0:
                    raise ValueError(
                        f"sampling component {component.name!r} indices are empty"
                    )


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
    """Parsed NRV1/NRV2 header fields.

    For v1: ``scale_table_size`` is always 0 (legacy fp16-only or
    broken-scaffold-int8). For v2: ``scale_table_size`` is
    ``num_state_dict_keys * 4`` (one float32 per key, byte order little
    endian) when ``weight_dtype == 1`` (int8), else 0 (fp16).
    """

    version: int
    num_freqs: int
    hidden_dim: int
    num_classes: int
    depth: int
    weight_dtype: int  # 0 = fp16, 1 = int8
    payload_size: int
    scale_table_size: int = 0


def encode_nerv_codec(
    codec: NeRVMaskCodec | None = None,
    *,
    weight_dtype: str = "fp16",
    version: int | None = None,
) -> bytes:
    """Encode NeRVMaskCodec state_dict to a self-describing NRV1/NRV2 payload.

    Args:
        codec: trained (or untrained) NeRVMaskCodec. Required.
        weight_dtype: "fp16" (default) or "int8". For int8 in v2, a per-tensor
            symmetric scale table is appended after the int8 weight payload
            so the decoder can reproduce float numerics.
        version: explicit format version. ``None`` → uses ``NERV_VERSION``
            (currently 2). Pass ``version=1`` for the legacy wire format
            (no scale table — int8 NOT round-trippable in v1).

    Returns:
        bytes — self-describing payload.

        - v1 layout: ``[magic 4B][hdr 18B][weights]`` — total 22B + weights.
        - v2 layout: ``[magic 4B][hdr 26B][weights][scale_table?]`` — total
          30B + weights + (4 × num_keys if int8 else 0).

    Raises:
        ValueError: codec is None, weight_dtype unsupported, or unsupported
            version requested.
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
    use_version = NERV_VERSION if version is None else int(version)
    if use_version not in (1, 2):
        raise ValueError(
            f"encode_nerv_codec: version must be 1 or 2; got {use_version}"
        )
    dtype_code = 0 if weight_dtype == "fp16" else 1
    # Flatten state_dict in deterministic order (sorted keys)
    sd = codec.state_dict()
    sorted_keys = sorted(sd.keys())
    scale_bytes = b""
    if weight_dtype == "fp16":
        flat = torch.cat(
            [sd[k].detach().to(torch.float16).reshape(-1) for k in sorted_keys]
        )
        weight_bytes = flat.cpu().numpy().tobytes()
    else:
        # int8: per-tensor symmetric quantization. v2 ships scale table so
        # the decoder can reproduce float numerics. v1 keeps the legacy
        # broken-scaffold behaviour (no scale persisted) for backwards
        # compatibility with any pre-2026-04-30 payloads.
        flat_list: list[np.ndarray] = []
        scale_list: list[float] = []
        for k in sorted_keys:
            t = sd[k].detach().cpu()
            if t.dtype != torch.int8:
                vmax = float(t.abs().max().clamp_min(1e-12))
                scale = vmax / 127.0
                qt = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
                flat_list.append(qt.reshape(-1).numpy())
                scale_list.append(scale)
            else:
                # Already-quantized tensor: use unit scale (caller's
                # responsibility to have applied a known scale).
                flat_list.append(t.reshape(-1).numpy())
                scale_list.append(1.0)
        weight_bytes = np.concatenate(flat_list).astype(np.int8).tobytes()
        if use_version == 2:
            scale_bytes = np.asarray(scale_list, dtype=np.float32).tobytes()
        # v1 + int8 → scale_bytes left empty (legacy broken-scaffold path)

    # Header layout (little-endian):
    #   magic            : 4 bytes  = b"NRV1"   (same across v1/v2)
    #   version          : 2 bytes  uint16     (1 or 2)
    #   num_freqs        : 2 bytes  uint16
    #   hidden_dim       : 2 bytes  uint16
    #   num_classes      : 2 bytes  uint16
    #   depth            : 2 bytes  uint16
    #   weight_dtype     : 2 bytes  uint16     (0=fp16, 1=int8)
    #   payload_size     : 8 bytes  uint64     (weight bytes only, NOT scale)
    #   scale_table_size : 8 bytes  uint64     [v2 only]
    #   payload          : payload_size bytes
    #   scale_table      : scale_table_size bytes [v2 + int8 only]
    header = io.BytesIO()
    header.write(NERV_MAGIC)
    header.write(struct.pack("<H", use_version))
    header.write(struct.pack("<H", int(codec.num_freqs)))
    header.write(struct.pack("<H", int(codec.hidden_dim)))
    header.write(struct.pack("<H", int(codec.num_classes)))
    header.write(struct.pack("<H", int(codec.depth)))
    header.write(struct.pack("<H", int(dtype_code)))
    header.write(struct.pack("<Q", len(weight_bytes)))
    if use_version == 2:
        header.write(struct.pack("<Q", len(scale_bytes)))
    return header.getvalue() + weight_bytes + scale_bytes


def _parse_nerv_header(blob: bytes) -> NeRVHeader:
    """Strict header parser; raises ValueError on malformed input.

    Accepts both NRV1 (v1, no scale table) and NRV2 (v2, with scale table)
    layouts. The version field disambiguates.
    """
    # v1 minimum size: 4 (magic) + 6*2 (uint16s) + 8 (payload_size) = 24
    if len(blob) < 4 + 2 * 6 + 8:
        raise ValueError(
            f"decode_nerv_codec: blob length {len(blob)} too small for NRV header"
        )
    if blob[:4] != NERV_MAGIC:
        raise ValueError(
            f"decode_nerv_codec: bad magic {blob[:4]!r}, expected {NERV_MAGIC!r}"
        )
    buf = io.BytesIO(blob)
    buf.read(4)  # magic
    (version,) = struct.unpack("<H", buf.read(2))
    if version not in (1, 2):
        raise ValueError(
            f"decode_nerv_codec: unsupported version {version}; "
            f"expected 1 or 2"
        )
    (num_freqs,) = struct.unpack("<H", buf.read(2))
    (hidden_dim,) = struct.unpack("<H", buf.read(2))
    (num_classes,) = struct.unpack("<H", buf.read(2))
    (depth,) = struct.unpack("<H", buf.read(2))
    (dtype_code,) = struct.unpack("<H", buf.read(2))
    (payload_size,) = struct.unpack("<Q", buf.read(8))
    scale_table_size = 0
    if version == 2:
        # v2 header includes a uint64 scale_table_size right after payload_size
        if len(blob) < 4 + 2 * 6 + 8 + 8:
            raise ValueError(
                f"decode_nerv_codec: blob length {len(blob)} too small for NRV2 header"
            )
        (scale_table_size,) = struct.unpack("<Q", buf.read(8))
    return NeRVHeader(
        version=int(version),
        num_freqs=int(num_freqs),
        hidden_dim=int(hidden_dim),
        num_classes=int(num_classes),
        depth=int(depth),
        weight_dtype=int(dtype_code),
        payload_size=int(payload_size),
        scale_table_size=int(scale_table_size),
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
    # Header size depends on version
    header_size = 4 + 2 * 6 + 8 + (8 if hdr.version == 2 else 0)
    payload = blob[header_size : header_size + hdr.payload_size]
    if len(payload) != hdr.payload_size:
        raise ValueError(
            f"decode_nerv_codec: truncated payload "
            f"(read {len(payload)} of {hdr.payload_size})"
        )
    sd = codec.state_dict()
    sorted_keys = sorted(sd.keys())
    # Read scale table (v2 + int8 only)
    scales: list[float] | None = None
    if hdr.version == 2 and hdr.scale_table_size > 0:
        scale_off = header_size + hdr.payload_size
        scale_blob = blob[scale_off : scale_off + hdr.scale_table_size]
        if len(scale_blob) != hdr.scale_table_size:
            raise ValueError(
                f"decode_nerv_codec: truncated scale table "
                f"(read {len(scale_blob)} of {hdr.scale_table_size})"
            )
        scales_arr = np.frombuffer(scale_blob, dtype=np.float32)
        if len(scales_arr) != len(sorted_keys):
            raise ValueError(
                f"decode_nerv_codec: scale table has {len(scales_arr)} entries "
                f"but codec has {len(sorted_keys)} state-dict keys"
            )
        scales = [float(s) for s in scales_arr]

    if hdr.weight_dtype == 0:
        # fp16 — direct cast
        flat = np.frombuffer(payload, dtype=np.float16).copy()
        cursor = 0
        for k in sorted_keys:
            n = int(sd[k].numel())
            chunk = flat[cursor : cursor + n]
            if len(chunk) != n:
                raise ValueError(
                    f"decode_nerv_codec: weight stream truncated at key {k!r} "
                    f"(need {n}, got {len(chunk)})"
                )
            sd[k] = torch.from_numpy(chunk.astype(np.float32)).reshape(sd[k].shape).to(sd[k].dtype)
            cursor += n
    elif hdr.weight_dtype == 1:
        # int8 — multiply by per-tensor scale (v2). v1 had no scale, so the
        # decoded floats are int8 codes cast to float (broken-scaffold v1).
        flat = np.frombuffer(payload, dtype=np.int8).astype(np.float32)
        cursor = 0
        for i, k in enumerate(sorted_keys):
            n = int(sd[k].numel())
            chunk = flat[cursor : cursor + n]
            if len(chunk) != n:
                raise ValueError(
                    f"decode_nerv_codec: weight stream truncated at key {k!r} "
                    f"(need {n}, got {len(chunk)})"
                )
            if scales is not None:
                chunk = chunk * scales[i]
            sd[k] = torch.from_numpy(chunk.copy()).reshape(sd[k].shape).to(sd[k].dtype)
            cursor += n
    else:
        raise ValueError(
            f"decode_nerv_codec: unknown weight_dtype code {hdr.weight_dtype}"
        )
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


# ── Trainer ───────────────────────────────────────────────────────────────


class NeRVMaskTrainer:
    """Per-sequence overfit trainer for ``NeRVMaskCodec``.

    Implements the CLAUDE.md non-negotiables for any training path:

    * ``device != "mps"`` — refused at construction (PoseNet drift 23x;
      MPS is FORBIDDEN for any kill/promote decision per CLAUDE.md
      "MPS auth eval is NOISE — NON-NEGOTIABLE").
    * ``EMA(model, decay=0.997)`` instantiated at construction and updated
      after every ``optimizer.step()`` (CLAUDE.md "EMA — NON-NEGOTIABLE").
      The EMA shadow is what ``encode()`` ships, NOT the live weights.
    * ``eval_roundtrip``-aware loss measurement: cross-entropy on raw 5-class
      logits vs argmax labels (NO ``.round()`` anywhere — argmax has zero
      gradient per Council A `.round()` zero-gradient bug class). The
      scorer-side eval roundtrip (``384 → 874 → uint8 → 384``) does not
      apply at the mask-CODEC layer; it applies at the auth-eval stage
      after decode → render → mask_video → SegNet, which is delegated to
      ``experiments/contest_auth_eval.py``.
    * Snapshot+restore at eval: ``ema.apply(model)`` is called inside a
      try/finally that restores the live state-dict after evaluation, so
      training continues from un-shadowed weights.
    * CUDA-required default; explicit opt-in to CPU for unit tests via
      ``device="cpu"`` (allowed only because per-sequence overfit is
      deterministic; CUDA still required for the production dispatch).

    Args:
        codec: instantiated NeRVMaskCodec on ``device``.
        device: "cuda" (production) or "cpu" (unit tests). MPS refused.
        learning_rate: Adam lr. Default 1e-3 per Phase B council verdict.
        ema_decay: EMA decay. Default 0.997 (CLAUDE.md non-negotiable).
        seed: RNG seed for batch sampling. Default 2026.

    Raises:
        ValueError: device is MPS, codec is None, ema_decay outside (0, 1).
    """

    def __init__(
        self,
        codec: "NeRVMaskCodec | None" = None,
        *,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        ema_decay: float = 0.997,
        seed: int = 2026,
    ) -> None:
        if codec is None:
            raise ValueError(
                "NeRVMaskTrainer: codec is required (no silent default)."
            )
        device_str = str(device)
        if device_str.startswith("mps"):
            raise ValueError(
                "NeRVMaskTrainer refuses device='mps'. MPS auth-eval drifts "
                "23x on PoseNet (CLAUDE.md non-negotiable; verified "
                "2026-04-25). CUDA only for production; CPU allowed for "
                "unit tests."
            )
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(
                f"NeRVMaskTrainer: ema_decay must be in (0, 1); got {ema_decay}"
            )
        self.codec = codec
        self.device = torch.device(device_str)
        self.codec.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.codec.parameters(), lr=float(learning_rate), betas=(0.9, 0.999)
        )
        self.seed = int(seed)
        # Canonical EMA per tac.training.EMA. Imported lazily so this module
        # does not depend on the heavy `training.py` import chain unless a
        # trainer is actually constructed.
        from tac.training import EMA

        self.ema = EMA(self.codec, decay=float(ema_decay))
        self._step = 0
        self._rng = torch.Generator(device="cpu").manual_seed(self.seed)

    # ── private: build coordinate grid for one frame (for batched sampling) ──

    @staticmethod
    def _normalize_coord(idx: torch.Tensor, size: int) -> torch.Tensor:
        """Map integer indices [0, size) → float coords in [-1, 1]."""
        if size <= 1:
            return torch.zeros_like(idx, dtype=torch.float32)
        return (idx.float() / float(size - 1)) * 2.0 - 1.0

    def _sample_batch(
        self,
        masks_THW: torch.Tensor,
        batch_size: int,
        sampling_pool: NeRVSamplingPool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (coords, labels) from a (T, H, W) integer mask tensor.

        Returns:
            coords: (batch_size, 3) float on self.device
            labels: (batch_size,) long on self.device
        """
        T, H, W = masks_THW.shape
        # Sample integer indices on CPU (deterministic generator) then move.
        flat_size = T * H * W
        if sampling_pool is None:
            flat_idx = torch.randint(0, flat_size, (batch_size,), generator=self._rng)
        else:
            component_weights = torch.tensor(
                [float(component.weight) for component in sampling_pool.components],
                dtype=torch.float64,
            )
            component_draws = torch.multinomial(
                component_weights,
                int(batch_size),
                replacement=True,
                generator=self._rng,
            )
            flat_idx = torch.empty(int(batch_size), dtype=torch.long)
            for component_index, component in enumerate(sampling_pool.components):
                mask = component_draws == int(component_index)
                n_draws = int(mask.sum().item())
                if n_draws == 0:
                    continue
                if component.flat_indices is None:
                    flat_idx[mask] = torch.randint(
                        0,
                        flat_size,
                        (n_draws,),
                        generator=self._rng,
                    )
                    continue
                component_indices = component.flat_indices.detach().cpu().long()
                min_index = int(component_indices.min().item())
                max_index = int(component_indices.max().item())
                if min_index < 0 or max_index >= flat_size:
                    raise ValueError(
                        f"sampling component {component.name!r} contains flat "
                        f"indices outside [0, {flat_size})"
                    )
                source_draws = torch.randint(
                    0,
                    int(component_indices.numel()),
                    (n_draws,),
                    generator=self._rng,
                )
                flat_idx[mask] = component_indices[source_draws]
        t_idx = flat_idx // (H * W)
        rem = flat_idx % (H * W)
        y_idx = rem // W
        x_idx = rem % W
        labels = masks_THW.reshape(-1)[flat_idx].long().to(self.device)
        coords = torch.stack(
            [
                self._normalize_coord(t_idx, T),
                self._normalize_coord(y_idx, H),
                self._normalize_coord(x_idx, W),
            ],
            dim=-1,
        ).to(self.device)
        return coords, labels

    # ── public: train one step ─────────────────────────────────────────────

    def step(
        self,
        masks_THW: torch.Tensor,
        batch_size: int = 4096,
        sampling_pool: NeRVSamplingPool | None = None,
    ) -> dict[str, float]:
        """Single SGD step on a uniform-coordinate batch of (T, H, W) masks.

        Args:
            masks_THW: (T, H, W) integer mask tensor on CPU (long/uint8).
            batch_size: number of (t, y, x) coords sampled per step.
            sampling_pool: optional weighted sampling pool. When omitted,
                sampling remains uniform over the full tensor.

        Returns:
            dict with keys ``loss`` (cross-entropy) and ``acc`` (argmax-match
            rate, in [0, 1]).
        """
        if masks_THW.ndim != 3:
            raise ValueError(
                f"NeRVMaskTrainer.step: masks must be (T, H, W); "
                f"got shape {tuple(masks_THW.shape)}"
            )
        self.codec.train()
        coords, labels = self._sample_batch(
            masks_THW,
            batch_size=batch_size,
            sampling_pool=sampling_pool,
        )
        logits = self.codec(coords)  # (B, num_classes)
        # Cross-entropy on raw logits — gradient flows through softmax. NO
        # `.round()` anywhere in the forward chain (argmax used only for
        # diagnostic acc; not differentiated).
        import torch.nn.functional as _F

        loss = _F.cross_entropy(logits, labels, reduction="mean")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        # EMA update AFTER optimizer.step() — canonical pattern (CLAUDE.md
        # "EMA — NON-NEGOTIABLE").
        self.ema.update(self.codec)
        self._step += 1
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
        return {"loss": float(loss.item()), "acc": float(acc)}

    # ── public: evaluate (with EMA snapshot+restore) ───────────────────────

    def evaluate_argmax_disagreement(
        self,
        masks_THW: torch.Tensor,
        *,
        batch_size: int = 16384,
    ) -> dict[str, float]:
        """Compute pixel-wise argmax disagreement using EMA shadow weights.

        Snapshots the live state, applies EMA shadow, runs a full-grid
        forward, computes argmax-vs-label disagreement, restores live
        state. Matches the CLAUDE.md canonical EMA snapshot+restore pattern
        (copied from ``experiments/train_distill.py``).

        Args:
            masks_THW: (T, H, W) integer mask tensor on CPU.
            batch_size: forward batch size.

        Returns:
            dict with keys ``disagreement_rate`` (fraction of pixels where
            argmax(logits) != label, in [0, 1]) and ``num_pixels``.
        """
        T, H, W = masks_THW.shape
        # Snapshot live state
        live_state = {
            k: v.detach().clone() for k, v in self.codec.state_dict().items()
        }
        try:
            self.ema.apply(self.codec)
            self.codec.eval()
            disagree = 0
            total = T * H * W
            # Stream through coords in batches to bound memory
            with torch.no_grad():
                # Build coord grid once (CPU); ship batches to device
                ts = torch.linspace(-1.0, 1.0, T)
                ys = torch.linspace(-1.0, 1.0, H)
                xs = torch.linspace(-1.0, 1.0, W)
                grid_t, grid_y, grid_x = torch.meshgrid(ts, ys, xs, indexing="ij")
                coords_all = torch.stack(
                    [grid_t.reshape(-1), grid_y.reshape(-1), grid_x.reshape(-1)],
                    dim=-1,
                )
                labels_all = masks_THW.reshape(-1).long()
                for start in range(0, total, batch_size):
                    chunk_coords = coords_all[start : start + batch_size].to(self.device)
                    chunk_labels = labels_all[start : start + batch_size].to(self.device)
                    chunk_logits = self.codec(chunk_coords)
                    chunk_pred = chunk_logits.argmax(dim=-1)
                    disagree += int((chunk_pred != chunk_labels).sum().item())
        finally:
            # Restore live weights — training resumes from un-shadowed state
            self.codec.load_state_dict(live_state)
            self.codec.train()
        return {
            "disagreement_rate": float(disagree) / float(total),
            "num_pixels": int(total),
        }

    # ── public: ship EMA shadow as the inference codec ─────────────────────

    def encode(self, *, weight_dtype: str = "fp16") -> bytes:
        """Encode the EMA shadow weights as an NRV2 payload.

        This is the production "ship-time" call. Per CLAUDE.md non-negotiable
        ("Inference / archive bytes come from ``ema.state_dict()``"), the
        EMA shadow is encoded — NOT the live weights.

        Args:
            weight_dtype: "fp16" (default; ~23 KB at hidden=64) or "int8"
                (~12 KB; v2 ships scale table).

        Returns:
            bytes — NRV2 self-describing payload.
        """
        # Snapshot live; apply EMA; encode; restore live.
        live_state = {
            k: v.detach().clone() for k, v in self.codec.state_dict().items()
        }
        try:
            self.ema.apply(self.codec)
            blob = encode_nerv_codec(self.codec, weight_dtype=weight_dtype)
        finally:
            self.codec.load_state_dict(live_state)
        return blob


# ── helpers: render argmax mask sequence (decode + grid forward) ──────────


def render_mask_argmax(
    codec: NeRVMaskCodec,
    num_frames: int,
    height: int,
    width: int,
    batch_size: int = 16384,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Run the codec over all (t, y, x) coords; return (T, H, W) argmax mask.

    Inflate-time helper: accepts a decoded codec (from ``decode_nerv_codec``),
    runs forward over the coord grid, returns argmax class IDs as a uint8
    tensor for direct AV1/AMRC-equivalent ingestion. Pure forward; no
    gradient tracking.

    Args:
        codec: decoded codec.
        num_frames: T.
        height: H.
        width: W.
        batch_size: forward batch size.
        device: device for the forward pass. CUDA at inflate time on T4
            (~2.4s for 384×512×1200); CPU acceptable for unit tests.

    Returns:
        (T, H, W) uint8 tensor with values in [0, num_classes - 1].
    """
    dev = torch.device(str(device))
    if str(dev).startswith("mps"):
        raise ValueError(
            "render_mask_argmax refuses device='mps'. CLAUDE.md non-negotiable."
        )
    codec.to(dev).eval()
    ts = torch.linspace(-1.0, 1.0, num_frames)
    ys = torch.linspace(-1.0, 1.0, height)
    xs = torch.linspace(-1.0, 1.0, width)
    grid_t, grid_y, grid_x = torch.meshgrid(ts, ys, xs, indexing="ij")
    coords_all = torch.stack(
        [grid_t.reshape(-1), grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1
    )
    out_argmax = torch.empty(num_frames * height * width, dtype=torch.uint8)
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], batch_size):
            chunk = coords_all[start : start + batch_size].to(dev)
            logits = codec(chunk)
            pred = logits.argmax(dim=-1).to(torch.uint8).cpu()
            out_argmax[start : start + len(pred)] = pred
    return out_argmax.reshape(num_frames, height, width)


__all__ = [
    "NERV_MAGIC",
    "NERV_VERSION",
    "NeRVHeader",
    "NeRVMaskCodec",
    "NeRVSamplingComponent",
    "NeRVSamplingPool",
    "NeRVMaskTrainer",
    "decode_nerv_codec",
    "encode_nerv_codec",
    "nerv_codec_bytes",
    "positional_encode",
    "raw_fp16_baseline_bytes",
    "render_mask_argmax",
    "render_mask_logits",
]
