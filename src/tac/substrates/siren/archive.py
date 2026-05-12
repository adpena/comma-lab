"""siren archive grammar SRV1 — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (HNeRV parity discipline
lesson L2):

::

    MAGIC(4)             b"SRV1"   SIREN Variant 1
    VERSION(1)           u8        schema version (currently 1)
    NUM_PAIRS(2)         u16       cfg.num_pairs (e.g. 600)
    HIDDEN_DIM(2)        u16       cfg.hidden_dim (e.g. 128)
    NUM_HIDDEN_LAYERS(1) u8        cfg.num_hidden_layers (e.g. 6)
    OUTPUT_HEIGHT(2)     u16       cfg.output_height
    OUTPUT_WIDTH(2)      u16       cfg.output_width
    DECODER_BLOB_LEN(4)  u32       brotli(state_dict) of MLP weights
    META_BLOB_LEN(4)     u32       utf-8 json meta bytes len
    DECODER_BLOB         ...       brotli(pickled state_dict, fp16 cpu)
    META_BLOB            ...       json: {first_omega, hidden_omega, coord_dim, output_dim, ...}

SRV1 = "SIREN Variant 1". No latents — SIREN is purely a coordinate->RGB MLP.
All trainable bytes live in the DECODER_BLOB. The grammar is fixed at design-
time; mutating it changes the schema VERSION and requires a new inflate.py.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 CPU state_dict, fixed brotli quality)
- No /tmp paths
- No scorer load
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

SRV1_MAGIC: bytes = b"SRV1"
"""siren variant 1 archive magic."""

SRV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + HIDDEN_DIM(2) + NUM_HIDDEN_LAYERS(1)
#                + OUTPUT_H(2) + OUTPUT_W(2) + DECODER_LEN(4) + META_LEN(4) = 22 bytes
SRV1_HEADER_FMT: str = "<4sBHHBHHII"
SRV1_HEADER_SIZE: int = struct.calcsize(SRV1_HEADER_FMT)
assert SRV1_HEADER_SIZE == 22, "SRV1 header size invariant (4+1+2+2+1+2+2+4+4 = 22)"

# Brotli quality
BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class SirenArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """MLP state_dict (all model weights — there are no latents)."""

    meta: dict[str, object]
    """Sidecar JSON meta with arch hparams (omega, coord_dim, output_dim, ...)."""

    schema_version: int
    num_pairs: int
    hidden_dim: int
    num_hidden_layers: int
    output_height: int
    output_width: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu).

    ``_spatial_coords`` is a deterministic meshgrid buffer, not payload. It is
    rebuilt from ``SirenConfig`` at inflate time and is refused here so the
    archive never pays bytes for it.
    """
    _validate_runtime_state_dict(sd)
    buf = io.BytesIO()
    sd_cpu = {k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()}
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _validate_runtime_state_dict(sd: dict[str, torch.Tensor]) -> None:
    if "_spatial_coords" in sd:
        raise ValueError(
            "SRV1 archive state_dict contains deterministic buffer _spatial_coords. "
            "Use SirenSubstrate.runtime_state_dict_for_archive()."
        )
    if not any(key.startswith("hidden.") for key in sd):
        raise ValueError("SRV1 runtime state_dict missing hidden.* MLP weights")
    if not any(key.startswith("output_layer.") for key in sd):
        raise ValueError("SRV1 runtime state_dict missing output_layer.* weights")


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    meta: dict[str, object],
    *,
    num_pairs: int,
    hidden_dim: int,
    num_hidden_layers: int,
    output_height: int,
    output_width: int,
    schema_version: int = SRV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained SIREN state into the monolithic 0.bin bytes."""
    if schema_version != SRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")

    for name, v, max_v in (
        ("num_pairs", num_pairs, 0xFFFF),
        ("hidden_dim", hidden_dim, 0xFFFF),
        ("num_hidden_layers", num_hidden_layers, 0xFF),
        ("output_height", output_height, 0xFFFF),
        ("output_width", output_width, 0xFFFF),
    ):
        if v <= 0 or v > max_v:
            raise ValueError(f"{name}={v} out of u8/u16 range (max {max_v})")

    decoder_blob = _serialize_state_dict(decoder_state_dict)
    meta_bytes = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        SRV1_HEADER_FMT,
        SRV1_MAGIC,
        schema_version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_height,
        output_width,
        len(decoder_blob),
        len(meta_bytes),
    )
    return header + decoder_blob + meta_bytes


def parse_archive(blob: bytes) -> SirenArchive:
    """Parse 0.bin bytes back into typed SirenArchive."""
    if len(blob) < SRV1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes; need >= {SRV1_HEADER_SIZE})")
    (
        magic,
        version,
        num_pairs,
        hidden_dim,
        num_hidden_layers,
        output_height,
        output_width,
        decoder_len,
        meta_len,
    ) = struct.unpack(SRV1_HEADER_FMT, blob[:SRV1_HEADER_SIZE])
    if magic != SRV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {SRV1_MAGIC!r})")
    if version != SRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    pos = SRV1_HEADER_SIZE
    decoder_blob = blob[pos : pos + decoder_len]
    pos += decoder_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    return SirenArchive(
        decoder_state_dict=sd,
        meta=meta,
        schema_version=int(version),
        num_pairs=int(num_pairs),
        hidden_dim=int(hidden_dim),
        num_hidden_layers=int(num_hidden_layers),
        output_height=int(output_height),
        output_width=int(output_width),
    )
