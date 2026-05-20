# SPDX-License-Identifier: MIT
"""pact_nerv_vq archive grammar — monolithic 0.bin (PVQ).

Catalog #124 STRICT archive-grammar 8 fields declared in package ``__init__``.
Export-first grammar (HNeRV parity L2):

::

    MAGIC(4)              b"PVQ\\x00"  Pact-NeRV VQ
    VERSION(1)            u8       schema version (currently 1)
    LATENT_DIM(2)         u16      cfg.latent_dim
    NUM_PAIRS(2)          u16      cfg.num_pairs
    CODEBOOK_SIZE(2)      u16      cfg.codebook_size
    DECODER_BLOB_LEN(4)   u32      brotli decoder state_dict len
    CODEBOOK_BLOB_LEN(4)  u32      raw int16 codebook (codebook_size * latent_dim * 2)
    INDICES_BLOB_LEN(4)   u32      raw uint16 codebook indices per pair
    META_BLOB_LEN(4)      u32      utf-8 json meta bytes len

Header: 4+1+2+2+2+4+4+4+4 = 27 bytes.

The codebook ships as fp16 int16-quantized; indices ship as uint16 (codebook
sizes up to 65535 supported).
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PVQ_MAGIC: bytes = b"PVQ\x00"
PVQ_SCHEMA_VERSION: int = 1

PVQ_HEADER_FMT: str = "<4sBHHHIIII"
PVQ_HEADER_SIZE: int = struct.calcsize(PVQ_HEADER_FMT)
assert PVQ_HEADER_SIZE == 27, "PVQ header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervVqArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    codebook: torch.Tensor
    """(codebook_size, latent_dim) fp32 reconstructed from int16."""
    indices: torch.Tensor
    """(num_pairs,) uint16 codebook indices."""
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("decoder_state_dict blob did not unpickle to a dict")
    return sd


def _quantize_tensor_to_int16(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    if tensor.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {tensor.dtype}")
    f = tensor.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_tensor(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    decoder_state_dict: dict[str, torch.Tensor],
    codebook: torch.Tensor,
    indices: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = PVQ_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PVQ_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if codebook.dim() != 2:
        raise ValueError(
            f"codebook must be 2-D (codebook_size, latent_dim); got {tuple(codebook.shape)}"
        )
    if indices.dim() != 1:
        raise ValueError(
            f"indices must be 1-D (num_pairs,); got {tuple(indices.shape)}"
        )

    codebook_size, latent_dim = int(codebook.shape[0]), int(codebook.shape[1])
    num_pairs = int(indices.shape[0])
    if codebook_size <= 0 or codebook_size > 0xFFFF:
        raise ValueError(f"codebook_size {codebook_size} out of u16 range")
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if int(indices.max().item()) >= codebook_size or int(indices.min().item()) < 0:
        raise ValueError("indices out of [0, codebook_size)")

    q_codebook, cb_scale, cb_zp = _quantize_tensor_to_int16(codebook)
    codebook_bytes = q_codebook.contiguous().numpy().tobytes()

    indices_uint16 = indices.to(torch.int64).clamp(0, 0xFFFF).numpy().astype("uint16")
    indices_bytes = indices_uint16.tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_codebook_quant_scale"] = float(cb_scale)
    meta_with_quant["_codebook_quant_zero_point"] = float(cb_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PVQ_HEADER_FMT,
        PVQ_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        codebook_size,
        len(decoder_blob),
        len(codebook_bytes),
        len(indices_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + codebook_bytes + indices_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervVqArchive:
    if len(blob) < PVQ_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PVQ_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        codebook_size,
        decoder_len,
        codebook_len,
        indices_len,
        meta_len,
    ) = struct.unpack(PVQ_HEADER_FMT, blob[:PVQ_HEADER_SIZE])
    if magic != PVQ_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PVQ_MAGIC!r})")
    if version != PVQ_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_codebook_bytes = codebook_size * latent_dim * 2
    if codebook_len != expected_codebook_bytes:
        raise ValueError(
            f"codebook_len {codebook_len} != codebook_size*latent_dim*2 = {expected_codebook_bytes}"
        )
    expected_indices_bytes = num_pairs * 2
    if indices_len != expected_indices_bytes:
        raise ValueError(
            f"indices_len {indices_len} != num_pairs*2 = {expected_indices_bytes}"
        )

    end_header = PVQ_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_codebook = end_decoder + codebook_len
    end_indices = end_codebook + indices_len
    end_meta = end_indices + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    codebook_blob = blob[end_decoder:end_codebook]
    indices_blob = blob[end_codebook:end_indices]
    meta_blob = blob[end_indices:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_codebook = torch.from_numpy(
        np.frombuffer(codebook_blob, dtype=np.int16).copy()
    ).view(codebook_size, latent_dim)
    indices = torch.from_numpy(
        np.frombuffer(indices_blob, dtype=np.uint16).copy().astype("int64")
    ).view(num_pairs).to(torch.long)

    cb_scale = float(meta.pop("_codebook_quant_scale"))
    cb_zp = float(meta.pop("_codebook_quant_zero_point"))
    codebook = _dequantize_tensor(q_codebook, cb_scale, cb_zp)

    return PactNervVqArchive(
        decoder_state_dict=sd,
        codebook=codebook,
        indices=indices,
        meta=meta,
        schema_version=int(version),
    )
