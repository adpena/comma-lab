# SPDX-License-Identifier: MIT
"""pact_nerv_mamba archive grammar — monolithic 0.bin (PNMB).

Catalog #124 STRICT archive-grammar 8 fields declared in package ``__init__``.
Export-first grammar (HNeRV parity L2):

::

    MAGIC(4)              b"PNMB"     Pact-NeRV Mamba
    VERSION(1)            u8          schema version (currently 1)
    LATENT_DIM(2)         u16         cfg.latent_dim
    NUM_PAIRS(2)          u16         cfg.num_pairs
    SSM_STATE_DIM(2)      u16         cfg.ssm_state_dim
    DECODER_BLOB_LEN(4)   u32         brotli decoder state_dict len
    LATENT_BLOB_LEN(4)    u32         raw int16 latents bytes len
    SSM_STATE_BLOB_LEN(4) u32         raw fp16 ssm state bytes len
    META_BLOB_LEN(4)      u32         utf-8 json meta bytes len

Header: 4+1+2+2+2+4+4+4+4 = 27 bytes.

The Mamba-2 SSM state IS the distinguishing primitive's byte residence — it
ships SEPARATELY from the decoder weights (logical surface for the inflate
runtime to optionally load only when the SSM block is consumed).

CLAUDE.md compliance: deterministic, no /tmp, no scorer load, no comma2k19
dependency at inflate time.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

PNMB_MAGIC: bytes = b"PNMB"
PNMB_SCHEMA_VERSION: int = 1

PNMB_HEADER_FMT: str = "<4sBHHHIIII"
PNMB_HEADER_SIZE: int = struct.calcsize(PNMB_HEADER_FMT)
assert PNMB_HEADER_SIZE == 27, "PNMB header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class PactNervMambaArchive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    ssm_state: torch.Tensor
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
        # FFFF Catalog #158 fix
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
    latents: torch.Tensor,
    ssm_state: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = PNMB_SCHEMA_VERSION,
) -> bytes:
    if schema_version != PNMB_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
    if ssm_state.dim() != 1:
        raise ValueError(
            f"ssm_state must be 1-D (state_dim,); got {tuple(ssm_state.shape)}"
        )

    num_pairs, latent_dim = int(latents.shape[0]), int(latents.shape[1])
    state_dim = int(ssm_state.shape[0])
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")
    if latent_dim <= 0 or latent_dim > 0xFFFF:
        raise ValueError(f"latent_dim {latent_dim} out of u16 range")
    if state_dim <= 0 or state_dim > 0xFFFF:
        raise ValueError(f"state_dim {state_dim} out of u16 range")

    q_latents, lat_scale, lat_zp = _quantize_tensor_to_int16(latents)
    latent_bytes = q_latents.contiguous().numpy().tobytes()

    ssm_state_fp16 = ssm_state.detach().to(dtype=torch.float16, device="cpu").contiguous()
    ssm_state_bytes = ssm_state_fp16.numpy().tobytes()

    decoder_blob = _serialize_state_dict(decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        PNMB_HEADER_FMT,
        PNMB_MAGIC,
        schema_version,
        latent_dim,
        num_pairs,
        state_dim,
        len(decoder_blob),
        len(latent_bytes),
        len(ssm_state_bytes),
        len(meta_bytes),
    )
    return header + decoder_blob + latent_bytes + ssm_state_bytes + meta_bytes


def parse_archive(blob: bytes) -> PactNervMambaArchive:
    if len(blob) < PNMB_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {PNMB_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        state_dim,
        decoder_len,
        latent_len,
        ssm_state_len,
        meta_len,
    ) = struct.unpack(PNMB_HEADER_FMT, blob[:PNMB_HEADER_SIZE])
    if magic != PNMB_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {PNMB_MAGIC!r})")
    if version != PNMB_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )
    expected_ssm_bytes = state_dim * 2
    if ssm_state_len != expected_ssm_bytes:
        raise ValueError(
            f"ssm_state_len {ssm_state_len} != state_dim*2 = {expected_ssm_bytes}"
        )

    end_header = PNMB_HEADER_SIZE
    end_decoder = end_header + decoder_len
    end_latents = end_decoder + latent_len
    end_ssm = end_latents + ssm_state_len
    end_meta = end_ssm + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    decoder_blob = blob[end_header:end_decoder]
    latent_blob = blob[end_decoder:end_latents]
    ssm_state_blob = blob[end_latents:end_ssm]
    meta_blob = blob[end_ssm:end_meta]

    sd = _deserialize_state_dict(decoder_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_latents = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    ssm_state = torch.from_numpy(
        np.frombuffer(ssm_state_blob, dtype=np.float16).copy()
    ).to(torch.float32)

    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequantize_tensor(q_latents, lat_scale, lat_zp)

    return PactNervMambaArchive(
        decoder_state_dict=sd,
        latents=latents,
        ssm_state=ssm_state,
        meta=meta,
        schema_version=int(version),
    )
