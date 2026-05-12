"""hybrid_renderer_residual archive grammar — monolithic single-file ``0.bin`` (γ).

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (lesson L2) for γ.

Per the council §4.2 γ candidate archive-grammar declaration, the γ archive
extends α's grammar with a residual-basis section:

::

    MAGIC(4)              b"HRRV1"  Hybrid-Renderer-Residual Variant 1
                                    NB: 5 bytes magic vs α's 4 to disambiguate
                                    -> use a 4-byte b"HRR1" magic for header
                                    fixed-width discipline (HRR1 + version byte)
    VERSION(1)            u8        schema version (currently 1)
    LATENT_DIM(2)         u16       cfg.latent_dim (renderer latent)
    RES_BASIS_DIM(2)      u16       cfg.residual_basis_dim (dictionary size)
    RES_COEFF_K(2)        u16       cfg.residual_coeffs_per_pair (sparsity k)
    NUM_PAIRS(2)          u16       cfg.num_pairs
    RENDERER_BLOB_LEN(4)  u32       brotli-compressed renderer state_dict bytes len
    RESDEC_BLOB_LEN(4)    u32       brotli-compressed residual_decoder state_dict bytes len
    LATENTS_BLOB_LEN(4)   u32       raw int16 renderer latent bytes len (= num_pairs*latent_dim*2)
    RESCOEFFS_BLOB_LEN(4) u32       raw int16 residual coefficient bytes len
                                    (= num_pairs * residual_coeffs_per_pair * 4
                                     [2 bytes index + 2 bytes int16 value])
    META_BLOB_LEN(4)      u32       utf-8 json meta bytes len
    RENDERER_BLOB         ...
    RESDEC_BLOB           ...
    LATENTS_BLOB          ...       int16 row-major (num_pairs, latent_dim)
    RESCOEFFS_BLOB        ...       per-pair: k * (u16 index || i16 value)
    META_BLOB             ...

The grammar is FIXED at design-time; mutating it changes the schema VERSION
and requires a new inflate.py.

Round-trip contract (tested in tests/test_hybrid_renderer_residual_roundtrip.py
per Catalog #91):

    bytes -> parse_archive -> HybridRendererResidualArchive
    HybridRendererResidualArchive components -> pack_archive -> bytes

The parse_archive() return type IS the inflate-time API.

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load
- Deterministic: same input -> same bytes (no timestamps, no host info)
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch


HRRV1_MAGIC: bytes = b"HRR1"
"""hybrid_renderer_residual variant 1 archive magic (4 bytes)."""

HRRV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
#   MAGIC(4) + VERSION(1) + LATENT_DIM(2) + RES_BASIS_DIM(2)
#   + RES_COEFF_K(2) + NUM_PAIRS(2)
#   + RENDERER_LEN(4) + RESDEC_LEN(4) + LATENTS_LEN(4) + RESCOEFFS_LEN(4)
#   + META_LEN(4)
# Total = 4 + 1 + 2 + 2 + 2 + 2 + 4*5 = 33 bytes
HRRV1_HEADER_FMT: str = "<4sBHHHHIIIII"
HRRV1_HEADER_SIZE: int = struct.calcsize(HRRV1_HEADER_FMT)
assert HRRV1_HEADER_SIZE == 33, f"header size invariant; got {HRRV1_HEADER_SIZE}"

BROTLI_QUALITY: int = 9

# Per-coefficient on-disk size in the rescoeffs blob: u16 index + i16 value
RESCOEFF_BYTES_PER_ELEMENT: int = 4


@dataclass(frozen=True)
class HybridRendererResidualArchive:
    """Parsed archive structure — the inflate-time data contract.

    Attributes:
        renderer_state_dict: HNeRV-class renderer state_dict (without latents).
        residual_decoder_state_dict: small residual-basis decoder state_dict
            (maps sparse-coeff vector -> per-pair RGB delta).
        latents: ``(num_pairs, latent_dim)`` float renderer latents
            (dequantized from int16 archive bytes).
        residual_basis_coefficients: ``(num_pairs, residual_coeffs_per_pair, 2)``
            int64 tensor whose [:, :, 0] are basis indices in [0, basis_dim)
            and [:, :, 1] are int16 quantized coefficient magnitudes. The
            inflate side dequantizes the magnitudes via meta["_res_quant_*"].
        meta: sidecar JSON meta (config, quant scales/zero-points,
            residual_basis_dim, residual_coeffs_per_pair, output_h/w, ...).
        schema_version: archive schema version (must == HRRV1_SCHEMA_VERSION).
    """

    renderer_state_dict: dict[str, torch.Tensor]
    residual_decoder_state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor
    residual_basis_coefficients: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 CPU)."""
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _quantize_to_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize ``t`` to int16. Returns ``(q, scale, zero_point)``.

    ``f = (q_int16 + 32767) * scale + zero_point``
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.zeros_like(f, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_from_int16(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def _pack_residual_coeffs(
    indices: torch.Tensor, values_int16: torch.Tensor
) -> bytes:
    """Pack per-pair sparse residual coefficients as ``u16 index || i16 value`` pairs.

    Args:
        indices: ``(num_pairs, k)`` int64 / int32 / long tensor of basis indices
            in ``[0, residual_basis_dim)``; must fit in u16.
        values_int16: ``(num_pairs, k)`` int16 quantized coefficient values.

    Returns deterministic bytes laid out row-major per pair.
    """
    if indices.shape != values_int16.shape:
        raise ValueError(
            f"indices {tuple(indices.shape)} vs values {tuple(values_int16.shape)} shape mismatch"
        )
    if int(indices.min().item()) < 0 or int(indices.max().item()) > 0xFFFF:
        raise ValueError("residual basis indices out of u16 range [0, 65535]")
    # Pack as little-endian u16 indices interleaved with little-endian i16 values
    import numpy as np  # local import; keep module's import-time light

    idx_np = indices.detach().to(device="cpu", dtype=torch.int64).numpy().astype(
        np.uint16, copy=False
    )
    val_np = values_int16.detach().to(device="cpu", dtype=torch.int16).numpy()
    if val_np.dtype != np.int16:
        raise ValueError(f"values_int16 dtype must be int16 on numpy side; got {val_np.dtype}")

    # Interleave: per pair, alternate (idx, val) pairs flattened to bytes.
    # We use np.empty with a structured dtype so layout is deterministic.
    num_pairs, k = idx_np.shape
    packed = np.empty((num_pairs, k), dtype=[("idx", "<u2"), ("val", "<i2")])
    packed["idx"] = idx_np
    packed["val"] = val_np
    return packed.tobytes()


def _unpack_residual_coeffs(
    blob: bytes, num_pairs: int, k: int
) -> torch.Tensor:
    """Unpack residual coefficient bytes into a ``(num_pairs, k, 2)`` int64 tensor.

    Returns ``out`` where ``out[:, :, 0]`` are basis indices (int64) and
    ``out[:, :, 1]`` are int16 quantized values widened to int64. The caller
    is responsible for dequantizing values via meta scale/zero-point.
    """
    import numpy as np

    expected = num_pairs * k * RESCOEFF_BYTES_PER_ELEMENT
    if len(blob) != expected:
        raise ValueError(
            f"residual coeff blob {len(blob)}B != expected {expected}B "
            f"({num_pairs} pairs * {k} coeffs * {RESCOEFF_BYTES_PER_ELEMENT} bytes)"
        )
    packed = np.frombuffer(
        blob, dtype=[("idx", "<u2"), ("val", "<i2")]
    ).copy().reshape(num_pairs, k)
    out = torch.empty((num_pairs, k, 2), dtype=torch.int64)
    out[:, :, 0] = torch.from_numpy(packed["idx"].astype(np.int64))
    out[:, :, 1] = torch.from_numpy(packed["val"].astype(np.int64))
    return out


def pack_archive(
    renderer_state_dict: dict[str, torch.Tensor],
    residual_decoder_state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor,
    residual_basis_indices: torch.Tensor,
    residual_basis_values: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = HRRV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize all γ substrate components into the monolithic ``0.bin`` bytes.

    The trainer ONLY calls this; everything else (framing, padding, section
    CRCs if added later) is the codec's responsibility, not the training
    loop's.

    Args:
        renderer_state_dict: HNeRV-class renderer weights (excluding per-pair latents).
        residual_decoder_state_dict: small residual-basis decoder weights.
        latents: ``(num_pairs, latent_dim)`` renderer latents (float).
        residual_basis_indices: ``(num_pairs, k)`` long tensor in
            ``[0, residual_basis_dim)``.
        residual_basis_values: ``(num_pairs, k)`` float tensor (will be
            int16-quantized).
        meta: sidecar JSON meta. Must include ``residual_basis_dim``,
            ``residual_coeffs_per_pair``, decoder/residual-decoder configs.
    """
    if schema_version != HRRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
    if residual_basis_indices.dim() != 2:
        raise ValueError(
            "residual_basis_indices must be 2-D (num_pairs, k); got "
            f"{tuple(residual_basis_indices.shape)}"
        )
    if residual_basis_values.shape != residual_basis_indices.shape:
        raise ValueError(
            f"residual_basis_values {tuple(residual_basis_values.shape)} "
            f"vs indices {tuple(residual_basis_indices.shape)} shape mismatch"
        )
    if residual_basis_indices.shape[0] != latents.shape[0]:
        raise ValueError(
            f"num_pairs mismatch: latents {latents.shape[0]} vs "
            f"residual {residual_basis_indices.shape[0]}"
        )

    num_pairs = int(latents.shape[0])
    latent_dim = int(latents.shape[1])
    residual_basis_dim = int(meta["residual_basis_dim"])
    residual_coeffs_per_pair = int(residual_basis_indices.shape[1])

    for name, val in (
        ("num_pairs", num_pairs),
        ("latent_dim", latent_dim),
        ("residual_basis_dim", residual_basis_dim),
        ("residual_coeffs_per_pair", residual_coeffs_per_pair),
    ):
        if val <= 0 or val > 0xFFFF:
            raise ValueError(f"{name}={val} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_to_int16(latents)
    q_res_vals, res_scale, res_zp = _quantize_to_int16(residual_basis_values)
    latent_bytes = q_lat.contiguous().numpy().tobytes()
    rescoeffs_bytes = _pack_residual_coeffs(residual_basis_indices, q_res_vals)

    renderer_blob = _serialize_state_dict(renderer_state_dict)
    resdec_blob = _serialize_state_dict(residual_decoder_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_with_quant["_res_quant_scale"] = float(res_scale)
    meta_with_quant["_res_quant_zero_point"] = float(res_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        HRRV1_HEADER_FMT,
        HRRV1_MAGIC,
        schema_version,
        latent_dim,
        residual_basis_dim,
        residual_coeffs_per_pair,
        num_pairs,
        len(renderer_blob),
        len(resdec_blob),
        len(latent_bytes),
        len(rescoeffs_bytes),
        len(meta_bytes),
    )
    return (
        header
        + renderer_blob
        + resdec_blob
        + latent_bytes
        + rescoeffs_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> HybridRendererResidualArchive:
    """Parse the ``0.bin`` bytes back into the γ substrate components.

    Pure-bytes function — no model class needed. inflate.py imports this +
    the model classes + builds + loads + renders + adds residual + writes
    PNGs, in ~140 LOC total.
    """
    if len(blob) < HRRV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {HRRV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        latent_dim,
        residual_basis_dim,
        residual_coeffs_per_pair,
        num_pairs,
        renderer_len,
        resdec_len,
        latent_len,
        rescoeffs_len,
        meta_len,
    ) = struct.unpack(HRRV1_HEADER_FMT, blob[:HRRV1_HEADER_SIZE])
    if magic != HRRV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {HRRV1_MAGIC!r})")
    if version != HRRV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2  # int16 = 2 bytes
    expected_rescoeffs_bytes = (
        num_pairs * residual_coeffs_per_pair * RESCOEFF_BYTES_PER_ELEMENT
    )
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )
    if rescoeffs_len != expected_rescoeffs_bytes:
        raise ValueError(
            f"rescoeffs_len {rescoeffs_len} != "
            f"num_pairs*k*{RESCOEFF_BYTES_PER_ELEMENT} = {expected_rescoeffs_bytes}"
        )

    end_header = HRRV1_HEADER_SIZE
    end_renderer = end_header + renderer_len
    end_resdec = end_renderer + resdec_len
    end_latents = end_resdec + latent_len
    end_rescoeffs = end_latents + rescoeffs_len
    end_meta = end_rescoeffs + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    renderer_blob = blob[end_header:end_renderer]
    resdec_blob = blob[end_renderer:end_resdec]
    latent_blob = blob[end_resdec:end_latents]
    rescoeffs_blob = blob[end_latents:end_rescoeffs]
    meta_blob = blob[end_rescoeffs:end_meta]

    renderer_sd = _deserialize_state_dict(renderer_blob)
    resdec_sd = _deserialize_state_dict(resdec_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import; keep module's import-time light
    q_lat = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequantize_from_int16(q_lat, lat_scale, lat_zp)

    rescoeffs = _unpack_residual_coeffs(
        rescoeffs_blob, num_pairs, residual_coeffs_per_pair
    )
    # Stash residual quant for the inflate side.
    res_scale = float(meta.pop("_res_quant_scale"))
    res_zp = float(meta.pop("_res_quant_zero_point"))
    # Surface back into meta under non-private names so the inflate side
    # can read them WITHOUT relying on private underscored keys.
    meta["residual_quant_scale"] = res_scale
    meta["residual_quant_zero_point"] = res_zp

    return HybridRendererResidualArchive(
        renderer_state_dict=renderer_sd,
        residual_decoder_state_dict=resdec_sd,
        latents=latents,
        residual_basis_coefficients=rescoeffs,
        meta=meta,
        schema_version=int(version),
    )
