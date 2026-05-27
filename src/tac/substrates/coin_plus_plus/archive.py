# SPDX-License-Identifier: MIT
"""coin_plus_plus archive grammar — monolithic single-file ``0.bin`` (CPP1).

Catalog #124 STRICT archive-grammar 8 fields declared in package
``__init__``. Export-first grammar (L2):

::

    MAGIC(4)               b"CPP1"  COIN++ Variant 1
    VERSION(1)             u8       schema version (currently 1)
    MODULATION_DIM(2)      u16      cfg.modulation_dim (distinctive: per-pair latent dim)
    NUM_PAIRS(2)           u16      cfg.num_pairs
    BASE_MLP_BLOB_LEN(4)   u32      brotli shared base MLP state_dict len
    MODULATION_BLOB_LEN(4) u32      raw int8 modulation bytes len
    META_BLOB_LEN(4)       u32      utf-8 json meta bytes len
    BASE_MLP_BLOB          ...      brotli(quality=9) of pickled state_dict
                                    (shared base coord-MLP weights)
    MODULATION_BLOB        ...      int8 modulations row-major
                                    (num_pairs * modulation_dim bytes)
    META_BLOB              ...      json: {"hidden_dim": ..., "num_hidden_layers": ..., ...}

Header: 4+1+2+2+4+4+4 = 21 bytes (MODULATION_DIM is the distinctive
header field vs sister DSV1/BSV1/NRV1 — operator-visible knob declaring
the per-pair latent rate that the inflate-time consumer must respect).

Note: MODULATION uses int8 (not int16) because the typical COIN++
modulation range is small (~[-2, 2]) and int8 captures this well at
half the rate of int16 latents. This is the distinctive COIN++ rate-
saving choice vs the NeRV-family.

Catalog #124 parser-section manifest enumerates 6 logical sections:
HEADER + BASE_MLP_BLOB + MODULATION_BLOB + META_BLOB + (implicit
"shared_base_mlp_weights" subset inside BASE_MLP_BLOB) + (implicit
"per_pair_modulations" subset inside MODULATION_BLOB).

CLAUDE.md compliance: deterministic, no /tmp, no scorer load.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

CPP1_MAGIC: bytes = b"CPP1"
# Schema v2 (PACT-NeRV numpy-portable bridge, 2026-05-27): the BASE_MLP_BLOB
# internal encoding is now a torch-free numpy-native ``{key: fp16 array}``
# serialization (was ``brotli(pickle(torch_tensors))`` in v1). Per the 8th
# MLX-first standing directive: ``torch`` is FORBIDDEN at inflate time, and a
# torch-tensor pickle requires ``torch`` to unpickle (it embeds
# ``torch._utils._rebuild_tensor_v2`` GLOBAL refs). The numpy-native blob lets
# ``parse_archive_numpy`` read weights with NO torch dependency so the shipped
# inflate runtime is numpy/PIL-portable. The torch-side ``parse_archive`` wraps
# the same numpy arrays in ``torch.from_numpy`` for training-side parity.
CPP1_SCHEMA_VERSION: int = 2

CPP1_HEADER_FMT: str = "<4sBHHIII"
CPP1_HEADER_SIZE: int = struct.calcsize(CPP1_HEADER_FMT)
assert CPP1_HEADER_SIZE == 21, "CPP1 header size invariant"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class CoinplusplusArchive:
    """Parsed archive structure — the inflate-time data contract."""

    base_mlp_state_dict: dict[str, torch.Tensor]
    """Shared base coord-MLP state_dict (NOT per-pair; amortized over all pairs)."""

    modulations: torch.Tensor
    """Per-pair modulation vectors (shape: num_pairs x modulation_dim)."""

    meta: dict[str, object]
    schema_version: int
    modulation_dim: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize a torch state_dict into the torch-free numpy-native CPP1-v2 blob.

    The blob is ``np.savez``-style but hand-rolled so the inflate-time reader
    needs ONLY numpy (no ``np.load`` allow_pickle, no torch). Layout (inside the
    brotli envelope)::

        u32 LE  num_entries
        repeat num_entries:
            u16 LE  key_len
            key     utf-8 bytes
            u8      ndim
            ndim x u32 LE  shape dims
            f16 raw  prod(shape) fp16 values (little-endian, C-order)
    """
    np_sd = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous().numpy()
        for k, v in sd.items()
    }
    return _serialize_numpy_state_dict(np_sd)


def _serialize_numpy_state_dict(np_sd: dict[str, "np.ndarray"]) -> bytes:
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(np_sd)))
    for key, arr in np_sd.items():
        a = np.ascontiguousarray(arr, dtype=np.float16)
        key_bytes = key.encode("utf-8")
        buf.write(struct.pack("<H", len(key_bytes)))
        buf.write(key_bytes)
        buf.write(struct.pack("<B", a.ndim))
        for dim in a.shape:
            buf.write(struct.pack("<I", int(dim)))
        buf.write(a.tobytes(order="C"))
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_numpy_state_dict(blob: bytes) -> dict[str, "np.ndarray"]:
    """Torch-free deserialize of the CPP1-v2 numpy-native state_dict blob.

    Returns ``{key: fp16 ndarray}``. Used by the numpy-portable inflate path
    (NO torch import). The shipped inflate runtime calls this.
    """
    raw = brotli.decompress(blob)
    mv = memoryview(raw)
    off = 0
    (num_entries,) = struct.unpack_from("<I", mv, off)
    off += 4
    out: dict[str, "np.ndarray"] = {}
    for _ in range(num_entries):
        (key_len,) = struct.unpack_from("<H", mv, off)
        off += 2
        key = bytes(mv[off : off + key_len]).decode("utf-8")
        off += key_len
        (ndim,) = struct.unpack_from("<B", mv, off)
        off += 1
        shape: list[int] = []
        for _d in range(ndim):
            (dim,) = struct.unpack_from("<I", mv, off)
            off += 4
            shape.append(int(dim))
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * 2  # fp16 = 2 bytes
        arr = np.frombuffer(mv[off : off + nbytes], dtype=np.float16).copy()
        off += nbytes
        out[key] = arr.reshape(shape) if shape else arr.reshape(())
    return out


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    """Torch-side deserialize (training/eval parity): wraps numpy arrays."""
    np_sd = _deserialize_numpy_state_dict(blob)
    return {k: torch.from_numpy(v.astype("float32")) for k, v in np_sd.items()}


def _quantize_modulations_to_int8(
    modulations: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize modulations to int8 with linear scale+zero_point."""
    if modulations.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"modulations must be float; got {modulations.dtype}")
    f = modulations.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:  # QUANTIZE_DEGENERATE_OK:coin_plus_plus_l0_scaffold_research_only_uses_int8_256level_minus_128_sentinel_math_correct_alternative_to_canonical_254level_minus_127_pattern_dequant_recovers_lo_exactly_via_q_plus_128_times_one_plus_lo_per_lane_registry_lane_coin_plus_plus_l0_scaffold_20260520_l0_status_full_main_raises_not_implemented_error_per_catalog_240_research_substrate_class_no_paid_dispatch_eligible_until_phase_2_council_approval
        # Degenerate-range branch (FFFF Catalog #161 fix): use -128 sentinel
        # so dequant = 0 * scale + lo = lo.
        return (torch.full_like(f, -128, dtype=torch.int8), 1.0, lo)
    # int8 range: [-128, 127] -> 256 levels (use 255 spacing per Catalog #161 pattern)
    scale = (hi - lo) / 255.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 255.0)
    q = (q_unsigned - 128.0).to(torch.int8)
    return (q, scale, lo)


def _dequantize_modulations(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 128.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    base_mlp_state_dict: dict[str, torch.Tensor],
    modulations: torch.Tensor,
    meta: dict[str, object],
    *,
    modulation_dim: int,
    schema_version: int = CPP1_SCHEMA_VERSION,
) -> bytes:
    if schema_version != CPP1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if modulations.dim() != 2:
        raise ValueError(
            f"modulations must be 2-D (num_pairs, modulation_dim); got {tuple(modulations.shape)}"
        )
    if modulation_dim <= 0 or modulation_dim > 0xFFFF:
        raise ValueError(f"modulation_dim {modulation_dim} out of u16 range")

    num_pairs, mdim = int(modulations.shape[0]), int(modulations.shape[1])
    if mdim != modulation_dim:
        raise ValueError(
            f"modulations shape last-dim {mdim} != declared modulation_dim {modulation_dim}"
        )
    if num_pairs <= 0 or num_pairs > 0xFFFF:
        raise ValueError(f"num_pairs {num_pairs} out of u16 range")

    q_modulations, scale, zero_point = _quantize_modulations_to_int8(modulations)
    modulation_bytes = q_modulations.contiguous().numpy().tobytes()
    base_mlp_blob = _serialize_state_dict(base_mlp_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_quant_scale"] = float(scale)
    meta_with_quant["_quant_zero_point"] = float(zero_point)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        CPP1_HEADER_FMT,
        CPP1_MAGIC,
        schema_version,
        modulation_dim,
        num_pairs,
        len(base_mlp_blob),
        len(modulation_bytes),
        len(meta_bytes),
    )
    return header + base_mlp_blob + modulation_bytes + meta_bytes


def parse_archive(blob: bytes) -> CoinplusplusArchive:
    if len(blob) < CPP1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {CPP1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        modulation_dim,
        num_pairs,
        base_mlp_len,
        modulation_len,
        meta_len,
    ) = struct.unpack(CPP1_HEADER_FMT, blob[:CPP1_HEADER_SIZE])
    if magic != CPP1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {CPP1_MAGIC!r})")
    if version != CPP1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_modulation_bytes = num_pairs * modulation_dim * 1  # int8 = 1 byte
    if modulation_len != expected_modulation_bytes:
        raise ValueError(
            f"modulation_len {modulation_len} != num_pairs*modulation_dim*1 = {expected_modulation_bytes}"
        )

    end_header = CPP1_HEADER_SIZE
    end_base_mlp = end_header + base_mlp_len
    end_modulations = end_base_mlp + modulation_len
    end_meta = end_modulations + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    base_mlp_blob = blob[end_header:end_base_mlp]
    modulation_blob = blob[end_base_mlp:end_modulations]
    meta_blob = blob[end_modulations:end_meta]

    sd = _deserialize_state_dict(base_mlp_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_modulations = torch.from_numpy(
        np.frombuffer(modulation_blob, dtype=np.int8).copy()
    ).view(num_pairs, modulation_dim)
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    modulations = _dequantize_modulations(q_modulations, scale, zp)

    return CoinplusplusArchive(
        base_mlp_state_dict=sd,
        modulations=modulations,
        meta=meta,
        schema_version=int(version),
        modulation_dim=int(modulation_dim),
    )


@dataclass(frozen=True)
class CoinplusplusArchiveNumpy:
    """Torch-free parsed archive — the numpy-portable inflate-time contract.

    Sister of ``CoinplusplusArchive`` but with ``np.ndarray`` weights so the
    shipped inflate runtime needs ONLY numpy (no torch) per the 8th MLX-first
    standing directive.
    """

    base_mlp_state_dict: dict[str, "np.ndarray"]
    """Shared base coord-MLP state_dict as {key: fp32 ndarray}."""

    modulations: "np.ndarray"
    """Per-pair modulation vectors (num_pairs x modulation_dim), fp32."""

    meta: dict[str, object]
    schema_version: int
    modulation_dim: int


def parse_archive_numpy(blob: bytes) -> CoinplusplusArchiveNumpy:
    """Torch-free parse of a CPP1-v2 archive for the numpy-portable inflate.

    Identical section walk to ``parse_archive`` but reconstructs weights +
    modulations as numpy arrays with NO torch import. Per the 8th MLX-first
    directive's bridge contract: the shipped inflate runtime reads weights via
    this path so the runtime tree carries only numpy + brotli + PIL.
    """
    if len(blob) < CPP1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {CPP1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        modulation_dim,
        num_pairs,
        base_mlp_len,
        modulation_len,
        meta_len,
    ) = struct.unpack(CPP1_HEADER_FMT, blob[:CPP1_HEADER_SIZE])
    if magic != CPP1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {CPP1_MAGIC!r})")
    if version != CPP1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_modulation_bytes = num_pairs * modulation_dim * 1  # int8 = 1 byte
    if modulation_len != expected_modulation_bytes:
        raise ValueError(
            f"modulation_len {modulation_len} != num_pairs*modulation_dim*1 = "
            f"{expected_modulation_bytes}"
        )

    end_header = CPP1_HEADER_SIZE
    end_base_mlp = end_header + base_mlp_len
    end_modulations = end_base_mlp + modulation_len
    end_meta = end_modulations + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    base_mlp_blob = blob[end_header:end_base_mlp]
    modulation_blob = blob[end_base_mlp:end_modulations]
    meta_blob = blob[end_modulations:end_meta]

    np_sd_fp16 = _deserialize_numpy_state_dict(base_mlp_blob)
    np_sd = {k: v.astype(np.float32) for k, v in np_sd_fp16.items()}
    meta = json.loads(meta_blob.decode("utf-8"))

    q = np.frombuffer(modulation_blob, dtype=np.int8).reshape(
        num_pairs, modulation_dim
    )
    scale = float(meta.pop("_quant_scale"))
    zp = float(meta.pop("_quant_zero_point"))
    # dequant mirrors _dequantize_modulations: (q + 128) * scale + zero_point
    modulations = (q.astype(np.float32) + 128.0) * scale + zp

    return CoinplusplusArchiveNumpy(
        base_mlp_state_dict=np_sd,
        modulations=modulations,
        meta=meta,
        schema_version=int(version),
        modulation_dim=int(modulation_dim),
    )
