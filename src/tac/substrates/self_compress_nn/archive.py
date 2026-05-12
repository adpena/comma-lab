"""self_compress_nn archive grammar — monolithic single-file ``0.bin`` (δ).

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (lesson L2) for δ.

Per the council §4.2 δ candidate archive-grammar declaration, the δ archive
stores ONLY (codebook + per-layer cluster_indices), NOT full weight tensors:

::

    MAGIC(4)              b"SCV1"  Self-Compress Variant 1
    VERSION(1)            u8       schema version (currently 1)
    NUM_LAYERS(2)         u16      number of quantized weight tensors
    NUM_PAIRS(2)          u16      cfg.num_pairs
    LATENT_DIM(2)         u16      cfg.latent_dim
    CODEBOOK_K(2)         u16      cfg.codebook_k (cluster count)
    CODEBOOK_DV(2)        u16      cfg.codebook_dv (per-cluster vector dim)
    CODEBOOK_BLOB_LEN(4)  u32      brotli-compressed codebook (fp16) bytes len
    LAYER_META_BLOB_LEN(4) u32     utf-8 json per-layer metadata bytes len
    INDICES_BLOB_LEN(4)   u32      raw int16 layer cluster_indices bytes len
                                    (sum over all layers of numel)
    LATENTS_BLOB_LEN(4)   u32      raw int16 latent bytes len
    META_BLOB_LEN(4)      u32      utf-8 json meta bytes len
    CODEBOOK_BLOB         ...      brotli(pickle({"codebook": fp16 (K, D_v)}))
    LAYER_META_BLOB       ...      utf-8 json list of {name, shape, numel}
    INDICES_BLOB          ...      concatenated int16 indices, ordered by layer
    LATENTS_BLOB          ...      int16 row-major (num_pairs, latent_dim)
    META_BLOB             ...

The grammar is FIXED at design-time; mutating it changes the schema VERSION
and requires a new inflate.py.

Round-trip contract (tested in tests/test_self_compress_nn_roundtrip.py per
Catalog #91):

    bytes -> parse_archive -> SelfCompressNnArchive
    SelfCompressNnArchive components -> pack_archive -> bytes

The parse_archive() return type IS the inflate-time API.

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths
- No scorer load
- Deterministic: same input -> same bytes (no timestamps, no host info)

MDL grounding (Selfcomp council position): per-weight rate cost is
``log2(K)`` bits for the cluster index + ``codebook_total_bits / total_numel``
bits amortized for the codebook. For ``K = 256`` and ``codebook ~ 4KB``
amortized over ~200K weights, that is ``8 + 0.16 = 8.16`` bits/weight, vs
``16`` for fp16. The ``2x`` rate saving is the δ score-axis attack vector
(orthogonal to α/β/γ which don't quantize weights).
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch


SCV1_MAGIC: bytes = b"SCV1"
"""self_compress_nn variant 1 archive magic (4 bytes)."""

SCV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
#   MAGIC(4) + VERSION(1) + NUM_LAYERS(2) + NUM_PAIRS(2)
#   + LATENT_DIM(2) + CODEBOOK_K(2) + CODEBOOK_DV(2)
#   + CODEBOOK_LEN(4) + LAYER_META_LEN(4) + INDICES_LEN(4)
#   + LATENTS_LEN(4) + META_LEN(4)
# Total = 4 + 1 + 5*2 + 5*4 = 35 bytes
SCV1_HEADER_FMT: str = "<4sBHHHHHIIIII"
SCV1_HEADER_SIZE: int = struct.calcsize(SCV1_HEADER_FMT)
assert SCV1_HEADER_SIZE == 35, f"header size invariant; got {SCV1_HEADER_SIZE}"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class SelfCompressNnArchive:
    """Parsed archive structure — the inflate-time data contract.

    Attributes:
        codebook: ``(K, D_v)`` float tensor of cluster centers (fp16 on disk).
        layer_cluster_indices: dict mapping tensor name -> int64 tensor of
            per-element cluster indices (shape matches the original weight
            tensor's flat numel for the int16 storage; reshaped to the
            target shape by inflate using layer_meta).
        layer_meta: list of dicts with keys ``name``, ``shape`` (target
            tensor shape as a list of ints), ``numel``. Preserves the
            on-disk layer order so int16 indices can be sliced back per
            layer.
        latents: ``(num_pairs, latent_dim)`` float tensor (dequantized from int16).
        meta: sidecar JSON meta (config, quant scales/zero-points, ...).
        schema_version: archive schema version (must == SCV1_SCHEMA_VERSION).
    """

    codebook: torch.Tensor
    layer_cluster_indices: dict[str, torch.Tensor]
    layer_meta: list[dict]
    latents: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_codebook(codebook: torch.Tensor) -> bytes:
    """Pickle + brotli the codebook deterministically (fp16 CPU)."""
    buf = io.BytesIO()
    cb_cpu = codebook.detach().to("cpu", dtype=torch.float16).contiguous()
    pickle.dump({"codebook": cb_cpu}, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_codebook(blob: bytes) -> torch.Tensor:
    raw = brotli.decompress(blob)
    obj = pickle.loads(raw)
    if not isinstance(obj, dict) or "codebook" not in obj:
        raise ValueError("codebook blob did not unpickle to {'codebook': tensor}")
    cb = obj["codebook"]
    if not isinstance(cb, torch.Tensor):
        raise ValueError("codebook entry must be a torch.Tensor")
    return cb


def _quantize_to_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize ``t`` to int16. Returns ``(q, scale, zero_point)``.

    ``f = (q_int16 + 32767) * scale + zero_point``
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        # FFFF Catalog #158 fix: -32767 fill so dequant = 0*scale + lo = lo
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_from_int16(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def _pack_indices_blob(
    layer_cluster_indices: dict[str, torch.Tensor], layer_meta: list[dict]
) -> bytes:
    """Concatenate per-layer cluster indices into one int16 byte stream.

    Layer order from layer_meta drives the order. Each layer's indices
    are flattened row-major.
    """
    import numpy as np  # local import; keep module's import-time light

    parts = []
    for entry in layer_meta:
        name = entry["name"]
        if name not in layer_cluster_indices:
            raise ValueError(f"layer_meta references missing layer indices: {name}")
        idx = layer_cluster_indices[name]
        if idx.numel() != int(entry["numel"]):
            raise ValueError(
                f"layer {name} numel mismatch: indices {idx.numel()} vs meta {entry['numel']}"
            )
        if int(idx.min().item()) < 0 or int(idx.max().item()) > 0x7FFF:
            raise ValueError(
                f"layer {name} cluster index out of int16 range [0, 32767]"
            )
        parts.append(idx.detach().to("cpu", dtype=torch.int16).contiguous().numpy())
    if not parts:
        return b""
    return np.concatenate(parts, axis=None).tobytes()


def _unpack_indices_blob(
    blob: bytes, layer_meta: list[dict]
) -> dict[str, torch.Tensor]:
    """Slice concatenated int16 cluster indices back into per-layer tensors."""
    import numpy as np

    flat = np.frombuffer(blob, dtype=np.int16).copy()
    total_expected = sum(int(e["numel"]) for e in layer_meta)
    if flat.size != total_expected:
        raise ValueError(
            f"indices blob has {flat.size} int16s but layer_meta expects {total_expected}"
        )
    out: dict[str, torch.Tensor] = {}
    cursor = 0
    for entry in layer_meta:
        n = int(entry["numel"])
        chunk = flat[cursor : cursor + n]
        cursor += n
        # store as int64 for downstream gather convenience (cheap copy)
        out[entry["name"]] = torch.from_numpy(chunk.astype(np.int64))
    return out


def pack_archive(
    codebook: torch.Tensor,
    layer_cluster_indices: dict[str, torch.Tensor],
    layer_meta: list[dict],
    latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = SCV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize all δ substrate components into the monolithic ``0.bin`` bytes.

    Args:
        codebook: ``(K, D_v)`` cluster-center vectors. Quantized fp16 on
            disk + brotli compressed.
        layer_cluster_indices: per-layer int tensor of cluster indices.
            Tensor name keys match ``layer_meta[i]["name"]``.
        layer_meta: list of ``{"name": str, "shape": [int, ...], "numel": int}``.
            Order is preserved on disk.
        latents: ``(num_pairs, latent_dim)`` renderer latents (float).
        meta: sidecar JSON meta with substrate config + quant scales.
    """
    if schema_version != SCV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if codebook.dim() != 2:
        raise ValueError(
            f"codebook must be 2-D (K, D_v); got {tuple(codebook.shape)}"
        )
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )

    K, D_v = int(codebook.shape[0]), int(codebook.shape[1])
    num_pairs = int(latents.shape[0])
    latent_dim = int(latents.shape[1])
    num_layers = len(layer_meta)

    for name, val in (
        ("K", K),
        ("D_v", D_v),
        ("num_pairs", num_pairs),
        ("latent_dim", latent_dim),
        ("num_layers", num_layers),
    ):
        if val <= 0 or val > 0xFFFF:
            raise ValueError(f"{name}={val} out of u16 range")

    q_lat, lat_scale, lat_zp = _quantize_to_int16(latents)
    latent_bytes = q_lat.contiguous().numpy().tobytes()

    cb_blob = _serialize_codebook(codebook)
    layer_meta_bytes = json.dumps(
        layer_meta, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    indices_bytes = _pack_indices_blob(layer_cluster_indices, layer_meta)

    meta_with_quant = dict(meta)
    meta_with_quant["_lat_quant_scale"] = float(lat_scale)
    meta_with_quant["_lat_quant_zero_point"] = float(lat_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        SCV1_HEADER_FMT,
        SCV1_MAGIC,
        schema_version,
        num_layers,
        num_pairs,
        latent_dim,
        K,
        D_v,
        len(cb_blob),
        len(layer_meta_bytes),
        len(indices_bytes),
        len(latent_bytes),
        len(meta_bytes),
    )
    return (
        header
        + cb_blob
        + layer_meta_bytes
        + indices_bytes
        + latent_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> SelfCompressNnArchive:
    """Parse the ``0.bin`` bytes back into the δ substrate components.

    Pure-bytes function — no model class needed. inflate.py imports this +
    the model class + builds + reconstructs weights from codebook+indices +
    renders, in ~170 LOC total.
    """
    if len(blob) < SCV1_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {SCV1_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_layers,
        num_pairs,
        latent_dim,
        K,
        D_v,
        cb_len,
        layer_meta_len,
        indices_len,
        latent_len,
        meta_len,
    ) = struct.unpack(SCV1_HEADER_FMT, blob[:SCV1_HEADER_SIZE])
    if magic != SCV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {SCV1_MAGIC!r})")
    if version != SCV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_latent_bytes = num_pairs * latent_dim * 2
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"latent_len {latent_len} != num_pairs*latent_dim*2 = {expected_latent_bytes}"
        )

    end_header = SCV1_HEADER_SIZE
    end_cb = end_header + cb_len
    end_layer_meta = end_cb + layer_meta_len
    end_indices = end_layer_meta + indices_len
    end_latents = end_indices + latent_len
    end_meta = end_latents + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    cb_blob = blob[end_header:end_cb]
    layer_meta_blob = blob[end_cb:end_layer_meta]
    indices_blob = blob[end_layer_meta:end_indices]
    latent_blob = blob[end_indices:end_latents]
    meta_blob = blob[end_latents:end_meta]

    codebook = _deserialize_codebook(cb_blob)
    if tuple(codebook.shape) != (K, D_v):
        raise ValueError(
            f"codebook shape {tuple(codebook.shape)} != header {(K, D_v)}"
        )
    layer_meta = json.loads(layer_meta_blob.decode("utf-8"))
    if not isinstance(layer_meta, list) or len(layer_meta) != num_layers:
        raise ValueError(
            f"layer_meta must be a list of len {num_layers}; "
            f"got {type(layer_meta).__name__} len {len(layer_meta) if isinstance(layer_meta, list) else '-'}"
        )
    layer_cluster_indices = _unpack_indices_blob(indices_blob, layer_meta)

    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np
    q_lat = torch.from_numpy(
        np.frombuffer(latent_blob, dtype=np.int16).copy()
    ).view(num_pairs, latent_dim)
    lat_scale = float(meta.pop("_lat_quant_scale"))
    lat_zp = float(meta.pop("_lat_quant_zero_point"))
    latents = _dequantize_from_int16(q_lat, lat_scale, lat_zp)

    return SelfCompressNnArchive(
        codebook=codebook,
        layer_cluster_indices=layer_cluster_indices,
        layer_meta=layer_meta,
        latents=latents,
        meta=meta,
        schema_version=int(version),
    )
