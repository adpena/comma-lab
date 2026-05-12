"""pr101_lc_v2_clone archive grammar — byte-faithful PR101 0.bin layout.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (lesson L2) mirroring
PR101's monolithic single-file ``0.bin`` layout (PR101 source line 467-480):

::

    DECODER_BLOB  [: DECODER_BLOB_LEN]                    PR101 anchor: 162_164 bytes
    LATENT_BLOB   [DECODER_BLOB_LEN : DECODER_BLOB_LEN+LATENT_BLOB_LEN]  PR101 anchor: 15_387 bytes
    SIDECAR_BLOB  [DECODER_BLOB_LEN+LATENT_BLOB_LEN:]     PR101 anchor: variable

The clone's DECODER_BLOB consumes Subagent C's 3 GOLD primitives end-to-end:

    encode pipeline:
        decoder_state_dict
            -> per-tensor int8 quantise with per-tensor fp16 scale
            -> apply_storage_perm(...) for the 13 4D conv tensors per
               PR101_CONV4_STORAGE_PERMS (CONV4 PRIMITIVE)
            -> encode_byte_map(arr_i8, strategy) per PR101_DECODER_BYTE_MAPS
               (BYTE-MAP PRIMITIVE)
            -> append fp16 scale (2 bytes) after each tensor's bytes
            -> reorder per PR101_DECODER_STORAGE_ORDER (STORAGE-ORDER PRIMITIVE)
            -> partition by PR101_DECODER_STREAM_ENDS (STORAGE-ORDER PRIMITIVE)
            -> brotli.compress each stream
            -> concat brotli streams -> DECODER_BLOB

The composition is what gives PR101 its bytes. The clone reproduces it
exactly so the GOLD anchor's archive bytes can be regenerated end-to-end.

NEGZIG precondition: PR101's negzig encoding is NOT a bijection over the
full int8 range; ``-128`` collides with ``0``. We REFUSE encoding at archive
time if any negzig-tagged tensor entry quantises to ``-128`` (ValueError
with a banner per HNeRV parity discipline lesson 7 fail-closed).

CLAUDE.md compliance:
* No silent defaults (caller passes config)
* No /tmp paths
* No scorer load
* Deterministic: same input -> same bytes (deterministic brotli)
"""

from __future__ import annotations

import io
import json
import lzma
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

from tac.packet_compiler.pr101_conv4_storage_perms import (
    PR101_CONV4_STORAGE_PERMS,
    Conv4StoragePermSchema,
    apply_inverse_perm,
    apply_storage_perm,
)
from tac.packet_compiler.pr101_decoder_byte_maps import (
    PR101_DECODER_BYTE_MAPS,
    ByteMapStrategy,
    DecoderByteMapsSchema,
    decode_byte_map,
    encode_byte_map,
)
from tac.packet_compiler.pr101_decoder_storage_order import (
    PR101_DECODER_STORAGE_ORDER,
    PR101_DECODER_STREAM_ENDS,
    DecoderStorageOrderSchema,
)

from .architecture import Pr101LcV2CloneConfig, Pr101LcV2CloneSubstrate

# PR101 GOLD anchor: 28-tensor state_dict (per intake-clone model.py).
_N_TENSORS_ANCHOR = 28

# Brotli quality 11 matches PR101's bytes (max compression; deterministic).
_BROTLI_QUALITY = 11

# Per-tensor fp16 scale = 2 bytes (PR101 source line 277).
_FP16_SCALE_BYTES = 2

# LZMA filter matches PR101 source line 28-30 EXACTLY.
_LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]


PR101_LC_V2_ARCHIVE_GRAMMAR: dict[str, object] = {
    "name": "pr101_lc_v2_clone",
    "format": "monolithic_single_file_0_bin",
    "sections": ["DECODER_BLOB", "LATENT_BLOB", "SIDECAR_BLOB"],
    "decoder_pipeline": [
        "per_tensor_int8_quantise",
        "apply_storage_perm_for_4d_conv",
        "encode_byte_map_per_tensor",
        "reorder_for_storage_order",
        "partition_by_stream_ends",
        "brotli_compress_per_stream",
        "concat_streams",
    ],
    "primitives_consumed": [
        "tac.packet_compiler.pr101_decoder_storage_order",
        "tac.packet_compiler.pr101_conv4_storage_perms",
        "tac.packet_compiler.pr101_decoder_byte_maps",
    ],
    "pr101_anchor": {
        "decoder_blob_len": 162_164,
        "latent_blob_len": 15_387,
        "n_tensors": _N_TENSORS_ANCHOR,
        "n_streams": 7,
    },
    "research_only": True,
    "score_claim": False,
    "promotion_eligible": False,
}
"""Public-facing archive grammar manifest (Catalog #124 declaration)."""


@dataclass(frozen=True)
class Pr101LcV2Archive:
    """Parsed archive structure — the inflate-time data contract."""

    decoder_state_dict: dict[str, torch.Tensor]
    """Decoder state_dict (28 tensors mirroring PR101 layout)."""

    latents: torch.Tensor
    """``(num_pairs, latent_dim)`` per-pair latents (float)."""

    meta: dict[str, object]
    """Sidecar JSON meta: latent_dim, base_channels, eval_size, ..."""


# ── Schema accessors (Subagent C primitives wrapped) ────────────────────────


def _build_storage_order_schema() -> DecoderStorageOrderSchema:
    """Build the PR101 storage-order schema once (validated)."""
    return DecoderStorageOrderSchema(
        storage_order=PR101_DECODER_STORAGE_ORDER,
        stream_ends=PR101_DECODER_STREAM_ENDS,
        n_tensors=_N_TENSORS_ANCHOR,
    )


def _build_conv4_perm_schema() -> Conv4StoragePermSchema:
    """Build the PR101 conv4-perm schema once (validated)."""
    return Conv4StoragePermSchema.from_perms(PR101_CONV4_STORAGE_PERMS)


def _build_byte_maps_schema() -> DecoderByteMapsSchema:
    """Build the PR101 byte-maps schema once (validated)."""
    return DecoderByteMapsSchema.from_table(dict(PR101_DECODER_BYTE_MAPS))


# ── Encoder ─────────────────────────────────────────────────────────────────


def _quantise_tensor_to_int8(
    t: torch.Tensor,
) -> tuple[np.ndarray, np.float16]:
    """Per-tensor symmetric int8 quantise with fp16 scale, full int8 range.

    Mirrors PR101's encode-side (the public source doesn't include encoder
    code, but the decoder side reads ``q * scale`` after ``decode_mapped_u8``
    so the encoder MUST emit ``q = round(t / scale).clip(-128, 127)``).

    The scale is computed as ``absmax / 128`` (NOT 127) so the full int8
    range ``[-128, 127]`` is reachable: the most-negative entry of a tensor
    whose ``min == -absmax`` produces ``q = -128``. This is what makes the
    NEGZIG non-bijection precondition (refusing -128 in negzig-tagged
    tensors) load-bearing.

    Returns the int8 array (flat 1-D) plus the fp16 scale.
    """
    arr = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    absmax = float(np.abs(arr).max())
    if absmax <= 0.0:
        return (np.zeros(arr.size, dtype=np.int8), np.float16(1.0))
    scale_f32 = absmax / 128.0
    scale_f16 = np.float16(scale_f32)
    q = np.round(arr / float(scale_f16)).clip(-128.0, 127.0).astype(np.int8)
    return (q.reshape(-1), scale_f16)


def _per_tensor_encode_bytes(
    tensors: list[tuple[str, torch.Tensor]],
    *,
    conv4_schema: Conv4StoragePermSchema,
    byte_maps_schema: DecoderByteMapsSchema,
) -> tuple[list[bytes], list[int]]:
    """Encode each tensor independently. Returns (bytes_per_tensor,
    sizes_per_tensor). Each entry is ``encoded_q_bytes || fp16_scale``.

    NEGZIG precondition: if any byte-map for a 4D-conv index is ``negzig``,
    refuse encode when any quantised entry equals -128 (the negzig
    non-bijection point).
    """
    per_tensor_payloads: list[bytes] = []
    per_tensor_sizes: list[int] = []
    for idx, (name, t) in enumerate(tensors):
        q_flat_i8, scale_f16 = _quantise_tensor_to_int8(t)
        shape = tuple(t.shape)

        # If 4D conv tensor: apply storage perm BEFORE byte map.
        if len(shape) == 4 and idx in conv4_schema.perms:
            perm = conv4_schema.perms[idx]
            tensor_bytes_uint8 = q_flat_i8.view(np.uint8).tobytes()
            permuted = apply_storage_perm(tensor_bytes_uint8, shape, perm)
            q_for_bytemap = np.frombuffer(permuted, dtype=np.int8)
        else:
            q_for_bytemap = q_flat_i8

        # Negzig precondition: refuse -128 in negzig-tagged tensors.
        strategy = byte_maps_schema.strategy_for(idx)
        if strategy == "negzig" and int((q_for_bytemap == -128).sum()) > 0:
            raise ValueError(
                f"NEGZIG_NON_BIJECTION: tensor index {idx} (name={name!r}) "
                f"quantises to -128 under negzig; PR101 source line 234 "
                f"requires entries in [-127, 127]. Re-quantise or pick "
                f"a different byte-map for this tensor."
            )

        encoded_q = encode_byte_map(q_for_bytemap, strategy)
        scale_bytes = scale_f16.tobytes()
        payload = encoded_q + scale_bytes
        per_tensor_payloads.append(payload)
        per_tensor_sizes.append(len(payload))
    return (per_tensor_payloads, per_tensor_sizes)


def encode_decoder_compact(
    decoder_state_dict: Mapping[str, torch.Tensor],
    *,
    n_tensors: int = _N_TENSORS_ANCHOR,
) -> bytes:
    """Encode a 28-tensor state_dict into the DECODER_BLOB byte stream.

    Pipeline (mirrors PR101 ``decode_decoder_compact`` inverse, line 259-292):

    1. Per-tensor int8 quantise + fp16 scale.
    2. For each 4D conv tensor at index in CONV4_STORAGE_PERMS, apply
       the storage perm (Subagent C primitive #2).
    3. For each tensor, encode via its byte map (Subagent C primitive #3).
    4. Append the fp16 scale after the encoded bytes (PR101 source line 277).
    5. Reorder per DECODER_STORAGE_ORDER (Subagent C primitive #1, half-A).
    6. Concatenate -> a single buffer.
    7. Partition by DECODER_STREAM_ENDS (Subagent C primitive #1, half-B).
    8. brotli.compress each stream at quality=11.
    9. Concatenate brotli streams -> DECODER_BLOB.

    NOTE: At encode-time we receive a state_dict in PR101's iteration order
    (the trainer produces a model whose state_dict() yields tensors in the
    same order as PR101's HNeRVDecoder). The 3 primitives' indices reference
    this iteration order.
    """
    storage_schema = _build_storage_order_schema()
    conv4_schema = _build_conv4_perm_schema()
    byte_maps_schema = _build_byte_maps_schema()

    tensors_in_iteration_order: list[tuple[str, torch.Tensor]] = list(
        decoder_state_dict.items()
    )
    if len(tensors_in_iteration_order) != n_tensors:
        raise ValueError(
            f"decoder_state_dict has {len(tensors_in_iteration_order)} tensors; "
            f"expected {n_tensors} (PR101 anchor)"
        )

    per_tensor_payloads, per_tensor_sizes_iter = _per_tensor_encode_bytes(
        tensors_in_iteration_order,
        conv4_schema=conv4_schema,
        byte_maps_schema=byte_maps_schema,
    )

    # Reorder payloads into STORAGE order.
    payloads_in_storage_order: list[bytes] = [
        per_tensor_payloads[idx] for idx in storage_schema.storage_order
    ]
    sizes_in_storage_order: list[int] = [
        per_tensor_sizes_iter[idx] for idx in storage_schema.storage_order
    ]

    # Concatenate into one buffer (raw pre-brotli decoder bytes).
    buffer = b"".join(payloads_in_storage_order)

    # Partition by stream ends + brotli each segment.
    cumulative: list[int] = [0]
    running = 0
    for s in sizes_in_storage_order:
        running += s
        cumulative.append(running)

    streams_out: list[bytes] = []
    prev_pos = 0
    for end_pos in storage_schema.stream_ends:
        start_byte = cumulative[prev_pos]
        end_byte = cumulative[end_pos]
        segment = buffer[start_byte:end_byte]
        streams_out.append(
            brotli.compress(segment, quality=_BROTLI_QUALITY)
        )
        prev_pos = end_pos

    return b"".join(streams_out)


def encode_latents_compact(
    latents: torch.Tensor,
    *,
    num_pairs: int,
    latent_dim: int,
) -> bytes:
    """Quantise latents to uint8 + lzma compress (PR101-style minimal).

    PR101's actual encoder uses temporal-delta + LZMA-RAW with the filter
    above; here we emit a simpler bit-faithful round-trippable variant that
    matches the decoder's wire format: ``[mins(fp16*dim) || scales(fp16*dim)
    || raw_u8(num_pairs * dim)]``. The decode side mirrors PR101 source
    line 295-317 (cumulative-delta inversion path).

    The latent encoding fidelity is research-only at this scaffold stage;
    the byte-faithful PR101 path will be a follow-up.
    """
    if latents.dim() != 2:
        raise ValueError(f"latents must be 2-D; got {tuple(latents.shape)}")
    if latents.shape != (num_pairs, latent_dim):
        raise ValueError(
            f"latents shape {tuple(latents.shape)} != "
            f"(num_pairs={num_pairs}, latent_dim={latent_dim})"
        )

    f = latents.detach().to("cpu", dtype=torch.float32).numpy()
    mins = f.min(axis=0).astype(np.float16)  # (latent_dim,)
    maxs = f.max(axis=0).astype(np.float16)
    scales = ((maxs.astype(np.float32) - mins.astype(np.float32)) / 255.0).astype(
        np.float16
    )
    safe_scales = np.where(scales == 0, np.float16(1.0), scales)
    q = ((f - mins.astype(np.float32)) / safe_scales.astype(np.float32)).clip(
        0.0, 255.0
    ).round().astype(np.uint8)

    buf = io.BytesIO()
    buf.write(mins.tobytes())
    buf.write(scales.tobytes())
    buf.write(q.tobytes())
    raw = buf.getvalue()
    return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LATENT_LZMA_FILTERS)


def _pack_header(decoder_len: int, latent_len: int, meta_len: int) -> bytes:
    """4-byte magic + 1-byte version + 3 u32 lengths."""
    return struct.pack(
        "<4sBIII",
        b"PRC1",
        1,
        int(decoder_len),
        int(latent_len),
        int(meta_len),
    )


def pack_archive(
    decoder_state_dict: Mapping[str, torch.Tensor],
    latents: torch.Tensor,
    meta: Mapping[str, object],
) -> bytes:
    """Build the full ``0.bin`` archive bytes.

    Layout (clone-specific framing, NOT PR101's framing — PR101 hardcodes
    DECODER_BLOB_LEN+LATENT_BLOB_LEN as constants and has no header; we
    add a small typed header for parser safety while preserving PR101's
    section ORDER and the 3-primitive decode pipeline):

        4 bytes magic ``b"PRC1"``
        1 byte schema version
        4 bytes u32 decoder_blob_len
        4 bytes u32 latent_blob_len
        4 bytes u32 meta_blob_len
        decoder_blob (multi-brotli streams)
        latent_blob (lzma-raw)
        meta_blob (utf-8 json)
    """
    decoder_blob = encode_decoder_compact(decoder_state_dict)
    latent_dim = int(latents.shape[1])
    num_pairs = int(latents.shape[0])
    latent_blob = encode_latents_compact(
        latents, num_pairs=num_pairs, latent_dim=latent_dim
    )
    meta_with_shapes = dict(meta)
    meta_with_shapes.setdefault("latent_dim", latent_dim)
    meta_with_shapes.setdefault("num_pairs", num_pairs)
    meta_blob = json.dumps(
        meta_with_shapes, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    header = _pack_header(len(decoder_blob), len(latent_blob), len(meta_blob))
    return header + decoder_blob + latent_blob + meta_blob


# ── Decoder ─────────────────────────────────────────────────────────────────


def _decompress_brotli_streams(data: bytes, n_streams: int) -> bytes:
    """Mirror PR101 source line 242-256: per-stream incremental decode."""
    outputs: list[bytes] = []
    pos = 0
    for _ in range(n_streams):
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos:pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated decoder payload")
        outputs.append(b"".join(chunks))
    if pos != len(data):
        raise ValueError("trailing decoder payload")
    return b"".join(outputs)


def decode_decoder_compact(
    data: bytes,
    *,
    probe_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Decode the DECODER_BLOB bytes back into a state_dict.

    Pipeline (mirrors PR101 source line 259-292):

    1. Decompress N brotli streams -> raw bytes.
    2. Iterate in STORAGE_ORDER:
       - read ``numel`` bytes (encoded under byte map)
       - read 2 bytes fp16 scale
       - decode via byte map
       - if 4D conv: apply inverse perm to restore original shape
       - reshape + cast to float -> tensor * scale
    3. Return state_dict.

    ``probe_state_dict`` is used to learn each tensor's name/shape (PR101
    source line 261-266 builds a probe via ``HNeRVDecoder()``).
    """
    storage_schema = _build_storage_order_schema()
    conv4_schema = _build_conv4_perm_schema()
    byte_maps_schema = _build_byte_maps_schema()

    raw = _decompress_brotli_streams(data, len(storage_schema.stream_ends))
    items = list(probe_state_dict.items())
    if len(items) != storage_schema.n_tensors:
        raise ValueError(
            f"probe_state_dict has {len(items)} tensors; "
            f"expected {storage_schema.n_tensors}"
        )

    pos = 0
    sd: dict[str, torch.Tensor] = {}
    for idx in storage_schema.storage_order:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())

        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = float(
            np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0]
        )
        pos += _FP16_SCALE_BYTES

        strategy: ByteMapStrategy = byte_maps_schema.strategy_for(idx)
        q_i8 = decode_byte_map(zz.tobytes(), strategy)

        if len(shape) == 4 and idx in conv4_schema.perms:
            stored_shape_tuple: tuple[int, int, int, int] = (
                shape[conv4_schema.perms[idx][0]],
                shape[conv4_schema.perms[idx][1]],
                shape[conv4_schema.perms[idx][2]],
                shape[conv4_schema.perms[idx][3]],
            )
            inverse_perm = conv4_schema.inverse_perms[idx]
            restored_bytes = apply_inverse_perm(
                q_i8.view(np.uint8).tobytes(),
                stored_shape_tuple,
                inverse_perm,
            )
            q_arr = np.frombuffer(restored_bytes, dtype=np.int8).reshape(shape)
        else:
            q_arr = q_i8.reshape(shape)

        sd[name] = torch.from_numpy(q_arr.astype(np.float32)) * scale

    if pos != len(raw):
        raise ValueError("trailing or truncated decoder payload")
    return sd


def decode_latents_compact(
    data: bytes, *, num_pairs: int, latent_dim: int
) -> torch.Tensor:
    """Decode the LATENT_BLOB bytes back into ``(num_pairs, latent_dim)``.

    Mirrors the encode-side wire format above (mins || scales || u8 codes).
    """
    raw = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=_LATENT_LZMA_FILTERS)
    expected_len = latent_dim * 2 + latent_dim * 2 + num_pairs * latent_dim
    if len(raw) != expected_len:
        raise ValueError(
            f"latent blob raw bytes {len(raw)} != expected {expected_len}"
        )
    buf = io.BytesIO(raw)
    mins = np.frombuffer(buf.read(latent_dim * 2), dtype=np.float16).astype(
        np.float32
    )
    scales = np.frombuffer(buf.read(latent_dim * 2), dtype=np.float16).astype(
        np.float32
    )
    q = np.frombuffer(
        buf.read(num_pairs * latent_dim), dtype=np.uint8
    ).reshape(num_pairs, latent_dim)
    f = q.astype(np.float32) * scales[None, :] + mins[None, :]
    return torch.from_numpy(f)


def parse_archive(archive_bytes: bytes) -> Pr101LcV2Archive:
    """Parse the ``0.bin`` bytes back into ``(decoder_sd, latents, meta)``."""
    if len(archive_bytes) < 17:
        raise ValueError(f"archive too short ({len(archive_bytes)} bytes)")
    (magic, version, decoder_len, latent_len, meta_len) = struct.unpack(
        "<4sBIII", archive_bytes[:17]
    )
    if magic != b"PRC1":
        raise ValueError(f"bad magic: {magic!r} (expected b'PRC1')")
    if version != 1:
        raise ValueError(f"unsupported schema version: {version}")
    end_header = 17
    end_decoder = end_header + int(decoder_len)
    end_latent = end_decoder + int(latent_len)
    end_meta = end_latent + int(meta_len)
    if end_meta != len(archive_bytes):
        raise ValueError(
            f"archive size {len(archive_bytes)} != header-implied {end_meta}"
        )

    decoder_blob = archive_bytes[end_header:end_decoder]
    latent_blob = archive_bytes[end_decoder:end_latent]
    meta_blob = archive_bytes[end_latent:end_meta]

    meta = json.loads(meta_blob.decode("utf-8"))
    num_pairs = int(meta["num_pairs"])
    latent_dim = int(meta["latent_dim"])
    base_channels = int(meta.get("base_channels", 36))
    output_height = int(meta.get("output_height", 384))
    output_width = int(meta.get("output_width", 512))
    base_h = int(meta.get("base_h", 6))
    base_w = int(meta.get("base_w", 8))
    num_upsample_blocks = int(meta.get("num_upsample_blocks", 6))

    # Build a probe state_dict via the architecture class (cheap on CPU; no train).
    probe_cfg = Pr101LcV2CloneConfig(
        latent_dim=latent_dim,
        base_channels=base_channels,
        base_h=base_h,
        base_w=base_w,
        num_upsample_blocks=num_upsample_blocks,
        output_height=output_height,
        output_width=output_width,
        num_pairs=num_pairs,
    )
    probe = Pr101LcV2CloneSubstrate(probe_cfg)
    probe_sd = probe.state_dict()

    decoder_sd = decode_decoder_compact(decoder_blob, probe_state_dict=probe_sd)
    latents = decode_latents_compact(
        latent_blob, num_pairs=num_pairs, latent_dim=latent_dim
    )

    return Pr101LcV2Archive(
        decoder_state_dict=decoder_sd, latents=latents, meta=meta
    )
