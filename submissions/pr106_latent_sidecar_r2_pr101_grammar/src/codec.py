"""Compression codec for the HNeRV decoder + per-frame-pair latents.

Decoder path: per-tensor symmetric INT8 quantization → zigzag → concat with
shape/scale metadata → brotli (quality 11). We previously used a hybrid that
added per-tensor categorical AC (via the constriction crate) for big tensors —
that was ~217 bytes smaller (~+0.0001 to score) but added a Python dependency
and embedded a feature-level compression choice into the inflation path. Pure
brotli is the simpler, more transparent default.

Latent path: per-dim min/max scaling to [0, 254] (uint8), then 1st-order
temporal delta, zigzag to uint16, split into lo/hi byte streams (lo brotli's
well, hi is mostly zero). Beats plain brotli by ~240 bytes on our latents.

Round-trip verified bit-exact.
"""
import io
import struct

import brotli
import numpy as np
import torch

N_QUANT = 127


FIXED_STATE_SCHEMA = [
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
]


PACKED_STATE_SCHEMA = sorted(FIXED_STATE_SCHEMA, key=lambda item: -int(np.prod(item[1])))
DECODER_STORAGE_ORDER = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)
DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)
CONV4_STORAGE_PERMS = {
    2: (3, 0, 2, 1),
    4: (3, 0, 2, 1),
    6: (0, 1, 2, 3),
    8: (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}
CONV4_INVERSE_PERMS = {
    idx: tuple(int(value) for value in np.argsort(perm))
    for idx, perm in CONV4_STORAGE_PERMS.items()
}
DECODER_BYTE_MAPS = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}


def decode_fixed_decoder(data):
    """Decode the fixed HNeRV state schema."""
    raw = brotli.decompress(data)
    pos = 0
    sd = {}
    for name, shape in FIXED_STATE_SCHEMA:
        scale = struct.unpack_from("<f", raw, pos)[0]
        pos += 4
        size = int(np.prod(shape))
        q = zigzag_decode_u8(np.frombuffer(raw[pos:pos + size], dtype=np.uint8))
        pos += size
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    if pos != len(raw):
        raise ValueError("bad fixed decoder payload")
    return sd


def decode_packed_decoder(data):
    """Decode the packed fixed HNeRV state schema."""
    try:
        raw = brotli.decompress(data)
    except brotli.error as legacy_error:
        if data[:4] in (b"HDM3", b"HDM4"):
            raw = decode_hdm_decoder_raw(data)
        else:
            raw = decode_pr101_schema_decoder_raw(data, legacy_error=legacy_error)
    return decode_packed_decoder_raw(raw)


def decode_packed_decoder_raw(raw):
    """Decode raw q-stream + f32-scale bytes into the packed fixed schema."""
    pos = 0
    quantized = []
    for _, shape in PACKED_STATE_SCHEMA:
        size = int(np.prod(shape))
        quantized.append(zigzag_decode_u8(np.frombuffer(raw[pos:pos + size], dtype=np.uint8)))
        pos += size
    scales_pos = pos
    expected = scales_pos + 4 * len(PACKED_STATE_SCHEMA)
    if expected != len(raw):
        raise ValueError("bad packed decoder payload")
    sd = {}
    for index, (name, shape) in enumerate(PACKED_STATE_SCHEMA):
        scale = struct.unpack_from("<f", raw, scales_pos + 4 * index)[0]
        sd[name] = torch.from_numpy(quantized[index].astype(np.float32).reshape(shape)) * scale
    return sd


def decode_pr101_schema_decoder_raw(data, *, legacy_error=None):
    """Decode PR101 schema-split decoder streams into PR106 packed raw bytes.

    This is a fail-closed runtime adapter: legacy PR106 single-Brotli remains
    the default path, and this fallback is used only when legacy Brotli decode
    rejects the decoder section.
    """
    try:
        schema_raw = decompress_concatenated_brotli_streams(data, len(DECODER_STREAM_ENDS))
    except ValueError as exc:
        prefix = "decoder section is neither legacy PR106 Brotli nor PR101 schema-split"
        if legacy_error is not None:
            prefix += f"; legacy_error={legacy_error}"
        raise ValueError(f"{prefix}; schema_error={exc}") from exc

    cursor = 0
    records_by_name = {}
    for fixed_index in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[fixed_index]
        value_count = int(np.prod(shape))
        mapped = np.frombuffer(
            read_exact(schema_raw, cursor, value_count, f"{name}:q"),
            dtype=np.uint8,
        )
        cursor += value_count
        scale_f32 = read_exact(schema_raw, cursor, 4, f"{name}:scale")
        cursor += 4
        q_storage = decode_mapped_u8(mapped, DECODER_BYTE_MAPS.get(fixed_index, "zig"))
        if len(shape) == 4:
            storage_perm = CONV4_STORAGE_PERMS[fixed_index]
            inverse_perm = CONV4_INVERSE_PERMS[fixed_index]
            stored_shape = tuple(shape[index] for index in storage_perm)
            q_original = np.transpose(q_storage.reshape(stored_shape), inverse_perm).copy()
        else:
            q_original = q_storage.reshape(shape)
        records_by_name[name] = (q_original.reshape(-1), scale_f32)
    if cursor != len(schema_raw):
        raise ValueError("PR101 schema decoder has trailing raw bytes")

    raw_parts = []
    scale_parts = []
    for name, shape in PACKED_STATE_SCHEMA:
        record = records_by_name.get(name)
        if record is None:
            raise ValueError(f"missing decoded schema record: {name}")
        q_original, scale_f32 = record
        if q_original.size != int(np.prod(shape)):
            raise ValueError(f"decoded schema shape mismatch for {name}")
        raw_parts.append(zigzag_encode_i8(q_original.astype(np.int8)).tobytes())
        scale_parts.append(scale_f32)
    return b"".join(raw_parts + scale_parts)


def decode_hdm_decoder_raw(data):
    """Decode HDM3/HDM4 fixed-schema q-Brotli/raw-scale decoder bytes.

    HDM3 and HDM4 are lossless section recodes for the same packed HNeRV
    decoder contract. Unknown or malformed HDM bytes fail closed.
    """
    if data[:4] == b"HDM3":
        return decode_hdm3_decoder_raw(data)
    if data[:4] == b"HDM4":
        return decode_hdm4_decoder_raw(data)
    raise ValueError("invalid HDM decoder magic")


def decode_hdm3_decoder_raw(data):
    """Decode HDM3 fixed-schema q-Brotli/raw-scale decoder bytes."""
    if data[:4] != b"HDM3":
        raise ValueError("invalid HDM3 decoder magic")
    cursor = 4
    compressed_len = int.from_bytes(
        read_exact(data, cursor, 3, "HDM3 q_brotli_len24"),
        "little",
    )
    cursor += 3
    compressed = read_exact(data, cursor, compressed_len, "HDM3 q_brotli")
    cursor += compressed_len
    scale_len = 4 * len(PACKED_STATE_SCHEMA)
    scale_stream = read_exact(data, cursor, scale_len, "HDM3 scale_stream")
    cursor += scale_len
    if cursor != len(data):
        raise ValueError("HDM3 decoder has trailing bytes")
    try:
        q_stream = brotli.decompress(compressed)
    except brotli.error as exc:
        raise ValueError(f"HDM3 q stream brotli decode failed: {exc}") from exc
    expected_q_len = sum(int(np.prod(shape)) for _, shape in PACKED_STATE_SCHEMA)
    if len(q_stream) != expected_q_len:
        raise ValueError("HDM3 q stream length mismatch")
    return q_stream + scale_stream


def decode_hdm4_decoder_raw(data):
    """Decode HDM4 fixed-recipe split q-Brotli/raw-scale decoder bytes."""
    if data[:4] != b"HDM4":
        raise ValueError("invalid HDM4 decoder magic")
    cursor = 4
    recipe_id = read_exact(data, cursor, 1, "HDM4 recipe_id")[0]
    cursor += 1
    if recipe_id != 1:
        raise ValueError(f"unsupported HDM4 recipe id: {recipe_id}")
    split_points = (6, 9, 26, 28)
    lengths = []
    for index in range(len(split_points)):
        lengths.append(
            int.from_bytes(
                read_exact(data, cursor, 3, f"HDM4 q_brotli_len24_{index}"),
                "little",
            )
        )
        cursor += 3
    chunks = []
    for index, length in enumerate(lengths):
        compressed = read_exact(data, cursor, length, f"HDM4 q_brotli_{index}")
        cursor += length
        try:
            chunks.append(brotli.decompress(compressed))
        except brotli.error as exc:
            raise ValueError(f"HDM4 q stream chunk {index} brotli decode failed: {exc}") from exc
    scale_len = 4 * len(PACKED_STATE_SCHEMA)
    scale_stream = read_exact(data, cursor, scale_len, "HDM4 scale_stream")
    cursor += scale_len
    if cursor != len(data):
        raise ValueError("HDM4 decoder has trailing bytes")

    ordered_schema = sorted(
        PACKED_STATE_SCHEMA,
        key=lambda item: (
            0 if len(item[1]) == 4 and item[1][2:] == (3, 3)
            else 1 if len(item[1]) == 4
            else 2 if item[0].endswith(".bias")
            else 3,
            -int(np.prod(item[1])),
            item[0],
        ),
    )
    records_by_name = {}
    split_start = 0
    for chunk, split_end in zip(chunks, split_points):
        schema_slice = ordered_schema[split_start:split_end]
        expected = sum(int(np.prod(shape)) for _, shape in schema_slice)
        if len(chunk) != expected:
            raise ValueError("HDM4 q chunk length mismatch")
        q_cursor = 0
        for name, shape in schema_slice:
            value_count = int(np.prod(shape))
            records_by_name[name] = chunk[q_cursor:q_cursor + value_count]
            q_cursor += value_count
        split_start = split_end
    if len(records_by_name) != len(PACKED_STATE_SCHEMA):
        raise ValueError("HDM4 decoded record count mismatch")
    raw_parts = []
    for name, shape in PACKED_STATE_SCHEMA:
        q = records_by_name.get(name)
        if q is None or len(q) != int(np.prod(shape)):
            raise ValueError(f"HDM4 decoded schema mismatch for {name}")
        raw_parts.append(q)
    return b"".join(raw_parts) + scale_stream


def decompress_concatenated_brotli_streams(payload, n_streams):
    outputs = []
    cursor = 0
    for _ in range(n_streams):
        decoder = brotli.Decompressor()
        chunks = []
        try:
            while cursor < len(payload) and not decoder.is_finished():
                chunks.append(decoder.process(payload[cursor:cursor + 1]))
                cursor += 1
        except brotli.error as exc:
            raise ValueError("invalid PR101 schema Brotli stream") from exc
        if not decoder.is_finished():
            raise ValueError("truncated PR101 schema Brotli stream")
        outputs.append(b"".join(chunks))
    if cursor != len(payload):
        raise ValueError("trailing PR101 schema Brotli stream bytes")
    return b"".join(outputs)


def decode_mapped_u8(values, byte_map):
    if byte_map == "zig":
        return zigzag_decode_u8(values)
    if byte_map == "negzig":
        return (-zigzag_decode_u8(values).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (values.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return values.view(np.int8)
    raise ValueError(f"unknown decoder byte map: {byte_map}")


def read_exact(payload, cursor, size, label):
    end = cursor + size
    if end > len(payload):
        raise ValueError(f"truncated PR101 schema decoder at {label}")
    return payload[cursor:end]


def decode_fixed_latents(data):
    """Decode fixed 600x28 latent payload."""
    raw = brotli.decompress(data)
    n, d = 600, 28
    meta_len = d * 4
    total = n * d
    lo = np.frombuffer(raw[:total], dtype=np.uint8).astype(np.uint16)
    mins = torch.from_numpy(np.frombuffer(raw[total:total + d * 2], dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(raw[total + d * 2:total + meta_len], dtype=np.float16).copy()).float()
    hi = np.frombuffer(raw[total + meta_len:total + meta_len + total], dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2,
                     -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def parse_fixed_archive(archive_bytes):
    """Parse fixed-schema HNeRV archive."""
    pos = 3
    dec_len = struct.unpack_from("<I", archive_bytes, pos)[0]
    pos += 4
    decoder_sd = decode_fixed_decoder(archive_bytes[pos:pos + dec_len])
    pos += dec_len
    lat_len = struct.unpack_from("<I", archive_bytes, pos)[0]
    pos += 4
    latents = decode_fixed_latents(archive_bytes[pos:pos + lat_len])
    if pos + lat_len != len(archive_bytes):
        raise ValueError("bad fixed archive")
    return decoder_sd, latents, {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}


def parse_packed_archive(archive_bytes):
    """Parse packed fixed-schema HNeRV archive."""
    dec_len = int.from_bytes(archive_bytes[1:4], "little")
    decoder_sd = decode_packed_decoder(archive_bytes[4:4 + dec_len])
    latents = decode_fixed_latents(archive_bytes[4 + dec_len:])
    return decoder_sd, latents, {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}


# ============================================================================
# Quantization
# ============================================================================

def quantize_state_dict(sd, n_quant=N_QUANT):
    """Per-tensor symmetric INT8 quant. Returns {name: (int8_flat_array, scale, shape)}."""
    out = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        m = t.abs().max().item()
        scale = m / n_quant if m > 0 else 1.0
        q = (t / scale).round().clamp(-n_quant, n_quant).to(torch.int8).numpy().flatten()
        out[name] = (q, scale, tuple(tensor.shape))
    return out


def zigzag_encode_i8(arr_i8):
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def zigzag_decode_u8(arr_u8):
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


# ============================================================================
# Decoder weights: pure brotli on the entire INT8-quantized state dict.
# We previously used a hybrid (per-tensor categorical AC for big tensors + brotli for small)
# but switched to pure brotli for simplicity — it's only ~217 bytes worse on our
# trained weights (~+0.0001 to score) and removes the constriction dependency.
# ============================================================================

def encode_decoder(q_sd):
    """Encode quantized state dict to compressed bytes via zigzag + brotli."""
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(q_sd)))
    for name, (q, scale, shape) in q_sd.items():
        nb = name.encode('utf-8')
        buf.write(struct.pack("<I", len(nb)))
        buf.write(nb)
        buf.write(struct.pack("<I", len(shape)))
        for s in shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<f", scale))
        buf.write(struct.pack("<I", q.size))
        buf.write(zigzag_encode_i8(q).tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def decode_decoder(data):
    """Inverse of encode_decoder. Returns {name: torch.Tensor (float32, dequantized)}."""
    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    n = struct.unpack("<I", buf.read(4))[0]
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl).decode('utf-8')
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8)
        q = zigzag_decode_u8(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


# ============================================================================
# Latents: delta + zigzag + brotli (lo/hi byte split)
# ============================================================================

def encode_latents(latents: torch.Tensor):
    """Encode (n_pairs, latent_dim) float tensor to bytes.

    Per-dim asymmetric UINT8 (min/max scaling to [0,254]) + 1st-order temporal
    delta + zigzag to uint16 + lo/hi byte split (lo brotli's well, hi mostly 0).
    """
    t = latents.detach().cpu().float()
    n, d = t.shape
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = ((t - mins.unsqueeze(0)) / scales.unsqueeze(0)).round().clamp(0, 254).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8).tobytes()
    hi = (delta_zz >> 8).astype(np.uint8).tobytes()
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += lo + hi
    return payload  # caller wraps in brotli


def decode_latents(raw):
    buf = io.BytesIO(raw)
    n, d = struct.unpack("<II", buf.read(8))
    mins = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz.astype(np.int32) // 2,
                     -(delta_zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


# ============================================================================
# Top-level archive: meta + decoder + latents
# ============================================================================

def build_archive(decoder_state_dict, latents, meta_dict):
    """Build the final archive blob.

    Layout:
      [meta_brotli_len:u32] [meta_brotli]
      [decoder_blob_len:u32] [decoder_blob]
      [latents_brotli_len:u32] [latents_brotli]
    """
    import json
    meta_raw = json.dumps(meta_dict).encode('utf-8')
    meta_brotli = brotli.compress(meta_raw, quality=11)

    q_sd = quantize_state_dict(decoder_state_dict)
    decoder_blob = encode_decoder(q_sd)

    latents_payload = encode_latents(latents)
    latents_brotli = brotli.compress(latents_payload, quality=11)

    out = io.BytesIO()
    out.write(struct.pack("<I", len(meta_brotli)))
    out.write(meta_brotli)
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
    out.write(struct.pack("<I", len(latents_brotli)))
    out.write(latents_brotli)
    return out.getvalue()


def parse_archive(archive_bytes):
    """Inverse of build_archive. Returns (decoder_sd, latents_tensor, meta_dict)."""
    if archive_bytes[:1] == b"\xff":
        return parse_packed_archive(archive_bytes)
    if archive_bytes.startswith(b"HN1"):
        return parse_fixed_archive(archive_bytes)
    import json
    buf = io.BytesIO(archive_bytes)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta = json.loads(brotli.decompress(buf.read(meta_len)))
    dec_len = struct.unpack("<I", buf.read(4))[0]
    decoder_sd = decode_decoder(buf.read(dec_len))
    lat_len = struct.unpack("<I", buf.read(4))[0]
    latents = decode_latents(brotli.decompress(buf.read(lat_len)))
    return decoder_sd, latents, meta
