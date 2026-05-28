"""Compact inflater-side codec for the PR #101 HNeRV-microcodec source payload.

This stores the fixed model schema in code and keeps all video-specific payload
inside archive.zip (member `x`, inside the PR #101 source-payload region, *not*
including the locally appended FEC6 selector wrapper):

  decoder: concatenated Brotli streams of q-bytes + fp16 scale per tensor (PR #101 grammar)
  latents: raw LZMA(fp16 min/scale per dim + centered temporal-delta uint8 latent codes) (PR #101 grammar)
  sidecar: Brotli((u8 dim, i8 delta_x100) per frame pair) (PR #101 grammar)

The HNeRV decoder architecture itself (model.py) originates in PR #95 by
@AaronLeslie138 and is byte-identical across PR #95 / PR #98 / PR #101 / this
packet. PR #101 by @SajayR is the immediate byte substrate for this packet.
"""
import io
import lzma

import brotli
import numpy as np
import torch

from codec_sidecar import apply_latent_sidecar
from model import HNeRVDecoder


DECODER_BLOB_LEN = 162_164
LATENT_BLOB_LEN = 15_387
N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)
LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]

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
    idx: tuple(np.argsort(perm)) for idx, perm in CONV4_STORAGE_PERMS.items()
}

DECODER_BYTE_MAPS = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}

LATENT_DIM_ORDER = (
    26, 0, 17, 15, 10, 24, 20, 12, 14, 21, 22, 18, 4, 11,
    3, 7, 16, 2, 6, 8, 19, 23, 5, 9, 1, 13, 27, 25,
)
def zigzag_decode_u8(arr_u8):
    """Map unsigned zigzag symbols back to signed int8 residuals."""
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def decode_mapped_u8(arr_u8, byte_map):
    """Decode one stored uint8 tensor stream using its declared byte map."""
    if byte_map == "zig":
        return zigzag_decode_u8(arr_u8)
    if byte_map == "negzig":
        return (-zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise ValueError(f"unknown decoder byte map: {byte_map}")


def decompress_brotli_streams(data, n_streams):
    """Concatenate n independent Brotli streams from a compact payload."""
    outputs = []
    pos = 0
    for _ in range(n_streams):
        dec = brotli.Decompressor()
        chunks = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos:pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated compact decoder payload")
        outputs.append(b"".join(chunks))
    if pos != len(data):
        raise ValueError("trailing compact decoder payload")
    return b"".join(outputs)


def decode_decoder_compact(data):
    """Decode the compact HNeRV state_dict without changing tensor order."""
    raw = decompress_brotli_streams(data, len(DECODER_STREAM_ENDS))
    probe = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    items = list(probe.state_dict().items())
    pos = 0
    sd = {}

    for idx in DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0]
        pos += 2

        q = decode_mapped_u8(zz, DECODER_BYTE_MAPS.get(idx, "zig"))
        if len(shape) == 4:
            storage_perm = CONV4_STORAGE_PERMS[idx]
            inverse_perm = CONV4_INVERSE_PERMS[idx]
            stored_shape = tuple(shape[i] for i in storage_perm)
            q = q.reshape(stored_shape)
            q = np.transpose(q, inverse_perm).copy()
        else:
            q = q.reshape(shape)
        sd[name] = torch.from_numpy(q.astype(np.float32)) * float(scale)

    if pos != len(raw):
        raise ValueError("trailing or truncated compact decoder payload")
    return sd


def decode_latents_compact(data):
    """Decode LZMA-compressed per-pair latent codes into float tensors."""
    raw = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    buf = io.BytesIO(raw)
    mins = torch.from_numpy(
        np.frombuffer(buf.read(LATENT_DIM * 2), dtype=np.float16).copy()
    ).float()
    scales = torch.from_numpy(
        np.frombuffer(buf.read(LATENT_DIM * 2), dtype=np.float16).copy()
    ).float()
    stored = np.frombuffer(buf.read(N_PAIRS * LATENT_DIM), dtype=np.uint8)
    if stored.size != N_PAIRS * LATENT_DIM:
        raise ValueError("truncated compact latent payload")
    delta_ordered = stored.reshape(LATENT_DIM, N_PAIRS)
    q_ordered = delta_ordered.copy()
    q_ordered[:, 1:] = np.cumsum(
        ((delta_ordered[:, 1:].astype(np.int16) - 128) & 255),
        axis=1,
        dtype=np.uint16,
    ).astype(np.uint8) + delta_ordered[:, :1]
    q_ordered = q_ordered.T.copy()
    q = np.empty((N_PAIRS, LATENT_DIM), dtype=np.uint8)
    q[:, LATENT_DIM_ORDER] = q_ordered
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def parse_archive(archive_bytes):
    """Parse archive-local bytes into decoder state, latents, and metadata."""
    decoder_blob = archive_bytes[:DECODER_BLOB_LEN]
    latent_blob = archive_bytes[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    if not decoder_blob or not latent_blob:
        raise ValueError("bad compact archive")
    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decode_decoder_compact(decoder_blob), latents, meta
