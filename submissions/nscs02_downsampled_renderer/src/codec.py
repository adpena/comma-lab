"""NSCS02 standalone codec (mirror of tac.substrates.nscs02_downsampled_renderer.archive).

Per HNeRV parity discipline lesson 9 (runtime closure) the inflate
runtime tree must be self-contained. This module is a byte-faithful
duplicate of ``src/tac/substrates/nscs02_downsampled_renderer/archive.py``
with no ``tac.*`` imports. Parity is enforced by
``src/tac/tests/test_substrate_nscs02_downsampled_renderer.py::test_submission_codec_byte_parity_with_substrate_codec``.
"""
import io
import struct

import brotli
import numpy as np
import torch

from model import NSCS02Decoder

LATENT_DIM = 28
BASE_CHANNELS = 36
N_PAIRS = 600
RENDER_HW = (192, 256)
NSCS02_ARCHIVE_MAGIC = b"NSCS02\x00\x01"
HEADER_LEN = 16
WEIGHT_DTYPE = np.float16
LATENT_DTYPE = np.float16


def _inflate_fp16_stream_to_state_dict(stream, template_state_dict):
    buf = io.BytesIO(stream)

    def _read(n):
        chunk = buf.read(n)
        if len(chunk) != n:
            raise ValueError(f"NSCS02 archive: short read ({len(chunk)} of {n})")
        return chunk

    out = type(template_state_dict)()
    template_dtypes = {k: v.dtype for k, v in template_state_dict.items()}
    for _expected_name, template_tensor in template_state_dict.items():
        name_len = struct.unpack("B", _read(1))[0]
        name = _read(name_len).decode("utf-8")
        if name != _expected_name:
            raise ValueError(
                f"NSCS02 archive: tensor name order mismatch "
                f"(expected {_expected_name!r}, got {name!r})"
            )
        ndim = struct.unpack("B", _read(1))[0]
        shape = tuple(struct.unpack("<I", _read(4))[0] for _ in range(ndim))
        if shape != tuple(template_tensor.shape):
            raise ValueError(
                f"NSCS02 archive: tensor shape mismatch for {name} "
                f"(expected {tuple(template_tensor.shape)}, got {shape})"
            )
        numel = int(np.prod(shape)) if shape else 1
        nbytes = numel * np.dtype(WEIGHT_DTYPE).itemsize
        fp16 = np.frombuffer(_read(nbytes), dtype=WEIGHT_DTYPE).reshape(shape).copy()
        out[name] = torch.from_numpy(fp16).to(template_dtypes[name])
    if buf.tell() != len(stream):
        raise ValueError(
            f"NSCS02 archive: trailing bytes after weight stream "
            f"({len(stream) - buf.tell()} bytes remain)"
        )
    return out


def _inflate_fp16_latent_stream(stream):
    expected_nbytes = N_PAIRS * LATENT_DIM * np.dtype(LATENT_DTYPE).itemsize
    if len(stream) != expected_nbytes:
        raise ValueError(
            f"NSCS02 archive: latent stream byte mismatch "
            f"(expected {expected_nbytes}, got {len(stream)})"
        )
    arr = np.frombuffer(stream, dtype=LATENT_DTYPE).reshape(N_PAIRS, LATENT_DIM).copy()
    return torch.from_numpy(arr).to(torch.float32)


def parse_nscs02_archive_bytes(archive_bytes):
    """Decode NSCS02 archive bytes -> (state_dict, latents)."""
    if len(archive_bytes) < HEADER_LEN:
        raise ValueError(
            f"NSCS02 archive: too short for header ({len(archive_bytes)} < {HEADER_LEN})"
        )
    if archive_bytes[: len(NSCS02_ARCHIVE_MAGIC)] != NSCS02_ARCHIVE_MAGIC:
        raise ValueError(
            f"NSCS02 archive: magic mismatch "
            f"(expected {NSCS02_ARCHIVE_MAGIC!r}, got {archive_bytes[:8]!r})"
        )
    D = struct.unpack_from("<I", archive_bytes, 8)[0]
    L = struct.unpack_from("<I", archive_bytes, 12)[0]
    expected_total = HEADER_LEN + D + L
    if len(archive_bytes) != expected_total:
        raise ValueError(
            f"NSCS02 archive: total length mismatch "
            f"(expected {expected_total}, got {len(archive_bytes)})"
        )
    weight_blob = archive_bytes[HEADER_LEN : HEADER_LEN + D]
    latent_blob = archive_bytes[HEADER_LEN + D : HEADER_LEN + D + L]

    weight_stream = brotli.decompress(weight_blob)
    latent_stream = brotli.decompress(latent_blob)

    template = NSCS02Decoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        render_hw=RENDER_HW,
    )
    decoder_sd = _inflate_fp16_stream_to_state_dict(weight_stream, template.state_dict())
    latents = _inflate_fp16_latent_stream(latent_stream)
    return decoder_sd, latents
