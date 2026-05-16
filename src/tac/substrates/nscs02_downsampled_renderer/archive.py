# SPDX-License-Identifier: MIT
"""NSCS02 archive grammar — packed weights + per-pair latents.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
lessons 2 (export-first), 3 (monolithic single-file 0.bin), 4 (inflate
<= 100 LOC), 11 (no-op detector).

Per the standing directive
``feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md``
this archive is a UNIQUE-AND-COMPLETE design for the NSCS02 5-stage
decoder. NOT shared with A1's PR98-fine-tuned codec (which uses
specialized 28-stream Brotli + per-tensor permutations + Huffman
sidecar — overkill for NSCS02's smaller weight tensor count).

Wire format (single-file ``0.bin`` per HNeRV parity L3):

    Offset 0:                  8 bytes ``NSCS02_ARCHIVE_MAGIC``
    Offset 8:                  4 bytes uint32 LE ``decoder_blob_len`` (D)
    Offset 12:                 4 bytes uint32 LE ``latent_blob_len`` (L)
    Offset 16:                 D bytes brotli-compressed fp16 weight stream
    Offset 16 + D:             L bytes brotli-compressed fp16 latent stream
    Offset 16 + D + L:         (no trailing bytes)

Total wire-format header overhead = 16 bytes (magic + 2 length prefixes).
The brotli quality is fixed at level 11 (max) for byte-deterministic
encoding; no quality knob.

Parser-section manifest (declared at design time per Catalog #124):
- ``magic``: bytes 0-7
- ``decoder_blob_len``: bytes 8-11
- ``latent_blob_len``: bytes 12-15
- ``decoder_blob``: bytes 16 .. 16+D
- ``latent_blob``: bytes 16+D .. 16+D+L

Catalog #220 mechanism status:
  ``score_improvement_mechanism_status=RESEARCH_ONLY`` until the resizing-chain
  ablation and paired exact eval prove the low-resolution inflate path is
  score-relevant. The local byte-mutation smoke is parser/runtime evidence,
  not promotion evidence.
"""

from __future__ import annotations

import io
import struct
from collections import OrderedDict as _OrderedDictT
from dataclasses import dataclass
from typing import TYPE_CHECKING

import brotli
import numpy as np
import torch

from . import NSCS02_ARCHIVE_MAGIC, NSCS02_LATENT_DIM, NSCS02_N_PAIRS

if TYPE_CHECKING:
    from .architecture import NSCS02DownsampledDecoder

# Header constants — each field is fixed-width so the parser is trivial.
HEADER_LEN: int = 16  # 8 magic + 4 D + 4 L
BROTLI_QUALITY: int = 11  # max; deterministic
WEIGHT_DTYPE = np.float16
LATENT_DTYPE = np.float16


@dataclass(frozen=True)
class NSCS02ParsedArchive:
    """Parser output: decoder state-dict + latent tensor."""

    decoder_state_dict: _OrderedDictT[str, torch.Tensor]
    latents: torch.Tensor  # shape (N_PAIRS, latent_dim) on CPU
    decoder_blob_len: int
    latent_blob_len: int


def _flatten_state_dict_to_fp16(
    state_dict: _OrderedDictT[str, torch.Tensor],
) -> tuple[bytes, list[tuple[str, tuple[int, ...], int]]]:
    """Concatenate every tensor in state_dict into one fp16 byte stream.

    Returns ``(stream_bytes, manifest)`` where ``manifest[i] = (name,
    shape, numel_offset)``. The manifest is encoded into the stream
    via fixed-width prefix (per-tensor): 1-byte name length + name +
    1-byte shape ndim + ndim * 4-byte uint32 dims + tensor bytes.

    This is a UNIQUE serializer for NSCS02 — A1's codec uses 28
    specialized streams + per-tensor permutations + per-stream brotli;
    NSCS02's smaller decoder (~165K params, ~12 conv tensors) does
    not justify that complexity. A SINGLE concatenated fp16 stream
    + ONE brotli pass is the simplest faithful design.
    """
    out = io.BytesIO()
    manifest: list[tuple[str, tuple[int, ...], int]] = []
    for name, tensor in state_dict.items():
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError(f"NSCS02 archive: tensor name too long ({len(name_bytes)}): {name}")
        out.write(struct.pack("B", len(name_bytes)))
        out.write(name_bytes)
        shape = tuple(int(d) for d in tensor.shape)
        if len(shape) > 255:
            raise ValueError(f"NSCS02 archive: tensor ndim > 255: {name}")
        out.write(struct.pack("B", len(shape)))
        for dim in shape:
            if dim < 0 or dim > (1 << 31) - 1:
                raise ValueError(f"NSCS02 archive: dim out of range for {name}: {dim}")
            out.write(struct.pack("<I", dim))
        fp16 = tensor.detach().cpu().numpy().astype(WEIGHT_DTYPE).tobytes()
        out.write(fp16)
        manifest.append((name, shape, tensor.numel()))
    return out.getvalue(), manifest


def _inflate_fp16_stream_to_state_dict(
    stream: bytes,
    template_state_dict: _OrderedDictT[str, torch.Tensor],
) -> _OrderedDictT[str, torch.Tensor]:
    """Reverse of ``_flatten_state_dict_to_fp16``.

    Templated parsing: walks the stream and uses the supplied
    ``template_state_dict`` (the decoder's current ``.state_dict()``)
    to validate name + shape ordering. Returns a NEW state-dict with
    the same key-order as the template.
    """
    buf = io.BytesIO(stream)

    def _read(n: int) -> bytes:
        chunk = buf.read(n)
        if len(chunk) != n:
            raise ValueError(f"NSCS02 archive: short read ({len(chunk)} of {n})")
        return chunk

    out: _OrderedDictT[str, torch.Tensor] = type(template_state_dict)()
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
        # Promote back to the template's dtype (typically float32).
        out[name] = torch.from_numpy(fp16).to(template_dtypes[name])
    if buf.tell() != len(stream):
        raise ValueError(
            f"NSCS02 archive: trailing bytes after weight stream "
            f"({len(stream) - buf.tell()} bytes remain)"
        )
    return out


def _flatten_latents_to_fp16(latents: torch.Tensor) -> bytes:
    if latents.shape != (NSCS02_N_PAIRS, NSCS02_LATENT_DIM):
        raise ValueError(
            f"NSCS02 archive: latents must be ({NSCS02_N_PAIRS}, {NSCS02_LATENT_DIM}); "
            f"got {tuple(latents.shape)}"
        )
    return latents.detach().cpu().numpy().astype(LATENT_DTYPE).tobytes()


def _inflate_fp16_latent_stream(stream: bytes) -> torch.Tensor:
    expected_nbytes = NSCS02_N_PAIRS * NSCS02_LATENT_DIM * np.dtype(LATENT_DTYPE).itemsize
    if len(stream) != expected_nbytes:
        raise ValueError(
            f"NSCS02 archive: latent stream byte mismatch "
            f"(expected {expected_nbytes}, got {len(stream)})"
        )
    arr = np.frombuffer(stream, dtype=LATENT_DTYPE).reshape(NSCS02_N_PAIRS, NSCS02_LATENT_DIM).copy()
    return torch.from_numpy(arr).to(torch.float32)


def pack_nscs02_archive(
    decoder: NSCS02DownsampledDecoder,
    latents: torch.Tensor,
) -> bytes:
    """Encode NSCS02 archive bytes (suitable for ZIP-member ``0.bin``).

    Args:
        decoder: trained NSCS02 5-stage decoder.
        latents: per-pair latents shaped (N_PAIRS, latent_dim).

    Returns:
        Byte string laid out per the wire-format spec at module-doc.
    """
    sd = decoder.state_dict()
    weight_stream, _manifest = _flatten_state_dict_to_fp16(sd)
    weight_blob = brotli.compress(weight_stream, quality=BROTLI_QUALITY)

    latent_stream = _flatten_latents_to_fp16(latents)
    latent_blob = brotli.compress(latent_stream, quality=BROTLI_QUALITY)

    D = len(weight_blob)
    L = len(latent_blob)
    if D > (1 << 32) - 1 or L > (1 << 32) - 1:
        raise ValueError(f"NSCS02 archive: blob too large for uint32 (D={D}, L={L})")

    header = NSCS02_ARCHIVE_MAGIC + struct.pack("<I", D) + struct.pack("<I", L)
    return header + weight_blob + latent_blob


def parse_nscs02_archive(
    archive_bytes: bytes,
    template_decoder: NSCS02DownsampledDecoder,
) -> NSCS02ParsedArchive:
    """Decode NSCS02 archive bytes into a state-dict + latents tensor.

    Args:
        archive_bytes: bytes packed by ``pack_nscs02_archive``.
        template_decoder: a freshly-constructed NSCS02 decoder; its
            state-dict is the structural template that validates
            tensor names + shapes during parse.

    Returns:
        ``NSCS02ParsedArchive`` ready to ``decoder.load_state_dict(...)``.
    """
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

    decoder_sd = _inflate_fp16_stream_to_state_dict(weight_stream, template_decoder.state_dict())
    latents = _inflate_fp16_latent_stream(latent_stream)
    return NSCS02ParsedArchive(
        decoder_state_dict=decoder_sd,
        latents=latents,
        decoder_blob_len=D,
        latent_blob_len=L,
    )


def parser_section_manifest(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return canonical parser-section manifest for the archive bytes.

    Every NSCS02 archive declares its sections at fixed offsets per the
    wire format. This helper makes the offsets machine-readable for
    Catalog #124 representation-lane archive-grammar compliance and
    for the no-op detector smoke (Catalog #139).
    """
    if len(archive_bytes) < HEADER_LEN:
        raise ValueError(f"NSCS02 archive: too short for header ({len(archive_bytes)})")
    if archive_bytes[: len(NSCS02_ARCHIVE_MAGIC)] != NSCS02_ARCHIVE_MAGIC:
        raise ValueError("NSCS02 archive: magic mismatch")
    D = struct.unpack_from("<I", archive_bytes, 8)[0]
    L = struct.unpack_from("<I", archive_bytes, 12)[0]
    return {
        "magic": (0, len(NSCS02_ARCHIVE_MAGIC)),
        "decoder_blob_len_field": (8, 4),
        "latent_blob_len_field": (12, 4),
        "decoder_blob": (HEADER_LEN, D),
        "latent_blob": (HEADER_LEN + D, L),
    }


__all__ = [
    "BROTLI_QUALITY",
    "HEADER_LEN",
    "NSCS02ParsedArchive",
    "pack_nscs02_archive",
    "parse_nscs02_archive",
    "parser_section_manifest",
]
