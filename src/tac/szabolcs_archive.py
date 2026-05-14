# SPDX-License-Identifier: MIT
"""SZv1 archive packer / unpacker for the szabolcs no-masks paradigm.

Layout
------
On disk a SZv1 binary is::

    SZv1 magic                               (4 bytes)
    header_len uint32 LE                     (4 bytes)
    header JSON (utf-8)                      (header_len bytes)
    tar.xz payload                           (rest of file)

The tar.xz payload contains the following file members:

    latent_grid.bfp           — block-FP packed shared latent canvas
    affine_params.bfp         — block-FP packed per-frame affine table
    body/<layer_name>.weight.bfp — block-FP packed Conv2d weight
    body/<layer_name>.bias.f32  — float32 little-endian bias vector
    body/__index__.json       — list[{"name": ..., "shape": [...], "kind": ...}]
    config.json               — architecture hyperparameters (mirrors ckpt['__meta__'])

The header JSON carries lightweight metadata so consumers can sanity-check the
binary before unpacking the (relatively expensive) tar.xz stream:

    {
        "version": 1,
        "renderer": "szabolcs",
        "param_count": int,
        "block_size": int,
        "tarxz_nbytes": int,
        "checksum_crc32": int,
        "predicted_band": [low, high],   # advisory only
    }

Why tar.xz and not zip+brotli
-----------------------------
The reference (`/tmp/szabolcs_re/inflate.py`) reads a .pt pickle, but the PR
that scored 0.36 used a tar.xz outer wrapper for the renderer + frame video.
For dense-ternary block-FP data the LZMA2 / xz back end achieves close to
Shannon entropy on the qint stream — empirically ~1.0-1.5 bits/weight — which
beats zip+brotli by ~0.5 bits/weight on the same payload. The cost is one
extra second of CPU at inflate, well inside the 30-min contest budget.

Strict-scorer-rule
------------------
Nothing in this module loads SegNet or PoseNet. The packed payload contains
only the renderer state + a static LUT recipe (the LUT itself is
reconstructed in code at inflate time — see ``szabolcs_renderer.create_gaussian_softmax_lut``).
"""
from __future__ import annotations

import io
import json
import lzma
import struct
import tarfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch

from tac.block_fp_codec import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CLIP_THRESHOLD,
    pack_block_fp,
    unpack_block_fp,
)


SZV1_MAGIC: bytes = b"SZv1"
SZV1_VERSION: int = 1


# ── Helpers: tarfile member writers ────────────────────────────────────────


def _tar_add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0  # deterministic archive bytes
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def _conv_layer_names(model: torch.nn.Module) -> list[str]:
    """Return the canonical body-layer ordering used by the szabolcs renderer.

    We DON'T just iterate ``state_dict()`` because we want a deterministic,
    versioned ordering that survives state_dict reorderings in PyTorch
    upgrades. The ordering matches the reference inflate's reconstruction:

        layer_in.weight, layer_in.bias,
        blocks.0.conv1.weight, blocks.0.conv1.bias,
        blocks.0.conv2.weight, blocks.0.conv2.bias,
        ...
        layer_out.weight, layer_out.bias

    Returned list is the leaf layer prefixes (no ".weight"/".bias" suffix).
    """
    names: list[str] = ["layer_in"]
    num_blocks = len(model.blocks)  # type: ignore[attr-defined]
    for i in range(num_blocks):
        names.append(f"blocks.{i}.conv1")
        names.append(f"blocks.{i}.conv2")
    names.append("layer_out")
    return names


# ── Packing ────────────────────────────────────────────────────────────────


@dataclass
class SzabolcsArchiveStats:
    """Bundle returned by :func:`pack_szabolcs_archive` for provenance."""

    raw_param_count: int
    raw_param_bytes: int  # if we stored fp32 naively
    packed_bytes: int  # final SZv1 binary size
    bits_per_weight: float
    tarxz_compressed_bytes: int


def pack_szabolcs_archive(
    model: torch.nn.Module,
    output_path: Optional[Union[Path, str]] = None,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    predicted_band: tuple[float, float] = (0.30, 0.50),
) -> tuple[bytes, SzabolcsArchiveStats]:
    """Serialize a SzabolcsRenderer to the SZv1 binary format.

    Args:
        model: A ``SzabolcsRenderer`` instance (eval mode preferred but not required).
        output_path: If provided, also writes the bytes to disk at this path.
        block_size: Block-FP partition size along axis 0 (default 16).
        clip_threshold: Ternary rounding threshold (default 0.5).
        predicted_band: Advisory contest score band, recorded in the header
            for downstream provenance — does NOT affect bytes-on-the-wire.

    Returns:
        ``(blob, stats)`` — the SZv1 bytes plus a bits/weight summary.
    """
    # ── Collect parameters ────────────────────────────────────────────────
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ── Build tar.xz payload ──────────────────────────────────────────────
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        # latent_grid (1, C, H, W) — block-FP packed along channel axis.
        latent = state["shared_latent_base"]
        latent_packed = pack_block_fp(latent.squeeze(0), block_size, clip_threshold)
        _tar_add_bytes(tar, "latent_grid.bfp", latent_packed)

        # affine_params (max_frame_index, 6) — per-frame embedding.
        affine = state["frame_affine_embedding.weight"]
        affine_packed = pack_block_fp(affine, block_size, clip_threshold)
        _tar_add_bytes(tar, "affine_params.bfp", affine_packed)

        # Body conv layers — block-FP weights + raw float32 biases. Biases
        # are tiny (one float per output channel) so block-FP'ing them
        # doesn't help; raw float32 wins the precision/size tradeoff.
        index_entries = []
        for layer_name in _conv_layer_names(model):
            w_key = f"{layer_name}.weight"
            b_key = f"{layer_name}.bias"
            w = state[w_key]
            b = state[b_key]

            w_packed = pack_block_fp(w, block_size, clip_threshold)
            _tar_add_bytes(tar, f"body/{layer_name}.weight.bfp", w_packed)

            b_bytes = b.contiguous().to(torch.float32).cpu().numpy().tobytes()
            _tar_add_bytes(tar, f"body/{layer_name}.bias.f32", b_bytes)

            index_entries.append({
                "name": layer_name,
                "weight_shape": list(w.shape),
                "bias_shape": list(b.shape),
                "kind": "conv2d",
            })

        body_index = json.dumps({"layers": index_entries}, sort_keys=True).encode("utf-8")
        _tar_add_bytes(tar, "body/__index__.json", body_index)

        # config.json — everything needed to re-instantiate the architecture.
        cfg = {
            "renderer": "szabolcs",
            "hidden": int(model.hidden),
            "block_hidden": int(model.block_hidden),
            "num_blocks": int(model.num_blocks),
            "num_classes": int(model.num_classes),
            "max_frame_index": int(model.max_frame_index),
            "shared_latent_channels": int(model.shared_latent_channels),
            "shared_latent_height": int(model.shared_latent_height),
            "shared_latent_width": int(model.shared_latent_width),
            "latent_canvas_scale": float(model.latent_canvas_scale),
            "affine_max_zoom_delta": float(model.max_zoom_delta),
            "affine_max_aspect_delta": float(model.max_aspect_delta),
            "affine_max_shear": float(model.max_shear),
            "affine_max_translation": float(model.max_translation),
            "latent_input_scale": float(model.latent_input_scale),
            "block_size": int(block_size),
            "clip_threshold": float(clip_threshold),
        }
        _tar_add_bytes(tar, "config.json", json.dumps(cfg, sort_keys=True).encode("utf-8"))

    tar_bytes = tar_buf.getvalue()

    # XZ outer compression — preset 9e for max ratio (the tar payload is small,
    # so encode time is well under a second even at preset 9).
    xz_bytes = lzma.compress(tar_bytes, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)

    # ── Header ────────────────────────────────────────────────────────────
    raw_param_count = sum(int(v.numel()) for v in state.values())
    raw_param_bytes = raw_param_count * 4
    header = {
        "version": SZV1_VERSION,
        "renderer": "szabolcs",
        "param_count": raw_param_count,
        "block_size": block_size,
        "clip_threshold": clip_threshold,
        "tarxz_nbytes": len(xz_bytes),
        "checksum_crc32": zlib.crc32(xz_bytes) & 0xFFFFFFFF,
        "predicted_band": [float(predicted_band[0]), float(predicted_band[1])],
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    blob = (
        SZV1_MAGIC
        + struct.pack("<I", len(header_bytes))
        + header_bytes
        + xz_bytes
    )

    if output_path is not None:
        Path(output_path).write_bytes(blob)

    bpw = (8.0 * len(blob)) / max(raw_param_count, 1)
    stats = SzabolcsArchiveStats(
        raw_param_count=raw_param_count,
        raw_param_bytes=raw_param_bytes,
        packed_bytes=len(blob),
        bits_per_weight=bpw,
        tarxz_compressed_bytes=len(xz_bytes),
    )
    return blob, stats


# ── Unpacking ──────────────────────────────────────────────────────────────


@dataclass
class SzabolcsArchiveContents:
    """Decoded contents of a SZv1 binary."""

    config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    header: dict[str, Any]


def unpack_szabolcs_archive(data: Union[bytes, Path, str]) -> SzabolcsArchiveContents:
    """Inverse of :func:`pack_szabolcs_archive`.

    Returns a config dict + a state_dict ready to load into a freshly built
    ``SzabolcsRenderer``.
    """
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    if not isinstance(data, (bytes, bytearray)):  # pragma: no cover
        raise TypeError(f"unpack_szabolcs_archive: bad input type {type(data)}")
    blob = bytes(data)

    if blob[:4] != SZV1_MAGIC:
        raise ValueError(
            f"unpack_szabolcs_archive: not a SZv1 binary (magic={blob[:4]!r})"
        )
    offset = 4
    (header_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    header = json.loads(blob[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    if header.get("version") != SZV1_VERSION:
        raise ValueError(
            f"unpack_szabolcs_archive: unsupported version {header.get('version')}"
        )
    xz_bytes = blob[offset:]
    if zlib.crc32(xz_bytes) & 0xFFFFFFFF != header.get("checksum_crc32"):
        raise ValueError("unpack_szabolcs_archive: tar.xz checksum mismatch")

    tar_bytes = lzma.decompress(xz_bytes, format=lzma.FORMAT_XZ)
    tar_buf = io.BytesIO(tar_bytes)
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=tar_buf, mode="r") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:  # pragma: no cover (no dirs in our pack)
                continue
            members[member.name] = f.read()

    cfg = json.loads(members["config.json"].decode("utf-8"))
    body_index = json.loads(members["body/__index__.json"].decode("utf-8"))

    state: dict[str, torch.Tensor] = {}

    # latent_grid: pack stripped the leading singleton; restore it.
    latent_packed = members["latent_grid.bfp"]
    latent = unpack_block_fp(latent_packed)
    state["shared_latent_base"] = latent.unsqueeze(0)

    affine_packed = members["affine_params.bfp"]
    state["frame_affine_embedding.weight"] = unpack_block_fp(affine_packed)

    for entry in body_index["layers"]:
        layer = entry["name"]
        w_packed = members[f"body/{layer}.weight.bfp"]
        b_bytes = members[f"body/{layer}.bias.f32"]
        w = unpack_block_fp(w_packed)
        b = torch.frombuffer(bytearray(b_bytes), dtype=torch.float32).clone()
        state[f"{layer}.weight"] = w
        state[f"{layer}.bias"] = b

    return SzabolcsArchiveContents(config=cfg, state_dict=state, header=header)


__all__ = [
    "SZV1_MAGIC",
    "SZV1_VERSION",
    "SzabolcsArchiveStats",
    "SzabolcsArchiveContents",
    "pack_szabolcs_archive",
    "unpack_szabolcs_archive",
]
