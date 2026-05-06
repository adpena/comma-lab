"""PR95-family HNeRV single-member archive wire helpers.

Public PR95/PR98-style HNeRV archives use one stored ZIP member named
``0.bin``. The member payload is three length-prefixed brotli blobs:
metadata, decoder weights, and uint8 latent rows. This module keeps that
wire grammar in ``tac`` so profilers, residual-atom planners, and replay tools
do not each carry their own parser.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import struct
import zipfile
from pathlib import Path
from typing import Sequence

import brotli
import numpy as np


FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


@dataclasses.dataclass(frozen=True)
class LatentPayload:
    n_pairs: int
    latent_dim: int
    mins_f16: bytes
    scales_f16: bytes
    quantized: tuple[tuple[int, ...], ...]

    def to_bytes(self) -> bytes:
        if len(self.mins_f16) != self.latent_dim * 2:
            raise ValueError("mins_f16 length does not match latent_dim")
        if len(self.scales_f16) != self.latent_dim * 2:
            raise ValueError("scales_f16 length does not match latent_dim")
        rows = _rows_to_uint8_matrix(
            self.quantized,
            n_pairs=self.n_pairs,
            latent_dim=self.latent_dim,
            label="latent quantized",
        )
        deltas = np.empty(rows.shape, dtype=np.int16)
        deltas[0] = rows[0].astype(np.int16)
        if self.n_pairs > 1:
            deltas[1:] = rows[1:].astype(np.int16) - rows[:-1].astype(np.int16)
        zz = np.where(deltas >= 0, deltas * 2, -2 * deltas - 1).astype(np.uint16)
        return (
            struct.pack("<II", self.n_pairs, self.latent_dim)
            + self.mins_f16
            + self.scales_f16
            + (zz & 0xFF).astype(np.uint8).tobytes()
            + (zz >> 8).astype(np.uint8).tobytes()
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(buf: io.BytesIO, n: int, label: str) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise ValueError(f"truncated {label}: wanted {n}, got {len(data)}")
    return data


def read_single_member_zip(path: Path) -> tuple[str, bytes, dict[str, int | list[int]]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected exactly one archive member, got {len(infos)}")
        info = infos[0]
        if info.filename != "0.bin":
            raise ValueError(f"PR95-family archive must contain exactly 0.bin, got {info.filename!r}")
        data = zf.read(info.filename)
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC validation failed for member {bad!r}")
        return (
            info.filename,
            data,
            {
                "compress_type": int(info.compress_type),
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "crc": int(info.CRC),
                "date_time": list(info.date_time),
            },
        )


def write_stored_zip(path: Path, member_name: str, payload: bytes) -> None:
    if member_name != "0.bin":
        raise ValueError(f"PR95-family archive member must be 0.bin, got {member_name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name, date_time=FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def parse_top_blob(blob: bytes) -> dict[str, bytes]:
    buf = io.BytesIO(blob)
    out: dict[str, bytes] = {}
    for label in ("meta", "decoder", "latents"):
        (size,) = struct.unpack("<I", _read_exact(buf, 4, f"{label}_brotli_len"))
        compressed = _read_exact(buf, size, f"{label}_brotli")
        out[f"{label}_brotli"] = compressed
        out[f"{label}_raw"] = brotli.decompress(compressed)
    rest = buf.read()
    if rest:
        raise ValueError(f"trailing bytes after PR95 blob: {len(rest)}")
    return out


def encode_top_blob(meta_brotli: bytes, decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    out = io.BytesIO()
    for payload in (meta_brotli, decoder_brotli, latents_brotli):
        out.write(struct.pack("<I", len(payload)))
        out.write(payload)
    return out.getvalue()


def parse_latents_raw(latents_raw: bytes) -> LatentPayload:
    buf = io.BytesIO(latents_raw)
    n_pairs, latent_dim = struct.unpack("<II", _read_exact(buf, 8, "latent header"))
    if n_pairs <= 0 or latent_dim <= 0:
        raise ValueError(f"latent header dimensions must be positive, got n_pairs={n_pairs}, latent_dim={latent_dim}")
    mins_f16 = _read_exact(buf, latent_dim * 2, "latent mins_f16")
    scales_f16 = _read_exact(buf, latent_dim * 2, "latent scales_f16")
    total = n_pairs * latent_dim
    lo = _read_exact(buf, total, "latent lo delta stream")
    hi = _read_exact(buf, total, "latent hi delta stream")
    rest = buf.read()
    if rest:
        raise ValueError(f"latent raw has trailing bytes: {len(rest)}")
    lo_u16 = np.frombuffer(lo, dtype=np.uint8).astype(np.uint16).reshape(n_pairs, latent_dim)
    hi_u16 = np.frombuffer(hi, dtype=np.uint8).astype(np.uint16).reshape(n_pairs, latent_dim)
    zz = lo_u16 | (hi_u16 << 8)
    deltas = np.where((zz & 1) == 0, zz // 2, -((zz // 2).astype(np.int32)) - 1).astype(np.int16)
    values = np.cumsum(deltas, axis=0, dtype=np.int16)
    bad = np.argwhere((values < 0) | (values > 255))
    if bad.size:
        pair_index, dim_index = (int(part) for part in bad[0])
        raise ValueError(
            f"latent quantized value out of uint8 range at pair {pair_index}, "
            f"dim {dim_index}: {int(values[pair_index, dim_index])}"
        )
    return LatentPayload(
        n_pairs=n_pairs,
        latent_dim=latent_dim,
        mins_f16=mins_f16,
        scales_f16=scales_f16,
        quantized=tuple(tuple(int(value) for value in row) for row in values.tolist()),
    )


def latent_rows(payload: LatentPayload) -> list[list[int]]:
    return [list(row) for row in payload.quantized]


def latent_payload_from_rows(source: LatentPayload, rows: Sequence[Sequence[int]]) -> LatentPayload:
    checked = _rows_to_uint8_matrix(
        rows,
        n_pairs=source.n_pairs,
        latent_dim=source.latent_dim,
        label="latent value",
    )
    return dataclasses.replace(
        source,
        quantized=tuple(tuple(int(value) for value in row) for row in checked.tolist()),
    )


def _rows_to_uint8_matrix(
    rows: Sequence[Sequence[int]],
    *,
    n_pairs: int,
    latent_dim: int,
    label: str,
) -> np.ndarray:
    if n_pairs <= 0 or latent_dim <= 0:
        raise ValueError(f"latent shape must be positive, got {(n_pairs, latent_dim)}")
    try:
        matrix = np.asarray(rows, dtype=np.int16)
    except Exception as exc:
        raise ValueError(f"{label} rows must be a rectangular integer matrix") from exc
    if matrix.shape != (n_pairs, latent_dim):
        raise ValueError(f"expected latent shape {(n_pairs, latent_dim)}, got {matrix.shape}")
    bad = np.argwhere((matrix < 0) | (matrix > 255))
    if bad.size:
        pair_index, dim_index = (int(part) for part in bad[0])
        raise ValueError(
            f"{label} out of uint8 range at pair {pair_index}, "
            f"dim {dim_index}: {int(matrix[pair_index, dim_index])}"
        )
    return matrix.astype(np.uint8)
