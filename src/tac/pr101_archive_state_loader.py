# SPDX-License-Identifier: MIT
"""Strict PR101 archive to HNeRV decoder state loader.

The public PR101 ``hnerv_ft_microcodec`` archive stores a single ZIP member
``x`` whose bytes are laid out as:

``decoder_blob[0:162164] + latent_blob[162164:177551] + sidecar_blob``.

This module exposes the reusable, scorer-free loader for that decoder slice.
It fails closed on archive/member/layout errors so training dispatchers cannot
silently fall back to fresh initialization.
"""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli
import torch

from tac.pr101_split_brotli_codec import (
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    LATENT_DIM,
    N_PAIRS,
    Pr101SplitBrotliCodecError,
    apply_latent_sidecar,
    decode_decoder_compact,
    decode_latents_compact,
)

PR101_INNER_MEMBER_NAME = "x"
PR101_TOTAL_INNER_BYTES_NOMINAL = DECODER_BLOB_LEN + LATENT_BLOB_LEN + 607


class Pr101ArchiveStateLoaderError(ValueError):
    """Raised when a PR101 archive cannot be safely decoded to state_dict."""


@dataclasses.dataclass(frozen=True)
class Pr101ArchiveState:
    """Decoded PR101 archive state plus byte-custody metadata."""

    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    decoder_blob: bytes = dataclasses.field(repr=False)
    latent_blob: bytes = dataclasses.field(repr=False)
    sidecar_blob: bytes = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class Pr101ArchiveLatents:
    """Decoded PR101 runtime latents plus byte-custody metadata."""

    latents: torch.Tensor
    metadata: dict[str, Any]


def load_pr101_archive_state(
    archive_path: str | Path,
    *,
    expected_member_name: str = PR101_INNER_MEMBER_NAME,
    decoder_blob_len: int = DECODER_BLOB_LEN,
    latent_blob_len: int = LATENT_BLOB_LEN,
    require_stored_zip_member: bool = True,
) -> Pr101ArchiveState:
    """Decode a PR101-shaped archive into a torch ``state_dict``.

    Args:
        archive_path: Path to a PR101 ``archive.zip``.
        expected_member_name: Strict single member name. Defaults to ``"x"``.
        decoder_blob_len: Byte length of the PR101 decoder slice.
        latent_blob_len: Byte length of the PR101 latent slice.
        require_stored_zip_member: Require the inner ZIP member to be stored,
            matching the public PR101 archive and avoiding ZIP-level parser
            ambiguity.

    Returns:
        :class:`Pr101ArchiveState` with decoded state_dict and custody metadata.

    Raises:
        Pr101ArchiveStateLoaderError: on any archive, member, split, decoder, or
            state schema mismatch.
    """

    archive = Path(archive_path)
    if not archive.is_file():
        raise Pr101ArchiveStateLoaderError(f"PR101 archive path does not exist: {archive}")
    if decoder_blob_len <= 0:
        raise Pr101ArchiveStateLoaderError(f"decoder_blob_len must be positive: {decoder_blob_len}")
    if latent_blob_len < 0:
        raise Pr101ArchiveStateLoaderError(f"latent_blob_len must be non-negative: {latent_blob_len}")

    inner_blob, zip_member = _read_strict_pr101_inner_blob(
        archive,
        expected_member_name=expected_member_name,
        require_stored_zip_member=require_stored_zip_member,
    )
    required_min = decoder_blob_len + latent_blob_len
    if len(inner_blob) < required_min:
        raise Pr101ArchiveStateLoaderError(
            f"inner member {expected_member_name!r} length {len(inner_blob)} < "
            f"required decoder+latent minimum {required_min} "
            f"(decoder_blob_len={decoder_blob_len}, latent_blob_len={latent_blob_len})"
        )

    decoder_blob = inner_blob[:decoder_blob_len]
    latent_blob = inner_blob[decoder_blob_len:required_min]
    sidecar_blob = inner_blob[required_min:]
    try:
        state_dict = decode_decoder_compact(decoder_blob)
    except (Pr101SplitBrotliCodecError, brotli.error, ValueError, RuntimeError) as exc:
        raise Pr101ArchiveStateLoaderError(
            "decoder_blob cannot be decoded as PR101 split-Brotli state_dict: "
            f"{exc}"
        ) from exc

    _validate_state_dict_schema(state_dict)
    metadata = {
        "kind": "pr101_archive_decoder_blob",
        "loader": "tac.pr101_archive_state_loader.load_pr101_archive_state",
        "archive_path": archive.as_posix(),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha256_file(archive),
        "inner_member_name": zip_member.filename,
        "inner_member_compress_type": int(zip_member.compress_type),
        "inner_member_crc": int(zip_member.CRC),
        "inner_member_bytes": len(inner_blob),
        "inner_member_sha256": _sha256_bytes(inner_blob),
        "decoder_blob_offset": 0,
        "decoder_blob_bytes": len(decoder_blob),
        "decoder_blob_sha256": _sha256_bytes(decoder_blob),
        "latent_blob_offset": len(decoder_blob),
        "latent_blob_bytes": len(latent_blob),
        "latent_blob_sha256": _sha256_bytes(latent_blob),
        "sidecar_blob_offset": len(decoder_blob) + len(latent_blob),
        "sidecar_blob_bytes": len(sidecar_blob),
        "sidecar_blob_sha256": _sha256_bytes(sidecar_blob),
        "state_dict_tensor_count": len(state_dict),
        "state_dict_schema": "tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA",
        "state_dict_keys": [name for name, _shape in FIXED_STATE_SCHEMA],
    }
    return Pr101ArchiveState(
        state_dict=state_dict,
        metadata=metadata,
        decoder_blob=decoder_blob,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
    )


def load_pr101_archive_state_dict(
    archive_path: str | Path,
    **kwargs: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compatibility wrapper returning ``(state_dict, metadata)``."""

    loaded = load_pr101_archive_state(archive_path, **kwargs)
    return loaded.state_dict, loaded.metadata


def load_pr101_archive_latents(
    archive_path: str | Path,
    **kwargs: Any,
) -> Pr101ArchiveLatents:
    """Decode the exact PR101 runtime latent rows from ``archive.zip``.

    This helper applies both the fixed ``latent_blob`` decoder and the PR101
    latent sidecar, so the returned tensor matches what ``inflate.py`` feeds to
    ``HNeRVDecoder``. A1 score-gradient fine-tuning must use these rows; using
    random latents trains a decoder that the rebuilt archive cannot reproduce.
    """

    loaded = load_pr101_archive_state(archive_path, **kwargs)
    try:
        base_latents = decode_latents_compact(loaded.latent_blob)
        latents = apply_latent_sidecar(base_latents, loaded.sidecar_blob)
    except (Pr101SplitBrotliCodecError, brotli.error, ValueError, RuntimeError) as exc:
        raise Pr101ArchiveStateLoaderError(
            "latent_blob/sidecar_blob cannot be decoded as PR101 runtime latents: "
            f"{exc}"
        ) from exc
    if tuple(int(dim) for dim in latents.shape) != (N_PAIRS, LATENT_DIM):
        raise Pr101ArchiveStateLoaderError(
            f"decoded PR101 latents shape {tuple(latents.shape)} != "
            f"({N_PAIRS}, {LATENT_DIM})"
        )
    if latents.dtype != torch.float32:
        raise Pr101ArchiveStateLoaderError(f"decoded PR101 latents dtype {latents.dtype} != torch.float32")
    metadata = {
        "kind": "pr101_archive_runtime_latents",
        "loader": "tac.pr101_archive_state_loader.load_pr101_archive_latents",
        "source_state_loader": loaded.metadata["loader"],
        "archive_path": loaded.metadata["archive_path"],
        "archive_size_bytes": loaded.metadata["archive_size_bytes"],
        "archive_sha256": loaded.metadata["archive_sha256"],
        "inner_member_name": loaded.metadata["inner_member_name"],
        "inner_member_bytes": loaded.metadata["inner_member_bytes"],
        "inner_member_sha256": loaded.metadata["inner_member_sha256"],
        "latent_blob_offset": loaded.metadata["latent_blob_offset"],
        "latent_blob_bytes": loaded.metadata["latent_blob_bytes"],
        "latent_blob_sha256": loaded.metadata["latent_blob_sha256"],
        "sidecar_blob_offset": loaded.metadata["sidecar_blob_offset"],
        "sidecar_blob_bytes": loaded.metadata["sidecar_blob_bytes"],
        "sidecar_blob_sha256": loaded.metadata["sidecar_blob_sha256"],
        "sidecar_applied": bool(loaded.sidecar_blob),
        "latent_shape": [int(dim) for dim in latents.shape],
        "latent_dtype": str(latents.dtype),
        "frame_pair_order": "non_overlapping_pairs_0_1_2_3",
    }
    return Pr101ArchiveLatents(latents=latents, metadata=metadata)


def _read_strict_pr101_inner_blob(
    archive: Path,
    *,
    expected_member_name: str,
    require_stored_zip_member: bool,
) -> tuple[bytes, zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            if names != [expected_member_name]:
                raise Pr101ArchiveStateLoaderError(
                    f"archive {archive} has members {names!r}; expected "
                    f"[{expected_member_name!r}] (PR101 layout)"
                )
            info = infos[0]
            if info.is_dir():
                raise Pr101ArchiveStateLoaderError(
                    f"archive member {info.filename!r} is a directory; expected file"
                )
            if (
                "\\" in info.filename
                or "\x00" in info.filename
                or any(ord(ch) < 32 for ch in info.filename)
                or info.filename.startswith("/")
                or ".." in Path(info.filename).parts
            ):
                raise Pr101ArchiveStateLoaderError(
                    f"unsafe archive member name for PR101 layout: {info.filename!r}"
                )
            if require_stored_zip_member and info.compress_type != zipfile.ZIP_STORED:
                raise Pr101ArchiveStateLoaderError(
                    f"archive member {info.filename!r} compress_type={info.compress_type}; "
                    "expected ZIP_STORED for PR101 layout"
                )
            bad_member = zf.testzip()
            if bad_member is not None:
                raise Pr101ArchiveStateLoaderError(f"ZIP CRC validation failed for {bad_member!r}")
            return zf.read(info.filename), info
    except zipfile.BadZipFile as exc:
        raise Pr101ArchiveStateLoaderError(f"bad ZIP archive {archive}: {exc}") from exc


def _validate_state_dict_schema(state_dict: Mapping[str, torch.Tensor]) -> None:
    expected = dict(FIXED_STATE_SCHEMA)
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise Pr101ArchiveStateLoaderError(
            f"decoded PR101 state_dict key mismatch: missing={missing}, extra={extra}"
        )
    shape_errors = []
    for name, shape in FIXED_STATE_SCHEMA:
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            shape_errors.append(f"{name}: not a torch.Tensor ({type(tensor).__name__})")
            continue
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != shape:
            shape_errors.append(f"{name}: expected {shape}, got {actual_shape}")
        if tensor.dtype != torch.float32:
            shape_errors.append(f"{name}: expected torch.float32, got {tensor.dtype}")
    if shape_errors:
        raise Pr101ArchiveStateLoaderError(
            "decoded PR101 state_dict schema mismatch: " + "; ".join(shape_errors)
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
