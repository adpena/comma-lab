"""Weight-derived procedural codebook generation.

The source member must already live inside ``archive.zip`` and be charged by the
rate term. The derived codebook adds no new payload bytes by construction.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from .hash_seed_codebook_generator import expand_seed_to_codebook

if TYPE_CHECKING:
    import numpy as np


def derive_codebook_from_archive_bytes(
    archive_path: Path,
    source_member: str,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Derive a deterministic int8 codebook from an existing archive member."""

    member_sha = freeze_source_member_sha256(archive_path, source_member)
    return expand_seed_to_codebook(bytes.fromhex(member_sha), target_shape)


def freeze_source_member_sha256(archive_path: Path, source_member: str) -> str:
    """Return the SHA-256 of an already-shipped archive member."""

    return hashlib.sha256(_read_unique_archive_member(Path(archive_path), source_member)).hexdigest()


def verify_no_new_bytes_added(before_archive: Path, after_archive: Path) -> bool:
    """Return true when the candidate archive adds no new members or bytes.

    This is intentionally conservative: the candidate archive must be no larger
    than the source archive, every non-directory member name in the candidate
    must have existed in the source archive, and every retained member must keep
    identical bytes. It does not make a score claim.
    """

    try:
        before = Path(before_archive)
        after = Path(after_archive)
        before_members = _unique_member_fingerprints(before)
        after_members = _unique_member_fingerprints(after)
        return (
            after.stat().st_size <= before.stat().st_size
            and after_members.keys() <= before_members.keys()
            and all(
                after_members[name] == before_members[name]
                for name in after_members
            )
        )
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _read_unique_archive_member(archive_path: Path, source_member: str) -> bytes:
    member = _normalize_member_name(source_member)
    with zipfile.ZipFile(archive_path) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if names.count(member) != 1:
            raise KeyError(f"expected exactly one source member {member!r} in archive")
        return zf.read(member)


def _unique_member_fingerprints(archive_path: Path) -> dict[str, tuple[int, int, str]]:
    with zipfile.ZipFile(archive_path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate archive members are not a valid no-new-bytes proof: {archive_path}")
        out: dict[str, tuple[int, int, str]] = {}
        for info in infos:
            payload = zf.read(info.filename)
            out[info.filename] = (
                int(info.file_size),
                int(info.CRC),
                hashlib.sha256(payload).hexdigest(),
            )
        return out


def _normalize_member_name(source_member: str) -> str:
    member = str(source_member).lstrip("/")
    if not member or member in {".", ".."} or member.startswith("../") or "/../" in member:
        raise ValueError(f"unsafe archive member path: {source_member!r}")
    return member
