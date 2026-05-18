"""Hash-seed procedural codebook generation.

The seed bytes must be shipped inside ``archive.zip`` before any generated
codebook can affect an inflate path. This module only provides deterministic
generation and structural mutation smoke checks; it does not promote a score.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

DEFAULT_SEED_BYTES = 8
_SEED_DOMAIN = b"tac.procedural_codebook_generator.seed.v1\0"
_SMOKE_SHAPE = (256,)
_SUPPORTED_DISTRIBUTIONS = frozenset({"uniform_int8"})


def emit_seed(
    target_codebook_shape: tuple[int, ...],
    target_distribution: str = "uniform_int8",
) -> bytes:
    """Emit a deterministic in-archive seed for the requested codebook shape.

    The default seed is intentionally derived from the declared target, rather
    than ambient randomness, so tests and archive builders are byte-stable. A
    substrate-specific search may still store a different seed inside the
    archive, then use :func:`expand_seed_to_codebook` to decode it.
    """

    shape = _validate_shape(target_codebook_shape)
    distribution = _validate_distribution(target_distribution)
    descriptor = json.dumps(
        {
            "distribution": distribution,
            "shape": shape,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_SEED_DOMAIN + descriptor).digest()[:DEFAULT_SEED_BYTES]


def expand_seed_to_codebook(
    seed: bytes,
    target_shape: tuple[int, ...],
    target_distribution: str = "uniform_int8",
) -> np.ndarray:
    """Expand shipped seed bytes to a deterministic numpy codebook."""

    shape = _validate_shape(target_shape)
    distribution = _validate_distribution(target_distribution)
    seed_bytes = _validate_seed(seed)
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed_bytes, "big")))

    if distribution == "uniform_int8":
        raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
        return raw.view(np.int8)

    raise AssertionError(f"unsupported distribution passed validation: {distribution}")


def verify_generator_seed_mutation_smoke(
    seed_path: Path,
    archive_path: Path,
    mutation_offsets: Sequence[int],
) -> bool:
    """Return true when mutating shipped seed bytes changes generated output.

    ``archive_path`` may be a ZIP archive or an extracted archive directory.
    ``seed_path`` is interpreted as the member path inside that archive unless
    it is an absolute path inside an extracted archive directory.

    This is a structural Catalog #272-style smoke: it proves the seed bytes are
    present and consumed by this generator. It is not an inflate-output smoke.
    """

    try:
        seed = _read_seed_from_archive(seed_path=Path(seed_path), archive_path=Path(archive_path))
        offsets = tuple(_coerce_offsets(mutation_offsets))
        if not offsets:
            return False

        baseline = expand_seed_to_codebook(seed, _SMOKE_SHAPE)
        for offset in offsets:
            if offset < 0 or offset >= len(seed):
                return False
            mutated = bytearray(seed)
            mutated[offset] ^= 0xFF
            candidate = expand_seed_to_codebook(bytes(mutated), _SMOKE_SHAPE)
            if np.array_equal(candidate, baseline):
                return False
        return True
    except (OSError, ValueError, zipfile.BadZipFile, KeyError):
        return False


def _validate_distribution(target_distribution: str) -> str:
    distribution = str(target_distribution)
    if distribution not in _SUPPORTED_DISTRIBUTIONS:
        supported = ", ".join(sorted(_SUPPORTED_DISTRIBUTIONS))
        raise ValueError(f"unsupported target_distribution {distribution!r}; supported: {supported}")
    return distribution


def _validate_shape(target_shape: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(target_shape, tuple) or not target_shape:
        raise ValueError("target_shape must be a non-empty tuple of positive integers")
    out: list[int] = []
    for dim in target_shape:
        try:
            dim_int = operator.index(dim)
        except TypeError as exc:
            raise ValueError("target_shape dimensions must be integers") from exc
        if dim_int <= 0:
            raise ValueError("target_shape dimensions must be positive")
        out.append(dim_int)
    if math.prod(out) <= 0:
        raise ValueError("target_shape must contain at least one element")
    return tuple(out)


def _validate_seed(seed: bytes) -> bytes:
    seed_bytes = bytes(seed)
    if not seed_bytes:
        raise ValueError("seed must contain at least one byte")
    return seed_bytes


def _coerce_offsets(mutation_offsets: Sequence[int]) -> Iterable[int]:
    for offset in mutation_offsets:
        if not isinstance(offset, int):
            raise ValueError("mutation_offsets must contain integers")
        yield offset


def _read_seed_from_archive(*, seed_path: Path, archive_path: Path) -> bytes:
    if archive_path.is_dir():
        archive_root = archive_path.resolve()
        candidate = seed_path if seed_path.is_absolute() else archive_root / seed_path
        candidate = candidate.resolve(strict=True)
        try:
            candidate.relative_to(archive_root)
        except ValueError as exc:
            raise ValueError("seed_path must be inside extracted archive_path") from exc
        return _validate_seed(candidate.read_bytes())

    member_name = _normalize_member_name(seed_path)
    with zipfile.ZipFile(archive_path) as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if names.count(member_name) != 1:
            raise KeyError(f"expected exactly one seed member {member_name!r} in archive")
        return _validate_seed(zf.read(member_name))


def _normalize_member_name(path: Path) -> str:
    member = path.as_posix().lstrip("/")
    if not member or member in {".", ".."} or member.startswith("../") or "/../" in member:
        raise ValueError(f"unsafe archive member path: {path}")
    return member
