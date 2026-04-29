"""CPU-only archive diet packer for renderer / SegMap submission archives.

Draft target path: src/tac/archive_diet_pack.py

The packer is intentionally lossless:
* ZIP entries are rewritten with deterministic metadata.
* Member payloads are Brotli-compressed at q=11 and ZIP_STORED.
* SegMap `segmap_weights.tar.xz` is logically replaced by SHv1
  `payload.bin`, using tac.arithmetic_qint_codec.
* Verification decompresses every Brotli member and checks byte-exact
  passthrough, plus bit-exact SegMap tensor decode for SHv1 replacement.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any


_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SEGMENT_WEIGHT_NAME = "segmap_weights.tar.xz"
_ARITH_PAYLOAD_NAME = "payload.bin"


def diet_pack(
    input_archive: Path,
    output_archive: Path,
    *,
    brotli_quality: int = 11,
    verify: bool = True,
) -> dict:
    """Repack an archive with SHv1 arithmetic qints and Brotli outer members.

    Args:
        input_archive: ZIP archive containing either `renderer.bin` or
            `segmap_weights.tar.xz` plus submission-side media/pose artifacts.
        output_archive: Destination archive path.
        brotli_quality: Brotli quality 0..11. Default 11 matches the current
            Quantizr-style compression setting.
        verify: If True, verify the output archive decodes bit-exactly before
            returning stats.

    Returns:
        Dict with byte totals, rate-only score savings [advisory only],
        verification status, and per-component input/output member sizes.
    """
    input_archive = Path(input_archive)
    output_archive = Path(output_archive)
    if not 0 <= int(brotli_quality) <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")

    input_bytes = input_archive.stat().st_size
    input_members = _read_zip_members(input_archive)
    if any(name.endswith(".br") for name in input_members):
        raise ValueError("archive already contains .br members; refusing double-Brotli repack")

    output_members: dict[str, bytes] = {}
    logical_replacements: dict[str, tuple[str, bytes]] = {}
    segmap_bit_exact = True

    if _SEGMENT_WEIGHT_NAME in input_members:
        with tempfile.TemporaryDirectory(prefix="archive-diet-") as td:
            payload, segmap_bit_exact, _ = _repack_segmap_weights_to_payload(
                input_members[_SEGMENT_WEIGHT_NAME],
                Path(td),
                verify=verify,
            )
        output_members[_ARITH_PAYLOAD_NAME] = payload
        logical_replacements[_ARITH_PAYLOAD_NAME] = (_SEGMENT_WEIGHT_NAME, input_members[_SEGMENT_WEIGHT_NAME])
    elif "renderer.bin" not in input_members:
        raise ValueError(
            "unsupported archive layout: expected renderer.bin or segmap_weights.tar.xz"
        )

    for name, data in input_members.items():
        if name == _SEGMENT_WEIGHT_NAME:
            continue
        output_members.setdefault(name, data)

    compressed_sizes = _write_deterministic_brotli_zip(
        output_archive,
        output_members,
        quality=brotli_quality,
    )

    bit_exact = bool(segmap_bit_exact)
    if verify:
        bit_exact = _verify_output_archive(
            input_members=input_members,
            output_archive=output_archive,
            logical_replacements=logical_replacements,
        )

    output_bytes = output_archive.stat().st_size
    savings_bytes = input_bytes - output_bytes
    components = _component_stats(input_members, output_members, compressed_sizes)

    from tac.submission_archive import ORIGINAL_VIDEO_BYTES

    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "savings_bytes": savings_bytes,
        "savings_score_pts": 25.0 * savings_bytes / ORIGINAL_VIDEO_BYTES,
        "bit_exact": bit_exact,
        "components": components,
    }


def _read_zip_members(path: Path) -> dict[str, bytes]:
    if not path.exists():
        raise FileNotFoundError(f"archive not found: {path}")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            if not infos:
                raise ValueError("empty archive")
            members: dict[str, bytes] = {}
            for info in infos:
                if info.is_dir():
                    continue
                _validate_member_name(info.filename)
                if info.filename in members:
                    raise ValueError(f"duplicate archive member: {info.filename}")
                members[info.filename] = zf.read(info)
    except zipfile.BadZipFile as e:
        raise ValueError(f"not a valid zip archive: {path}") from e
    if not members:
        raise ValueError("empty archive")
    return members


def _validate_member_name(name: str) -> None:
    p = Path(name)
    if name.startswith("/") or ".." in p.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if not name or name.endswith("/"):
        raise ValueError(f"invalid archive member path: {name!r}")


def _repack_segmap_weights_to_payload(
    weights_tar_xz: bytes,
    tmpdir: Path,
    *,
    verify: bool,
) -> tuple[bytes, bool, dict[str, Any]]:
    src = tmpdir / _SEGMENT_WEIGHT_NAME
    dst = tmpdir / _ARITH_PAYLOAD_NAME
    src.write_bytes(weights_tar_xz)

    from tac.arithmetic_qint_codec import (
        repack_payload_tar_xz_to_arithmetic,
        unpack_arithmetic_payload,
    )

    stats = repack_payload_tar_xz_to_arithmetic(str(src), str(dst))
    payload = dst.read_bytes()
    bit_exact = True

    if verify:
        from tac.block_fp_codec import unpack_payload_tar_xz

        original = unpack_payload_tar_xz(src)
        decoded = unpack_arithmetic_payload(str(dst))
        bit_exact = _tensor_dict_bit_exact(original, decoded)
        if not bit_exact:
            raise RuntimeError("SegMap SHv1 payload decode is not bit-exact")

    return payload, bit_exact, stats


def _tensor_dict_bit_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if getattr(a, "shape", None) != getattr(b, "shape", None):
            return False
        if getattr(a, "dtype", None) != getattr(b, "dtype", None):
            return False
        if hasattr(a, "equal"):
            if not a.equal(b):
                return False
        elif a != b:
            return False
    return True


def _write_deterministic_brotli_zip(
    output_archive: Path,
    members: dict[str, bytes],
    *,
    quality: int,
) -> dict[str, int]:
    import brotli

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    compressed_sizes: dict[str, int] = {}
    with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(members):
            br_name = f"{name}.br"
            compressed = brotli.compress(members[name], quality=quality, lgwin=24)
            info = zipfile.ZipInfo(filename=br_name, date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.extra = b""
            info.comment = b""
            zf.writestr(info, compressed)
            compressed_sizes[name] = len(compressed)
    return compressed_sizes


def _verify_output_archive(
    *,
    input_members: dict[str, bytes],
    output_archive: Path,
    logical_replacements: dict[str, tuple[str, bytes]],
) -> bool:
    import brotli

    with zipfile.ZipFile(output_archive, "r") as zf:
        output_infos = [info for info in zf.infolist() if not info.is_dir()]
        if not output_infos:
            raise RuntimeError("output archive is empty")

        decoded: dict[str, bytes] = {}
        for info in output_infos:
            if info.date_time != _FIXED_ZIP_TIMESTAMP:
                raise RuntimeError(f"non-deterministic timestamp on {info.filename}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError(f"member is not ZIP_STORED: {info.filename}")
            if info.extra:
                raise RuntimeError(f"member has ZIP extra metadata: {info.filename}")
            if not info.filename.endswith(".br"):
                raise RuntimeError(f"member is not Brotli-suffixed: {info.filename}")
            logical_name = info.filename[:-3]
            try:
                decoded[logical_name] = brotli.decompress(zf.read(info))
            except brotli.error as e:
                raise RuntimeError(f"Brotli decode failed for {info.filename}") from e

    replaced_input_names = {original_name for original_name, _ in logical_replacements.values()}
    expected_names = set(input_members) - replaced_input_names
    expected_names |= set(logical_replacements)
    if set(decoded) != expected_names:
        raise RuntimeError(
            f"output member set mismatch: got {sorted(decoded)}, expected {sorted(expected_names)}"
        )

    for name, data in decoded.items():
        if name in logical_replacements:
            original_name, original_data = logical_replacements[name]
            _verify_segmap_payload_bytes(original_name, original_data, name, data)
        elif decoded[name] != input_members[name]:
            raise RuntimeError(f"member {name} is not byte-exact after Brotli decode")

    return True


def _verify_segmap_payload_bytes(
    original_name: str,
    original_data: bytes,
    payload_name: str,
    payload_data: bytes,
) -> None:
    if original_name != _SEGMENT_WEIGHT_NAME or payload_name != _ARITH_PAYLOAD_NAME:
        raise RuntimeError(f"unsupported logical replacement: {original_name} -> {payload_name}")
    with tempfile.TemporaryDirectory(prefix="archive-diet-verify-") as td:
        td_path = Path(td)
        src = td_path / original_name
        dst = td_path / payload_name
        src.write_bytes(original_data)
        dst.write_bytes(payload_data)
        from tac.arithmetic_qint_codec import unpack_arithmetic_payload
        from tac.block_fp_codec import unpack_payload_tar_xz

        if not _tensor_dict_bit_exact(unpack_payload_tar_xz(src), unpack_arithmetic_payload(str(dst))):
            raise RuntimeError("logical replacement payload.bin is not bit-exact")


def _component_stats(
    input_members: dict[str, bytes],
    output_members: dict[str, bytes],
    compressed_sizes: dict[str, int],
) -> dict[str, dict[str, int]]:
    components: dict[str, dict[str, int]] = {}
    for name, data in input_members.items():
        if name == _SEGMENT_WEIGHT_NAME and _ARITH_PAYLOAD_NAME in compressed_sizes:
            components[name] = {"in": len(data), "out": compressed_sizes[_ARITH_PAYLOAD_NAME]}
        elif name in compressed_sizes:
            components[name] = {"in": len(data), "out": compressed_sizes[name]}
    for name, data in output_members.items():
        if name == _ARITH_PAYLOAD_NAME:
            continue
        if name not in input_members and name in compressed_sizes:
            components[name] = {"in": 0, "out": compressed_sizes[name]}
    return components


__all__ = ["diet_pack"]
