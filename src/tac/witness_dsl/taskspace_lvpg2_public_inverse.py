# SPDX-License-Identifier: MIT
"""Generic public LVPG2 inverse and LVLS1 decoder adapter.

This file is intentionally self-contained.  The public-runtime compiler copies
its exact bytes to ``inflate.py`` next to a byte-owned generic LVLS1 renderer.
It contains only the reversible LVPG2 wire grammar and process adapter; every
instance-derived weight, latent, selector, threshold, or exception remains in
the counted archive member.
"""

from __future__ import annotations

import bz2
import hashlib
import itertools
import json
import lzma
import os
import runpy
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import brotli
import numpy as np

LVPG2_MAGIC = b"LVPG2\x00"
LVLS1_MAGIC = b"LVLS1\x00"
_U32 = struct.Struct("<I")
_INNER_CODECS = (
    "raw",
    "brotli9",
    "brotli10",
    "brotli11",
    "zlib9",
    "bz2",
    "lzma9",
)
_PERMUTATIONS = tuple(itertools.permutations(range(3)))
_TRACE_SCHEMA = "tac.taskspace_lvpg2_public_inverse_trace.v1"


class PublicInverseError(ValueError):
    """The public inverse wire, environment, or output contract failed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PublicInverseError("value is not finite canonical ASCII JSON") from exc


def _take(payload: bytes, offset: int, size: int, *, label: str) -> tuple[bytes, int]:
    end = offset + size
    if size < 0 or end > len(payload):
        raise PublicInverseError(f"truncated {label}: need {size} byte(s) at {offset}, member has {len(payload)}")
    return payload[offset:end], end


def _read_four_blocks(member: bytes) -> tuple[dict[str, Any], tuple[bytes, bytes, bytes, bytes]]:
    if type(member) is not bytes or not member.startswith(LVPG2_MAGIC):
        raise PublicInverseError("input is not immutable LVPG2 bytes")
    offset = len(LVPG2_MAGIC)
    blocks: list[bytes] = []
    for label in ("manifest", "base", "population", "pose"):
        if offset + _U32.size > len(member):
            raise PublicInverseError(f"truncated {label} length")
        (size,) = _U32.unpack_from(member, offset)
        offset += _U32.size
        block, offset = _take(member, offset, size, label=label)
        blocks.append(block)
    if offset != len(member):
        raise PublicInverseError(f"LVPG2 member has {len(member) - offset} unconsumed trailing byte(s)")
    try:
        manifest = json.loads(blocks[0].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicInverseError("LVPG2 manifest is not ASCII JSON") from exc
    if type(manifest) is not dict or _canonical_json(manifest) != blocks[0]:
        raise PublicInverseError("LVPG2 manifest is not canonical sorted compact JSON")
    return manifest, (blocks[0], blocks[1], blocks[2], blocks[3])


def _decompress(payload: bytes, codec_wire: int) -> bytes:
    if type(codec_wire) is not int or codec_wire not in range(len(_INNER_CODECS)):
        raise PublicInverseError("LVPG2 inner-codec wire is invalid")
    codec = _INNER_CODECS[codec_wire]
    try:
        if codec == "raw":
            return payload
        if codec.startswith("brotli"):
            return brotli.decompress(payload)
        if codec == "zlib9":
            return zlib.decompress(payload)
        if codec == "bz2":
            return bz2.decompress(payload)
        if codec == "lzma9":
            return lzma.decompress(payload)
    except (brotli.error, bz2.BZ2Error, lzma.LZMAError, zlib.error) as exc:
        raise PublicInverseError(f"LVPG2 {codec} payload failed exact decode") from exc
    raise PublicInverseError(f"unsupported LVPG2 codec {codec!r}")


def _unzigzag_i8(value: np.ndarray) -> np.ndarray:
    unsigned = np.asarray(value, dtype=np.uint8).astype(np.int16)
    signed = np.where((unsigned & 1) == 0, unsigned // 2, -(unsigned // 2) - 1)
    return signed.astype(np.int8)


def _validate_unique_indices(value: object, *, size: int, label: str) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise PublicInverseError(f"{label} must be an integer list")
    indices = tuple(value)
    if tuple(sorted(set(indices))) != indices or any(item < 0 or item >= size for item in indices):
        raise PublicInverseError(f"{label} must be unique, increasing, and in range")
    return indices


def _decode_base(
    raw: bytes,
    manifest: dict[str, Any],
    *,
    transpose_indices: tuple[int, ...],
    zigzag_indices: tuple[int, ...],
) -> bytes:
    order = manifest.get("base_param_order")
    shapes = manifest.get("base_shapes")
    if (
        type(order) is not list
        or not order
        or any(type(name) is not str or not name for name in order)
        or len(set(order)) != len(order)
        or type(shapes) is not dict
        or set(shapes) != set(order)
    ):
        raise PublicInverseError("LVPG2 base order/shapes are malformed")
    transposed = frozenset(transpose_indices)
    zigzagged = frozenset(zigzag_indices)
    offset = 0
    decoded: list[bytes] = []
    for index, name in enumerate(order):
        shape_wire = shapes[name]
        if type(shape_wire) is not list or not shape_wire or any(type(dim) is not int or dim < 1 for dim in shape_wire):
            raise PublicInverseError(f"LVPG2 base shape for {name!r} is invalid")
        shape = tuple(shape_wire)
        size = int(np.prod(shape, dtype=np.int64))
        chunk, offset = _take(raw, offset, size, label=f"base tensor {name}")
        if index in transposed:
            if len(shape) != 2 or min(shape) <= 1:
                raise PublicInverseError(f"LVPG2 transpose targets invalid tensor {name!r}")
            stored_shape = (shape[1], shape[0])
        else:
            stored_shape = shape
        unsigned = np.frombuffer(chunk, dtype=np.uint8).reshape(stored_shape)
        value = _unzigzag_i8(unsigned) if index in zigzagged else unsigned.view(np.int8).copy()
        if index in transposed:
            value = value.T.copy()
        decoded.append(np.ascontiguousarray(value, dtype=np.int8).tobytes())
    if offset != len(raw):
        raise PublicInverseError(f"LVPG2 base stream has {len(raw) - offset} unconsumed trailing byte(s)")
    return b"".join(decoded)


def _unpredict_axis(
    encoded: np.ndarray,
    *,
    axis: int,
    family_wire: int,
    reset_interval: int,
) -> np.ndarray:
    out = np.asarray(encoded, dtype=np.uint8).copy()
    for index in range(1, encoded.shape[axis]):
        current = [slice(None)] * encoded.ndim
        previous = [slice(None)] * encoded.ndim
        current[axis] = index
        previous[axis] = index - 1
        if axis == 0 and reset_interval and index % reset_interval == 0:
            out[tuple(current)] = encoded[tuple(current)]
        elif family_wire == 0:
            out[tuple(current)] = (
                encoded[tuple(current)].astype(np.uint16) + out[tuple(previous)].astype(np.uint16)
            ).astype(np.uint8)
        elif family_wire == 1:
            out[tuple(current)] = encoded[tuple(current)] ^ out[tuple(previous)]
        else:
            raise PublicInverseError("LVPG2 predictor-family wire is invalid")
    return out


def _decode_population(raw: bytes, manifest: dict[str, Any], transform: object) -> bytes:
    n_pairs = manifest.get("n_pairs")
    code_shape = manifest.get("code_shape")
    if (
        type(n_pairs) is not int
        or n_pairs < 1
        or type(code_shape) is not list
        or len(code_shape) != 2
        or any(type(item) is not int or item < 1 for item in code_shape)
        or code_shape[0] != 2 * n_pairs
    ):
        raise PublicInverseError("LVPG2 population shape is inconsistent with n_pairs")
    if type(transform) is not list or len(transform) != 5 or any(type(item) is not int for item in transform):
        raise PublicInverseError("LVPG2 population transform must be five integers")
    family_wire, permutation_wire, axis_mask, reset_interval, zigzag_wire = transform
    if (
        family_wire not in (0, 1)
        or permutation_wire not in range(len(_PERMUTATIONS))
        or axis_mask not in range(8)
        or reset_interval < 0
        or (not axis_mask & 1 and reset_interval != 0)
        or zigzag_wire not in (0, 1)
    ):
        raise PublicInverseError("LVPG2 population transform wire is invalid")
    shape = (n_pairs, 2, code_shape[1])
    expected = int(np.prod(shape, dtype=np.int64))
    if len(raw) != expected:
        raise PublicInverseError(f"LVPG2 population stream has {len(raw)} byte(s), expected {expected}")
    permutation = _PERMUTATIONS[permutation_wire]
    permuted_shape = tuple(shape[axis] for axis in permutation)
    permuted = np.frombuffer(raw, dtype=np.uint8).reshape(permuted_shape)
    inverse_permutation = tuple(int(item) for item in np.argsort(permutation))
    value = np.ascontiguousarray(np.transpose(permuted, inverse_permutation))
    if zigzag_wire:
        value = _unzigzag_i8(value).view(np.uint8)
    for axis in reversed(range(3)):
        if (axis_mask >> axis) & 1:
            value = _unpredict_axis(
                value,
                axis=axis,
                family_wire=family_wire,
                reset_interval=reset_interval if axis == 0 else 0,
            )
    return np.ascontiguousarray(value.reshape(code_shape).view(np.int8)).tobytes()


def _pack_lvls1(manifest: bytes, base: bytes, code: bytes, pose: bytes) -> bytes:
    return b"".join(
        (
            LVLS1_MAGIC,
            _U32.pack(len(manifest)),
            manifest,
            _U32.pack(len(base)),
            base,
            _U32.pack(len(code)),
            code,
            _U32.pack(len(pose)),
            pose,
        )
    )


def lvpg2_to_lvls1(member: bytes) -> bytes:
    """Invert any well-formed LVPG2 member to an ordinary logical LVLS1 member."""

    manifest, blocks = _read_four_blocks(member)
    descriptor = manifest.get("pg2")
    if type(descriptor) is not dict or set(descriptor) != {"b", "l", "o", "t", "z"}:
        raise PublicInverseError("LVPG2 pg2 descriptor fields drifted")
    order = manifest.get("base_param_order")
    if type(order) is not list:
        raise PublicInverseError("LVPG2 base order is not a list")
    transpose_indices = _validate_unique_indices(descriptor["b"], size=len(order), label="LVPG2 transpose indices")
    zigzag_indices = _validate_unique_indices(descriptor["z"], size=len(order), label="LVPG2 zigzag indices")
    layout = descriptor["l"]
    if type(layout) is not list or len(layout) != 3 or any(type(item) is not int for item in layout):
        raise PublicInverseError("LVPG2 layout/codec wire must be three integers")
    layout_wire, base_codec_wire, population_codec_wire = layout
    if layout_wire == 0:
        base_raw = _decompress(blocks[1], base_codec_wire)
        population_raw = _decompress(blocks[2], population_codec_wire)
    elif layout_wire == 1:
        if population_codec_wire != 0 or blocks[2]:
            raise PublicInverseError("joint LVPG2 requires raw empty population slot")
        joint = _decompress(blocks[1], base_codec_wire)
        shapes = manifest.get("base_shapes")
        if type(shapes) is not dict or set(shapes) != set(order):
            raise PublicInverseError("LVPG2 base shapes are malformed")
        base_size = 0
        for name in order:
            shape = shapes[name]
            if type(shape) is not list or any(type(dim) is not int or dim < 1 for dim in shape):
                raise PublicInverseError(f"LVPG2 base shape for {name!r} is invalid")
            base_size += int(np.prod(shape, dtype=np.int64))
        code_shape = manifest.get("code_shape")
        if type(code_shape) is not list or len(code_shape) != 2:
            raise PublicInverseError("LVPG2 code shape is malformed")
        population_size = int(np.prod(code_shape, dtype=np.int64))
        if len(joint) != base_size + population_size:
            raise PublicInverseError(
                f"joint LVPG2 stream has {len(joint)} byte(s), expected {base_size + population_size}"
            )
        base_raw, population_raw = joint[:base_size], joint[base_size:]
    else:
        raise PublicInverseError("LVPG2 layout wire is invalid")
    logical_base = _decode_base(
        base_raw,
        manifest,
        transpose_indices=transpose_indices,
        zigzag_indices=zigzag_indices,
    )
    logical_population = _decode_population(population_raw, manifest, descriptor["t"])
    logical_manifest = dict(manifest)
    logical_manifest.pop("pg2")
    logical_manifest.pop("xcodec", None)
    manifest_bytes = _canonical_json(logical_manifest)
    return _pack_lvls1(
        manifest_bytes,
        brotli.compress(logical_base, quality=11),
        brotli.compress(logical_population, quality=11),
        blocks[3],
    )


def _atomic_write(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def _bytecode_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts)
        )
    )


def _require_authority_environment(runtime_dir: Path) -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PublicInverseError("public inverse requires PYTHONDONTWRITEBYTECODE=1")
    if int(os.environ.get("INFLATE_MAX_PAIRS", "0")) != 0:
        raise PublicInverseError("public authority decode forbids INFLATE_MAX_PAIRS")
    if os.environ.get("INFLATE_FP32", "0") != "0":
        raise PublicInverseError("public authority decode forbids nonportable INFLATE_FP32")
    contamination = _bytecode_paths(runtime_dir)
    if contamination:
        raise PublicInverseError(f"runtime bytecode contamination refused: {contamination}")
    expected = {"inflate.sh", "inflate.py", "lvls1_runtime.py"}
    entries = tuple(runtime_dir.iterdir())
    if {path.name for path in entries} != expected or any(not path.is_file() or path.is_symlink() for path in entries):
        raise PublicInverseError("public runtime directory is not the exact three-file non-symlink tree")


def _install_decoder_audit_policy(
    *,
    runtime_path: Path,
    source_member: Path,
    logical_member: Path,
    output_path: Path,
    scratch_dir: Path,
    trace_path: Path | None,
) -> str:
    """Deny inverse/renderer escapes before the counted source member is read."""

    read_roots = tuple(
        dict.fromkeys(Path(value).resolve() for value in (Path(sys.prefix), Path(sys.base_prefix), scratch_dir))
    )
    write_root_values = [output_path.parent.resolve(), scratch_dir.resolve()]
    if trace_path is not None:
        write_root_values.append(trace_path.parent.resolve())
    write_roots = tuple(dict.fromkeys(write_root_values))
    exact_read_files = {
        source_member.resolve(),
        logical_member.resolve(),
        output_path.resolve(),
        Path(__file__).resolve(),
        runtime_path.resolve(),
    }
    exact_write_files = {
        output_path.resolve(),
        output_path.with_name(output_path.name + ".partial").resolve(),
    }
    if trace_path is not None:
        exact_write_files.add(trace_path.resolve())

    def under(path: Path, roots: tuple[Path, ...]) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event in {
            "os.exec",
            "os.getrandom",
            "os.posix_spawn",
            "os.spawn",
            "os.system",
            "os.urandom",
            "socket.__new__",
            "socket.bind",
            "socket.connect",
            "socket.getaddrinfo",
            "subprocess.Popen",
        }:
            raise PermissionError(f"public decoder policy denied {event}")
        if event == "ctypes.dlopen":
            library = args[0] if args else None
            if library is None:
                return
            if not isinstance(library, (str, bytes, os.PathLike)):
                raise PermissionError("public decoder policy denied non-path ctypes load")
            library_path = Path(os.fsdecode(library)).resolve()
            abi_roots = tuple(Path(value).resolve() for value in (sys.prefix, sys.base_prefix))
            if not under(library_path, abi_roots):
                raise PermissionError(
                    f"public decoder policy denied ctypes load outside ABI roots: {library_path.name}"
                )
            return
        if event in {"os.remove", "os.rmdir"} and args:
            removed = Path(os.fsdecode(args[0])).resolve()
            if removed not in exact_write_files and not under(removed, write_roots):
                raise PermissionError(f"public decoder policy denied external removal: {removed.name}")
            return
        if event == "os.rename" and len(args) >= 2:
            source = Path(os.fsdecode(args[0])).resolve()
            destination = Path(os.fsdecode(args[1])).resolve()
            if any(path not in exact_write_files and not under(path, write_roots) for path in (source, destination)):
                raise PermissionError("public decoder policy denied external rename")
            return
        if event != "open" or not args or isinstance(args[0], int):
            return
        raw_path = args[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            raise PermissionError("public decoder policy denied non-path open")
        path = Path(os.fsdecode(raw_path)).resolve()
        mode_or_flags = args[1] if len(args) > 1 else "r"
        writing = (
            isinstance(mode_or_flags, str) and any(token in mode_or_flags for token in ("w", "a", "x", "+"))
        ) or (
            isinstance(mode_or_flags, int)
            and bool(mode_or_flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        )
        if writing:
            if path not in exact_write_files and not under(path, write_roots):
                raise PermissionError(f"public decoder policy denied write outside output roots: {path.name}")
        elif path not in exact_read_files and not under(path, read_roots):
            raise PermissionError(f"public decoder policy denied external read: {path.name}")

    sys.addaudithook(audit)
    return "PYTHON_AUDIT_DENY_NETWORK_EXEC_ENTROPY_AND_EXTERNAL_FILE_READS_V2"


def _expected_raw_nbytes(manifest: dict[str, Any]) -> int:
    values = tuple(manifest.get(name) for name in ("n_pairs", "camera_h", "camera_w"))
    if any(type(value) is not int or value < 1 for value in values):
        raise PublicInverseError("LVPG2 raw contract dimensions are invalid")
    n_pairs, camera_h, camera_w = values
    return 2 * n_pairs * camera_h * camera_w * 3


def _write_trace(
    *,
    trace_path: Path,
    source_member: Path,
    source_bytes: bytes,
    logical_member: bytes,
    runtime_path: Path,
    output_path: Path,
    argv: list[str],
    decode_wall_seconds: float,
    filesystem_policy: str,
) -> None:
    trace = {
        "argv": argv,
        "decoder_phase_scope": "LVPG2_INVERSE_PLUS_GENERIC_LVLS1_RENDERER_ONLY",
        "dependency_paths": ["inflate.py", "lvls1_runtime.py"],
        "decode_wall_seconds": decode_wall_seconds,
        "entropy_denied": True,
        "external_reads_denied": True,
        "filesystem_policy": filesystem_policy,
        "inverse_source_sha256": _sha256_file(Path(__file__).resolve()),
        "logical_lvls1_sha256": _sha256(logical_member),
        "network_denied": True,
        "output_raw_bytes": output_path.stat().st_size,
        "output_raw_sha256": _sha256_file(output_path),
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "runtime_sha256": _sha256_file(runtime_path),
        "schema": _TRACE_SCHEMA,
        "source_member_name": source_member.name,
        "source_member_sha256": _sha256(source_bytes),
        "subprocess_denied": True,
    }
    _atomic_write(trace_path, _canonical_json(trace))


def materialize_lvls1(source: Path, destination: Path) -> None:
    """Write a conformance LVLS1 object atomically; this does not render frames."""

    source_bytes = source.read_bytes()
    logical = lvpg2_to_lvls1(source_bytes)
    _atomic_write(destination, logical)


def inflate(source: Path, destination: Path) -> None:
    """Run the deterministic generic LVLS1 renderer on the exact inverse."""

    runtime_dir = Path(__file__).resolve().parent
    _require_authority_environment(runtime_dir)
    runtime_path = runtime_dir / "lvls1_runtime.py"
    if not runtime_path.is_file():
        raise PublicInverseError("public runtime is missing lvls1_runtime.py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    trace_value = os.environ.get("PACT_AUTH_TRACE_PATH")
    trace_path = None if not trace_value else Path(trace_value)
    decode_started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lvpg2_inverse_", dir=destination.parent) as scratch:
        scratch_path = Path(scratch)
        logical_path = scratch_path / "logical.lvls1"
        filesystem_policy = _install_decoder_audit_policy(
            runtime_path=runtime_path,
            source_member=source,
            logical_member=logical_path,
            output_path=destination,
            scratch_dir=scratch_path,
            trace_path=trace_path,
        )
        source_bytes = source.read_bytes()
        manifest, _blocks = _read_four_blocks(source_bytes)
        logical = lvpg2_to_lvls1(source_bytes)
        _atomic_write(logical_path, logical)
        prior_argv = sys.argv
        try:
            sys.argv = [os.fspath(runtime_path), os.fspath(logical_path), os.fspath(destination)]
            try:
                runpy.run_path(os.fspath(runtime_path), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise PublicInverseError(f"generic LVLS1 renderer exited {exc.code}") from exc
        finally:
            sys.argv = prior_argv
    expected = _expected_raw_nbytes(manifest)
    if not destination.is_file() or destination.stat().st_size != expected:
        actual = destination.stat().st_size if destination.exists() else -1
        raise PublicInverseError(f"public raw output has {actual} byte(s), expected exactly {expected}")
    contamination = _bytecode_paths(runtime_dir)
    if contamination:
        raise PublicInverseError(f"runtime generated forbidden bytecode: {contamination}")
    if trace_path is not None:
        _write_trace(
            trace_path=trace_path,
            source_member=source,
            source_bytes=source_bytes,
            logical_member=logical,
            runtime_path=runtime_path,
            output_path=destination,
            argv=list(sys.argv),
            decode_wall_seconds=time.monotonic() - decode_started,
            filesystem_policy=filesystem_policy,
        )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 3 and arguments[0] == "--materialize-lvls1":
        if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
            raise PublicInverseError("conformance materialization requires PYTHONDONTWRITEBYTECODE=1")
        materialize_lvls1(Path(arguments[1]), Path(arguments[2]))
        return 0
    if len(arguments) != 2:
        raise SystemExit(
            "usage: inflate.py SOURCE_LVPG2 DEST_RAW\n       inflate.py --materialize-lvls1 SOURCE_LVPG2 DEST_LVLS1"
        )
    inflate(Path(arguments[0]), Path(arguments[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
