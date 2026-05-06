#!/usr/bin/env python3
"""Build PR85 side-channel lossless recode candidates.

This is a local byte-screening tool. It parses the public PR85 single-member
``x`` bundle, proposes runtime-supported encodings for side-channel choice
streams, verifies decoded-semantics parity without importing the CUDA runtime,
and emits deterministic candidate archives only when the public PR85 parser can
recover the new segment boundaries. It does not claim score, touch dispatch
state, or submit remote jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
PROFILE_PATH = EXPERIMENTS_ROOT / "profile_pr85_adaptive_masking_sidechannel_attribution.py"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_REPLAY_RUNTIME_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/replay_submission"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/public_pr85_sidechannel_recodes_20260503_codex"
TOOL = "experiments/build_pr85_sidechannel_recode_candidates.py"
SCHEMA = "pr85_sidechannel_recode_candidates_v1"
MANIFEST_SCHEMA = "pr85_sidechannel_recode_candidate_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAIR_COUNT = 600
PR85_FRONTIER_SCORE = 0.25806611029397786  # [external: PR-85 contest-CUDA T4 frontier]
PR85_FRONTIER_BYTES = 236_328
PR85_FRONTIER_SHA256 = "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
SIDECHANNELS = ("post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti")
FIXED_V5_LENGTHS = {"bias": 223, "region": 273}
BROTLIS = tuple((q, lgwin) for q in (5, 9, 11) for lgwin in (18, 20, 22, 24))


def _load_profile_module() -> Any:
    spec = importlib.util.spec_from_file_location("pr85_sidechannel_profile_for_recode", PROFILE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load PR85 profile helper from {PROFILE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile = _load_profile_module()
SEGMENT_ORDER: tuple[str, ...] = tuple(profile.SEGMENT_ORDER)
HEADERLESS_RANDMULTI_SPECS: tuple[tuple[int, int, int, int], ...] = tuple(
    profile.HEADERLESS_RANDMULTI_SPECS
)


@dataclass(frozen=True)
class Encoding:
    segment: str
    variant: str
    decoded: bytes
    raw: bytes
    compressed: bytes
    brotli_params: dict[str, int | str] | None
    runtime_supported: bool
    support_reason: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _u24(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError(f"cannot encode {value} as uint24")
    return int(value).to_bytes(3, "little")


def _safe_zip_member(name: str) -> str:
    path = Path(name)
    if name != "x" or name.startswith("/") or ".." in path.parts:
        raise ValueError(f"PR85 recode archives must use safe single member 'x', got {name!r}")
    return name


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_zip_member(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), payload)


def _archive_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one member in {path}, found {len(infos)}")
        info = infos[0]
        raw = zf.read(info)
    return {
        "archive_path": str(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_bytes": int(len(raw)),
        "member_sha256": _sha256(raw),
        "zip_stored": info.compress_type == zipfile.ZIP_STORED,
    }


def _runtime_dependency_closure(replay_runtime_dir: Path) -> dict[str, Any]:
    required = ("inflate.sh", "inflate.py", "range_mask_codec.cpp")
    files = []
    missing = []
    for name in required:
        path = replay_runtime_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        files.append(
            {
                "name": name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    status = "passed" if not missing else "failed"
    return {
        "status": status,
        "runtime_dir": str(replay_runtime_dir),
        "required_files": files,
        "missing_files": missing,
        "hidden_archive_sidecars": False,
        "closure_scope": "candidate archive contains only member x; replay runtime files are external exact-eval custody inputs",
    }


def _read_pr85_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    archive, raw = profile._read_single_member_archive(path, "x")  # noqa: SLF001
    return archive, raw


def _parse_bundle(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(raw) >= 30:
        lengths = {
            "mask": int.from_bytes(raw[0:3], "little"),
            "model": int.from_bytes(raw[3:6], "little"),
            "pose": int.from_bytes(raw[6:9], "little"),
            "post": int.from_bytes(raw[9:12], "little"),
            "shift": int.from_bytes(raw[12:15], "little"),
            "frac": int.from_bytes(raw[15:18], "little"),
            "frac2": int.from_bytes(raw[18:21], "little"),
            "frac3": int.from_bytes(raw[21:24], "little"),
            "bias": int.from_bytes(raw[24:27], "little"),
            "region": int.from_bytes(raw[27:30], "little"),
        }
        if (
            lengths["mask"] > 1000
            and lengths["model"] > 1000
            and lengths["pose"] > 100
            and all(0 < lengths[name] < 10000 for name in SEGMENT_ORDER[3:-1])
        ):
            return _slice_bundle(raw, lengths, 30, "pr85_explicit_30byte_lengths")
    bundle, segments = profile.parse_pr85_v5_bundle(raw)
    return bundle, segments


def _slice_bundle(
    raw: bytes,
    lengths: dict[str, int],
    header_bytes: int,
    fmt: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    pos = header_bytes
    segments: dict[str, bytes] = {}
    offsets: dict[str, int] = {}
    for name in SEGMENT_ORDER[:-1]:
        size = lengths[name]
        end = pos + size
        if size <= 0 or end > len(raw):
            raise ValueError(f"invalid PR85 segment slice for {name}: size={size}")
        offsets[name] = pos
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise ValueError("PR85 bundle is missing randmulti tail")
    offsets["randmulti"] = pos
    segments["randmulti"] = raw[pos:]
    return (
        {
            "format": fmt,
            "header_bytes": header_bytes,
            "segment_offsets": offsets,
            "segment_lengths": {name: len(segments[name]) for name in SEGMENT_ORDER},
            "fixed_length_segments": {} if header_bytes == 30 else dict(FIXED_V5_LENGTHS),
        },
        segments,
    )


def _pack_bundle(segments: dict[str, bytes], *, header_mode: str) -> bytes:
    if header_mode == "v5":
        for name, fixed_len in FIXED_V5_LENGTHS.items():
            actual = len(segments[name])
            if actual != fixed_len:
                raise ValueError(
                    f"cannot pack PR85 v5 bundle with changed fixed-length segment {name!r}: "
                    f"got {actual}, expected {fixed_len}"
                )
        header_names = SEGMENT_ORDER[:8]
        return b"".join(_u24(len(segments[name])) for name in header_names) + b"".join(
            segments[name] for name in SEGMENT_ORDER
        )
    if header_mode == "explicit_30":
        header_names = SEGMENT_ORDER[:10]
        return b"".join(_u24(len(segments[name])) for name in header_names) + b"".join(
            segments[name] for name in SEGMENT_ORDER
        )
    raise ValueError(f"unknown header mode {header_mode!r}")


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode negative values")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    for _ in range(count):
        acc = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise ValueError("truncated varint stream")
            byte = raw[pos]
            pos += 1
            acc |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
                if shift > 28:
                    raise ValueError("oversized varint")
            else:
                values.append(acc)
                break
    return values, pos


def _brotli_best(raw: bytes) -> tuple[bytes, dict[str, int | str]]:
    best = brotli.compress(raw, quality=11)
    best_params: dict[str, int | str] = {"quality": 11, "lgwin": "default"}
    for quality, lgwin in BROTLIS:
        candidate = brotli.compress(raw, quality=quality, lgwin=lgwin)
        if len(candidate) < len(best) or (len(candidate) == len(best) and candidate < best):
            best = candidate
            best_params = {"quality": quality, "lgwin": lgwin}
    if brotli.decompress(best) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    return best, best_params


def _decode_segment_raw(name: str, segment: bytes) -> bytes:
    if name == "mask":
        return segment
    return brotli.decompress(segment)


def _choice_payload_values(raw: bytes, *, default_center: int | None = None) -> bytes:
    magic = raw[:3]
    values = raw[3:]
    if magic in {b"SH4", b"FH1", b"FH2", b"FH3", b"BH1", b"RH1"}:
        return bytes(values)
    if magic in {b"SD4", b"FD3", b"BD1", b"RD1"}:
        if default_center is None:
            raise ValueError(f"default center required for {magic!r}")
        return bytes(default_center if value == 0 else value - 1 for value in values)
    raise ValueError(f"unsupported dense choice magic {magic!r}")


def _sparse_values(raw: bytes, *, magic: bytes, default_choice: int) -> bytes:
    if not raw.startswith(magic) or len(raw) < 5:
        raise ValueError(f"bad sparse payload magic for {magic!r}")
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    vals = raw[pos : pos + count]
    if len(vals) != count or pos + count != len(raw):
        raise ValueError("sparse payload length mismatch")
    out = bytearray([default_choice] * PAIR_COUNT)
    idx = -1
    for gap, value in zip(gaps, vals):
        idx += gap + 1
        if not 0 <= idx < PAIR_COUNT:
            raise ValueError("sparse payload index out of range")
        out[idx] = value - 1
    return bytes(out)


def _decode_choice_semantics(name: str, raw: bytes) -> bytes:
    if name == "shift":
        return _choice_payload_values(raw, default_center=40)
    if name == "frac":
        if raw[:3] == b"FV1":
            return _sparse_values(raw, magic=b"FV1", default_choice=4)
        if raw[:3] == b"FH1":
            return _choice_payload_values(raw)
    if name == "frac2":
        if raw[:3] == b"FH2":
            return _choice_payload_values(raw)
    if name == "frac3":
        return _choice_payload_values(raw, default_center=4)
    if name == "bias":
        if raw[:3] == b"BV1":
            return _sparse_values(raw, magic=b"BV1", default_choice=13)
        return _choice_payload_values(raw, default_center=13)
    if name == "region":
        if raw[:3] == b"RV1":
            return _sparse_values(raw, magic=b"RV1", default_choice=0)
        return _choice_payload_values(raw, default_center=0)
    raise ValueError(f"no simple choice decoder for {name!r}")


def _encode_sparse_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    indices = [idx for idx, value in enumerate(values) if value != default_choice]
    if len(indices) > 0xFFFF:
        raise ValueError("too many sparse choices")
    gaps = []
    last = -1
    vals = bytearray()
    for idx in indices:
        gaps.append(idx - last - 1)
        vals.append(values[idx] + 1)
        last = idx
    return magic + len(indices).to_bytes(2, "little") + b"".join(_varint(gap) for gap in gaps) + bytes(vals)


def _encode_delta_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    encoded = bytes(0 if value == default_choice else value + 1 for value in values)
    return magic + encoded


def _decode_post_semantics(raw: bytes) -> bytes:
    if raw[:4] == b"PCD1":
        pos = 4
        stage_count = raw[pos]
        pos += 1
        out = bytearray()
        for _ in range(stage_count):
            if pos + 3 > len(raw):
                raise ValueError("truncated PCD1 post stage")
            stage_id = raw[pos]
            pos += 1
            count = int.from_bytes(raw[pos : pos + 2], "little")
            pos += 2
            choices = raw[pos : pos + count]
            pos += count
            out += bytes([stage_id]) + count.to_bytes(2, "little") + choices
        if pos != len(raw):
            raise ValueError("PCD1 post payload has trailing bytes")
        return bytes(out)
    if len(raw) % PAIR_COUNT:
        raise ValueError("bad headerless post length")
    stage_count = len(raw) // PAIR_COUNT
    if stage_count not in (3, 4):
        raise ValueError(f"unsupported headerless post stage count {stage_count}")
    out = bytearray()
    pos = 0
    for stage_id in range(1, stage_count + 1):
        choices = raw[pos : pos + PAIR_COUNT]
        pos += PAIR_COUNT
        out += bytes([stage_id]) + PAIR_COUNT.to_bytes(2, "little") + choices
    return bytes(out)


def _post_raw_from_semantics(semantic: bytes, *, variant: str) -> bytes:
    pos = 0
    stages: list[tuple[int, bytes]] = []
    while pos < len(semantic):
        if pos + 3 > len(semantic):
            raise ValueError("truncated post semantic record")
        stage_id = semantic[pos]
        pos += 1
        count = int.from_bytes(semantic[pos : pos + 2], "little")
        pos += 2
        choices = semantic[pos : pos + count]
        pos += count
        if len(choices) != count:
            raise ValueError("truncated post semantic choices")
        stages.append((stage_id, choices))
    if variant == "headerless":
        if [stage for stage, _ in stages] != list(range(1, len(stages) + 1)):
            raise ValueError("post stages are not headerless-compatible")
        if len(stages) not in (3, 4) or any(len(choices) != PAIR_COUNT for _, choices in stages):
            raise ValueError("post stages are not PR85 headerless-compatible")
        return b"".join(choices for _, choices in stages)
    if variant == "pcd1":
        if len(stages) > 255:
            raise ValueError("too many post stages")
        out = bytearray(b"PCD1" + bytes([len(stages)]))
        for stage_id, choices in stages:
            out.append(stage_id)
            out += len(choices).to_bytes(2, "little")
            out += choices
        return bytes(out)
    raise ValueError(f"unknown post variant {variant!r}")


def _decode_randmulti_semantics(raw: bytes) -> bytes:
    groups = _decode_randmulti_groups(raw)
    out = bytearray()
    for group_index, (lh, lw, amp, scount, rows) in enumerate(groups):
        out += group_index.to_bytes(2, "little")
        out += int(lh).to_bytes(4, "little")
        out += int(lw).to_bytes(4, "little")
        out += int(amp).to_bytes(2, "little")
        out += int(scount).to_bytes(2, "little")
        for row in rows:
            out += bytes(row)
    return bytes(out)


def _decode_randmulti_groups(raw: bytes) -> list[tuple[int, int, int, int, list[bytes]]]:
    if raw[:3] == b"NM1" and len(raw) >= 4:
        scount = raw[3]
        payload = raw[4:]
        if len(payload) != scount * PAIR_COUNT:
            raise ValueError("NM1 randmulti length mismatch")
        return [(24, 32, 1, scount, [payload[i * PAIR_COUNT : (i + 1) * PAIR_COUNT] for i in range(scount)])]
    if raw[:3] == b"NM2" and len(raw) >= 4:
        pos = 4
        groups = []
        for _ in range(raw[3]):
            if pos + 4 > len(raw):
                raise ValueError("truncated NM2 group")
            lh, lw, amp, scount = raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]
            pos += 4
            rows = []
            for _row in range(scount):
                row = raw[pos : pos + PAIR_COUNT]
                pos += PAIR_COUNT
                if len(row) != PAIR_COUNT:
                    raise ValueError("truncated NM2 row")
                rows.append(bytes(row))
            groups.append((lh, lw, amp, scount, rows))
        if pos != len(raw):
            raise ValueError("NM2 randmulti trailing bytes")
        return groups
    pos = 0
    groups = []
    for lh, lw, amp, scount in HEADERLESS_RANDMULTI_SPECS:
        rows = []
        for _ in range(scount):
            if pos >= len(raw):
                raise ValueError("truncated headerless randmulti count")
            count = raw[pos]
            pos += 1
            if count == 255:
                if pos + 2 > len(raw):
                    raise ValueError("truncated extended headerless randmulti count")
                count = int.from_bytes(raw[pos : pos + 2], "little")
                pos += 2
            gaps, pos = _read_varints(raw, pos, count)
            values = raw[pos : pos + count]
            pos += count
            if len(values) != count:
                raise ValueError("truncated headerless randmulti values")
            row = bytearray(PAIR_COUNT)
            idx = -1
            for gap, value in zip(gaps, values):
                idx += gap + 1
                if not 0 <= idx < PAIR_COUNT:
                    raise ValueError("headerless randmulti index out of range")
                row[idx] = value
            rows.append(bytes(row))
        groups.append((lh, lw, amp, scount, rows))
    if pos != len(raw):
        raise ValueError("headerless randmulti trailing bytes")
    return groups


def _encode_randmulti_headerless(semantic: bytes) -> bytes:
    groups = _semantic_randmulti_groups(semantic)
    if [(lh, lw, amp, scount) for lh, lw, amp, scount, _ in groups] != [
        (lh, lw, amp, scount) for lh, lw, amp, scount in HEADERLESS_RANDMULTI_SPECS
    ]:
        raise ValueError("randmulti semantic groups do not match PR85 headerless schedule")
    out = bytearray()
    for _lh, _lw, _amp, _scount, rows in groups:
        for row in rows:
            indices = [idx for idx, value in enumerate(row) if value]
            if len(indices) >= 255:
                out.append(255)
                out += len(indices).to_bytes(2, "little")
            else:
                out.append(len(indices))
            last = -1
            for idx in indices:
                out += _varint(idx - last - 1)
                last = idx
            out += bytes(row[idx] for idx in indices)
    return bytes(out)


def _semantic_randmulti_groups(semantic: bytes) -> list[tuple[int, int, int, int, list[bytes]]]:
    pos = 0
    groups = []
    group_index = 0
    while pos < len(semantic):
        if pos + 14 > len(semantic):
            raise ValueError("truncated randmulti semantic header")
        observed_index = int.from_bytes(semantic[pos : pos + 2], "little")
        pos += 2
        if observed_index != group_index:
            raise ValueError("randmulti semantic group index mismatch")
        lh = int.from_bytes(semantic[pos : pos + 4], "little")
        lw = int.from_bytes(semantic[pos + 4 : pos + 8], "little")
        amp = int.from_bytes(semantic[pos + 8 : pos + 10], "little")
        scount = int.from_bytes(semantic[pos + 10 : pos + 12], "little")
        pos += 12
        rows = []
        for _ in range(scount):
            row = semantic[pos : pos + PAIR_COUNT]
            pos += PAIR_COUNT
            if len(row) != PAIR_COUNT:
                raise ValueError("truncated randmulti semantic row")
            rows.append(bytes(row))
        groups.append((lh, lw, amp, scount, rows))
        group_index += 1
    return groups


def _encode_randmulti_nm2(semantic: bytes) -> bytes:
    groups = _semantic_randmulti_groups(semantic)
    out = bytearray(b"NM2" + bytes([len(groups)]))
    for lh, lw, amp, scount, rows in groups:
        if not all(0 <= value <= 255 for value in (lh, lw, amp, scount)):
            raise ValueError("NM2 cannot encode this randmulti group header")
        out += bytes([lh, lw, amp, scount])
        for row in rows:
            out += row
    return bytes(out)


def _segment_semantics(name: str, decoded_raw: bytes) -> bytes:
    if name == "post":
        return _decode_post_semantics(decoded_raw)
    if name == "randmulti":
        return _decode_randmulti_semantics(decoded_raw)
    if name in {"shift", "frac", "frac2", "frac3", "bias", "region"}:
        values = _decode_choice_semantics(name, decoded_raw)
        if len(values) != PAIR_COUNT:
            raise ValueError(f"{name} decoded choice count {len(values)} != {PAIR_COUNT}")
        return values
    raise ValueError(f"unsupported side-channel segment {name!r}")


def _segment_raw_variants(name: str, semantic: bytes, source_decoded: bytes) -> list[tuple[str, bytes, bool, str]]:
    if name == "post":
        variants = [("raw_resweep", source_decoded, True, "public runtime decodes current post grammar")]
        for variant in ("headerless", "pcd1"):
            try:
                variants.append(
                    (
                        variant,
                        _post_raw_from_semantics(semantic, variant=variant),
                        True,
                        f"public runtime supports {variant} post grammar",
                    )
                )
            except ValueError as exc:
                variants.append((variant, source_decoded, False, str(exc)))
        return variants
    if name == "shift":
        return [
            ("raw_resweep", source_decoded, True, "public runtime decodes current shift grammar"),
            ("SH4_absolute", b"SH4" + semantic, True, "public runtime supports SH4 absolute choices"),
            ("SD4_delta_center40", _encode_delta_choice(b"SD4", semantic, default_choice=40), True, "public runtime supports SD4 delta choices"),
        ]
    if name == "frac":
        return [
            ("raw_resweep", source_decoded, True, "public runtime decodes current frac grammar"),
            ("FH1_absolute", b"FH1" + semantic, True, "public runtime supports FH1 absolute choices"),
            ("FV1_sparse_default4", _encode_sparse_choice(b"FV1", semantic, default_choice=4), True, "public runtime supports FV1 sparse choices"),
        ]
    if name == "frac2":
        return [
            ("raw_resweep", source_decoded, True, "public runtime decodes current frac2 grammar"),
            ("FH2_absolute", b"FH2" + semantic, True, "public runtime supports FH2 absolute choices"),
        ]
    if name == "frac3":
        return [
            ("raw_resweep", source_decoded, True, "public runtime decodes current frac3 grammar"),
            ("FH3_absolute", b"FH3" + semantic, True, "public runtime supports FH3 absolute choices"),
            ("FD3_delta_center4", _encode_delta_choice(b"FD3", semantic, default_choice=4), True, "public runtime supports FD3 delta choices"),
        ]
    if name == "bias":
        return [
            ("raw_resweep", source_decoded, True, "public runtime decodes current bias grammar"),
            ("BH1_absolute_explicit_header", b"BH1" + semantic, True, "requires explicit 30-byte bundle header if compressed length changes"),
            ("BD1_delta_center13_explicit_header", _encode_delta_choice(b"BD1", semantic, default_choice=13), True, "requires explicit 30-byte bundle header if compressed length changes"),
            ("BV1_sparse_default13_explicit_header", _encode_sparse_choice(b"BV1", semantic, default_choice=13), True, "requires explicit 30-byte bundle header if compressed length changes"),
        ]
    if name == "region":
        return [
            ("raw_resweep", source_decoded, True, "public runtime decodes current region grammar"),
            ("RH1_absolute_explicit_header", b"RH1" + semantic, True, "requires explicit 30-byte bundle header if compressed length changes"),
            ("RD1_delta_zero_explicit_header", _encode_delta_choice(b"RD1", semantic, default_choice=0), True, "requires explicit 30-byte bundle header if compressed length changes"),
            ("RV1_sparse_default0_explicit_header", _encode_sparse_choice(b"RV1", semantic, default_choice=0), True, "requires explicit 30-byte bundle header if compressed length changes"),
        ]
    if name == "randmulti":
        variants = [
            ("raw_resweep", source_decoded, True, "public runtime decodes current randmulti grammar"),
            ("headerless_sparse_canonical", _encode_randmulti_headerless(semantic), True, "public runtime supports PR85 headerless randmulti sparse tables"),
        ]
        try:
            variants.append(("NM2_dense", _encode_randmulti_nm2(semantic), True, "public runtime supports NM2 dense randmulti tables"))
        except ValueError as exc:
            variants.append(("NM2_dense", source_decoded, False, str(exc)))
        return variants
    raise ValueError(f"unsupported segment {name!r}")


def _make_encodings(name: str, source_segment: bytes) -> list[Encoding]:
    decoded = _decode_segment_raw(name, source_segment)
    semantic = _segment_semantics(name, decoded)
    encodings: list[Encoding] = []
    seen: set[tuple[str, str]] = set()
    for variant, raw, runtime_supported, reason in _segment_raw_variants(name, semantic, decoded):
        compressed, params = _brotli_best(raw)
        decoded_check = _decode_segment_raw(name, compressed)
        if _segment_semantics(name, decoded_check) != semantic:
            raise ValueError(f"{name} variant {variant} failed decoded-semantics parity")
        key = (variant, _sha256(compressed))
        if key in seen:
            continue
        seen.add(key)
        encodings.append(
            Encoding(
                segment=name,
                variant=variant,
                decoded=semantic,
                raw=raw,
                compressed=compressed,
                brotli_params=params,
                runtime_supported=runtime_supported,
                support_reason=reason,
            )
        )
    return encodings


def _best_encoding(name: str, source_segment: bytes) -> Encoding:
    encodings = [encoding for encoding in _make_encodings(name, source_segment) if encoding.runtime_supported]
    if not encodings:
        raise ValueError(f"no runtime-supported encodings for {name}")
    return min(encodings, key=lambda enc: (len(enc.compressed), enc.variant, enc.compressed))


def _transform_record(name: str, source_segment: bytes, encoding: Encoding) -> dict[str, Any]:
    source_decoded = _decode_segment_raw(name, source_segment)
    source_semantic = _segment_semantics(name, source_decoded)
    candidate_semantic = _segment_semantics(name, _decode_segment_raw(name, encoding.compressed))
    parity = source_semantic == candidate_semantic
    return {
        "segment": name,
        "variant": encoding.variant,
        "runtime_supported": encoding.runtime_supported,
        "support_reason": encoding.support_reason,
        "source_segment_bytes": int(len(source_segment)),
        "candidate_segment_bytes": int(len(encoding.compressed)),
        "segment_byte_delta": int(len(encoding.compressed) - len(source_segment)),
        "source_segment_sha256": _sha256(source_segment),
        "candidate_segment_sha256": _sha256(encoding.compressed),
        "source_decoded_bytes": int(len(source_decoded)),
        "candidate_decoded_bytes": int(len(encoding.raw)),
        "source_semantic_sha256": _sha256(source_semantic),
        "candidate_semantic_sha256": _sha256(candidate_semantic),
        "decoded_parity_status": "passed" if parity else "failed",
        "brotli_params": encoding.brotli_params,
        "noop_segment": source_segment == encoding.compressed,
    }


def _candidate_policy_segments(policy_id: str, source_segments: dict[str, bytes]) -> dict[str, Encoding]:
    if policy_id.startswith("segment_") and policy_id.endswith("_best"):
        name = policy_id.removeprefix("segment_").removesuffix("_best")
        if name not in SIDECHANNELS:
            raise ValueError(f"unknown segment policy {policy_id!r}")
        return {name: _best_encoding(name, source_segments[name])}
    best_by_segment = {name: _best_encoding(name, source_segments[name]) for name in SIDECHANNELS}
    if policy_id == "all_best":
        return best_by_segment
    if policy_id == "all_positive_best":
        return {
            name: encoding
            for name, encoding in best_by_segment.items()
            if len(encoding.compressed) < len(source_segments[name])
        }
    if policy_id == "micro_positive_best":
        return {
            name: best_by_segment[name]
            for name in ("shift", "frac", "frac2", "frac3", "bias", "region")
            if len(best_by_segment[name].compressed) < len(source_segments[name])
        }
    if policy_id == "randmulti_best":
        return {"randmulti": best_by_segment["randmulti"]}
    raise ValueError(f"unknown policy {policy_id!r}")


def _header_mode_for_segments(source_segments: dict[str, bytes], candidate_segments: dict[str, bytes]) -> str:
    fixed_changed = [
        name
        for name, fixed_len in FIXED_V5_LENGTHS.items()
        if len(candidate_segments[name]) != fixed_len or len(source_segments[name]) != fixed_len
    ]
    return "explicit_30" if fixed_changed else "v5"


def _validate_candidate_bundle(
    payload: bytes,
    expected_segments: dict[str, bytes],
) -> dict[str, Any]:
    bundle, parsed = _parse_bundle(payload)
    mismatches = []
    for name in SEGMENT_ORDER:
        if parsed[name] != expected_segments[name]:
            mismatches.append(name)
    return {
        "status": "passed" if not mismatches else "failed",
        "bundle_format": bundle["format"],
        "header_bytes": int(bundle["header_bytes"]),
        "segment_lengths": bundle["segment_lengths"],
        "fixed_length_segments": bundle.get("fixed_length_segments", {}),
        "mismatched_segments": mismatches,
    }


def build_candidates(
    archive: Path,
    out_dir: Path,
    *,
    policy_ids: list[str] | None = None,
    replay_runtime_dir: Path = DEFAULT_REPLAY_RUNTIME_DIR,
) -> dict[str, Any]:
    source_archive, raw = _read_pr85_archive(archive)
    source_bundle, source_segments = _parse_bundle(raw)
    runtime_closure = _runtime_dependency_closure(replay_runtime_dir)
    selected = policy_ids or [
        "all_positive_best",
        "micro_positive_best",
        "randmulti_best",
        "segment_post_best",
        "segment_shift_best",
        "segment_frac_best",
        "segment_frac2_best",
        "segment_frac3_best",
        "segment_bias_best",
        "segment_region_best",
    ]
    rows = []
    for policy_id in selected:
        replacements = _candidate_policy_segments(policy_id, source_segments)
        candidate_segments = dict(source_segments)
        transforms = []
        for name, encoding in replacements.items():
            candidate_segments[name] = encoding.compressed
            transforms.append(_transform_record(name, source_segments[name], encoding))
        header_mode = _header_mode_for_segments(source_segments, candidate_segments)
        payload = _pack_bundle(candidate_segments, header_mode=header_mode)
        validation = _validate_candidate_bundle(payload, candidate_segments)
        if validation["status"] != "passed":
            raise ValueError(f"candidate {policy_id} failed bundle parse validation")

        candidate_dir = out_dir / policy_id
        archive_path = candidate_dir / "archive.zip"
        _write_archive(archive_path, payload)
        info = _archive_info(archive_path)
        changed = [row["segment"] for row in transforms if not row["noop_segment"]]
        all_parity = all(row["decoded_parity_status"] == "passed" for row in transforms)
        byte_delta = int(info["archive_bytes"] - source_archive["bytes"])
        dispatchable = bool(changed) and all_parity and byte_delta < 0 and all(
            row["runtime_supported"] for row in transforms
        ) and runtime_closure["status"] == "passed"
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool": TOOL,
            "policy_id": policy_id,
            "score_claim": False,
            "dispatch_performed": False,
            "evidence_grade": "empirical_lossless_decoded_semantics_recode",
            "source_frontier": {
                "score": PR85_FRONTIER_SCORE,
                "archive_bytes": PR85_FRONTIER_BYTES,
                "archive_sha256": PR85_FRONTIER_SHA256,
                "score_source": "exact T4 PR85 replay; included for comparison only",
            },
            "source_archive": source_archive,
            "source_bundle": {
                "format": source_bundle["format"],
                "header_bytes": int(source_bundle["header_bytes"]),
                "segment_lengths": source_bundle["segment_lengths"],
                "fixed_length_segments": source_bundle.get("fixed_length_segments", {}),
            },
            "candidate": info,
            "candidate_bundle_validation": validation,
            "runtime_dependency_closure": runtime_closure,
            "header_mode": header_mode,
            "fixed_length_parser_safety": {
                "status": "passed",
                "v5_fixed_lengths_preserved": header_mode == "v5",
                "explicit_30byte_header_used": header_mode == "explicit_30",
                "reason": (
                    "bias/region fixed v5 lengths preserved"
                    if header_mode == "v5"
                    else "candidate uses public-runtime explicit bias/region length parser branch"
                ),
            },
            "transforms": transforms,
            "changed_segments": changed,
            "noop": not changed and info["archive_sha256"] == source_archive["sha256"],
            "byte_delta_vs_source_archive": byte_delta,
            "formula_only_rate_score_delta_vs_source": byte_delta * RATE_SCORE_PER_BYTE,
            "decoded_parity_metadata": {
                "status": "passed" if all_parity else "failed",
                "segments_checked": sorted(replacements),
                "semantic_hash_fields": [
                    "source_semantic_sha256",
                    "candidate_semantic_sha256",
                ],
            },
            "dispatch_gate": (
                "eligible_for_cuda_auth_eval_after_lane_claim"
                if dispatchable
                else "planning_only/no_remote_dispatch"
            ),
            "next_gate": (
                "Before exact eval, claim the lane with tools/claim_lane_dispatch.py and run PR85 public-runtime CUDA auth eval on this exact archive."
            ),
        }
        _write_json(candidate_dir / "manifest.json", manifest)
        rows.append(manifest)

    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "source_archive": source_archive,
        "runtime_dependency_closure": runtime_closure,
        "source_bundle": {
            "format": source_bundle["format"],
            "header_bytes": int(source_bundle["header_bytes"]),
            "segment_lengths": source_bundle["segment_lengths"],
            "fixed_length_segments": source_bundle.get("fixed_length_segments", {}),
        },
        "candidate_count": len(rows),
        "dispatchable_candidate_count": sum(
            1 for row in rows if row["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
        ),
        "candidates": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def _policy_choices() -> list[str]:
    return [
        "all_positive_best",
        "micro_positive_best",
        "all_best",
        "randmulti_best",
        *[f"segment_{name}_best" for name in SIDECHANNELS],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--replay-runtime-dir", type=Path, default=DEFAULT_REPLAY_RUNTIME_DIR)
    parser.add_argument("--policy", action="append", dest="policies", choices=_policy_choices())
    args = parser.parse_args(argv)

    payload = build_candidates(
        args.archive,
        args.out_dir,
        policy_ids=args.policies,
        replay_runtime_dir=args.replay_runtime_dir,
    )
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
