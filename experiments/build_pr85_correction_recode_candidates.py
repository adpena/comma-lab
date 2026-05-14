#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build strict PR85 correction-stream recode candidates.

This is a local-only byte-screening worker for Dalton's PR85 correction atom
waterfill plan.  It never mutates scorer inputs, never dispatches GPU work, and
only writes archive candidates whose correction streams decode to exactly the
same canonical semantics as the source PR85 archive while reducing stored ZIP
bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    PR85_HEADERLESS_RANDMULTI_SPECS,
    SEGMENT_ORDER,
    Pr85BundleError,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/build_pr85_correction_recode_candidates.py"
SCHEMA = "pr85_correction_recode_candidates_v1"
MANIFEST_SCHEMA = "pr85_correction_recode_candidate_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PLAN_JSON = Path("/tmp/pr85_correction_atom_waterfill_plan.json")
DEFAULT_REPLAY_RUNTIME_DIR = (
    REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/replay_submission"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_correction_recodes_20260504_codex"
PLAN_SCRIPT = REPO_ROOT / "experiments/plan_pr85_correction_atom_waterfill.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAIR_COUNT = 600
SOURCE_PR85_BYTES = 236_328
SOURCE_PR85_SHA256 = "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
SOURCE_PR85_SCORE = 0.25806611029397786  # [external: PR-85 contest-CUDA T4 frontier]
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
CORRECTION_STREAMS = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
REQUIRED_RUNTIME_FILES = ("inflate.sh", "inflate.py", "range_mask_codec.cpp")
REQUIRED_RUNTIME_TOKENS = (
    "PCD1",
    "SH4",
    "SD4",
    "FH1",
    "FV1",
    "FH2",
    "FH3",
    "FD3",
    "BH1",
    "BD1",
    "BV1",
    "RH1",
    "RD1",
    "RV1",
    "NM1",
    "NM2",
    "bad headerless post_codes length",
    "bad headerless f1 randmulti payload",
)
BROTLI_SWEEP = tuple(
    (quality, lgwin)
    for quality in (1, 5, 9, 11)
    for lgwin in (10, 16, 18, 20, 22, 24)
)


class CorrectionRecodeError(ValueError):
    """Raised when a PR85 correction recode would be unsafe or non-deterministic."""


@dataclass(frozen=True)
class Encoding:
    segment: str
    variant: str
    raw: bytes
    compressed: bytes
    brotli_params: Mapping[str, int | str] | None
    runtime_tokens: tuple[str, ...]
    support_reason: str


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _varint(value: int) -> bytes:
    if value < 0:
        raise CorrectionRecodeError("varint cannot encode negative values")
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
                raise CorrectionRecodeError("truncated varint stream")
            byte = raw[pos]
            pos += 1
            acc |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
                if shift > 63:
                    raise CorrectionRecodeError("overlong varint stream")
            else:
                values.append(acc)
                break
    return values, pos


def _brotli_decode(segment: bytes, name: str) -> bytes:
    try:
        return brotli.decompress(segment)
    except brotli.error as exc:
        raise CorrectionRecodeError(f"PR85 correction segment {name!r} is not Brotli-decodable") from exc


def _brotli_best(raw: bytes) -> tuple[bytes, dict[str, int | str]]:
    best = brotli.compress(raw, quality=11)
    best_params: dict[str, int | str] = {"quality": 11, "lgwin": "default"}
    for quality, lgwin in BROTLI_SWEEP:
        candidate = brotli.compress(raw, quality=quality, lgwin=lgwin)
        if len(candidate) < len(best) or (len(candidate) == len(best) and candidate < best):
            best = candidate
            best_params = {"quality": quality, "lgwin": lgwin}
    if brotli.decompress(best) != raw:
        raise CorrectionRecodeError("selected Brotli stream failed round-trip")
    return best, best_params


def _choice_payload_values(raw: bytes, *, default_center: int | None = None) -> bytes:
    magic = raw[:3]
    values = raw[3:]
    if len(values) != PAIR_COUNT:
        raise CorrectionRecodeError(f"choice payload {magic!r} has {len(values)} values")
    if magic in {b"SH4", b"FH1", b"FH2", b"FH3", b"BH1", b"RH1"}:
        return bytes(values)
    if magic in {b"SD4", b"FD3", b"BD1", b"RD1"}:
        if default_center is None:
            raise CorrectionRecodeError(f"default center required for {magic!r}")
        return bytes(default_center if value == 0 else value - 1 for value in values)
    raise CorrectionRecodeError(f"unsupported dense choice magic {magic!r}")


def _sparse_values(raw: bytes, *, magic: bytes, default_choice: int) -> bytes:
    if not raw.startswith(magic) or len(raw) < 5:
        raise CorrectionRecodeError(f"bad sparse payload magic for {magic!r}")
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    vals = raw[pos : pos + count]
    if len(vals) != count or pos + count != len(raw):
        raise CorrectionRecodeError("sparse payload length mismatch")
    out = bytearray([default_choice] * PAIR_COUNT)
    idx = -1
    for gap, value in zip(gaps, vals, strict=True):
        idx += gap + 1
        if not 0 <= idx < PAIR_COUNT:
            raise CorrectionRecodeError("sparse payload index out of range")
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
    if name == "frac2" and raw[:3] == b"FH2":
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
    raise CorrectionRecodeError(f"unsupported choice stream {name!r}")


def _encode_sparse_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    indices = [idx for idx, value in enumerate(values) if value != default_choice]
    if len(indices) > 0xFFFF:
        raise CorrectionRecodeError("too many sparse choices")
    gaps: list[int] = []
    vals = bytearray()
    last = -1
    for idx in indices:
        gaps.append(idx - last - 1)
        vals.append(values[idx] + 1)
        last = idx
    return magic + len(indices).to_bytes(2, "little") + b"".join(_varint(gap) for gap in gaps) + bytes(vals)


def _encode_delta_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    return magic + bytes(0 if value == default_choice else value + 1 for value in values)


def _decode_post_semantics(raw: bytes) -> bytes:
    if raw[:4] == b"PCD1":
        pos = 5
        if len(raw) < pos:
            raise CorrectionRecodeError("truncated PCD1 post header")
        out = bytearray()
        for _ in range(raw[4]):
            if pos + 3 > len(raw):
                raise CorrectionRecodeError("truncated PCD1 post stage")
            stage_id = raw[pos]
            count = int.from_bytes(raw[pos + 1 : pos + 3], "little")
            pos += 3
            choices = raw[pos : pos + count]
            pos += count
            if len(choices) != count:
                raise CorrectionRecodeError("truncated PCD1 post choices")
            out += bytes([stage_id]) + count.to_bytes(2, "little") + choices
        if pos != len(raw):
            raise CorrectionRecodeError("PCD1 post payload has trailing bytes")
        return bytes(out)
    if len(raw) % PAIR_COUNT:
        raise CorrectionRecodeError("bad headerless post length")
    stage_count = len(raw) // PAIR_COUNT
    if stage_count not in (3, 4):
        raise CorrectionRecodeError(f"unsupported headerless post stage count {stage_count}")
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
            raise CorrectionRecodeError("truncated post semantic record")
        stage_id = semantic[pos]
        count = int.from_bytes(semantic[pos + 1 : pos + 3], "little")
        pos += 3
        choices = semantic[pos : pos + count]
        pos += count
        if len(choices) != count:
            raise CorrectionRecodeError("truncated post semantic choices")
        stages.append((stage_id, choices))
    if variant == "headerless":
        if [stage for stage, _choices in stages] != list(range(1, len(stages) + 1)):
            raise CorrectionRecodeError("post stages are not headerless-compatible")
        if len(stages) not in (3, 4) or any(len(choices) != PAIR_COUNT for _stage, choices in stages):
            raise CorrectionRecodeError("post stages are not PR85 headerless-compatible")
        return b"".join(choices for _stage, choices in stages)
    if variant == "pcd1":
        if len(stages) > 255:
            raise CorrectionRecodeError("too many post stages")
        out = bytearray(b"PCD1" + bytes([len(stages)]))
        for stage_id, choices in stages:
            out.append(stage_id)
            out += len(choices).to_bytes(2, "little")
            out += choices
        return bytes(out)
    raise CorrectionRecodeError(f"unknown post variant {variant!r}")


def _decode_randmulti_groups(raw: bytes) -> list[tuple[int, int, int, int, list[bytes]]]:
    if raw[:3] == b"NM1" and len(raw) >= 4:
        scount = raw[3]
        payload = raw[4:]
        if len(payload) != scount * PAIR_COUNT:
            raise CorrectionRecodeError("NM1 randmulti length mismatch")
        rows = [payload[idx * PAIR_COUNT : (idx + 1) * PAIR_COUNT] for idx in range(scount)]
        return [(24, 32, 1, scount, [bytes(row) for row in rows])]
    if raw[:3] == b"NM2" and len(raw) >= 4:
        pos = 4
        groups = []
        for _ in range(raw[3]):
            if pos + 4 > len(raw):
                raise CorrectionRecodeError("truncated NM2 group")
            lh, lw, amp, scount = raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]
            pos += 4
            rows = []
            for _row in range(scount):
                row = raw[pos : pos + PAIR_COUNT]
                pos += PAIR_COUNT
                if len(row) != PAIR_COUNT:
                    raise CorrectionRecodeError("truncated NM2 row")
                rows.append(bytes(row))
            groups.append((lh, lw, amp, scount, rows))
        if pos != len(raw):
            raise CorrectionRecodeError("NM2 randmulti trailing bytes")
        return groups
    pos = 0
    groups = []
    for lh, lw, amp, scount in PR85_HEADERLESS_RANDMULTI_SPECS:
        rows = []
        for _ in range(scount):
            if pos >= len(raw):
                raise CorrectionRecodeError("truncated headerless randmulti count")
            count = raw[pos]
            pos += 1
            if count == 255:
                if pos + 2 > len(raw):
                    raise CorrectionRecodeError("truncated extended headerless randmulti count")
                count = int.from_bytes(raw[pos : pos + 2], "little")
                pos += 2
            gaps, pos = _read_varints(raw, pos, count)
            values = raw[pos : pos + count]
            pos += count
            if len(values) != count:
                raise CorrectionRecodeError("truncated headerless randmulti values")
            row = bytearray(PAIR_COUNT)
            idx = -1
            for gap, value in zip(gaps, values, strict=True):
                idx += gap + 1
                if not 0 <= idx < PAIR_COUNT:
                    raise CorrectionRecodeError("headerless randmulti index out of range")
                row[idx] = value
            rows.append(bytes(row))
        groups.append((lh, lw, amp, scount, rows))
    if pos != len(raw):
        raise CorrectionRecodeError("headerless randmulti trailing bytes")
    return groups


def _decode_randmulti_semantics(raw: bytes) -> bytes:
    out = bytearray()
    for group_index, (lh, lw, amp, scount, rows) in enumerate(_decode_randmulti_groups(raw)):
        out += group_index.to_bytes(2, "little")
        out += int(lh).to_bytes(4, "little")
        out += int(lw).to_bytes(4, "little")
        out += int(amp).to_bytes(2, "little")
        out += int(scount).to_bytes(2, "little")
        for row in rows:
            out += row
    return bytes(out)


def _semantic_randmulti_groups(semantic: bytes) -> list[tuple[int, int, int, int, list[bytes]]]:
    pos = 0
    groups = []
    group_index = 0
    while pos < len(semantic):
        if pos + 14 > len(semantic):
            raise CorrectionRecodeError("truncated randmulti semantic header")
        observed_index = int.from_bytes(semantic[pos : pos + 2], "little")
        pos += 2
        if observed_index != group_index:
            raise CorrectionRecodeError("randmulti semantic group index mismatch")
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
                raise CorrectionRecodeError("truncated randmulti semantic row")
            rows.append(bytes(row))
        groups.append((lh, lw, amp, scount, rows))
        group_index += 1
    return groups


def _encode_randmulti_headerless(semantic: bytes) -> bytes:
    groups = _semantic_randmulti_groups(semantic)
    expected = [(lh, lw, amp, scount) for lh, lw, amp, scount in PR85_HEADERLESS_RANDMULTI_SPECS]
    if [(lh, lw, amp, scount) for lh, lw, amp, scount, _rows in groups] != expected:
        raise CorrectionRecodeError("randmulti semantic groups do not match PR85 headerless schedule")
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


def _encode_randmulti_nm2(semantic: bytes) -> bytes:
    groups = _semantic_randmulti_groups(semantic)
    out = bytearray(b"NM2" + bytes([len(groups)]))
    for lh, lw, amp, scount, rows in groups:
        if not all(0 <= value <= 255 for value in (lh, lw, amp, scount)):
            raise CorrectionRecodeError("NM2 cannot encode this randmulti group header")
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
            raise CorrectionRecodeError(f"{name} decoded choice count {len(values)} != {PAIR_COUNT}")
        return values
    raise CorrectionRecodeError(f"unsupported correction segment {name!r}")


def _segment_semantics_from_segment(name: str, segment: bytes) -> bytes:
    return _segment_semantics(name, _brotli_decode(segment, name))


def _source_runtime_tokens(name: str, decoded_raw: bytes) -> tuple[str, ...]:
    if name == "post":
        return ("PCD1",) if decoded_raw[:4] == b"PCD1" else ("bad headerless post_codes length",)
    if name == "randmulti":
        if decoded_raw[:3] == b"NM1":
            return ("NM1",)
        if decoded_raw[:3] == b"NM2":
            return ("NM2",)
        return ("bad headerless f1 randmulti payload",)
    magic = decoded_raw[:3].decode("ascii", errors="replace")
    return (magic,)


def _segment_raw_variants(name: str, semantic: bytes, source_decoded: bytes) -> list[tuple[str, bytes, tuple[str, ...], str]]:
    if name == "post":
        variants = [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source post grammar")
        ]
        for variant, token in (("headerless", "bad headerless post_codes length"), ("pcd1", "PCD1")):
            try:
                variants.append(
                    (
                        variant,
                        _post_raw_from_semantics(semantic, variant=variant),
                        (token,),
                        f"runtime supports {variant} post grammar",
                    )
                )
            except CorrectionRecodeError:
                continue
        return variants
    if name == "shift":
        return [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source shift grammar"),
            ("SH4_absolute", b"SH4" + semantic, ("SH4",), "runtime supports SH4 absolute choices"),
            ("SD4_delta_center40", _encode_delta_choice(b"SD4", semantic, default_choice=40), ("SD4",), "runtime supports SD4 delta choices"),
        ]
    if name == "frac":
        return [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source frac grammar"),
            ("FH1_absolute", b"FH1" + semantic, ("FH1",), "runtime supports FH1 absolute choices"),
            ("FV1_sparse_default4", _encode_sparse_choice(b"FV1", semantic, default_choice=4), ("FV1",), "runtime supports FV1 sparse choices"),
        ]
    if name == "frac2":
        return [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source frac2 grammar"),
            ("FH2_absolute", b"FH2" + semantic, ("FH2",), "runtime supports FH2 absolute choices"),
        ]
    if name == "frac3":
        return [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source frac3 grammar"),
            ("FH3_absolute", b"FH3" + semantic, ("FH3",), "runtime supports FH3 absolute choices"),
            ("FD3_delta_center4", _encode_delta_choice(b"FD3", semantic, default_choice=4), ("FD3",), "runtime supports FD3 delta choices"),
        ]
    if name == "bias":
        return [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source bias grammar"),
            ("BH1_absolute", b"BH1" + semantic, ("BH1",), "runtime supports BH1 absolute choices"),
            ("BD1_delta_center13", _encode_delta_choice(b"BD1", semantic, default_choice=13), ("BD1",), "runtime supports BD1 delta choices"),
            ("BV1_sparse_default13", _encode_sparse_choice(b"BV1", semantic, default_choice=13), ("BV1",), "runtime supports BV1 sparse choices"),
        ]
    if name == "region":
        return [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source region grammar"),
            ("RH1_absolute", b"RH1" + semantic, ("RH1",), "runtime supports RH1 absolute choices"),
            ("RD1_delta_zero", _encode_delta_choice(b"RD1", semantic, default_choice=0), ("RD1",), "runtime supports RD1 delta choices"),
            ("RV1_sparse_default0", _encode_sparse_choice(b"RV1", semantic, default_choice=0), ("RV1",), "runtime supports RV1 sparse choices"),
        ]
    if name == "randmulti":
        variants = [
            ("raw_resweep", source_decoded, _source_runtime_tokens(name, source_decoded), "runtime decodes source randmulti grammar"),
            (
                "headerless_sparse_canonical",
                _encode_randmulti_headerless(semantic),
                ("bad headerless f1 randmulti payload",),
                "runtime supports PR85 headerless randmulti sparse tables",
            ),
        ]
        try:
            variants.append(
                (
                    "NM2_dense",
                    _encode_randmulti_nm2(semantic),
                    ("NM2",),
                    "runtime supports NM2 dense randmulti tables",
                )
            )
        except CorrectionRecodeError:
            pass
        return variants
    raise CorrectionRecodeError(f"unsupported segment {name!r}")


def _runtime_support_report(replay_runtime_dir: Path) -> dict[str, Any]:
    files = []
    missing_files = []
    for name in REQUIRED_RUNTIME_FILES:
        path = replay_runtime_dir / name
        if not path.is_file():
            missing_files.append(name)
            continue
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    inflate_py = replay_runtime_dir / "inflate.py"
    text = inflate_py.read_text(encoding="utf-8") if inflate_py.is_file() else ""
    missing_tokens = [token for token in REQUIRED_RUNTIME_TOKENS if token not in text]
    status = "passed" if not missing_files and not missing_tokens else "failed"
    return {
        "status": status,
        "runtime_dir": _rel(replay_runtime_dir),
        "required_files": files,
        "missing_files": missing_files,
        "required_decode_tokens": list(REQUIRED_RUNTIME_TOKENS),
        "missing_decode_tokens": missing_tokens,
        "support_check": "static public replay runtime grammar-token closure",
    }


def _ensure_runtime_support(report: Mapping[str, Any], tokens: Iterable[str], *, variant: str) -> None:
    if report.get("status") != "passed":
        raise CorrectionRecodeError("missing PR85 replay runtime support closure")
    missing = [token for token in tokens if token in set(report.get("missing_decode_tokens", []))]
    if missing:
        raise CorrectionRecodeError(f"runtime does not support {variant}: missing {missing}")


def _make_encodings(
    name: str,
    source_segment: bytes,
    *,
    runtime_report: Mapping[str, Any],
) -> list[Encoding]:
    source_decoded = _brotli_decode(source_segment, name)
    semantic = _segment_semantics(name, source_decoded)
    source_exact = Encoding(
        segment=name,
        variant="source_exact",
        raw=source_decoded,
        compressed=source_segment,
        brotli_params=None,
        runtime_tokens=_source_runtime_tokens(name, source_decoded),
        support_reason="source PR85 bytes copied unchanged",
    )
    _ensure_runtime_support(runtime_report, source_exact.runtime_tokens, variant=f"{name}:source_exact")
    encodings = [source_exact]
    seen = {("source_exact", _sha256(source_segment))}
    for variant, raw, tokens, reason in _segment_raw_variants(name, semantic, source_decoded):
        _ensure_runtime_support(runtime_report, tokens, variant=f"{name}:{variant}")
        compressed, params = _brotli_best(raw)
        _assert_segment_semantic_parity(name, source_segment, compressed, variant=variant)
        key = (variant, _sha256(compressed))
        if key in seen:
            continue
        seen.add(key)
        encodings.append(
            Encoding(
                segment=name,
                variant=variant,
                raw=raw,
                compressed=compressed,
                brotli_params=params,
                runtime_tokens=tokens,
                support_reason=reason,
            )
        )
    return encodings


def _best_encoding(name: str, source_segment: bytes, *, runtime_report: Mapping[str, Any]) -> Encoding:
    encodings = _make_encodings(name, source_segment, runtime_report=runtime_report)
    return min(
        encodings,
        key=lambda enc: (
            len(enc.compressed),
            enc.variant != "source_exact",
            enc.variant,
            enc.compressed,
        ),
    )


def _assert_segment_semantic_parity(
    name: str,
    source_segment: bytes,
    candidate_segment: bytes,
    *,
    variant: str,
) -> dict[str, Any]:
    source_semantic = _segment_semantics_from_segment(name, source_segment)
    candidate_semantic = _segment_semantics_from_segment(name, candidate_segment)
    if source_semantic != candidate_semantic:
        raise CorrectionRecodeError(f"{name} variant {variant} failed decoded-semantics parity")
    return {
        "segment": name,
        "variant": variant,
        "status": "passed",
        "source_semantic_sha256": _sha256(source_semantic),
        "candidate_semantic_sha256": _sha256(candidate_semantic),
        "source_decoded_semantic_bytes": len(source_semantic),
        "candidate_decoded_semantic_bytes": len(candidate_semantic),
    }


def _parity_records(source_segments: Mapping[str, bytes], candidate_segments: Mapping[str, bytes]) -> list[dict[str, Any]]:
    rows = []
    for name in CORRECTION_STREAMS:
        variant = "unchanged" if source_segments[name] == candidate_segments[name] else "candidate"
        row = _assert_segment_semantic_parity(name, source_segments[name], candidate_segments[name], variant=variant)
        row["changed"] = source_segments[name] != candidate_segments[name]
        row["source_segment_bytes"] = len(source_segments[name])
        row["candidate_segment_bytes"] = len(candidate_segments[name])
        row["segment_byte_delta"] = len(candidate_segments[name]) - len(source_segments[name])
        rows.append(row)
    return rows


def _read_pr85_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise CorrectionRecodeError(f"PR85 source archive is missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise CorrectionRecodeError(f"PR85 archive must contain exactly one member named 'x', got {names!r}")
        validate_pr85_member_name(infos[0].filename)
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise CorrectionRecodeError("PR85 archive member 'x' must be ZIP_STORED")
        raw = zf.read(infos[0])
    return (
        {
            "path": _rel(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "member_name": "x",
            "member_bytes": len(raw),
            "member_sha256": _sha256(raw),
            "zip_stored": True,
            "expected_pr85_frontier_match": (
                path.stat().st_size == SOURCE_PR85_BYTES and _sha256_file(path) == SOURCE_PR85_SHA256
            ),
        },
        raw,
    )


def _zip_info() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("x", FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _zip_bytes(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(_zip_info(), payload)
    return buffer.getvalue()


def _archive_info_from_bytes(path: Path, archive_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise CorrectionRecodeError(f"expected one member in generated archive, found {len(infos)}")
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
        if info.compress_type != zipfile.ZIP_STORED:
            raise CorrectionRecodeError("generated archive member is not ZIP_STORED")
    return {
        "archive_path": _rel(path),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": _sha256(archive_bytes),
        "member_name": "x",
        "member_bytes": len(raw),
        "member_sha256": _sha256(raw),
        "zip_stored": True,
    }


def _validate_candidate_bundle(payload: bytes, expected_segments: Mapping[str, bytes]) -> dict[str, Any]:
    try:
        bundle = parse_pr85_bundle(payload)
    except Pr85BundleError as exc:
        raise CorrectionRecodeError(f"candidate bundle parse failed: {exc}") from exc
    mismatches = [name for name in SEGMENT_ORDER if bytes(bundle.segments[name]) != expected_segments[name]]
    if mismatches:
        raise CorrectionRecodeError(f"candidate bundle segment mismatch: {mismatches}")
    return {
        "status": "passed",
        "bundle_format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "segment_lengths": bundle.segment_lengths,
        "fixed_length_segments": dict(bundle.fixed_length_segments),
        "mismatched_segments": [],
    }


def _header_mode_for_segments(
    *,
    source_header_bytes: int,
    source_segments: Mapping[str, bytes],
    candidate_segments: Mapping[str, bytes],
) -> str:
    if source_header_bytes == 30:
        return "explicit_30"
    fixed_changed = any(
        len(candidate_segments[name]) != len(source_segments[name])
        for name in ("bias", "region")
    )
    return "explicit_30" if fixed_changed else "v5"


def _candidate_payload(
    *,
    source_header_bytes: int,
    source_segments: Mapping[str, bytes],
    replacements: Mapping[str, Encoding],
) -> tuple[bytes, dict[str, bytes], str, dict[str, Any]]:
    candidate_segments = {name: bytes(source_segments[name]) for name in SEGMENT_ORDER}
    for name, encoding in replacements.items():
        candidate_segments[name] = encoding.compressed
    header_mode = _header_mode_for_segments(
        source_header_bytes=source_header_bytes,
        source_segments=source_segments,
        candidate_segments=candidate_segments,
    )
    try:
        payload = pack_pr85_bundle(candidate_segments, header_mode=header_mode)
    except Pr85BundleError as exc:
        raise CorrectionRecodeError(f"candidate pack failed: {exc}") from exc
    validation = _validate_candidate_bundle(payload, candidate_segments)
    return payload, candidate_segments, header_mode, validation


def _policy_id_for_subset(subset: Sequence[str], *, all_changed: Sequence[str]) -> str:
    if len(subset) == 1:
        return f"segment_{subset[0]}_best"
    if list(subset) == list(all_changed):
        return "decoded_parity_all_positive_best"
    return "combo_" + "_".join(subset) + "_best"


def _load_plan_module() -> Any:
    spec = importlib.util.spec_from_file_location("pr85_correction_atom_waterfill_plan_for_recode", PLAN_SCRIPT)
    if spec is None or spec.loader is None:
        raise CorrectionRecodeError(f"could not load planner from {_rel(PLAN_SCRIPT)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_or_build_plan(plan_json: Path | None, archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if plan_json is not None and plan_json.is_file():
        payload = _read_json(plan_json)
        return payload, {"source": _rel(plan_json), "rebuilt": False}
    module = _load_plan_module()
    payload = module.build_plan(archive=archive)
    if plan_json is not None:
        _write_json(plan_json, payload)
    return payload, {"source": _rel(plan_json) if plan_json is not None else "in_memory", "rebuilt": True}


def _planner_recode_policy(plan: Mapping[str, Any]) -> dict[str, Any]:
    policies = plan.get("candidate_policies", {}).get("policies", [])
    for policy in policies:
        if policy.get("candidate_policy_id") == "decoded_parity_recode_all_correction_streams":
            return dict(policy)
    raise CorrectionRecodeError("Dalton plan is missing decoded_parity_recode_all_correction_streams")


def _planner_selected_streams(plan: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    selected_ids = set(policy.get("selected_atom_ids", []))
    atoms = plan.get("atom_ledger", {}).get("atoms", [])
    streams = sorted(
        {
            atom.get("stream")
            for atom in atoms
            if atom.get("atom_id") in selected_ids and atom.get("stream") in CORRECTION_STREAMS
        }
    )
    return streams or list(CORRECTION_STREAMS)


def _validate_plan_source(plan: Mapping[str, Any], source_archive: Mapping[str, Any]) -> dict[str, Any]:
    plan_archive = plan.get("source_archive") or plan.get("atom_ledger", {}).get("archive")
    if not isinstance(plan_archive, dict):
        return {"status": "not_present", "source_archive_checked": False}
    matches = (
        plan_archive.get("sha256") == source_archive.get("sha256")
        and int(plan_archive.get("bytes", -1)) == int(source_archive.get("bytes", -2))
    )
    if not matches:
        raise CorrectionRecodeError("Dalton plan source archive does not match candidate source archive")
    return {
        "status": "passed",
        "source_archive_checked": True,
        "plan_archive_sha256": plan_archive.get("sha256"),
        "plan_archive_bytes": plan_archive.get("bytes"),
    }


def _screen_record(name: str, source_segment: bytes, encoding: Encoding) -> dict[str, Any]:
    parity = _assert_segment_semantic_parity(name, source_segment, encoding.compressed, variant=encoding.variant)
    return {
        **parity,
        "best_variant": encoding.variant,
        "source_segment_bytes": len(source_segment),
        "candidate_segment_bytes": len(encoding.compressed),
        "segment_byte_delta": len(encoding.compressed) - len(source_segment),
        "source_segment_sha256": _sha256(source_segment),
        "candidate_segment_sha256": _sha256(encoding.compressed),
        "changed": source_segment != encoding.compressed,
        "brotli_params": encoding.brotli_params,
        "runtime_tokens": list(encoding.runtime_tokens),
        "support_reason": encoding.support_reason,
    }


def build_candidates(
    archive: Path = DEFAULT_ARCHIVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    plan_json: Path | None = DEFAULT_PLAN_JSON,
    replay_runtime_dir: Path = DEFAULT_REPLAY_RUNTIME_DIR,
    max_archive_candidates: int = 16,
    require_known_source: bool = True,
) -> dict[str, Any]:
    source_archive, raw = _read_pr85_archive(archive)
    if require_known_source and source_archive["sha256"] != SOURCE_PR85_SHA256:
        raise CorrectionRecodeError(
            f"source archive SHA mismatch: got {source_archive['sha256']}, expected {SOURCE_PR85_SHA256}"
        )
    try:
        bundle = parse_pr85_bundle(raw)
    except Pr85BundleError as exc:
        raise CorrectionRecodeError(f"failed to parse PR85 bundle: {exc}") from exc
    source_segments = {name: bytes(bundle.segments[name]) for name in SEGMENT_ORDER}
    plan, plan_meta = _load_or_build_plan(plan_json, archive)
    policy = _planner_recode_policy(plan)
    selected_streams = _planner_selected_streams(plan, policy)
    unsupported_streams = [stream for stream in selected_streams if stream not in CORRECTION_STREAMS]
    if unsupported_streams:
        raise CorrectionRecodeError(f"planner selected unsupported correction streams: {unsupported_streams}")
    plan_source_validation = _validate_plan_source(plan, source_archive)
    runtime_report = _runtime_support_report(replay_runtime_dir)
    if runtime_report["status"] != "passed":
        raise CorrectionRecodeError("PR85 replay runtime support check failed")

    best_by_stream = {
        name: _best_encoding(name, source_segments[name], runtime_report=runtime_report)
        for name in selected_streams
    }
    screen = [_screen_record(name, source_segments[name], best_by_stream[name]) for name in selected_streams]
    changed_streams = [name for name in selected_streams if best_by_stream[name].compressed != source_segments[name]]

    subset_screens: list[dict[str, Any]] = []
    for size in range(1, len(changed_streams) + 1):
        for subset in combinations(changed_streams, size):
            replacements = {name: best_by_stream[name] for name in subset}
            payload, candidate_segments, header_mode, validation = _candidate_payload(
                source_header_bytes=bundle.header_bytes,
                source_segments=source_segments,
                replacements=replacements,
            )
            archive_bytes_a = _zip_bytes(payload)
            archive_bytes_b = _zip_bytes(payload)
            if archive_bytes_a != archive_bytes_b:
                raise CorrectionRecodeError(f"candidate {subset!r} archive construction is non-deterministic")
            byte_delta = len(archive_bytes_a) - int(source_archive["bytes"])
            subset_screens.append(
                {
                    "policy_id": _policy_id_for_subset(list(subset), all_changed=changed_streams),
                    "changed_segments": list(subset),
                    "header_mode": header_mode,
                    "byte_delta_vs_source_archive": byte_delta,
                    "formula_only_rate_score_delta_vs_source": byte_delta * RATE_SCORE_PER_BYTE,
                    "candidate_payload_bytes": len(payload),
                    "candidate_payload_sha256": _sha256(payload),
                    "candidate_archive_bytes": len(archive_bytes_a),
                    "candidate_archive_sha256": _sha256(archive_bytes_a),
                    "candidate_bundle_validation": validation,
                    "archive_byte_win": byte_delta < 0,
                    "archive_bytes_for_write": archive_bytes_a,
                    "candidate_segments": candidate_segments,
                }
            )

    byte_wins = sorted(
        [row for row in subset_screens if row["archive_byte_win"]],
        key=lambda row: (row["byte_delta_vs_source_archive"], len(row["changed_segments"]), row["policy_id"]),
    )
    archived_rows = []
    for rank, row in enumerate(byte_wins[:max_archive_candidates], start=1):
        candidate_dir = out_dir / f"rank{rank:03d}_{row['policy_id']}"
        archive_path = candidate_dir / "archive.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(row["archive_bytes_for_write"])
        info = _archive_info_from_bytes(archive_path, row["archive_bytes_for_write"])
        parity_rows = _parity_records(source_segments, row["candidate_segments"])
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool": TOOL,
            "policy_id": row["policy_id"],
            "rank": rank,
            "score_claim": False,
            "dispatch_performed": False,
            "evidence_grade": "empirical_lossless_decoded_semantics_recode",
            "source_frontier": {
                "archive_bytes": SOURCE_PR85_BYTES,
                "archive_sha256": SOURCE_PR85_SHA256,
                "score": SOURCE_PR85_SCORE,
                "score_source": "PR85 exact T4 replay baseline; comparison only",
            },
            "source_archive": source_archive,
            "source_bundle": {
                "format": bundle.format,
                "header_bytes": bundle.header_bytes,
                "segment_lengths": bundle.segment_lengths,
                "fixed_length_segments": dict(bundle.fixed_length_segments),
            },
            "planner": {
                **plan_meta,
                "policy": policy,
                "selected_streams": selected_streams,
                "source_validation": plan_source_validation,
            },
            "runtime_support": runtime_report,
            "candidate": info,
            "header_mode": row["header_mode"],
            "changed_segments": row["changed_segments"],
            "transforms": [
                screen_row for screen_row in screen if screen_row["segment"] in set(row["changed_segments"])
            ],
            "decoded_parity_metadata": {
                "status": "passed",
                "segments_checked": list(CORRECTION_STREAMS),
                "rows": parity_rows,
            },
            "candidate_bundle_validation": row["candidate_bundle_validation"],
            "byte_delta_vs_source_archive": row["byte_delta_vs_source_archive"],
            "formula_only_rate_score_delta_vs_source": row["formula_only_rate_score_delta_vs_source"],
            "dispatch_gate": "eligible_for_cuda_auth_eval_after_lane_claim",
            "exact_eval_unlocked": True,
            "next_gate": (
                "Claim a dispatch lane with tools/claim_lane_dispatch.py before any CUDA auth eval; "
                "this manifest is not a score claim."
            ),
        }
        _write_json(candidate_dir / "manifest.json", manifest)
        archived_rows.append(manifest)

    sanitized_subset_screens = [
        {key: value for key, value in row.items() if key not in {"archive_bytes_for_write", "candidate_segments"}}
        for row in sorted(
            subset_screens,
            key=lambda row: (row["byte_delta_vs_source_archive"], len(row["changed_segments"]), row["policy_id"]),
        )
    ]
    best_delta = sanitized_subset_screens[0]["byte_delta_vs_source_archive"] if sanitized_subset_screens else 0
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "source_archive": source_archive,
        "source_bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
            "fixed_length_segments": dict(bundle.fixed_length_segments),
        },
        "planner": {
            **plan_meta,
            "policy": policy,
            "selected_streams": selected_streams,
            "source_validation": plan_source_validation,
        },
        "runtime_support": runtime_report,
        "segment_screen": screen,
        "subset_screen": sanitized_subset_screens,
        "archive_candidate_count": len(archived_rows),
        "byte_winning_subset_count": len(byte_wins),
        "not_archived_byte_winning_subset_count": max(0, len(byte_wins) - len(archived_rows)),
        "best_byte_delta_vs_source_archive": best_delta,
        "best_candidate": (
            {
                "policy_id": archived_rows[0]["policy_id"],
                "archive_path": archived_rows[0]["candidate"]["archive_path"],
                "archive_bytes": archived_rows[0]["candidate"]["archive_bytes"],
                "archive_sha256": archived_rows[0]["candidate"]["archive_sha256"],
                "byte_delta_vs_source_archive": archived_rows[0]["byte_delta_vs_source_archive"],
            }
            if archived_rows
            else None
        ),
        "candidates": archived_rows,
        "result_class": (
            "candidate_byte_win"
            if archived_rows
            else "exact_local_negative_no_byte_winning_recode"
        ),
        "dispatch_gate": (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if archived_rows
            else "planning_only/no_remote_dispatch"
        ),
        "exact_eval_unlocked": bool(archived_rows),
        "negative_reason": (
            None
            if archived_rows
            else "all decoded-parity-preserving runtime-supported recodes were byte-neutral or byte-negative at archive level"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--replay-runtime-dir", type=Path, default=DEFAULT_REPLAY_RUNTIME_DIR)
    parser.add_argument("--max-archive-candidates", type=int, default=16)
    parser.add_argument(
        "--allow-source-sha-mismatch",
        action="store_true",
        help="Permit non-frontier fixture archives; exact PR85 runs should leave this unset.",
    )
    args = parser.parse_args(argv)
    if args.max_archive_candidates < 1:
        raise CorrectionRecodeError("--max-archive-candidates must be positive")
    payload = build_candidates(
        archive=args.archive,
        out_dir=args.out_dir,
        plan_json=args.plan_json,
        replay_runtime_dir=args.replay_runtime_dir,
        max_archive_candidates=args.max_archive_candidates,
        require_known_source=not args.allow_source_sha_mismatch,
    )
    print(_json_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
