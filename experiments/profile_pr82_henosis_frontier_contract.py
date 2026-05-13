#!/usr/bin/env python3
"""Static PR82/Henosis Frontier archive anatomy profiler.

This is a planning-only profiler. It reads the public PR82 intake archive and
replay inflate source, but it does not import the replay runtime, load scorers,
inflate frames, or dispatch GPU work.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE_DIR = (
    REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex"
)
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_REPLAY_INFLATE = DEFAULT_INTAKE_DIR / "replay_submission/inflate.py"
DEFAULT_OUTPUT_JSON = DEFAULT_INTAKE_DIR / "pr82_henosis_frontier_static_profile.json"

SCHEMA = "pr82_henosis_frontier_static_profile_v1"
TOOL = "experiments/profile_pr82_henosis_frontier_contract.py"
EVIDENCE_GRADE = "empirical_static_archive_profile"
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_REFERENCE_LABEL = "PR79/S2 A++ T4 frontier"
DEFAULT_REFERENCE_ARCHIVE_BYTES = 277_321
DEFAULT_REFERENCE_ARCHIVE_SHA256 = (
    "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
)
DEFAULT_REFERENCE_SCORE = 0.31453355357318635
CUDA_GATE = (
    "Planning only: do not promote, rank, or transfer until exact CUDA auth eval "
    "on identical archive bytes completes through archive.zip -> inflate.sh -> "
    "upstream/evaluate.py, with dispatch lane claim handled outside this profiler."
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _brotli_decompress(data: bytes) -> tuple[bytes | None, str | None]:
    try:
        import brotli
    except ImportError:
        return None, "brotli module unavailable"
    try:
        return brotli.decompress(data), None
    except Exception as exc:  # pragma: no cover - exact message is codec-specific
        return None, f"{type(exc).__name__}: {exc}"


def _magic_ascii(data: bytes, n: int = 8) -> str:
    return "".join(chr(value) if 32 <= value < 127 else "." for value in data[:n])


def _read_single_member_archive(archive: Path) -> tuple[dict[str, Any], bytes]:
    archive_bytes = archive.read_bytes()
    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        duplicate_names = sorted(
            name
            for name in {info.filename for info in infos}
            if sum(row.filename == name for row in infos) > 1
        )
        unsafe_names = [
            info.filename
            for info in infos
            if info.filename.startswith("/") or ".." in Path(info.filename).parts
        ]
        if duplicate_names:
            raise ValueError(f"duplicate ZIP members are not strict custody: {duplicate_names}")
        if unsafe_names:
            raise ValueError(f"unsafe ZIP member names are not strict custody: {unsafe_names}")
        if len(infos) != 1:
            raise ValueError(f"expected one PR82 ZIP member, found {len(infos)}")
        info = infos[0]
        payload = zf.read(info.filename)
        member = {
            "compress_size": int(info.compress_size),
            "compress_type": int(info.compress_type),
            "crc": int(info.CRC),
            "date_time": list(info.date_time),
            "external_attr": int(info.external_attr),
            "file_size": int(info.file_size),
            "filename": info.filename,
            "payload_sha256": _sha256_bytes(payload),
            "stored": info.compress_type == zipfile.ZIP_STORED,
        }
    zip_container = {
        "archive_bytes": len(archive_bytes),
        "archive_path": _repo_rel(archive),
        "archive_sha256": _sha256_bytes(archive_bytes),
        "member_count": 1,
        "members": [member],
        "strict_zip_overhead_bytes": int(len(archive_bytes) - len(payload)),
    }
    return zip_container, payload


def _extract_replay_contract(replay_inflate: Path) -> dict[str, Any]:
    text = replay_inflate.read_text(encoding="utf-8")
    tree = ast.parse(text)
    constants: dict[str, Any] = {
        "fixed_bias_bytes": None,
        "fixed_region_bytes": None,
        "randmulti_headerless_specs": [],
    }
    # Catalog #168 fix 2026-05-12: handle both bare Assign and AnnAssign
    # forms so future PR vendor edits with annotated constants aren't missed.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif (isinstance(node, ast.AnnAssign)
              and node.value is not None
              and isinstance(node.target, ast.Name)):
            target = node.target
            value_node = node.value
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id in {"l_bias", "l_region"}
            and isinstance(value_node, ast.Constant)
            and isinstance(value_node.value, int)
        ):
            key = "fixed_bias_bytes" if target.id == "l_bias" else "fixed_region_bytes"
            constants[key] = int(value_node.value)
        if isinstance(target, ast.Name) and target.id == "specs_n":
            try:
                constants["randmulti_headerless_specs"] = [
                    [int(v) for v in row] for row in ast.literal_eval(value_node)
                ]
            except (ValueError, SyntaxError):
                constants["randmulti_headerless_specs"] = []
    constants["source_path"] = _repo_rel(replay_inflate)
    constants["source_sha256"] = _sha256_file(replay_inflate)
    return constants


def _u24(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "little")


def _segment_record(name: str, encoded: bytes, offset: int) -> dict[str, Any]:
    decoded, error = _brotli_decompress(encoded)
    record: dict[str, Any] = {
        "encoded_bytes": len(encoded),
        "encoded_offset": int(offset),
        "encoded_sha256": _sha256_bytes(encoded),
        "encoded_magic8_ascii": _magic_ascii(encoded),
        "encoded_magic8_hex": encoded[:8].hex(),
        "name": name,
    }
    if decoded is None:
        record.update(
            {
                "brotli_decodable": False,
                "decode_error": error,
                "decoded_bytes": None,
                "decoded_magic8_ascii": None,
                "decoded_magic8_hex": None,
                "decoded_sha256": None,
            }
        )
    else:
        record.update(
            {
                "brotli_decodable": True,
                "decode_error": None,
                "decoded_bytes": len(decoded),
                "decoded_magic8_ascii": _magic_ascii(decoded),
                "decoded_magic8_hex": decoded[:8].hex(),
                "decoded_sha256": _sha256_bytes(decoded),
            }
        )
    return record


def _parse_compact_bundle(raw: bytes, replay_contract: dict[str, Any]) -> dict[str, Any]:
    if len(raw) < 24:
        raise ValueError("PR82 compact bundle is too short for the v5 micro header")
    fixed_bias = replay_contract.get("fixed_bias_bytes")
    fixed_region = replay_contract.get("fixed_region_bytes")
    if not isinstance(fixed_bias, int) or not isinstance(fixed_region, int):
        fixed_bias = 223
        fixed_region = 273
    names = ["mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3"]
    lengths = {name: _u24(raw, index * 3) for index, name in enumerate(names)}
    pos = 24
    segments: list[dict[str, Any]] = []
    for name in names:
        nbytes = lengths[name]
        encoded = raw[pos : pos + nbytes]
        if len(encoded) != nbytes:
            raise ValueError(f"truncated PR82 compact segment {name}")
        segments.append(_segment_record(name, encoded, pos))
        pos += nbytes
    for name, nbytes in (("bias", fixed_bias), ("region", fixed_region)):
        encoded = raw[pos : pos + nbytes]
        if len(encoded) != nbytes:
            raise ValueError(f"truncated fixed PR82 compact segment {name}")
        segments.append(_segment_record(name, encoded, pos))
        pos += nbytes
    randmulti = raw[pos:]
    if not randmulti:
        raise ValueError("missing PR82 randmulti tail")
    segments.append(_segment_record("randmulti", randmulti, pos))
    total_segment_bytes = sum(int(row["encoded_bytes"]) for row in segments)
    if total_segment_bytes + 24 != len(raw):
        raise ValueError("PR82 segment accounting does not close")
    return {
        "format": "public_pr82_henosis_compact_v5_micro_header",
        "header_bytes": 24,
        "header_lengths_u24": lengths,
        "fixed_tail_lengths_from_replay": {
            "bias": int(fixed_bias),
            "region": int(fixed_region),
        },
        "payload_bytes": len(raw),
        "segment_total_bytes": int(total_segment_bytes),
        "segments": segments,
    }


def _decoded_segment(segment_bytes: dict[str, bytes], name: str) -> bytes | None:
    encoded = segment_bytes.get(name)
    if encoded is None:
        return None
    # The decode is intentionally repeated instead of storing raw bytes in the
    # profile object; this keeps emitted JSON byte-free and deterministic.
    return _brotli_decompress(encoded)[0]


def _parse_pose(decoded: bytes | None) -> dict[str, Any]:
    if decoded is None:
        return {"available": False}
    if decoded[:4] == b"P1D1":
        if len(decoded) < 5:
            return {"available": True, "format": "P1D1", "valid": False}
        count = decoded[4]
        pos = 5
        streams: list[dict[str, Any]] = []
        for _ in range(count):
            if pos + 3 > len(decoded):
                return {"available": True, "format": "P1D1", "valid": False}
            dim = int(decoded[pos])
            nbytes = int.from_bytes(decoded[pos + 1 : pos + 3], "little")
            pos += 3
            streams.append({"dimension": dim, "encoded_delta_stream_bytes": nbytes})
        payload_start = pos
        for stream in streams:
            pos += int(stream["encoded_delta_stream_bytes"])
        for stream in streams:
            stream["samples_targeted"] = 600
        return {
            "available": True,
            "format": "P1D1_delta_varint",
            "valid": pos == len(decoded),
            "decoded_bytes": len(decoded),
            "dimension_count": int(count),
            "dimensions": [row["dimension"] for row in streams],
            "header_bytes": int(payload_start),
            "stream_payload_bytes": int(len(decoded) - payload_start),
            "streams": streams,
        }
    if decoded[:4] in (b"PQ12", b"PQB1"):
        return {"available": True, "format": decoded[:4].decode("ascii"), "decoded_bytes": len(decoded)}
    return {
        "available": True,
        "format": "unknown_or_numpy_payload",
        "decoded_bytes": len(decoded),
        "magic8_hex": decoded[:8].hex(),
    }


def _post_def_count(stage_id: int) -> int | None:
    if stage_id == 1:
        return 12
    if stage_id == 2:
        return 65
    if stage_id == 3:
        return 125
    if stage_id == 4:
        return 27
    return None


def _parse_postprocess(decoded: bytes | None) -> dict[str, Any]:
    if decoded is None:
        return {"available": False}
    if decoded[:4] == b"PCD1":
        return {"available": True, "format": "PCD1_self_describing", "decoded_bytes": len(decoded)}
    valid_headerless = len(decoded) % 600 == 0 and len(decoded) // 600 in (3, 4)
    stages: list[dict[str, Any]] = []
    if valid_headerless:
        for index in range(len(decoded) // 600):
            choices = decoded[index * 600 : (index + 1) * 600]
            counts = Counter(choices)
            stage_id = index + 1
            defs = _post_def_count(stage_id)
            stages.append(
                {
                    "choice_count": 600,
                    "defined_variant_count": defs,
                    "max_choice": max(choices) if choices else None,
                    "nonzero_choices": int(sum(1 for value in choices if value != 0)),
                    "stage_id": stage_id,
                    "unique_choice_count": len(counts),
                    "valid_choice_range": None if defs is None else max(choices) < defs,
                }
            )
    return {
        "available": True,
        "decoded_bytes": len(decoded),
        "format": "headerless_fixed_public_test" if valid_headerless else "unknown",
        "stage_count": len(stages),
        "stages": stages,
        "valid": bool(valid_headerless and all(row["valid_choice_range"] for row in stages)),
    }


def _read_varints(data: bytes, pos: int, count: int) -> tuple[list[int], int, bool]:
    values: list[int] = []
    for _ in range(count):
        acc = 0
        shift = 0
        while True:
            if pos >= len(data):
                return values, pos, False
            by = data[pos]
            pos += 1
            acc |= (by & 127) << shift
            if by & 128:
                shift += 7
            else:
                break
        values.append(acc)
    return values, pos, True


def _parse_sparse_choice_payload(decoded: bytes, *, default_length: int, center: int) -> dict[str, Any]:
    count = int.from_bytes(decoded[3:5], "little") if len(decoded) >= 5 else 0
    pos = 5
    deltas, pos, ok = _read_varints(decoded, pos, count)
    values_len = max(0, len(decoded) - pos)
    return {
        "center_choice": center,
        "default_length": default_length,
        "format": decoded[:3].decode("ascii", errors="replace"),
        "sparse_count": int(count),
        "valid": bool(ok and values_len == count),
        "varint_index_bytes": int(pos - 5),
        "value_bytes": int(values_len),
    }


def _parse_control_field(name: str, decoded: bytes | None) -> dict[str, Any]:
    if decoded is None:
        return {"available": False, "name": name}
    magic = decoded[:3]
    record: dict[str, Any] = {
        "available": True,
        "decoded_bytes": len(decoded),
        "magic": magic.decode("ascii", errors="replace"),
        "name": name,
    }
    if magic in {b"SH4", b"SD4", b"FH2", b"FH3", b"FD3", b"BH1", b"BD1", b"RH1", b"RD1"}:
        choices = decoded[3:]
        record.update(
            {
                "choice_count": len(choices),
                "format": "dense_or_delta_choice_array",
                "nonzero_bytes": int(sum(1 for value in choices if value != 0)),
                "unique_encoded_byte_count": len(set(choices)),
                "valid_length_600": len(choices) == 600,
            }
        )
    elif magic == b"FV1":
        record.update(_parse_sparse_choice_payload(decoded, default_length=600, center=4))
    elif magic == b"BV1":
        record.update(_parse_sparse_choice_payload(decoded, default_length=600, center=13))
    elif magic == b"RV1":
        record.update(_parse_sparse_choice_payload(decoded, default_length=600, center=0))
    else:
        record.update({"format": "unknown", "valid_length_600": None})
    return record


def _parse_randmulti(decoded: bytes | None, replay_contract: dict[str, Any]) -> dict[str, Any]:
    if decoded is None:
        return {"available": False}
    if decoded[:3] in {b"NM1", b"NM2"}:
        return {
            "available": True,
            "decoded_bytes": len(decoded),
            "format": decoded[:3].decode("ascii"),
            "headerless": False,
        }
    specs = replay_contract.get("randmulti_headerless_specs") or []
    pos = 0
    groups: list[dict[str, Any]] = []
    valid = True
    for index, raw_spec in enumerate(specs):
        if len(raw_spec) != 4:
            valid = False
            break
        lh, lw, amp, scount = [int(value) for value in raw_spec]
        sparse_counts: list[int] = []
        group_start = pos
        for _ in range(scount):
            if pos >= len(decoded):
                valid = False
                break
            count = int(decoded[pos])
            pos += 1
            if count == 255:
                if pos + 2 > len(decoded):
                    valid = False
                    break
                count = int.from_bytes(decoded[pos : pos + 2], "little")
                pos += 2
            _, pos, ok = _read_varints(decoded, pos, count)
            if not ok or pos + count > len(decoded):
                valid = False
                break
            pos += count
            sparse_counts.append(count)
        if not valid:
            break
        groups.append(
            {
                "amplitude": amp,
                "group_index": index,
                "height": lh,
                "nonzero_choice_total": int(sum(sparse_counts)),
                "payload_bytes": int(pos - group_start),
                "scount": scount,
                "width": lw,
            }
        )
    return {
        "available": True,
        "decoded_bytes": len(decoded),
        "format": "headerless_sparse_randmulti",
        "group_count": len(groups),
        "groups": groups,
        "headerless": True,
        "parsed_bytes": int(pos),
        "valid": bool(valid and pos == len(decoded)),
    }


def _joint_generator_qconv_modules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_embedding(name: str, num_embeddings: int, emb_dim: int) -> None:
        rows.append(
            {
                "bias_numel": 0,
                "kind": "QEmbedding",
                "name": name,
                "numel": int(num_embeddings * emb_dim),
                "shape": [num_embeddings, emb_dim],
            }
        )

    def add_conv(name: str, in_ch: int, out_ch: int, k: int, *, groups: int = 1, bias: bool = True) -> None:
        rows.append(
            {
                "bias_numel": int(out_ch if bias else 0),
                "kind": "QConv2d",
                "name": name,
                "numel": int(out_ch * (in_ch // groups) * k * k),
                "shape": [out_ch, in_ch // groups, k, k],
            }
        )

    def sep_conv(prefix: str, in_ch: int, out_ch: int, *, k: int = 3, depth_mult: int = 1, norm: bool = False) -> None:
        mid = in_ch * depth_mult
        add_conv(f"{prefix}.dw", in_ch, mid, k, groups=in_ch, bias=False)
        add_conv(f"{prefix}.pw", mid, out_ch, 1, bias=True)

    def sep_res(prefix: str, ch: int) -> None:
        sep_conv(f"{prefix}.conv1", ch, ch)
        sep_conv(f"{prefix}.conv2", ch, ch)

    def film_sep_res(prefix: str, ch: int) -> None:
        sep_conv(f"{prefix}.conv1", ch, ch)
        sep_conv(f"{prefix}.conv2", ch, ch)

    add_embedding("shared_trunk.embedding", 5, 6)
    sep_conv("shared_trunk.stem_conv", 8, 56)
    sep_res("shared_trunk.stem_block", 56)
    sep_conv("shared_trunk.down_conv", 56, 64)
    sep_res("shared_trunk.down_block", 64)
    sep_conv("shared_trunk.up.1", 64, 56)
    sep_conv("shared_trunk.fuse", 112, 56)
    sep_res("shared_trunk.fuse_block", 56)
    film_sep_res("frame1_head.block1", 56)
    sep_res("frame1_head.block2", 56)
    sep_conv("frame1_head.pre", 56, 52)
    add_conv("frame1_head.head", 52, 3, 1, bias=True)
    sep_res("frame2_head.block1", 56)
    sep_res("frame2_head.block2", 56)
    sep_conv("frame2_head.pre", 56, 52)
    add_conv("frame2_head.head", 52, 3, 1, bias=True)
    return rows


def _joint_generator_dense_state() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def gn(prefix: str, ch: int) -> None:
        rows.append({"name": f"{prefix}.weight", "numel": ch, "rows": ch, "shape": [ch]})
        rows.append({"name": f"{prefix}.bias", "numel": ch, "rows": ch, "shape": [ch]})

    def linear(prefix: str, out_ch: int, in_ch: int) -> None:
        rows.append({"name": f"{prefix}.weight", "numel": out_ch * in_ch, "rows": out_ch, "shape": [out_ch, in_ch]})
        rows.append({"name": f"{prefix}.bias", "numel": out_ch, "rows": out_ch, "shape": [out_ch]})

    def sep_gn(prefix: str, out_ch: int) -> None:
        gn(f"{prefix}.norm", out_ch)

    def sep_res_gn(prefix: str, ch: int) -> None:
        sep_gn(f"{prefix}.conv1", ch)
        gn(f"{prefix}.norm2", ch)

    sep_gn("shared_trunk.stem_conv", 56)
    sep_res_gn("shared_trunk.stem_block", 56)
    sep_gn("shared_trunk.down_conv", 64)
    sep_res_gn("shared_trunk.down_block", 64)
    sep_gn("shared_trunk.up.1", 56)
    sep_gn("shared_trunk.fuse", 56)
    sep_res_gn("shared_trunk.fuse_block", 56)
    linear("pose_mlp.0", 48, 6)
    linear("pose_mlp.2", 48, 48)
    sep_gn("frame1_head.block1.conv1", 56)
    gn("frame1_head.block1.norm2", 56)
    linear("frame1_head.block1.film_proj", 112, 48)
    sep_res_gn("frame1_head.block2", 56)
    sep_gn("frame1_head.pre", 52)
    sep_res_gn("frame2_head.block1", 56)
    sep_res_gn("frame2_head.block2", 56)
    sep_gn("frame2_head.pre", 52)
    return rows


def _parse_qh0_model(decoded: bytes | None) -> dict[str, Any]:
    if decoded is None:
        return {"available": False}
    if decoded[:3] not in (b"QH0", b"QM0"):
        return {"available": True, "format": "non_custom_torch_payload", "decoded_bytes": len(decoded)}
    hilosplit = decoded[:3] == b"QH0"
    pos = 3
    conv_rows: list[dict[str, Any]] = []
    totals = Counter()
    valid = True
    for spec in _joint_generator_qconv_modules():
        if pos >= len(decoded):
            valid = False
            break
        kind = int(decoded[pos])
        pos += 1
        numel = int(spec["numel"])
        if kind == 1:
            blocks = math.ceil(numel / 32)
            packed_len = (blocks * 32 + 1) // 2
            weight_bytes = packed_len + blocks * 2
            quant = "fp4_hilosplit" if hilosplit else "fp4"
        elif kind == 0:
            blocks = None
            weight_bytes = numel * 2
            quant = "fp16_hilosplit" if hilosplit else "fp16"
        else:
            valid = False
            break
        start = pos - 1
        pos += weight_bytes
        bias_bytes = int(spec["bias_numel"]) * 2
        pos += bias_bytes
        if pos > len(decoded):
            valid = False
            break
        totals[quant] += weight_bytes
        totals["bias_fp16_bytes"] += bias_bytes
        conv_rows.append(
            {
                "bias_bytes": bias_bytes,
                "kind_byte": kind,
                "name": spec["name"],
                "quantization": quant,
                "record_bytes": int(pos - start),
                "shape": spec["shape"],
                "weight_bytes": int(weight_bytes),
            }
        )
    dense_rows: list[dict[str, Any]] = []
    if valid:
        for spec in _joint_generator_dense_state():
            if pos >= len(decoded):
                valid = False
                break
            kind = int(decoded[pos])
            pos += 1
            numel = int(spec["numel"])
            rows = int(spec["rows"])
            if kind == 2:
                payload_bytes = numel + rows * 2
                quant = "int8_row_scale_hilosplit" if hilosplit else "int8_row_scale"
            elif kind == 0:
                payload_bytes = numel * 2
                quant = "fp16_hilosplit" if hilosplit else "fp16"
            else:
                valid = False
                break
            start = pos - 1
            pos += payload_bytes
            if pos > len(decoded):
                valid = False
                break
            totals[quant] += payload_bytes
            dense_rows.append(
                {
                    "kind_byte": kind,
                    "name": spec["name"],
                    "quantization": quant,
                    "record_bytes": int(pos - start),
                    "shape": spec["shape"],
                    "tensor_bytes": int(payload_bytes),
                }
            )
    return {
        "available": True,
        "covered_qconv_or_qembedding_count": len(conv_rows),
        "decoded_bytes": len(decoded),
        "dense_state_count": len(dense_rows),
        "format": decoded[:3].decode("ascii"),
        "hilo_split": bool(hilosplit),
        "parse_valid": bool(valid and pos == len(decoded)),
        "parsed_bytes": int(pos),
        "qconv_quantization_counts": dict(Counter(row["quantization"] for row in conv_rows)),
        "qconv_total_record_bytes": int(sum(row["record_bytes"] for row in conv_rows)),
        "dense_quantization_counts": dict(Counter(row["quantization"] for row in dense_rows)),
        "dense_total_record_bytes": int(sum(row["record_bytes"] for row in dense_rows)),
        "top_records_by_bytes": sorted(
            conv_rows + dense_rows,
            key=lambda row: int(row["record_bytes"]),
            reverse=True,
        )[:12],
        "totals_by_quantization": {key: int(value) for key, value in sorted(totals.items())},
    }


def _static_rate_break_even(
    *,
    archive_bytes: int,
    archive_sha256: str,
    reference_label: str,
    reference_archive_bytes: int | None,
    reference_archive_sha256: str | None,
    reference_score: float | None,
) -> dict[str, Any]:
    if reference_archive_bytes is None:
        return {
            "available": False,
            "reason": "no reference archive bytes supplied",
            "score_claim": False,
        }
    delta_bytes = int(archive_bytes - reference_archive_bytes)
    rate_delta = 25.0 * delta_bytes / ORIGINAL_VIDEO_BYTES
    return {
        "available": True,
        "profiled_archive_bytes": int(archive_bytes),
        "profiled_archive_sha256": archive_sha256,
        "reference_archive_bytes": int(reference_archive_bytes),
        "reference_archive_sha256": reference_archive_sha256,
        "reference_label": reference_label,
        "reference_score": reference_score,
        "score_claim": False,
        "static_archive_delta_bytes_vs_reference": delta_bytes,
        "static_rate_term_delta_vs_reference": round(rate_delta, 15),
        "required_distortion_term_improvement_to_break_even_if_reference_score_is_current": round(
            max(rate_delta, 0.0), 15
        ),
        "valid_only_if_reference_profile_matches_current_frontier": True,
    }


def _transfer_recommendations(bundle: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {row["name"]: row for row in bundle["segments"]}
    recs = [
        {
            "candidate": "container/header port",
            "bytes_at_stake": int(bundle["header_bytes"]),
            "exact_cuda_eval_required": True,
            "recommendation": "Study only after replay eval returns; header-byte savings alone cannot justify PR79 transfer.",
        },
        {
            "candidate": "postprocess/control-field transfer",
            "bytes_at_stake": int(
                sum(by_name[name]["encoded_bytes"] for name in by_name if name in {
                    "post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti"
                })
            ),
            "exact_cuda_eval_required": True,
            "recommendation": "Treat as high-risk scorer-coupled action stream; require semantic parity and exact CUDA component gates before transplant.",
        },
        {
            "candidate": "QH0 model self-compression",
            "bytes_at_stake": int(by_name.get("model", {}).get("encoded_bytes", 0)),
            "exact_cuda_eval_required": True,
            "recommendation": "Use decoded QH0 byte anatomy as a search map only; any model-byte replacement is score-affecting and gated on PR82 replay plus exact CUDA eval.",
        },
        {
            "candidate": "mask stream transplant",
            "bytes_at_stake": int(by_name.get("mask", {}).get("encoded_bytes", 0)),
            "exact_cuda_eval_required": True,
            "recommendation": "Largest stream, but not portable from bytes alone; require decoded-frame or archive-level safety proof before any exact-eval planning.",
        },
    ]
    if model.get("parse_valid") is False:
        recs.append(
            {
                "candidate": "model parser hardening",
                "bytes_at_stake": int(by_name.get("model", {}).get("encoded_bytes", 0)),
                "exact_cuda_eval_required": False,
                "recommendation": "Static QH0 table did not close; inspect replay architecture before deriving layer-level transfer atoms.",
            }
        )
    return recs


def build_profile(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    replay_inflate: Path = DEFAULT_REPLAY_INFLATE,
    output_json: Path | None = DEFAULT_OUTPUT_JSON,
    reference_archive_bytes: int | None = DEFAULT_REFERENCE_ARCHIVE_BYTES,
    reference_archive_sha256: str | None = DEFAULT_REFERENCE_ARCHIVE_SHA256,
    reference_label: str = DEFAULT_REFERENCE_LABEL,
    reference_score: float | None = DEFAULT_REFERENCE_SCORE,
) -> dict[str, Any]:
    archive = Path(archive)
    replay_inflate = Path(replay_inflate)
    zip_container, payload = _read_single_member_archive(archive)
    replay_contract = _extract_replay_contract(replay_inflate)
    bundle = _parse_compact_bundle(payload, replay_contract)

    segment_bytes: dict[str, bytes] = {}
    for row in bundle["segments"]:
        start = int(row["encoded_offset"])
        end = start + int(row["encoded_bytes"])
        segment_bytes[row["name"]] = payload[start:end]

    decoded = {
        name: _decoded_segment(segment_bytes, name)
        for name in ["model", "pose", "post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti"]
    }
    model = _parse_qh0_model(decoded["model"])
    profile = {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_performed": False,
        "cuda_eval_status": "not_run_by_profiler",
        "cuda_gate": CUDA_GATE,
        "zip_container": zip_container,
        "compact_bundle": bundle,
        "replay_inflate_contract": replay_contract,
        "anatomy": {
            "model_qh0": model,
            "pose": _parse_pose(decoded["pose"]),
            "postprocess": _parse_postprocess(decoded["post"]),
            "control_fields": [
                _parse_control_field(name, decoded[name])
                for name in ["shift", "frac", "frac2", "frac3", "bias", "region"]
            ],
            "randmulti": _parse_randmulti(decoded["randmulti"], replay_contract),
        },
    }
    profile["static_rate_break_even_vs_reference"] = _static_rate_break_even(
        archive_bytes=int(zip_container["archive_bytes"]),
        archive_sha256=str(zip_container["archive_sha256"]),
        reference_archive_bytes=reference_archive_bytes,
        reference_archive_sha256=reference_archive_sha256,
        reference_label=reference_label,
        reference_score=reference_score,
    )
    profile["transfer_recommendations"] = _transfer_recommendations(bundle, model)
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_bytes(_json_bytes(profile))
    return profile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--replay-inflate", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--reference-archive-bytes", type=int, default=DEFAULT_REFERENCE_ARCHIVE_BYTES)
    parser.add_argument("--reference-archive-sha256", default=DEFAULT_REFERENCE_ARCHIVE_SHA256)
    parser.add_argument("--reference-label", default=DEFAULT_REFERENCE_LABEL)
    parser.add_argument("--reference-score", type=float, default=DEFAULT_REFERENCE_SCORE)
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="omit static PR79/S2 rate break-even calculation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    reference_archive_bytes = None if args.no_reference else args.reference_archive_bytes
    reference_archive_sha256 = None if args.no_reference else args.reference_archive_sha256
    reference_score = None if args.no_reference else args.reference_score
    profile = build_profile(
        archive=args.archive,
        replay_inflate=args.replay_inflate,
        output_json=args.output_json,
        reference_archive_bytes=reference_archive_bytes,
        reference_archive_sha256=reference_archive_sha256,
        reference_label=args.reference_label,
        reference_score=reference_score,
    )
    print(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
