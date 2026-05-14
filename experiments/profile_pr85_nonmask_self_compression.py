#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit PR85 non-mask self-compression and single-blob overhead.

This is a local static profiler. It slices the PR85 single-member ``x`` bundle
with the canonical parser, profiles every non-mask segment at byte/bit level,
probes lossless recompression baselines, and emits fail-closed builder
candidates. It does not inflate frames, load scorers, train, dispatch remote
jobs, or claim score.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import math
import sys
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    FIXED_V5_LENGTHS,
    Pr85BundleError,
    SEGMENT_ORDER,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/profile_pr85_nonmask_self_compression.py"
SCHEMA = "pr85_nonmask_self_compression_audit_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PR90_PROBE = (
    REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json"
)
DEFAULT_PR91_ARCHIVE = REPO_ROOT / "experiments/results/public_pr91_intake_20260504_worker/archive.zip"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_nonmask_self_compression_audit_20260504_worker"
)
DEFAULT_JSON_OUT = DEFAULT_OUT_DIR / "pr85_nonmask_self_compression_audit.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT_DIR / "pr85_nonmask_self_compression_audit.md"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
NON_MASK_SEGMENTS = tuple(name for name in SEGMENT_ORDER if name != "mask")
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class AuditError(RuntimeError):
    """Raised when the audit cannot safely parse its source artifacts."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(p)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{_rel(path)}: expected JSON object")
    return payload


def _rate_score_delta(delta_bytes: int) -> float:
    return round(float(delta_bytes) * RATE_SCORE_PER_BYTE, 12)


def _magic_ascii(data: bytes, *, limit: int = 12) -> str:
    return "".join(chr(value) if 32 <= value <= 126 else "." for value in data[:limit])


def _entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = float(len(data))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _longest_equal_run(data: bytes) -> dict[str, Any]:
    if not data:
        return {"byte": None, "length": 0, "start": None}
    best_byte = data[0]
    best_start = 0
    best_len = 1
    cur_byte = data[0]
    cur_start = 0
    cur_len = 1
    for idx, value in enumerate(data[1:], start=1):
        if value == cur_byte:
            cur_len += 1
            continue
        if cur_len > best_len:
            best_byte = cur_byte
            best_start = cur_start
            best_len = cur_len
        cur_byte = value
        cur_start = idx
        cur_len = 1
    if cur_len > best_len:
        best_byte = cur_byte
        best_start = cur_start
        best_len = cur_len
    return {"byte": int(best_byte), "length": int(best_len), "start": int(best_start)}


def _byte_profile(data: bytes) -> dict[str, Any]:
    counts = Counter(data)
    entropy = _entropy_bits_per_byte(data)
    entropy_bytes = int(math.ceil(entropy * len(data) / 8.0))
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    return {
        "bytes": int(len(data)),
        "sha256": _sha256_bytes(data),
        "magic_hex": data[:12].hex(),
        "magic_ascii": _magic_ascii(data),
        "entropy_bits_per_byte": round(entropy, 12),
        "zero_order_entropy_bytes": entropy_bytes,
        "zero_order_entropy_delta_vs_input_bytes": entropy_bytes - len(data),
        "unique_byte_count": int(len(counts)),
        "zero_byte_fraction": round(data.count(0) / len(data), 12) if data else None,
        "ff_byte_fraction": round(data.count(255) / len(data), 12) if data else None,
        "longest_equal_byte_run": _longest_equal_run(data),
        "top_byte_frequencies": [
            {
                "byte": int(value),
                "hex": f"{value:02x}",
                "count": int(count),
                "fraction": round(count / len(data), 12) if data else None,
            }
            for value, count in top
        ],
    }


def _brotli_available() -> bool:
    try:
        import brotli  # noqa: F401
    except ImportError:
        return False
    return True


def _brotli_compress(data: bytes, *, quality: int) -> bytes | None:
    try:
        import brotli
    except ImportError:
        return None
    return brotli.compress(data, quality=quality)


def _brotli_decompress(data: bytes) -> tuple[bytes | None, str | None]:
    try:
        import brotli
    except ImportError:
        return None, "brotli_python_package_missing"
    try:
        return brotli.decompress(data), None
    except brotli.error as exc:
        return None, f"brotli_decode_failed:{exc}"


def _compression_probes(data: bytes) -> dict[str, Any]:
    probes = [
        ("zlib_9", zlib.compress(data, level=9)),
        ("lzma_preset9", lzma.compress(data, preset=9)),
    ]
    for quality in (1, 5, 9, 11):
        compressed = _brotli_compress(data, quality=quality)
        if compressed is not None:
            probes.append((f"brotli_q{quality}", compressed))
    rows = [
        {
            "codec": name,
            "bytes": int(len(encoded)),
            "delta_vs_input_bytes": int(len(encoded) - len(data)),
            "ratio": round(len(encoded) / len(data), 12) if data else None,
            "sha256": _sha256_bytes(encoded),
        }
        for name, encoded in probes
    ]
    best = min(rows, key=lambda row: (int(row["bytes"]), str(row["codec"]))) if rows else None
    return {
        "input_bytes": int(len(data)),
        "brotli_available": _brotli_available(),
        "probes": rows,
        "best_probe": best,
        "best_probe_delta_vs_input_bytes": None if best is None else best["delta_vs_input_bytes"],
        "generic_recompression_improves": bool(best and int(best["bytes"]) < len(data)),
    }


def _brotli_decoded_recode_probes(decoded: bytes) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for quality in (1, 5, 9, 11):
        compressed = _brotli_compress(decoded, quality=quality)
        if compressed is None:
            continue
        roundtrip, error = _brotli_decompress(compressed)
        rows.append(
            {
                "codec": f"brotli_q{quality}",
                "bytes": int(len(compressed)),
                "sha256": _sha256_bytes(compressed),
                "roundtrip_decoded_sha256": _sha256_bytes(roundtrip) if roundtrip is not None else None,
                "roundtrip_ok": roundtrip == decoded,
                "roundtrip_error": error,
            }
        )
    best = min(rows, key=lambda row: (int(row["bytes"]), str(row["codec"]))) if rows else None
    return {"probes": rows, "best_probe": best}


def _read_pr85_archive(path: Path) -> tuple[bytes, dict[str, Any], zipfile.ZipInfo]:
    if not path.is_file():
        raise AuditError(f"source PR85 archive is missing: {_rel(path)}")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if len(infos) != 1:
                raise AuditError(f"expected one non-directory ZIP member, got {len(infos)}")
            info = infos[0]
            validate_pr85_member_name(info.filename)
            raw = zf.read(info.filename)
    except (zipfile.BadZipFile, Pr85BundleError) as exc:
        raise AuditError(f"strict PR85 archive read failed closed: {exc}") from exc
    return raw, {
        "archive_path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": _sha256_bytes(raw),
        "zip_stored": bool(info.compress_type == zipfile.ZIP_STORED),
    }, info


def _deterministic_zip_size(raw: bytes, *, member_name: str, compression: int) -> int | None:
    try:
        handle = io.BytesIO()
        info = zipfile.ZipInfo(member_name, FIXED_ZIP_TIMESTAMP)
        info.compress_type = compression
        info.external_attr = 0o644 << 16
        with zipfile.ZipFile(handle, "w") as zf:
            zf.writestr(info, raw)
        return int(len(handle.getvalue()))
    except (RuntimeError, NotImplementedError, zlib.error, lzma.LZMAError):
        return None


def _container_overhead(
    *,
    archive_info: Mapping[str, Any],
    zip_info: zipfile.ZipInfo,
    raw: bytes,
    bundle_header_bytes: int,
) -> dict[str, Any]:
    archive_bytes = int(archive_info["archive_bytes"])
    member_bytes = int(archive_info["member_file_size"])
    zip_overhead = archive_bytes - member_bytes
    filename_bytes = len(zip_info.filename.encode("utf-8"))
    local_header = 30 + filename_bytes + len(zip_info.extra)
    central_header = 46 + filename_bytes + len(zip_info.extra) + len(zip_info.comment)
    eocd = 22
    structural_min = 30 + 1 + 46 + 1 + 22
    baselines: list[dict[str, Any]] = []
    for name, method, risk in (
        ("zip_stored", zipfile.ZIP_STORED, "low"),
        ("zip_deflated", zipfile.ZIP_DEFLATED, "low"),
        ("zip_bzip2", zipfile.ZIP_BZIP2, "medium"),
        ("zip_lzma", zipfile.ZIP_LZMA, "medium"),
    ):
        size = _deterministic_zip_size(raw, member_name=str(archive_info["member_name"]), compression=method)
        if size is None:
            continue
        baselines.append(
            {
                "container": name,
                "archive_bytes": size,
                "delta_vs_source_archive_bytes": int(size - archive_bytes),
                "formula_only_rate_score_delta": _rate_score_delta(size - archive_bytes),
                "runtime_risk": risk,
                "member_payload_after_unzip_identical": True,
            }
        )
    best = min(baselines, key=lambda row: (int(row["archive_bytes"]), str(row["container"]))) if baselines else None
    return {
        "zip_container_overhead_bytes": int(zip_overhead),
        "zip_container_overhead_bits": int(zip_overhead * 8),
        "zip_structural_breakdown_bytes": {
            "local_file_header": int(local_header),
            "central_directory_header": int(central_header),
            "end_of_central_directory": int(eocd),
            "accounted_total": int(local_header + central_header + eocd),
        },
        "theoretical_min_single_member_zip_overhead_bytes": int(structural_min),
        "arbitrary_extra_zip_overhead_bytes": int(max(0, zip_overhead - structural_min)),
        "zip_has_extra_fields": bool(zip_info.extra),
        "zip_member_comment_bytes": int(len(zip_info.comment)),
        "bundle_header_bytes": int(bundle_header_bytes),
        "bundle_header_bits": int(bundle_header_bytes * 8),
        "v5_header_saves_bytes_vs_explicit_30": 6,
        "outer_container_baselines": baselines,
        "best_outer_container_baseline": best,
        "overhead_recommendation": (
            "No direct single-blob overhead candidate: current one-byte member name and no extra/comment fields are already at the 100-byte ZIP structural floor."
            if zip_overhead <= structural_min
            else "Remove non-structural ZIP extra/comment overhead before any eval."
        ),
    }


def _decoded_kind(name: str, decoded: bytes | None) -> str | None:
    if decoded is None:
        return None
    prefix = decoded[:4]
    if name == "model" and prefix.startswith((b"QH0", b"QH1", b"QM0")):
        return "pr85_joint_frame_model"
    if name == "pose" and prefix == b"P1D1":
        return "pr85_p1d1_pose"
    if name in {"post", "shift", "frac", "frac2", "frac3", "bias", "region"}:
        return f"pr85_qpost_{decoded[:3].decode('ascii', errors='replace')}"
    if name == "randmulti":
        return "pr85_headerless_sparse_randmulti"
    return "unknown_brotli_payload"


def _runtime_risk_for_segment(name: str, *, fixed_length: bool) -> str:
    if fixed_length:
        return "high"
    if name in {"model", "pose", "randmulti"}:
        return "medium"
    return "low"


def _segment_profile(name: str, data: bytes, offset: int) -> dict[str, Any]:
    encoded = _byte_profile(data)
    encoded_probes = _compression_probes(data)
    decoded, decode_error = _brotli_decompress(data)
    fixed_length = name in FIXED_V5_LENGTHS
    row: dict[str, Any] = {
        "name": name,
        "offset_in_x_member": int(offset),
        "bit_range": {"start": int(offset * 8), "end_exclusive": int((offset + len(data)) * 8)},
        "fixed_length_public_v5": fixed_length,
        "fixed_length_bytes": FIXED_V5_LENGTHS.get(name),
        "encoded": encoded,
        "encoded_recompression_baselines": encoded_probes,
        "container": {
            "brotli_decodable": decoded is not None,
            "brotli_decode_error": decode_error,
            "decoded_payload_kind": _decoded_kind(name, decoded),
            "decoded_magic_hex": None if decoded is None else decoded[:12].hex(),
            "decoded_magic_ascii": None if decoded is None else _magic_ascii(decoded),
        },
        "fixed_constant_signal": {
            "encoded_single_byte_constant": encoded["unique_byte_count"] == 1,
            "encoded_longest_run_fraction": (
                round(encoded["longest_equal_byte_run"]["length"] / len(data), 12) if data else None
            ),
            "public_runtime_fixed_length_constant": fixed_length,
        },
        "candidate_assessment": {
            "lossless_decoded_recode_candidate": None,
            "blocked_or_rejected_reasons": [],
        },
    }
    if decoded is not None:
        decoded_profile = _byte_profile(decoded)
        decoded_recode = _brotli_decoded_recode_probes(decoded)
        best = decoded_recode["best_probe"]
        best_delta = None if best is None else int(best["bytes"]) - len(data)
        row["decoded"] = decoded_profile
        row["decoded_brotli_recode_baselines"] = decoded_recode
        row["fixed_constant_signal"].update(
            {
                "decoded_single_byte_constant": decoded_profile["unique_byte_count"] == 1,
                "decoded_longest_run_fraction": (
                    round(decoded_profile["longest_equal_byte_run"]["length"] / len(decoded), 12)
                    if decoded
                    else None
                ),
                "decoded_default_or_constant_heavy": bool(
                    decoded_profile["entropy_bits_per_byte"] <= 2.0
                    or (
                        decoded_profile["longest_equal_byte_run"]["length"] / len(decoded)
                        if decoded
                        else 0
                    )
                    >= 0.5
                ),
            }
        )
        if best is not None and best_delta is not None and best_delta < 0:
            if fixed_length:
                row["candidate_assessment"]["blocked_or_rejected_reasons"].append(
                    "smaller Brotli stream would change a public-v5 fixed-length segment; requires reviewed runtime/header change"
                )
            elif not bool(best.get("roundtrip_ok")):
                row["candidate_assessment"]["blocked_or_rejected_reasons"].append(
                    "best decoded recode failed Brotli roundtrip"
                )
            else:
                row["candidate_assessment"]["lossless_decoded_recode_candidate"] = {
                    "candidate_id": f"lossless_brotli_recode_pr85_{name}_segment",
                    "segment": name,
                    "archive_builder_action": (
                        f"decompress PR85 {name} segment, recompress decoded bytes with {best['codec']}, "
                        "then pack the original segment map with header_mode='v5'"
                    ),
                    "expected_archive_delta_bytes": int(best_delta),
                    "expected_rate_score_delta_formula_only": _rate_score_delta(best_delta),
                    "source_segment_bytes": int(len(data)),
                    "candidate_segment_bytes": int(best["bytes"]),
                    "candidate_segment_sha256": best["sha256"],
                    "decoded_sha256_preserved": decoded_profile["sha256"],
                    "state_change_required": True,
                    "no_op": False,
                    "runtime_risk": _runtime_risk_for_segment(name, fixed_length=fixed_length),
                    "runtime_risk_reason": "lossless decoded bytes are identical, but runtime-output parity is still required before eval",
                    "dispatchable_now": False,
                    "next_gate": "build candidate archive, prove segment decoded SHA parity and PR85 runtime output parity locally",
                }
        else:
            row["candidate_assessment"]["blocked_or_rejected_reasons"].append(
                "decoded Brotli recode is non-improving or unavailable"
            )
    else:
        row["decoded"] = None
        row["decoded_brotli_recode_baselines"] = None
        row["candidate_assessment"]["blocked_or_rejected_reasons"].append(
            "segment is not Brotli-decodable with local brotli package"
        )
    if not bool(encoded_probes["generic_recompression_improves"]):
        row["candidate_assessment"]["blocked_or_rejected_reasons"].append(
            "wrapping encoded bytes in zlib/brotli/lzma is a no-op or larger"
        )
    return row


def _pr91_identity_comparison(pr85_segments: Mapping[str, bytes], pr91_archive: Path | None) -> dict[str, Any]:
    if pr91_archive is None or not pr91_archive.is_file():
        return {"available": False, "path": _rel(pr91_archive), "reason": "PR91 archive missing"}
    try:
        raw, archive_info, _info = _read_pr85_archive(pr91_archive)
        pr91 = parse_pr85_bundle(raw)
    except (AuditError, Pr85BundleError) as exc:
        return {"available": False, "path": _rel(pr91_archive), "reason": str(exc)}
    rows = []
    for name in NON_MASK_SEGMENTS:
        pr85_data = bytes(pr85_segments[name])
        pr91_data = bytes(pr91.segments[name])
        rows.append(
            {
                "name": name,
                "pr85_bytes": int(len(pr85_data)),
                "pr91_bytes": int(len(pr91_data)),
                "byte_delta_vs_pr85": int(len(pr91_data) - len(pr85_data)),
                "pr85_sha256": _sha256_bytes(pr85_data),
                "pr91_sha256": _sha256_bytes(pr91_data),
                "sha_equal": pr85_data == pr91_data,
            }
        )
    return {
        "available": True,
        "archive": archive_info,
        "nonmask_all_identity": all(bool(row["sha_equal"]) for row in rows),
        "segments": rows,
        "useful_conclusion": (
            "PR91 changes the mask segment only; any true PR85 non-mask byte reduction should stack byte-for-byte with PR91."
            if all(bool(row["sha_equal"]) for row in rows)
            else "PR91 has non-mask differences; inspect per-segment rows before stacking."
        ),
    }


def _pr90_comparison(
    pr85_segments: Mapping[str, bytes],
    pr90_probe_json: Path | None,
) -> dict[str, Any]:
    probe = _read_json(pr90_probe_json)
    if not probe:
        return {"available": False, "path": _rel(pr90_probe_json), "reason": "PR90 probe JSON missing"}
    slices = probe.get("slices", {})
    compact = probe.get("compact_constants", {})
    if not isinstance(slices, Mapping) or not isinstance(compact, Mapping):
        return {"available": False, "path": _rel(pr90_probe_json), "reason": "malformed PR90 probe"}
    model_body = int(compact.get("model_body_bytes", slices.get("model_body", {}).get("len", 0)))
    pose_qrgb = int(slices.get("pose_qrgb_body", {}).get("len", 0))
    pr85_model = len(pr85_segments["model"])
    pr85_controls = sum(len(pr85_segments[name]) for name in NON_MASK_SEGMENTS if name != "model")
    pr85_nonmask = pr85_model + pr85_controls
    pr90_nonmask = model_body + pose_qrgb
    candidates = []
    if model_body and model_body < pr85_model:
        delta = model_body - pr85_model
        candidates.append(
            {
                "candidate_id": "pr90_qfq4_style_pr85_model_serializer_probe",
                "surface": "PR85 model segment",
                "archive_builder_action": "prototype a PR85 decoded-QH0 to QFQ4/grouped-FP serializer and paired runtime loader",
                "expected_archive_delta_bytes_formula_only": int(delta),
                "expected_rate_score_delta_formula_only": _rate_score_delta(delta),
                "source_basis": "PR90 model body is smaller than PR85 model segment, but not byte-identical or runtime-compatible",
                "identity_preserving": False,
                "state_change_required": True,
                "no_op": False,
                "runtime_risk": "high",
                "runtime_risk_reason": "different model grammar; requires tensor/output parity and exact CUDA component gates",
                "dispatchable_now": False,
                "next_gate": "build a local serializer parity harness; do not exact-eval until runtime output parity passes",
            }
        )
    if pose_qrgb and pose_qrgb < pr85_controls:
        delta = pose_qrgb - pr85_controls
        candidates.append(
            {
                "candidate_id": "pr90_qrgb_control_stack_recode_probe",
                "surface": "PR85 pose/post/motion/bias/region/randmulti control stack",
                "archive_builder_action": "lower PR90-style compact pose plus sparse QRGB/control actions into an explicit PR85-family qpost candidate",
                "expected_archive_delta_bytes_formula_only": int(delta),
                "expected_rate_score_delta_formula_only": _rate_score_delta(delta),
                "source_basis": "PR90 compact control body is smaller than the PR85 non-mask control stack, but semantics are different",
                "identity_preserving": False,
                "state_change_required": True,
                "no_op": False,
                "runtime_risk": "high",
                "runtime_risk_reason": "changes pose/color/control semantics; component deltas can dominate byte savings",
                "dispatchable_now": False,
                "next_gate": "build local non-noop candidate with raw-output/runtime parity diagnostics before any eval claim",
            }
        )
    return {
        "available": True,
        "path": _rel(pr90_probe_json),
        "payload_len": probe.get("payload_len"),
        "split_mode": probe.get("split_mode"),
        "slice_sizes": {
            "model_body": model_body,
            "pose_qrgb_body": pose_qrgb,
            "pr85_model_segment": int(pr85_model),
            "pr85_control_stack_nonmodel_nonmask": int(pr85_controls),
            "pr85_total_nonmask": int(pr85_nonmask),
            "pr90_model_plus_pose_qrgb": int(pr90_nonmask),
        },
        "deltas_vs_pr85": {
            "model_body_vs_pr85_model_bytes": int(model_body - pr85_model) if model_body else None,
            "pose_qrgb_vs_pr85_control_stack_bytes": int(pose_qrgb - pr85_controls)
            if pose_qrgb
            else None,
            "total_nonmask_vs_pr85_nonmask_bytes": int(pr90_nonmask - pr85_nonmask)
            if pr90_nonmask
            else None,
        },
        "identity_preserving": False,
        "useful_conclusion": "PR90 is a non-identity architecture signal, not a PR85 self-compression proof.",
        "architecture_transfer_candidates": candidates,
    }


def _collect_lossless_candidates(segment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in segment_rows:
        candidate = row.get("candidate_assessment", {}).get("lossless_decoded_recode_candidate")
        if isinstance(candidate, Mapping):
            candidates.append(dict(candidate))
    return sorted(
        candidates,
        key=lambda row: (int(row["expected_archive_delta_bytes"]), str(row["candidate_id"])),
    )


def _rejected_noop_rows(
    segment_rows: Sequence[Mapping[str, Any]],
    container: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in segment_rows:
        assessment = row.get("candidate_assessment", {})
        candidate = assessment.get("lossless_decoded_recode_candidate") if isinstance(assessment, Mapping) else None
        if candidate is None:
            best = None
            decoded_recode = row.get("decoded_brotli_recode_baselines")
            if isinstance(decoded_recode, Mapping):
                best = decoded_recode.get("best_probe")
            rows.append(
                {
                    "surface": f"segment:{row['name']}",
                    "source_bytes": row["encoded"]["bytes"],
                    "best_runtime_compatible_recode_bytes": None if not isinstance(best, Mapping) else best.get("bytes"),
                    "rejected_as_noop_or_blocked": True,
                    "reasons": assessment.get("blocked_or_rejected_reasons", [])
                    if isinstance(assessment, Mapping)
                    else [],
                }
            )
    best_container = container.get("best_outer_container_baseline")
    if isinstance(best_container, Mapping) and int(best_container["delta_vs_source_archive_bytes"]) >= 0:
        rows.append(
            {
                "surface": "single_blob_zip_container",
                "source_bytes": None,
                "best_runtime_compatible_recode_bytes": best_container["archive_bytes"],
                "rejected_as_noop_or_blocked": True,
                "reasons": ["deterministic stored/deflated ZIP baselines do not reduce archive bytes"],
            }
        )
    return rows


def _markdown_report(profile: Mapping[str, Any]) -> str:
    lines = [
        "# PR85 Non-Mask Self-Compression Audit",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- dispatch_performed: false",
        f"- archive: `{profile['archive']['archive_path']}`",
        f"- archive_bytes: {profile['archive']['archive_bytes']}",
        f"- x_member_bytes: {profile['archive']['member_file_size']}",
        f"- zip_overhead_bytes: {profile['single_blob_container_overhead']['zip_container_overhead_bytes']}",
        "",
        "## Direct Lossless Candidates",
        "",
    ]
    direct = profile["lossless_archive_builder_candidates"]
    if direct:
        for candidate in direct:
            lines.extend(
                [
                    f"- `{candidate['candidate_id']}`: {candidate['expected_archive_delta_bytes']} bytes, "
                    f"rate-score delta {candidate['expected_rate_score_delta_formula_only']}, "
                    f"runtime risk {candidate['runtime_risk']}.",
                ]
            )
    else:
        lines.append("- none: all non-mask decoded Brotli recodes and outer-container baselines were non-improving or blocked.")
    lines.extend(["", "## Architecture Transfer Candidates", ""])
    transfers = profile["architecture_transfer_candidates"]
    if transfers:
        for candidate in transfers:
            lines.extend(
                [
                    f"- `{candidate['candidate_id']}`: {candidate['expected_archive_delta_bytes_formula_only']} bytes "
                    f"(formula-only rate delta {candidate['expected_rate_score_delta_formula_only']}), "
                    f"runtime risk {candidate['runtime_risk']}; {candidate['source_basis']}.",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "## PR91 Identity", ""])
    pr91 = profile["comparisons"]["pr91_nonmask_identity"]
    lines.append(f"- available: {pr91.get('available')}")
    lines.append(f"- nonmask_all_identity: {pr91.get('nonmask_all_identity')}")
    lines.append(f"- conclusion: {pr91.get('useful_conclusion')}")
    return "\n".join(lines) + "\n"


def build_profile(
    archive: Path = DEFAULT_ARCHIVE,
    *,
    pr90_probe_json: Path | None = DEFAULT_PR90_PROBE,
    pr91_archive: Path | None = DEFAULT_PR91_ARCHIVE,
) -> dict[str, Any]:
    raw, archive_info, zip_info = _read_pr85_archive(Path(archive))
    try:
        bundle = parse_pr85_bundle(raw)
    except Pr85BundleError as exc:
        raise AuditError(f"canonical PR85 parser rejected source bundle: {exc}") from exc
    segment_rows = [
        _segment_profile(name, bytes(bundle.segments[name]), int(bundle.segment_offsets[name]))
        for name in NON_MASK_SEGMENTS
    ]
    container = _container_overhead(
        archive_info=archive_info,
        zip_info=zip_info,
        raw=raw,
        bundle_header_bytes=int(bundle.header_bytes),
    )
    lossless_candidates = _collect_lossless_candidates(segment_rows)
    pr90 = _pr90_comparison(bundle.segments, pr90_probe_json)
    transfer_candidates = (
        pr90.get("architecture_transfer_candidates", []) if isinstance(pr90, Mapping) else []
    )
    profile: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "local_only": True,
        "deterministic": True,
        "evidence_grade": "planning_only_static_nonmask_self_compression_audit",
        "cuda_score_truth": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "archive": archive_info,
        "inputs": {
            "archive": _rel(archive),
            "pr90_probe_json": _rel(pr90_probe_json),
            "pr91_archive": _rel(pr91_archive),
        },
        "bundle": {
            "format": bundle.format,
            "header_bytes": int(bundle.header_bytes),
            "segment_order": list(SEGMENT_ORDER),
            "nonmask_segments_audited": list(NON_MASK_SEGMENTS),
            "mask_segment_excluded_from_candidate_search": {
                "bytes": int(len(bundle.segments["mask"])),
                "sha256": _sha256_bytes(bytes(bundle.segments["mask"])),
                "reason": "task scope is non-mask self-compression",
            },
            "segment_lengths": bundle.segment_lengths,
            "segment_offsets": {name: int(offset) for name, offset in bundle.segment_offsets.items()},
            "fixed_length_segments": dict(bundle.fixed_length_segments),
        },
        "score_rate_formula": {
            "formula_only": True,
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "score_claim_from_this_profile": False,
        },
        "single_blob_container_overhead": container,
        "nonmask_segments": segment_rows,
        "lossless_archive_builder_candidates": lossless_candidates,
        "architecture_transfer_candidates": list(transfer_candidates),
        "top_implementable_byte_saving_candidates": [
            *lossless_candidates,
            *list(transfer_candidates),
        ],
        "rejected_noop_or_blocked_recommendations": _rejected_noop_rows(segment_rows, container),
        "comparisons": {
            "pr90_nonidentity_size_signal": pr90,
            "pr91_nonmask_identity": _pr91_identity_comparison(bundle.segments, pr91_archive),
        },
        "fail_closed_summary": {
            "direct_lossless_candidate_count": int(len(lossless_candidates)),
            "architecture_transfer_candidate_count": int(len(transfer_candidates)),
            "no_op_recommendations_promoted": 0,
            "score_claim": False,
            "dispatch_performed": False,
        },
    }
    return profile


def write_outputs(profile: Mapping[str, Any], *, json_out: Path, markdown_out: Path | None) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(_json_text(profile), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_markdown_report(profile), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--pr90-probe-json", type=Path, default=DEFAULT_PR90_PROBE)
    parser.add_argument("--pr91-archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args(argv)

    try:
        profile = build_profile(
            args.archive.resolve(),
            pr90_probe_json=args.pr90_probe_json.resolve() if args.pr90_probe_json else None,
            pr91_archive=args.pr91_archive.resolve() if args.pr91_archive else None,
        )
    except AuditError as exc:
        print(f"{TOOL}: failed closed: {exc}", file=sys.stderr)
        return 2
    write_outputs(profile, json_out=args.json_out.resolve(), markdown_out=args.markdown_out.resolve())
    print(_json_text(profile), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
