#!/usr/bin/env python3
"""Static PR81/QMA9 semantic range-mask intake profiler.

This tool is local planning/forensics only. It parses charged archive bytes,
fixed PR81 payload slices, and the QMA9 range-mask header. It does not run the
contest scorer, does not require CUDA, and does not dispatch GPU or remote
jobs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/profile_pr81_qma9_range_mask_contract.py"
SCHEMA = "pr81_qma9_semantic_range_mask_contract_v1"
EVIDENCE_GRADE = "external/planning_only"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_LAMBDA = 25.0 / ORIGINAL_VIDEO_BYTES

DEFAULT_PR81_DIR = REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex"
DEFAULT_PR81_ARCHIVE = DEFAULT_PR81_DIR / "archive.zip"
DEFAULT_PR81_INFLATE = DEFAULT_PR81_DIR / "replay_submission/inflate.py"
DEFAULT_RANGE_MASK_CODEC = DEFAULT_PR81_DIR / "range_mask_codec.cpp"
DEFAULT_OUTPUT_JSON = DEFAULT_PR81_DIR / "pr81_qma9_semantic_range_mask_profile.json"
DEFAULT_PR79_PROFILE = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr79/"
    "pr79_minp_grammar_profile.json"
)
DEFAULT_PR79_S2_PROFILE = (
    REPO_ROOT
    / "experiments/results/pr79_runtime_parity_20260503_codex/"
    "pr79_s2_runtime_parity.json"
)


@dataclass(frozen=True)
class ZipMemberProfile:
    name: str
    compress_type: int
    file_size: int
    compress_size: int
    crc32_hex: str
    sha256: str
    header_offset: int


@dataclass(frozen=True)
class ArchiveProfile:
    path: str
    bytes: int
    sha256: str
    payload_container: str
    payload_bytes: int
    payload_sha256: str
    zip_overhead_bytes: int
    members: tuple[ZipMemberProfile, ...]


@dataclass(frozen=True)
class QMA9MaskProfile:
    magic: str
    frame_count: int
    width: int
    height: int
    bitstream_bytes: int
    header_bytes: int
    packed_bytes: int
    decoded_mask_bytes: int
    context_symbol_count: int
    class_symbol_count: int
    sentinel: int
    payload_sha256: str
    bitstream_sha256: str


@dataclass(frozen=True)
class PayloadSegment:
    name: str
    offset: int
    bytes: int
    sha256: str
    prefix_hex: str
    codec: str


@dataclass(frozen=True)
class PR81SplitProfile:
    payload_format: str
    boundary_authority: str
    expected_payload_bytes: int
    segments: tuple[PayloadSegment, ...]


@dataclass(frozen=True)
class ReferenceProfile:
    label: str
    available: bool
    archive_bytes: int | None = None
    archive_sha256: str | None = None
    payload_bytes: int | None = None
    payload_sha256: str | None = None
    mask_charged_bytes: int | None = None
    model_charged_bytes: int | None = None
    pose_charged_bytes: int | None = None
    router_or_action_charged_bytes: int | None = None
    source_json: str | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class BreakEvenProfile:
    reference_label: str
    reference_archive_bytes: int
    pr81_archive_bytes: int
    archive_byte_delta_pr81_minus_reference: int
    rate_score_delta_if_components_unchanged: float
    component_score_worsening_budget_before_equal_total: float
    bytes_saved_before_equal_rate: int
    note: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _eval_int_expr(node: ast.AST, env: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left, env)
        right = _eval_int_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise ValueError(f"unsupported integer constant expression: {ast.dump(node)}")


def parse_split_constants(path: Path) -> dict[str, int]:
    """Parse public QMA9 inflate constants without importing Torch code.

    PR81 records an explicit ``PACKED_PAYLOAD_BYTES`` and
    ``POSE_STREAM_BYTES``.  PR84's no-router runtime computes the same boundary
    inline with a literal 899-byte pose stream.  Keep both forms admissible so
    public replay intake can profile the actual runtime contract instead of
    requiring synthetic constants.
    """
    tree = ast.parse(path.read_text())
    env: dict[str, int] = {}
    wanted = {
        "RANGE_MASK_BYTES",
        "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
        "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
        "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
        "SPLIT_MODEL_REORDERED_BYTES",
        "POSE_STREAM_BYTES",
        "ROUTER_ACTION_BYTES",
        "ROUTER_ACTION_COUNT",
        "ROUTER_ACTION_BITS",
        "PACKED_PAYLOAD_BYTES",
    }
    # Catalog #168 fix 2026-05-12: handle both bare Assign and AnnAssign forms.
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            value_node = stmt.value
        elif (isinstance(stmt, ast.AnnAssign)
              and stmt.value is not None):
            target = stmt.target
            value_node = stmt.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        try:
            env[target.id] = _eval_int_expr(value_node, env)
        except ValueError:
            continue
    mandatory = {
        "RANGE_MASK_BYTES",
        "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
        "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
        "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
        "SPLIT_MODEL_REORDERED_BYTES",
    }
    missing = sorted(mandatory.difference(env))
    if missing:
        raise ValueError(f"{path} is missing QMA9 split constants: {missing}")
    if "POSE_STREAM_BYTES" not in env:
        env["POSE_STREAM_BYTES"] = 899
    if "ROUTER_ACTION_BYTES" not in env:
        env["ROUTER_ACTION_BYTES"] = 0
    if "ROUTER_ACTION_COUNT" not in env:
        env["ROUTER_ACTION_COUNT"] = 0
    if "ROUTER_ACTION_BITS" not in env:
        env["ROUTER_ACTION_BITS"] = 0
    if "PACKED_PAYLOAD_BYTES" not in env:
        env["PACKED_PAYLOAD_BYTES"] = (
            env["RANGE_MASK_BYTES"]
            + env["SPLIT_MODEL_REORDERED_BYTES"]
            + env["POSE_STREAM_BYTES"]
            + env["ROUTER_ACTION_BYTES"]
        )
    return {key: env[key] for key in sorted(wanted)}


def read_single_payload_zip(path: Path) -> tuple[bytes, ArchiveProfile]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["p"]:
            raise ValueError(f"{path} must contain exactly one non-directory member 'p'; got {names!r}")
        info = infos[0]
        payload = zf.read(info)
        members = (
            ZipMemberProfile(
                name=info.filename,
                compress_type=int(info.compress_type),
                file_size=int(info.file_size),
                compress_size=int(info.compress_size),
                crc32_hex=f"{int(info.CRC):08x}",
                sha256=_sha256_bytes(payload),
                header_offset=int(info.header_offset),
            ),
        )
    archive_bytes = int(path.stat().st_size)
    return payload, ArchiveProfile(
        path=str(path),
        bytes=archive_bytes,
        sha256=_sha256_file(path),
        payload_container="p",
        payload_bytes=len(payload),
        payload_sha256=_sha256_bytes(payload),
        zip_overhead_bytes=archive_bytes - len(payload),
        members=members,
    )


def parse_qma9_mask(mask_payload: bytes) -> QMA9MaskProfile:
    if len(mask_payload) < 20:
        raise ValueError("QMA9 mask payload is shorter than its 20-byte header")
    magic, frame_count, width, height, bitstream_bytes = struct.unpack_from("<4sIIII", mask_payload, 0)
    if magic != b"QMA9":
        raise ValueError(f"not a QMA9 range-mask payload: {magic!r}")
    packed_bytes = 20 + int(bitstream_bytes)
    if packed_bytes != len(mask_payload):
        raise ValueError(f"QMA9 mask length mismatch: header declares {packed_bytes}, got {len(mask_payload)}")
    return QMA9MaskProfile(
        magic=magic.decode("ascii"),
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        bitstream_bytes=int(bitstream_bytes),
        header_bytes=20,
        packed_bytes=int(packed_bytes),
        decoded_mask_bytes=int(frame_count) * int(width) * int(height),
        context_symbol_count=6**9,
        class_symbol_count=5,
        sentinel=5,
        payload_sha256=_sha256_bytes(mask_payload),
        bitstream_sha256=_sha256_bytes(mask_payload[20:]),
    )


def build_pr81_split(payload: bytes, constants: dict[str, int]) -> tuple[PR81SplitProfile, QMA9MaskProfile]:
    range_mask_bytes = int(constants["RANGE_MASK_BYTES"])
    model_bytes = int(constants["SPLIT_MODEL_REORDERED_BYTES"])
    pose_bytes = int(constants.get("POSE_STREAM_BYTES", 899))
    router_bytes = int(constants.get("ROUTER_ACTION_BYTES", 0))
    base_expected = range_mask_bytes + model_bytes + pose_bytes
    router_expected = base_expected + router_bytes
    expected = int(constants["PACKED_PAYLOAD_BYTES"])
    has_router = router_bytes > 0 and len(payload) == router_expected
    if len(payload) == base_expected:
        expected = base_expected
    elif len(payload) == router_expected:
        expected = router_expected
    elif len(payload) != expected:
        raise ValueError(
            "QMA9 payload length mismatch: "
            f"expected one of {[base_expected, router_expected, expected]}, got {len(payload)}"
        )
    pieces = [
        ("range_mask.qma9", 0, range_mask_bytes, "qma9_adaptive9_binary_range_mask"),
        ("split_model_reordered.br_bundle", range_mask_bytes, model_bytes, "brotli_reordered_qzs3_model_bundle"),
        ("optimized_poses.qp1.br", range_mask_bytes + model_bytes, pose_bytes, "brotli_qp1_pose_stream"),
    ]
    if has_router:
        pieces.append(
            (
                "router_actions.3bit",
                range_mask_bytes + model_bytes + pose_bytes,
                router_bytes,
                "packed_3bit_pair_router_actions",
            )
        )
    segments = tuple(
        PayloadSegment(
            name=name,
            offset=offset,
            bytes=size,
            sha256=_sha256_bytes(payload[offset : offset + size]),
            prefix_hex=payload[offset : offset + min(size, 16)].hex(),
            codec=codec,
        )
        for name, offset, size, codec in pieces
    )
    mask_profile = parse_qma9_mask(payload[:range_mask_bytes])
    return PR81SplitProfile(
        payload_format=(
            "public_pr81_qma9_range_mask_qzs3_split_model_qp1_router_v1"
            if has_router
            else "public_qma9_range_mask_qzs3_split_model_qp1_no_router_v1"
        ),
        boundary_authority="public_pr81_inflate_constant_table_plus_qma9_self_header",
        expected_payload_bytes=expected,
        segments=segments,
    ), mask_profile


def _dig(mapping: dict[str, Any], *path: str) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _stream_bytes(streams: dict[str, Any] | None, name: str) -> int | None:
    if not isinstance(streams, dict):
        return None
    row = streams.get(name)
    if isinstance(row, dict) and row.get("charged_bytes") is not None:
        return int(row["charged_bytes"])
    return None


def load_pr79_reference(path: Path, *, label: str) -> ReferenceProfile:
    if not path.exists():
        return ReferenceProfile(label=label, available=False, source_json=str(path), unavailable_reason="missing_json")
    payload = json.loads(path.read_text())
    public = payload.get("public_archive") if isinstance(payload.get("public_archive"), dict) else payload
    archive = public.get("archive", {}) if isinstance(public, dict) else {}
    streams = public.get("decoded_streams") if isinstance(public, dict) else None
    return ReferenceProfile(
        label=label,
        available=True,
        archive_bytes=int(archive["bytes"]) if archive.get("bytes") is not None else None,
        archive_sha256=archive.get("sha256"),
        payload_bytes=int(_dig(public, "payload", "bytes")) if _dig(public, "payload", "bytes") is not None else None,
        payload_sha256=_dig(public, "payload", "sha256"),
        mask_charged_bytes=_stream_bytes(streams, "masks.mkv"),
        model_charged_bytes=_stream_bytes(streams, "renderer.bin"),
        pose_charged_bytes=_stream_bytes(streams, "optimized_poses.qp1"),
        router_or_action_charged_bytes=_stream_bytes(streams, "seg_tile_actions.bin"),
        source_json=str(path),
    )


def load_s2_reference(path: Path, *, label: str) -> ReferenceProfile:
    if not path.exists():
        return ReferenceProfile(label=label, available=False, source_json=str(path), unavailable_reason="missing_json")
    payload = json.loads(path.read_text())
    archive = payload.get("archive", {})
    members = _dig(payload, "profiles", "robust_current", "diagnostics", "header", "members")
    by_name = {row.get("name"): row for row in members or [] if isinstance(row, dict)}
    return ReferenceProfile(
        label=label,
        available=True,
        archive_bytes=int(archive["bytes"]) if archive.get("bytes") is not None else None,
        archive_sha256=archive.get("sha256"),
        payload_bytes=int(archive["payload_bytes"]) if archive.get("payload_bytes") is not None else None,
        payload_sha256=archive.get("payload_sha256"),
        mask_charged_bytes=int(by_name["masks.mkv"]["bytes"]) if "masks.mkv" in by_name else None,
        model_charged_bytes=int(by_name["renderer.bin"]["bytes"]) if "renderer.bin" in by_name else None,
        pose_charged_bytes=int(by_name["optimized_poses.qp1"]["bytes"]) if "optimized_poses.qp1" in by_name else None,
        router_or_action_charged_bytes=int(by_name["seg_tile_actions.bin"]["bytes"]) if "seg_tile_actions.bin" in by_name else None,
        source_json=str(path),
    )


def break_even_vs_reference(pr81_archive_bytes: int, ref: ReferenceProfile) -> BreakEvenProfile | None:
    if not ref.available or ref.archive_bytes is None:
        return None
    delta = int(pr81_archive_bytes) - int(ref.archive_bytes)
    rate_delta = RATE_LAMBDA * delta
    return BreakEvenProfile(
        reference_label=ref.label,
        reference_archive_bytes=int(ref.archive_bytes),
        pr81_archive_bytes=int(pr81_archive_bytes),
        archive_byte_delta_pr81_minus_reference=delta,
        rate_score_delta_if_components_unchanged=rate_delta,
        component_score_worsening_budget_before_equal_total=max(0.0, -rate_delta),
        bytes_saved_before_equal_rate=max(0, -delta),
        note=(
            "Static contest-rate-term arithmetic only: 25 * archive_bytes / "
            f"{ORIGINAL_VIDEO_BYTES}. This is not a score claim and assumes no component evidence."
        ),
    )


def component_delta(name: str, pr81_bytes: int, ref_bytes: int | None, ref_label: str) -> dict[str, Any]:
    if ref_bytes is None:
        return {"component": name, "reference": ref_label, "available": False}
    delta = int(pr81_bytes) - int(ref_bytes)
    return {
        "component": name,
        "reference": ref_label,
        "available": True,
        "pr81_bytes": int(pr81_bytes),
        "reference_bytes": int(ref_bytes),
        "delta_bytes_pr81_minus_reference": delta,
        "rate_term_delta": RATE_LAMBDA * delta,
    }


def transfer_recommendations(
    split: PR81SplitProfile,
    references: list[ReferenceProfile],
) -> list[dict[str, Any]]:
    by_name = {segment.name: segment for segment in split.segments}
    primary_ref = next((ref for ref in references if ref.available and ref.mask_charged_bytes is not None), None)
    mask_savings = None
    router_savings = None
    model_delta = None
    pose_delta = None
    if primary_ref is not None:
        mask_savings = primary_ref.mask_charged_bytes - by_name["range_mask.qma9"].bytes
        if primary_ref.router_or_action_charged_bytes is not None:
            router_savings = primary_ref.router_or_action_charged_bytes - (
                by_name["router_actions.3bit"].bytes if "router_actions.3bit" in by_name else 0
            )
        if primary_ref.model_charged_bytes is not None:
            model_delta = by_name["split_model_reordered.br_bundle"].bytes - primary_ref.model_charged_bytes
        if primary_ref.pose_charged_bytes is not None:
            pose_delta = by_name["optimized_poses.qp1.br"].bytes - primary_ref.pose_charged_bytes
    return [
        {
            "rank": 1,
            "target": "range_mask.qma9",
            "recommendation": "isolate_QMA9_semantic_mask_coder_as_first_transfer_candidate",
            "planning_signal": "large_static_mask_byte_reduction_vs_PR79_family" if mask_savings is not None and mask_savings > 0 else "mask_reference_missing",
            "estimated_charged_byte_savings_vs_primary_reference": mask_savings,
            "required_before_dispatch": [
                "decode_hash_or_raw_semantic_mask_parity_against_public_PR81_runtime",
                "archive_payload_closure_with_QMA9_decoder_charged_or_fixed",
                "exact_cuda_auth_eval_for_any_score_or_component_claim",
            ],
            "evidence_grade": EVIDENCE_GRADE,
        },
        {
            "rank": 2,
            "target": "router_actions.3bit",
            "recommendation": "study_pair_router_as_semantic_action_compressor_not_drop_in_PR79_tile_action_replacement",
            "planning_signal": "small_3bit_pair_router_stream" if router_savings is not None and router_savings > 0 else "router_reference_missing_or_not_smaller",
            "estimated_charged_byte_savings_vs_primary_reference": router_savings,
            "required_before_dispatch": [
                "prove_action_semantics_or_rendered_output_effect_not_just_byte_savings",
                "record transfer mapping from PR79 tile actions to PR81 pair router actions",
                "exact_cuda_auth_eval_for_any_score_or_component_claim",
            ],
            "evidence_grade": EVIDENCE_GRADE,
        },
        {
            "rank": 3,
            "target": "split_model_reordered.br_bundle",
            "recommendation": "treat_as_runtime_coupled_model_split_with_low_standalone_transfer_value",
            "planning_signal": "near_byte_neutral_vs_PR79_renderer" if model_delta is not None and abs(model_delta) < 128 else "model_reference_missing_or_large_delta",
            "delta_bytes_pr81_minus_primary_reference": model_delta,
            "required_before_dispatch": [
                "loader_compatibility_and_runtime_tree_custody",
                "rendered_output_parity_or_exact_eval_custody",
            ],
            "evidence_grade": EVIDENCE_GRADE,
        },
        {
            "rank": 4,
            "target": "optimized_poses.qp1.br",
            "recommendation": "do_not_prioritize_pose_transfer_from_PR81_static_bytes_alone",
            "planning_signal": "pose_stream_byte_delta_is_negligible" if pose_delta is not None and abs(pose_delta) <= 8 else "pose_reference_missing_or_material_delta",
            "delta_bytes_pr81_minus_primary_reference": pose_delta,
            "required_before_dispatch": [
                "pose_float_or_QP1_hash_comparison_before_claiming_identity",
                "exact_cuda_auth_eval_for_any_pose_component_claim",
            ],
            "evidence_grade": EVIDENCE_GRADE,
        },
    ]


def optional_cpp_decode_hash(
    *,
    codec_cpp: Path,
    mask_payload: bytes,
    enabled: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not enabled:
        return {"attempted": False, "status": "skipped_not_requested"}
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        return {"attempted": True, "status": "unavailable", "reason": "no_cxx_compiler_on_path"}
    if not codec_cpp.exists():
        return {"attempted": True, "status": "unavailable", "reason": f"missing_codec_cpp:{codec_cpp}"}
    try:
        with tempfile.TemporaryDirectory(prefix="pr81_qma9_decode_") as tmp:
            tmpdir = Path(tmp)
            binary = tmpdir / "range_mask_codec"
            mask_bin = tmpdir / "range_mask.qma9"
            raw_out = tmpdir / "range_mask.raw"
            mask_bin.write_bytes(mask_payload)
            compile_cmd = [compiler, "-O2", "-std=c++17", str(codec_cpp), "-o", str(binary)]
            compile_proc = subprocess.run(
                compile_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if compile_proc.returncode != 0:
                return {
                    "attempted": True,
                    "status": "compile_failed",
                    "returncode": compile_proc.returncode,
                    "stderr_tail": compile_proc.stderr[-2000:],
                }
            decode_proc = subprocess.run(
                [str(binary), "decode", str(mask_bin), str(raw_out)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if decode_proc.returncode != 0:
                return {
                    "attempted": True,
                    "status": "decode_failed",
                    "returncode": decode_proc.returncode,
                    "stderr_tail": decode_proc.stderr[-2000:],
                }
            return {
                "attempted": True,
                "status": "ok",
                "decoded_bytes": raw_out.stat().st_size,
                "decoded_sha256": _sha256_file(raw_out),
                "codec_cpp": str(codec_cpp),
                "compiler": compiler,
            }
    except Exception as exc:  # pragma: no cover - fail-soft path depends on host toolchain.
        return {"attempted": True, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def build_profile(
    *,
    archive_path: Path,
    split_constants_path: Path,
    pr79_profile_path: Path,
    pr79_s2_profile_path: Path,
    range_mask_codec_cpp: Path,
    try_cpp_decode_hash: bool,
    cpp_timeout_seconds: int,
) -> dict[str, Any]:
    constants = parse_split_constants(split_constants_path)
    payload, archive = read_single_payload_zip(archive_path)
    split, qma9 = build_pr81_split(payload, constants)
    pr79 = load_pr79_reference(pr79_profile_path, label="PR79_public")
    s2 = load_s2_reference(pr79_s2_profile_path, label="PR79_S2")
    references = [pr79, s2]
    by_segment = {segment.name: segment for segment in split.segments}
    break_even = [be for ref in references if (be := break_even_vs_reference(archive.bytes, ref)) is not None]
    component_deltas: list[dict[str, Any]] = []
    for ref in references:
        if not ref.available:
            continue
        component_deltas.extend(
            [
                component_delta("mask_or_range_mask", by_segment["range_mask.qma9"].bytes, ref.mask_charged_bytes, ref.label),
                component_delta("model_or_renderer", by_segment["split_model_reordered.br_bundle"].bytes, ref.model_charged_bytes, ref.label),
                component_delta("pose", by_segment["optimized_poses.qp1.br"].bytes, ref.pose_charged_bytes, ref.label),
                component_delta(
                    "router_or_actions",
                    by_segment["router_actions.3bit"].bytes if "router_actions.3bit" in by_segment else 0,
                    ref.router_or_action_charged_bytes,
                    ref.label,
                ),
            ]
        )
    decode_hash = optional_cpp_decode_hash(
        codec_cpp=range_mask_codec_cpp,
        mask_payload=payload[: constants["RANGE_MASK_BYTES"]],
        enabled=try_cpp_decode_hash,
        timeout_seconds=cpp_timeout_seconds,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "notes": [
            "External/public artifact intake plus planning-only byte arithmetic.",
            "No scorer invocation, no CUDA dependency, and no remote/GPU dispatch.",
            "Static rate-term deltas are not component or score evidence.",
        ],
        "archive": asdict(archive),
        "payload_split": asdict(split),
        "qma9": asdict(qma9),
        "split_constants": constants,
        "references": [asdict(ref) for ref in references],
        "static_break_even": [asdict(item) for item in break_even],
        "component_byte_deltas": component_deltas,
        "transfer_recommendations": transfer_recommendations(split, references),
        "optional_cpp_decode_hash_validation": decode_hash,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR81_ARCHIVE)
    parser.add_argument("--split-constants-py", type=Path, default=DEFAULT_PR81_INFLATE)
    parser.add_argument("--pr79-profile-json", type=Path, default=DEFAULT_PR79_PROFILE)
    parser.add_argument("--pr79-s2-json", type=Path, default=DEFAULT_PR79_S2_PROFILE)
    parser.add_argument("--range-mask-codec-cpp", type=Path, default=DEFAULT_RANGE_MASK_CODEC)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--try-cpp-decode-hash", action="store_true")
    parser.add_argument("--cpp-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = build_profile(
        archive_path=args.archive,
        split_constants_path=args.split_constants_py,
        pr79_profile_path=args.pr79_profile_json,
        pr79_s2_profile_path=args.pr79_s2_json,
        range_mask_codec_cpp=args.range_mask_codec_cpp,
        try_cpp_decode_hash=args.try_cpp_decode_hash,
        cpp_timeout_seconds=args.cpp_timeout_seconds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(_json_bytes(profile))
    print(f"wrote {args.output_json}")
    print(f"evidence_grade={profile['evidence_grade']} score_claim={profile['score_claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
