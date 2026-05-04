#!/usr/bin/env python3
"""Build a C-063 breakthrough candidate matrix.

This is a local, non-dispatch planner/builder for the public-floor basin.  It
builds only closed byte-screen archive candidates where the current runtime
already has a decoder, and it emits H100 dispatch specs for candidates that
need GPU-side construction such as pose-manifold line search.

All outputs are non-promotable and keep ``score_claim=false`` until exact CUDA
auth eval runs on identical archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRODUCER = "experiments/build_c063_breakthrough_candidate_matrix.py"
SCHEMA_VERSION = 1
EXPECTED_SAMPLES = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c063_breakthrough_candidate_matrix_20260502"
DEFAULT_C063_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z"
)
DEFAULT_C063_ARCHIVE = DEFAULT_C063_DIR / "archive.zip"
DEFAULT_C063_EVAL = DEFAULT_C063_DIR / "contest_auth_eval.json"
DEFAULT_C063_TRACE = DEFAULT_C063_DIR / "component_trace.json"
DEFAULT_POSE_PLAN = (
    REPO_ROOT / "experiments/results/pose_atom_plan_c063_h100nvl_20260502/pose_atom_policies.json"
)
DEFAULT_PR65_ARCHIVE = REPO_ROOT / "reports/raw/leaderboard_intel_20260501/pr65_archive.zip"
DEFAULT_PR67_ARCHIVE = REPO_ROOT / "reports/raw/leaderboard_intel_20260501/pr67_archive.zip"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_QPOST_EVIDENCE = (
    REPO_ROOT
    / "experiments/results/vast_harvest/archive_eval_qpost_r13_bias_20260502T0158Z/"
    "contest_auth_eval.json",
    REPO_ROOT
    / "experiments/results/vast_harvest/archive_eval_qpost_r13_region_20260502T0158Z/"
    "contest_auth_eval.json",
    REPO_ROOT
    / "experiments/results/vast_harvest/archive_eval_qpost_r13_post_20260502T0158Z/"
    "contest_auth_eval.json",
    REPO_ROOT
    / "experiments/results/vast_harvest/archive_eval_c059_pairgated_qpost_top16_20260502/"
    "contest_auth_eval.json",
)

DEFAULT_MQZ_POLICIES = (
    "component-aware-v1:frame2_all64",
    "component-aware-v1:frame2_block2_pre64",
)
DEFAULT_QPOST_SPECS = (
    ("c063_qpost_bias_top032", "bias", 32),
    ("c063_qpost_region_bias_top032", "region,bias", 32),
    ("c063_qpost_bias_top016", "bias", 16),
)

CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NON_PROMOTABLE_WARNING = (
    "This artifact is byte-screen/planning evidence only. It cannot promote, "
    "rank, kill, retire, or support a score claim until exact CUDA auth eval "
    "runs on the identical archive bytes."
)


class BreakthroughMatrixError(ValueError):
    """Raised when candidate matrix inputs fail custody or schema checks."""


@dataclass(frozen=True)
class QPostSpec:
    candidate_id: str
    include_streams: tuple[str, ...]
    top_pair_count: int | None


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BreakthroughMatrixError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BreakthroughMatrixError(f"{label} must be a JSON object: {path}")
    return payload


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BreakthroughMatrixError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise BreakthroughMatrixError(f"{field} must be finite")
    return out


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BreakthroughMatrixError(f"{field} must be an integer")
    return int(value)


def _file_meta(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    meta: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        meta["size_bytes"] = resolved.stat().st_size
        meta["sha256"] = _sha256_file(resolved)
    return meta


def _archive_members_meta(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        return {
            info.filename: {
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "sha256": _sha256_bytes(zf.read(info)),
            }
            for info in zf.infolist()
            if not info.is_dir()
        }


def _read_single_member(path: Path, *, member: str = "p") -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member]:
            raise BreakthroughMatrixError(
                f"expected single archive member {member!r} in {path}; got {names!r}"
            )
        return zf.read(infos[0])


def _write_single_member_stored_zip(path: Path, *, member: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _looks_like_renderer_payload(data: bytes) -> bool:
    return (
        data.startswith(b"QZS3")
        or data.startswith(b"MQZ1")
        or data.startswith(b"QFAI")
        or data.startswith(b"\x80\x02")
        or data.startswith(b"PK\x03\x04")
    )


def _parse_pr64_len_table_raw_payload(raw_payload: bytes) -> tuple[bytes, bytes, bytes]:
    header_size = struct.calcsize("<III")
    if len(raw_payload) < header_size:
        raise BreakthroughMatrixError("frontier Brotli payload is too short for PR64 len table")
    first_len, second_len, pose_len = struct.unpack_from("<III", raw_payload, 0)
    expected = header_size + first_len + second_len + pose_len
    if min(first_len, second_len, pose_len) <= 0 or expected != len(raw_payload):
        raise BreakthroughMatrixError(
            "frontier Brotli payload is not a PR64 len table: "
            f"lengths=({first_len}, {second_len}, {pose_len}) raw={len(raw_payload)}"
        )
    first_start = header_size
    first_end = first_start + first_len
    second_end = first_end + second_len
    first = raw_payload[first_start:first_end]
    second = raw_payload[first_end:second_end]
    pose = raw_payload[second_end:]
    if _looks_like_renderer_payload(first) and not _looks_like_renderer_payload(second):
        renderer_raw, mask_raw = first, second
    elif _looks_like_renderer_payload(second) and not _looks_like_renderer_payload(first):
        mask_raw, renderer_raw = first, second
    else:
        raise BreakthroughMatrixError("could not identify mask/model order in PR64 len table")
    if not pose.startswith(b"QP1"):
        raise BreakthroughMatrixError(
            "C-063 line-search source builder expected QP1 pose payload; "
            f"got magic={pose[:4]!r}"
        )
    return mask_raw, renderer_raw, pose


def _load_unpacker_module():
    spec = importlib.util.spec_from_file_location(
        "pact_breakthrough_matrix_unpacker", UNPACKER_PATH
    )
    if spec is None or spec.loader is None:
        raise BreakthroughMatrixError(f"cannot load unpacker module: {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixedslice_source_records(
    *,
    archive_path: Path,
    metadata_path: Path,
    provenance_path: Path,
    source_archive: Path,
    source_payload: bytes,
    blob: bytes,
    mask_br: bytes,
    model_br: bytes,
    pose_br: bytes,
    decoded_members: Mapping[str, bytes],
    payload_format: str,
    source_raw_payload_sha256: str | None,
    brotli_quality: int | None,
) -> dict[str, Any]:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "evidence_grade": "empirical_metadata_for_line_search",
        "payload_format": payload_format,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "blob_bytes": len(blob),
        "blob_sha256": _sha256_bytes(blob),
        "mask_br_bytes": len(mask_br),
        "mask_br_sha256": _sha256_bytes(mask_br),
        "model_br_bytes": len(model_br),
        "model_br_sha256": _sha256_bytes(model_br),
        "pose_br_bytes": len(pose_br),
        "pose_br_sha256": _sha256_bytes(pose_br),
        "mask_uncompressed_bytes": len(decoded_members["masks.mkv"]),
        "model_uncompressed_bytes": len(decoded_members["renderer.bin"]),
        "pose_uncompressed_bytes": len(decoded_members["optimized_poses.bin"]),
        "pose_codec": "qp1",
        "source_archive": str(source_archive),
        "source_archive_sha256": _sha256_file(source_archive),
        "source_payload_sha256": _sha256_bytes(source_payload),
        "source_raw_payload_sha256": source_raw_payload_sha256,
        "decoded_member_sha256": {
            "masks.mkv": _sha256_bytes(decoded_members["masks.mkv"]),
            "renderer.bin": _sha256_bytes(decoded_members["renderer.bin"]),
            "optimized_poses.bin": _sha256_bytes(decoded_members["optimized_poses.bin"]),
        },
        "brotli_quality": brotli_quality,
    }
    _write_json(metadata_path, metadata)
    _write_json(
        provenance_path,
        {
            "schema_version": SCHEMA_VERSION,
            "tool": PRODUCER,
            "score_claim": False,
            "source_archive": _file_meta(source_archive),
            "output_archive": _file_meta(archive_path),
            "output_metadata": _file_meta(metadata_path),
            "payload_format": payload_format,
            "determinism": {
                "zip_compress_type": "ZIP_STORED",
                "zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
                "zip_permissions": "0644",
                "single_member_order": ["p"],
                "brotli_quality": brotli_quality,
            },
        },
    )
    return metadata


def _build_line_search_source_from_public_fixedslice(
    *,
    source_archive: Path,
    source_payload: bytes,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    unpacker = _load_unpacker_module()
    try:
        header, decoded_members = unpacker._parse_payload(source_payload)  # noqa: SLF001
    except Exception as exc:
        raise BreakthroughMatrixError(
            "frontier member is neither a Brotli PR64 len-table payload nor "
            "a recognized public fixed-slice payload"
        ) from exc
    if header.get("payload_format") != "public_pr67_qzs3_qp1_fixed_slices":
        raise BreakthroughMatrixError(
            "default fixed-slice line-search source generation only supports "
            "public_pr67_qzs3_qp1_fixed_slices; pass explicit line-search "
            f"metadata for payload_format={header.get('payload_format')!r}"
        )
    member_by_name = {
        str(item.get("name")): item
        for item in header.get("members", [])
        if isinstance(item, Mapping)
    }
    try:
        mask_n = int(member_by_name["masks.mkv"]["bytes"])
        model_n = int(member_by_name["renderer.bin"]["bytes"])
        pose_n = int(member_by_name["optimized_poses.bin"]["bytes"])
    except Exception as exc:
        raise BreakthroughMatrixError("public fixed-slice header lacks segment byte counts") from exc
    expected = mask_n + model_n + pose_n
    if expected != len(source_payload):
        raise BreakthroughMatrixError(
            "public fixed-slice byte counts do not sum to source payload: "
            f"{expected} != {len(source_payload)}"
        )
    mask_br = source_payload[:mask_n]
    model_br = source_payload[mask_n : mask_n + model_n]
    pose_br = source_payload[mask_n + model_n :]
    archive_path = output_dir / "line_search_source_c067_fixedslice" / "archive.zip"
    metadata_path = archive_path.with_name("metadata.json")
    provenance_path = archive_path.with_name("build_provenance.json")
    _write_single_member_stored_zip(archive_path, member="p", payload=source_payload)
    metadata = _write_fixedslice_source_records(
        archive_path=archive_path,
        metadata_path=metadata_path,
        provenance_path=provenance_path,
        source_archive=source_archive,
        source_payload=source_payload,
        blob=source_payload,
        mask_br=mask_br,
        model_br=model_br,
        pose_br=pose_br,
        decoded_members=decoded_members,
        payload_format="frontier_public_pr67_fixedslice_reused_for_line_search",
        source_raw_payload_sha256=None,
        brotli_quality=None,
    )
    return archive_path, metadata_path, metadata


def _build_line_search_source_from_frontier(
    *,
    source_archive: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build a C-063-equivalent PR67 fixed-slice source for pose line search."""
    source_payload = _read_single_member(source_archive)
    try:
        raw_payload = brotli.decompress(source_payload)
    except brotli.error as exc:
        try:
            return _build_line_search_source_from_public_fixedslice(
                source_archive=source_archive,
                source_payload=source_payload,
                output_dir=output_dir,
            )
        except BreakthroughMatrixError as fixedslice_exc:
            raise fixedslice_exc from exc
    mask_raw, renderer_raw, pose_raw = _parse_pr64_len_table_raw_payload(raw_payload)
    mask_br = brotli.compress(mask_raw, quality=11)
    model_br = brotli.compress(renderer_raw, quality=11)
    pose_br = brotli.compress(pose_raw, quality=11)
    blob = mask_br + model_br + pose_br
    archive_path = output_dir / "line_search_source_c063_fixedslice" / "archive.zip"
    metadata_path = archive_path.with_name("metadata.json")
    provenance_path = archive_path.with_name("build_provenance.json")
    _write_single_member_stored_zip(archive_path, member="p", payload=blob)
    metadata = _write_fixedslice_source_records(
        archive_path=archive_path,
        metadata_path=metadata_path,
        provenance_path=provenance_path,
        source_archive=source_archive,
        source_payload=source_payload,
        blob=blob,
        mask_br=mask_br,
        model_br=model_br,
        pose_br=pose_br,
        decoded_members={
            "masks.mkv": mask_raw,
            "renderer.bin": renderer_raw,
            "optimized_poses.bin": pose_raw,
        },
        payload_format="c063_frontier_pr64_len_table_to_pr67_fixedslice",
        source_raw_payload_sha256=_sha256_bytes(raw_payload),
        brotli_quality=11,
    )
    return archive_path, metadata_path, metadata


def _load_frontier_eval(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="frontier_eval")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BreakthroughMatrixError("frontier_eval.provenance must be an object")
    n_samples = _int_value(payload.get("n_samples"), field="frontier_eval.n_samples")
    if n_samples != EXPECTED_SAMPLES:
        raise BreakthroughMatrixError(
            f"frontier_eval.n_samples must be {EXPECTED_SAMPLES}, got {n_samples}"
        )
    if provenance.get("device") != "cuda":
        raise BreakthroughMatrixError("frontier_eval.provenance.device must be cuda")
    archive_sha = provenance.get("archive_sha256")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise BreakthroughMatrixError("frontier_eval.provenance.archive_sha256 must be SHA-256")
    return {
        "path": str(path.resolve()),
        "file": _file_meta(path),
        "archive_size_bytes": _int_value(
            payload.get("archive_size_bytes"),
            field="frontier_eval.archive_size_bytes",
        ),
        "archive_sha256": archive_sha,
        "score_recomputed_from_components": _finite_float(
            payload.get("score_recomputed_from_components"),
            field="frontier_eval.score_recomputed_from_components",
        ),
        "avg_posenet_dist": _finite_float(
            payload.get("avg_posenet_dist"),
            field="frontier_eval.avg_posenet_dist",
        ),
        "avg_segnet_dist": _finite_float(
            payload.get("avg_segnet_dist"),
            field="frontier_eval.avg_segnet_dist",
        ),
        "n_samples": n_samples,
        "gpu_model": provenance.get("gpu_model"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "device": provenance.get("device"),
        "sys_argv": provenance.get("sys_argv"),
    }


def _load_trace_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="frontier_component_trace")
    if payload.get("score_claim") is not False:
        raise BreakthroughMatrixError("frontier_component_trace.score_claim must be false")
    cross = payload.get("contest_auth_eval_cross_check")
    if not isinstance(cross, Mapping) or cross.get("all_match") is not True:
        raise BreakthroughMatrixError("frontier_component_trace must cross-check auth eval")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise BreakthroughMatrixError("frontier_component_trace.samples must have 600 rows")
    hard_pairs: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pair_index = sample.get("pair_index")
        if not isinstance(pair_index, int):
            continue
        pose = float(sample.get("posenet_dist", 0.0))
        seg = float(sample.get("segnet_dist", 0.0))
        hard_pairs.append(
            {
                "pair_index": pair_index,
                "posenet_dist": pose,
                "segnet_dist": seg,
                "hardness": pose + seg,
            }
        )
    hard_pairs.sort(key=lambda item: (-item["hardness"], item["pair_index"]))
    return {
        "file": _file_meta(path),
        "top_pair_indices": [item["pair_index"] for item in hard_pairs[:64]],
    }


def _load_pose_plan(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="pose_atom_plan")
    if payload.get("score_claim") is not False:
        raise BreakthroughMatrixError("pose_atom_plan.score_claim must be false")
    policies = payload.get("recommended_policies")
    if not isinstance(policies, list) or not policies:
        raise BreakthroughMatrixError("pose_atom_plan.recommended_policies must be non-empty")
    normalized: list[dict[str, Any]] = []
    for item in policies:
        if not isinstance(item, Mapping):
            continue
        selected = item.get("selected_pair_indices")
        if not isinstance(selected, list) or not all(isinstance(x, int) for x in selected):
            continue
        normalized.append(
            {
                "policy_name": str(item.get("policy_name") or f"pose_atoms_top{len(selected):03d}"),
                "selected_pair_indices": [int(x) for x in selected],
                "pair_count": len(selected),
                "charged_bytes_estimate": float(item.get("charged_bytes_estimate") or 0.0),
                "expected_score_saved_sum": float(item.get("expected_score_saved_sum") or 0.0),
                "measured_delta_atom_count": int(item.get("measured_delta_atom_count") or 0),
            }
        )
    if not normalized:
        raise BreakthroughMatrixError("pose_atom_plan has no usable recommended policies")
    normalized.sort(
        key=lambda item: (
            -item["expected_score_saved_sum"],
            item["charged_bytes_estimate"],
            item["policy_name"],
        )
    )
    top_pairs = normalized[0]["selected_pair_indices"]
    return {
        "file": _file_meta(path),
        "frontier": payload.get("frontier"),
        "recommended_policies": normalized,
        "top_policy": normalized[0],
        "top_pair_indices": top_pairs,
    }


def _load_evidence_summaries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        meta = _file_meta(path)
        if not meta["exists"]:
            summaries.append(meta)
            continue
        try:
            payload = _read_json(path, label=f"evidence:{path}")
        except BreakthroughMatrixError:
            summaries.append({**meta, "json_parseable": False})
            continue
        provenance = payload.get("provenance") if isinstance(payload.get("provenance"), Mapping) else {}
        summaries.append(
            {
                **meta,
                "json_parseable": True,
                "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
                "archive_size_bytes": payload.get("archive_size_bytes"),
                "avg_posenet_dist": payload.get("avg_posenet_dist"),
                "avg_segnet_dist": payload.get("avg_segnet_dist"),
                "archive_sha256": provenance.get("archive_sha256"),
                "gpu_model": provenance.get("gpu_model"),
                "device": provenance.get("device"),
            }
        )
    return summaries


def _parse_qpost_spec(raw: str) -> QPostSpec:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) not in (2, 3):
        raise BreakthroughMatrixError(
            "qpost spec must be candidate_id:stream,stream[:top_pair_count]"
        )
    streams = tuple(part.strip() for part in parts[1].split(",") if part.strip())
    if not streams:
        raise BreakthroughMatrixError(f"qpost spec {raw!r} has no streams")
    count = int(parts[2]) if len(parts) == 3 and parts[2] else None
    if count is not None and count <= 0:
        raise BreakthroughMatrixError(f"qpost spec {raw!r} top_pair_count must be positive")
    return QPostSpec(candidate_id=parts[0], include_streams=streams, top_pair_count=count)


def _default_qpost_specs() -> tuple[QPostSpec, ...]:
    return tuple(
        QPostSpec(name, tuple(part for part in streams.split(",") if part), count)
        for name, streams, count in DEFAULT_QPOST_SPECS
    )


def _candidate_common(
    *,
    candidate_id: str,
    family: str,
    archive_path: Path,
    frontier: Mapping[str, Any],
    rationale: str,
    risk: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    archive_bytes = archive_path.stat().st_size
    byte_delta = archive_bytes - int(frontier["archive_size_bytes"])
    min_gain = max(0.0, byte_delta * RATE_SCORE_PER_BYTE)
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "family": family,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_byte_screen_only_until_cuda_auth_eval",
        "exact_evaluable_archive": True,
        "required_eval": CUDA_AUTH_EVAL_PATH,
        "output_archive": str(archive_path.resolve()),
        "output_archive_bytes": archive_bytes,
        "output_archive_sha256": _sha256_file(archive_path),
        "archive_byte_delta_vs_c063": byte_delta,
        "formula_rate_score_delta_vs_c063": byte_delta * RATE_SCORE_PER_BYTE,
        "minimum_component_score_gain_to_beat_c063": min_gain,
        "archive_members": _archive_members_meta(archive_path),
        "rationale": rationale,
        "risk": risk,
    }
    if extra:
        payload.update(extra)
    return payload


def _build_mqz_candidates(
    *,
    source_archive: Path,
    output_dir: Path,
    frontier: Mapping[str, Any],
    policies: Sequence[str],
    source_evidence_path: Path,
) -> list[dict[str, Any]]:
    from experiments import build_mixed_qzs_block_candidate as mixed

    candidates: list[dict[str, Any]] = []
    for spec in policies:
        policy = mixed.parse_block_policy(spec)
        candidate_id = f"c063_mqz_{policy.name}"
        candidate_dir = output_dir / candidate_id
        archive = candidate_dir / "archive.zip"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        meta = mixed.build_candidate_for_policy(
            source_archive,
            archive,
            policy=policy,
            source_evidence_path=source_evidence_path,
        )
        _write_json(candidate_dir / "build_provenance.json", meta)
        candidates.append(
            _candidate_common(
                candidate_id=candidate_id,
                family="archive_side_qzs3_mqz1_block_policy",
                archive_path=archive,
                frontier=frontier,
                rationale=(
                    "Closed archive using the reviewed MQZ1/QZS3 runtime path; "
                    "only frame2-side policy variants are kept because global "
                    "larger blocks have exact PoseNet-collapse negatives."
                ),
                risk="medium: renderer quantization changes score components and needs H100 screen",
                extra={
                    "builder_provenance": str((candidate_dir / "build_provenance.json").resolve()),
                    "policy": meta.get("block_policy", {}).get("policy"),
                    "renderer": meta.get("renderer"),
                },
            )
        )
    return candidates


def _build_qpost_candidates(
    *,
    source_archive: Path,
    pr65_archive: Path,
    output_dir: Path,
    frontier: Mapping[str, Any],
    qpost_specs: Sequence[QPostSpec],
    pose_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from experiments import build_qzs3_postprocess_candidate as qpost

    top_pairs = list(pose_plan["top_pair_indices"])
    candidates: list[dict[str, Any]] = []
    for spec in qpost_specs:
        candidate_dir = output_dir / spec.candidate_id
        archive = candidate_dir / "archive.zip"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        pair_indices: tuple[int, ...] | None = None
        if spec.top_pair_count is not None:
            pair_indices = tuple(top_pairs[: spec.top_pair_count])
        meta = qpost.build_candidate(
            source_archive,
            pr65_archive,
            archive,
            include_streams=spec.include_streams,
            pair_indices=pair_indices,
        )
        _write_json(candidate_dir / "build_provenance.json", meta)
        candidates.append(
            _candidate_common(
                candidate_id=spec.candidate_id,
                family="archive_side_pr65_qpost_atoms",
                archive_path=archive,
                frontier=frontier,
                rationale=(
                    "Counted PR65 qpost sidecar filtered to C-063 pose-waterfill "
                    "hard pairs; included as a closed archive screen, not as a "
                    "promotion recommendation."
                ),
                risk=(
                    "high: prior PR65 qpost and pairgated screens regressed; "
                    "kept for custody-complete matrix coverage only"
                ),
                extra={
                    "builder_provenance": str((candidate_dir / "build_provenance.json").resolve()),
                    "include_streams": list(spec.include_streams),
                    "selected_pair_indices": list(pair_indices or ()),
                    "pair_count": len(pair_indices or ()),
                    "qpost_member": meta.get("members", {}).get("qpost.bin"),
                },
            )
        )
    return candidates


def _pose_dispatch_rows(
    *,
    frontier: Mapping[str, Any],
    pose_plan: Mapping[str, Any],
    source_archive: Path,
    line_search_source_archive: Path,
    line_search_source_metadata: Path,
    pr67_archive: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in pose_plan["recommended_policies"]:
        if policy["pair_count"] not in (32, 48):
            continue
        charged_bytes = float(policy["charged_bytes_estimate"])
        expected_saved = float(policy["expected_score_saved_sum"])
        expected_rate_cost = charged_bytes * RATE_SCORE_PER_BYTE
        candidate_id = f"c063_pose_waterfill_{policy['pair_count']:03d}_pr67_pr65_basis"
        candidate_dir = output_dir / candidate_id
        candidate_archive = candidate_dir / "archive.zip"
        metadata_path = candidate_dir / "metadata.json"
        pair_csv = ",".join(str(x) for x in policy["selected_pair_indices"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": "h100_pose_renderer_subspace_waterfill_dispatch_matrix",
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": "planning_only_until_closed_archive_cuda_auth_eval",
                "exact_evaluable_archive": False,
                "build_required_on_h100": True,
                "requires_dispatch_claim_before_remote_job": True,
                "source_archive": str(source_archive.resolve()),
                "line_search_source_archive": str(line_search_source_archive.resolve()),
                "line_search_source_metadata": str(line_search_source_metadata.resolve()),
                "post_search_lossless_repack_recommended": True,
                "public_anatomy_reference_archive": str(pr67_archive.resolve()),
                "output_archive": str(candidate_archive.resolve()),
                "output_metadata": str(metadata_path.resolve()),
                "selected_pair_indices": policy["selected_pair_indices"],
                "pair_count": policy["pair_count"],
                "charged_bytes_estimate": charged_bytes,
                "expected_component_score_saved": expected_saved,
                "expected_rate_score_cost": expected_rate_cost,
                "expected_net_score_delta_vs_c063": expected_rate_cost - expected_saved,
                "h100_gate_min_score_gain_vs_c063_before_t4": 0.00025,
                "exact_eval_command_after_build": (
                    ".venv/bin/python -u experiments/contest_auth_eval.py "
                    f"--archive {candidate_archive} --device cuda "
                    "--output-json "
                    f"{candidate_dir / 'contest_auth_eval.json'}"
                ),
                "build_command_no_dispatch_claim_included": (
                    ".venv/bin/python -u experiments/line_search_pose_refinement.py "
                    f"--archive-path {line_search_source_archive} "
                    f"--metadata-path {line_search_source_metadata} "
                    f"--output-path {candidate_archive} "
                    f"--output-metadata {metadata_path} "
                    "--posenet-path upstream/models/posenet.safetensors "
                    "--gt-mkv upstream/videos/0.mkv --device cuda:0 --batch-size 16 "
                    "--candidate-chunk 32 --basis-delta-sets 'pair_window:1,2,3;dct:1,2' "
                    f"--basis-pair-indices '{pair_csv}' "
                    "--basis-window-radius 1 --passes 2 --progress-every-candidates 64"
                ),
                "rationale": (
                    "Highest local EV because the C-063 pose atom plan predicts "
                    "component movement larger than byte-only polish, while "
                    "CRF/RPK1 mask repair and PR65 qpost have fresh exact negatives."
                ),
                "risk": (
                    "medium-high: H100/T4 drift is known; exact H100 gain must "
                    "clear the gate before spending T4/equivalent confirmation"
                ),
                "frontier_score_reference": frontier["score_recomputed_from_components"],
            }
        )
    rows.sort(
        key=lambda item: (
            item["expected_net_score_delta_vs_c063"],
            -item["pair_count"],
            item["candidate_id"],
        )
    )
    return rows


def build_breakthrough_matrix(
    *,
    source_archive: Path = DEFAULT_C063_ARCHIVE,
    frontier_eval: Path = DEFAULT_C063_EVAL,
    frontier_component_trace: Path = DEFAULT_C063_TRACE,
    pose_plan_path: Path = DEFAULT_POSE_PLAN,
    line_search_source_archive: Path | None = None,
    line_search_source_metadata: Path | None = None,
    pr65_archive: Path = DEFAULT_PR65_ARCHIVE,
    pr67_archive: Path = DEFAULT_PR67_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mqz_policies: Sequence[str] = DEFAULT_MQZ_POLICIES,
    qpost_specs: Sequence[QPostSpec] = _default_qpost_specs(),
    qpost_evidence_paths: Sequence[Path] = DEFAULT_QPOST_EVIDENCE,
    build_archive_candidates: bool = True,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    if not source_archive.is_file():
        raise BreakthroughMatrixError(f"source archive missing: {source_archive}")
    frontier = _load_frontier_eval(frontier_eval.resolve())
    source_sha = _sha256_file(source_archive)
    if source_sha != frontier["archive_sha256"]:
        raise BreakthroughMatrixError(
            "source archive SHA does not match frontier eval: "
            f"{source_sha} != {frontier['archive_sha256']}"
        )
    if source_archive.stat().st_size != frontier["archive_size_bytes"]:
        raise BreakthroughMatrixError("source archive bytes do not match frontier eval")

    output_dir.mkdir(parents=True, exist_ok=True)
    if (line_search_source_archive is None) != (line_search_source_metadata is None):
        raise BreakthroughMatrixError(
            "--line-search-source-archive and --line-search-source-metadata must be supplied together"
        )
    if line_search_source_archive is None:
        (
            line_search_source_archive,
            line_search_source_metadata,
            line_search_source_generation,
        ) = _build_line_search_source_from_frontier(
            source_archive=source_archive,
            output_dir=output_dir,
        )
    else:
        line_search_source_archive = line_search_source_archive.resolve()
        line_search_source_metadata = line_search_source_metadata.resolve()
        if not line_search_source_archive.is_file():
            raise BreakthroughMatrixError(
                f"line-search source archive missing: {line_search_source_archive}"
            )
        if not line_search_source_metadata.is_file():
            raise BreakthroughMatrixError(
                f"line-search source metadata missing: {line_search_source_metadata}"
            )
        line_search_source_generation = {
            "score_claim": False,
            "evidence_grade": "external_line_search_source_supplied",
            "archive_path": str(line_search_source_archive),
            "metadata_path": str(line_search_source_metadata),
        }
    trace = _load_trace_summary(frontier_component_trace.resolve())
    pose_plan = _load_pose_plan(pose_plan_path.resolve())
    qpost_evidence = _load_evidence_summaries(tuple(path.resolve() for path in qpost_evidence_paths))

    archive_candidates: list[dict[str, Any]] = []
    if build_archive_candidates:
        archive_candidates.extend(
            _build_mqz_candidates(
                source_archive=source_archive,
                output_dir=output_dir / "archives",
                frontier=frontier,
                policies=mqz_policies,
                source_evidence_path=frontier_eval.resolve(),
            )
        )
        if qpost_specs:
            archive_candidates.extend(
                _build_qpost_candidates(
                    source_archive=source_archive,
                    pr65_archive=pr65_archive.resolve(),
                    output_dir=output_dir / "archives",
                    frontier=frontier,
                    qpost_specs=qpost_specs,
                    pose_plan=pose_plan,
                )
            )

    archive_candidates.sort(
        key=lambda item: (
            item["minimum_component_score_gain_to_beat_c063"],
            0 if item["family"] == "archive_side_qzs3_mqz1_block_policy" else 1,
            item["output_archive_bytes"],
            item["candidate_id"],
        )
    )
    dispatch_rows = _pose_dispatch_rows(
        frontier=frontier,
        pose_plan=pose_plan,
        source_archive=source_archive,
        line_search_source_archive=line_search_source_archive,
        line_search_source_metadata=line_search_source_metadata,
        pr67_archive=pr67_archive.resolve(),
        output_dir=output_dir / "h100_dispatch_matrix",
    )

    recommended = dispatch_rows[0] if dispatch_rows else (archive_candidates[0] if archive_candidates else None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "non_promotable_warning": NON_PROMOTABLE_WARNING,
        "required_promotion_eval": CUDA_AUTH_EVAL_PATH,
        "output_dir": str(output_dir),
        "frontier_label": "C-063",
        "frontier": frontier,
        "source_archive": _file_meta(source_archive),
        "line_search_source_archive": _file_meta(line_search_source_archive),
        "line_search_source_metadata": _file_meta(line_search_source_metadata),
        "line_search_source_generation": line_search_source_generation,
        "frontier_component_trace": trace,
        "pose_atom_plan": {
            "file": pose_plan["file"],
            "top_policy": pose_plan["top_policy"],
            "recommended_policies": pose_plan["recommended_policies"],
        },
        "public_anatomy_inputs": {
            "pr65_archive": _file_meta(pr65_archive.resolve()),
            "pr67_archive": _file_meta(pr67_archive.resolve()),
        },
        "negative_or_cautionary_evidence": {
            "qpost_exact_screens": qpost_evidence,
            "charged_mask_crf_repair_status": (
                "fresh H100 exact screen regressed to about 2.32187 despite "
                "clean AMR1 application; broad CRF/RPK1 repair is not ranked"
            ),
            "global_qzs_block_status": (
                "larger global QZS blocks have exact PoseNet-collapse negatives; "
                "only frame2-protected MQZ screens are included locally"
            ),
        },
        "archive_candidate_count": len(archive_candidates),
        "archive_candidates": archive_candidates,
        "h100_dispatch_matrix": dispatch_rows,
        "recommended_first_h100_candidate": recommended,
    }
    _write_json(output_dir / "c063_breakthrough_candidate_matrix.json", payload)
    _write_json(
        output_dir / "exact_eval_recommendation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "score_claim": False,
            "promotion_eligible": False,
            "recommended_first_h100_candidate": recommended,
        },
    )
    _write_json(
        output_dir / "archive_candidate_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "score_claim": False,
            "promotion_eligible": False,
            "archive_candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "output_archive": item["output_archive"],
                    "output_archive_bytes": item["output_archive_bytes"],
                    "output_archive_sha256": item["output_archive_sha256"],
                    "archive_byte_delta_vs_c063": item["archive_byte_delta_vs_c063"],
                }
                for item in archive_candidates
            ],
        },
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_C063_ARCHIVE)
    parser.add_argument("--frontier-eval", type=Path, default=DEFAULT_C063_EVAL)
    parser.add_argument("--frontier-component-trace", type=Path, default=DEFAULT_C063_TRACE)
    parser.add_argument("--pose-plan", type=Path, default=DEFAULT_POSE_PLAN)
    parser.add_argument(
        "--line-search-source-archive",
        type=Path,
        default=None,
        help=(
            "Raw fixed-slice archive compatible with line_search_pose_refinement.py. "
            "Default: derive a C-063-equivalent fixed-slice source from the "
            "lossless outer-Brotli frontier archive."
        ),
    )
    parser.add_argument(
        "--line-search-source-metadata",
        type=Path,
        default=None,
        help="Metadata matching --line-search-source-archive.",
    )
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--pr67-archive", type=Path, default=DEFAULT_PR67_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mqz-policy",
        action="append",
        default=None,
        help="Repeatable MQZ/QZS block policy. Defaults to protected frame2 policies.",
    )
    parser.add_argument(
        "--qpost-spec",
        action="append",
        default=None,
        help="Repeatable qpost candidate_id:stream,stream[:top_pair_count] spec.",
    )
    parser.add_argument(
        "--qpost-evidence",
        action="append",
        type=Path,
        default=None,
        help="Repeatable qpost exact-screen JSON to fingerprint as cautionary evidence.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the dispatch matrix without building local archive candidates.",
    )
    args = parser.parse_args(argv)

    payload = build_breakthrough_matrix(
        source_archive=args.source_archive,
        frontier_eval=args.frontier_eval,
        frontier_component_trace=args.frontier_component_trace,
        pose_plan_path=args.pose_plan,
        line_search_source_archive=args.line_search_source_archive,
        line_search_source_metadata=args.line_search_source_metadata,
        pr65_archive=args.pr65_archive,
        pr67_archive=args.pr67_archive,
        output_dir=args.output_dir,
        mqz_policies=tuple(args.mqz_policy) if args.mqz_policy else DEFAULT_MQZ_POLICIES,
        qpost_specs=(
            tuple(_parse_qpost_spec(spec) for spec in args.qpost_spec)
            if args.qpost_spec
            else _default_qpost_specs()
        ),
        qpost_evidence_paths=tuple(args.qpost_evidence or DEFAULT_QPOST_EVIDENCE),
        build_archive_candidates=not args.plan_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
