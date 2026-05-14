#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic PR82/Henosis atom-transfer candidates.

This tool is local-only.  It mines PR82 postprocess and P1D1 pose-control
atoms, transfers only runtime-compatible charged atoms onto a PR79/S2 or
QZS3-family single-payload archive, and emits fail-closed dispatch gates.
It never claims score and never dispatches GPU work.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.archive_byte_profile import profile_archive
from tac.henosis_pr82_transfer import (
    QPOST_STREAM_NAMES,
    decode_control_arrays,
    decode_pr82_p1d1_pose,
    decode_randmulti_activity,
    decode_randmulti_groups,
    decode_randmulti_qrm1,
    encode_randmulti_nm2,
    encode_randmulti_qrm1,
    encode_qpost,
    filter_qpost_streams_to_pairs,
    parse_pr82_bundle,
    parse_replay_contract,
    pose_velocity_atom_ranking,
    randmulti_qrm1_parity_profile,
    qpost_stream_summary,
    randmulti_group_qps1_nm2_compatible,
    randmulti_group_summary,
    rank_pairs_by_activity,
    sha256_bytes,
    sha256_path,
    summarize_pair_activity,
)
from tac.qp1_pose_codec import decode_qp1, encode_qp1


TOOL = "experiments/build_pr82_henosis_atom_transfer_candidates.py"
SCHEMA = "pr82_henosis_atom_transfer_candidates_v1"
MANIFEST_SCHEMA = "pr82_henosis_atom_transfer_manifest_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
CURRENT_A_PLUS_PLUS_T4_FRONTIER = 0.31453355357318635
SUB314_TARGET = 0.314
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex"
DEFAULT_PR82_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_REPLAY_INFLATE = DEFAULT_INTAKE_DIR / "replay_submission/inflate.py"
DEFAULT_PR79_S2_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/"
    "pr79_s2_fixed_adaptive_actions/archive.zip"
)
DEFAULT_PR79_S2_EXACT_JSON = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr79_s2_fixed_adaptive_actions_t4_20260503T173023Z/contest_auth_eval.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr82_henosis_atom_transfer_20260503_codex"
EXPECTED_PR82_SHA256 = "a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4"
EXPECTED_PR79_S2_SHA256 = "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_QPOST_STREAMS = ("post", "shift", "frac", "frac2", "frac3", "bias", "region")
DEFAULT_QPOST_TOPKS = (8, 16, 32)
DEFAULT_POSE_TOPKS = (8, 16, 32)
DEFAULT_RANDMULTI_TOPKS = (1, 4, 8)
NON_POSE_MEMBERS = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin")


class Pr82TransferBuildError(ValueError):
    """Raised when PR82 transfer candidate construction fails a guard."""


@dataclass(frozen=True)
class SourceArchive:
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    raw_segments: dict[str, bytes]
    decoded_members: dict[str, bytes]
    member_summary: dict[str, dict[str, Any]]


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Pr82TransferBuildError(f"expected JSON object: {path}")
    return payload


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name in sorted(members):
            member_path = Path(name)
            if not name or name.startswith("/") or ".." in member_path.parts or len(member_path.parts) != 1:
                raise Pr82TransferBuildError(f"unsafe output member name: {name!r}")
            zf.writestr(_zip_info(name), members[name])


def _read_single_member(archive: Path, expected_name: str) -> bytes:
    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_name]:
            raise Pr82TransferBuildError(f"{archive}: expected single member {expected_name!r}, got {names!r}")
        return zf.read(expected_name)


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr82_henosis_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise Pr82TransferBuildError(f"cannot import unpacker: {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _member_summary_from_header(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = header.get("members")
    if not isinstance(rows, list):
        raise Pr82TransferBuildError("runtime payload header lacks member list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name"))
        out[name] = dict(row)
    return out


def _extract_raw_segments(payload: bytes, member_summary: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    if payload.startswith(b"P6"):
        if len(payload) < 12:
            raise Pr82TransferBuildError("P6 payload too short")
        mask_len, renderer_len, actions_len, _record_count = struct.unpack_from("<IHHH", payload, 2)
        cursor = 2 + struct.calcsize("<IHHH")
        pose_start = cursor + mask_len + renderer_len + actions_len
        return {
            "masks.mkv": payload[cursor : cursor + mask_len],
            "renderer.bin": payload[cursor + mask_len : cursor + mask_len + renderer_len],
            "seg_tile_actions.bin": payload[cursor + mask_len + renderer_len : pose_start],
            "optimized_poses.qp1": payload[pose_start:],
        }
    if payload.startswith(b"P3"):
        if len(payload) < 10:
            raise Pr82TransferBuildError("P3 payload too short")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        cursor = 2 + struct.calcsize("<IHH")
        pose_start = cursor + mask_len + renderer_len + actions_len
        return {
            "masks.mkv": payload[cursor : cursor + mask_len],
            "renderer.bin": payload[cursor + mask_len : cursor + mask_len + renderer_len],
            "seg_tile_actions.bin": payload[cursor + mask_len + renderer_len : pose_start],
            "optimized_poses.qp1": payload[pose_start:],
        }
    cursor = 0
    raw_segments: dict[str, bytes] = {}
    for name in (*NON_POSE_MEMBERS, "optimized_poses.qp1"):
        if name not in member_summary:
            raise Pr82TransferBuildError(f"source runtime summary missing {name}")
        n_bytes = int(member_summary[name]["bytes"])
        raw_segments[name] = payload[cursor : cursor + n_bytes]
        cursor += n_bytes
    if cursor != len(payload):
        raise Pr82TransferBuildError(f"fixed-slice source consumed {cursor} bytes, payload has {len(payload)}")
    return raw_segments


def _build_payload_like_source(source: SourceArchive, pose_br: bytes) -> bytes:
    if source.payload.startswith(b"P6"):
        raw = source.raw_segments
        if max(len(raw["renderer.bin"]), len(raw["seg_tile_actions.bin"]), len(pose_br)) > 0xFFFF:
            raise Pr82TransferBuildError("P6 u16 stream length limit exceeded")
        return (
            b"P6"
            + struct.pack(
                "<IHHH",
                len(raw["masks.mkv"]),
                len(raw["renderer.bin"]),
                len(raw["seg_tile_actions.bin"]),
                int(source.member_summary["seg_tile_actions.bin"].get("decoded_bytes", 0)) // 4,
            )
            + raw["masks.mkv"]
            + raw["renderer.bin"]
            + raw["seg_tile_actions.bin"]
            + pose_br
        )
    if source.payload.startswith(b"P3"):
        raw = source.raw_segments
        return (
            b"P3"
            + struct.pack("<IHH", len(raw["masks.mkv"]), len(raw["renderer.bin"]), len(raw["seg_tile_actions.bin"]))
            + raw["masks.mkv"]
            + raw["renderer.bin"]
            + raw["seg_tile_actions.bin"]
            + pose_br
        )
    raw = source.raw_segments
    return raw["masks.mkv"] + raw["renderer.bin"] + raw["seg_tile_actions.bin"] + pose_br


def _load_source_archive(path: Path, *, expected_sha256: str | None) -> SourceArchive:
    path = path.resolve()
    if expected_sha256:
        actual = sha256_path(path)
        if actual != expected_sha256:
            raise Pr82TransferBuildError(f"source archive SHA mismatch: expected {expected_sha256}, got {actual}")
    payload = _read_single_member(path, "p")
    unpacker = _load_unpacker()
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded_members = {str(name): bytes(data) for name, data in decoded.items()}
    member_summary = _member_summary_from_header(header)
    for name in (*NON_POSE_MEMBERS, "optimized_poses.qp1"):
        if name not in decoded_members:
            raise Pr82TransferBuildError(f"source runtime parse missing {name}")
    raw_segments = _extract_raw_segments(payload, member_summary)
    for name, raw in raw_segments.items():
        if sha256_bytes(raw) != str(member_summary[name]["sha256"]):
            raise Pr82TransferBuildError(f"source raw slice SHA mismatch for {name}")
    return SourceArchive(
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=sha256_path(path),
        payload=payload,
        payload_sha256=sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        raw_segments=raw_segments,
        decoded_members=decoded_members,
        member_summary={name: dict(value) for name, value in member_summary.items()},
    )


def _load_pr82(pr82_archive: Path, replay_inflate: Path, *, expected_sha256: str | None) -> tuple[dict[str, Any], Any]:
    pr82_archive = pr82_archive.resolve()
    archive_sha = sha256_path(pr82_archive)
    if expected_sha256 and archive_sha != expected_sha256:
        raise Pr82TransferBuildError(f"PR82 archive SHA mismatch: expected {expected_sha256}, got {archive_sha}")
    raw = _read_single_member(pr82_archive, "x")
    contract = parse_replay_contract(replay_inflate)
    bundle = parse_pr82_bundle(raw, contract)
    return (
        {
            "archive_bytes": pr82_archive.stat().st_size,
            "archive_path": _repo_rel(pr82_archive),
            "archive_sha256": archive_sha,
            "contract": {
                "fixed_bias_bytes": contract.fixed_bias_bytes,
                "fixed_region_bytes": contract.fixed_region_bytes,
                "randmulti_group_count": len(contract.randmulti_specs),
                "replay_inflate": _repo_rel(replay_inflate),
                "replay_inflate_sha256": contract.source_sha256,
            },
            "payload_bytes": bundle.payload_bytes,
            "payload_sha256": sha256_bytes(raw),
        },
        (contract, bundle),
    )


def _source_exact_score(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "canonical_score": CURRENT_A_PLUS_PLUS_T4_FRONTIER,
            "source": "hardcoded_current_A++_T4_frontier",
        }
    payload = _read_json(path)
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        "archive_sha256": provenance.get("archive_sha256"),
        "canonical_score": float(payload.get("canonical_score", CURRENT_A_PLUS_PLUS_T4_FRONTIER)),
        "eval_json": _repo_rel(path),
        "n_samples": payload.get("n_samples"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
    }


def _brotli_best(raw: bytes, *, source: bytes | None = None) -> tuple[bytes, dict[str, int] | str]:
    best: bytes | None = None
    best_params: dict[str, int] | str = {}
    if source is not None:
        try:
            if brotli.decompress(source) == raw:
                best = source
                best_params = "source"
        except brotli.error:
            pass
    for quality in (11, 10, 9, 6, 4, 2, 0):
        for mode in (0, 1, 2):
            for lgwin in (10, 16, 22, 24):
                candidate = brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin)
                if best is None or len(candidate) < len(best):
                    best = candidate
                    best_params = {"quality": quality, "mode": mode, "lgwin": lgwin}
    if best is None:
        raise Pr82TransferBuildError("no Brotli pose candidate produced")
    if brotli.decompress(best) != raw:
        raise Pr82TransferBuildError("selected Brotli pose stream failed round-trip")
    return best, best_params


def _candidate_gate(
    *,
    source_score: float,
    archive_delta_bytes: int,
    changed_atoms: int,
    runtime_compatible: bool,
    raw_delta_proof: bool,
    proxy_component_gain: float = 0.0,
    risk_flags: Sequence[str] = (),
) -> dict[str, Any]:
    blockers: list[str] = []
    if changed_atoms <= 0:
        blockers.append("candidate is a no-op after atom filtering")
    if not runtime_compatible:
        blockers.append("candidate is not supported by current robust runtime")
    if not raw_delta_proof:
        blockers.append("no local raw-output delta proof is attached")
    if "known_negative_family" in risk_flags:
        blockers.append("nearby transfer family already has negative exact T4 evidence")
    rate_delta = archive_delta_bytes * RATE_SCORE_PER_BYTE
    required_to_beat_frontier = max(0.0, rate_delta)
    required_for_sub314 = max(0.0, source_score - SUB314_TARGET + rate_delta)
    if proxy_component_gain <= required_to_beat_frontier:
        blockers.append("no component proxy clears the rate break-even against PR79/S2")
    return {
        "component_gain_required_for_sub314": required_for_sub314,
        "component_gain_required_to_beat_pr79_s2": required_to_beat_frontier,
        "dispatch_ready_now": False,
        "formula_rate_score_delta_vs_source": rate_delta,
        "lane_claim_required_before_any_exact_eval": True,
        "proxy_component_gain": proxy_component_gain,
        "remote_dispatch_performed": False,
        "recommendation": "do_not_dispatch",
        "reason": "; ".join(blockers) if blockers else "local candidate still requires lane claim and exact CUDA eval",
        "score_claim": False,
    }


def _build_qpost_candidate(
    *,
    source: SourceArchive,
    pr82_encoded: Mapping[str, bytes],
    selected_pairs: Sequence[int],
    include_streams: Sequence[str],
    candidate_id: str,
    output_dir: Path,
    source_score: float,
) -> dict[str, Any]:
    streams = filter_qpost_streams_to_pairs(
        pr82_encoded,
        selected_pairs,
        include_streams=include_streams,
    )
    qpost = encode_qpost(streams)
    candidate_dir = output_dir / candidate_id
    archive = candidate_dir / "archive.zip"
    _write_archive(archive, {"p": source.payload, "qpost.bin": qpost})
    arrays = decode_control_arrays(pr82_encoded)
    active_atoms = 0
    for pair in selected_pairs:
        for name in include_streams:
            if name == "post":
                active_atoms += int(np.count_nonzero(arrays[name][:, pair] != 0))
            else:
                default = {"shift": 40, "frac": 4, "frac2": 4, "frac3": 4, "bias": 13, "region": 0}[name]
                active_atoms += int(arrays[name][pair] != default)
    archive_delta = archive.stat().st_size - source.archive_bytes
    manifest = {
        "archive_byte_profile": profile_archive(archive),
        "candidate_id": candidate_id,
        "candidate_kind": "pr82_runtime_compatible_qpost_pair_filter",
        "dispatch_gate": _candidate_gate(
            source_score=source_score,
            archive_delta_bytes=archive_delta,
            changed_atoms=active_atoms,
            runtime_compatible=True,
            raw_delta_proof=False,
            risk_flags=("known_negative_family",),
        ),
        "evidence_grade": "empirical_local_archive_build_and_atom_accounting",
        "include_streams": list(include_streams),
        "manifest_schema": MANIFEST_SCHEMA,
        "no_remote_dispatch": True,
        "no_op_detection": {
            "is_noop": active_atoms == 0,
            "selected_active_atoms_total": active_atoms,
            "selected_pair_count": len(selected_pairs),
            "selected_pairs": list(selected_pairs),
        },
        "output_archive": {
            "bytes": archive.stat().st_size,
            "path": _repo_rel(archive),
            "sha256": sha256_path(archive),
        },
        "qpost": {
            "bytes": len(qpost),
            "runtime_contract": "QPS1",
            "runtime_helper": _repo_rel(REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py"),
            "sha256": sha256_bytes(qpost),
            "streams": qpost_stream_summary(streams, pr82_encoded),
        },
        "score_claim": False,
        "schema": SCHEMA,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_source": archive_delta,
            "qpost_charged_member_bytes": len(qpost),
        },
        "tool": TOOL,
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": candidate_id,
        "candidate_kind": manifest["candidate_kind"],
        "dispatch_gate": manifest["dispatch_gate"],
        "manifest_path": _repo_rel(candidate_dir / "manifest.json"),
        "selected_active_atoms_total": active_atoms,
        "stream_delta": manifest["stream_delta"],
    }


def _build_pose_candidate(
    *,
    source: SourceArchive,
    pr82_pose: np.ndarray,
    selected_atoms: Sequence[Mapping[str, Any]],
    candidate_id: str,
    output_dir: Path,
    source_score: float,
) -> dict[str, Any]:
    base_pose = decode_qp1(source.decoded_members["optimized_poses.qp1"])
    candidate_pose = base_pose.copy()
    for atom in selected_atoms:
        pair = int(atom["pair_index"])
        delta_q = int(atom["delta_q"])
        candidate_pose[pair, 0] += float(delta_q) / 512.0
    pose_qp1 = encode_qp1(candidate_pose)
    pose_br, params = _brotli_best(pose_qp1, source=source.raw_segments["optimized_poses.qp1"])
    payload = _build_payload_like_source(source, pose_br)
    candidate_dir = output_dir / candidate_id
    archive = candidate_dir / "archive.zip"
    _write_archive(archive, {"p": payload})
    changed_words = int(np.count_nonzero(decode_qp1(pose_qp1)[:, 0] != base_pose[:, 0]))
    archive_delta = archive.stat().st_size - source.archive_bytes
    manifest = {
        "archive_byte_profile": profile_archive(archive),
        "candidate_id": candidate_id,
        "candidate_kind": "pr82_p1d1_velocity_atoms_reencoded_qp1",
        "dispatch_gate": _candidate_gate(
            source_score=source_score,
            archive_delta_bytes=archive_delta,
            changed_atoms=changed_words,
            runtime_compatible=True,
            raw_delta_proof=False,
            risk_flags=("known_negative_family",),
        ),
        "evidence_grade": "empirical_local_archive_build_and_pose_word_accounting",
        "manifest_schema": MANIFEST_SCHEMA,
        "no_remote_dispatch": True,
        "no_op_detection": {
            "changed_pose_word_count": changed_words,
            "is_noop": changed_words == 0,
            "selected_atom_count": len(selected_atoms),
        },
        "output_archive": {
            "bytes": archive.stat().st_size,
            "path": _repo_rel(archive),
            "sha256": sha256_path(archive),
        },
        "pose_change": {
            "base_pose_brotli_bytes": len(source.raw_segments["optimized_poses.qp1"]),
            "base_pose_qp1_sha256": sha256_bytes(source.decoded_members["optimized_poses.qp1"]),
            "candidate_pose_brotli_bytes": len(pose_br),
            "candidate_pose_brotli_params": params,
            "candidate_pose_brotli_sha256": sha256_bytes(pose_br),
            "candidate_pose_qp1_sha256": sha256_bytes(pose_qp1),
            "pr82_pose_float32_sha256": sha256_bytes(pr82_pose.astype("<f4", copy=False).tobytes()),
            "selected_atoms": list(selected_atoms),
        },
        "score_claim": False,
        "schema": SCHEMA,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "stream_closure": {
            "non_pose_streams_preserved": True,
            "source_payload_format": source.payload_format,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_source": archive_delta,
            "pose_brotli_delta_bytes_vs_source": len(pose_br) - len(source.raw_segments["optimized_poses.qp1"]),
        },
        "tool": TOOL,
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": candidate_id,
        "candidate_kind": manifest["candidate_kind"],
        "dispatch_gate": manifest["dispatch_gate"],
        "manifest_path": _repo_rel(candidate_dir / "manifest.json"),
        "selected_active_atoms_total": changed_words,
        "stream_delta": manifest["stream_delta"],
    }


def _build_randmulti_candidate(
    *,
    source: SourceArchive,
    pr82_encoded: Mapping[str, bytes],
    selected_groups: Sequence[Any],
    candidate_id: str,
    output_dir: Path,
    source_score: float,
) -> dict[str, Any]:
    randmulti = encode_randmulti_nm2(selected_groups)
    streams = {name: b"" for name in QPOST_STREAM_NAMES}
    streams["randmulti"] = randmulti
    qpost = encode_qpost(streams)
    candidate_dir = output_dir / candidate_id
    archive = candidate_dir / "archive.zip"
    _write_archive(archive, {"p": source.payload, "qpost.bin": qpost})
    active_atoms = int(sum(np.count_nonzero(group.rows) for group in selected_groups))
    archive_delta = archive.stat().st_size - source.archive_bytes
    group_rows = [randmulti_group_summary(group) for group in selected_groups]
    manifest = {
        "archive_byte_profile": profile_archive(archive),
        "candidate_id": candidate_id,
        "candidate_kind": "pr82_randmulti_qps1_nm2_generic_group_transfer",
        "dispatch_gate": _candidate_gate(
            source_score=source_score,
            archive_delta_bytes=archive_delta,
            changed_atoms=active_atoms,
            runtime_compatible=True,
            raw_delta_proof=False,
            risk_flags=("known_negative_family",),
        ),
        "evidence_grade": "empirical_local_archive_build_and_randmulti_byte_screen",
        "manifest_schema": MANIFEST_SCHEMA,
        "no_remote_dispatch": True,
        "no_op_detection": {
            "is_noop": active_atoms == 0,
            "selected_active_atoms_total": active_atoms,
            "selected_group_count": len(selected_groups),
            "selected_group_indices": [int(group.group_index) for group in selected_groups],
        },
        "output_archive": {
            "bytes": archive.stat().st_size,
            "path": _repo_rel(archive),
            "sha256": sha256_path(archive),
        },
        "qpost": {
            "bytes": len(qpost),
            "runtime_contract": "QPS1/NM2",
            "runtime_helper": _repo_rel(REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py"),
            "sha256": sha256_bytes(qpost),
            "streams": qpost_stream_summary(streams, pr82_encoded),
        },
        "randmulti": {
            "charged_representation": "NM2_dense_runtime_compatible_subset",
            "groups": group_rows,
            "semantic_scope": "generic_frame0_nearest_random_pattern_only",
        },
        "score_claim": False,
        "schema": SCHEMA,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_source": archive_delta,
            "qpost_charged_member_bytes": len(qpost),
            "randmulti_brotli_bytes": len(randmulti),
        },
        "tool": TOOL,
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": candidate_id,
        "candidate_kind": manifest["candidate_kind"],
        "dispatch_gate": manifest["dispatch_gate"],
        "manifest_path": _repo_rel(candidate_dir / "manifest.json"),
        "selected_active_atoms_total": active_atoms,
        "stream_delta": manifest["stream_delta"],
    }


def _build_randmulti_qrm1_candidate(
    *,
    source: SourceArchive,
    pr82_encoded: Mapping[str, bytes],
    replay_specs: Sequence[Sequence[int]],
    selected_groups: Sequence[Any],
    candidate_id: str,
    output_dir: Path,
    source_score: float,
) -> dict[str, Any]:
    randmulti = encode_randmulti_qrm1(selected_groups)
    decoded_groups = decode_randmulti_qrm1(randmulti, replay_specs)
    qrm1_profile = randmulti_qrm1_parity_profile(
        selected_groups,
        decoded_groups,
        encoded=randmulti,
        source_encoded=pr82_encoded["randmulti"],
    )
    streams = {name: b"" for name in QPOST_STREAM_NAMES}
    streams["randmulti"] = randmulti
    qpost = encode_qpost(streams)
    candidate_dir = output_dir / candidate_id
    archive = candidate_dir / "archive.zip"
    _write_archive(archive, {"p": source.payload, "qpost.bin": qpost})
    active_atoms = int(sum(np.count_nonzero(group.rows) for group in selected_groups))
    archive_delta = archive.stat().st_size - source.archive_bytes
    group_rows = [randmulti_group_summary(group) for group in decoded_groups]
    robust_runtime_ready = False
    manifest = {
        "archive_byte_profile": profile_archive(archive),
        "candidate_id": candidate_id,
        "candidate_kind": "pr82_randmulti_qps1_qrm1_native_sparse_group_transfer",
        "dispatch_gate": _candidate_gate(
            source_score=source_score,
            archive_delta_bytes=archive_delta,
            changed_atoms=active_atoms,
            runtime_compatible=robust_runtime_ready,
            raw_delta_proof=False,
            risk_flags=("known_negative_family",),
        ),
        "evidence_grade": "empirical_local_archive_build_and_qrm1_group_row_parity",
        "manifest_schema": MANIFEST_SCHEMA,
        "no_remote_dispatch": True,
        "no_op_detection": {
            "is_noop": active_atoms == 0,
            "selected_active_atoms_total": active_atoms,
            "selected_group_count": len(selected_groups),
            "selected_group_indices": [int(group.group_index) for group in selected_groups],
        },
        "output_archive": {
            "bytes": archive.stat().st_size,
            "path": _repo_rel(archive),
            "sha256": sha256_path(archive),
        },
        "qpost": {
            "bytes": len(qpost),
            "runtime_contract": "QPS1/QRM1",
            "runtime_helper": _repo_rel(REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py"),
            "robust_current_runtime_ready": robust_runtime_ready,
            "sha256": sha256_bytes(qpost),
            "streams": qpost_stream_summary(streams, pr82_encoded),
        },
        "randmulti": {
            "charged_representation": "QRM1_sparse_group_id_runtime_contract",
            "groups": group_rows,
            "local_decode_profile": qrm1_profile,
            "runtime_extension_required": [
                "read QRM1 in the randmulti stream after Brotli decompression",
                "look up PR82 replay group specs by carried u16 group id",
                "apply generic groups to frame 0 with existing nearest random-pattern path",
                "apply replay-special f2 tile, boundary, class-conditioned, and global RGB-bias branches to frame 1",
            ],
            "semantic_scope": "all PR82 randmulti groups represented exactly at sparse choice-row level",
        },
        "score_claim": False,
        "schema": SCHEMA,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_source": archive_delta,
            "qpost_charged_member_bytes": len(qpost),
            "randmulti_brotli_bytes": len(randmulti),
            "randmulti_brotli_delta_bytes_vs_pr82_tail": qrm1_profile["qrm1_brotli_delta_bytes_vs_source_tail"],
        },
        "tool": TOOL,
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": candidate_id,
        "candidate_kind": manifest["candidate_kind"],
        "dispatch_gate": manifest["dispatch_gate"],
        "manifest_path": _repo_rel(candidate_dir / "manifest.json"),
        "qrm1_local_decode_profile": qrm1_profile,
        "selected_active_atoms_total": active_atoms,
        "stream_delta": manifest["stream_delta"],
    }


def build_candidates(
    *,
    pr82_archive: Path = DEFAULT_PR82_ARCHIVE,
    replay_inflate: Path = DEFAULT_REPLAY_INFLATE,
    source_archive: Path = DEFAULT_PR79_S2_ARCHIVE,
    source_exact_json: Path | None = DEFAULT_PR79_S2_EXACT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    qpost_topks: Sequence[int] = DEFAULT_QPOST_TOPKS,
    pose_topks: Sequence[int] = DEFAULT_POSE_TOPKS,
    randmulti_topks: Sequence[int] = DEFAULT_RANDMULTI_TOPKS,
    include_streams: Sequence[str] = DEFAULT_QPOST_STREAMS,
    expected_pr82_sha256: str | None = EXPECTED_PR82_SHA256,
    expected_source_sha256: str | None = EXPECTED_PR79_S2_SHA256,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pr82_profile, (contract, bundle) = _load_pr82(
        pr82_archive,
        replay_inflate,
        expected_sha256=expected_pr82_sha256,
    )
    source = _load_source_archive(source_archive, expected_sha256=expected_source_sha256)
    source_exact = _source_exact_score(source_exact_json)
    source_score = float(source_exact["canonical_score"])

    arrays = decode_control_arrays(bundle.encoded_segments)
    randmulti = decode_randmulti_activity(bundle.encoded_segments["randmulti"], contract.randmulti_specs)
    randmulti_groups = decode_randmulti_groups(bundle.encoded_segments["randmulti"], contract.randmulti_specs)
    compatible_randmulti_groups = [
        group for group in randmulti_groups if randmulti_group_qps1_nm2_compatible(group)
    ]
    compatible_randmulti_groups = sorted(
        compatible_randmulti_groups,
        key=lambda group: (-int(np.count_nonzero(group.rows)), int(group.group_index)),
    )
    incompatible_randmulti_groups = [
        group for group in randmulti_groups if not randmulti_group_qps1_nm2_compatible(group)
    ]
    randmulti_qrm1_all = encode_randmulti_qrm1(randmulti_groups)
    randmulti_qrm1_decoded = decode_randmulti_qrm1(randmulti_qrm1_all, contract.randmulti_specs)
    randmulti_qrm1_profile = randmulti_qrm1_parity_profile(
        randmulti_groups,
        randmulti_qrm1_decoded,
        encoded=randmulti_qrm1_all,
        source_encoded=bundle.encoded_segments["randmulti"],
    )
    randmulti_lowlevel_profile = {
        "decoded_brotli_bytes": randmulti["decoded_bytes"],
        "encoded_brotli_bytes": len(bundle.encoded_segments["randmulti"]),
        "encoded_sha256": sha256_bytes(bundle.encoded_segments["randmulti"]),
        "group_count": len(randmulti_groups),
        "groups": [randmulti_group_summary(group) for group in randmulti_groups],
        "qps1_nm2_compatible_group_count": len(compatible_randmulti_groups),
        "qps1_nm2_compatible_nonzero_choice_total": int(
            sum(np.count_nonzero(group.rows) for group in compatible_randmulti_groups)
        ),
        "replay_only_group_count": len(incompatible_randmulti_groups),
        "runtime_compatibility": {
            "compatible_representation": "QPS1 randmulti stream containing Brotli(NM2 dense groups)",
            "compatible_scope": "generic frame-0 random-pattern groups with u8 dimensions only",
            "incompatible_reasons": [
                "current robust QPS1 helper does not yet read QRM1",
                "NM2 stores height/width/amplitude/scount as u8, so large PR82 groups cannot be represented by the older dense subset",
                "replay special-case f2 tile/boundary/class-bias groups need distinct frame-1 semantics in the robust runtime apply path",
            ],
            "minimum_contract_extension": "QRM1 sparse group-id stream: magic + u16 group count + repeated u16 replay_group_id + sparse rows, with runtime table lookup for generic and special semantics",
        },
        "qrm1_native_contract_profile": randmulti_qrm1_profile,
    }
    randmulti_profile_path = output_dir / "pr82_randmulti_lowlevel_profile.json"
    _write_json(randmulti_profile_path, randmulti_lowlevel_profile)
    pair_rows = summarize_pair_activity(arrays, randmulti_counts=randmulti["per_pair_nonzero_counts"])
    ranked_pairs = rank_pairs_by_activity(pair_rows)
    qpost_candidates = []
    for topk in qpost_topks:
        if topk <= 0:
            continue
        selected = tuple(sorted(ranked_pairs[: int(topk)]))
        qpost_candidates.append(
            _build_qpost_candidate(
                source=source,
                pr82_encoded=bundle.encoded_segments,
                selected_pairs=selected,
                include_streams=tuple(include_streams),
                candidate_id=f"pr82_qpost_top{int(topk):03d}_on_source",
                output_dir=output_dir,
                source_score=source_score,
            )
        )

    randmulti_candidates = []
    for topk in randmulti_topks:
        if topk <= 0:
            continue
        selected_groups = compatible_randmulti_groups[: int(topk)]
        if not selected_groups:
            continue
        randmulti_candidates.append(
            _build_randmulti_candidate(
                source=source,
                pr82_encoded=bundle.encoded_segments,
                selected_groups=selected_groups,
                candidate_id=f"pr82_randmulti_generic_top{int(topk):03d}_on_source",
                output_dir=output_dir,
                source_score=source_score,
            )
        )
    randmulti_qrm1_candidates = [
        _build_randmulti_qrm1_candidate(
            source=source,
            pr82_encoded=bundle.encoded_segments,
            replay_specs=contract.randmulti_specs,
            selected_groups=randmulti_groups,
            candidate_id="pr82_randmulti_qrm1_all072_on_source",
            output_dir=output_dir,
            source_score=source_score,
        )
    ]

    pr82_pose = decode_pr82_p1d1_pose(bundle.encoded_segments["pose"])
    source_pose = decode_qp1(source.decoded_members["optimized_poses.qp1"])
    pose_atoms = pose_velocity_atom_ranking(source_pose, pr82_pose)
    pose_candidates = []
    for topk in pose_topks:
        if topk <= 0:
            continue
        selected_atoms = pose_atoms[: int(topk)]
        pose_candidates.append(
            _build_pose_candidate(
                source=source,
                pr82_pose=pr82_pose,
                selected_atoms=selected_atoms,
                candidate_id=f"pr82_pose_velocity_top{int(topk):03d}_on_source",
                output_dir=output_dir,
                source_score=source_score,
            )
        )

    adversarial_profile = {
        "known_rate_problem": {
            "pr82_archive_bytes": pr82_profile["archive_bytes"],
            "source_archive_bytes": source.archive_bytes,
            "static_delta_bytes": pr82_profile["archive_bytes"] - source.archive_bytes,
            "static_rate_score_delta": (pr82_profile["archive_bytes"] - source.archive_bytes) * RATE_SCORE_PER_BYTE,
        },
        "nontransferable_or_blocked_atoms": [
            {
                "atom_family": "randmulti",
                "compatible_group_count": len(compatible_randmulti_groups),
                "compatible_representation": "QPS1/NM2 dense generic subset",
                "qrm1_native_group_count": len(randmulti_groups),
                "qrm1_native_local_parity": randmulti_qrm1_profile["exact_group_row_parity"],
                "reason": "PR82 replay declares 72 headerless randmulti groups; QRM1 now represents all groups exactly at sparse row level, but robust_current apply_qzs3_postprocess.py still needs the QRM1 decoder and f2 special-branch application before dispatch",
                "replay_only_group_count": len(incompatible_randmulti_groups),
                "runtime_compatible": "QPS1/NM2 partial_generic_subset_only; QPS1/QRM1 local_contract_ready_not_robust_runtime_ready",
                "nonzero_pair_count": randmulti["nonzero_pair_count"],
            },
            {
                "atom_family": "P1D1_dim2_pose",
                "reason": "PR79/S2 QP1 carries velocity column only; non-velocity transfer needs a separate reviewed PVR1/QP2 archive contract",
                "runtime_compatible": False,
            },
        ],
        "dispatch_policy": {
            "no_remote_dispatch": True,
            "lane_claim_required_before_any_future_exact_eval": True,
            "recommendation": "do_not_dispatch_without_raw_delta_and_component_trace",
        },
    }
    summary = {
        "adversarial_profile": adversarial_profile,
        "candidate_count": len(qpost_candidates) + len(randmulti_candidates) + len(randmulti_qrm1_candidates) + len(pose_candidates),
        "evidence_grade": "empirical_local_archive_build_and_atom_accounting",
        "include_streams": list(include_streams),
        "no_remote_dispatch": True,
        "pair_activity_top16": [pair_rows[pair] for pair in ranked_pairs[:16]],
        "pose_atom_top16": pose_atoms[:16],
        "pr82_profile": pr82_profile,
        "qpost_candidates": qpost_candidates,
        "randmulti_candidates": randmulti_candidates,
        "randmulti_qrm1_candidates": randmulti_qrm1_candidates,
        "randmulti_lowlevel_profile_path": _repo_rel(randmulti_profile_path),
        "randmulti_qrm1_native_contract_profile": randmulti_qrm1_profile,
        "randmulti_profile_top16_compatible_groups": [
            randmulti_group_summary(group) for group in compatible_randmulti_groups[:16]
        ],
        "randmulti_profile_top16_replay_only_groups": [
            randmulti_group_summary(group)
            for group in sorted(
                incompatible_randmulti_groups,
                key=lambda group: (-int(np.count_nonzero(group.rows)), int(group.group_index)),
            )[:16]
        ],
        "pose_candidates": pose_candidates,
        "schema": SCHEMA,
        "score_claim": False,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "payload_format": source.payload_format,
            "sha256": source.archive_sha256,
        },
        "source_exact_reference": source_exact,
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    (output_dir / "DESIGN_NOTE.md").write_text(
        "# PR82/Henosis Atom Transfer\n\n"
        "No remote GPU dispatch was performed.\n\n"
        "The generated archives are deterministic local candidates only.  "
        "Every dispatch gate is fail-closed because the candidates lack a "
        "local raw-output delta proof and component-trace support against the "
        "current PR79/S2 A++ T4 frontier.  PR82 randmulti is deconstructed "
        "into 72 replay groups; only the generic u8-sized frame-0 subset is "
        "byte-screened through the current QPS1/NM2 helper.  A QPS1/QRM1 "
        "native sparse group-id stream now parity-checks all 72 PR82 groups "
        "locally, but robust_current still needs QRM1 decode/apply support for "
        "the replay-only large and f2-special groups before exact eval "
        "dispatch.\n",
        encoding="utf-8",
    )
    return summary


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() in {"", "none"}:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_streams(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(values) - set(QPOST_STREAM_NAMES))
    if unknown:
        raise Pr82TransferBuildError(f"unknown streams: {unknown}")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr82-archive", type=Path, default=DEFAULT_PR82_ARCHIVE)
    parser.add_argument("--replay-inflate", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_PR79_S2_ARCHIVE)
    parser.add_argument("--source-exact-json", type=Path, default=DEFAULT_PR79_S2_EXACT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-streams", default=",".join(DEFAULT_QPOST_STREAMS))
    parser.add_argument("--qpost-topks", default=",".join(str(value) for value in DEFAULT_QPOST_TOPKS))
    parser.add_argument("--pose-topks", default=",".join(str(value) for value in DEFAULT_POSE_TOPKS))
    parser.add_argument("--randmulti-topks", default=",".join(str(value) for value in DEFAULT_RANDMULTI_TOPKS))
    parser.add_argument("--no-pr82-sha-check", action="store_true")
    parser.add_argument("--no-source-sha-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_candidates(
        pr82_archive=args.pr82_archive,
        replay_inflate=args.replay_inflate,
        source_archive=args.source_archive,
        source_exact_json=args.source_exact_json,
        output_dir=args.output_dir,
        qpost_topks=_parse_csv_ints(args.qpost_topks),
        pose_topks=_parse_csv_ints(args.pose_topks),
        randmulti_topks=_parse_csv_ints(args.randmulti_topks),
        include_streams=_parse_streams(args.include_streams),
        expected_pr82_sha256=None if args.no_pr82_sha_check else EXPECTED_PR82_SHA256,
        expected_source_sha256=None if args.no_source_sha_check else EXPECTED_PR79_S2_SHA256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
