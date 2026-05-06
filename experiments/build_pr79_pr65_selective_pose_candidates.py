#!/usr/bin/env python3
"""Build local PR79/S2 plus selective PR65 pose candidates.

This tool is local-only.  It keeps the PR79/S2 mask, renderer, and action
streams byte-identical, rewrites only the charged QP1 pose slice, validates
the resulting archive with the contest runtime unpacker, and emits manifests
with byte/proxy accounting.  It never dispatches remote or GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
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
from tac.qp1_pose_codec import decode_qp1, encode_qp1


BASE_BUILDER_PATH = REPO_ROOT / "experiments/build_pr79_action_lossless_repack_candidates.py"
PR65_TOOL_PATH = REPO_ROOT / "experiments/plan_pr65_henosis_stream_transfer.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr79_pr65_selective_pose_candidates_20260503_worker"
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
DEFAULT_PR65_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_POSE_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/public_pose_manifold_compare_20260503_worker/"
    "pose_manifold_compare_plan.json"
)
DEFAULT_NEGATIVE_EXACT_JSON = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_fix1_20260503T121925Z/"
    "contest_auth_eval.json"
)
TOOL = "experiments/build_pr79_pr65_selective_pose_candidates.py"
SCHEMA = "pr79_pr65_selective_pose_candidates_v1"
MANIFEST_SCHEMA = "pr79_pr65_selective_pose_manifest_v1"
EXPECTED_PR79_S2_SHA256 = "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
EXPECTED_PR65_SHA256 = "b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68"
PR79_S2_SCORE = 0.31453355357318635
SUB314_TARGET = 0.314
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
POSE_ATOM_TOPKS = (8, 16, 24, 40)
NON_POSE_SEGMENTS = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin")


@dataclass(frozen=True)
class PoseCandidateSpec:
    candidate_id: str
    kind: str
    selected_atoms: tuple[dict[str, Any], ...]
    proxy_benefit: float
    proxy_source: str
    risk_flags: tuple[str, ...]


@dataclass(frozen=True)
class SelectiveSourceArchive:
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    raw_segments: dict[str, bytes]
    decoded: dict[str, bytes]


class SelectivePoseError(ValueError):
    """Raised when local PR79/PR65 pose candidate building fails a guard."""


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SelectivePoseError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_BUILDER_PATH, "pr79_pr65_pose_base_builder")
PR65 = _load_module(PR65_TOOL_PATH, "pr79_pr65_pose_pr65_tool")


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SelectivePoseError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SelectivePoseError(f"expected JSON object: {path}")
    return payload


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _brotli_pose(raw_qp1: bytes) -> tuple[bytes, dict[str, int]]:
    best: bytes | None = None
    best_params: dict[str, int] | None = None
    for quality in range(11, -1, -1):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                candidate = brotli.compress(raw_qp1, quality=quality, mode=mode, lgwin=lgwin)
                if best is None or len(candidate) < len(best):
                    best = candidate
                    best_params = {"quality": quality, "mode": mode, "lgwin": lgwin}
    if best is None or best_params is None:
        raise SelectivePoseError("no Brotli pose candidate generated")
    if brotli.decompress(best) != raw_qp1:
        raise SelectivePoseError("selected pose Brotli stream failed round-trip")
    return best, best_params


def _pose_words_from_qp1(raw_qp1: bytes) -> np.ndarray:
    pose = decode_qp1(raw_qp1)
    words = np.rint((pose[:, 0].astype(np.float64) - 20.0) * 512.0).astype(np.int64)
    if (words < 0).any() or (words > 0xFFFF).any():
        raise SelectivePoseError("QP1 velocity words are outside uint16 range")
    return words.astype(np.uint16)


def _pose_summary(raw_qp1: bytes, label: str) -> dict[str, Any]:
    pose = decode_qp1(raw_qp1)
    words = _pose_words_from_qp1(raw_qp1)
    return {
        "label": label,
        "pose_qp1_bytes": len(raw_qp1),
        "pose_qp1_sha256": _sha256_bytes(raw_qp1),
        "pose_float32_sha256": _sha256_bytes(pose.astype("<f4", copy=False).tobytes()),
        "pose_word_count": int(words.size),
        "pose_words_sha256": _sha256_bytes(words.astype("<u2", copy=False).tobytes()),
    }


def _build_payload(source: Any, pose_br: bytes) -> bytes:
    if source.payload.startswith(b"P3"):
        return (
            b"P3"
            + struct.pack(
                "<IHH",
                len(source.raw_segments["masks.mkv"]),
                len(source.raw_segments["renderer.bin"]),
                len(source.raw_segments["seg_tile_actions.bin"]),
            )
            + source.raw_segments["masks.mkv"]
            + source.raw_segments["renderer.bin"]
            + source.raw_segments["seg_tile_actions.bin"]
            + pose_br
        )
    if source.payload.startswith((b"P4", b"P5", b"P6")):
        raise SelectivePoseError(
            f"unsupported source container for PR79/S2 pose isolation: {source.payload[:2]!r}"
        )
    return (
        source.raw_segments["masks.mkv"]
        + source.raw_segments["renderer.bin"]
        + source.raw_segments["seg_tile_actions.bin"]
        + pose_br
    )


def _extract_raw_segments(payload: bytes, runtime_members: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        pose_start = header_size + mask_len + model_len + actions_len
        raw_segments = {
            "masks.mkv": payload[header_size:header_size + mask_len],
            "renderer.bin": payload[header_size + mask_len:header_size + mask_len + model_len],
            "seg_tile_actions.bin": payload[header_size + mask_len + model_len:pose_start],
            "optimized_poses.qp1": payload[pose_start:],
        }
    elif payload.startswith((b"P4", b"P5", b"P6")):
        raise SelectivePoseError(f"unsupported source container for PR79/S2 pose isolation: {payload[:2]!r}")
    else:
        offset = 0
        raw_segments = {}
        for name in ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1"):
            size = int(runtime_members[name]["bytes"])
            raw_segments[name] = payload[offset:offset + size]
            offset += size
        if offset != len(payload):
            raise SelectivePoseError(f"fixed source slices consume {offset}, payload has {len(payload)}")
    for name, raw in raw_segments.items():
        expected_sha = str(runtime_members[name]["sha256"])
        if _sha256_bytes(raw) != expected_sha:
            raise SelectivePoseError(f"raw source slice SHA mismatch for {name}")
    return raw_segments


def _load_source_archive(path: Path, unpacker: Any) -> SelectiveSourceArchive:
    path = path.resolve()
    payload = BASE._read_single_payload(path)  # noqa: SLF001
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    runtime_members = BASE._member_summary(header)  # noqa: SLF001
    missing = sorted(set((*NON_POSE_SEGMENTS, "optimized_poses.qp1")) - set(runtime_members))
    if missing:
        raise SelectivePoseError(f"source archive missing runtime members: {missing}")
    raw_segments = _extract_raw_segments(payload, runtime_members)
    BASE.validate_seg_tile_actions_payload(  # noqa: SLF001
        decoded["seg_tile_actions.bin"],
        source_name="PR79/S2 decoded seg_tile_actions.bin",
    )
    return SelectiveSourceArchive(
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        raw_segments=raw_segments,
        decoded=decoded,
    )


def _validate_candidate_payload(source: Any, payload: bytes, unpacker: Any) -> dict[str, Any]:
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    changed: list[str] = []
    for name in (*NON_POSE_SEGMENTS, "optimized_poses.qp1"):
        if name not in decoded:
            raise SelectivePoseError(f"candidate runtime parse missing {name}")
        if decoded[name] != source.decoded[name]:
            changed.append(name)
    unexpected = sorted(set(changed) - {"optimized_poses.qp1"})
    if unexpected:
        raise SelectivePoseError(f"non-pose streams changed: {unexpected}")
    runtime_members = BASE._member_summary(header)  # noqa: SLF001
    return {
        "changed_decoded_streams_vs_source": changed,
        "payload_format": str(header.get("payload_format")),
        "runtime_members": runtime_members,
        "runtime_parser": _repo_rel(BASE.UNPACKER_PATH),
        "status": "passed",
    }


def _load_pr65_pose_qp1(pr65_archive: Path, expected_sha256: str | None) -> bytes:
    anatomy = PR65.parse_pr65_henosis_archive(pr65_archive, expected_sha256=expected_sha256)
    pr65_pose = PR65.decode_pr65_p1d1_pose(anatomy["_segments_bytes"]["pose"])
    return encode_qp1(pr65_pose)


def _load_pr65_atom_plan(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    for row in payload.get("ranked_candidates", []):
        if isinstance(row, dict) and row.get("basis_id") == "public_difference_pr65":
            return row
    raise SelectivePoseError(f"missing public_difference_pr65 ranked candidate in {path}")


def _selected_atom_specs(plan: Mapping[str, Any]) -> list[PoseCandidateSpec]:
    atoms_raw = plan.get("selected_coefs")
    if not isinstance(atoms_raw, list) or not atoms_raw:
        raise SelectivePoseError("PR65 pose plan has no selected_coefs")
    atoms = tuple(dict(item) for item in atoms_raw if isinstance(item, dict))
    full_abs = sum(abs(int(item.get("delta_q", 0))) for item in atoms) or 1
    full_proxy = float(plan.get("expected_benefit_proxy", 0.0))
    specs: list[PoseCandidateSpec] = []
    for topk in POSE_ATOM_TOPKS:
        selected = atoms[:topk]
        selected_abs = sum(abs(int(item.get("delta_q", 0))) for item in selected)
        proxy = full_proxy * selected_abs / full_abs
        specs.append(
            PoseCandidateSpec(
                candidate_id=f"pr79_s2_pr65_pose_atoms_top{topk:03d}",
                kind="selective_pr65_qp1_velocity_atoms",
                selected_atoms=tuple(selected),
                proxy_benefit=proxy,
                proxy_source=(
                    "scaled_from_public_pose_manifold_compare public_difference_pr65 "
                    "expected_benefit_proxy by abs(delta_q) prefix mass"
                ),
                risk_flags=(
                    "planning_proxy_not_score_evidence",
                    "pr65_direct_transfer_exact_negative_exists",
                    "qp1_velocity_col0_only",
                ),
            )
        )
    return specs


def _apply_atoms(base_qp1: bytes, atoms: Sequence[Mapping[str, Any]]) -> bytes:
    pose = decode_qp1(base_qp1)
    for atom in atoms:
        pair_index = int(atom["pair_index"])
        delta_q = int(atom["delta_q"])
        if pair_index < 0 or pair_index >= pose.shape[0]:
            raise SelectivePoseError(f"pose atom pair index out of bounds: {pair_index}")
        pose[pair_index, 0] += float(delta_q) / 512.0
    return encode_qp1(pose)


def _exact_eval_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _read_json(path)
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        "path": _repo_rel(path),
        "archive_bytes": int(payload.get("archive_size_bytes", 0)),
        "archive_sha256": str(provenance.get("archive_sha256", "")),
        "avg_posenet_dist": float(payload.get("avg_posenet_dist", 0.0)),
        "avg_segnet_dist": float(payload.get("avg_segnet_dist", 0.0)),
        "canonical_score": float(payload.get("canonical_score", payload.get("score_recomputed_from_components", 0.0))),
        "n_samples": int(payload.get("n_samples", 0)),
    }


def _candidate_screen(
    *,
    source_score: float,
    archive_delta_bytes: int,
    proxy_benefit: float,
    exact_negative_present: bool,
    risk_flags: Sequence[str],
) -> dict[str, Any]:
    rate_delta = archive_delta_bytes * RATE_SCORE_PER_BYTE
    required_to_beat_frontier = max(0.0, rate_delta)
    required_for_sub314 = max(0.0, source_score - SUB314_TARGET + rate_delta)
    proxy_margin_vs_frontier = proxy_benefit - required_to_beat_frontier
    proxy_margin_vs_sub314 = proxy_benefit - required_for_sub314
    high_ev = (
        proxy_benefit > 0.0
        and proxy_margin_vs_frontier > 0.00005
        and not exact_negative_present
        and "wholesale_pr65_pose_exact_negative_on_nearby_frontier" not in risk_flags
    )
    return {
        "formula_rate_score_delta_vs_pr79_s2": rate_delta,
        "component_gain_required_to_beat_pr79_s2": required_to_beat_frontier,
        "component_gain_required_for_sub314": required_for_sub314,
        "proxy_benefit": proxy_benefit,
        "proxy_margin_vs_pr79_s2": proxy_margin_vs_frontier,
        "proxy_margin_vs_sub314": proxy_margin_vs_sub314,
        "high_ev_enough_for_exact_eval": high_ev,
        "dispatch_recommendation": {
            "recommended": high_ev,
            "dispatch_ready_now": False,
            "lane_claim_required_before_any_exact_eval": True,
            "remote_dispatch_performed": False,
            "reason": (
                "local archive-valid candidate has enough proxy margin to justify a claimed exact eval"
                if high_ev
                else "local archive-valid candidate does not clear the PR79/S2 proxy and risk screen"
            ),
        },
    }


def _build_one(
    *,
    source: Any,
    unpacker: Any,
    spec: PoseCandidateSpec,
    pose_qp1: bytes,
    pose_brotli_source: str,
    output_dir: Path,
    force: bool,
    source_score: float,
    negative_exact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pose_br, pose_params = _brotli_pose(pose_qp1)
    payload = _build_payload(source, pose_br)
    candidate_dir = output_dir / spec.candidate_id
    archive = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive.exists() and not force:
        raise FileExistsError(f"{archive} exists; pass --force")
    BASE._write_archive(archive, payload)  # noqa: SLF001
    if BASE._read_single_payload(archive) != payload:  # noqa: SLF001
        raise SelectivePoseError(f"{spec.candidate_id}: archive readback mismatch")
    validation = _validate_candidate_payload(source, payload, unpacker)
    archive_sha = _sha256_file(archive)
    payload_sha = _sha256_bytes(payload)
    base_pose_raw = source.raw_segments["optimized_poses.qp1"]
    base_pose_qp1 = source.decoded["optimized_poses.qp1"]
    changed_pose_words = int(
        np.count_nonzero(_pose_words_from_qp1(base_pose_qp1) != _pose_words_from_qp1(pose_qp1))
    )
    archive_delta = archive.stat().st_size - source.archive_bytes
    screen = _candidate_screen(
        source_score=source_score,
        archive_delta_bytes=archive_delta,
        proxy_benefit=spec.proxy_benefit,
        exact_negative_present=negative_exact is not None,
        risk_flags=spec.risk_flags,
    )
    manifest = {
        "archive_byte_profile": profile_archive(archive),
        "candidate_id": spec.candidate_id,
        "candidate_kind": spec.kind,
        "evidence_grade": "empirical_local_archive_valid_byte_and_proxy_screen",
        "manifest_schema": MANIFEST_SCHEMA,
        "no_remote_dispatch": True,
        "output_archive": {
            "bytes": archive.stat().st_size,
            "path": str(archive),
            "repo_relative_path": _repo_rel(archive),
            "sha256": archive_sha,
        },
        "payload": {
            "bytes": len(payload),
            "member": BASE.MEMBER_NAME,
            "sha256": payload_sha,
        },
        "pose_change": {
            "base_pose_raw_brotli_bytes": len(base_pose_raw),
            "base_pose_raw_brotli_sha256": _sha256_bytes(base_pose_raw),
            "candidate_pose_brotli_bytes": len(pose_br),
            "candidate_pose_brotli_params": pose_params,
            "candidate_pose_brotli_sha256": _sha256_bytes(pose_br),
            "changed_pose_word_count": changed_pose_words,
            "pose_brotli_delta_bytes_vs_pr79_s2": len(pose_br) - len(base_pose_raw),
            "pose_brotli_source": pose_brotli_source,
            "selected_atoms": list(spec.selected_atoms),
        },
        "pose_summary": _pose_summary(pose_qp1, spec.candidate_id),
        "proxy_screen": screen,
        "risk_flags": list(spec.risk_flags),
        "runtime_parse_validation": validation,
        "score_claim": False,
        "schema": SCHEMA,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
            "exact_score": source_score,
        },
        "stream_closure": {
            "decoded_mask_sha256": _sha256_bytes(source.decoded["masks.mkv"]),
            "decoded_renderer_sha256": _sha256_bytes(source.decoded["renderer.bin"]),
            "decoded_actions_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
            "non_pose_streams_preserved": True,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_pr79_s2": archive_delta,
            "payload_delta_bytes_vs_pr79_s2": len(payload) - len(source.payload),
        },
        "tool": TOOL,
    }
    if negative_exact is not None:
        manifest["nearby_pr65_pose_exact_negative"] = dict(negative_exact)
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["repo_relative_path"],
        "archive_sha256": archive_sha,
        "candidate_id": spec.candidate_id,
        "changed_pose_word_count": changed_pose_words,
        "dispatch_recommendation": screen["dispatch_recommendation"],
        "high_ev_enough_for_exact_eval": screen["high_ev_enough_for_exact_eval"],
        "manifest_path": _repo_rel(manifest_path),
        "pose_brotli_delta_bytes_vs_pr79_s2": manifest["pose_change"]["pose_brotli_delta_bytes_vs_pr79_s2"],
        "proxy_margin_vs_pr79_s2": screen["proxy_margin_vs_pr79_s2"],
        "proxy_margin_vs_sub314": screen["proxy_margin_vs_sub314"],
        "score_claim": False,
    }


def build_candidates(
    *,
    pr79_s2_archive: Path = DEFAULT_PR79_S2_ARCHIVE,
    pr65_archive: Path = DEFAULT_PR65_ARCHIVE,
    pose_plan_json: Path = DEFAULT_POSE_PLAN_JSON,
    pr79_s2_exact_json: Path | None = DEFAULT_PR79_S2_EXACT_JSON,
    negative_exact_json: Path | None = DEFAULT_NEGATIVE_EXACT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
    expected_pr79_s2_sha256: str | None = EXPECTED_PR79_S2_SHA256,
    expected_pr65_sha256: str | None = EXPECTED_PR65_SHA256,
) -> dict[str, Any]:
    unpacker = BASE._load_unpacker()  # noqa: SLF001
    source = _load_source_archive(pr79_s2_archive, unpacker=unpacker)
    if expected_pr79_s2_sha256 and source.archive_sha256 != expected_pr79_s2_sha256:
        raise SelectivePoseError(
            f"PR79/S2 archive SHA mismatch: expected {expected_pr79_s2_sha256}, got {source.archive_sha256}"
        )
    exact_anchor = _exact_eval_summary(pr79_s2_exact_json)
    source_score = (
        float(exact_anchor["canonical_score"])
        if exact_anchor is not None and exact_anchor.get("canonical_score")
        else PR79_S2_SCORE
    )
    pr65_pose_qp1 = _load_pr65_pose_qp1(pr65_archive, expected_pr65_sha256)
    pose_plan = _load_pr65_atom_plan(pose_plan_json)
    atom_specs = _selected_atom_specs(pose_plan)
    negative_exact = _exact_eval_summary(negative_exact_json)

    wholesale_spec = PoseCandidateSpec(
        candidate_id="pr79_s2_pr65_pose_wholesale_qp1",
        kind="wholesale_pr65_p1d1_reencoded_qp1_velocity_only",
        selected_atoms=(),
        proxy_benefit=0.0,
        proxy_source="no positive proxy; exact negative exists on nearby C091/C089 frontier",
        risk_flags=(
            "wholesale_pr65_pose_exact_negative_on_nearby_frontier",
            "pr65_direct_transfer_exact_negative_exists",
            "qp1_velocity_col0_only",
        ),
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _build_one(
            source=source,
            unpacker=unpacker,
            spec=wholesale_spec,
            pose_qp1=pr65_pose_qp1,
            pose_brotli_source="PR65 P1D1 decoded then reencoded to QP1",
            output_dir=output_dir,
            force=force,
            source_score=source_score,
            negative_exact=negative_exact,
        )
    ]
    base_pose_qp1 = source.decoded["optimized_poses.qp1"]
    for spec in atom_specs:
        candidates.append(
            _build_one(
                source=source,
                unpacker=unpacker,
                spec=spec,
                pose_qp1=_apply_atoms(base_pose_qp1, spec.selected_atoms),
                pose_brotli_source=spec.proxy_source,
                output_dir=output_dir,
                force=force,
                source_score=source_score,
                negative_exact=negative_exact,
            )
        )

    best = max(
        candidates,
        key=lambda item: (
            bool(item["high_ev_enough_for_exact_eval"]),
            float(item["proxy_margin_vs_pr79_s2"]),
            -int(item["archive_bytes"]),
        ),
    )
    matrix = {
        "anchor_exact_eval": exact_anchor,
        "candidate_matrix": candidates,
        "dispatch_decision": {
            "best_candidate_id": best["candidate_id"],
            "exact_eval_recommended": any(bool(item["high_ev_enough_for_exact_eval"]) for item in candidates),
            "no_remote_dispatch_performed": True,
            "reason": (
                "one or more local archive-valid candidates cleared the proxy screen"
                if any(bool(item["high_ev_enough_for_exact_eval"]) for item in candidates)
                else "no PR79/S2 + PR65 pose candidate cleared the local proxy/risk screen"
            ),
            "required_before_any_exact_eval": [
                "claim a non-conflicting lane with tools/claim_lane_dispatch.py claim",
                "run experiments/contest_auth_eval.py --device cuda on the exact archive SHA/bytes",
                "record contest_auth_eval.json, runtime tree hash, component gates, and terminal dispatch claim",
            ],
        },
        "evidence_grade": "empirical_local_archive_valid_byte_and_proxy_screen",
        "nearby_pr65_pose_exact_negative": negative_exact,
        "no_remote_dispatch_performed": True,
        "pose_sources": {
            "pr79_s2": _pose_summary(source.decoded["optimized_poses.qp1"], "pr79_s2"),
            "pr65": _pose_summary(pr65_pose_qp1, "pr65_p1d1_reencoded_qp1"),
        },
        "score_claim": False,
        "schema": SCHEMA,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", matrix)
    return matrix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr79-s2-archive", type=Path, default=DEFAULT_PR79_S2_ARCHIVE)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--pose-plan-json", type=Path, default=DEFAULT_POSE_PLAN_JSON)
    parser.add_argument("--pr79-s2-exact-json", type=Path, default=DEFAULT_PR79_S2_EXACT_JSON)
    parser.add_argument("--negative-exact-json", type=Path, default=DEFAULT_NEGATIVE_EXACT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-source-sha-mismatch", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    matrix = build_candidates(
        pr79_s2_archive=args.pr79_s2_archive,
        pr65_archive=args.pr65_archive,
        pose_plan_json=args.pose_plan_json,
        pr79_s2_exact_json=args.pr79_s2_exact_json,
        negative_exact_json=args.negative_exact_json,
        output_dir=args.output_dir,
        force=bool(args.force),
        expected_pr79_s2_sha256=None if args.allow_source_sha_mismatch else EXPECTED_PR79_S2_SHA256,
        expected_pr65_sha256=None if args.allow_source_sha_mismatch else EXPECTED_PR65_SHA256,
    )
    print(json.dumps(matrix["candidate_matrix"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
