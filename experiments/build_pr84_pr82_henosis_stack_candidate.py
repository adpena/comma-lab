#!/usr/bin/env python3
"""Build local PR84/QMA9 + PR82/Henosis stack candidates.

This is a local custody and byte-accounting tool.  It never invokes the scorer
and never dispatches remote/GPU work.  Candidates are deterministic archives
that expand PR84's public single-member payload into robust-current runtime
members, then attach charged PR82 ``QPS1`` sidecars for local screening.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.build_pr81_pr82_henosis_stack_candidate import (
    DEFAULT_PR82_ARCHIVE,
    DEFAULT_REPLAY_INFLATE,
    EXPECTED_PR82_SHA256,
    ORIGINAL_VIDEO_BYTES,
    PR82_POSENET_DIST,
    PR82_SEGNET_DIST,
    RUNTIME_INFLATE_RENDERER_PATH,
    _changed_qpost_atoms_from_pr82,
    _classify_qrm1_stream,
    _load_runtime_inflate_renderer,
    _qrm1_exclusion_report,
    _repo_rel,
    _synthetic_raw_delta_proof,
    _write_archive,
    _write_json,
    contest_score_from_components,
)
from tac.henosis_pr82_transfer import (
    QPOST_STREAM_NAMES,
    decode_control_arrays,
    decode_randmulti_groups,
    decode_randmulti_qrm1,
    encode_qpost,
    encode_randmulti_qrm1,
    filter_qpost_streams_to_pairs,
    parse_pr82_bundle,
    parse_replay_contract,
    randmulti_group_summary,
    randmulti_qrm1_parity_profile,
    rank_pairs_by_activity,
    sha256_bytes,
    sha256_path,
    summarize_pair_activity,
)
from tac.qma9_range_mask_contract import (
    parse_qma9_header,
    read_single_member_zip,
    split_qma9_pr81_payload,
)

TOOL = "experiments/build_pr84_pr82_henosis_stack_candidate.py"
SCHEMA = "pr84_pr82_henosis_stack_candidate_v1"
EXPECTED_PR84_SHA256 = "a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f"
EXPECTED_PR84_BYTES = 215_735
EXPECTED_PR84_RANGE_MASK_SHA256 = "4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179"
PR84_REPORTED_SCORE = 0.2751402303839512  # [external: PR-84 reported score (contest-CUDA replay pending verification)]
PR84_POSE_STREAM_BYTES = 899
PR81_REORDERED_QZS3_MAGIC = b"Q81R"
DEFAULT_PR84_DIR = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr84"
DEFAULT_PR84_ARCHIVE = DEFAULT_PR84_DIR / "archive.zip"
DEFAULT_PR84_SOURCE_INFLATE = DEFAULT_PR84_DIR / "sources/inflate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr84_pr82_henosis_stack_20260503_codex"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
ARCHIVE_LAYOUT_EXPANDED_MEMBERS = "expanded_runtime_members"
ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST = "public_payload_plus_qpost"
ARCHIVE_LAYOUTS = (ARCHIVE_LAYOUT_EXPANDED_MEMBERS, ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST)
DEFAULT_TOPK_QPOST_PAIRS: tuple[int, ...] = ()
DEFAULT_TOPK_QRM1_GROUPS: tuple[int, ...] = ()
DEFAULT_QPOST_PAIR_STREAMS = ("post", "shift", "frac", "frac2", "frac3", "bias", "region")


class Pr84Pr82StackError(ValueError):
    """Raised when the PR84/PR82 local stack cannot be built safely."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Pr84Pr82StackError(f"expected JSON object: {path}")
    return payload


def _eval_int_expr(node: ast.AST, constants: Mapping[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_int_expr(node.operand, constants)
        return None if value is None else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_int_expr(node.left, constants)
        right = _eval_int_expr(node.right, constants)
        if left is not None and right is not None:
            return left + right
    return None


def _parse_pr84_source_constants(source_inflate: Path) -> dict[str, int]:
    text = source_inflate.read_text(encoding="utf-8")
    tree = ast.parse(text)
    constants: dict[str, int] = {}
    # Catalog #168 fix 2026-05-12: handle both `X = 16` (Assign) and
    # `X: int = 16` (AnnAssign) module-level constants.
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif (isinstance(node, ast.AnnAssign)
              and node.value is not None):
            target = node.target
            value_node = node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        value = _eval_int_expr(value_node, constants)
        if value is not None:
            constants[target.id] = value
    required = {
        "RANGE_MASK_BYTES",
        "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
        "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
        "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
        "SPLIT_MODEL_REORDERED_BYTES",
        "ROUTER_ACTION_BYTES",
    }
    missing = sorted(required - set(constants))
    if missing:
        raise Pr84Pr82StackError(f"PR84 source inflater missing required constants: {missing}")
    summed_model = (
        constants["SPLIT_MODEL_PACKED_REORDERED_BR_BYTES"]
        + constants["SPLIT_MODEL_SCALES_REORDERED_BR_BYTES"]
        + constants["SPLIT_MODEL_TAIL_REORDERED_BR_BYTES"]
    )
    if summed_model != constants["SPLIT_MODEL_REORDERED_BYTES"]:
        raise Pr84Pr82StackError(
            "PR84 source reordered model constants are inconsistent: "
            f"{summed_model} != {constants['SPLIT_MODEL_REORDERED_BYTES']}"
        )
    constants.setdefault("POSE_STREAM_BYTES", PR84_POSE_STREAM_BYTES)
    return constants


def _pr84_reordered_model_restore_preflight(model_payload: bytes) -> dict[str, Any]:
    runtime = _load_runtime_inflate_renderer()
    try:
        restored = runtime._restore_pr81_reordered_qzs3_model_payload(model_payload)
    except Exception as exc:
        raise Pr84Pr82StackError(
            f"PR84 reordered QZS3 restore preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(restored, (bytes, bytearray)):
        raise Pr84Pr82StackError("PR84 reordered QZS3 restore preflight returned non-bytes")
    restored_bytes = bytes(restored)
    if not restored_bytes.startswith(b"QZS3") or len(restored_bytes) < 6:
        raise Pr84Pr82StackError("PR84 reordered QZS3 restore preflight did not emit QZS3 bytes")
    block_size = int.from_bytes(restored_bytes[4:6], "little")
    if block_size <= 0:
        raise Pr84Pr82StackError(f"PR84 reordered QZS3 restore emitted invalid block size {block_size}")
    return {
        "input_model_payload_bytes": len(model_payload),
        "input_model_payload_sha256": sha256_bytes(model_payload),
        "restored_block_size": block_size,
        "restored_model_bytes": len(restored_bytes),
        "restored_model_sha256": sha256_bytes(restored_bytes),
        "runtime_inflate_renderer": _repo_rel(RUNTIME_INFLATE_RENDERER_PATH),
        "runtime_inflate_renderer_sha256": sha256_path(RUNTIME_INFLATE_RENDERER_PATH),
        "status": "passed",
    }


def _decode_qp1_pose_stream(pose_br: bytes) -> dict[str, Any]:
    try:
        raw = brotli.decompress(pose_br)
    except brotli.error as exc:
        raise Pr84Pr82StackError("PR84 pose stream is not Brotli-decodable") from exc
    if len(raw) < 5 or raw[:3] != b"QP1":
        raise Pr84Pr82StackError(f"PR84 pose stream did not decode to QP1: {raw[:4]!r}")
    values = [int.from_bytes(raw[3:5], "little")]
    cursor = 5
    while cursor < len(raw):
        shift = 0
        acc = 0
        while True:
            if cursor >= len(raw):
                raise Pr84Pr82StackError("PR84 QP1 pose stream is truncated")
            byte = raw[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
            if shift > 63:
                raise Pr84Pr82StackError("PR84 QP1 pose stream has overlong VLQ")
        delta = (acc >> 1) ^ -(acc & 1)
        values.append((values[-1] + delta) & 0xFFFF)
    return {
        "decoded_pose_bytes": len(raw),
        "decoded_pose_count": len(values),
        "decoded_pose_sha256": sha256_bytes(raw),
        "encoded_pose_bytes": len(pose_br),
        "encoded_pose_sha256": sha256_bytes(pose_br),
        "magic": "QP1",
        "status": "passed",
    }


def _load_runtime_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("robust_current_unpack_renderer_payload", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise Pr84Pr82StackError(f"cannot load runtime payload unpacker: {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _public_payload_unpack_preflight(payload: bytes) -> dict[str, Any]:
    """Prove robust_current can unpack PR84's public single-member payload."""

    unpacker = _load_runtime_unpacker()
    with tempfile.TemporaryDirectory(prefix="pr84_public_payload_unpack_") as tmp:
        archive_dir = Path(tmp)
        (archive_dir / "p").write_bytes(payload)
        try:
            summary = unpacker.unpack_renderer_payload(archive_dir)
        except Exception as exc:
            raise Pr84Pr82StackError(
                f"PR84 public payload unpack preflight failed: {type(exc).__name__}: {exc}"
            ) from exc
        members = {
            row["name"]: {
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in summary.get("members", [])
        }
        required = {"masks.qma9", "renderer.bin", "optimized_poses.qp1"}
        missing = sorted(required - set(members))
        if missing:
            raise Pr84Pr82StackError(f"PR84 public payload unpack preflight missing members: {missing}")
        renderer = (archive_dir / "renderer.bin").read_bytes()
        pose = (archive_dir / "optimized_poses.qp1").read_bytes()
        masks = (archive_dir / "masks.qma9").read_bytes()
        if not renderer.startswith(PR81_REORDERED_QZS3_MAGIC):
            raise Pr84Pr82StackError("PR84 public payload unpack preflight emitted non-Q81R renderer")
        if not pose.startswith(b"QP1"):
            raise Pr84Pr82StackError("PR84 public payload unpack preflight emitted non-QP1 pose stream")
        if not masks.startswith(b"QMA9"):
            raise Pr84Pr82StackError("PR84 public payload unpack preflight emitted non-QMA9 masks")
    return {
        "payload_bytes": len(payload),
        "payload_format": summary.get("payload_format"),
        "schema": summary.get("schema"),
        "members": members,
        "runtime_unpacker": _repo_rel(UNPACKER_PATH),
        "runtime_unpacker_sha256": sha256_path(UNPACKER_PATH),
        "status": "passed",
    }


def _load_pr84_source(
    archive_path: Path,
    source_inflate: Path,
    *,
    expected_sha256: str | None,
    expected_bytes: int | None,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    if expected_bytes is not None and archive_path.stat().st_size != int(expected_bytes):
        raise Pr84Pr82StackError(
            f"PR84 archive byte mismatch: expected {expected_bytes}, got {archive_path.stat().st_size}"
        )
    if expected_sha256 is not None:
        actual = sha256_path(archive_path)
        if actual != expected_sha256:
            raise Pr84Pr82StackError(f"PR84 archive SHA mismatch: expected {expected_sha256}, got {actual}")
    constants = _parse_pr84_source_constants(source_inflate)
    payload, zip_profile = read_single_member_zip(archive_path, expected_member="p")
    base_payload_bytes = (
        int(constants["RANGE_MASK_BYTES"])
        + int(constants["SPLIT_MODEL_REORDERED_BYTES"])
        + int(constants["POSE_STREAM_BYTES"])
    )
    if len(payload) not in {base_payload_bytes, base_payload_bytes + int(constants["ROUTER_ACTION_BYTES"])}:
        raise Pr84Pr82StackError(
            "PR84 payload length does not match source inflater fixed-slice contract: "
            f"payload={len(payload)} expected={base_payload_bytes}"
            f" or {base_payload_bytes + int(constants['ROUTER_ACTION_BYTES'])}"
        )
    router_bytes = len(payload) - base_payload_bytes
    split = split_qma9_pr81_payload(
        payload,
        range_mask_bytes=int(constants["RANGE_MASK_BYTES"]),
        model_bytes=int(constants["SPLIT_MODEL_REORDERED_BYTES"]),
        pose_bytes=int(constants["POSE_STREAM_BYTES"]),
        router_bytes=router_bytes,
    )
    qma9_header = parse_qma9_header(split.range_mask)
    if expected_sha256 is not None and sha256_bytes(split.range_mask) != EXPECTED_PR84_RANGE_MASK_SHA256:
        raise Pr84Pr82StackError(
            "PR84 range_mask.qma9 SHA mismatch: "
            f"expected {EXPECTED_PR84_RANGE_MASK_SHA256}, got {sha256_bytes(split.range_mask)}"
        )
    if qma9_header.frame_count != 600 or (qma9_header.width, qma9_header.height) not in {(512, 384), (384, 512)}:
        raise Pr84Pr82StackError(
            f"unexpected PR84 QMA9 mask shape: {(qma9_header.frame_count, qma9_header.width, qma9_header.height)}"
        )
    restore_preflight = _pr84_reordered_model_restore_preflight(split.model)
    pose_preflight = _decode_qp1_pose_stream(split.pose)
    if pose_preflight["decoded_pose_count"] != 600:
        raise Pr84Pr82StackError(f"PR84 QP1 decoded {pose_preflight['decoded_pose_count']} poses, expected 600")
    unpack_preflight = _public_payload_unpack_preflight(payload)
    return {
        "archive": zip_profile,
        "payload": payload,
        "source_constants": constants,
        "source_inflate": source_inflate.resolve(),
        "source_inflate_sha256": sha256_path(source_inflate),
        "qma9_header": qma9_header,
        "payload_contract": {
            "base_payload_bytes_without_router": base_payload_bytes,
            "contract": "pr84_qma9_reordered_qzs3_qp1_no_router"
            if router_bytes == 0
            else "pr81_style_qma9_reordered_qzs3_qp1_router",
            "expected_pr84_no_router": router_bytes == 0,
            "router_action_bytes": router_bytes,
            "stale_pr81_packed_payload_bytes_avoided": True,
        },
        "runtime_restore_preflight": restore_preflight,
        "public_payload_unpack_preflight": unpack_preflight,
        "pose_preflight": pose_preflight,
        "split": split,
    }


def _load_pr82_source(path: Path, replay_inflate: Path, *, expected_sha256: str | None) -> dict[str, Any]:
    path = path.resolve()
    if expected_sha256 is not None:
        actual = sha256_path(path)
        if actual != expected_sha256:
            raise Pr84Pr82StackError(f"PR82 archive SHA mismatch: expected {expected_sha256}, got {actual}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise Pr84Pr82StackError(f"{path}: expected single member 'x', got {names!r}")
        raw = zf.read("x")
    contract = parse_replay_contract(replay_inflate)
    bundle = parse_pr82_bundle(raw, contract)
    return {
        "archive_bytes": path.stat().st_size,
        "archive_path": path,
        "archive_sha256": sha256_path(path),
        "bundle": bundle,
        "contract": contract,
        "payload_bytes": len(raw),
        "payload_sha256": sha256_bytes(raw),
    }


def _runtime_members_from_pr84(pr84: Mapping[str, Any]) -> dict[str, bytes]:
    split = pr84["split"]
    pose = brotli.decompress(split.pose)
    members = {
        "masks.qma9": split.range_mask,
        "optimized_poses.qp1": pose,
        "renderer.bin": PR81_REORDERED_QZS3_MAGIC + split.model,
    }
    if split.router:
        members["router_actions.3bit"] = split.router
    return members


def _archive_members_for_layout(
    *,
    pr84: Mapping[str, Any],
    qpost: bytes,
    archive_layout: str,
) -> dict[str, bytes]:
    if archive_layout == ARCHIVE_LAYOUT_EXPANDED_MEMBERS:
        members = _runtime_members_from_pr84(pr84)
        members["qpost.bin"] = qpost
        return members
    if archive_layout == ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST:
        return {"p": pr84["payload"], "qpost.bin": qpost}
    raise Pr84Pr82StackError(f"unsupported PR84 archive layout: {archive_layout!r}")


def _member_table(members: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "bytes": len(data),
            "name": name,
            "sha256": sha256_bytes(data),
        }
        for name, data in sorted(members.items())
    ]


def _layout_candidate_id(candidate_id: str, archive_layout: str) -> str:
    if archive_layout == ARCHIVE_LAYOUT_EXPANDED_MEMBERS:
        return candidate_id
    if archive_layout == ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST:
        return f"{candidate_id}_packedp"
    raise Pr84Pr82StackError(f"unsupported PR84 archive layout: {archive_layout!r}")


def _layout_candidate_kind(candidate_kind: str, archive_layout: str) -> str:
    if archive_layout == ARCHIVE_LAYOUT_EXPANDED_MEMBERS:
        return candidate_kind
    if archive_layout == ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST:
        return candidate_kind.replace("pr84_expanded_qma9_qzs3_qp1", "pr84_public_payload_p")
    raise Pr84Pr82StackError(f"unsupported PR84 archive layout: {archive_layout!r}")


def _build_qpost_candidate(
    *,
    pr84: Mapping[str, Any],
    streams: Mapping[str, bytes],
    candidate_id: str,
    candidate_kind: str,
    qpost_contract: str,
    changed_atoms: int,
    runtime_blockers: Sequence[str],
    output_dir: Path,
    archive_layout: str = ARCHIVE_LAYOUT_EXPANDED_MEMBERS,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    qpost = encode_qpost(streams)
    candidate_id = _layout_candidate_id(candidate_id, archive_layout)
    candidate_kind = _layout_candidate_kind(candidate_kind, archive_layout)
    archive = output_dir / candidate_id / "archive.zip"
    members = _archive_members_for_layout(pr84=pr84, qpost=qpost, archive_layout=archive_layout)
    _write_archive(archive, members)
    candidate_dir = output_dir / candidate_id
    blockers = list(runtime_blockers)
    raw_output_delta_proof = None
    try:
        raw_output_delta_proof = _synthetic_raw_delta_proof(
            archive_path=archive,
            candidate_dir=candidate_dir,
            qpost=qpost,
        )
    except Exception as exc:
        blockers.append(str(exc))
    if changed_atoms <= 0:
        blockers.append("candidate is a no-op after stream filtering")
    if raw_output_delta_proof is None:
        blockers.append("missing positive raw-output delta proof")
    dispatch_ready = len(blockers) == 0
    archive_bytes = archive.stat().st_size
    manifest: dict[str, Any] = {
        "archive_bytes": archive_bytes,
        "archive_path": _repo_rel(archive),
        "archive_sha256": sha256_path(archive),
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "dispatch_gate": {
            "dispatch_ready_now": dispatch_ready,
            "dispatch_claim_required_before_any_exact_eval": True,
            "no_remote_dispatch": True,
            "reason": "ready_for_exact_cuda_eval_after_lane_claim" if dispatch_ready else "; ".join(blockers),
            "remote_dispatch_performed": False,
            "score_claim": False,
        },
        "evidence_grade": "empirical_local_archive_build_and_runtime_preflight_only",
        "manifest_schema": SCHEMA,
        "no_op_detection": {
            "changed_atom_count": int(changed_atoms),
            "is_noop": int(changed_atoms) == 0,
        },
        "output_archive": {
            "bytes": archive_bytes,
            "layout": archive_layout,
            "members": sorted(members),
            "member_table": _member_table(members),
            "path": _repo_rel(archive),
            "sha256": sha256_path(archive),
        },
        "payload_packing_parity": {
            "archive_layout": archive_layout,
            "exact_pr84_public_payload_preserved": (
                archive_layout == ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST
                and members.get("p") == pr84["payload"]
            ),
            "output_p_member_bytes": len(members["p"]) if "p" in members else None,
            "output_p_member_sha256": sha256_bytes(members["p"]) if "p" in members else None,
            "source_pr84_payload_bytes": int(pr84["archive"].member_bytes),
            "source_pr84_payload_sha256": pr84["archive"].member_sha256,
            "source_runtime_unpack_preflight": dict(pr84["public_payload_unpack_preflight"]),
        },
        "qpost": {
            "bytes": len(qpost),
            "runtime_contract": qpost_contract,
            "sha256": sha256_bytes(qpost),
        },
        "raw_output_delta_proof": dict(raw_output_delta_proof) if raw_output_delta_proof is not None else None,
        "score_claim": False,
        "source_pr84_archive": {
            "bytes": int(pr84["archive"].archive_bytes),
            "path": _repo_rel(Path(pr84["archive"].archive_path)),
            "reported_score_external": PR84_REPORTED_SCORE,
            "sha256": pr84["archive"].archive_sha256,
        },
        "source_pr84_payload_preflight": {
            "payload_bytes": int(pr84["archive"].member_bytes),
            "payload_contract": dict(pr84["payload_contract"]),
            "payload_sha256": pr84["archive"].member_sha256,
            "pose": dict(pr84["pose_preflight"]),
            "qma9": {
                "decoded_mask_bytes": int(pr84["qma9_header"].decoded_mask_bytes),
                "frame_count": int(pr84["qma9_header"].frame_count),
                "height": int(pr84["qma9_header"].height),
                "packed_bytes": int(pr84["qma9_header"].packed_bytes),
                "width": int(pr84["qma9_header"].width),
            },
            "runtime_restore_preflight": dict(pr84["runtime_restore_preflight"]),
            "source_inflate": _repo_rel(Path(pr84["source_inflate"])),
            "source_inflate_sha256": pr84["source_inflate_sha256"],
        },
        "static_score_band_if_pr82_components_carried": {
            "archive_bytes": archive_bytes,
            "bytes_delta_vs_pr84_public_archive": archive_bytes - int(pr84["archive"].archive_bytes),
            "component_assumption": "PR82 exact T4 components copied unchanged; planning-only lower bound, not evidence",
            "expected_score": contest_score_from_components(
                archive_bytes,
                segnet_dist=PR82_SEGNET_DIST,
                posenet_dist=PR82_POSENET_DIST,
            ),
            "pr82_posenet_dist": PR82_POSENET_DIST,
            "pr82_segnet_dist": PR82_SEGNET_DIST,
        },
        "stream_delta": {
            "archive_delta_bytes_vs_pr84_public_archive": archive_bytes - int(pr84["archive"].archive_bytes),
            "qpost_charged_member_bytes": len(qpost),
        },
        "tool": TOOL,
    }
    if extra:
        manifest.update(extra)
    _write_json(output_dir / candidate_id / "manifest.json", manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "archive_layout": archive_layout,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "dispatch_gate": manifest["dispatch_gate"],
        "manifest_path": _repo_rel(output_dir / candidate_id / "manifest.json"),
        "qpost_bytes": len(qpost),
        "score_if_pr82_components_carried": manifest["static_score_band_if_pr82_components_carried"]["expected_score"],
        "stream_delta": manifest["stream_delta"],
    }


def _count_selected_control_atoms(
    arrays: Mapping[str, np.ndarray],
    selected_pairs: Sequence[int],
    include_streams: Sequence[str],
) -> int:
    defaults = {"shift": 40, "frac": 4, "frac2": 4, "frac3": 4, "bias": 13, "region": 0}
    total = 0
    for pair in selected_pairs:
        pair_i = int(pair)
        for name in include_streams:
            if name == "post":
                total += int(np.count_nonzero(arrays[name][:, pair_i] != 0))
            else:
                total += int(arrays[name][pair_i] != defaults[name])
    return total


def build_candidates(
    *,
    pr84_archive: Path = DEFAULT_PR84_ARCHIVE,
    pr84_source_inflate: Path = DEFAULT_PR84_SOURCE_INFLATE,
    pr82_archive: Path = DEFAULT_PR82_ARCHIVE,
    replay_inflate: Path = DEFAULT_REPLAY_INFLATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_pr84_sha256: str | None = EXPECTED_PR84_SHA256,
    expected_pr84_bytes: int | None = EXPECTED_PR84_BYTES,
    expected_pr82_sha256: str | None = EXPECTED_PR82_SHA256,
    archive_layout: str = ARCHIVE_LAYOUT_EXPANDED_MEMBERS,
    qpost_pair_topks: Sequence[int] = DEFAULT_TOPK_QPOST_PAIRS,
    qrm1_group_topks: Sequence[int] = DEFAULT_TOPK_QRM1_GROUPS,
    qpost_pair_streams: Sequence[str] = DEFAULT_QPOST_PAIR_STREAMS,
) -> dict[str, Any]:
    if archive_layout not in ARCHIVE_LAYOUTS:
        raise Pr84Pr82StackError(f"unsupported PR84 archive layout: {archive_layout!r}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pr84 = _load_pr84_source(
        pr84_archive,
        pr84_source_inflate,
        expected_sha256=expected_pr84_sha256,
        expected_bytes=expected_pr84_bytes,
    )
    pr82 = _load_pr82_source(pr82_archive, replay_inflate, expected_sha256=expected_pr82_sha256)
    bundle = pr82["bundle"]
    groups = decode_randmulti_groups(bundle.encoded_segments["randmulti"], pr82["contract"].randmulti_specs)
    qrm1 = encode_randmulti_qrm1(groups)
    qrm1_support_report = _classify_qrm1_stream(qrm1)
    qrm1_exclusion = _qrm1_exclusion_report(groups, support_report=qrm1_support_report)
    if not bool(qrm1_support_report.get("dispatchable_qrm1")):
        raise Pr84Pr82StackError(
            "PR82 all-72 QRM1 stream is not runtime-dispatchable; refusing PR84 stack build: "
            f"{qrm1_support_report}"
        )
    qrm1_decoded = decode_randmulti_qrm1(qrm1, pr82["contract"].randmulti_specs)
    qrm1_profile = randmulti_qrm1_parity_profile(
        groups,
        qrm1_decoded,
        encoded=qrm1,
        source_encoded=bundle.encoded_segments["randmulti"],
    )
    controls_streams = {name: bundle.encoded_segments[name] for name in QPOST_STREAM_NAMES if name != "randmulti"}
    controls_streams["randmulti"] = b""
    control_atoms = _changed_qpost_atoms_from_pr82(bundle, include_randmulti=False)
    qrm1_streams = dict.fromkeys(QPOST_STREAM_NAMES, b"")
    qrm1_streams["randmulti"] = qrm1
    qrm1_atoms = int(sum(np.count_nonzero(group.rows) for group in groups))
    common_blockers: list[str] = []
    qps1_controls = _build_qpost_candidate(
        pr84=pr84,
        streams=controls_streams,
        candidate_id="pr84_qma9_pr82_qps1_controls_all600",
        candidate_kind="pr84_expanded_qma9_qzs3_qp1_plus_pr82_qps1_control_sidecar",
        qpost_contract="QPS1/post_shift_frac_bias_region_without_randmulti",
        changed_atoms=control_atoms,
        runtime_blockers=common_blockers,
        output_dir=output_dir,
        archive_layout=archive_layout,
    )
    qps1_qrm1 = _build_qpost_candidate(
        pr84=pr84,
        streams=qrm1_streams,
        candidate_id="pr84_qma9_pr82_qps1_qrm1_all072_randmulti",
        candidate_kind="pr84_expanded_qma9_qzs3_qp1_plus_pr82_qrm1_all72_randmulti_sidecar",
        qpost_contract="QPS1/QRM1_native_sparse_group_id_randmulti_all72",
        changed_atoms=qrm1_atoms,
        runtime_blockers=common_blockers,
        output_dir=output_dir,
        archive_layout=archive_layout,
        extra={
            "randmulti": {
                "excluded_group_policy": qrm1_exclusion,
                "groups": [randmulti_group_summary(group) for group in groups],
                "local_decode_profile": qrm1_profile,
                "runtime_support_report": qrm1_support_report,
                "semantic_scope": "all 72 PR82 randmulti groups represented exactly at sparse row level",
            }
        },
    )
    full_streams = dict(controls_streams)
    full_streams["randmulti"] = qrm1
    qps1_full = _build_qpost_candidate(
        pr84=pr84,
        streams=full_streams,
        candidate_id="pr84_qma9_pr82_qps1_controls_qrm1_all072",
        candidate_kind="pr84_expanded_qma9_qzs3_qp1_plus_pr82_controls_and_qrm1_all72_sidecar",
        qpost_contract="QPS1/full_controls_plus_QRM1_all72",
        changed_atoms=control_atoms + qrm1_atoms,
        runtime_blockers=common_blockers,
        output_dir=output_dir,
        archive_layout=archive_layout,
        extra={
            "randmulti": {
                "excluded_group_policy": qrm1_exclusion,
                "groups": [randmulti_group_summary(group) for group in groups],
                "local_decode_profile": qrm1_profile,
                "runtime_support_report": qrm1_support_report,
                "semantic_scope": "all PR82 controls plus all 72 randmulti groups represented as charged sidecar bytes",
            }
        },
    )
    candidate_rows = [qps1_controls, qps1_qrm1, qps1_full]
    atom_filtered_candidates: list[dict[str, Any]] = []
    arrays = decode_control_arrays(bundle.encoded_segments)
    ranked_pairs = rank_pairs_by_activity(summarize_pair_activity(arrays))
    for topk in qpost_pair_topks:
        topk_i = int(topk)
        if topk_i <= 0:
            continue
        selected_pairs = tuple(sorted(ranked_pairs[:topk_i]))
        streams = filter_qpost_streams_to_pairs(
            bundle.encoded_segments,
            selected_pairs,
            include_streams=tuple(qpost_pair_streams),
        )
        active_atoms = _count_selected_control_atoms(arrays, selected_pairs, qpost_pair_streams)
        atom_filtered_candidates.append(
            _build_qpost_candidate(
                pr84=pr84,
                streams=streams,
                candidate_id=f"pr84_qma9_pr82_qps1_controls_top{topk_i:03d}",
                candidate_kind="pr84_expanded_qma9_qzs3_qp1_plus_pr82_qps1_control_pair_topk_sidecar",
                qpost_contract=f"QPS1/top{topk_i:03d}_control_pair_filter",
                changed_atoms=active_atoms,
                runtime_blockers=common_blockers,
                output_dir=output_dir,
                archive_layout=archive_layout,
                extra={
                    "atom_filter": {
                        "filter_kind": "rank_pairs_by_pr82_control_activity",
                        "include_streams": list(qpost_pair_streams),
                        "selected_pair_count": len(selected_pairs),
                        "selected_pairs": list(selected_pairs),
                        "topk": topk_i,
                    }
                },
            )
        )
    qrm1_groups_ranked = sorted(
        groups,
        key=lambda group: (-int(np.count_nonzero(group.rows)), int(group.group_index)),
    )
    for topk in qrm1_group_topks:
        topk_i = int(topk)
        if topk_i <= 0:
            continue
        selected_groups = tuple(qrm1_groups_ranked[:topk_i])
        if not selected_groups:
            continue
        selected_qrm1 = encode_randmulti_qrm1(selected_groups)
        selected_support = _classify_qrm1_stream(selected_qrm1)
        runtime_blockers = list(common_blockers)
        if not bool(selected_support.get("dispatchable_qrm1")):
            runtime_blockers.append(f"selected QRM1 top{topk_i:03d} is not runtime-dispatchable: {selected_support}")
        selected_decoded = decode_randmulti_qrm1(selected_qrm1, pr82["contract"].randmulti_specs)
        selected_profile = randmulti_qrm1_parity_profile(
            selected_groups,
            selected_decoded,
            encoded=selected_qrm1,
            source_encoded=bundle.encoded_segments["randmulti"],
        )
        streams = dict.fromkeys(QPOST_STREAM_NAMES, b"")
        streams["randmulti"] = selected_qrm1
        atom_filtered_candidates.append(
            _build_qpost_candidate(
                pr84=pr84,
                streams=streams,
                candidate_id=f"pr84_qma9_pr82_qps1_qrm1_top{topk_i:03d}",
                candidate_kind="pr84_expanded_qma9_qzs3_qp1_plus_pr82_qrm1_topk_group_sidecar",
                qpost_contract=f"QPS1/QRM1_native_sparse_group_id_randmulti_top{topk_i:03d}",
                changed_atoms=int(sum(np.count_nonzero(group.rows) for group in selected_groups)),
                runtime_blockers=runtime_blockers,
                output_dir=output_dir,
                archive_layout=archive_layout,
                extra={
                    "randmulti": {
                        "groups": [randmulti_group_summary(group) for group in selected_decoded],
                        "local_decode_profile": selected_profile,
                        "runtime_support_report": selected_support,
                        "semantic_scope": f"top {topk_i} PR82 randmulti groups by sparse nonzero count",
                    }
                },
            )
        )
    candidate_rows.extend(atom_filtered_candidates)
    summary = {
        "atom_filter_config": {
            "qpost_pair_streams": list(qpost_pair_streams),
            "qpost_pair_topks": [int(value) for value in qpost_pair_topks],
            "qrm1_group_topks": [int(value) for value in qrm1_group_topks],
        },
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "evidence_grade": "empirical_local_archive_build_and_runtime_preflight_only",
        "exact_scores_used_for_planning": {
            "pr84_external_reported": {
                "archive_bytes": EXPECTED_PR84_BYTES,
                "archive_sha256": EXPECTED_PR84_SHA256,
                "score": PR84_REPORTED_SCORE,
            },
            "pr82_components": {
                "posenet_dist": PR82_POSENET_DIST,
                "segnet_dist": PR82_SEGNET_DIST,
            },
        },
        "highest_ev_local_candidate": min(candidate_rows, key=lambda row: int(row["archive_bytes"])),
        "no_remote_dispatch": True,
        "original_video_bytes": ORIGINAL_VIDEO_BYTES,
        "archive_layout": archive_layout,
        "pr82_profile": {
            "archive_bytes": pr82["archive_bytes"],
            "archive_sha256": pr82["archive_sha256"],
            "contract": {
                "fixed_bias_bytes": pr82["contract"].fixed_bias_bytes,
                "fixed_region_bytes": pr82["contract"].fixed_region_bytes,
                "randmulti_group_count": len(pr82["contract"].randmulti_specs),
                "replay_inflate": _repo_rel(replay_inflate),
                "replay_inflate_sha256": pr82["contract"].source_sha256,
            },
            "payload_bytes": pr82["payload_bytes"],
            "payload_sha256": pr82["payload_sha256"],
        },
        "pr84_profile": {
            "archive_bytes": int(pr84["archive"].archive_bytes),
            "archive_sha256": pr84["archive"].archive_sha256,
            "payload_bytes": int(pr84["archive"].member_bytes),
            "payload_sha256": pr84["archive"].member_sha256,
            "payload_contract": dict(pr84["payload_contract"]),
            "public_payload_unpack_preflight": dict(pr84["public_payload_unpack_preflight"]),
            "runtime_member_contract": (
                "archive keeps public member p; inflate.sh expands it via unpack_renderer_payload.py, then applies qpost.bin"
                if archive_layout == ARCHIVE_LAYOUT_PUBLIC_PAYLOAD_QPOST
                else "expanded robust_current members: masks.qma9, renderer.bin(Q81R), optimized_poses.qp1, optional qpost.bin"
            ),
            "segments": [
                {
                    "bytes": segment.size_bytes,
                    "codec": segment.codec,
                    "name": segment.name,
                    "offset": segment.offset,
                    "sha256": segment.sha256,
                }
                for segment in pr84["split"].segments
            ],
            "source_inflate": _repo_rel(Path(pr84["source_inflate"])),
            "source_inflate_sha256": pr84["source_inflate_sha256"],
        },
        "schema": SCHEMA,
        "score_claim": False,
        "tool": TOOL,
        "unsupported_semantics_before_t4_dispatch": [
            "no exact CUDA eval was run by this builder",
            "a lane dispatch claim is required before any future exact eval",
            "static score bands assume PR82 components carry and are not score evidence",
        ],
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    (output_dir / "DESIGN_NOTE.md").write_text(
        "# PR84 QMA9 + PR82 Henosis Stack\n\n"
        "No remote GPU dispatch was performed.\n\n"
        f"Archive layout: `{archive_layout}`.\n\n"
        "No exact CUDA eval or score claim was made.\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr84-archive", type=Path, default=DEFAULT_PR84_ARCHIVE)
    parser.add_argument("--pr84-source-inflate", type=Path, default=DEFAULT_PR84_SOURCE_INFLATE)
    parser.add_argument("--pr82-archive", type=Path, default=DEFAULT_PR82_ARCHIVE)
    parser.add_argument("--replay-inflate", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-pr84-sha-check", action="store_true")
    parser.add_argument("--no-pr84-byte-check", action="store_true")
    parser.add_argument("--no-pr82-sha-check", action="store_true")
    parser.add_argument(
        "--archive-layout",
        choices=ARCHIVE_LAYOUTS,
        default=ARCHIVE_LAYOUT_EXPANDED_MEMBERS,
        help=(
            "Candidate archive packing. Default preserves the existing expanded-member "
            "builder output; public_payload_plus_qpost keeps PR84 member p intact and adds qpost.bin."
        ),
    )
    parser.add_argument(
        "--qpost-pair-topks",
        default="",
        help="Comma-separated PR82 control-pair top-k filters to build, or empty/none to skip.",
    )
    parser.add_argument(
        "--qrm1-group-topks",
        default="",
        help="Comma-separated PR82 QRM1 group top-k filters to build, or empty/none to skip.",
    )
    parser.add_argument(
        "--qpost-pair-streams",
        default=",".join(DEFAULT_QPOST_PAIR_STREAMS),
        help="Comma-separated QPOST control streams for pair-topk candidates.",
    )
    return parser


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() in {"", "none"}:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_streams(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(set(values) - (set(QPOST_STREAM_NAMES) - {"randmulti"}))
    if unknown:
        raise Pr84Pr82StackError(f"unknown or unsupported QPOST pair stream(s): {unknown}")
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_candidates(
        pr84_archive=args.pr84_archive,
        pr84_source_inflate=args.pr84_source_inflate,
        pr82_archive=args.pr82_archive,
        replay_inflate=args.replay_inflate,
        output_dir=args.output_dir,
        expected_pr84_sha256=None if args.no_pr84_sha_check else EXPECTED_PR84_SHA256,
        expected_pr84_bytes=None if args.no_pr84_byte_check else EXPECTED_PR84_BYTES,
        expected_pr82_sha256=None if args.no_pr82_sha_check else EXPECTED_PR82_SHA256,
        archive_layout=args.archive_layout,
        qpost_pair_topks=_parse_csv_ints(args.qpost_pair_topks),
        qrm1_group_topks=_parse_csv_ints(args.qrm1_group_topks),
        qpost_pair_streams=_parse_csv_streams(args.qpost_pair_streams),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
