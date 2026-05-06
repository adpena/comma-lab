"""Analyze PR90 topband masks as a PR85 QMA9 mask-prior transfer.

This is a local, planning-only analyzer. It reads the public PR90 intake archive
and the PR85 intake bundle already present in ``experiments/results/``, compares
their decoded semantic mask tensors, and emits a ranked policy JSON. It never
dispatches, trains, runs the scorer, or claims score.
"""

from __future__ import annotations

import argparse
import brotli
import importlib
import json
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import parse_pr85_bundle, validate_pr85_member_name  # noqa: E402
from tac.qma9_range_mask_contract import (  # noqa: E402
    ORIGINAL_VIDEO_BYTES,
    parse_qma9_header,
    sha256_bytes,
    sha256_file,
)


TOOL = "experiments/analyze_pr90_qma9_mask_prior_transfer.py"
SCHEMA = "pr90_qma9_mask_prior_transfer_analysis_v1"
POLICY_SCHEMA = "pr90_qma9_mask_prior_transfer_ranked_policy_v1"

DEFAULT_PR90_DIR = REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker"
DEFAULT_PR85_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker"
DEFAULT_ANALYSIS_JSON = DEFAULT_OUTPUT_DIR / "analysis.json"
DEFAULT_POLICY_JSON = DEFAULT_OUTPUT_DIR / "ranked_candidate_policy.json"
DEFAULT_REPORT_MD = DEFAULT_OUTPUT_DIR / "summary.md"

PR90_COMPACT_MASK_BODY_BYTES = 152_431
PR90_COMPACT_MODEL_BODY_BYTES = 56_385
PR90_STBM_MAGIC = b"STBM1BR\0"
PR90_QTBM_MAGICS = {
    b"QTBM1\0": {"has_residual_order": False, "sparse_plain": False},
    b"QTBM2\0": {"has_residual_order": True, "sparse_plain": False},
    b"QTBM3\0": {"has_residual_order": True, "sparse_plain": True},
    b"QTBM4\0": {"has_residual_order": True, "sparse_plain": True},
    b"QTBM5\0": {"has_residual_order": True, "sparse_plain": True},
}
PR90_RESIDUAL_SYMBOLS = 4
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES


class TransferAnalysisError(ValueError):
    """Raised when the local transfer analysis cannot be built safely."""


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_strict_single_member_zip(path: Path, *, expected_member: str) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise TransferAnalysisError(f"archive is missing: {_repo_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_member]:
            raise TransferAnalysisError(
                f"{_repo_rel(path)} must contain exactly one member {expected_member!r}; got {names!r}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise TransferAnalysisError(f"{_repo_rel(path)}:{expected_member} must be ZIP_STORED")
        if expected_member == "x":
            validate_pr85_member_name(info.filename)
        payload = zf.read(info)
        if len(payload) != int(info.file_size):
            raise TransferAnalysisError(f"{_repo_rel(path)}:{expected_member} size mismatch")
    archive_bytes = int(path.stat().st_size)
    return payload, {
        "archive_path": _repo_rel(path),
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256_file(path),
        "member_name": expected_member,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_overhead_bytes": archive_bytes - len(payload),
    }


def _class_counts(raw: bytes | np.ndarray) -> dict[str, int]:
    arr = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray, memoryview)) else raw.reshape(-1)
    counts = np.bincount(arr.astype(np.uint8, copy=False), minlength=5)
    return {str(cls): int(counts[cls]) for cls in range(5)}


def parse_pr90_qtbm_blob(blob: bytes) -> dict[str, Any]:
    """Parse the actual PR90 QTBM topband stream header and component slices."""

    pos = 0
    matched_magic = None
    magic_config: dict[str, Any] | None = None
    for magic, config in sorted(PR90_QTBM_MAGICS.items(), key=lambda item: len(item[0]), reverse=True):
        if blob.startswith(magic):
            matched_magic = magic
            magic_config = config
            pos = len(magic)
            break
    if matched_magic is None or magic_config is None:
        raise TransferAnalysisError(f"PR90 topband blob has unsupported magic: {blob[:8]!r}")

    if pos + struct.calcsize("<HHHBBBbb") > len(blob):
        raise TransferAnalysisError("PR90 topband blob is truncated before stream dimensions")
    n_pairs, height, width, precision, top_bins, boundary_xbins, shift_dy, shift_dx = struct.unpack_from(
        "<HHHBBBbb", blob, pos
    )
    pos += struct.calcsize("<HHHBBBbb")

    residual_order: list[int] | None = None
    if bool(magic_config["has_residual_order"]):
        if pos + PR90_RESIDUAL_SYMBOLS > len(blob):
            raise TransferAnalysisError("PR90 topband blob is truncated before residual order")
        residual_order = [int(v) for v in blob[pos : pos + PR90_RESIDUAL_SYMBOLS]]
        if sorted(residual_order) != [0, 1, 2, 3]:
            raise TransferAnalysisError(f"invalid PR90 residual order: {residual_order!r}")
        pos += PR90_RESIDUAL_SYMBOLS

    if pos + 8 > len(blob):
        raise TransferAnalysisError("PR90 topband blob is truncated before top/road lengths")
    top_len, road_len = struct.unpack_from("<II", blob, pos)
    pos += 8
    top_start = pos
    top_end = top_start + int(top_len)
    road_start = top_end
    road_end = road_start + int(road_len)
    if road_end > len(blob):
        raise TransferAnalysisError("PR90 topband blob top/road payloads exceed blob length")
    top_payload = blob[top_start:top_end]
    road_payload = blob[road_start:road_end]
    pos = road_end

    if pos + 4 > len(blob):
        raise TransferAnalysisError("PR90 topband blob is truncated before sparse table lengths")
    spatial_size, m5_size = struct.unpack_from("<HH", blob, pos)
    pos += 4
    spatial_start = pos
    spatial_end = spatial_start + int(spatial_size)
    m5_start = spatial_end
    m5_end = m5_start + int(m5_size)
    if m5_end > len(blob):
        raise TransferAnalysisError("PR90 sparse table slices exceed blob length")
    pos = m5_end

    if pos >= len(blob):
        raise TransferAnalysisError("PR90 topband blob is missing sparse feature count")
    n_feats = int(blob[pos])
    pos += 1
    feat_ids = [int(v) for v in blob[pos : pos + n_feats]]
    if len(feat_ids) != n_feats:
        raise TransferAnalysisError("PR90 topband blob is truncated in sparse feature ids")
    pos += n_feats

    if pos + 6 > len(blob):
        raise TransferAnalysisError("PR90 topband blob is truncated before sparse payload")
    threshold_q8, sparse_len = struct.unpack_from("<HI", blob, pos)
    pos += 6
    sparse_start = pos
    sparse_end = sparse_start + int(sparse_len)
    if sparse_end + 4 > len(blob):
        raise TransferAnalysisError("PR90 sparse payload exceeds blob length")
    pos = sparse_end
    (bitstream_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    bitstream_start = pos
    bitstream_end = bitstream_start + int(bitstream_len)
    if bitstream_end != len(blob):
        raise TransferAnalysisError(
            f"PR90 residual bitstream does not consume blob exactly: end={bitstream_end} len={len(blob)}"
        )

    return {
        "magic": matched_magic.decode("ascii", "replace"),
        "bytes": len(blob),
        "sha256": sha256_bytes(blob),
        "n_pairs": int(n_pairs),
        "height": int(height),
        "width": int(width),
        "precision": int(precision),
        "top_bins": int(top_bins),
        "boundary_xbins": int(boundary_xbins),
        "shift_dy": int(shift_dy),
        "shift_dx": int(shift_dx),
        "residual_order": residual_order,
        "top_payload": {"offset": top_start, "bytes": len(top_payload), "sha256": sha256_bytes(top_payload)},
        "road_payload": {"offset": road_start, "bytes": len(road_payload), "sha256": sha256_bytes(road_payload)},
        "spatial_table": {"offset": spatial_start, "bytes": int(spatial_size)},
        "m5_table": {"offset": m5_start, "bytes": int(m5_size)},
        "sparse_features": {"count": n_feats, "ids": feat_ids, "threshold_q8": int(threshold_q8)},
        "sparse_table": {"offset": sparse_start, "bytes": int(sparse_len)},
        "residual_bitstream": {"offset": bitstream_start, "bytes": int(bitstream_len)},
        "_top_payload_bytes": top_payload,
        "_road_payload_bytes": road_payload,
    }


def _import_pr90_seg_codec(pr90_source_dir: Path):
    qrepro_dir = pr90_source_dir / "submissions/qrepro"
    if not qrepro_dir.is_dir():
        raise TransferAnalysisError(f"PR90 qrepro source directory is missing: {_repo_rel(qrepro_dir)}")
    inserted = False
    if str(qrepro_dir) not in sys.path:
        sys.path.insert(0, str(qrepro_dir))
        inserted = True
    try:
        return importlib.import_module("seg_sparse_m5_codec")
    finally:
        if inserted:
            try:
                sys.path.remove(str(qrepro_dir))
            except ValueError:
                pass


def decode_pr90_support_masks(*, qtbm: dict[str, Any], pr90_source_dir: Path) -> dict[str, Any]:
    """Decode the PR90 top support and road-boundary support masks."""

    codec = _import_pr90_seg_codec(pr90_source_dir)
    top = codec.decode_topband_payload(
        qtbm["_top_payload_bytes"], qtbm["n_pairs"], qtbm["height"], qtbm["width"]
    )
    road = codec.decode_boundary_mask_payload(
        qtbm["_road_payload_bytes"], qtbm["n_pairs"], qtbm["height"], qtbm["width"]
    )
    top_bool = top.astype(bool, copy=False)
    road_bool = road.astype(bool, copy=False)
    top_pixels = int(top_bool.sum())
    road_pixels = int(road_bool.sum())
    road_non_top_pixels = int(np.logical_and(road_bool, ~top_bool).sum())
    overlap_pixels = int(np.logical_and(top_bool, road_bool).sum())
    total_pixels = int(qtbm["n_pairs"] * qtbm["height"] * qtbm["width"])
    residual_pixels = total_pixels - top_pixels - road_non_top_pixels
    return {
        "top_support_shape": [int(v) for v in top.shape],
        "road_support_shape": [int(v) for v in road.shape],
        "top_support_sha256": sha256_bytes(top.tobytes()),
        "road_support_sha256": sha256_bytes(road.tobytes()),
        "top_support_pixels": top_pixels,
        "road_support_pixels": road_pixels,
        "road_non_top_pixels": road_non_top_pixels,
        "top_road_overlap_pixels": overlap_pixels,
        "residual_coded_pixels_after_support": int(residual_pixels),
        "total_pixels": total_pixels,
        "top_support_fraction": top_pixels / total_pixels,
        "road_non_top_fraction": road_non_top_pixels / total_pixels,
        "residual_coded_fraction": residual_pixels / total_pixels,
    }


def decode_pr90_full_mask(*, qtbm_blob: bytes, pr90_source_dir: Path) -> np.ndarray:
    """Decode PR90's full semantic topband stream into render-order uint8 masks."""

    codec = _import_pr90_seg_codec(pr90_source_dir)
    with tempfile.NamedTemporaryFile(suffix=".qtbm", delete=True) as handle:
        handle.write(qtbm_blob)
        handle.flush()
        return np.asarray(codec.decode_seg_topband(handle.name), dtype=np.uint8)


def load_pr90_mask_stream(*, pr90_archive: Path, pr90_source_dir: Path) -> dict[str, Any]:
    payload, archive = _read_strict_single_member_zip(pr90_archive, expected_member="p")
    min_compact_len = PR90_COMPACT_MASK_BODY_BYTES + PR90_COMPACT_MODEL_BODY_BYTES
    if len(payload) <= min_compact_len:
        raise TransferAnalysisError(
            f"PR90 compact payload is too short for fixed mask/model slices: {len(payload)} <= {min_compact_len}"
        )
    mask_body = payload[:PR90_COMPACT_MASK_BODY_BYTES]
    qtbm_blob = brotli.decompress(mask_body)
    qtbm = parse_pr90_qtbm_blob(qtbm_blob)
    support = decode_pr90_support_masks(qtbm=qtbm, pr90_source_dir=pr90_source_dir)
    qtbm_public = {key: value for key, value in qtbm.items() if not key.startswith("_")}
    return {
        "archive": archive,
        "compact_split": {
            "mode": "pr90_len_lt_260000_fixed_offsets",
            "payload_bytes": len(payload),
            "mask_body_bytes": len(mask_body),
            "model_body_bytes": PR90_COMPACT_MODEL_BODY_BYTES,
            "pose_qrgb_body_bytes": len(payload) - min_compact_len,
            "stbm_magic_bytes_required_for_self_describing_pr85_segment": len(PR90_STBM_MAGIC),
            "mask_body_sha256": sha256_bytes(mask_body),
            "candidate_stbm_segment_sha256": sha256_bytes(PR90_STBM_MAGIC + mask_body),
        },
        "topband_stream": qtbm_public,
        "support_masks": support,
        "_qtbm_blob": qtbm_blob,
    }


def load_pr85_mask_stream(*, pr85_archive: Path, pr85_token_profile: Path, pr85_token_bin: Path) -> dict[str, Any]:
    raw_member, archive = _read_strict_single_member_zip(pr85_archive, expected_member="x")
    bundle = parse_pr85_bundle(raw_member)
    mask_segment = bytes(bundle.segments["mask"])
    header = parse_qma9_header(mask_segment)
    if header.packed_bytes != len(mask_segment):
        raise TransferAnalysisError(
            f"PR85 mask segment has trailing bytes: packed={header.packed_bytes} segment={len(mask_segment)}"
        )
    profile: dict[str, Any] = {}
    if pr85_token_profile.is_file():
        profile = json.loads(pr85_token_profile.read_text(encoding="utf-8"))
        expected_sha = profile.get("mask_segment_identity", {}).get("sha256")
        if expected_sha and expected_sha != sha256_bytes(mask_segment):
            raise TransferAnalysisError("PR85 token profile does not match archive mask segment SHA")
    if not pr85_token_bin.is_file():
        raise TransferAnalysisError(f"PR85 decoded token source is missing: {_repo_rel(pr85_token_bin)}")
    tokens = pr85_token_bin.read_bytes()
    expected_len = header.frame_count * header.width * header.height
    if len(tokens) != expected_len:
        raise TransferAnalysisError(f"PR85 decoded token length {len(tokens)} != expected {expected_len}")
    token_sha = sha256_bytes(tokens)
    profile_token_sha = profile.get("token_source", {}).get("sha256")
    if profile_token_sha and profile_token_sha != token_sha:
        raise TransferAnalysisError("PR85 token profile raw-token SHA does not match token file")
    arr_storage = np.frombuffer(tokens, dtype=np.uint8).reshape(header.frame_count, header.width, header.height)
    arr_render = arr_storage.transpose(0, 2, 1).copy()
    return {
        "archive": archive,
        "bundle_format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "mask_segment": {
            "bytes": len(mask_segment),
            "sha256": sha256_bytes(mask_segment),
            "offset": int(bundle.segment_offsets["mask"]),
            "qma9_header": {
                "magic": header.magic,
                "frame_count": header.frame_count,
                "width": header.width,
                "height": header.height,
                "bitstream_bytes": header.bitstream_bytes,
                "decoded_mask_bytes": header.decoded_mask_bytes,
                "bitstream_sha256": header.bitstream_sha256,
                "payload_sha256": header.payload_sha256,
            },
        },
        "decoded_tokens": {
            "path": _repo_rel(pr85_token_bin),
            "storage_order_shape": [header.frame_count, header.width, header.height],
            "render_order_shape": [header.frame_count, header.height, header.width],
            "storage_order_sha256": token_sha,
            "render_order_sha256": sha256_bytes(arr_render.tobytes()),
            "class_counts": _class_counts(tokens),
        },
        "_render_array": arr_render,
    }


def build_ranked_policy(
    *,
    pr90_mask: dict[str, Any],
    pr85_mask: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    """Build a ranked policy JSON with byte estimates and fail-closed gates."""

    pr85_archive_bytes = int(pr85_mask["archive"]["archive_bytes"])
    pr85_mask_bytes = int(pr85_mask["mask_segment"]["bytes"])
    candidate_mask_bytes = int(pr90_mask["compact_split"]["mask_body_bytes"]) + len(PR90_STBM_MAGIC)
    delta_mask_bytes = candidate_mask_bytes - pr85_mask_bytes
    candidate_archive_bytes = pr85_archive_bytes + delta_mask_bytes
    rate_delta = delta_mask_bytes * RATE_SCORE_PER_BYTE
    parity_passed = bool(parity["decoded_mask_equal"])

    decision = {
        "status": "ready_for_local_archive_builder_after_runtime_port" if parity_passed else "fail_closed_co_trained_or_mismatched_mask",
        "implementable_next_archive_builder": parity_passed,
        "fixed_pr85_runtime_preserved": False,
        "reason": (
            "PR90 topband stream decodes to the exact PR85 mask tensor; a builder can replace only the mask segment after adding an explicit STBM decoder path."
            if parity_passed
            else "PR90 decoded mask differs from PR85 QMA9 tokens, so the representation is too co-trained or mismatched for a lossless PR85 mask-prior transfer."
        ),
    }
    candidate = {
        "rank": 1,
        "policy_id": "pr90_stbm1br_lossless_pr85_mask_recode",
        "surface": "PR85 QMA9 mask segment",
        "source_signal": "PR90 semantic topband/road-boundary decomposition",
        "target_score_term": "archive_bytes_rate_only_if_decoded_mask_and_runtime_output_parity_hold",
        "status": "builder_ready_after_runtime_port" if parity_passed else "fail_closed_not_buildable",
        "dispatch_unlocked": False,
        "score_claim": False,
        "no_op_status": {
            "archive_changing": parity_passed and delta_mask_bytes != 0,
            "decoded_mask_changing": False if parity_passed else None,
            "source_byte_reuse": False,
            "candidate_mask_segment_differs_from_pr85_qma9": parity_passed and delta_mask_bytes != 0,
            "no_op": not (parity_passed and delta_mask_bytes != 0),
            "reason": (
                "charged mask bytes change while decoded semantic tokens remain identical"
                if parity_passed and delta_mask_bytes != 0
                else "candidate is blocked before archive mutation"
            ),
        },
        "charged_byte_estimate": {
            "pr85_archive_bytes": pr85_archive_bytes,
            "pr85_mask_segment_bytes": pr85_mask_bytes,
            "pr90_mask_body_bytes": int(pr90_mask["compact_split"]["mask_body_bytes"]),
            "required_stbm_magic_bytes": len(PR90_STBM_MAGIC),
            "candidate_mask_segment_bytes": candidate_mask_bytes,
            "delta_mask_segment_bytes": delta_mask_bytes,
            "estimated_archive_bytes_if_only_mask_segment_changes": candidate_archive_bytes,
            "delta_archive_bytes": delta_mask_bytes,
            "rate_score_delta_if_components_unchanged": rate_delta,
        },
        "parity_evidence": {
            "decoded_mask_equal": parity_passed,
            "diff_pixels": None if parity.get("diff_pixels") is None else int(parity["diff_pixels"]),
            "pr90_render_order_sha256": parity["pr90_render_order_sha256"],
            "pr85_render_order_sha256": parity["pr85_render_order_sha256"],
            "render_order_shape": parity["render_order_shape"],
        },
        "topband_road_boundary_profile": pr90_mask["support_masks"],
        "exact_builder_requirements": [
            "Build a PR85-family archive by replacing only the mask segment in the single-member x bundle.",
            "Charge a self-describing STBM1BR\\0 mask segment; do not rely on PR90 compact fixed offsets inside PR85.",
            "Port or reimplement the PR90 QTBM topband/road-boundary decoder into the PR85 inflate runtime with a distinct magic path.",
            "Prove decoded-token SHA parity against the PR85 QMA9 render-order tensor before archive construction is considered valid.",
            "Prove PR85 inflate/runtime output parity against the baseline archive before any exact CUDA eval dispatch.",
            "Run archive validator checks for deterministic ZIP structure, member ordering, no sidecars, and updated bundle length header.",
            "Only after local parity gates pass, claim a lane before any remote exact eval dispatch.",
        ],
        "blockers": [
            {
                "blocker_class": "fixed_pr85_runtime_does_not_decode_stbm1br",
                "status": "blocking_fixed_runtime_transfer",
                "requirement": "Current PR85 runtime expects the QMA9 mask path; STBM requires an explicit reviewed decoder path.",
            },
            {
                "blocker_class": "public_pr90_decoder_port_required",
                "status": "blocking_until_vendored_and_reviewed",
                "requirement": "Do not depend on mutable PR90 source paths at inflate time; vendor/reimplement the decoder in the archive runtime.",
            },
        ],
    }
    if not parity_passed:
        candidate["blockers"].insert(
            0,
            {
                "blocker_class": "decoded_mask_mismatch",
                "status": "fail_closed",
                "requirement": "No archive builder may transfer PR90 topband bytes unless decoded mask parity is exact.",
            },
        )
    fixed_runtime_route = {
        "rank": 2,
        "policy_id": "fixed_runtime_qma9_geometry_prior_reencode",
        "surface": "PR85 QMA9 mask segment",
        "source_signal": "PR90 topband/road-boundary support masks",
        "target_score_term": "archive_bytes_rate",
        "status": "research_blocker_no_archive_changing_policy",
        "dispatch_unlocked": False,
        "score_claim": False,
        "no_op_status": {
            "archive_changing": False,
            "decoded_mask_changing": False,
            "source_byte_reuse": True,
            "no_op": True,
            "reason": "QMA9 has no charged prior side channel; changing its model requires runtime grammar changes.",
        },
        "charged_byte_estimate": {
            "pr85_mask_segment_bytes": pr85_mask_bytes,
            "candidate_mask_segment_bytes": pr85_mask_bytes,
            "delta_mask_segment_bytes": 0,
            "rate_score_delta_if_components_unchanged": 0.0,
        },
        "exact_builder_requirements": [
            "None under the fixed PR85 QMA9 runtime; a geometry prior must become a new mask grammar or a token-parity runtime extension.",
        ],
        "blockers": [
            {
                "blocker_class": "qma9_runtime_has_no_topband_prior_contract",
                "status": "fail_closed_fixed_runtime",
                "requirement": "A PR90-inspired prior cannot change QMA9 bytes while preserving the exact QMA9 decoder semantics.",
            }
        ],
    }
    return {
        "schema": POLICY_SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "decision": decision,
        "ranked_candidates": [candidate, fixed_runtime_route],
    }


def build_analysis(
    *,
    pr90_archive: Path,
    pr90_source_dir: Path,
    pr85_archive: Path,
    pr85_token_profile: Path,
    pr85_token_bin: Path,
    output_json: Path,
    policy_json: Path,
    report_md: Path,
    full_decode: bool = True,
) -> dict[str, Any]:
    pr90_mask = load_pr90_mask_stream(pr90_archive=pr90_archive, pr90_source_dir=pr90_source_dir)
    pr85_mask = load_pr85_mask_stream(
        pr85_archive=pr85_archive,
        pr85_token_profile=pr85_token_profile,
        pr85_token_bin=pr85_token_bin,
    )

    if full_decode:
        pr90_render = decode_pr90_full_mask(qtbm_blob=pr90_mask["_qtbm_blob"], pr90_source_dir=pr90_source_dir)
        pr85_render = pr85_mask["_render_array"]
        if list(pr90_render.shape) != list(pr85_render.shape):
            diff_pixels = -1
            decoded_equal = False
        else:
            diff_pixels = int(np.count_nonzero(pr90_render != pr85_render))
            decoded_equal = diff_pixels == 0
        parity = {
            "performed": True,
            "decoded_mask_equal": decoded_equal,
            "diff_pixels": diff_pixels,
            "render_order_shape": [int(v) for v in pr90_render.shape],
            "pr90_render_order_sha256": sha256_bytes(pr90_render.tobytes()),
            "pr85_render_order_sha256": pr85_mask["decoded_tokens"]["render_order_sha256"],
            "class_counts": _class_counts(pr90_render),
        }
    else:
        parity = {
            "performed": False,
            "decoded_mask_equal": False,
            "diff_pixels": None,
            "render_order_shape": pr85_mask["decoded_tokens"]["render_order_shape"],
            "pr90_render_order_sha256": None,
            "pr85_render_order_sha256": pr85_mask["decoded_tokens"]["render_order_sha256"],
            "class_counts": None,
            "fail_closed_reason": "full PR90 topband decode was skipped",
        }

    pr90_public = {key: value for key, value in pr90_mask.items() if not key.startswith("_")}
    pr85_public = {key: value for key, value in pr85_mask.items() if not key.startswith("_")}
    policy = build_ranked_policy(pr90_mask=pr90_public, pr85_mask=pr85_public, parity=parity)
    analysis = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "evidence_grade": "empirical_local_exact_token_parity_no_score",
        "pr90": pr90_public,
        "pr85": pr85_public,
        "decoded_mask_parity": parity,
        "co_training_assessment": {
            "whole_pr90_stack_is_co_trained": True,
            "mask_stream_transfer_too_co_trained": not bool(parity["decoded_mask_equal"]),
            "fixed_pr85_runtime_transfer_blocked": True,
            "lossless_mask_recode_transfer_supported": bool(parity["decoded_mask_equal"]),
        },
        "policy_json": _repo_rel(policy_json),
    }
    _write_json(output_json, analysis)
    _write_json(policy_json, policy)
    _write_text(report_md, render_markdown_summary(analysis=analysis, policy=policy))
    return analysis


def render_markdown_summary(*, analysis: dict[str, Any], policy: dict[str, Any]) -> str:
    decision = policy["decision"]
    top = policy["ranked_candidates"][0]
    byte = top["charged_byte_estimate"]
    parity = analysis["decoded_mask_parity"]
    lines = [
        "# PR90 QMA9 Mask-Prior Transfer Analysis",
        "",
        "Local-only artifact. No training, scorer run, remote dispatch, or score claim.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Implementable next archive builder: `{decision['implementable_next_archive_builder']}`",
        f"- Fixed PR85 runtime preserved: `{decision['fixed_pr85_runtime_preserved']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Exact Mask Parity",
        "",
        f"- PR90/PR85 decoded mask equal: `{parity['decoded_mask_equal']}`",
        f"- Diff pixels: `{parity['diff_pixels']}`",
        f"- PR90 render-order SHA-256: `{parity['pr90_render_order_sha256']}`",
        f"- PR85 render-order SHA-256: `{parity['pr85_render_order_sha256']}`",
        "",
        "## Top Candidate",
        "",
        f"- Policy: `{top['policy_id']}`",
        f"- Candidate mask bytes: `{byte['candidate_mask_segment_bytes']}`",
        f"- Delta mask bytes: `{byte['delta_mask_segment_bytes']}`",
        f"- Estimated archive bytes: `{byte['estimated_archive_bytes_if_only_mask_segment_changes']}`",
        f"- Rate score delta if components unchanged: `{byte['rate_score_delta_if_components_unchanged']}`",
        "",
        "## Blocker",
        "",
        "The fixed PR85 runtime cannot consume STBM1BR. The next builder must include a reviewed runtime decoder port and parity gates.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr90-archive", type=Path, default=DEFAULT_PR90_DIR / "archive.zip")
    parser.add_argument("--pr90-source-dir", type=Path, default=DEFAULT_PR90_DIR / "pr90_src")
    parser.add_argument("--pr85-archive", type=Path, default=DEFAULT_PR85_DIR / "archive.zip")
    parser.add_argument(
        "--pr85-token-profile",
        type=Path,
        default=DEFAULT_PR85_DIR / "qma9_token_source/pr85_qma9_token_source_profile.json",
    )
    parser.add_argument(
        "--pr85-token-bin",
        type=Path,
        default=DEFAULT_PR85_DIR / "qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_ANALYSIS_JSON)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument(
        "--skip-full-decode",
        action="store_true",
        help="Skip the full PR90 semantic decode and fail closed; intended for parser-only debugging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = build_analysis(
        pr90_archive=args.pr90_archive,
        pr90_source_dir=args.pr90_source_dir,
        pr85_archive=args.pr85_archive,
        pr85_token_profile=args.pr85_token_profile,
        pr85_token_bin=args.pr85_token_bin,
        output_json=args.output_json,
        policy_json=args.policy_json,
        report_md=args.report_md,
        full_decode=not bool(args.skip_full_decode),
    )
    decision_status = "ready" if analysis["co_training_assessment"]["lossless_mask_recode_transfer_supported"] else "blocked"
    print(
        "pr90_qma9_mask_prior_transfer "
        f"decision={decision_status} "
        f"analysis={_repo_rel(args.output_json)} "
        f"policy={_repo_rel(args.policy_json)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
