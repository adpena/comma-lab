#!/usr/bin/env python3
"""Build the PR85 STBM1BR lossless mask-recode candidate.

This is a local-only, fail-closed builder.  It replaces only PR85's mask
segment in the single-member ``x`` bundle with ``STBM1BR\\0`` plus the public
PR90 topband mask body, proves decoded render-order mask parity, writes a
deterministic single-member candidate archive, and records custody artifacts.

It does not load scorers, run exact eval, claim a score, dispatch jobs, or
mutate dispatch state.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    Pr85BundleError,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)
from tac.qma9_range_mask_contract import (  # noqa: E402
    decode_qma9_mask,
    parse_qma9_header,
    sha256_bytes,
    sha256_file,
)
from tac.stbm1br_mask_codec import (  # noqa: E402
    STBM1BR_MAGIC,
    decode_stbm1br_mask_segment,
    metadata_as_dict,
    parse_stbm1br_metadata,
)


TOOL = "experiments/build_pr85_stbm1br_mask_recode_candidate.py"
SCHEMA = "pr85_stbm1br_mask_recode_candidate_summary_v1"
MANIFEST_SCHEMA = "pr85_stbm1br_mask_recode_candidate_v1"
DEFAULT_PR85_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PR90_ARCHIVE = REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker/archive.zip"
DEFAULT_POLICY_JSON = (
    REPO_ROOT
    / "experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/ranked_candidate_policy.json"
)
DEFAULT_PR85_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_tokens_u8_storage_order.bin"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker"
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
DEFAULT_PR85_REPLAY_RUNTIME_DIR: Path | None = None
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
KNOWN_PR85_ARCHIVE_BYTES = 236_328
KNOWN_PR85_ARCHIVE_SHA256 = "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
KNOWN_PR90_ARCHIVE_BYTES = 218_080
KNOWN_PR90_ARCHIVE_SHA256 = "608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d"
EXPECTED_PR85_RENDER_ORDER_SHA256 = "0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45"
EXPECTED_PR90_MASK_BODY_BYTES = 152_431
EXPECTED_PR85_QMA9_MASK_BYTES = 159_011
EXPECTED_SHAPE = (600, 384, 512)
POLICY_ID = "pr90_stbm1br_lossless_pr85_mask_recode"


class STBMRecodeBuildError(ValueError):
    """Raised when the local candidate cannot pass fail-closed preflight."""


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _read_single_member_zip(path: Path, *, expected_member: str) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise STBMRecodeBuildError(f"archive is missing: {_repo_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_member]:
            raise STBMRecodeBuildError(
                f"{_repo_rel(path)} must contain exactly one member {expected_member!r}; got {names!r}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise STBMRecodeBuildError(f"{_repo_rel(path)}:{expected_member} must be ZIP_STORED")
        if expected_member == "x":
            validate_pr85_member_name(info.filename)
        payload = zf.read(info)
        if len(payload) != int(info.file_size):
            raise STBMRecodeBuildError(f"{_repo_rel(path)}:{expected_member} size mismatch")
    archive_bytes = int(path.stat().st_size)
    return payload, {
        "path": _repo_rel(path),
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256_file(path),
        "member_name": expected_member,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_overhead_bytes": archive_bytes - len(payload),
        "zip_compress_type": int(zipfile.ZIP_STORED),
    }


def _load_policy(policy_json: Path) -> dict[str, Any]:
    if not policy_json.is_file():
        raise STBMRecodeBuildError(f"policy JSON is missing: {_repo_rel(policy_json)}")
    policy = json.loads(policy_json.read_text(encoding="utf-8"))
    ranked = policy.get("ranked_candidates", [])
    if not ranked or ranked[0].get("policy_id") != POLICY_ID:
        raise STBMRecodeBuildError(f"policy JSON top candidate is not {POLICY_ID!r}")
    if policy.get("dispatch_performed") is not False or policy.get("score_claim") is not False:
        raise STBMRecodeBuildError("source policy must be planning-only with no dispatch/score claim")
    return policy


def _validate_known_archive(
    meta: Mapping[str, Any],
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    label: str,
) -> None:
    if expected_bytes is not None and int(meta["archive_bytes"]) != int(expected_bytes):
        raise STBMRecodeBuildError(
            f"{label} archive bytes {meta['archive_bytes']} != expected {expected_bytes}"
        )
    if expected_sha256 is not None and str(meta["archive_sha256"]) != str(expected_sha256):
        raise STBMRecodeBuildError(
            f"{label} archive sha256 {meta['archive_sha256']} != expected {expected_sha256}"
        )


def _extract_pr90_mask_body(pr90_payload: bytes, *, expected_body_bytes: int) -> bytes:
    if len(pr90_payload) < int(expected_body_bytes):
        raise STBMRecodeBuildError(
            f"PR90 payload is shorter than expected mask body: {len(pr90_payload)} < {expected_body_bytes}"
        )
    mask_body = pr90_payload[: int(expected_body_bytes)]
    segment = STBM1BR_MAGIC + mask_body
    metadata = parse_stbm1br_metadata(segment)
    if metadata.brotli_body_bytes != int(expected_body_bytes):
        raise STBMRecodeBuildError("parsed STBM body length does not match fixed PR90 split")
    return mask_body


def _render_order_from_storage_tokens(
    token_bytes: bytes,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> np.ndarray:
    expected = int(frame_count) * int(width) * int(height)
    if len(token_bytes) != expected:
        raise STBMRecodeBuildError(f"PR85 token source bytes {len(token_bytes)} != expected {expected}")
    storage = np.frombuffer(token_bytes, dtype=np.uint8).reshape(frame_count, width, height)
    return storage.transpose(0, 2, 1).copy()


def _decode_pr85_render_order(
    source_mask: bytes,
    *,
    token_source: Path | None,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    header = parse_qma9_header(source_mask)
    if token_source is not None and token_source.is_file():
        token_bytes = token_source.read_bytes()
        render = _render_order_from_storage_tokens(
            token_bytes,
            frame_count=header.frame_count,
            width=header.width,
            height=header.height,
        )
        method = "predecoded_pr85_qma9_token_source"
        token_meta = {
            "path": _repo_rel(token_source),
            "storage_order_bytes": len(token_bytes),
            "storage_order_sha256": sha256_bytes(token_bytes),
            "storage_order": "frame_major_header_width_by_header_height",
        }
    else:
        decoded = decode_qma9_mask(source_mask)
        render = _render_order_from_storage_tokens(
            decoded.data,
            frame_count=decoded.header.frame_count,
            width=decoded.header.width,
            height=decoded.header.height,
        )
        method = "decoded_source_qma9_segment_with_repo_python_codec"
        token_meta = {
            "path": None,
            "storage_order_bytes": len(decoded.data),
            "storage_order_sha256": decoded.sha256,
            "storage_order": decoded.storage_order,
        }
    if tuple(int(v) for v in render.shape) != tuple(expected_shape):
        raise STBMRecodeBuildError(f"PR85 render-order shape {tuple(render.shape)} != {expected_shape}")
    token_meta.update(
        {
            "method": method,
            "render_order_shape": [int(v) for v in render.shape],
            "render_order_sha256": sha256_bytes(render.tobytes()),
            "qma9_header": {
                "frame_count": int(header.frame_count),
                "width": int(header.width),
                "height": int(header.height),
                "bitstream_bytes": int(header.bitstream_bytes),
                "payload_sha256": header.payload_sha256,
                "bitstream_sha256": header.bitstream_sha256,
            },
        }
    )
    return render, token_meta


def _zip_member_bytes(member_name: str, payload: bytes) -> bytes:
    if member_name != "x":
        raise STBMRecodeBuildError(f"candidate archive member must be 'x', got {member_name!r}")
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _write_single_x_archive(path: Path, x_payload: bytes) -> dict[str, Any]:
    first = _zip_member_bytes("x", x_payload)
    second = _zip_member_bytes("x", x_payload)
    if first != second:
        raise STBMRecodeBuildError("deterministic ZIP writer produced non-identical archives")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(first)
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise STBMRecodeBuildError(f"candidate archive must contain only ['x']; got {names!r}")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise STBMRecodeBuildError("candidate x member is not ZIP_STORED")
        readback = zf.read(info)
    if readback != x_payload:
        raise STBMRecodeBuildError("candidate x readback differs from written payload")
    return {
        "path": _repo_rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": sha256_file(path),
        "member_name": "x",
        "member_bytes": len(x_payload),
        "member_sha256": sha256_bytes(x_payload),
        "zip_overhead_bytes": int(path.stat().st_size) - len(x_payload),
        "zip_storage": "stored",
        "deterministic_rewrite_identical": True,
    }


def _runtime_support(runtime_dir: Path) -> dict[str, Any]:
    inflate_renderer = runtime_dir / "inflate_renderer.py"
    text = inflate_renderer.read_text(encoding="utf-8", errors="replace") if inflate_renderer.is_file() else ""
    checks = {
        "inflate_renderer_present": inflate_renderer.is_file(),
        "stbm1br_magic_declared": "STBM1BR" in text,
        "stbm1br_loader_present": "_load_masks_from_stbm1br" in text,
        "archive_loader_routes_stbm1br": "return _load_masks_from_stbm1br" in text,
        "qma9_loader_still_present": "_load_masks_from_qma9" in text and "QMA9" in text,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "runtime_dir": _repo_rel(runtime_dir),
        "checks": checks,
        "runtime_support_present": not blockers,
        "remaining_blockers": blockers,
        "support_scope": "mask_loader_only_no_scorer_no_dispatch",
    }


def _runtime_files_contract(runtime_dir: Path) -> dict[str, Any]:
    files = []
    tree_hasher = hashlib.sha256()
    for path in sorted(p for p in runtime_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(runtime_dir).as_posix()
        if rel.startswith("__pycache__/") or "/__pycache__/" in rel:
            continue
        digest = sha256_file(path)
        size = int(path.stat().st_size)
        files.append(
            {
                "path": rel,
                "bytes": size,
                "sha256": digest,
            }
        )
        tree_hasher.update(rel.encode("utf-8"))
        tree_hasher.update(b"\0")
        tree_hasher.update(str(size).encode("ascii"))
        tree_hasher.update(b"\0")
        tree_hasher.update(digest.encode("ascii"))
        tree_hasher.update(b"\n")
    return {
        "file_count": len(files),
        "runtime_tree_sha256": tree_hasher.hexdigest(),
        "files": files,
    }


def _load_runtime_expectations(contract_json: Path | None) -> dict[str, Any]:
    if contract_json is None:
        return {}
    if not contract_json.is_file():
        raise STBMRecodeBuildError(f"runtime contract JSON is missing: {_repo_rel(contract_json)}")
    payload = json.loads(contract_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise STBMRecodeBuildError("runtime contract JSON must contain an object")
    return payload


def _pr85_replay_runtime_contract(
    runtime_dir: Path | None,
    *,
    candidate_archive: Mapping[str, Any],
    candidate_mask: Mapping[str, Any],
    runtime_contract_json: Path | None = None,
) -> dict[str, Any]:
    expectations = _load_runtime_expectations(runtime_contract_json)
    expected = expectations.get("expected_candidate", expectations)
    if expected is None or not isinstance(expected, Mapping):
        expected = {}
    if runtime_dir is None:
        blockers = [
            {
                "code": "exact_runtime:missing_explicit_pr85_replay_runtime",
                "severity": "blocking",
                "detail": "Pass --pr85-replay-runtime-dir for the exact eval replay runtime; robust_current is not sufficient.",
            }
        ]
        return {
            "schema": "pr85_stbm1br_exact_replay_runtime_contract_v1",
            "runtime_dir": None,
            "runtime_contract_json": _repo_rel(runtime_contract_json),
            "status": "missing",
            "ready_for_exact_eval_runtime": False,
            "checks": {"explicit_runtime_dir_supplied": False},
            "remaining_blockers": blockers,
            "runtime_tree_sha256": None,
            "files": [],
            "support_scope": "exact_eval_replay_runtime_required_not_robust_current",
        }
    runtime_dir = runtime_dir.resolve()
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_py = runtime_dir / "inflate.py"
    inflate_renderer = runtime_dir / "inflate_renderer.py"
    range_codec = runtime_dir / "range_mask_codec.cpp"
    py_texts = []
    for path in (inflate_py, inflate_renderer):
        if path.is_file():
            py_texts.append(path.read_text(encoding="utf-8", errors="replace"))
    combined = "\n".join(py_texts)
    file_contract = _runtime_files_contract(runtime_dir) if runtime_dir.is_dir() else {
        "file_count": 0,
        "runtime_tree_sha256": None,
        "files": [],
    }
    expected_runtime_tree_sha256 = expected.get("runtime_tree_sha256")
    expected_archive_sha256 = expected.get("candidate_archive_sha256") or expected.get("archive_sha256")
    expected_archive_bytes = expected.get("candidate_archive_bytes") or expected.get("archive_bytes")
    expected_mask_sha256 = expected.get("candidate_mask_sha256") or expected.get("mask_sha256")
    expected_mask_bytes = expected.get("candidate_mask_bytes") or expected.get("mask_bytes")
    checks = {
        "explicit_runtime_dir_supplied": True,
        "runtime_dir_exists": runtime_dir.is_dir(),
        "inflate_sh_present": inflate_sh.is_file(),
        "replay_python_runtime_present": inflate_py.is_file() or inflate_renderer.is_file(),
        "pr85_single_member_x_loader_present": (
            ("load_compact_archive_bundle" in combined and 'data_dir / "x"' in combined)
            or "parse_pr85_bundle" in combined
        ),
        "stbm_magic_branch_present": "STBM1BR" in combined,
        "stbm_decoder_present": "decode_stbm1br_mask_segment" in combined or "_load_masks_from_stbm1br" in combined,
        "stbm_branch_before_brotli_fallback_present": (
            "bundle[\"mask\"][:8] == b\"STBM1BR\\0\"" in combined
            or "bundle['mask'][:8] == b'STBM1BR\\0'" in combined
            or "_load_masks_from_stbm1br" in combined
        ),
        "qma9_or_range_mask_fallback_present": "QMA9" in combined or range_codec.is_file(),
        "runtime_tree_matches_contract_json": (
            expected_runtime_tree_sha256 is None
            or expected_runtime_tree_sha256 == file_contract["runtime_tree_sha256"]
        ),
        "candidate_archive_sha256_matches_contract_json": (
            expected_archive_sha256 is None
            or expected_archive_sha256 == candidate_archive["archive_sha256"]
        ),
        "candidate_archive_bytes_matches_contract_json": (
            expected_archive_bytes is None
            or int(expected_archive_bytes) == int(candidate_archive["archive_bytes"])
        ),
        "candidate_mask_sha256_matches_contract_json": (
            expected_mask_sha256 is None
            or expected_mask_sha256 == candidate_mask["sha256"]
        ),
        "candidate_mask_bytes_matches_contract_json": (
            expected_mask_bytes is None
            or int(expected_mask_bytes) == int(candidate_mask["bytes"])
        ),
    }
    blockers = [
        {
            "code": f"exact_runtime:{name}",
            "severity": "blocking",
            "detail": f"{name} failed",
        }
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema": "pr85_stbm1br_exact_replay_runtime_contract_v1",
        "runtime_dir": _repo_rel(runtime_dir),
        "runtime_contract_json": _repo_rel(runtime_contract_json),
        "status": "passed" if not blockers else "failed",
        "ready_for_exact_eval_runtime": not blockers,
        "checks": checks,
        "remaining_blockers": blockers,
        "runtime_tree_sha256": file_contract["runtime_tree_sha256"],
        "files": file_contract["files"],
        "support_scope": "explicit_pr85_replay_runtime_for_archive_zip_to_inflate_sh_to_evaluate_py",
        "contract_expectations": dict(expected),
    }


def _class_counts(arr: np.ndarray) -> dict[str, int]:
    counts = np.bincount(arr.reshape(-1).astype(np.uint8, copy=False), minlength=5)
    return {str(i): int(counts[i]) for i in range(5)}


def build_pr85_stbm1br_mask_recode_candidate(
    *,
    pr85_archive: Path,
    pr90_archive: Path,
    policy_json: Path,
    out_dir: Path,
    token_source: Path | None = DEFAULT_PR85_TOKEN_SOURCE,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
    pr85_replay_runtime_dir: Path | None = DEFAULT_PR85_REPLAY_RUNTIME_DIR,
    pr85_replay_runtime_contract_json: Path | None = None,
    require_exact_eval_runtime: bool = False,
    expected_shape: tuple[int, int, int] = EXPECTED_SHAPE,
    expected_pr85_render_sha256: str = EXPECTED_PR85_RENDER_ORDER_SHA256,
    expected_pr85_archive_bytes: int | None = KNOWN_PR85_ARCHIVE_BYTES,
    expected_pr85_archive_sha256: str | None = KNOWN_PR85_ARCHIVE_SHA256,
    expected_pr90_archive_bytes: int | None = KNOWN_PR90_ARCHIVE_BYTES,
    expected_pr90_archive_sha256: str | None = KNOWN_PR90_ARCHIVE_SHA256,
    expected_pr90_mask_body_bytes: int = EXPECTED_PR90_MASK_BODY_BYTES,
) -> dict[str, Any]:
    policy = _load_policy(policy_json)
    pr85_raw, pr85_archive_meta = _read_single_member_zip(pr85_archive, expected_member="x")
    pr90_payload, pr90_archive_meta = _read_single_member_zip(pr90_archive, expected_member="p")
    _validate_known_archive(
        pr85_archive_meta,
        expected_bytes=expected_pr85_archive_bytes,
        expected_sha256=expected_pr85_archive_sha256,
        label="PR85",
    )
    _validate_known_archive(
        pr90_archive_meta,
        expected_bytes=expected_pr90_archive_bytes,
        expected_sha256=expected_pr90_archive_sha256,
        label="PR90",
    )

    pr85_bundle = parse_pr85_bundle(pr85_raw)
    source_segments = dict(pr85_bundle.segments)
    source_mask = bytes(source_segments["mask"])
    if not source_mask.startswith(b"QMA9"):
        raise STBMRecodeBuildError(f"source PR85 mask segment is not QMA9: {source_mask[:8]!r}")
    if len(source_mask) != EXPECTED_PR85_QMA9_MASK_BYTES and expected_pr85_archive_sha256 == KNOWN_PR85_ARCHIVE_SHA256:
        raise STBMRecodeBuildError(
            f"known PR85 source mask bytes {len(source_mask)} != {EXPECTED_PR85_QMA9_MASK_BYTES}"
        )

    pr90_mask_body = _extract_pr90_mask_body(
        pr90_payload,
        expected_body_bytes=expected_pr90_mask_body_bytes,
    )
    candidate_mask = STBM1BR_MAGIC + pr90_mask_body
    candidate_metadata = parse_stbm1br_metadata(candidate_mask)
    candidate_render = decode_stbm1br_mask_segment(candidate_mask, expected_shape=expected_shape)
    candidate_render_sha = sha256_bytes(candidate_render.tobytes())

    pr85_render, pr85_token_meta = _decode_pr85_render_order(
        source_mask,
        token_source=token_source,
        expected_shape=expected_shape,
    )
    pr85_render_sha = pr85_token_meta["render_order_sha256"]
    if pr85_render_sha != expected_pr85_render_sha256:
        raise STBMRecodeBuildError(
            f"PR85 render-order sha256 {pr85_render_sha} != expected {expected_pr85_render_sha256}"
        )
    if candidate_render_sha != expected_pr85_render_sha256:
        raise STBMRecodeBuildError(
            f"STBM render-order sha256 {candidate_render_sha} != expected {expected_pr85_render_sha256}"
        )

    decoded_equal = bool(np.array_equal(candidate_render, pr85_render))
    diff_pixels = 0 if decoded_equal else int(np.count_nonzero(candidate_render != pr85_render))
    if not decoded_equal:
        raise STBMRecodeBuildError(f"decoded STBM mask differs from PR85 QMA9 by {diff_pixels} pixels")

    if candidate_mask == source_mask:
        raise STBMRecodeBuildError("candidate mask segment is a byte-level no-op")
    if len(candidate_mask) >= len(source_mask):
        raise STBMRecodeBuildError(
            f"candidate mask is not byte-positive: {len(candidate_mask)} >= {len(source_mask)}"
        )

    candidate_segments = dict(source_segments)
    candidate_segments["mask"] = candidate_mask
    header_mode = "v5" if pr85_bundle.header_bytes == 24 else "explicit_30"
    candidate_x = pack_pr85_bundle(candidate_segments, header_mode=header_mode)
    parsed_candidate = parse_pr85_bundle(candidate_x)
    if bytes(parsed_candidate.segments["mask"]) != candidate_mask:
        raise STBMRecodeBuildError("candidate bundle parse/readback lost the STBM mask segment")
    if parsed_candidate.segment_lengths["mask"] != len(candidate_mask):
        raise STBMRecodeBuildError("candidate bundle header did not record updated mask length")

    candidate_id = POLICY_ID
    candidate_dir = out_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    candidate_archive_meta = _write_single_x_archive(archive_path, candidate_x)
    if int(candidate_archive_meta["archive_bytes"]) >= int(pr85_archive_meta["archive_bytes"]):
        raise STBMRecodeBuildError("candidate archive is not byte-positive versus source PR85")

    mask_segment_path = candidate_dir / "mask_segment.stbm1br"
    mask_segment_path.write_bytes(candidate_mask)
    runtime = _runtime_support(robust_current_dir)
    fail_closed_checks = {
        "candidate_non_noop_at_byte_level": candidate_mask != source_mask,
        "candidate_mask_byte_positive": len(candidate_mask) < len(source_mask),
        "candidate_archive_byte_positive": int(candidate_archive_meta["archive_bytes"]) < int(pr85_archive_meta["archive_bytes"]),
        "decoded_mask_equal": decoded_equal,
        "decoded_render_order_sha_matches_expected": candidate_render_sha == expected_pr85_render_sha256,
        "single_member_x_only": candidate_archive_meta["member_name"] == "x",
        "runtime_support_present": runtime["runtime_support_present"],
        "no_scorer_load": True,
        "remote_dispatch_not_performed": True,
    }
    if not all(fail_closed_checks.values()):
        failed = [name for name, passed in fail_closed_checks.items() if not passed]
        raise STBMRecodeBuildError(f"fail-closed preflight failed: {failed}")

    source_segment_meta = {
        "name": "mask",
        "codec": "QMA9",
        "bytes": len(source_mask),
        "sha256": sha256_bytes(source_mask),
        "magic_hex": source_mask[:8].hex(),
    }
    candidate_segment_meta = {
        "name": "mask",
        "codec": "STBM1BR_brotli_qtbm_topband",
        "path": _repo_rel(mask_segment_path),
        "bytes": len(candidate_mask),
        "sha256": sha256_bytes(candidate_mask),
        "magic_hex": candidate_mask[:8].hex(),
        "brotli_body_bytes": len(pr90_mask_body),
        "brotli_body_sha256": sha256_bytes(pr90_mask_body),
        "metadata": metadata_as_dict(candidate_metadata),
    }
    exact_runtime = _pr85_replay_runtime_contract(
        pr85_replay_runtime_dir,
        candidate_archive=candidate_archive_meta,
        candidate_mask=candidate_segment_meta,
        runtime_contract_json=pr85_replay_runtime_contract_json,
    )
    ready_for_exact_eval_after_lane_claim = bool(exact_runtime["ready_for_exact_eval_runtime"])
    if require_exact_eval_runtime and not ready_for_exact_eval_after_lane_claim:
        blockers = [row["code"] for row in exact_runtime["remaining_blockers"]]
        raise STBMRecodeBuildError(f"exact-eval runtime contract failed: {blockers}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "policy_id": POLICY_ID,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "source_policy": {
            "path": _repo_rel(policy_json),
            "schema": policy.get("schema"),
            "top_candidate_status": policy["ranked_candidates"][0].get("status"),
        },
        "source_archive": pr85_archive_meta,
        "pr90_archive": pr90_archive_meta,
        "source_bundle": {
            "format": pr85_bundle.format,
            "header_bytes": int(pr85_bundle.header_bytes),
            "segment_lengths": pr85_bundle.segment_lengths,
        },
        "candidate_archive": candidate_archive_meta,
        "candidate_bundle": {
            "format": parsed_candidate.format,
            "header_mode": header_mode,
            "header_bytes": int(parsed_candidate.header_bytes),
            "segment_lengths": parsed_candidate.segment_lengths,
            "member_sha256": sha256_bytes(candidate_x),
            "member_bytes": len(candidate_x),
        },
        "segments": {
            "source_mask": source_segment_meta,
            "candidate_mask": candidate_segment_meta,
            "byte_delta_vs_source_mask": len(candidate_mask) - len(source_mask),
            "archive_byte_delta_vs_source": int(candidate_archive_meta["archive_bytes"]) - int(pr85_archive_meta["archive_bytes"]),
        },
        "parity": {
            "decoded_mask_equal": decoded_equal,
            "diff_pixels": diff_pixels,
            "render_order_shape": [int(v) for v in expected_shape],
            "pr85_render_order_sha256": pr85_render_sha,
            "candidate_render_order_sha256": candidate_render_sha,
            "expected_render_order_sha256": expected_pr85_render_sha256,
            "candidate_class_counts": _class_counts(candidate_render),
            "pr85_token_source": pr85_token_meta,
        },
        "runtime_support": runtime,
        "exact_eval_runtime_contract": exact_runtime,
        "fail_closed_preflight": {
            "status": "passed",
            "checks": fail_closed_checks,
            "remote_dispatch_allowed": False,
            "ready_for_exact_eval_after_lane_claim": ready_for_exact_eval_after_lane_claim,
            "exact_eval_requires_lane_claim": True,
            "exact_eval_requires_explicit_pr85_replay_runtime": True,
            "readiness_status": "ready" if ready_for_exact_eval_after_lane_claim else "non_dispatchable",
            "remaining_exact_eval_blockers": exact_runtime["remaining_blockers"],
            "next_gate": (
                "main_must_claim_lane_before_any_exact_eval_dispatch"
                if ready_for_exact_eval_after_lane_claim
                else "provide_stbm_aware_pr85_replay_runtime_before_lane_claim"
            ),
        },
    }
    manifest_path = candidate_dir / "manifest.json"
    preflight_path = candidate_dir / "stbm1br_preflight.json"
    _write_json(manifest_path, manifest)
    _write_json(preflight_path, manifest["fail_closed_preflight"] | {"parity": manifest["parity"]})
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "candidate_count": 1,
        "ready_for_exact_eval_after_lane_claim": ready_for_exact_eval_after_lane_claim,
        "exact_eval_readiness_status": "ready" if ready_for_exact_eval_after_lane_claim else "non_dispatchable",
        "exact_eval_runtime_contract": {
            "runtime_dir": exact_runtime["runtime_dir"],
            "runtime_tree_sha256": exact_runtime["runtime_tree_sha256"],
            "status": exact_runtime["status"],
            "remaining_blockers": exact_runtime["remaining_blockers"],
        },
        "candidate_archive": candidate_archive_meta,
        "source_archive": pr85_archive_meta,
        "mask_byte_delta": len(candidate_mask) - len(source_mask),
        "archive_byte_delta": int(candidate_archive_meta["archive_bytes"]) - int(pr85_archive_meta["archive_bytes"]),
        "render_order_sha256": candidate_render_sha,
        "artifacts": {
            "manifest": _repo_rel(manifest_path),
            "preflight": _repo_rel(preflight_path),
            "mask_segment": _repo_rel(mask_segment_path),
        },
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return manifest


def _shape_arg(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.replace("x", ",").split(",") if part]
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("shape must be frames,height,width")
    return tuple(parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr85-archive", type=Path, default=DEFAULT_PR85_ARCHIVE)
    parser.add_argument("--pr90-archive", type=Path, default=DEFAULT_PR90_ARCHIVE)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pr85-token-source", type=Path, default=DEFAULT_PR85_TOKEN_SOURCE)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument(
        "--pr85-replay-runtime-dir",
        type=Path,
        default=DEFAULT_PR85_REPLAY_RUNTIME_DIR,
        help="Explicit STBM-aware PR85 replay runtime used for exact eval readiness. robust_current is local support only.",
    )
    parser.add_argument("--pr85-replay-runtime-contract-json", type=Path, default=None)
    parser.add_argument("--require-exact-eval-runtime", action="store_true")
    parser.add_argument("--expected-shape", type=_shape_arg, default=EXPECTED_SHAPE)
    parser.add_argument("--expected-render-order-sha256", default=EXPECTED_PR85_RENDER_ORDER_SHA256)
    parser.add_argument("--allow-unanchored-pr85-source", action="store_true")
    parser.add_argument("--allow-unanchored-pr90-source", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_pr85_stbm1br_mask_recode_candidate(
        pr85_archive=args.pr85_archive,
        pr90_archive=args.pr90_archive,
        policy_json=args.policy_json,
        out_dir=args.out_dir,
        token_source=args.pr85_token_source,
        robust_current_dir=args.robust_current_dir,
        pr85_replay_runtime_dir=args.pr85_replay_runtime_dir,
        pr85_replay_runtime_contract_json=args.pr85_replay_runtime_contract_json,
        require_exact_eval_runtime=args.require_exact_eval_runtime,
        expected_shape=args.expected_shape,
        expected_pr85_render_sha256=args.expected_render_order_sha256,
        expected_pr85_archive_bytes=None if args.allow_unanchored_pr85_source else KNOWN_PR85_ARCHIVE_BYTES,
        expected_pr85_archive_sha256=None if args.allow_unanchored_pr85_source else KNOWN_PR85_ARCHIVE_SHA256,
        expected_pr90_archive_bytes=None if args.allow_unanchored_pr90_source else KNOWN_PR90_ARCHIVE_BYTES,
        expected_pr90_archive_sha256=None if args.allow_unanchored_pr90_source else KNOWN_PR90_ARCHIVE_SHA256,
    )
    print(_json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
