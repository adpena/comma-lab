#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# pyc-recovery pass2: rehydrated from git blob 960bf56b34bdc2aad35623c6362c56fcc0f65ee8 via `git fsck --lost-found`
# original path: experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
"""Build the PR85_STBM1BR + PR92 RMB1 randmulti decoded-parity candidate.

This is a local deterministic candidate builder. It replaces only the STBM
frontier archive's randmulti segment with PR92's charged RMB1 randmulti segment
after proving decoded sparse-row parity. It does not run scorers, dispatch
remote work, or claim a score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import shutil
import struct
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.pr85_bundle import (
    SEGMENT_ORDER,
    compare_pr85_randmulti_decoded_rows,
    decode_pr85_randmulti_to_headerless_rows,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)
from tac.repo_io import json_text, read_json, sha256_file, write_json

_sha256_file = sha256_file


TOOL = "experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py"
SCHEMA = "pr85_stbm1br_pr92_rmb1_randmulti_candidate_v1"
SUMMARY_SCHEMA = "pr85_stbm1br_pr92_rmb1_randmulti_summary_v1"
DEFAULT_PR85_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_STBM_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "pr90_stbm1br_lossless_pr85_mask_recode/archive.zip"
)
DEFAULT_STBM_MANIFEST = DEFAULT_STBM_ARCHIVE.parent / "manifest.json"
DEFAULT_PR92_ARCHIVE = REPO_ROOT / "experiments/results/public_pr92_intake_20260504_codex/archive.zip"
DEFAULT_PR92_PROFILE = DEFAULT_PR92_ARCHIVE.parent / "public_frontier_intake_profile.json"
DEFAULT_STBM_REPLAY_RUNTIME = (
    REPO_ROOT / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker"
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker.md"
DEFAULT_STBM_EXACT_T4 = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/"
    "contest_auth_eval.adjudicated.json"
)
CANDIDATE_ID = "pr85_stbm1br_plus_pr92_rmb1_randmulti_recode"
LANE_ID = "pr85_stbm1br_pr92_rmb1_randmulti"

EXPECTED = {
    "pr85_archive_bytes": 236_328,
    "pr85_archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
    "stbm_archive_bytes": 229_756,
    "stbm_archive_sha256": "c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6",
    "pr92_archive_bytes": 236_516,
    "pr92_archive_sha256": "f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490",
    "stbm_randmulti_bytes": 16_101,
    "pr92_rmb1_randmulti_bytes": 15_825,
    "pr92_rmb1_randmulti_sha256": "4b10018eab64d8755da3def355881f51f2d450c9f19dd1457a6ec813cddd6f7c",
    "decoded_randmulti_rows_bytes": 27_105,
    "decoded_randmulti_rows_sha256": "87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9",
    "stbm_mask_sha256": "1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313",
}

RMB1_RUNTIME_HELPER = '''

def decode_randmulti_bitmask_payload(encoded_randmulti: bytes) -> bytes:
    """Decode PR92 RMB1 bitmask+value randmulti to headerless sparse rows."""
    if len(encoded_randmulti) < 6 or encoded_randmulti[:4] != b"RMB1":
        raise ValueError("bad RMB1 randmulti payload")
    mask_len = int.from_bytes(encoded_randmulti[4:6], "little")
    mask_br = encoded_randmulti[6:6 + mask_len]
    vals_br = encoded_randmulti[6 + mask_len:]
    if not mask_br or not vals_br:
        raise ValueError("truncated RMB1 randmulti payload")
    mask = brotli.decompress(mask_br)
    vals = brotli.decompress(vals_br)
    if len(mask) % 75:
        raise ValueError("bad RMB1 mask length")
    out = bytearray()
    vals_pos = 0
    for row_start in range(0, len(mask), 75):
        row_mask = mask[row_start:row_start + 75]
        indices = []
        row_values = []
        for byte_i, byte in enumerate(row_mask):
            for bit in range(8):
                frame_i = byte_i * 8 + bit
                if frame_i >= 600:
                    break
                if byte & (1 << bit):
                    if vals_pos >= len(vals):
                        raise ValueError("truncated RMB1 values")
                    indices.append(frame_i)
                    row_values.append(vals[vals_pos])
                    vals_pos += 1
        count = len(indices)
        if count < 255:
            out.append(count)
        else:
            out.append(255)
            out.extend(count.to_bytes(2, "little"))
        last = -1
        for idx in indices:
            delta = idx - last - 1
            last = idx
            while True:
                byte = delta & 0x7F
                delta >>= 7
                if delta:
                    out.append(byte | 0x80)
                else:
                    out.append(byte)
                    break
        out.extend(row_values)
    if vals_pos != len(vals):
        raise ValueError("unused RMB1 values")
    return bytes(out)
'''


class CandidateBuildError(ValueError):
    """Raised when the candidate must fail closed."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateBuildError(f"missing JSON file: {_rel(path)}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise CandidateBuildError(f"JSON file must contain an object: {_rel(path)}")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _read_archive_member(path: Path, *, member: str = "x", allow_extra_members: bool = False) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise CandidateBuildError(f"archive missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if not allow_extra_members and names != [member]:
            raise CandidateBuildError(f"expected exactly one archive member {member!r}, got {names!r}")
        if names.count(member) != 1:
            raise CandidateBuildError(f"expected exactly one archive member {member!r}, got {names!r}")
        info = next(row for row in infos if row.filename == member)
        validate_pr85_member_name(info.filename)
        payload = zf.read(info)
    meta = {
        "path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": sha256_file(path),
        "member_name": member,
        "member_bytes": len(payload),
        "member_sha256": _sha256_bytes(payload),
        "zip_compress_type": int(info.compress_type),
        "zip_overhead_bytes": int(path.stat().st_size) - len(payload),
        "archive_members": names,
    }
    return meta, payload


def _expect_archive(meta: Mapping[str, Any], *, bytes_key: str, sha_key: str) -> None:
    if meta.get("archive_bytes") != EXPECTED[bytes_key]:
        raise CandidateBuildError(f"{bytes_key} mismatch: {meta.get('archive_bytes')} != {EXPECTED[bytes_key]}")
    if meta.get("archive_sha256") != EXPECTED[sha_key]:
        raise CandidateBuildError(f"{sha_key} mismatch: {meta.get('archive_sha256')} != {EXPECTED[sha_key]}")


def _zip_single_x_bytes(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _local_header_name(archive_path: Path, info: zipfile.ZipInfo) -> str:
    data = archive_path.read_bytes()
    offset = int(info.header_offset)
    if data[offset : offset + 4] != b"PK\x03\x04":
        raise CandidateBuildError("candidate ZIP local header signature mismatch")
    name_len, extra_len = struct.unpack_from("<HH", data, offset + 26)
    name_start = offset + 30
    name_end = name_start + int(name_len)
    _extra_end = name_end + int(extra_len)
    return data[name_start:name_end].decode("utf-8")


def _strict_zip_report(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise CandidateBuildError(f"candidate ZIP must contain exactly one member 'x', got {names!r}")
        info = infos[0]
        local_name = _local_header_name(archive_path, info)
        payload = zf.read("x")
    deterministic = _zip_single_x_bytes(payload)
    actual = archive_path.read_bytes()
    blockers = []
    if info.compress_type != zipfile.ZIP_STORED:
        blockers.append("member_x_not_zip_stored")
    if local_name != "x":
        blockers.append("central_local_name_mismatch")
    if deterministic != actual:
        blockers.append("deterministic_rewrite_mismatch")
    if blockers:
        raise CandidateBuildError(f"strict ZIP validation failed: {blockers}")
    return {
        "schema": "strict_single_member_x_zip_v1",
        "valid": True,
        "member_count": 1,
        "members": [
            {
                "name": "x",
                "local_header_name": local_name,
                "central_local_name_match": local_name == "x",
                "bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "method_id": int(info.compress_type),
                "sha256": _sha256_bytes(payload),
                "crc32": f"{info.CRC:08x}",
            }
        ],
        "deterministic_rewrite_identical": deterministic == actual,
        "archive_bytes": len(actual),
        "archive_sha256": _sha256_bytes(actual),
    }


def _segment_diff(left_raw: bytes, right_raw: bytes) -> list[dict[str, Any]]:
    left = parse_pr85_bundle(left_raw)
    right = parse_pr85_bundle(right_raw)
    diffs = []
    for name in SEGMENT_ORDER:
        l_bytes = bytes(left.segments[name])
        r_bytes = bytes(right.segments[name])
        if l_bytes != r_bytes:
            diffs.append(
                {
                    "segment": name,
                    "left_bytes": len(l_bytes),
                    "right_bytes": len(r_bytes),
                    "left_sha256": _sha256_bytes(l_bytes),
                    "right_sha256": _sha256_bytes(r_bytes),
                }
            )
    return diffs


def _stbm_manifest_report(path: Path, stbm_meta: Mapping[str, Any], pr85_meta: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_json(path)
    checks = {
        "score_claim_false": payload.get("score_claim") is False,
        "dispatch_performed_false": payload.get("dispatch_performed") is False,
        "candidate_sha_matches": payload.get("candidate_archive", {}).get("archive_sha256") == stbm_meta.get("archive_sha256"),
        "source_sha_matches": payload.get("source_archive", {}).get("archive_sha256") == pr85_meta.get("archive_sha256"),
        "decoded_mask_equal": payload.get("parity", {}).get("decoded_mask_equal") is True,
        "diff_pixels_zero": payload.get("parity", {}).get("diff_pixels") == 0,
        "stbm_exact_runtime_ready": payload.get("exact_eval_runtime_contract", {}).get("ready_for_exact_eval_runtime") is True,
    }
    return {
        "path": _rel(path),
        "sha256": sha256_file(path),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "candidate_id": payload.get("candidate_id"),
        "runtime_tree_sha256": payload.get("exact_eval_runtime_contract", {}).get("runtime_tree_sha256"),
        "render_order_sha256": payload.get("parity", {}).get("candidate_render_order_sha256"),
        "mask_sha256": payload.get("segments", {}).get("candidate_mask", {}).get("sha256"),
    }


def _pr92_profile_report(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    rows = payload.get("primary_member", {}).get("segments", [])
    randmulti = next(
        (row for row in rows if isinstance(row, Mapping) and row.get("name") == "randmulti"),
        {},
    )
    checks = {
        "score_claim_false": payload.get("score_claim") is False,
        "promotion_eligible_false": payload.get("promotion_eligible") is False,
        "randmulti_bytes_expected": randmulti.get("bytes") == EXPECTED["pr92_rmb1_randmulti_bytes"],
        "randmulti_sha_expected": randmulti.get("sha256") == EXPECTED["pr92_rmb1_randmulti_sha256"],
        "randmulti_codec_rmb1": randmulti.get("codec") == "RMB1_side_info_backed_randmulti",
    }
    return {
        "path": _rel(path),
        "sha256": sha256_file(path),
        "label": payload.get("label"),
        "evidence_grade": payload.get("evidence_grade"),
        "side_info_charged_bytes": payload.get("side_info", {}).get("charged_bytes"),
        "randmulti_segment": dict(randmulti),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _runtime_tree_manifest(runtime_dir: Path) -> dict[str, Any]:
    files = []
    tree = hashlib.sha256()
    for path in sorted(p for p in runtime_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = path.relative_to(runtime_dir).as_posix()
        sha = sha256_file(path)
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha})
        tree.update(rel.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\0")
    return {
        "runtime_dir": _rel(runtime_dir),
        "runtime_file_count": len(files),
        "runtime_tree_sha256": tree.hexdigest(),
        "files": files,
    }


def _patch_replay_runtime_for_rmb1(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_dir.iterdir()):
        if child.is_file() and child.name != "README.md":
            shutil.copy2(child, out_dir / child.name)
        elif child.is_file() and child.name == "README.md":
            text = child.read_text(encoding="utf-8", errors="replace")
            text += "\nPR92 RMB1 randmulti support: candidate-local copy generated by " + TOOL + "\n"
            (out_dir / child.name).write_text(text, encoding="utf-8")
    inflate_py = out_dir / "inflate.py"
    if not inflate_py.is_file():
        raise CandidateBuildError("runtime copy is missing inflate.py")
    text = inflate_py.read_text(encoding="utf-8", errors="replace")
    if "def decode_randmulti_bitmask_payload" not in text:
        marker = "\ndef main():"
        if marker not in text:
            raise CandidateBuildError("cannot locate replay runtime main() for RMB1 helper insertion")
        text = text.replace(marker, RMB1_RUNTIME_HELPER + marker, 1)
    old = '        raw_n = brotli.decompress(bundle["randmulti"])'
    new = (
        '        encoded_n = bundle["randmulti"]\n'
        '        if encoded_n[:4] == b"RMB1":\n'
        "            raw_n = decode_randmulti_bitmask_payload(encoded_n)\n"
        "        else:\n"
        "            raw_n = brotli.decompress(encoded_n)"
    )
    if old in text:
        text = text.replace(old, new, 1)
    inflate_py.write_text(text, encoding="utf-8")
    checks = {
        "source_runtime_exists": source_dir.is_dir(),
        "inflate_sh_present": (out_dir / "inflate.sh").is_file(),
        "inflate_py_present": inflate_py.is_file(),
        "stbm_support_present": "STBM1BR" in text and "load_stbm1br_mask" in text,
        "rmb1_helper_present": "def decode_randmulti_bitmask_payload" in text and "RMB1" in text,
        "rmb1_randmulti_branch_present": 'encoded_n[:4] == b"RMB1"' in text,
    }
    report = _runtime_tree_manifest(out_dir)
    report.update(
        {
            "schema": "pr85_stbm1br_pr92_rmb1_replay_runtime_v1",
            "source_runtime_dir": _rel(source_dir),
            "checks": checks,
            "status": "passed" if all(checks.values()) else "failed",
            "score_claim": False,
            "dispatch_performed": False,
        }
    )
    return report


def _robust_current_rmb1_report() -> dict[str, Any]:
    path = REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    scorer_load_markers = (
        "from upstream",
        "import upstream",
        "load_segnet",
        "load_posenet",
        "SegNet(",
        "PoseNet(",
    )
    checks = {
        "apply_qzs3_postprocess_present": path.is_file(),
        "rmb1_decode_helper_present": "_decode_rmb1_randmulti_payload" in text,
        "rmb1_decode_branch_present": 'blob[:4] == b"RMB1"' in text,
        "rmb1_no_scorer_load": not any(marker in text for marker in scorer_load_markers),
    }
    return {
        "path": _rel(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _stbm_exact_t4_report(path: Path, stbm_meta: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": _rel(path), "status": "missing", "checks": {"artifact_present": False}}
    payload = _load_json(path)
    provenance = payload.get("provenance", {}) if isinstance(payload.get("provenance"), Mapping) else {}
    runtime = provenance.get("inflate_runtime_manifest", {}) if isinstance(provenance.get("inflate_runtime_manifest"), Mapping) else {}
    checks = {
        "artifact_present": True,
        "archive_sha_matches_stbm": provenance.get("archive_sha256") == stbm_meta.get("archive_sha256"),
        "archive_bytes_matches_stbm": payload.get("archive_size_bytes") == stbm_meta.get("archive_bytes"),
        "cuda_device": provenance.get("device") == "cuda",
        "t4_match": provenance.get("gpu_t4_match") is True,
        "full_sample_count": payload.get("n_samples") == 600,
        "runtime_tree_recorded": isinstance(runtime.get("runtime_tree_sha256"), str),
    }
    return {
        "path": _rel(path),
        "sha256": sha256_file(path),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "canonical_score": payload.get("canonical_score"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
    }


def build_candidate(
    *,
    pr85_archive: Path = DEFAULT_PR85_ARCHIVE,
    stbm_archive: Path = DEFAULT_STBM_ARCHIVE,
    stbm_manifest: Path = DEFAULT_STBM_MANIFEST,
    pr92_archive: Path = DEFAULT_PR92_ARCHIVE,
    pr92_profile: Path = DEFAULT_PR92_PROFILE,
    stbm_replay_runtime: Path = DEFAULT_STBM_REPLAY_RUNTIME,
    stbm_exact_t4_json: Path = DEFAULT_STBM_EXACT_T4,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    pr85_meta, pr85_raw = _read_archive_member(pr85_archive)
    stbm_meta, stbm_raw = _read_archive_member(stbm_archive)
    pr92_meta, pr92_raw = _read_archive_member(pr92_archive, allow_extra_members=True)
    _expect_archive(pr85_meta, bytes_key="pr85_archive_bytes", sha_key="pr85_archive_sha256")
    _expect_archive(stbm_meta, bytes_key="stbm_archive_bytes", sha_key="stbm_archive_sha256")
    _expect_archive(pr92_meta, bytes_key="pr92_archive_bytes", sha_key="pr92_archive_sha256")

    pr85_bundle = parse_pr85_bundle(pr85_raw)
    stbm_bundle = parse_pr85_bundle(stbm_raw)
    pr92_bundle = parse_pr85_bundle(pr92_raw)
    stbm_randmulti = bytes(stbm_bundle.segments["randmulti"])
    pr92_randmulti = bytes(pr92_bundle.segments["randmulti"])
    stbm_mask = bytes(stbm_bundle.segments["mask"])

    if len(stbm_randmulti) != EXPECTED["stbm_randmulti_bytes"]:
        raise CandidateBuildError("STBM randmulti encoded byte count mismatch")
    if len(pr92_randmulti) != EXPECTED["pr92_rmb1_randmulti_bytes"]:
        raise CandidateBuildError("PR92 RMB1 randmulti encoded byte count mismatch")
    if not pr92_randmulti.startswith(b"RMB1"):
        raise CandidateBuildError("PR92 randmulti segment is not RMB1")
    if _sha256_bytes(pr92_randmulti) != EXPECTED["pr92_rmb1_randmulti_sha256"]:
        raise CandidateBuildError("PR92 RMB1 randmulti SHA mismatch")
    decoded_rows, decoded_profile = decode_pr85_randmulti_to_headerless_rows(pr92_randmulti)
    if len(decoded_rows) != EXPECTED["decoded_randmulti_rows_bytes"]:
        raise CandidateBuildError("PR92 RMB1 decoded rows byte count mismatch")
    if _sha256_bytes(decoded_rows) != EXPECTED["decoded_randmulti_rows_sha256"]:
        raise CandidateBuildError("PR92 RMB1 decoded rows SHA mismatch")

    stbm_manifest_report = _stbm_manifest_report(stbm_manifest, stbm_meta, pr85_meta)
    pr92_profile_report = _pr92_profile_report(pr92_profile)
    if stbm_manifest_report["status"] != "passed":
        raise CandidateBuildError("STBM source manifest did not pass review checks")
    if pr92_profile_report["status"] != "passed":
        raise CandidateBuildError("PR92 intake profile did not pass randmulti checks")
    if stbm_manifest_report["mask_sha256"] != _sha256_bytes(stbm_mask):
        raise CandidateBuildError("STBM manifest mask SHA does not match source archive")
    if _sha256_bytes(stbm_mask) != EXPECTED["stbm_mask_sha256"]:
        raise CandidateBuildError("STBM source mask SHA mismatch")

    stbm_vs_pr85 = _segment_diff(pr85_raw, stbm_raw)
    if [row["segment"] for row in stbm_vs_pr85] != ["mask"]:
        raise CandidateBuildError("STBM source is not mask-only vs PR85")
    if bytes(pr85_bundle.segments["randmulti"]) != stbm_randmulti:
        raise CandidateBuildError("STBM randmulti source is not unchanged vs PR85")
    randmulti_parity = compare_pr85_randmulti_decoded_rows(stbm_randmulti, pr92_randmulti)
    if randmulti_parity.get("parity_status") != "passed":
        raise CandidateBuildError("PR92 RMB1 randmulti decoded rows do not match STBM")

    candidate_segments = {name: bytes(stbm_bundle.segments[name]) for name in SEGMENT_ORDER}
    candidate_segments["randmulti"] = pr92_randmulti
    candidate_raw = pack_pr85_bundle(candidate_segments, header_mode="v5")
    candidate_dir = out_dir / CANDIDATE_ID
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    archive_path.write_bytes(_zip_single_x_bytes(candidate_raw))
    strict_zip = _strict_zip_report(archive_path)
    candidate_meta, archive_member = _read_archive_member(archive_path)
    if archive_member != candidate_raw:
        raise CandidateBuildError("candidate archive member does not match built payload")
    candidate_vs_stbm = _segment_diff(stbm_raw, candidate_raw)
    candidate_vs_pr85 = _segment_diff(pr85_raw, candidate_raw)
    if [row["segment"] for row in candidate_vs_stbm] != ["randmulti"]:
        raise CandidateBuildError("candidate changed non-randmulti segment vs STBM")
    if bytes(parse_pr85_bundle(candidate_raw).segments["mask"]) != stbm_mask:
        raise CandidateBuildError("candidate did not preserve STBM mask bytes")

    runtime_dir = out_dir / "replay_submission_stbm_rmb1"
    runtime_report = _patch_replay_runtime_for_rmb1(stbm_replay_runtime, runtime_dir)
    runtime_blockers = [
        name
        for name, passed in runtime_report.get("checks", {}).items()
        if passed is not True
    ]
    exact_eval_runtime_contract = {
        "schema": "pr85_stbm1br_pr92_rmb1_exact_eval_runtime_contract_v1",
        "ready_for_exact_eval_runtime": runtime_report["status"] == "passed",
        "runtime_dir": runtime_report.get("runtime_dir"),
        "runtime_tree_sha256": runtime_report.get("runtime_tree_sha256"),
        "runtime_file_count": runtime_report.get("runtime_file_count"),
        "required_inflate_sh": _rel(runtime_dir / "inflate.sh"),
        "checks": runtime_report.get("checks", {}),
        "remaining_blockers": runtime_blockers,
        "score_claim": False,
        "dispatch_performed": False,
    }
    robust_current_rmb1 = _robust_current_rmb1_report()
    stbm_exact = _stbm_exact_t4_report(stbm_exact_t4_json, stbm_meta)
    rate_delta_bytes = int(candidate_meta["archive_bytes"]) - int(stbm_meta["archive_bytes"])
    formula_rate_score_delta = 25.0 * rate_delta_bytes / 37_545_489.0
    run_stamp = str(candidate_meta["archive_sha256"])[:16]
    job_name = f"exact_eval_pr85_stbm1br_pr92_rmb1_t4_{run_stamp}"
    claim_command = (
        ".venv/bin/python tools/claim_lane_dispatch.py claim "
        f"--lane-id {LANE_ID} --platform lightning "
        f"--instance-job-id {job_name} --agent codex:gpt-5.5 "
        "--predicted-eta-utc ${PREDICTED_ETA_UTC} --status eval "
        f"--notes \"T4 exact eval for {CANDIDATE_ID}; archive_sha256={candidate_meta['archive_sha256']}\""
    )

    readiness_checks = {
        "strict_zip_single_member_x": strict_zip["valid"] is True,
        "candidate_changes_only_randmulti_vs_stbm": [row["segment"] for row in candidate_vs_stbm] == ["randmulti"],
        "stbm_mask_unchanged": bytes(parse_pr85_bundle(candidate_raw).segments["mask"]) == stbm_mask,
        "randmulti_decoded_rows_match_stbm": randmulti_parity["decoded_rows_match"] is True,
        "candidate_score_claim_false": True,
        "remote_dispatch_not_performed": True,
        "candidate_runtime_rmb1_ready": runtime_report["status"] == "passed",
        "robust_current_rmb1_guard_present": robust_current_rmb1["status"] == "passed",
        "stbm_standalone_exact_t4_positive_present": stbm_exact["status"] == "passed",
    }
    exact_t4_dispatch_justified = all(readiness_checks.values())
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": CANDIDATE_ID,
        "build_status": "built",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "scorer_load_performed": False,
        "gpu_required_for_build": False,
        "evidence_grade": "empirical_local_archive_build_only",
        "source_archive": pr85_meta,
        "stbm_source_archive": stbm_meta,
        "pr92_source_archive": pr92_meta,
        "stbm_manifest_review": stbm_manifest_report,
        "pr92_profile_review": pr92_profile_report,
        "candidate_archive": candidate_meta,
        "strict_zip": strict_zip,
        "segment_diffs": {
            "stbm_vs_pr85": stbm_vs_pr85,
            "candidate_vs_stbm": candidate_vs_stbm,
            "candidate_vs_pr85": candidate_vs_pr85,
        },
        "non_noop_byte_change": {
            "changed_segments_vs_stbm": [row["segment"] for row in candidate_vs_stbm],
            "archive_delta_bytes_vs_stbm": rate_delta_bytes,
            "randmulti_delta_bytes_vs_stbm": len(pr92_randmulti) - len(stbm_randmulti),
            "source_randmulti_sha256": _sha256_bytes(stbm_randmulti),
            "candidate_randmulti_sha256": _sha256_bytes(pr92_randmulti),
        },
        "stbm_mask_preservation": {
            "unchanged": True,
            "bytes": len(stbm_mask),
            "sha256": _sha256_bytes(stbm_mask),
        },
        "randmulti_decoded_row_parity": randmulti_parity,
        "pr92_rmb1_decoded_profile": decoded_profile,
        "exact_eval_runtime_contract": exact_eval_runtime_contract,
        "runtime_support": {
            "candidate_replay_runtime": runtime_report,
            "robust_current_apply_qzs3_rmb1": robust_current_rmb1,
        },
        "stbm_standalone_exact_t4_reference": stbm_exact,
        "formula_only_rate_delta_vs_stbm": {
            "archive_delta_bytes": rate_delta_bytes,
            "score_delta_from_rate_only": formula_rate_score_delta,
            "score_claim": False,
        },
        "dispatch_readiness": {
            "checks": readiness_checks,
            "exact_t4_dispatch_justified_after_claim": exact_t4_dispatch_justified,
            "score_claim": False,
            "remote_dispatch_performed": False,
            "required_inflate_sh": _rel(runtime_dir / "inflate.sh"),
            "lane_id": LANE_ID,
            "next_claim_command": claim_command,
            "notes": [
                "Candidate is byte-positive and decoded-randmulti-parity-preserving versus the T4-positive STBM source.",
                "Exact CUDA auth eval remains the score truth; this manifest makes no score claim.",
            ],
        },
    }
    write_json(candidate_dir / "manifest.json", manifest)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "candidate_count": 1,
        "candidate_archive": candidate_meta,
        "candidate_manifest": _rel(candidate_dir / "manifest.json"),
        "archive_delta_bytes_vs_stbm": rate_delta_bytes,
        "randmulti_decoded_rows_sha256": randmulti_parity["decoded_rows_sha256"],
        "strict_zip_valid": True,
        "exact_t4_dispatch_justified_after_claim": exact_t4_dispatch_justified,
        "next_claim_command": claim_command,
    }
    write_json(out_dir / "candidate_summary.json", summary)
    return summary


def render_ledger(summary: Mapping[str, Any]) -> str:
    candidate = summary.get("candidate_archive", {}) if isinstance(summary.get("candidate_archive"), Mapping) else {}
    lines = [
        "# PR85 STBM1BR + PR92 RMB1 Randmulti Recode - 2026-05-04",
        "",
        f"- tool: `{TOOL}`",
        "- score_claim: false",
        "- dispatch_performed: false",
        "- remote_jobs_dispatched: false",
        "",
        "## Candidate",
        "",
        f"- archive: `{candidate.get('path')}`",
        f"- bytes: `{candidate.get('archive_bytes')}`",
        f"- sha256: `{candidate.get('archive_sha256')}`",
        f"- manifest: `{summary.get('candidate_manifest')}`",
        f"- archive_delta_bytes_vs_stbm: `{summary.get('archive_delta_bytes_vs_stbm')}`",
        f"- randmulti decoded rows SHA: `{summary.get('randmulti_decoded_rows_sha256')}`",
        "",
        "## Readiness",
        "",
        f"- strict_zip_valid: `{summary.get('strict_zip_valid')}`",
        f"- exact_t4_dispatch_justified_after_claim: `{summary.get('exact_t4_dispatch_justified_after_claim')}`",
        "- exact CUDA eval is required before any score claim.",
        "",
        "## Exact Next Claim Command",
        "",
        "```bash",
        str(summary.get("next_claim_command")),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_ledger(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(summary), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr85-archive", type=Path, default=DEFAULT_PR85_ARCHIVE)
    parser.add_argument("--stbm-archive", type=Path, default=DEFAULT_STBM_ARCHIVE)
    parser.add_argument("--stbm-manifest", type=Path, default=DEFAULT_STBM_MANIFEST)
    parser.add_argument("--pr92-archive", type=Path, default=DEFAULT_PR92_ARCHIVE)
    parser.add_argument("--pr92-profile", type=Path, default=DEFAULT_PR92_PROFILE)
    parser.add_argument("--stbm-replay-runtime", type=Path, default=DEFAULT_STBM_REPLAY_RUNTIME)
    parser.add_argument("--stbm-exact-t4-json", type=Path, default=DEFAULT_STBM_EXACT_T4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_candidate(
        pr85_archive=args.pr85_archive,
        stbm_archive=args.stbm_archive,
        stbm_manifest=args.stbm_manifest,
        pr92_archive=args.pr92_archive,
        pr92_profile=args.pr92_profile,
        stbm_replay_runtime=args.stbm_replay_runtime,
        stbm_exact_t4_json=args.stbm_exact_t4_json,
        out_dir=args.out_dir,
    )
    write_ledger(args.ledger_md, summary)
    if args.stdout:
        sys.stdout.write(json_text(summary))
    else:
        print(
            json_text(
                {
                    "candidate_archive": summary["candidate_archive"],
                    "candidate_manifest": summary["candidate_manifest"],
                    "exact_t4_dispatch_justified_after_claim": summary[
                        "exact_t4_dispatch_justified_after_claim"
                    ],
                }
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
