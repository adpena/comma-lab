#!/usr/bin/env python3
"""Build PR85 STBM1BR plus PR92 RMB1 randmulti lossless recode candidate.

This local builder replaces only the ``randmulti`` segment in the current
STBM1BR frontier archive with PR92's byte-smaller ``RMB1`` recode, then proves
the decoded headerless sparse randmulti rows are identical. It does not run
scorers, dispatch remote work, or claim a score.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (
    decode_pr85_randmulti_to_headerless_rows,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)
from tac.stbm1br_mask_codec import STBM1BR_MAGIC

TOOL = "experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py"
SCHEMA = "pr85_stbm1br_rmb1_randmulti_candidate_summary_v1"
MANIFEST_SCHEMA = "pr85_stbm1br_rmb1_randmulti_candidate_v1"
DEFAULT_STBM_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "pr90_stbm1br_lossless_pr85_mask_recode/archive.zip"
)
DEFAULT_PR92_ARCHIVE = REPO_ROOT / "experiments/results/public_pr92_intake_20260504_codex/archive.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EXPECTED_STBM_SHA256 = "c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6"
EXPECTED_STBM_BYTES = 229_756
EXPECTED_PR92_SHA256 = "f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490"
DEFAULT_DISPATCH_LANE_ID = "pr85_stbm1br_pr92_rmb1_randmulti"
DEFAULT_DISPATCH_PLATFORM = "lightning"
DISPATCH_CLAIMS_PATH = ".omx/state/active_lane_dispatch_claims.md"
ROBUST_CURRENT_DIR = REPO_ROOT / "submissions/robust_current"
CANONICAL_RMB1_BUILDER = "experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py"


class Rmb1CandidateBuildError(ValueError):
    """Raised when the candidate must fail closed."""


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _read_single_x_archive(path: Path, *, allow_extra_members: bool = False) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise Rmb1CandidateBuildError(f"archive is missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if "x" not in names or (not allow_extra_members and names != ["x"]):
            raise Rmb1CandidateBuildError(f"archive must contain exactly one member 'x'; got {names!r}")
        if names.count("x") != 1:
            raise Rmb1CandidateBuildError(f"archive must contain exactly one member named 'x'; got {names!r}")
        info = next(info for info in infos if info.filename == "x")
        validate_pr85_member_name(info.filename)
        payload = zf.read(info)
        side_members = [info.filename for info in infos if info.filename != "x"]
    return (
        {
            "path": _rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": _sha256_file(path),
            "member_name": "x",
            "member_bytes": len(payload),
            "member_sha256": _sha256_bytes(payload),
            "zip_stored": info.compress_type == zipfile.ZIP_STORED,
            "zip_timestamp": list(info.date_time),
            "side_members_ignored_for_x_intake": side_members,
        },
        payload,
    )


def _zip_member_bytes(member_name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _write_single_x_archive(path: Path, payload: bytes) -> dict[str, Any]:
    first = _zip_member_bytes("x", payload)
    second = _zip_member_bytes("x", payload)
    if first != second:
        raise Rmb1CandidateBuildError("deterministic ZIP writer produced non-identical bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(first)
    meta, readback = _read_single_x_archive(path)
    if readback != payload:
        raise Rmb1CandidateBuildError("candidate archive readback differs from written payload")
    meta["deterministic_rewrite_identical"] = True
    return meta


def _segment_meta(segment: bytes, *, codec: str) -> dict[str, Any]:
    return {
        "bytes": len(segment),
        "sha256": _sha256_bytes(segment),
        "magic_hex": segment[:8].hex(),
        "codec": codec,
    }


def _robust_current_runtime_support_report(runtime_dir: Path | None = None) -> dict[str, Any]:
    """Check that the default submission runtime has this archive's decoders."""

    runtime_dir = runtime_dir or ROBUST_CURRENT_DIR
    inflate_renderer = runtime_dir / "inflate_renderer.py"
    qpost_runtime = runtime_dir / "apply_qzs3_postprocess.py"
    inflate_renderer_text = (
        inflate_renderer.read_text(encoding="utf-8", errors="replace")
        if inflate_renderer.is_file()
        else ""
    )
    qpost_text = (
        qpost_runtime.read_text(encoding="utf-8", errors="replace")
        if qpost_runtime.is_file()
        else ""
    )
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh_text = (
        inflate_sh.read_text(encoding="utf-8", errors="replace")
        if inflate_sh.is_file()
        else ""
    )
    unpack_runtime = runtime_dir / "unpack_renderer_payload.py"
    unpack_text = (
        unpack_runtime.read_text(encoding="utf-8", errors="replace")
        if unpack_runtime.is_file()
        else ""
    )
    scorer_markers = (
        "import upstream",
        "from upstream",
        "load_posenet",
        "load_segnet",
        "PoseNet(",
        "SegNet(",
    )
    checks = {
        "inflate_renderer_present": inflate_renderer.is_file(),
        "inflate_sh_present": inflate_sh.is_file(),
        "qpost_runtime_present": qpost_runtime.is_file(),
        "unpack_runtime_present": unpack_runtime.is_file(),
        "single_member_x_unpack_stage_present": '[ -f "$ARCHIVE_DIR/x" ]' in inflate_sh_text
        or '[ -f "${ARCHIVE_DIR}/x" ]' in inflate_sh_text
        or "PAYLOAD_SHORT_BR = \"x\"" in unpack_text
        or "PAYLOAD_SHORT_BR = 'x'" in unpack_text,
        "stbm1br_magic_present": "STBM1BR_MAGIC" in inflate_renderer_text,
        "stbm1br_loader_present": "_load_masks_from_stbm1br" in inflate_renderer_text,
        "rmb1_helper_present": "_decode_rmb1_randmulti_payload" in qpost_text,
        "rmb1_branch_present": 'blob[:4] == b"RMB1"' in qpost_text,
        "qpost_runtime_no_scorer_load": not any(marker in qpost_text for marker in scorer_markers),
    }
    return {
        "schema": "robust_current_stbm1br_rmb1_runtime_support_v1",
        "runtime_dir": _rel(runtime_dir),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "files": {
            "inflate_renderer.py": {
                "path": _rel(inflate_renderer),
                "sha256": _sha256_file(inflate_renderer) if inflate_renderer.is_file() else None,
            },
            "apply_qzs3_postprocess.py": {
                "path": _rel(qpost_runtime),
                "sha256": _sha256_file(qpost_runtime) if qpost_runtime.is_file() else None,
            },
            "inflate.sh": {
                "path": _rel(inflate_sh),
                "sha256": _sha256_file(inflate_sh) if inflate_sh.is_file() else None,
            },
            "unpack_renderer_payload.py": {
                "path": _rel(unpack_runtime),
                "sha256": _sha256_file(unpack_runtime) if unpack_runtime.is_file() else None,
            },
        },
    }


def _module_string_constants(path: Path, names: set[str]) -> dict[str, str]:
    constants: dict[str, str] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Catalog #168 fix 2026-05-12: handle both `X = "y"` (Assign) and
    # `X: str = "y"` (AnnAssign) module-level string constants.
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    constants[target.id] = node.value.value
        elif (isinstance(node, ast.AnnAssign)
              and node.value is not None
              and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)
              and isinstance(node.target, ast.Name)
              and node.target.id in names):
            constants[node.target.id] = node.value.value
    return constants


def _duplicate_builder_coordination_report() -> dict[str, Any]:
    """Fail closed if the legacy and canonical RMB1 builders drift lanes."""

    canonical_path = REPO_ROOT / CANONICAL_RMB1_BUILDER
    checks: dict[str, bool] = {
        "canonical_builder_exists": canonical_path.is_file(),
    }
    constants: dict[str, str] = {}
    if canonical_path.is_file():
        constants = _module_string_constants(canonical_path, {"LANE_ID", "TOOL", "SCHEMA"})
        checks.update(
            {
                "canonical_tool_matches_path": constants.get("TOOL") == CANONICAL_RMB1_BUILDER,
                "lane_id_matches_canonical_builder": constants.get("LANE_ID") == DEFAULT_DISPATCH_LANE_ID,
                "canonical_schema_is_pr92_rmb1": str(constants.get("SCHEMA", "")).startswith(
                    "pr85_stbm1br_pr92_rmb1_randmulti"
                ),
            }
        )
    return {
        "schema": "duplicate_rmb1_builder_coordination_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "legacy_builder": TOOL,
        "canonical_builder": CANONICAL_RMB1_BUILDER,
        "legacy_lane_id": DEFAULT_DISPATCH_LANE_ID,
        "canonical_constants": constants,
        "checks": checks,
    }


def _dispatch_claim_template(*, candidate_id: str, archive_sha256: str, manifest_path: Path) -> dict[str, Any]:
    command = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--claims-path",
        DISPATCH_CLAIMS_PATH,
        "--lane-id",
        DEFAULT_DISPATCH_LANE_ID,
        "--platform",
        DEFAULT_DISPATCH_PLATFORM,
        "--instance-job-id",
        f"exact_eval_{candidate_id}_t4_${{UTC_STAMP}}",
        "--agent",
        "${AGENT_ID}",
        "--predicted-eta-utc",
        "${PREDICTED_ETA_UTC}",
        "--status",
        "exact_eval_ready",
        "--notes",
        f"archive_sha256={archive_sha256} manifest={_rel(manifest_path)}",
    ]
    return {
        "claim_required": True,
        "command_not_executed_by_builder": True,
        "command_template": command,
        "lane_id": DEFAULT_DISPATCH_LANE_ID,
        "platform": DEFAULT_DISPATCH_PLATFORM,
        "claims_path": DISPATCH_CLAIMS_PATH,
        "status": "exact_eval_ready",
        "required_placeholders": ["AGENT_ID", "PREDICTED_ETA_UTC", "UTC_STAMP"],
    }


def build_pr85_stbm1br_rmb1_randmulti_candidate(
    *,
    stbm_archive: Path = DEFAULT_STBM_ARCHIVE,
    pr92_archive: Path = DEFAULT_PR92_ARCHIVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    candidate_id: str = "pr85_stbm1br_plus_pr92_rmb1_randmulti",
) -> dict[str, Any]:
    stbm_meta, stbm_raw = _read_single_x_archive(stbm_archive)
    pr92_meta, pr92_raw = _read_single_x_archive(pr92_archive, allow_extra_members=True)
    stbm_bundle = parse_pr85_bundle(stbm_raw)
    pr92_bundle = parse_pr85_bundle(pr92_raw)

    stbm_segments = {name: bytes(stbm_bundle.segments[name]) for name in stbm_bundle.segments}
    pr92_randmulti = bytes(pr92_bundle.segments["randmulti"])
    source_randmulti = stbm_segments["randmulti"]
    if not bytes(stbm_bundle.segments["mask"]).startswith(STBM1BR_MAGIC):
        raise Rmb1CandidateBuildError("source archive mask is not STBM1BR")
    if not pr92_randmulti.startswith(b"RMB1"):
        raise Rmb1CandidateBuildError("PR92 randmulti segment is not RMB1")
    if pr92_randmulti == source_randmulti:
        raise Rmb1CandidateBuildError("RMB1 candidate would be a randmulti byte no-op")
    source_rows, source_profile = decode_pr85_randmulti_to_headerless_rows(source_randmulti)
    candidate_rows, candidate_profile = decode_pr85_randmulti_to_headerless_rows(pr92_randmulti)
    if source_rows != candidate_rows:
        raise Rmb1CandidateBuildError("RMB1 decoded randmulti rows differ from STBM source")
    if len(pr92_randmulti) >= len(source_randmulti):
        raise Rmb1CandidateBuildError("RMB1 randmulti is not byte-positive")

    candidate_segments = dict(stbm_segments)
    candidate_segments["randmulti"] = pr92_randmulti
    header_mode = "v5" if stbm_bundle.header_bytes == 24 else "explicit_30"
    candidate_x = pack_pr85_bundle(candidate_segments, header_mode=header_mode)
    parsed_candidate = parse_pr85_bundle(candidate_x)
    if bytes(parsed_candidate.segments["mask"]) != stbm_segments["mask"]:
        raise Rmb1CandidateBuildError("candidate changed STBM mask bytes")
    if bytes(parsed_candidate.segments["randmulti"]) != pr92_randmulti:
        raise Rmb1CandidateBuildError("candidate lost RMB1 randmulti bytes")

    candidate_dir = out_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    candidate_meta = _write_single_x_archive(archive_path, candidate_x)
    if int(candidate_meta["archive_bytes"]) >= int(stbm_meta["archive_bytes"]):
        raise Rmb1CandidateBuildError("candidate archive is not byte-positive versus STBM")
    runtime_support = _robust_current_runtime_support_report()
    duplicate_builder_coordination = _duplicate_builder_coordination_report()

    checks = {
        "source_archive_matches_expected_stbm": stbm_meta["archive_sha256"] == EXPECTED_STBM_SHA256
        and stbm_meta["archive_bytes"] == EXPECTED_STBM_BYTES,
        "pr92_archive_matches_expected": pr92_meta["archive_sha256"] == EXPECTED_PR92_SHA256,
        "source_mask_stbm1br": bytes(stbm_bundle.segments["mask"]).startswith(STBM1BR_MAGIC),
        "mask_bytes_unchanged": bytes(parsed_candidate.segments["mask"]) == stbm_segments["mask"],
        "randmulti_non_noop": pr92_randmulti != source_randmulti,
        "randmulti_byte_positive": len(pr92_randmulti) < len(source_randmulti),
        "randmulti_decoded_rows_equal": source_rows == candidate_rows,
        "candidate_archive_byte_positive": candidate_meta["archive_bytes"] < stbm_meta["archive_bytes"],
        "single_member_x_only": candidate_meta["member_name"] == "x",
        "zip_stored": bool(candidate_meta["zip_stored"]),
        "remote_dispatch_not_performed": True,
        "score_claim_false": True,
        "robust_current_runtime_support": runtime_support["status"] == "passed",
        "duplicate_rmb1_builder_lane_coordination": duplicate_builder_coordination["status"] == "passed",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise Rmb1CandidateBuildError(f"fail-closed checks failed: {failed}")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "evidence_grade": "empirical_local_lossless_recode_candidate",
        "source_archive": stbm_meta,
        "pr92_archive": pr92_meta,
        "candidate_archive": candidate_meta,
        "bundle": {
            "source_format": stbm_bundle.format,
            "candidate_format": parsed_candidate.format,
            "header_mode": header_mode,
            "segment_lengths": parsed_candidate.segment_lengths,
        },
        "segments": {
            "mask": _segment_meta(stbm_segments["mask"], codec="STBM1BR_lossless_mask_recode"),
            "source_randmulti": _segment_meta(source_randmulti, codec=source_profile["codec"]),
            "candidate_randmulti": _segment_meta(pr92_randmulti, codec=candidate_profile["codec"]),
            "randmulti_byte_delta": len(pr92_randmulti) - len(source_randmulti),
            "archive_byte_delta_vs_source": candidate_meta["archive_bytes"] - stbm_meta["archive_bytes"],
        },
        "parity": {
            "randmulti_decoded_rows_equal": True,
            "decoded_rows_bytes": len(source_rows),
            "decoded_rows_sha256": _sha256_bytes(source_rows),
            "source_profile": source_profile,
            "candidate_profile": candidate_profile,
        },
        "fail_closed_preflight": {
            "status": "passed",
            "checks": checks,
            "ready_for_exact_eval_after_lane_claim": True,
            "exact_eval_requires_lane_claim": True,
        },
        "runtime_support": runtime_support,
        "duplicate_builder_coordination": duplicate_builder_coordination,
        "dispatch_gate": {
            "status": "eligible_for_exact_eval_after_level2_lane_claim",
            "dispatch_performed": False,
            "remote_jobs_dispatched": False,
            "claim": _dispatch_claim_template(
                candidate_id=candidate_id,
                archive_sha256=str(candidate_meta["archive_sha256"]),
                manifest_path=manifest_path,
            ),
        },
        "expected_rate_only_delta": {
            "bytes": candidate_meta["archive_bytes"] - stbm_meta["archive_bytes"],
            "score_delta_if_components_unchanged": 25
            * (candidate_meta["archive_bytes"] - stbm_meta["archive_bytes"])
            / 37_545_489,
        },
    }
    _write_json(manifest_path, manifest)
    _write_json(candidate_dir / "preflight.json", manifest["fail_closed_preflight"] | {"parity": manifest["parity"]})
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "candidate_id": candidate_id,
        "archive": manifest["candidate_archive"],
        "manifest_path": _rel(candidate_dir / "manifest.json"),
        "preflight_path": _rel(candidate_dir / "preflight.json"),
        "byte_delta_vs_source": manifest["segments"]["archive_byte_delta_vs_source"],
        "ready_for_exact_eval_after_lane_claim": True,
        "canonical_builder": CANONICAL_RMB1_BUILDER,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stbm-archive", type=Path, default=DEFAULT_STBM_ARCHIVE)
    parser.add_argument("--pr92-archive", type=Path, default=DEFAULT_PR92_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-id", default="pr85_stbm1br_plus_pr92_rmb1_randmulti")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Compatibility no-op; the builder always prints the JSON summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_pr85_stbm1br_rmb1_randmulti_candidate(
        stbm_archive=args.stbm_archive,
        pr92_archive=args.pr92_archive,
        out_dir=args.out_dir,
        candidate_id=args.candidate_id,
    )
    print(_json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
