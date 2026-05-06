#!/usr/bin/env python3
"""Build local PR85 fixed-runtime bridge candidate archives.

This tool does not run inflate, load scorers, claim a score, dispatch remote
jobs, or mutate dispatch state.  It converts the public PR85 single-member
``x`` payload into the logical member names that ``submissions/robust_current``
can inspect: ``masks.qma9``, ``renderer.bin``, ``optimized_poses.bin``, and
``qpost.bin``.  The QH0 renderer member remains a runtime-loader blocker until
robust_current learns that wire format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    Pr85BundleError,
    expand_pr85_bundle_to_runtime_members,
    validate_pr85_member_name,
)


TOOL = "experiments/build_pr85_fixed_runtime_bridge_candidate.py"
SCHEMA = "pr85_fixed_runtime_bridge_candidates_v1"
MANIFEST_SCHEMA = "pr85_fixed_runtime_bridge_candidate_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_fixed_runtime_bridge_candidates_20260504_codex"
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_ORDER = ("masks.qma9", "renderer.bin", "optimized_poses.bin", "qpost.bin")
KNOWN_PUBLIC_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _read_pr85_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise Pr85BundleError(
                f"PR85 source archive must contain exactly one member 'x'; got {names!r}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    archive_sha = _sha256_file(path)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": archive_sha,
            "known_public_pr85_v5_match": {
                "matches": (
                    int(path.stat().st_size) == KNOWN_PUBLIC_PR85["archive_bytes"]
                    and archive_sha == KNOWN_PUBLIC_PR85["archive_sha256"]
                ),
                "expected_archive_bytes": KNOWN_PUBLIC_PR85["archive_bytes"],
                "expected_archive_sha256": KNOWN_PUBLIC_PR85["archive_sha256"],
            },
            "member_name": info.filename,
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc32_hex": f"{info.CRC:08x}",
            "member_sha256": _sha256_bytes(raw),
            "zip_compress_type": int(info.compress_type),
        },
        raw,
    )


def _zip_info(name: str, *, compress_type: int) -> zipfile.ZipInfo:
    if name not in MEMBER_ORDER:
        raise ValueError(f"unexpected PR85 bridge member: {name!r}")
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_candidate_archive(path: Path, members: dict[str, bytes]) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MEMBER_ORDER if name not in members]
    if missing:
        raise Pr85BundleError(f"PR85 bridge expansion is missing member(s): {missing}")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in MEMBER_ORDER:
            zf.writestr(
                _zip_info(name, compress_type=zipfile.ZIP_DEFLATED),
                members[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    with zipfile.ZipFile(path, "r") as zf:
        rows = []
        for info in zf.infolist():
            rows.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc32_hex": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(zf.read(info)),
                    "zip_compress_type": int(info.compress_type),
                }
            )
    return rows


def _runtime_gate(runtime_dir: Path) -> dict[str, Any]:
    inflate_renderer = runtime_dir / "inflate_renderer.py"
    apply_qpost = runtime_dir / "apply_qzs3_postprocess.py"
    inflate_sh = runtime_dir / "inflate.sh"
    text = inflate_renderer.read_text(encoding="utf-8", errors="replace") if inflate_renderer.is_file() else ""
    qpost_text = apply_qpost.read_text(encoding="utf-8", errors="replace") if apply_qpost.is_file() else ""
    inflate_text = inflate_sh.read_text(encoding="utf-8", errors="replace") if inflate_sh.is_file() else ""
    gates = {
        "qma9_mask_decode_available": "_load_masks_from_qma9" in text and "masks.qma9" in text,
        "qpost_runtime_available": "qpost.bin" in inflate_text and "read_qpost" in qpost_text,
        "qrm1_randmulti_runtime_available": "QRM1" in qpost_text and "_decode_qrm1_randmulti" in qpost_text,
        "p1d1_materialized_to_raw_fp16": True,
        "qh0_model_loader_available": "QH0" in text,
    }
    blockers = []
    if not gates["qh0_model_loader_available"]:
        blockers.append("qh0_model_loader_available")
    for gate_id, passed in gates.items():
        if gate_id != "qh0_model_loader_available" and not passed:
            blockers.append(gate_id)
    return {
        "runtime_dir": _repo_rel(runtime_dir),
        "gates": gates,
        "remaining_blockers": blockers,
        "ready_for_exact_eval_dispatch_claim": not blockers,
        "claim_required_before_remote_eval": True,
    }


def build_bridge_candidate(
    archive: Path,
    out_dir: Path,
    *,
    candidate_id: str = "expanded_qpost_qrm1_posefp16",
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
) -> dict[str, Any]:
    source, raw = _read_pr85_archive(archive)
    expansion = expand_pr85_bundle_to_runtime_members(raw, transcode_randmulti_qrm1=True)
    candidate_dir = out_dir / candidate_id
    candidate_archive = candidate_dir / "archive.zip"
    zip_members = _write_candidate_archive(candidate_archive, dict(expansion.members))
    runtime_gate = _runtime_gate(robust_current_dir)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "source_archive": source,
        "candidate_archive": {
            "path": _repo_rel(candidate_archive),
            "archive_bytes": int(candidate_archive.stat().st_size),
            "archive_sha256": _sha256_file(candidate_archive),
            "member_count": len(zip_members),
            "members": zip_members,
        },
        "runtime_expansion": expansion.manifest,
        "runtime_gate": runtime_gate,
        "dispatch_gate": (
            "eligible_for_exact_eval_after_lane_claim"
            if runtime_gate["ready_for_exact_eval_dispatch_claim"]
            else "blocked_local_runtime_bridge"
        ),
    }
    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-id", default="expanded_qpost_qrm1_posefp16")
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    args = parser.parse_args(argv)

    manifest = build_bridge_candidate(
        args.archive,
        args.out_dir,
        candidate_id=args.candidate_id,
        robust_current_dir=args.robust_current_dir,
    )
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "candidate_count": 1,
        "ready_for_exact_eval_dispatch_claim": manifest["runtime_gate"][
            "ready_for_exact_eval_dispatch_claim"
        ],
        "remaining_blockers": manifest["runtime_gate"]["remaining_blockers"],
        "candidates": [manifest],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidate_summary.json").write_text(_json_text(summary), encoding="utf-8")
    print(_json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
