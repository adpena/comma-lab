#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build PR85-family archives stacked with PR89's charged final-bias atom.

PR89's public archive keeps PR85's single-member ``x`` payload byte-identical
and adds a 300-byte ``fb`` member consumed by the public PR89 inflate runtime.
This tool makes that atom reusable for PR85-derived archive candidates: it
copies each candidate's ``x`` member, appends the charged ``fb`` member, and
emits a deterministic two-member ZIP plus manifest.

The emitted archives are exact-eval candidates only. They are not score claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FB_ARCHIVE = REPO_ROOT / "experiments/results/public_pr89_intake_20260504_codex/archive.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_final_bias_stack_candidates_20260504_codex"
DEFAULT_INPUTS = [
    REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip",
    REPO_ROOT / "experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/waterfill_top001/archive.zip",
    REPO_ROOT / "experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/waterfill_top004/archive.zip",
    REPO_ROOT / "experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/waterfill_top008/archive.zip",
    REPO_ROOT / "experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/waterfill_top016/archive.zip",
]
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SCHEMA = "pr85_final_bias_stack_candidates_v1"
MANIFEST_SCHEMA = "pr85_final_bias_stack_candidate_v1"
TOOL = "experiments/build_pr85_final_bias_stack_candidates.py"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    if name not in {"x", "fb"}:
        raise ValueError(f"unsupported member name {name!r}")
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _read_required_member(path: Path, member_name: str) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if member_name not in names:
            raise ValueError(f"{path} does not contain required member {member_name!r}; found {names}")
        if len(names) != len(set(names)):
            raise ValueError(f"{path} contains duplicate zip member names")
        info = zf.getinfo(member_name)
        raw = zf.read(info)
    return raw, {
        "archive_path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": member_name,
        "member_bytes": int(len(raw)),
        "member_sha256": _sha256(raw),
        "source_zip_stored": info.compress_type == zipfile.ZIP_STORED,
        "source_zip_compress_type": int(info.compress_type),
    }


def _archive_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        members = []
        for info in infos:
            raw = zf.read(info)
            members.append(
                {
                    "name": info.filename,
                    "bytes": int(len(raw)),
                    "sha256": _sha256(raw),
                    "zip_stored": info.compress_type == zipfile.ZIP_STORED,
                }
            )
    return {
        "archive_path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "members": members,
    }


def _candidate_id(path: Path) -> str:
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    return path.stem


def build_candidates(
    *,
    input_archives: Sequence[Path],
    fb_archive: Path,
    out_dir: Path,
) -> dict[str, Any]:
    fb_raw, fb_info = _read_required_member(fb_archive, "fb")
    if len(fb_raw) != 300:
        raise ValueError(f"expected PR89 fb member to be 300 bytes, got {len(fb_raw)}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source_archive in input_archives:
        source_archive = source_archive.resolve()
        x_raw, x_info = _read_required_member(source_archive, "x")
        cid = _candidate_id(source_archive)
        if cid in seen_ids:
            raise ValueError(f"duplicate candidate id {cid!r}")
        seen_ids.add(cid)
        candidate_dir = out_dir / f"{cid}_fb"
        archive_path = candidate_dir / "archive.zip"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr(_zip_info("x"), x_raw)
            zf.writestr(_zip_info("fb"), fb_raw)
        candidate = _archive_info(archive_path)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool": TOOL,
            "score_claim": False,
            "dispatch_performed": False,
            "evidence_grade": "empirical_archive_stack_candidate",
            "candidate_id": f"{cid}_fb",
            "source_x": x_info,
            "source_fb": fb_info,
            "candidate": candidate,
            "byte_delta_vs_source_x_archive": int(candidate["archive_bytes"] - x_info["archive_bytes"]),
            "charged_fb_bytes": int(len(fb_raw)),
            "runtime_contract": {
                "required_inflate_family": "public_pr89_henosis_final_bias",
                "required_members": ["x", "fb"],
                "score_truth": "exact CUDA auth eval only",
            },
            "next_gate": "Exact CUDA eval with PR89 public inflate runtime after lane claim.",
        }
        _write_json(candidate_dir / "manifest.json", manifest)
        rows.append(manifest)
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "fb_source": fb_info,
        "candidate_count": len(rows),
        "candidates": rows,
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-archive", action="append", type=Path, dest="input_archives")
    parser.add_argument("--fb-archive", type=Path, default=DEFAULT_FB_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_archives = args.input_archives or DEFAULT_INPUTS
    summary = build_candidates(
        input_archives=[Path(p) for p in input_archives],
        fb_archive=args.fb_archive,
        out_dir=args.out_dir,
    )
    print(_json_text({"out_dir": _rel(args.out_dir), "candidate_count": summary["candidate_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
