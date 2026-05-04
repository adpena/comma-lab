#!/usr/bin/env python3
"""Build deterministic PR85 side-channel ablation archives.

The archives produced here are attribution candidates only. They keep PR85's
single-member ``x`` bundle contract but replace selected side-channel segments
with neutral byte-closed payloads. They do not claim score or dispatch GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import brotli

from profile_pr85_adaptive_masking_sidechannel_attribution import (
    DEFAULT_ARCHIVE,
    FIXED_V5_BIAS_BYTES,
    FIXED_V5_REGION_BYTES,
    HEADERLESS_RANDMULTI_SPECS,
    SEGMENT_ORDER,
    _read_single_member_archive,
    parse_pr85_v5_bundle,
)


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_OUT_DIR = Path("experiments/results/public_pr85_sidechannel_ablations_20260503_codex")
TOOL = "experiments/build_pr85_sidechannel_ablation_candidates.py"
SCHEMA = "pr85_sidechannel_ablation_candidates_v1"
FIXED_LENGTH_SEGMENTS = {
    "bias": FIXED_V5_BIAS_BYTES,
    "region": FIXED_V5_REGION_BYTES,
}


ABLATION_POLICIES: dict[str, tuple[str, ...]] = {
    "minus_motion_stack": ("shift", "frac", "frac2", "frac3"),
    "minus_randmulti": ("randmulti",),
    "minus_post": ("post",),
    "minus_post_motion": ("post", "shift", "frac", "frac2", "frac3"),
    "minus_all_safe_corrections": (
        "post",
        "shift",
        "frac",
        "frac2",
        "frac3",
        "randmulti",
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _u24(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError(f"cannot encode {value} as uint24")
    return int(value).to_bytes(3, "little")


def _zip_info(name: str) -> zipfile.ZipInfo:
    if name != "x":
        raise ValueError("PR85 ablation archives must use member x")
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), payload)


def _neutral_segment(name: str) -> bytes:
    if name == "post":
        return brotli.compress(bytes(600 * 4), quality=11)
    if name == "shift":
        return brotli.compress(b"SD4" + bytes(600), quality=11)
    if name == "frac":
        return brotli.compress(b"FV1" + (0).to_bytes(2, "little"), quality=11)
    if name == "frac2":
        return brotli.compress(b"FH2" + bytes([4]) * 600, quality=11)
    if name == "frac3":
        return brotli.compress(b"FD3" + bytes(600), quality=11)
    if name == "bias":
        return brotli.compress(b"BD1" + bytes(600), quality=11)
    if name == "region":
        return brotli.compress(b"RH1" + bytes(600), quality=11)
    if name == "randmulti":
        rows = sum(spec[3] for spec in HEADERLESS_RANDMULTI_SPECS)
        return brotli.compress(bytes(rows), quality=11)
    raise ValueError(f"no neutralizer for segment {name!r}")


def _validate_replacement_segment(name: str, source: bytes, replacement: bytes) -> None:
    """Fail closed when PR85's public runtime cannot recover segment boundaries."""

    fixed_len = FIXED_LENGTH_SEGMENTS.get(name)
    if fixed_len is None:
        return
    if len(source) != fixed_len:
        raise ValueError(
            f"source {name!r} length {len(source)} does not match PR85 v5 fixed length {fixed_len}"
        )
    if len(replacement) != fixed_len:
        raise ValueError(
            f"cannot replace fixed-length PR85 v5 segment {name!r}: "
            f"replacement has {len(replacement)} bytes, runtime requires {fixed_len}"
        )


def _pack_bundle(segments: dict[str, bytes]) -> bytes:
    header_names = SEGMENT_ORDER[:8]
    header = b"".join(_u24(len(segments[name])) for name in header_names)
    return header + b"".join(segments[name] for name in SEGMENT_ORDER)


def _member_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one zip member, found {len(infos)}")
        info = infos[0]
        member = zf.read(info)
    return {
        "archive_path": str(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_bytes": int(len(member)),
        "member_sha256": _sha256(member),
        "zip_stored": info.compress_type == zipfile.ZIP_STORED,
    }


def build_candidates(
    archive: Path,
    out_dir: Path,
    *,
    policy_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_archive, raw = _read_single_member_archive(archive, "x")
    bundle, source_segments = parse_pr85_v5_bundle(raw)
    selected = policy_ids or sorted(ABLATION_POLICIES)
    rows = []
    for policy_id in selected:
        if policy_id not in ABLATION_POLICIES:
            raise ValueError(f"unknown policy {policy_id!r}")
        neutralized = ABLATION_POLICIES[policy_id]
        segments = dict(source_segments)
        replacements = {}
        for name in neutralized:
            replacement = _neutral_segment(name)
            _validate_replacement_segment(name, source_segments[name], replacement)
            segments[name] = replacement
            replacements[name] = {
                "source_bytes": int(len(source_segments[name])),
                "source_sha256": _sha256(source_segments[name]),
                "neutral_bytes": int(len(replacement)),
                "neutral_sha256": _sha256(replacement),
                "byte_delta": int(len(replacement) - len(source_segments[name])),
            }
        candidate_dir = out_dir / policy_id
        archive_path = candidate_dir / "archive.zip"
        payload = _pack_bundle(segments)
        _write_archive(archive_path, payload)
        info = _member_info(archive_path)
        manifest = {
            "schema": "pr85_sidechannel_ablation_candidate_v1",
            "tool": TOOL,
            "policy_id": policy_id,
            "score_claim": False,
            "dispatch_status": "not_dispatched",
            "evidence_grade": "empirical_archive_attribution_candidate",
            "source_archive": source_archive,
            "bundle_format": bundle["format"],
            "fixed_length_segments": bundle.get("fixed_length_segments", {}),
            "neutralized_segments": list(neutralized),
            "replacements": replacements,
            "candidate": info,
            "byte_delta_vs_source_archive": int(info["archive_bytes"] - source_archive["bytes"]),
            "next_gate": "Run PR85 public-runtime exact CUDA eval only after lane claim; compare against exact PR85 replay.",
        }
        (candidate_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(manifest)
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "source_archive": source_archive,
        "candidate_count": len(rows),
        "candidates": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", action="append", dest="policies")
    args = parser.parse_args(argv)

    payload = build_candidates(args.archive, args.out_dir, policy_ids=args.policies)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
