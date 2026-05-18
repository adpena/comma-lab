#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DP1+PR101 composition no-op detector and byte-closure probe.

This probe verifies the current L1 DP1+fec6/PR101 composition packet without
running the scorer or launching provider work. It starts from bytes:

    DPCOMP header(13) + DP1 prior bytes + fec6 base archive bytes

The expected L1 verdict is intentionally conservative: the packet is
byte-closed and structurally consumes the DP1 prefix during inflate, but the
frame-axis prior effect is still gated behind the L2
``PACT_DP1_PRIOR_STRENGTH > 0.0`` RuntimeError. That means this artifact can
certify rate-axis cost and no-op discipline; it cannot claim score movement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.pretrained_driving_prior.composition import (  # noqa: E402
    DPCOMP_HEADER_SIZE,
    DPCOMP_MAGIC,
    DPCOMP_SCHEMA_VERSION,
)

SCHEMA = "dp1_pr101_composition_noop_detector_v1"
PROBE_ID = "dp1_pr101_composition_noop_detector"
LANE_ID = "lane_dp1_plus_fec6_dual_stacking_build_20260517"
DEFAULT_PACKET_DIR = Path("experiments/results/dp1_plus_fec6_composition_20260517")
DEFAULT_OUTPUT_JSON = Path(
    ".omx/research/dp1_pr101_composition_noop_probe_20260518_codex.json"
)
CONTEST_ARCHIVE_NORMALIZER_BYTES = 37_545_489.0

_DPCOMP_HEADER_FMT = "<4sBI4s"
_BASE_TAGS = {
    b"A1\x00\x00": "a1",
    b"PR01": "pr101",
    b"HDM8": "hdm8",
    b"YUCR": "yucr",
    b"TT5L": "time_traveler_l5",
    b"SHN1": "sane_hnerv",
}

FALSE_AUTHORITY_FLAGS = {
    "research_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}

NO_SCORE_AUTHORITY_BLOCKERS = [
    "no_paired_contest_cuda_cpu_eval",
    "frame_axis_effect_deferred_to_l2",
    "not_score_authority",
]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path, blockers: list[str], label: str) -> dict[str, Any]:
    if not path.exists():
        blockers.append(f"{label}_missing")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append(f"{label}_unreadable")
        return {}
    if not isinstance(payload, dict):
        blockers.append(f"{label}_not_json_object")
        return {}
    return payload


def _repo_rel(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_manifest_path(raw: Any, repo_root: Path) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _manifest_sha(
    manifest: dict[str, Any],
    *keys: str,
) -> str | None:
    cur: Any = manifest
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return str(cur) if cur not in (None, "") else None


def _manifest_int(manifest: dict[str, Any], *keys: str) -> int | None:
    cur: Any = manifest
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    try:
        return int(cur)
    except (TypeError, ValueError):
        return None


def _compare_sha(
    blockers: list[str],
    *,
    label: str,
    expected: str | None,
    actual: str,
) -> None:
    if not expected:
        blockers.append(f"{label}_missing")
    elif expected != actual:
        blockers.append(f"{label}_mismatch")


def _compare_int(
    blockers: list[str],
    *,
    label: str,
    expected: int | None,
    actual: int,
) -> None:
    if expected is None:
        blockers.append(f"{label}_missing")
    elif expected != actual:
        blockers.append(f"{label}_mismatch:{expected}!={actual}")


def parse_dcomp_components(archive_bytes: bytes) -> tuple[dict[str, Any], list[str]]:
    """Parse DPCOMP bytes without requiring the embedded DP1 archive to parse."""

    blockers: list[str] = []
    if len(archive_bytes) < DPCOMP_HEADER_SIZE:
        return (
            {
                "header_size_bytes": DPCOMP_HEADER_SIZE,
                "archive_size_bytes": len(archive_bytes),
                "dp1_prefix_size_bytes": None,
                "base_archive_size_bytes": None,
                "base_substrate": None,
                "schema_version": None,
            },
            [f"archive_too_short_for_dcomp_header:{len(archive_bytes)}"],
        )

    magic, version, dp1_len, base_tag = struct.unpack(
        _DPCOMP_HEADER_FMT, archive_bytes[:DPCOMP_HEADER_SIZE]
    )
    if magic != DPCOMP_MAGIC:
        blockers.append("dcomp_magic_mismatch")
    if version != DPCOMP_SCHEMA_VERSION:
        blockers.append(f"dcomp_schema_version_mismatch:{version}")
    base_substrate = _BASE_TAGS.get(base_tag)
    if base_substrate is None:
        blockers.append(f"dcomp_base_tag_unknown:{base_tag!r}")

    base_start = DPCOMP_HEADER_SIZE + int(dp1_len)
    if base_start > len(archive_bytes):
        blockers.append("dcomp_declared_dp1_length_exceeds_archive")
        dp1_bytes = archive_bytes[DPCOMP_HEADER_SIZE:]
        base_bytes = b""
    else:
        dp1_bytes = archive_bytes[DPCOMP_HEADER_SIZE:base_start]
        base_bytes = archive_bytes[base_start:]

    if len(archive_bytes) != DPCOMP_HEADER_SIZE + len(dp1_bytes) + len(base_bytes):
        blockers.append("dcomp_length_identity_failed")

    return (
        {
            "archive_size_bytes": len(archive_bytes),
            "archive_sha256": _sha256_bytes(archive_bytes),
            "header_size_bytes": DPCOMP_HEADER_SIZE,
            "schema_version": version,
            "base_substrate": base_substrate,
            "base_tag_hex": base_tag.hex(),
            "dp1_prefix_size_bytes": len(dp1_bytes),
            "dp1_prefix_sha256": _sha256_bytes(dp1_bytes),
            "base_archive_size_bytes": len(base_bytes),
            "base_archive_sha256": _sha256_bytes(base_bytes),
        },
        blockers,
    )


def _check_source_file(
    *,
    label: str,
    path: Path | None,
    expected_sha: str | None,
    expected_size: int | None,
    blockers: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    if path is None:
        blockers.append(f"{label}_source_path_missing")
        return {
            "path": None,
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        }
    row: dict[str, Any] = {
        "path": _repo_rel(path, repo_root),
        "exists": path.exists(),
        "sha256": None,
        "size_bytes": None,
    }
    if not path.exists():
        blockers.append(f"{label}_source_file_missing")
        return row
    blob = path.read_bytes()
    actual_sha = _sha256_bytes(blob)
    row["sha256"] = actual_sha
    row["size_bytes"] = len(blob)
    _compare_sha(
        blockers,
        label=f"{label}_source_sha256",
        expected=expected_sha,
        actual=actual_sha,
    )
    _compare_int(
        blockers,
        label=f"{label}_source_size_bytes",
        expected=expected_size,
        actual=len(blob),
    )
    return row


def _check_inflate_l2_guard(inflate_py: Path, blockers: list[str]) -> dict[str, Any]:
    row = {
        "path": str(inflate_py),
        "exists": inflate_py.exists(),
        "has_strength_env": False,
        "has_l2_runtime_guard": False,
        "has_strength_gt_zero_guard": False,
    }
    if not inflate_py.exists():
        blockers.append("inflate_py_missing")
        return row
    text = inflate_py.read_text(encoding="utf-8", errors="ignore")
    row["has_strength_env"] = "PACT_DP1_PRIOR_STRENGTH" in text
    row["has_l2_runtime_guard"] = "requires L2 INTEGRATION" in text
    row["has_strength_gt_zero_guard"] = "strength > 0.0" in text
    if not row["has_strength_env"]:
        blockers.append("inflate_py_dp1_strength_env_missing")
    if not row["has_l2_runtime_guard"] or not row["has_strength_gt_zero_guard"]:
        blockers.append("inflate_py_l2_strength_guard_missing")
    return row


def build_probe_payload(
    packet_dir: Path = DEFAULT_PACKET_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Return a structural no-op proof payload for a DP1+PR101 packet."""

    packet_dir = packet_dir if packet_dir.is_absolute() else repo_root / packet_dir
    archive_path = packet_dir / "archive.zip"
    archive_manifest_path = packet_dir / "archive_manifest.json"
    build_manifest_path = packet_dir / "build_manifest.json"
    inflate_py_path = packet_dir / "inflate.py"

    structural_blockers: list[str] = []
    archive_manifest = _load_json(
        archive_manifest_path, structural_blockers, "archive_manifest"
    )
    build_manifest = _load_json(
        build_manifest_path, structural_blockers, "build_manifest"
    )

    if archive_path.exists():
        archive_bytes = archive_path.read_bytes()
    else:
        archive_bytes = b""
        structural_blockers.append("archive_zip_missing")

    components, parse_blockers = parse_dcomp_components(archive_bytes)
    structural_blockers.extend(parse_blockers)

    _compare_sha(
        structural_blockers,
        label="archive_manifest_archive_sha256",
        expected=_manifest_sha(archive_manifest, "archive_sha256"),
        actual=str(components.get("archive_sha256") or ""),
    )
    _compare_int(
        structural_blockers,
        label="archive_manifest_archive_size_bytes",
        expected=_manifest_int(archive_manifest, "archive_size_bytes"),
        actual=int(components.get("archive_size_bytes") or 0),
    )
    _compare_sha(
        structural_blockers,
        label="build_manifest_archive_sha256",
        expected=_manifest_sha(build_manifest, "archive_sha256"),
        actual=str(components.get("archive_sha256") or ""),
    )
    _compare_int(
        structural_blockers,
        label="build_manifest_archive_size_bytes",
        expected=_manifest_int(build_manifest, "archive_size_bytes"),
        actual=int(components.get("archive_size_bytes") or 0),
    )
    _compare_int(
        structural_blockers,
        label="archive_manifest_header_size_bytes",
        expected=_manifest_int(archive_manifest, "header_size_bytes"),
        actual=DPCOMP_HEADER_SIZE,
    )
    _compare_int(
        structural_blockers,
        label="archive_manifest_composition_schema_version",
        expected=_manifest_int(archive_manifest, "composition_schema_version"),
        actual=DPCOMP_SCHEMA_VERSION,
    )

    build_manifest_lane_id = build_manifest.get("lane_id")
    if build_manifest_lane_id != LANE_ID:
        structural_blockers.append("build_manifest_lane_id_mismatch")
    if archive_manifest.get("base_substrate") != "pr101":
        structural_blockers.append("archive_manifest_base_substrate_not_pr101")
    if components.get("base_substrate") != "pr101":
        structural_blockers.append("dcomp_base_substrate_not_pr101")

    dp1_sha = str(components.get("dp1_prefix_sha256") or "")
    dp1_size = int(components.get("dp1_prefix_size_bytes") or 0)
    base_sha = str(components.get("base_archive_sha256") or "")
    base_size = int(components.get("base_archive_size_bytes") or 0)

    _compare_sha(
        structural_blockers,
        label="archive_manifest_dp1_source_sha256",
        expected=_manifest_sha(archive_manifest, "dp1_source_sha256"),
        actual=dp1_sha,
    )
    _compare_int(
        structural_blockers,
        label="archive_manifest_dp1_source_size_bytes",
        expected=_manifest_int(archive_manifest, "dp1_source_size_bytes"),
        actual=dp1_size,
    )
    _compare_sha(
        structural_blockers,
        label="archive_manifest_fec6_source_sha256",
        expected=_manifest_sha(archive_manifest, "fec6_source_sha256"),
        actual=base_sha,
    )
    _compare_int(
        structural_blockers,
        label="archive_manifest_fec6_source_size_bytes",
        expected=_manifest_int(archive_manifest, "fec6_source_size_bytes"),
        actual=base_size,
    )

    _compare_sha(
        structural_blockers,
        label="build_manifest_dp1_source_sha256",
        expected=_manifest_sha(build_manifest, "dp1_source", "sha256"),
        actual=dp1_sha,
    )
    _compare_int(
        structural_blockers,
        label="build_manifest_dp1_source_size_bytes",
        expected=_manifest_int(build_manifest, "dp1_source", "size_bytes"),
        actual=dp1_size,
    )
    _compare_sha(
        structural_blockers,
        label="build_manifest_fec6_source_sha256",
        expected=_manifest_sha(build_manifest, "fec6_source", "sha256"),
        actual=base_sha,
    )
    _compare_int(
        structural_blockers,
        label="build_manifest_fec6_source_size_bytes",
        expected=_manifest_int(build_manifest, "fec6_source", "size_bytes"),
        actual=base_size,
    )

    dp1_source_path = _resolve_manifest_path(
        (build_manifest.get("dp1_source") or {}).get("path"), repo_root
    )
    fec6_source_path = _resolve_manifest_path(
        (build_manifest.get("fec6_source") or {}).get("path"), repo_root
    )
    source_files = {
        "dp1_source": _check_source_file(
            label="dp1",
            path=dp1_source_path,
            expected_sha=dp1_sha,
            expected_size=dp1_size,
            blockers=structural_blockers,
            repo_root=repo_root,
        ),
        "fec6_source": _check_source_file(
            label="fec6",
            path=fec6_source_path,
            expected_sha=base_sha,
            expected_size=base_size,
            blockers=structural_blockers,
            repo_root=repo_root,
        ),
    }

    if build_manifest.get("operational_mechanism_status") != "OPERATIONAL_DEFERRED_TO_L2":
        structural_blockers.append("operational_mechanism_status_not_deferred_to_l2")
    inflate_guard = _check_inflate_l2_guard(inflate_py_path, structural_blockers)

    archive_size = int(components.get("archive_size_bytes") or 0)
    rate_axis_delta = (
        25.0 * (archive_size - base_size) / CONTEST_ARCHIVE_NORMALIZER_BYTES
        if archive_size >= base_size and base_size > 0
        else None
    )
    if rate_axis_delta is not None and not math.isfinite(rate_axis_delta):
        structural_blockers.append("rate_axis_delta_not_finite")
        rate_axis_delta = None

    structural_pass = not structural_blockers
    verdict = (
        "l1_rate_only_noop_verified"
        if structural_pass
        else "blocked_structural_mismatch"
    )

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "created_utc": created_utc or _utc_now(),
        "packet_dir": _repo_rel(packet_dir, repo_root),
        "archive_path": _repo_rel(archive_path, repo_root),
        "archive_manifest_path": _repo_rel(archive_manifest_path, repo_root),
        "build_manifest_path": _repo_rel(build_manifest_path, repo_root),
        "build_manifest_lane_id": build_manifest_lane_id,
        "verdict": verdict,
        "structural_pass": structural_pass,
        "evidence_axis": "[byte-closed local structural proof]",
        "archive_components": components,
        "source_files": source_files,
        "inflate_l2_guard": inflate_guard,
        "operational_mechanism_status": build_manifest.get(
            "operational_mechanism_status"
        ),
        "frame_axis_effect_status": "deferred_l2_strength_zero",
        "rate_axis_delta_if_frames_identical": (
            round(rate_axis_delta, 12) if rate_axis_delta is not None else None
        ),
        "rate_axis_delta_formula": (
            "25 * (composed_archive_bytes - base_archive_bytes) / 37545489"
        ),
        "structural_blockers": list(dict.fromkeys(structural_blockers)),
        "blockers": list(
            dict.fromkeys([*structural_blockers, *NO_SCORE_AUTHORITY_BLOCKERS])
        ),
        "result_review": {
            "classification": (
                "l1_rate_axis_cost_and_noop_guard_verified"
                if structural_pass
                else "blocked_structural_or_custody_mismatch"
            ),
            "score_authority": "none",
            "reactivation_criteria": [
                "paired same-runtime fec6-vs-composed full-frame inflate parity",
                "L2 DP1 prior application implementation for strength > 0.0",
                "claimed CPU and CUDA exact-eval artifacts for the byte-closed packet",
            ],
        },
        **FALSE_AUTHORITY_FLAGS,
    }


def write_json(payload: dict[str, Any], path: Path, *, repo_root: Path) -> Path:
    out = path if path.is_absolute() else repo_root / path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(  # BARE_WRITE_OK:atomic_replace_single_output_probe_artifact
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)
    return out


def _render_human(payload: dict[str, Any]) -> str:
    components = payload["archive_components"]
    return "\n".join(
        [
            "=== DP1+PR101 composition no-op detector ===",
            f"  verdict: {payload['verdict']}",
            f"  structural_pass: {payload['structural_pass']}",
            f"  axis: {payload['evidence_axis']}",
            f"  archive_sha256: {components.get('archive_sha256')}",
            f"  archive_bytes: {components.get('archive_size_bytes')}",
            f"  dp1_prefix_bytes: {components.get('dp1_prefix_size_bytes')}",
            f"  base_archive_bytes: {components.get('base_archive_size_bytes')}",
            f"  rate_axis_delta_if_frames_identical: {payload['rate_axis_delta_if_frames_identical']}",
            "  false_authority: score_claim=false promotion_eligible=false ready_for_paid_dispatch=false",
            f"  blockers: {len(payload['blockers'])}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON payload")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    payload = build_probe_payload(args.packet_dir, repo_root=repo_root)
    if args.output is not None:
        out = write_json(payload, args.output, repo_root=repo_root)
        print(f"[dp1-noop-probe] wrote artifact: {out}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_human(payload))

    return 0 if payload["structural_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
