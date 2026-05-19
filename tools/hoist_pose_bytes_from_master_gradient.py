#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""OP-7 planning manifest for pose-axis master-gradient byte hoists.

This tool is intentionally planning-only. It turns the aggregate master-gradient
pose-axis dominance selector into a durable manifest of typed
``CandidateModificationSpec`` rows, while keeping raw diagnostic byte indices
non-dispatchable until a grammar-aware mutation builder proves packet closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

DEFAULT_ARCHIVE_SHA256 = "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "pose_byte_hoist_op7_20260518"
DEFAULT_MANIFEST_PATH = REPO_ROOT / ".omx" / "research" / "pose_byte_hoist_op7_manifest_20260518.json"
SCHEMA_VERSION = "pose_byte_hoist_op7_manifest_v1"
TOOL_ID = "tools/hoist_pose_bytes_from_master_gradient.py"


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(path_obj: object) -> Path | None:
    if not isinstance(path_obj, str) or not path_obj:
        return None
    path = Path(path_obj)
    if path.is_absolute():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return Path.cwd() / path


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdefABCDEF" for c in value)
    )


def build_pose_byte_hoist_manifest(
    *,
    archive_sha256: str = DEFAULT_ARCHIVE_SHA256,
    top_k: int = 128,
    axis_dominance_threshold: float = 0.7,
    anchor_path: Path | None = None,
    layout_manifest_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build the OP-7 planning manifest and companion selector sidecar."""

    from tac.master_gradient import MASTER_GRADIENT_LEDGER_PATH, contest_axis_authority_violation_reason
    from tac.master_gradient_consumers import (
        load_aggregate_gradient_from_anchor,
        select_pose_axis_dominant_bytes,
    )
    from tac.master_gradient_operator_plan import build_pose_axis_operator_candidates

    output_dir = output_dir.resolve()
    sidecar_root = output_dir / "master_gradient_consumers"
    sidecar_root.mkdir(parents=True, exist_ok=True)
    selector_sidecar_path = sidecar_root / f"pose_axis_dominant_bytes_{archive_sha256[:12]}_op7_manifest_v1.json"

    _array, anchor = load_aggregate_gradient_from_anchor(
        archive_sha256=archive_sha256,
        anchor_path=anchor_path,
    )
    specs = select_pose_axis_dominant_bytes(
        archive_sha256,
        top_k=top_k,
        axis_dominance_threshold=axis_dominance_threshold,
        anchor_path=anchor_path,
        write_sidecar=True,
        output_root=sidecar_root,
        sidecar_path=selector_sidecar_path,
    )

    spec_manifests = [spec.to_manifest() for spec in specs]
    blockers = [str(blocker) for spec in specs for blocker in spec.blockers]
    if not specs:
        blockers.append("no_pose_axis_dominant_bytes_selected")
    if not isinstance(anchor.get("score_axis_dominance"), dict):
        blockers.append("anchor_score_axis_dominance_not_persisted")

    resolved_layout_manifest_path = None
    operator_candidate_resolution: dict[str, Any] | None = None
    operator_candidate_resolution_error: str | None = None
    if layout_manifest_path is None:
        blockers.extend([
            "grammar_aware_layout_manifest_missing",
            "grammar_aware_pose_axis_mutation_builder_missing",
        ])
    else:
        resolved_layout_manifest_path = (
            layout_manifest_path
            if layout_manifest_path.is_absolute()
            else REPO_ROOT / layout_manifest_path
        )
        try:
            layout_payload = json.loads(resolved_layout_manifest_path.read_text(encoding="utf-8"))
            selector_payload = json.loads(selector_sidecar_path.read_text(encoding="utf-8"))
            operator_candidate_resolution = build_pose_axis_operator_candidates(
                selector_payload,
                layout_payload,
                packet_proofs_available=False,
            )
            blockers.extend(str(blocker) for blocker in operator_candidate_resolution["blockers"])
            if operator_candidate_resolution.get("resolved_count"):
                blockers = [
                    blocker for blocker in blockers
                    if blocker != "grammar_aware_pose_axis_mutation_builder_missing"
                ]
            if not operator_candidate_resolution.get("resolved_count"):
                blockers.append("grammar_aware_pose_axis_candidates_unresolved")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            operator_candidate_resolution_error = f"{type(exc).__name__}: {exc}"
            blockers.append("grammar_aware_layout_manifest_unreadable")
    blockers.append("packet_proofs_missing")

    anchor_violation = contest_axis_authority_violation_reason(anchor)
    anchor_ledger_path = (anchor_path or MASTER_GRADIENT_LEDGER_PATH).resolve()
    gradient_array_path = _resolve_path(anchor.get("gradient_array_path"))
    scored_archive_custody_available = (
        _is_sha256(anchor.get("scored_archive_sha256"))
        and isinstance(anchor.get("scored_archive_bytes"), int)
        and not isinstance(anchor.get("scored_archive_bytes"), bool)
    )
    source_anchor = {
        "archive_sha256": anchor.get("archive_sha256"),
        "scored_archive_sha256": anchor.get("scored_archive_sha256"),
        "scored_archive_bytes": anchor.get("scored_archive_bytes"),
        "scored_archive_custody_available": scored_archive_custody_available,
        "anchor_row_canonical_json_sha256": _sha256_json(anchor),
        "anchor_ledger_path": _repo_rel(anchor_ledger_path),
        "anchor_ledger_sha256": _sha256_file(anchor_ledger_path),
        "gradient_array_path": anchor.get("gradient_array_path"),
        "gradient_array_sha256": _sha256_file(gradient_array_path),
        "gradient_tensor_kind": anchor.get("gradient_tensor_kind"),
        "gradient_byte_domain": anchor.get("gradient_byte_domain"),
        "measurement_axis": anchor.get("measurement_axis"),
        "measurement_hardware": anchor.get("measurement_hardware"),
        "measurement_method": anchor.get("measurement_method"),
        "measurement_utc": anchor.get("measurement_utc"),
        "n_bytes": anchor.get("n_bytes"),
        "n_pairs_used": anchor.get("n_pairs_used"),
        "n_pairs_total": anchor.get("n_pairs_total"),
        "score_axis_dominance_available": isinstance(anchor.get("score_axis_dominance"), dict),
        "score_axis_dominance_source": (
            "anchor_field"
            if isinstance(anchor.get("score_axis_dominance"), dict)
            else "derived_from_gradient_tensor_at_runtime"
        ),
        "contest_axis_authority_violation_reason": anchor_violation,
    }

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_ID,
        "archive_sha256": archive_sha256,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "evidence_grade": "[diagnostic; master-gradient planning manifest]",
        "source_anchor": source_anchor,
        "authority_boundary": {
            "planning_only": True,
            "score_axis_authority": False,
            "source_archive_custody_available": scored_archive_custody_available,
            "raw_archive_byte_authority": False,
            "candidate_specs_are_dispatchable": False,
            "reason": (
                "diagnostic gradient-subject coordinates require scored archive custody, "
                "grammar resolution, packet proofs, and exact eval before promotion"
            ),
        },
        "selection": {
            "selector": "tac.master_gradient_consumers.select_pose_axis_dominant_bytes",
            "top_k": int(top_k),
            "axis_dominance_threshold": float(axis_dominance_threshold),
            "selected_count": len(specs),
            "axis_label": "pose",
            "coordinate_system": "grammar_aware_operator_response",
            "raw_archive_byte_coordinates_allowed": False,
        },
        "selector_sidecar_path": _repo_rel(selector_sidecar_path),
        "selector_sidecar_sha256": _sha256_file(selector_sidecar_path),
        "grammar_aware_layout_manifest_path": (
            _repo_rel(resolved_layout_manifest_path)
            if resolved_layout_manifest_path is not None
            else None
        ),
        "grammar_aware_layout_manifest_sha256": _sha256_file(resolved_layout_manifest_path),
        "grammar_aware_operator_candidate_resolution": operator_candidate_resolution,
        "grammar_aware_operator_candidate_resolution_error": operator_candidate_resolution_error,
        "candidate_modification_specs": spec_manifests,
        "packet_proofs": {
            "repacked_archive": False,
            "updated_zip_headers": False,
            "updated_zip_crc": False,
            "inflate_success_proof": False,
            "byte_consumption_noop_detector": False,
        },
        "smoke": {
            "attempted": False,
            "status": (
                "blocked_missing_packet_proofs"
                if operator_candidate_resolution is not None
                and operator_candidate_resolution.get("resolved_count")
                else "blocked_missing_grammar_aware_pose_axis_mutation_builder"
            ),
            "reason": (
                "OP-7 resolved at least one diagnostic gradient-subject byte "
                "through a grammar layout; mutation remains blocked until "
                "repack/CRC/inflate/no-op proof."
                if operator_candidate_resolution is not None
                and operator_candidate_resolution.get("resolved_count")
                else (
                    "OP-7 selected diagnostic gradient-subject byte indices only; "
                    "no raw archive-byte mutation is packet-valid without grammar "
                    "resolution plus repack/CRC/inflate/no-op proof."
                )
            ),
        },
        "blockers": _unique(blockers),
        "next_step": (
            "Build a grammar-aware pose-axis mutation builder for the highest-EV "
            "selected spec, then prove repack, CRC, inflate success, and byte "
            "consumption before any operator probe or provider dispatch."
        ),
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_path"] = _repo_rel(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-sha256", default=DEFAULT_ARCHIVE_SHA256)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--axis-dominance-threshold", type=float, default=0.7)
    parser.add_argument("--anchor-path", type=Path, default=None)
    parser.add_argument("--layout-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir
    manifest = build_pose_byte_hoist_manifest(
        archive_sha256=args.archive_sha256,
        top_k=args.top_k,
        axis_dominance_threshold=args.axis_dominance_threshold,
        anchor_path=args.anchor_path,
        layout_manifest_path=args.layout_manifest,
        output_dir=output_dir,
    )
    path = write_manifest(manifest, args.manifest_path)
    print(json.dumps({"manifest_path": _repo_rel(path), "selected_count": manifest["selection"]["selected_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
