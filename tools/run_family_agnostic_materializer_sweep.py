#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded empirical sweeps for family-agnostic materializers."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.family_agnostic_materializers import (  # noqa: E402
    PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    FamilyAgnosticMaterializerError,
    materialize_packet_member_zip_header_elide_candidate,
)
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    require_no_truthy_authority_fields,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_line,
    json_text,
    sha256_file,
    write_json_artifact,
    write_text_artifact,
)
from tac.score_composition import (  # noqa: E402
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
)

SWEEP_SCHEMA = "family_agnostic_materializer_empirical_sweep.v1"
OBSERVATION_SCHEMA = "family_agnostic_materializer_empirical_observation.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}
LOCAL_MATERIALIZER_AXIS = "[local-materializer-proof]"
LOCAL_RATE_SCORE_PER_BYTE = CANONICAL_RATE_MULTIPLIER / float(CANONICAL_RATE_DENOM_BYTES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-kind",
        default=PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        choices=(PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,),
    )
    parser.add_argument(
        "--archive",
        action="append",
        required=True,
        help="Archive path to sweep. Use label=path to provide a stable row label.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--observation-jsonl", type=Path)
    parser.add_argument("--member-name")
    parser.add_argument("--member-names", action="append", default=[])
    parser.add_argument("--all-members", action="store_true")
    parser.add_argument("--packet-member-manifest", type=Path)
    parser.add_argument("--header-elision-contract", type=Path)
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-existing-output-json-sha256")
    parser.add_argument("--expected-existing-observation-jsonl-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = args.output_json or args.output_dir / "sweep.json"
    observation_jsonl = args.observation_jsonl
    try:
        payload = build_materializer_empirical_sweep(
            target_kind=args.target_kind,
            archives=args.archive,
            output_dir=args.output_dir,
            member_name=args.member_name,
            member_names=tuple(args.member_names),
            packet_member_manifest=args.packet_member_manifest,
            header_elision_contract=args.header_elision_contract,
            all_members=args.all_members,
            min_free_bytes=args.min_free_bytes,
            allow_overwrite=args.allow_overwrite,
        )
        write_json_artifact(
            output_json,
            payload,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_existing_output_json_sha256,
            min_free_bytes=args.min_free_bytes,
        )
        if observation_jsonl is not None:
            write_text_artifact(
                observation_jsonl,
                "".join(json_line(row) for row in payload["observations"]),
                allow_overwrite=args.allow_overwrite,
                expected_existing_sha256=args.expected_existing_observation_jsonl_sha256,
                min_free_bytes=args.min_free_bytes,
            )
    except (OSError, ArtifactWriteError, FamilyAgnosticMaterializerError, ValueError) as exc:
        print(f"FATAL: materializer sweep failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(payload), end="")
    return 0


def build_materializer_empirical_sweep(
    *,
    target_kind: str,
    archives: Sequence[str],
    output_dir: str | Path,
    member_name: str | None = None,
    member_names: Sequence[str] = (),
    all_members: bool = False,
    packet_member_manifest: str | Path | Mapping[str, Any] | None = None,
    header_elision_contract: str | Path | Mapping[str, Any] | None = None,
    min_free_bytes: int = 0,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    if target_kind != PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        raise FamilyAgnosticMaterializerError(f"unsupported target kind: {target_kind}")
    archive_specs = [_parse_archive_spec(value) for value in archives]
    if not archive_specs:
        raise FamilyAgnosticMaterializerError("at least one archive is required")
    output_root = _resolve(output_dir)
    rows: list[dict[str, Any]] = []
    for index, (label, archive_path) in enumerate(archive_specs, start=1):
        archive = _resolve(archive_path)
        archive_sha = sha256_file(archive)
        row_slug = _row_slug(label or archive.stem, index=index, archive_sha=archive_sha)
        row_dir = output_root / "rows" / row_slug
        manifest_path = row_dir / "candidate.json"
        candidate_path = row_dir / "candidate.zip"
        proof_path = row_dir / "candidate.runtime_consumption_proof.json"
        result = materialize_packet_member_zip_header_elide_candidate(
            archive_path=archive,
            output_archive=candidate_path,
            packet_member_manifest=packet_member_manifest,
            member_name=member_name,
            member_names=member_names,
            all_members=all_members,
            header_elision_contract=header_elision_contract,
            runtime_consumption_proof_out=proof_path,
            repo_root=REPO_ROOT,
            allow_overwrite=allow_overwrite,
            min_free_bytes=min_free_bytes,
        )
        write_json_artifact(
            manifest_path,
            result,
            allow_overwrite=allow_overwrite,
            min_free_bytes=min_free_bytes,
        )
        rows.append(
            _observation_from_manifest(
                label=label,
                row_slug=row_slug,
                manifest_path=manifest_path,
                result=result,
            )
        )
    rate_positive = [row for row in rows if row["rate_positive"] is True]
    max_saved_bytes = max((int(row["saved_bytes"]) for row in rows), default=0)
    total_saved_bytes = sum(max(0, int(row["saved_bytes"])) for row in rows)
    blockers = []
    if not rate_positive:
        blockers.append("materializer_sweep_found_no_rate_positive_archives")
    return {
        "schema": SWEEP_SCHEMA,
        "target_kind": target_kind,
        "materializer_id": PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER_ID,
        "archive_count": len(rows),
        "observation_count": len(rows),
        "rate_positive_count": len(rate_positive),
        "rate_nonpositive_count": len(rows) - len(rate_positive),
        "max_saved_bytes": max_saved_bytes,
        "total_positive_saved_bytes": total_saved_bytes,
        "planner_feedback": _planner_feedback(rows),
        "blockers": blockers,
        "observations": rows,
        **FALSE_AUTHORITY,
    }


def _observation_from_manifest(
    *,
    label: str | None,
    row_slug: str,
    manifest_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    require_no_truthy_authority_fields(
        result,
        context="family_agnostic_materializer_empirical_sweep.manifest",
    )
    selected = result.get("selected_elision") if isinstance(result.get("selected_elision"), Mapping) else {}
    saved_bytes = int(selected.get("saved_bytes") or 0)
    readiness_blockers = [str(item) for item in result.get("readiness_blockers") or []]
    source_archive = result.get("source_archive") if isinstance(result.get("source_archive"), Mapping) else {}
    candidate_archive = (
        result.get("candidate_archive") if isinstance(result.get("candidate_archive"), Mapping) else {}
    )
    proof_write = (
        result.get("runtime_consumption_proof_write")
        if isinstance(result.get("runtime_consumption_proof_write"), Mapping)
        else {}
    )
    receiver_verification = (
        result.get("receiver_verification")
        if isinstance(result.get("receiver_verification"), Mapping)
        else {}
    )
    rate_positive = saved_bytes > 0 and "candidate_not_rate_positive" not in readiness_blockers
    observed_rate_gain = LOCAL_RATE_SCORE_PER_BYTE * float(saved_bytes) if rate_positive else 0.0
    return {
        "schema": OBSERVATION_SCHEMA,
        "observation_kind": "family_agnostic_materializer_empirical_observation",
        "observation_id": row_slug,
        "candidate_id": row_slug,
        "axis": LOCAL_MATERIALIZER_AXIS,
        "resource_kind": "local_cpu",
        "runtime_identity": {
            "runtime_contract_sha256": proof_write.get("sha256"),
            "scorer_version": SWEEP_SCHEMA,
        },
        "cache_identity": {
            "cache_sha256": candidate_archive.get("sha256")
            or source_archive.get("sha256"),
            "source_archive_sha256": source_archive.get("sha256"),
        },
        "archive_label": label,
        "target_kind": result.get("target_kind"),
        "materializer_id": result.get("materializer_id"),
        "receiver_contract_kind": result.get("receiver_contract_kind"),
        "source_archive_path": source_archive.get("path"),
        "source_archive_sha256": source_archive.get("sha256"),
        "source_archive_bytes": source_archive.get("bytes"),
        "candidate_archive_sha256": candidate_archive.get("sha256"),
        "candidate_archive_bytes": candidate_archive.get("bytes"),
        "artifact_bytes": candidate_archive.get("bytes") or source_archive.get("bytes") or 0,
        "saved_bytes": saved_bytes,
        "observed_rate_gain": observed_rate_gain,
        "observed_score_gain": observed_rate_gain,
        "rate_positive": rate_positive,
        "receiver_contract_satisfied": result.get("receiver_contract_satisfied") is True,
        "receiver_verification_blockers": receiver_verification.get("blockers") or [],
        "readiness_blockers": readiness_blockers,
        "runtime_consumption_proof_path": result.get("runtime_consumption_proof_path"),
        "manifest_path": manifest_path.as_posix(),
        "candidate_archive_path": candidate_archive.get("path"),
        "selected_member_name": result.get("selected_member_name"),
        "selected_member_names": result.get("selected_member_names") or [],
        "selection_scope": result.get("selection_scope"),
        "selected_elision": dict(selected),
        "recommended_planner_action": (
            "keep_rate_positive_candidate_for_inflate_parity_gate"
            if rate_positive
            else "demote_matching_archive_class_for_header_elide"
        ),
        "observation_feedback_is_not_score_authority": True,
        **FALSE_AUTHORITY,
    }


def _planner_feedback(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rate_positive_count = sum(1 for row in rows if row.get("rate_positive") is True)
    blocker_counts: dict[str, int] = {}
    for row in rows:
        for blocker in row.get("readiness_blockers") or []:
            key = str(blocker)
            blocker_counts[key] = blocker_counts.get(key, 0) + 1
    return {
        "schema": "family_agnostic_materializer_planner_feedback.v1",
        "observation_kind": "materializer_rate_observation",
        "target_kind": PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        "rate_positive_count": rate_positive_count,
        "rate_nonpositive_count": len(rows) - rate_positive_count,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "recommended_acquisition_rule": (
            "rank_rate_positive_header_elide_after_inflate_parity"
            if rate_positive_count
            else "demote_packet_member_zip_header_elide_for_matching_archive_class"
        ),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _parse_archive_spec(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise FamilyAgnosticMaterializerError(
            "--archive label=path specs require both label and path"
        )
    return label, Path(path)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve(strict=False)


def _row_slug(label: str, *, index: int, archive_sha: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("._-")
    if not cleaned:
        cleaned = "archive"
    return f"{index:03d}_{cleaned[:72]}_{archive_sha[:12]}"


__all__ = [
    "OBSERVATION_SCHEMA",
    "SWEEP_SCHEMA",
    "build_materializer_empirical_sweep",
]


if __name__ == "__main__":
    raise SystemExit(main())
