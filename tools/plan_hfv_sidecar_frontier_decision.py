#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an HFV sidecar frontier decision packet.

This tool compares the currently parity-proven HFV7/HFV8/HFV9 sidecar
artifacts and emits an operator-routable exact-eval recommendation. It does not
score anything and does not dispatch Modal; it verifies local custody, parity,
ZIP anatomy, and paired-dispatch plan readiness before ranking candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import claim_lane_dispatch
except ModuleNotFoundError:  # pragma: no cover - imported as tools.* in tests
    import tools.claim_lane_dispatch as claim_lane_dispatch


HFV_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "hfv7_exp_golomb",
        "manifest_path": "experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/hfv7_exp_golomb_manifest.json",
        "parity_path": "experiments/results/hfv7_exp_golomb_shell_inflate_parity_source_runtime_20260521T194843Z/shell_inflate_parity.json",
        "dispatch_plan_path": "experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/paired_dispatch_plan.json",
        "payload_bytes_key": "output_hfv7_payload_bytes",
        "payload_sha_key": "output_hfv7_payload_sha256",
        "compliance_profile": "rate_minimal_profile_row_runtime_profile",
        "policy_rank": 3,
        "policy_blockers": [
            "active_row_profile_not_archive_contained",
            "implicit_12_byte_trailer_format",
        ],
    },
    {
        "candidate_id": "hfv8_explicit_row",
        "manifest_path": "experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/hfv8_explicit_row_manifest.json",
        "parity_path": "experiments/results/hfv8_explicit_row_shell_inflate_parity_source_runtime_20260521T195718Z/shell_inflate_parity.json",
        "dispatch_plan_path": "experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/paired_dispatch_plan.json",
        "payload_bytes_key": "output_hfv8_payload_bytes",
        "payload_sha_key": "output_hfv8_payload_sha256",
        "compliance_profile": "row_archive_contained_length_discriminated",
        "policy_rank": 2,
        "policy_blockers": [
            "length_only_32_byte_format_discriminator",
        ],
    },
    {
        "candidate_id": "hfv9_magic_explicit_row",
        "manifest_path": "experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/hfv9_magic_explicit_row_manifest.json",
        "parity_path": "experiments/results/hfv9_magic_explicit_row_shell_inflate_parity_source_runtime_20260521T200930Z/shell_inflate_parity.json",
        "dispatch_plan_path": "experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/paired_dispatch_plan.json",
        "payload_bytes_key": "output_hfv9_payload_bytes",
        "payload_sha_key": "output_hfv9_payload_sha256",
        "compliance_profile": "row_archive_contained_magic_identified",
        "policy_rank": 1,
        "policy_blockers": [],
    },
)


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    compliance_profile: str
    archive: str
    archive_bytes: int
    archive_sha256: str
    payload_bytes: int
    payload_sha256: str
    bytes_delta_vs_baseline_archive: int
    rate_delta_vs_baseline_archive: float
    output_submission_dir: str
    output_inflate_py_sha256: str
    manifest_path: str
    manifest_sha256: str
    parity_path: str
    parity_sha256: str
    dispatch_plan_path: str
    dispatch_plan_sha256: str
    zip_single_stored_member_x: bool
    zip_member_bytes: int
    zip_extra_fields_absent: bool
    shell_parity_cmp_equal: bool
    shell_parity_output_sha256_match: bool
    shell_parity_raw_sha256: str
    paired_dispatch_plan_ready: bool
    paired_dispatch_pair_group_id: str
    paired_dispatch_required_axes: list[str]
    paired_dispatch_lanes: dict[str, str]
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool
    policy_rank: int
    policy_blockers: list[str]
    audit_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _zip_audit(path: Path) -> tuple[bool, int, bool, list[str]]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1:
            errors.append(f"expected exactly one ZIP member, got {len(infos)}")
            return False, 0, False, errors
        info = infos[0]
        if info.filename != "x":
            errors.append(f"expected ZIP member 'x', got {info.filename!r}")
        if info.compress_type != zipfile.ZIP_STORED:
            errors.append(f"expected ZIP_STORED member, got method {info.compress_type}")
        extra_absent = not bool(info.extra)
        if not extra_absent:
            errors.append(f"expected no ZIP extra fields, got {len(info.extra)} bytes")
        return (
            info.filename == "x" and info.compress_type == zipfile.ZIP_STORED,
            int(info.file_size),
            extra_absent,
            errors,
        )


def _audit_candidate(spec: dict[str, Any]) -> CandidateAudit:
    errors: list[str] = []
    manifest_path = Path(spec["manifest_path"])
    parity_path = Path(spec["parity_path"])
    dispatch_plan_path = Path(spec["dispatch_plan_path"])
    for label, path in (
        ("manifest", manifest_path),
        ("parity", parity_path),
        ("dispatch_plan", dispatch_plan_path),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")

    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    parity = _load_json(parity_path) if parity_path.is_file() else {}
    dispatch_plan = _load_json(dispatch_plan_path) if dispatch_plan_path.is_file() else {}

    archive_path = Path(str(manifest.get("output_archive", "")))
    archive_sha256 = str(manifest.get("output_archive_sha256", ""))
    if not archive_path.is_file():
        errors.append(f"missing archive: {archive_path}")
    elif _sha256_file(archive_path) != archive_sha256:
        errors.append("archive sha256 mismatch against manifest")

    zip_single_stored, zip_member_bytes, zip_extra_absent, zip_errors = (
        _zip_audit(archive_path) if archive_path.is_file() else (False, 0, False, ["zip missing"])
    )
    errors.extend(zip_errors)

    parity_right_sha = str(parity.get("right", {}).get("archive_sha256", ""))
    if parity_right_sha != archive_sha256:
        errors.append("shell parity right archive sha does not match manifest archive sha")
    if not bool(parity.get("cmp_equal")):
        errors.append("shell parity cmp_equal is false")
    if not bool(parity.get("output_sha256_match")):
        errors.append("shell parity output_sha256_match is false")

    plan_archive_sha = str(dispatch_plan.get("archive", {}).get("sha256", ""))
    plan_expected_sha = str(dispatch_plan.get("archive", {}).get("expected_sha256", ""))
    if plan_archive_sha != archive_sha256 or plan_expected_sha != archive_sha256:
        errors.append("paired dispatch plan archive sha does not match manifest archive sha")
    required_axes = list(dispatch_plan.get("required_axes") or [])
    if sorted(required_axes) != ["contest_cpu", "contest_cuda"]:
        errors.append(f"paired dispatch plan required axes mismatch: {required_axes!r}")
    commands = dispatch_plan.get("commands") or {}
    if not commands.get("contest_cpu") or not commands.get("contest_cuda"):
        errors.append("paired dispatch plan is missing CPU or CUDA command")
    lanes = dispatch_plan.get("lanes") or {}
    if not isinstance(lanes, dict):
        lanes = {}

    return CandidateAudit(
        candidate_id=str(spec["candidate_id"]),
        compliance_profile=str(spec["compliance_profile"]),
        archive=_repo_rel(archive_path),
        archive_bytes=int(manifest.get("output_archive_bytes", 0)),
        archive_sha256=archive_sha256,
        payload_bytes=int(manifest.get(str(spec["payload_bytes_key"]), 0)),
        payload_sha256=str(manifest.get(str(spec["payload_sha_key"]), "")),
        bytes_delta_vs_baseline_archive=int(manifest.get("bytes_delta_vs_baseline_archive", 0)),
        rate_delta_vs_baseline_archive=float(manifest.get("rate_delta_vs_baseline_archive", 0.0)),
        output_submission_dir=str(manifest.get("output_submission_dir", "")),
        output_inflate_py_sha256=str(manifest.get("output_inflate_py_sha256", "")),
        manifest_path=_repo_rel(manifest_path),
        manifest_sha256=_sha256_file(manifest_path) if manifest_path.is_file() else "",
        parity_path=_repo_rel(parity_path),
        parity_sha256=_sha256_file(parity_path) if parity_path.is_file() else "",
        dispatch_plan_path=_repo_rel(dispatch_plan_path),
        dispatch_plan_sha256=_sha256_file(dispatch_plan_path) if dispatch_plan_path.is_file() else "",
        zip_single_stored_member_x=zip_single_stored,
        zip_member_bytes=zip_member_bytes,
        zip_extra_fields_absent=zip_extra_absent,
        shell_parity_cmp_equal=bool(parity.get("cmp_equal")),
        shell_parity_output_sha256_match=bool(parity.get("output_sha256_match")),
        shell_parity_raw_sha256=str(parity.get("right", {}).get("output_raw_sha256", "")),
        paired_dispatch_plan_ready=not any(error.startswith("paired dispatch") for error in errors),
        paired_dispatch_pair_group_id=str(dispatch_plan.get("pair_group_id", "")),
        paired_dispatch_required_axes=required_axes,
        paired_dispatch_lanes={str(axis): str(lane_id) for axis, lane_id in lanes.items()},
        score_claim=bool(manifest.get("score_claim")),
        promotion_eligible=bool(manifest.get("promotion_eligible")),
        ready_for_exact_eval_dispatch=bool(manifest.get("ready_for_exact_eval_dispatch")),
        policy_rank=int(spec["policy_rank"]),
        policy_blockers=list(spec["policy_blockers"]),
        audit_errors=errors,
    )


def _candidate_ready(candidate: CandidateAudit) -> bool:
    return (
        not candidate.audit_errors
        and candidate.zip_single_stored_member_x
        and candidate.zip_extra_fields_absent
        and candidate.shell_parity_cmp_equal
        and candidate.shell_parity_output_sha256_match
        and candidate.paired_dispatch_plan_ready
        and not candidate.score_claim
        and not candidate.promotion_eligible
    )


def _active_claim_rows(claims_path: Path) -> list[dict[str, Any]]:
    if not claims_path.is_file():
        return []
    claims = claim_lane_dispatch._parse_claims(claims_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for claim in claim_lane_dispatch._latest_claims_by_job(claims).values():
        if claim_lane_dispatch._is_terminal(claim.status):
            continue
        rows.append(asdict(claim))
    return sorted(rows, key=lambda row: (row["lane_id"], row["instance_job_id"]))


def _candidate_lane_ids(candidate: CandidateAudit | None) -> set[str]:
    if candidate is None:
        return set()
    return {lane_id for lane_id in candidate.paired_dispatch_lanes.values() if lane_id}


def build_packet(
    *,
    active_claim_count: int | None,
    active_claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = [_audit_candidate(spec) for spec in HFV_CANDIDATES]
    ready_candidates = [candidate for candidate in candidates if _candidate_ready(candidate)]
    ranked = sorted(
        ready_candidates,
        key=lambda row: (row.policy_rank, row.archive_bytes, row.candidate_id),
    )
    byte_ranked = sorted(
        ready_candidates,
        key=lambda row: (row.archive_bytes, row.policy_rank, row.candidate_id),
    )
    recommended = ranked[0] if ranked else None
    rate_minimal = byte_ranked[0] if byte_ranked else None
    active_claims_known = active_claims is not None
    active_claims = active_claims or []
    active_claim_count = len(active_claims) if active_claims_known else active_claim_count
    recommended_lane_ids = _candidate_lane_ids(recommended)
    same_lane_conflicts = [
        row for row in active_claims if str(row.get("lane_id") or "") in recommended_lane_ids
    ]
    unrelated_active_claims = [
        row for row in active_claims if str(row.get("lane_id") or "") not in recommended_lane_ids
    ]
    dispatch_blockers: list[str] = []
    if same_lane_conflicts:
        lanes = ",".join(sorted({str(row["lane_id"]) for row in same_lane_conflicts}))
        dispatch_blockers.append(f"same_lane_active_dispatch_claim_present:{lanes}")
    elif not active_claims_known and active_claim_count is not None and active_claim_count > 0:
        dispatch_blockers.append(f"active_dispatch_claims_present:{active_claim_count}")
    if not recommended:
        dispatch_blockers.append("no_ready_hfv_candidate")
    if active_claims_known and unrelated_active_claims:
        nonblocking_note = (
            "unrelated active dispatch claims present; no same-lane HFV conflict detected"
        )
    else:
        nonblocking_note = ""

    return {
        "schema": "hfv_sidecar_frontier_decision_packet_v1",
        "generated_at_utc": _utc_iso(),
        "score_claim": False,
        "promotion_eligible": False,
        "exact_eval_executed": False,
        "active_dispatch_claim_count": active_claim_count,
        "active_dispatch_claims_known": active_claims_known,
        "same_lane_dispatch_conflicts": same_lane_conflicts,
        "unrelated_active_dispatch_claims": unrelated_active_claims,
        "dispatch_blocked": bool(dispatch_blockers),
        "dispatch_blockers": dispatch_blockers,
        "nonblocking_dispatch_claim_note": nonblocking_note,
        "recommended_for_paired_exact_eval": recommended.to_dict() if recommended else None,
        "recommended_for_paired_exact_eval_after_claims_clear": recommended.to_dict() if recommended else None,
        "rate_minimal_if_profile_code_allowed": rate_minimal.to_dict() if rate_minimal else None,
        "policy_interpretation": {
            "hfv7_exp_golomb": "minimum archive bytes, but active-row profile lives in runtime policy",
            "hfv8_explicit_row": "active row is archive-contained, but format is selected by 32-byte trailer length",
            "hfv9_magic_explicit_row": "active row and format magic are archive-contained; pays 4 bytes over HFV8",
        },
        "ranked_candidates": [candidate.to_dict() for candidate in ranked],
        "byte_ranked_candidates": [candidate.to_dict() for candidate in byte_ranked],
        "all_candidate_audits": [candidate.to_dict() for candidate in candidates],
        "recommended_next_action": (
            "Dispatch HFV9 first if there is no same-lane active claim conflict; "
            "unrelated live claims are coordination warnings, not HFV dispatch blockers. "
            "Retain HFV7 as the rate-minimal alternative only if runtime-profile "
            "interpretation is explicitly accepted."
        ),
    }


def render_markdown(packet: dict[str, Any]) -> str:
    recommended = packet["recommended_for_paired_exact_eval_after_claims_clear"] or {}
    lines = [
        "# HFV Sidecar Frontier Decision Packet",
        "",
        f"- Generated UTC: {packet['generated_at_utc']}",
        f"- Active dispatch claim count: {packet['active_dispatch_claim_count']}",
        f"- Dispatch blocked: {str(packet['dispatch_blocked']).lower()}",
        f"- Same-lane conflicts: {len(packet['same_lane_dispatch_conflicts'])}",
        f"- Score claim: {str(packet['score_claim']).lower()}",
        f"- Exact eval executed: {str(packet['exact_eval_executed']).lower()}",
        "",
        "## Recommendation",
        "",
        f"- Exact-eval candidate after claims clear: `{recommended.get('candidate_id', 'none')}`",
        f"- Archive bytes: {recommended.get('archive_bytes', 'n/a')}",
        f"- Archive SHA-256: `{recommended.get('archive_sha256', '')}`",
        f"- Compliance profile: `{recommended.get('compliance_profile', '')}`",
        "",
        packet["recommended_next_action"],
        "",
        "## Dispatch coordination",
        "",
        f"- Nonblocking active-claim note: {packet['nonblocking_dispatch_claim_note'] or 'none'}",
        f"- Same-lane conflicts: {len(packet['same_lane_dispatch_conflicts'])}",
        f"- Unrelated active claims: {len(packet['unrelated_active_dispatch_claims'])}",
        "",
        "## Ranked candidates",
        "",
        "| rank | candidate | bytes | +baseline bytes | profile | blockers |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for index, candidate in enumerate(packet["ranked_candidates"], start=1):
        blockers = ", ".join(candidate["policy_blockers"]) or "none"
        lines.append(
            f"| {index} | `{candidate['candidate_id']}` | {candidate['archive_bytes']} | "
            f"{candidate['bytes_delta_vs_baseline_archive']} | `{candidate['compliance_profile']}` | {blockers} |"
        )
    lines.extend(
        [
            "",
            "## Byte ranking",
            "",
            "| rank | candidate | bytes | compliance profile |",
            "| ---: | --- | ---: | --- |",
        ]
    )
    for index, candidate in enumerate(packet["byte_ranked_candidates"], start=1):
        lines.append(
            f"| {index} | `{candidate['candidate_id']}` | {candidate['archive_bytes']} | "
            f"`{candidate['compliance_profile']}` |"
        )
    lines.extend(["", "## Audit errors", ""])
    for candidate in packet["all_candidate_audits"]:
        errors = candidate["audit_errors"]
        lines.append(f"- `{candidate['candidate_id']}`: {', '.join(errors) if errors else 'none'}")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-claim-count", type=int, default=None)
    parser.add_argument(
        "--claims-path",
        type=Path,
        default=Path(".omx/state/active_lane_dispatch_claims.md"),
        help=(
            "Live claim ledger used to distinguish same-lane HFV conflicts from "
            "unrelated active dispatches. Pass an empty/nonexistent path to fall "
            "back to --active-claim-count compatibility behavior."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv_sidecar_frontier_decision_packet_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    active_claims = _active_claim_rows(args.claims_path) if args.claims_path.is_file() else None
    packet = build_packet(
        active_claim_count=args.active_claim_count,
        active_claims=active_claims,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "hfv_sidecar_frontier_decision_packet.json"
    md_path = args.output_dir / "hfv_sidecar_frontier_decision_packet.md"
    json_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(packet), encoding="utf-8")
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
