"""Fail-closed readiness contract for public PR91/HPM1 replay recovery."""

from __future__ import annotations

import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.pr85_bundle import HPM1_MAGIC, Pr85BundleError, parse_pr85_bundle
from tac.pr91_hpm1_codec import (
    DEFAULT_PR91_ARCHIVE,
    DEFAULT_PR91_RUNTIME_SOURCE_DIR,
    EXPECTED_PR91_ARCHIVE_BYTES,
    EXPECTED_PR91_ARCHIVE_SHA256,
    EXPECTED_PR91_HPM1_HPAC_SHA256,
    EXPECTED_PR91_HPM1_MASK_BYTES,
    EXPECTED_PR91_HPM1_MASK_SHA256,
    EXPECTED_PR91_HPM1_TOKENS_SHA256,
    EXPECTED_PR91_MEMBER_X_BYTES,
    EXPECTED_PR91_MEMBER_X_SHA256,
    analyze_pr91_hpm1_runtime_sources,
    repo_rel,
    sha256_bytes,
    sha256_path,
    split_hpm1_mask_segment,
)

SCHEMA_VERSION = 1
KIND = "pr91_hpm1_readiness"


@dataclass(frozen=True)
class Gate:
    """One deterministic readiness gate."""

    status: str
    passed: bool
    required_for_dispatch: bool
    reason: str


def _gate(
    *,
    passed: bool,
    reason: str,
    required_for_dispatch: bool = True,
    failed_status: str = "failed_closed",
) -> dict[str, Any]:
    status = "passed" if passed else failed_status
    return asdict(
        Gate(
            status=status,
            passed=passed,
            required_for_dispatch=required_for_dispatch,
            reason=reason,
        )
    )


def _expected_file_record(path: Path, *, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_sha = sha256_path(path) if exists else ""
    return {
        "path": repo_rel(path),
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "exists": exists,
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "matches_expected": exists
        and actual_bytes == expected_bytes
        and actual_sha == expected_sha256,
    }


def _read_single_x_member(archive: Path) -> tuple[bytes | None, dict[str, Any]]:
    if not archive.is_file():
        return None, {"status": "missing_archive", "members": [], "duplicates": []}
    try:
        with zipfile.ZipFile(archive) as zf:
            names = [info.filename for info in zf.infolist()]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if names != ["x"]:
                return None, {
                    "status": "not_single_x_archive",
                    "members": names,
                    "duplicates": duplicates,
                }
            return zf.read("x"), {"status": "passed", "members": names, "duplicates": duplicates}
    except Exception as exc:
        return None, {"status": "zip_read_failed", "error": f"{type(exc).__name__}: {exc}"}


def audit_pr91_hpm1_readiness(
    *,
    archive: str | Path = DEFAULT_PR91_ARCHIVE,
    runtime_source_dir: str | Path = DEFAULT_PR91_RUNTIME_SOURCE_DIR,
) -> dict[str, Any]:
    """Return a deterministic non-dispatchable readiness report for PR91/HPM1.

    This audit only validates byte custody and records the known remaining
    gates. It does not decode the HPM1 stream, load scorers, or dispatch remote
    work.
    """

    archive_path = Path(archive)
    runtime_dir = Path(runtime_source_dir)
    gates: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    archive_record = _expected_file_record(
        archive_path,
        expected_bytes=EXPECTED_PR91_ARCHIVE_BYTES,
        expected_sha256=EXPECTED_PR91_ARCHIVE_SHA256,
    )
    gates["static_archive_custody"] = _gate(
        passed=bool(archive_record["matches_expected"]),
        reason="archive bytes and sha256 match public PR91 custody"
        if archive_record["matches_expected"]
        else "archive missing or does not match public PR91 custody",
        failed_status="missing" if not archive_record["exists"] else "failed_closed",
    )

    member_x, zip_report = _read_single_x_member(archive_path)
    member_x_record = {
        "name": "x",
        "expected_bytes": EXPECTED_PR91_MEMBER_X_BYTES,
        "expected_sha256": EXPECTED_PR91_MEMBER_X_SHA256,
        "exists": member_x is not None,
        "bytes": len(member_x) if member_x is not None else None,
        "sha256": sha256_bytes(member_x) if member_x is not None else "",
        "matches_expected": member_x is not None
        and len(member_x) == EXPECTED_PR91_MEMBER_X_BYTES
        and sha256_bytes(member_x) == EXPECTED_PR91_MEMBER_X_SHA256,
        "zip_report": zip_report,
    }
    gates["member_x_custody"] = _gate(
        passed=bool(member_x_record["matches_expected"]),
        reason="single ZIP member x matches public PR91 custody"
        if member_x_record["matches_expected"]
        else "archive must contain exactly one byte-matching member named x",
        failed_status="missing" if member_x is None else "failed_closed",
    )

    mask_segment: bytes | None = None
    bundle_error = ""
    if member_x is not None:
        try:
            bundle = parse_pr85_bundle(member_x)
            mask_segment = bytes(bundle.segments["mask"])
        except (KeyError, Pr85BundleError, ValueError) as exc:
            bundle_error = f"{type(exc).__name__}: {exc}"

    hpm1_record: dict[str, Any] = {
        "expected_bytes": EXPECTED_PR91_HPM1_MASK_BYTES,
        "expected_sha256": EXPECTED_PR91_HPM1_MASK_SHA256,
        "exists": mask_segment is not None,
        "bytes": len(mask_segment) if mask_segment is not None else None,
        "sha256": sha256_bytes(mask_segment) if mask_segment is not None else "",
        "matches_expected": mask_segment is not None
        and len(mask_segment) == EXPECTED_PR91_HPM1_MASK_BYTES
        and sha256_bytes(mask_segment) == EXPECTED_PR91_HPM1_MASK_SHA256,
        "bundle_error": bundle_error,
    }
    if mask_segment is not None and not mask_segment.startswith(HPM1_MAGIC):
        hpm1_record["magic"] = mask_segment[:4].hex()
        hpm1_record["magic_matches_hpm1"] = False
    elif mask_segment is not None:
        hpm1_record["magic"] = HPM1_MAGIC.hex()
        hpm1_record["magic_matches_hpm1"] = True

    gates["hpm1_segment_custody"] = _gate(
        passed=bool(hpm1_record["matches_expected"]) and hpm1_record.get("magic_matches_hpm1") is True,
        reason="mask segment is byte-matching HPM1 payload"
        if hpm1_record["matches_expected"]
        else "HPM1 mask segment must match public PR91 custody",
        failed_status="missing" if mask_segment is None else "failed_closed",
    )

    hpm1_payload_record: dict[str, Any] = {
        "tokens_expected_sha256": EXPECTED_PR91_HPM1_TOKENS_SHA256,
        "hpac_expected_sha256": EXPECTED_PR91_HPM1_HPAC_SHA256,
        "tokens_sha256": "",
        "hpac_sha256": "",
        "config": {},
        "parse_error": "",
    }
    if mask_segment is not None and mask_segment.startswith(HPM1_MAGIC):
        try:
            payload = split_hpm1_mask_segment(mask_segment)
            hpm1_payload_record.update(
                {
                    "tokens_sha256": sha256_bytes(payload.tokens),
                    "hpac_sha256": sha256_bytes(payload.hpac),
                    "tokens_match_expected": sha256_bytes(payload.tokens)
                    == EXPECTED_PR91_HPM1_TOKENS_SHA256,
                    "hpac_matches_expected": sha256_bytes(payload.hpac)
                    == EXPECTED_PR91_HPM1_HPAC_SHA256,
                    "config": payload.config(),
                }
            )
        except (KeyError, Pr85BundleError, ValueError) as exc:
            hpm1_payload_record["parse_error"] = f"{type(exc).__name__}: {exc}"

    gates["hpm1_token_hpac_custody"] = _gate(
        passed=(
            hpm1_payload_record.get("tokens_match_expected") is True
            and hpm1_payload_record.get("hpac_matches_expected") is True
        ),
        reason="embedded HPM1 token stream and HPAC model match expected public custody"
        if hpm1_payload_record.get("tokens_match_expected") is True
        and hpm1_payload_record.get("hpac_matches_expected") is True
        else "embedded HPM1 token stream and HPAC model must match expected public custody",
        failed_status="missing" if not hpm1_payload_record["tokens_sha256"] else "failed_closed",
    )

    runtime_inventory = analyze_pr91_hpm1_runtime_sources(source_dir=runtime_dir)
    gates["runtime_source_inventory"] = _gate(
        passed=runtime_inventory.get("status") == "passed_static_source_inventory",
        reason="public PR91 runtime source inventory is present"
        if runtime_inventory.get("status") == "passed_static_source_inventory"
        else "public PR91 runtime source inventory missing",
        required_for_dispatch=False,
        failed_status="missing",
    )

    gates["full_hpm1_decode_600_frames"] = _gate(
        passed=False,
        reason="full 600-frame HPM1 probability/range decode is not recovered",
        failed_status="blocked",
    )
    gates["byte_exact_hpm1_reencode"] = _gate(
        passed=False,
        reason="byte-exact HPM1 re-encode is not recovered",
        failed_status="blocked",
    )
    gates["runtime_hpm1_loader_without_sidecars"] = _gate(
        passed=False,
        reason="contest inflate runtime has not proven HPM1 loading without uncharged sidecars or fallback",
        failed_status="blocked",
    )
    gates["exact_cuda_auth_eval_after_parity"] = _gate(
        passed=False,
        reason="exact CUDA auth eval is required after byte parity before any score claim",
        failed_status="blocked",
    )

    blockers = [
        name
        for name, gate in sorted(gates.items())
        if gate["required_for_dispatch"] and not gate["passed"]
    ]
    if member_x is not None and zip_report.get("duplicates"):
        warnings.append("duplicate ZIP members detected")

    ready = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": ready,
        "promotion_eligible": False,
        "evidence_grade": "archive_readiness_audit" if ready else "static_custody_plus_blocked_replay",
        "source_archive": archive_record,
        "member_x": member_x_record,
        "hpm1_mask_segment": hpm1_record,
        "hpm1_payload": hpm1_payload_record,
        "runtime_source_inventory": runtime_inventory,
        "gates": gates,
        "dispatch_blockers": blockers,
        "warnings": warnings,
        "next_safe_actions": [
            "Recover full 600-frame HPM1 probability/range decode against the byte-matching PR91 payload.",
            "Prove byte-exact HPM1 decode/re-encode parity before mutating or stacking the stream.",
            "Wire the contest inflate runtime to consume HPM1 without uncharged sidecars or STBM/QMA9 fallback.",
            "Only then run exact CUDA auth eval through archive.zip -> inflate.sh -> upstream/evaluate.py.",
        ],
    }


__all__ = [
    "KIND",
    "SCHEMA_VERSION",
    "audit_pr91_hpm1_readiness",
]
