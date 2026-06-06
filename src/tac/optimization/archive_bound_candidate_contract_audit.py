# SPDX-License-Identifier: MIT
"""Audit archive-like candidate payloads for shared contract custody.

This scanner is intentionally contract-first.  It treats an invalid or stale
``tac_archive_bound_candidate_contract.v1`` surface as a hard blocker, and
it treats archive-like candidate rows without the shared contract as migration
work that acquisition/briefing can route instead of losing as side reports.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    ArchiveBoundCandidateContractError,
    archive_bound_candidate_contract_stale_field_blockers,
    archive_bound_candidate_contracts_from_payload,
    has_archive_bound_candidate_contract_payload,
)

ARCHIVE_BOUND_CONTRACT_AUDIT_SCHEMA = "tac_archive_bound_candidate_contract_audit.v1"
ARCHIVE_BOUND_CONTRACT_MIGRATION_BACKLOG_QUEUE_SCHEMA = (
    "tac_archive_bound_candidate_contract_migration_backlog_queue.v1"
)
ARCHIVE_BOUND_CONTRACT_MIGRATION_BACKLOG_ROW_SCHEMA = (
    "tac_archive_bound_candidate_contract_migration_backlog_row.v1"
)

_CONTRACT_SCHEMAS = frozenset(
    {
        ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    }
)

_ARCHIVE_EVIDENCE_KEYS = frozenset(
    {
        "archive_bound_candidate_ready",
        "archive_bound_candidate_ready_for_exact_handoff",
        "archive_bytes",
        "archive_file_custody",
        "archive_path",
        "archive_sha256",
        "archive_zip_bytes",
        "archive_zip_path",
        "archive_zip_sha256",
        "byte_closed_archive",
        "byte_closed_candidate_emitted",
        "candidate_archive",
        "candidate_archive_bytes",
        "candidate_archive_path",
        "candidate_archive_sha256",
        "charged_bits_changed",
        "exact_ready_queue_path",
        "inflate_sh_path",
        "receiver_contract_satisfied",
        "runtime_adapter_manifest",
        "runtime_adapter_ready",
        "runtime_consumption_proof_path",
        "runtime_consumption_proof_status",
        "runtime_tree_sha256",
        "score_affecting_payload_changed",
        "source_archive",
        "source_archive_bytes",
        "source_archive_path",
        "source_archive_sha256",
    }
)

_STRONG_ARCHIVE_CUSTODY_KEYS = frozenset(
    {
        "archive_file_custody",
        "archive_path",
        "archive_sha256",
        "archive_zip_path",
        "archive_zip_sha256",
        "byte_closed_archive",
        "byte_closed_candidate_emitted",
        "byte_closed_candidate_materialized",
        "candidate_archive",
        "candidate_archive_materialized",
        "candidate_archive_path",
        "candidate_archive_sha256",
        "exact_ready_queue_path",
        "runtime_consumption_proof_path",
    }
)

_CANDIDATE_INTENT_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_family",
        "family_id",
        "materializer_backlog_row_count",
        "materializer_work_queue_executable_row_count",
        "mlx_triage_argv",
        "public_pr",
        "replay_argv",
        "selected_archive_transform_variant",
        "target_kind",
        "transform_kind",
    }
)

_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ArchiveBoundContractAuditFinding:
    """One contract hygiene finding at a JSON pointer or markdown block."""

    path: str
    pointer: str
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ArchiveBoundContractAuditResult:
    """Structured result consumed by operator briefing and preflight."""

    paths_scanned: int
    json_payload_count: int
    contract_surface_count: int
    valid_contract_surface_count: int
    blocking_findings: tuple[ArchiveBoundContractAuditFinding, ...]
    migration_required_findings: tuple[ArchiveBoundContractAuditFinding, ...]
    advisory_findings: tuple[ArchiveBoundContractAuditFinding, ...]
    skipped_paths: tuple[str, ...] = ()

    @property
    def finding_count(self) -> int:
        return (
            len(self.blocking_findings)
            + len(self.migration_required_findings)
            + len(self.advisory_findings)
        )

    @property
    def passed(self) -> bool:
        return not self.blocking_findings

    def as_dict(self) -> dict[str, Any]:
        migration_backlog_groups = archive_bound_migration_backlog_groups(
            self.migration_required_findings
        )
        migration_backlog_queue = archive_bound_migration_backlog_queue_from_groups(
            migration_backlog_groups
        )
        return {
            "schema": ARCHIVE_BOUND_CONTRACT_AUDIT_SCHEMA,
            "passed": self.passed,
            "paths_scanned": self.paths_scanned,
            "json_payload_count": self.json_payload_count,
            "contract_surface_count": self.contract_surface_count,
            "valid_contract_surface_count": self.valid_contract_surface_count,
            "blocking_finding_count": len(self.blocking_findings),
            "migration_required_finding_count": len(
                self.migration_required_findings
            ),
            "advisory_finding_count": len(self.advisory_findings),
            "finding_count": self.finding_count,
            "skipped_path_count": len(self.skipped_paths),
            "skipped_paths": list(self.skipped_paths),
            "blocking_findings": [
                asdict(finding) for finding in self.blocking_findings
            ],
            "migration_required_findings": [
                asdict(finding) for finding in self.migration_required_findings
            ],
            "migration_backlog_group_count": len(migration_backlog_groups),
            "migration_backlog_groups": migration_backlog_groups,
            "migration_backlog_queue": migration_backlog_queue,
            "advisory_findings": [
                asdict(finding) for finding in self.advisory_findings
            ],
        }


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(
            repo_root.resolve(strict=False)
        ).as_posix()
    except ValueError:
        return path.as_posix()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _walk_json_files(
    roots: Iterable[Path],
    *,
    include_markdown: bool,
) -> list[Path]:
    paths: list[Path] = []
    suffixes = {".json"}
    if include_markdown:
        suffixes.update({".md", ".markdown"})
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in suffixes:
                paths.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                paths.append(path)
    return sorted(dict.fromkeys(paths), key=lambda item: item.as_posix())


def _tracked_paths_under_roots(repo_root: Path, roots: Sequence[Path]) -> set[str]:
    pathspecs: list[str] = []
    for root in roots:
        try:
            pathspecs.append(
                root.resolve(strict=False)
                .relative_to(repo_root.resolve(strict=False))
                .as_posix()
            )
        except ValueError:
            continue
    if not pathspecs:
        return set()
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", *pathspecs],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return set()
    return {
        item.decode("utf-8")
        for item in proc.stdout.split(b"\0")
        if item
    }


def _has_contract_surface(payload: Mapping[str, Any]) -> bool:
    return (
        has_archive_bound_candidate_contract_payload(payload)
        or payload.get("schema") in _CONTRACT_SCHEMAS
    )


def _archive_like_candidate_payload(
    payload: Mapping[str, Any],
    *,
    path_label: str,
) -> bool:
    keys = set(payload)
    if not keys.intersection(_ARCHIVE_EVIDENCE_KEYS):
        return False
    if not keys.intersection(_STRONG_ARCHIVE_CUSTODY_KEYS):
        return False
    if keys.intersection(_CANDIDATE_INTENT_KEYS):
        return True
    return "candidate_archive" in payload or "source_archive" in payload


def _candidate_archive_digest(payload: Mapping[str, Any]) -> str:
    archive = payload.get("candidate_archive")
    if isinstance(archive, Mapping):
        archive_path = archive.get("path") or archive.get("archive_path")
        archive_sha = archive.get("sha256") or archive.get("archive_sha256")
        archive_bytes = archive.get("bytes") or archive.get("archive_bytes")
        return (
            f"path={archive_path!r} sha256={archive_sha!r} bytes={archive_bytes!r} "
            f"runtime={payload.get('runtime_consumption_proof_status')!r}"
        )
    return (
        f"path={payload.get('candidate_archive_path') or payload.get('archive_path')!r} "
        f"sha256={payload.get('candidate_archive_sha256') or payload.get('archive_sha256')!r} "
        f"bytes={payload.get('candidate_archive_bytes') or payload.get('archive_bytes')!r} "
        f"runtime={payload.get('runtime_consumption_proof_status')!r}"
    )


def _infer_migration_family(text: str) -> str:
    lowered = text.lower()
    ordered = (
        ("pr95", "pr95"),
        ("pr103", "pr103"),
        ("dqs1", "dqs1"),
        ("byte_shaving", "byte_shaving"),
        ("public_frontier", "public_frontier"),
        ("public_pr", "public_frontier"),
        ("range", "range_coder"),
        ("arithmetic", "range_coder"),
        ("ans", "ans_coder"),
        ("huffman", "huffman"),
        ("fec", "fec"),
        ("selector", "selector"),
        ("header", "header"),
        ("zip", "zip_ordering"),
        ("repack", "zip_ordering"),
    )
    for needle, family in ordered:
        if needle in lowered:
            return family
    return "archive_candidate"


def _infer_migration_stage(text: str) -> str:
    lowered = text.lower()
    if "exact" in lowered or "handoff" in lowered or "preclaim" in lowered:
        return "exact_handoff"
    if "receiver" in lowered or "runtime" in lowered or "inflate" in lowered:
        return "receiver_proof"
    if "mlx" in lowered or "triage" in lowered:
        return "mlx_triage"
    if "materializer" in lowered or "candidate_archive" in lowered or "archive" in lowered:
        return "byte_closed_materializer"
    return "contract_migration"


def _infer_migration_scope(text: str) -> str:
    lowered = text.lower()
    if "archive" in lowered:
        return "archive"
    ordered = (
        ("full_video", "full_video"),
        ("full-video", "full_video"),
        ("batch", "batch"),
        ("pair", "pair"),
        ("frame", "frame"),
        ("region", "region"),
        ("boundary", "boundary"),
        ("byte", "byte"),
        ("bit", "bit"),
    )
    for needle, scope in ordered:
        if needle in lowered:
            return scope
    return "archive"


def _infer_migration_entropy_position(family: str, text: str) -> str:
    lowered = f"{family} {text}".lower()
    if family in {"dqs1", "fec", "selector", "public_frontier"}:
        return "before_entropy_coder"
    if family in {"range_coder", "ans_coder", "huffman"}:
        return "at_entropy_coder"
    if family in {"header", "zip_ordering"}:
        return "after_entropy_coder"
    if "before" in lowered:
        return "before_entropy_coder"
    if "after" in lowered:
        return "after_entropy_coder"
    if "entropy" in lowered:
        return "at_entropy_coder"
    return "archive_entropy_position_unknown"


def archive_bound_migration_backlog_groups(
    findings: Sequence[ArchiveBoundContractAuditFinding],
) -> list[dict[str, Any]]:
    """Group missing-contract findings into executable migration backlog slices."""

    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for finding in findings:
        text = " ".join(
            [
                finding.path,
                finding.pointer,
                finding.code,
                finding.message,
            ]
        )
        family = _infer_migration_family(text)
        stage = _infer_migration_stage(text)
        scope = _infer_migration_scope(text)
        entropy_position = _infer_migration_entropy_position(family, text)
        key = (family, stage, scope, entropy_position)
        group = groups.setdefault(
            key,
            {
                "schema": "tac_archive_bound_candidate_contract_migration_backlog_group.v1",
                "group_key": "|".join(key),
                "family": family,
                "stage": stage,
                "scope": scope,
                "entropy_position_label": entropy_position,
                "finding_count": 0,
                "paths": [],
                "sample_pointers": [],
                "task_kind": "smallest_byte_closed_materializer_contract_migration",
                "smallest_executable_task": (
                    "emit tac_archive_bound_candidate_contract.v1 for the "
                    f"{family}/{stage}/{scope} row, then rerun receiver proof "
                    "or record an exact blocker"
                ),
                "contest_space_grounding_requirements": [
                    "byte_closed_archive_custody",
                    "contest_inflate_runtime_consumption",
                    "upstream_video_content_tree_or_runtime_tree_custody",
                    "segnet_posenet_rate_component_axis_label",
                    "exact_cpu_or_cuda_replay_or_precise_blocker",
                    "posterior_ledger_update_for_positive_or_negative_result",
                ],
                "acquisition_spend_preconditions": [
                    "shared_contract_valid",
                    "receiver_proof_gate_passed",
                    "exact_axis_preclaim_or_blocker_recorded",
                    "posterior_budget_route_updated",
                ],
                "allowed_use": "archive_bound_contract_migration_backlog_routing",
                "forbidden_use": "score_claim_or_dispatch_authority",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )
        group["finding_count"] += 1
        if finding.path not in group["paths"]:
            group["paths"].append(finding.path)
        if len(group["sample_pointers"]) < 8:
            group["sample_pointers"].append(finding.pointer)
    return sorted(
        groups.values(),
        key=lambda row: (-int(row["finding_count"]), row["group_key"]),
    )


def archive_bound_migration_backlog_queue_from_groups(
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile migration groups into false-authority executable backlog rows.

    The queue is deliberately not a dispatch queue.  Its rows are the smallest
    byte-closed contract-migration tasks acquisition is allowed to route before
    spending budget on a family/stage/scope again.
    """

    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group_key = str(group.get("group_key") or "")
        family = str(group.get("family") or "archive_candidate")
        stage = str(group.get("stage") or "contract_migration")
        scope = str(group.get("scope") or "archive")
        entropy_position = str(
            group.get("entropy_position_label")
            or "archive_entropy_position_unknown"
        )
        row_id = (
            "archive_bound_contract_migration__"
            f"{index:04d}__{family}__{stage}__{scope}__{entropy_position}"
        )
        rows.append(
            {
                "schema": ARCHIVE_BOUND_CONTRACT_MIGRATION_BACKLOG_ROW_SCHEMA,
                "row_id": row_id,
                "group_key": group_key,
                "family": family,
                "stage": stage,
                "scope": scope,
                "entropy_position_label": entropy_position,
                "finding_count": int(group.get("finding_count") or 0),
                "source_paths": list(group.get("paths") or []),
                "source_sample_pointers": list(group.get("sample_pointers") or []),
                "work_selection_kind": "contract_migration_or_blocker_work",
                "smallest_executable_task": group.get("smallest_executable_task"),
                "required_output_contract_schema": (
                    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
                ),
                "required_evidence": list(
                    group.get("contest_space_grounding_requirements") or []
                ),
                "acquisition_spend_preconditions": list(
                    group.get("acquisition_spend_preconditions") or []
                ),
                "contract_required_before_acquisition_spend": True,
                "posterior_ledger_required_before_acquisition_spend": True,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "allowed_use": "migration_backlog_routing_only",
                "forbidden_use": (
                    "score_claim_or_budget_spend_or_promotion_or_dispatch_authority"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return {
        "schema": ARCHIVE_BOUND_CONTRACT_MIGRATION_BACKLOG_QUEUE_SCHEMA,
        "row_count": len(rows),
        "rows": rows,
        "acquisition_contract": {
            "schema": "archive_bound_contract_migration_acquisition_guard.v1",
            "shared_contract_surface_required": True,
            "posterior_ledger_surface_required": True,
            "migration_rows_may_not_spend_budget": True,
            "migration_rows_may_only_open_materializer_or_blocker_work": True,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def archive_bound_migration_backlog_queue(
    findings: Sequence[ArchiveBoundContractAuditFinding],
) -> dict[str, Any]:
    """Compile missing-contract findings directly into backlog queue rows."""

    return archive_bound_migration_backlog_queue_from_groups(
        archive_bound_migration_backlog_groups(findings)
    )


def _scan_mapping(
    payload: Mapping[str, Any],
    *,
    path_label: str,
    pointer: str,
    ancestor_contract_surface: bool,
) -> tuple[list[ArchiveBoundContractAuditFinding], int, int, int]:
    findings: list[ArchiveBoundContractAuditFinding] = []
    contract_surfaces = 0
    valid_contract_surfaces = 0
    json_payloads = 1

    has_contract = _has_contract_surface(payload)
    if has_contract:
        contract_surfaces += 1
        try:
            archive_bound_candidate_contracts_from_payload(
                payload,
                label=f"{path_label}{pointer}",
            )
        except ArchiveBoundCandidateContractError as exc:
            findings.append(
                ArchiveBoundContractAuditFinding(
                    path=path_label,
                    pointer=pointer,
                    severity="blocking",
                    code="archive_bound_candidate_contract_invalid",
                    message=str(exc),
                )
            )
        else:
            valid_contract_surfaces += 1
            stale = archive_bound_candidate_contract_stale_field_blockers(payload)
            if stale:
                findings.append(
                    ArchiveBoundContractAuditFinding(
                        path=path_label,
                        pointer=pointer,
                        severity="blocking",
                        code="archive_bound_candidate_contract_stale_duplicate_fields",
                        message=", ".join(stale),
                    )
                )

    if (
        not has_contract
        and not ancestor_contract_surface
        and _archive_like_candidate_payload(payload, path_label=path_label)
    ):
        findings.append(
            ArchiveBoundContractAuditFinding(
                path=path_label,
                pointer=pointer,
                severity="migration_required",
                code="archive_like_candidate_payload_missing_shared_contract",
                message=_candidate_archive_digest(payload),
            )
        )

    if has_contract:
        return findings, json_payloads, contract_surfaces, valid_contract_surfaces

    child_ancestor_contract = ancestor_contract_surface or has_contract
    for key, value in payload.items():
        child_pointer = f"{pointer}/{key}" if pointer else f"/{key}"
        child_findings, child_payloads, child_surfaces, child_valid = _scan_value(
            value,
            path_label=path_label,
            pointer=child_pointer,
            ancestor_contract_surface=child_ancestor_contract,
        )
        findings.extend(child_findings)
        json_payloads += child_payloads
        contract_surfaces += child_surfaces
        valid_contract_surfaces += child_valid

    return findings, json_payloads, contract_surfaces, valid_contract_surfaces


def _scan_value(
    value: Any,
    *,
    path_label: str,
    pointer: str,
    ancestor_contract_surface: bool,
) -> tuple[list[ArchiveBoundContractAuditFinding], int, int, int]:
    if isinstance(value, Mapping):
        return _scan_mapping(
            value,
            path_label=path_label,
            pointer=pointer,
            ancestor_contract_surface=ancestor_contract_surface,
        )
    if _is_sequence(value):
        findings: list[ArchiveBoundContractAuditFinding] = []
        json_payloads = 0
        contract_surfaces = 0
        valid_contract_surfaces = 0
        for index, item in enumerate(value):
            child_findings, child_payloads, child_surfaces, child_valid = _scan_value(
                item,
                path_label=path_label,
                pointer=f"{pointer}/{index}" if pointer else f"/{index}",
                ancestor_contract_surface=ancestor_contract_surface,
            )
            findings.extend(child_findings)
            json_payloads += child_payloads
            contract_surfaces += child_surfaces
            valid_contract_surfaces += child_valid
        return findings, json_payloads, contract_surfaces, valid_contract_surfaces
    return [], 0, 0, 0


def _markdown_payloads(text: str) -> list[Any]:
    payloads: list[Any] = []
    for match in _JSON_FENCE_RE.finditer(text):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            payloads.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return payloads


def audit_archive_bound_candidate_contracts(
    roots: Iterable[str | Path],
    *,
    repo_root: str | Path,
    include_markdown: bool = False,
    max_files: int | None = None,
    max_file_bytes: int = 4_000_000,
    tracked_only: bool = False,
) -> ArchiveBoundContractAuditResult:
    """Scan JSON/optional Markdown artifacts for shared contract hygiene."""

    repo = Path(repo_root)
    root_paths = [
        path if isinstance(path, Path) else Path(path)
        for path in roots
    ]
    resolved_roots = [
        path if path.is_absolute() else repo / path
        for path in root_paths
    ]
    if tracked_only:
        tracked = _tracked_paths_under_roots(repo, resolved_roots)
        suffixes = {".json"}
        if include_markdown:
            suffixes.update({".md", ".markdown"})
        candidate_paths = [
            repo / rel_path
            for rel_path in sorted(tracked)
            if Path(rel_path).suffix.lower() in suffixes
        ]
    else:
        candidate_paths = _walk_json_files(
            resolved_roots,
            include_markdown=include_markdown,
        )
    if max_files is not None:
        candidate_paths = candidate_paths[: max(0, max_files)]

    blocking: list[ArchiveBoundContractAuditFinding] = []
    migration_required: list[ArchiveBoundContractAuditFinding] = []
    advisory: list[ArchiveBoundContractAuditFinding] = []
    skipped: list[str] = []
    paths_scanned = 0
    json_payload_count = 0
    contract_surface_count = 0
    valid_contract_surface_count = 0

    for path in candidate_paths:
        path_label = _repo_rel(path, repo)
        try:
            size = path.stat().st_size
        except OSError as exc:
            advisory.append(
                ArchiveBoundContractAuditFinding(
                    path=path_label,
                    pointer="",
                    severity="advisory",
                    code="archive_bound_contract_audit_stat_failed",
                    message=str(exc),
                )
            )
            continue
        if size > max_file_bytes:
            skipped.append(path_label)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped.append(path_label)
            continue
        except OSError as exc:
            advisory.append(
                ArchiveBoundContractAuditFinding(
                    path=path_label,
                    pointer="",
                    severity="advisory",
                    code="archive_bound_contract_audit_read_failed",
                    message=str(exc),
                )
            )
            continue

        payloads: list[tuple[str, Any]] = []
        if path.suffix.lower() == ".json":
            try:
                payloads.append(("", json.loads(text)))
            except json.JSONDecodeError as exc:
                advisory.append(
                    ArchiveBoundContractAuditFinding(
                        path=path_label,
                        pointer="",
                        severity="advisory",
                        code="archive_bound_contract_audit_json_parse_failed",
                        message=str(exc),
                    )
                )
                continue
        else:
            for index, payload in enumerate(_markdown_payloads(text)):
                payloads.append((f"/markdown_json_fence/{index}", payload))
            if not payloads and (
                "archive_bound_candidate_contract" in text
                or "candidate_archive_sha256" in text
                or "candidate_archive_path" in text
            ):
                advisory.append(
                    ArchiveBoundContractAuditFinding(
                        path=path_label,
                        pointer="",
                        severity="advisory",
                        code="archive_contract_signal_in_markdown_prose",
                        message="markdown prose mentions archive contract/custody; no JSON block audited",
                    )
                )
        paths_scanned += 1

        for root_pointer, payload in payloads:
            findings, payloads_seen, surfaces, valid_surfaces = _scan_value(
                payload,
                path_label=path_label,
                pointer=root_pointer,
                ancestor_contract_surface=False,
            )
            json_payload_count += payloads_seen
            contract_surface_count += surfaces
            valid_contract_surface_count += valid_surfaces
            for finding in findings:
                if finding.severity == "blocking":
                    blocking.append(finding)
                elif finding.severity == "migration_required":
                    migration_required.append(finding)
                else:
                    advisory.append(finding)

    return ArchiveBoundContractAuditResult(
        paths_scanned=paths_scanned,
        json_payload_count=json_payload_count,
        contract_surface_count=contract_surface_count,
        valid_contract_surface_count=valid_contract_surface_count,
        blocking_findings=tuple(blocking),
        migration_required_findings=tuple(migration_required),
        advisory_findings=tuple(advisory),
        skipped_paths=tuple(skipped),
    )


def format_archive_bound_candidate_contract_audit(
    result: ArchiveBoundContractAuditResult,
    *,
    limit: int = 12,
) -> str:
    """Human-readable rollup for operator briefing and direct CLI use."""

    lines = [
        f"passed: {result.passed}",
        f"paths_scanned: {result.paths_scanned}",
        f"json_payloads: {result.json_payload_count}",
        (
            "contract_surfaces: "
            f"{result.valid_contract_surface_count}/{result.contract_surface_count} valid"
        ),
        f"blocking_findings: {len(result.blocking_findings)}",
        f"migration_required_findings: {len(result.migration_required_findings)}",
        (
            "migration_backlog_groups: "
            f"{len(archive_bound_migration_backlog_groups(result.migration_required_findings))}"
        ),
        f"advisory_findings: {len(result.advisory_findings)}",
        f"skipped_paths: {len(result.skipped_paths)}",
    ]
    ranked = (
        list(result.blocking_findings)
        + list(result.migration_required_findings)
        + list(result.advisory_findings)
    )
    if ranked:
        lines.append("top findings:")
        for finding in ranked[: max(0, limit)]:
            lines.append(
                "  - "
                f"{finding.severity} {finding.code} "
                f"{finding.path}{finding.pointer}: {finding.message}"
            )
    return "\n".join(lines)
