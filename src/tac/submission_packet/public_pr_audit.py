# SPDX-License-Identifier: MIT
"""Public GitHub PR audit for contest submission bundles.

This module turns the manual PR-review ritual into a typed, repeatable
pre-submit audit. It is observability-only: a clean audit does not claim score
authority or promotion eligibility. Exact contest eval remains the authority.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    archive_bound_candidate_contract_fields_for_row,
)
from tac.submission_packet.linter import LintSeverity, lint_pr_body

PUBLIC_PR_AUDIT_SCHEMA_VERSION = "public_submission_pr_audit_v1_20260527"
PUBLIC_PR_AUDIT_HELPER = "tac.submission_packet.audit_public_submission_pr"
DEFAULT_TARGET_REPO = "commaai/comma_video_compression_challenge"
DEFAULT_RELEASE_ASSET_NAME = "archive.zip"
PUBLIC_PR_AUDIT_ARCHIVE_BOUND_FAMILY = "public_submission_pr_audit"
PUBLIC_PR_AUDIT_ARCHIVE_BOUND_TRANSFORM_KIND = "public_frontier_pr_archive_audit"

FORBIDDEN_PUBLIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"Claude", "forbidden_token_claude"),
    (r"Anthropic", "forbidden_token_anthropic"),
    (r"Co-Authored", "forbidden_token_co_authored"),
    (r"claude\.com", "forbidden_token_claude_dot_com"),
    (r"anthropic\.com", "forbidden_token_anthropic_dot_com"),
    (r"/Users/[A-Za-z0-9._-]+/", "local_absolute_path_users"),
    (r"/private/tmp/", "local_absolute_path_private_tmp"),
    (r"\.omx/", "internal_omx_path"),
    (r"_codex", "internal_codex_suffix"),
    (r"Yousfi-style", "unnecessary_public_name_drop"),
    (r"current top merged", "stale_leaderboard_phrase"),
    (r"Modal A100", "stale_hardware_phrase"),
    (r"reconstruct time", "stale_inflate_wording"),
    (r"per-frame selector", "stale_selector_wording"),
    (r"per-frame mode index", "stale_selector_wording"),
)

BLOB_SUFFIXES: tuple[str, ...] = (
    ".zip",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".npy",
    ".npz",
)

REQUIRED_PR_HEADINGS: tuple[str, ...] = (
    "# submission name:",
    "# upload zipped `archive.zip`",
    "# report.txt",
    "# does your submission require gpu for evaluation",
    "# did you include the compression script?",
    "# is this submission competitive or innovative?",
)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


Runner = Callable[[Sequence[str], Path | None, float | None], CommandResult]


def default_runner(
    args: Sequence[str],
    cwd: Path | None = None,
    timeout_s: float | None = None,
) -> CommandResult:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return CommandResult(
        args=tuple(str(part) for part in args),
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    surface: str
    rule: str
    message: str
    path: str | None = None
    suggestion: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ZipMemberInfo:
    filename: str
    compress_type: int
    file_size: int
    compress_size: int
    crc: int
    date_time: tuple[int, int, int, int, int, int]
    extra_len: int

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ZipAudit:
    path: str
    size_bytes: int
    sha256: str
    members: tuple[ZipMemberInfo, ...]
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "members": [member.as_dict() for member in self.members],
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


@dataclass(frozen=True)
class InflateSmokeProof:
    video_name: str
    expected_output_sha256: str | None
    raw_sha256: str | None = None
    raw_path: str | None = None
    ok: bool = False

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PublicSubmissionAuditConfig:
    target_repo: str = DEFAULT_TARGET_REPO
    pr_number: int = 0
    submission_name: str | None = None
    work_dir: Path | None = None
    keep_work_dir: bool = False
    expected_archive_sha256: str | None = None
    expected_archive_bytes: int | None = None
    run_inflate_smoke: bool = False
    expected_output_sha256: str | None = None
    video_name: str = "0.mkv"
    python_bin: Path | None = None
    run_encoder_rebuild: bool = False
    encoder_rebuild_command: tuple[str, ...] = ()
    command_timeout_s: float = 120.0
    network_timeout_s: float = 30.0


@dataclass(frozen=True)
class PublicSubmissionAuditResult:
    schema_version: str
    target_repo: str
    pr_number: int
    pr_url: str | None
    head_sha: str | None
    submission_name: str | None
    archive_url: str | None
    archive_sha256: str | None
    archive_bytes: int | None
    findings: tuple[AuditFinding, ...]
    zip_audit: ZipAudit | None = None
    inflate_smoke: InflateSmokeProof | None = None
    body_patch_suggestions: tuple[str, ...] = ()
    checked_commands: tuple[dict[str, object], ...] = ()
    work_dir: str | None = None
    elapsed_seconds: float = 0.0
    measurement_utc: str = ""
    score_claim: bool = False
    promotable: bool = False
    ready_for_exact_eval_dispatch: bool = False

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "error")

    @property
    def warn_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "warn")

    @property
    def info_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "info")

    @property
    def overall_clean(self) -> bool:
        return self.error_count == 0

    def _archive_bound_contract_fields(self) -> dict[str, object]:
        zip_path = self.zip_audit.path if self.zip_audit is not None else ""
        zip_ok = self.zip_audit.ok if self.zip_audit is not None else False
        smoke_ok = self.inflate_smoke.ok if self.inflate_smoke is not None else False
        blocker_ids = [
            f"{finding.surface}:{finding.rule}"
            for finding in self.findings
            if finding.severity == "error"
        ]
        if self.zip_audit is not None:
            blocker_ids.extend(self.zip_audit.blockers)
        if self.inflate_smoke is None:
            blocker_ids.append("public_pr_audit_inflate_smoke_not_run")
        elif not smoke_ok:
            blocker_ids.append("public_pr_audit_inflate_smoke_failed")
        archive_sha = self.archive_sha256 or (
            self.zip_audit.sha256 if self.zip_audit is not None else None
        )
        candidate_id_suffix = archive_sha[:16] if isinstance(archive_sha, str) else "missing"
        row = {
            "schema": "public_submission_pr_audit_archive_bound_candidate_row.v1",
            "candidate_id": f"public_pr_audit_{self.pr_number}_{candidate_id_suffix}",
            "candidate_family": PUBLIC_PR_AUDIT_ARCHIVE_BOUND_FAMILY,
            "archive_native_transform_kind": PUBLIC_PR_AUDIT_ARCHIVE_BOUND_TRANSFORM_KIND,
            "candidate_archive_path": zip_path,
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": self.archive_bytes,
            "byte_closed_candidate_materialized": zip_ok,
            "candidate_archive_materialized": zip_ok,
            "runtime_adapter_ready": smoke_ok,
            "contest_runtime_decoder_adapter_ready": smoke_ok,
            "runtime_adapter_manifest": {
                "schema": "public_pr_audit_runtime_adapter_manifest.v1",
                "submission_name": self.submission_name,
                "head_sha": self.head_sha,
                "inflate_smoke_ok": smoke_ok,
                "inflate_smoke_raw_sha256": (
                    self.inflate_smoke.raw_sha256 if self.inflate_smoke is not None else None
                ),
                "runtime_adapter_ready": smoke_ok,
                "contest_runtime_decoder_adapter_ready": smoke_ok,
            },
            "runtime_consumption_proof_ready": False,
            "receiver_contract_satisfied": False,
            "receiver_contract_kind": "public_pr_audit_requires_shared_receiver_proof",
            "semantic_payload_changed": False,
            "score_affecting_payload_changed": True,
            "exact_axis_score_affecting_adjudication_required": True,
            "charged_bits_changed": True,
            "blockers": blocker_ids,
            "canonical_anti_pattern_ids": [
                "proxy_or_advisory_signal_masquerades_as_score_authority",
                "public_pr_audit_without_archive_bound_receiver_proof",
            ],
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
        }
        return archive_bound_candidate_contract_fields_for_row(
            row,
            family_id=PUBLIC_PR_AUDIT_ARCHIVE_BOUND_FAMILY,
            candidate_chain_id=str(row["candidate_id"]),
        )

    def as_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "target_repo": self.target_repo,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "head_sha": self.head_sha,
            "submission_name": self.submission_name,
            "archive_url": self.archive_url,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "overall_clean": self.overall_clean,
            "error_count": self.error_count,
            "warn_count": self.warn_count,
            "info_count": self.info_count,
            "findings": [finding.as_dict() for finding in self.findings],
            "zip_audit": self.zip_audit.as_dict() if self.zip_audit else None,
            "inflate_smoke": self.inflate_smoke.as_dict() if self.inflate_smoke else None,
            "body_patch_suggestions": list(self.body_patch_suggestions),
            "checked_commands": list(self.checked_commands),
            "work_dir": self.work_dir,
            "elapsed_seconds": self.elapsed_seconds,
            "measurement_utc": self.measurement_utc,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "canonical_helper_invocation": PUBLIC_PR_AUDIT_HELPER,
            "axis_tag": "[predicted]",
            "evidence_grade": "[predicted; public-submission-pr-audit]",
        }
        payload.update(self._archive_bound_contract_fields())
        return payload


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_loads(text: str, *, command: Sequence[str]) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-json output from {list(command)!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object output from {list(command)!r}")
    return payload


def extract_submission_name(body: str) -> str | None:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower() == "# submission name:":
            for candidate in lines[index + 1 :]:
                stripped = candidate.strip()
                if not stripped or stripped.startswith("<!--"):
                    continue
                return stripped.strip("` ")
    return None


def extract_archive_url(body: str) -> str | None:
    match = re.search(r"\[archive\.zip\]\((https://[^)\s]+)\)", body)
    if match:
        return match.group(1)
    match = re.search(r"https://github\.com/[A-Za-z0-9._/-]+/releases/download/[^\s)]+/archive\.zip", body)
    return match.group(0) if match else None


def extract_archive_sha256(body: str) -> str | None:
    patterns = (
        r"Archive SHA-256:\s*([0-9a-fA-F]{64})",
        r"SHA-256 [`']?([0-9a-fA-F]{64})",
        r"sha256:([0-9a-fA-F]{64})",
    )
    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            return match.group(1).lower()
    return None


def extract_archive_bytes(body: str) -> int | None:
    patterns = (
        r"Archive size bytes:\s*([0-9,]+)",
        r"Submission file size:\s*([0-9,]+)\s*bytes",
        r"Archive bytes\s*\|?\s*`?([0-9,]+)`?",
    )
    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def parse_github_release_asset_url(url: str) -> tuple[str, str, str, str] | None:
    match = re.match(
        r"^https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/?#]+)$",
        url,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4)


def audit_zip(path: Path) -> ZipAudit:
    blockers: list[str] = []
    warnings: list[str] = []
    members: list[ZipMemberInfo] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            for info in infos:
                members.append(
                    ZipMemberInfo(
                        filename=info.filename,
                        compress_type=int(info.compress_type),
                        file_size=int(info.file_size),
                        compress_size=int(info.compress_size),
                        crc=int(info.CRC),
                        date_time=tuple(info.date_time),
                        extra_len=len(info.extra or b""),
                    )
                )
    except zipfile.BadZipFile as exc:
        return ZipAudit(
            path=str(path),
            size_bytes=path.stat().st_size if path.exists() else 0,
            sha256=_sha256_file(path) if path.exists() else "",
            members=(),
            blockers=(f"bad_zip:{exc}",),
        )
    if not members:
        blockers.append("zip_has_no_members")
    if any(member.filename.startswith("__MACOSX/") for member in members):
        blockers.append("zip_contains_macos_resource_fork_dir")
    if any(Path(member.filename).name.startswith(".") for member in members):
        blockers.append("zip_contains_hidden_member")
    if any(member.extra_len for member in members):
        warnings.append("zip_member_extra_fields_present")
    if len(members) != 1:
        warnings.append("zip_not_single_member")
    return ZipAudit(
        path=str(path),
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
        members=tuple(members),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def body_patch_suggestions(body: str, *, head_sha: str | None, submission_name: str | None) -> tuple[str, ...]:
    suggestions: list[str] = []
    if head_sha and submission_name:
        expected_short = head_sha[:7]
        expected_url_fragment = f"/tree/{head_sha}/submissions/{submission_name}"
        runtime_line = next((line for line in body.splitlines() if "Runtime tree at PR head" in line), "")
        if runtime_line and (expected_short not in runtime_line or expected_url_fragment not in runtime_line):
            suggestions.append(
                "Runtime tree line should point at current head "
                f"`{expected_short}` and `/tree/{head_sha}/submissions/{submission_name}`."
            )
        elif not runtime_line:
            suggestions.append(
                "Add a Runtime tree line pointing at the current PR head submission directory."
            )
    if "Yousfi-style" in body:
        suggestions.append("Replace unnecessary public name-drop `Yousfi-style` with technical wording.")
    if "scorer-disagreement (scorer-disagreement" in body:
        suggestions.append("Remove repeated scorer-disagreement wording in the opportunity paragraph.")
    return tuple(suggestions)


def _finding_from_linter(finding: Any) -> AuditFinding:
    severity = "error" if finding.severity == LintSeverity.ERROR.value else "warn"
    return AuditFinding(
        severity=severity,
        surface=f"pr_body:{finding.surface}",
        rule=str(finding.rule),
        message=str(finding.fix_suggestion),
        path=str(finding.file_path),
        suggestion=str(finding.fix_suggestion),
    )


def _add_pattern_findings(findings: list[AuditFinding], *, root: Path, surface: str) -> None:
    text_suffixes = {".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ""}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix in BLOB_SUFFIXES:
            findings.append(
                AuditFinding(
                    severity="error",
                    surface=surface,
                    rule="tracked_or_bundled_binary_blob",
                    message=f"submission bundle contains binary/blob artifact {rel}",
                    path=rel,
                    suggestion="Keep archive/model/generated blobs in release assets or custody dirs, not the PR tree.",
                )
            )
            continue
        if path.suffix not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern, rule in FORBIDDEN_PUBLIC_PATTERNS:
            match = re.search(pattern, text)
            if match:
                findings.append(
                    AuditFinding(
                        severity="error",
                        surface=surface,
                        rule=rule,
                        message=f"{rel} contains public-surface hazard {match.group(0)!r}",
                        path=rel,
                        suggestion="Remove internal/public-tone hazard before submitting.",
                    )
                )


def _run_checked(
    checked: list[dict[str, object]],
    findings: list[AuditFinding],
    runner: Runner,
    args: Sequence[str],
    *,
    cwd: Path,
    surface: str,
    rule: str,
    timeout_s: float,
    env_pythonpath: Path | None = None,
) -> CommandResult:
    old_pythonpath = os.environ.get("PYTHONPATH")
    if env_pythonpath is not None:
        paths = [str(env_pythonpath)]
        if old_pythonpath:
            paths.append(old_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        result = runner(args, cwd, timeout_s)
    finally:
        if env_pythonpath is not None:
            if old_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pythonpath
    checked.append(
        {
            "args": list(args),
            "cwd": str(cwd),
            "returncode": result.returncode,
            "stdout_preview": result.stdout[:500],
            "stderr_preview": result.stderr[:500],
        }
    )
    if not result.ok:
        findings.append(
            AuditFinding(
                severity="error",
                surface=surface,
                rule=rule,
                message=f"command failed rc={result.returncode}: {' '.join(args)}",
                path=str(cwd),
                suggestion=(result.stderr or result.stdout or "Fix failing command.").strip()[:500],
            )
        )
    return result


def _download_archive(url: str, output_path: Path, *, timeout_s: float) -> None:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        output_path.write_bytes(response.read())


def _head_repo_clone_url(pr_payload: dict[str, Any]) -> tuple[str, str, str] | None:
    owner = pr_payload.get("headRepositoryOwner") or {}
    repo = pr_payload.get("headRepository") or {}
    owner_login = owner.get("login") if isinstance(owner, dict) else None
    repo_name = repo.get("name") if isinstance(repo, dict) else None
    branch = pr_payload.get("headRefName")
    if not (isinstance(owner_login, str) and isinstance(repo_name, str) and isinstance(branch, str)):
        return None
    return f"https://github.com/{owner_login}/{repo_name}.git", branch, f"{owner_login}/{repo_name}"


def _fetch_pr_payload(config: PublicSubmissionAuditConfig, runner: Runner) -> dict[str, Any]:
    args = [
        "gh",
        "pr",
        "view",
        str(config.pr_number),
        "--repo",
        config.target_repo,
        "--json",
        "number,url,state,isDraft,mergeable,headRefOid,headRefName,headRepository,headRepositoryOwner,body,comments,files,reviewRequests",
    ]
    result = runner(args, None, config.command_timeout_s)
    if not result.ok:
        raise RuntimeError(f"gh pr view failed: {result.stderr or result.stdout}")
    return _json_loads(result.stdout, command=args)


def _audit_pr_surface(
    payload: dict[str, Any],
    config: PublicSubmissionAuditConfig,
) -> tuple[list[AuditFinding], str, str | None, str | None, int | None, tuple[str, ...]]:
    findings: list[AuditFinding] = []
    body = str(payload.get("body") or "")
    head_sha = payload.get("headRefOid") if isinstance(payload.get("headRefOid"), str) else None
    submission_name = config.submission_name or extract_submission_name(body)
    archive_url = extract_archive_url(body)
    archive_sha = config.expected_archive_sha256 or extract_archive_sha256(body)
    archive_bytes = config.expected_archive_bytes or extract_archive_bytes(body)

    if payload.get("state") != "OPEN":
        findings.append(AuditFinding("error", "pr", "pr_not_open", "PR is not open."))
    if payload.get("isDraft") is True:
        findings.append(AuditFinding("error", "pr", "pr_is_draft", "PR is still draft."))
    if payload.get("mergeable") not in {"MERGEABLE", "UNKNOWN", None}:
        findings.append(AuditFinding("warn", "pr", "pr_not_mergeable", f"mergeable={payload.get('mergeable')}"))
    if not submission_name:
        findings.append(AuditFinding("error", "pr_body", "submission_name_missing", "PR body lacks submission name."))
    if not archive_url:
        findings.append(AuditFinding("error", "pr_body", "archive_url_missing", "PR body lacks archive.zip release URL."))
    elif parse_github_release_asset_url(archive_url) is None:
        findings.append(
            AuditFinding(
                "error",
                "pr_body",
                "archive_url_not_github_release_asset",
                f"archive URL is not a GitHub release asset: {archive_url}",
                suggestion="Use a github.com/.../releases/download/.../archive.zip URL.",
            )
        )
    if not archive_sha:
        findings.append(AuditFinding("warn", "pr_body", "archive_sha_missing", "PR body does not expose archive SHA-256."))
    if not archive_bytes:
        findings.append(AuditFinding("warn", "pr_body", "archive_bytes_missing", "PR body does not expose archive byte size."))

    lowered = body.lower()
    for heading in REQUIRED_PR_HEADINGS:
        if heading not in lowered:
            findings.append(
                AuditFinding(
                    "warn",
                    "pr_body",
                    "template_heading_missing",
                    f"PR body may be missing template heading: {heading}",
                    suggestion="Update PR body to the current contest template.",
                )
            )
    for linter_finding in lint_pr_body(body, target_repo=config.target_repo, file_path="PR_BODY.md"):
        findings.append(_finding_from_linter(linter_finding))
    for comment in payload.get("comments") or []:
        if not isinstance(comment, dict):
            continue
        comment_body = str(comment.get("body") or "")
        for pattern, rule in FORBIDDEN_PUBLIC_PATTERNS:
            match = re.search(pattern, comment_body)
            if match:
                findings.append(
                    AuditFinding(
                        "error",
                        "pr_comment",
                        rule,
                        f"comment contains public-surface hazard {match.group(0)!r}",
                        path=str(comment.get("url") or ""),
                        suggestion="Edit or supersede the comment with neutral technical wording.",
                    )
                )
    for suggestion in body_patch_suggestions(body, head_sha=head_sha, submission_name=submission_name):
        findings.append(
            AuditFinding(
                "warn",
                "pr_body",
                "body_patch_suggestion",
                suggestion,
                path="PR_BODY.md",
                suggestion=suggestion,
            )
        )
    return findings, body, submission_name, archive_url, archive_bytes, body_patch_suggestions(
        body,
        head_sha=head_sha,
        submission_name=submission_name,
    )


def _audit_release_asset(
    config: PublicSubmissionAuditConfig,
    runner: Runner,
    archive_url: str | None,
    archive_path: Path,
    findings: list[AuditFinding],
) -> ZipAudit | None:
    if not archive_url:
        return None
    parsed = parse_github_release_asset_url(archive_url)
    if parsed is not None:
        owner, repo, tag, asset_name = parsed
        args = ["gh", "api", f"repos/{owner}/{repo}/releases/tags/{tag}"]
        result = runner(args, None, config.command_timeout_s)
        if result.ok:
            release = _json_loads(result.stdout, command=args)
            if release.get("draft") is True or release.get("prerelease") is True:
                findings.append(AuditFinding("error", "release", "release_not_final", "Release is draft or prerelease."))
            assets = release.get("assets") if isinstance(release.get("assets"), list) else []
            matching = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == asset_name]
            if not matching:
                findings.append(AuditFinding("error", "release", "release_asset_missing", f"Missing release asset {asset_name}."))
            else:
                asset = matching[0]
                digest = str(asset.get("digest") or "")
                if config.expected_archive_sha256 and digest and digest != f"sha256:{config.expected_archive_sha256}":
                    findings.append(
                        AuditFinding("error", "release", "release_digest_mismatch", f"asset digest {digest}")
                    )
                if config.expected_archive_bytes and int(asset.get("size") or 0) != config.expected_archive_bytes:
                    findings.append(
                        AuditFinding("error", "release", "release_size_mismatch", f"asset size {asset.get('size')}")
                    )
        else:
            findings.append(AuditFinding("warn", "release", "release_api_unavailable", result.stderr or result.stdout))
    try:
        _download_archive(archive_url, archive_path, timeout_s=config.network_timeout_s)
    except Exception as exc:
        findings.append(AuditFinding("error", "archive", "archive_download_failed", str(exc)))
        return None
    zip_audit = audit_zip(archive_path)
    for blocker in zip_audit.blockers:
        findings.append(AuditFinding("error", "archive_zip", blocker, blocker, path=str(archive_path)))
    for warning in zip_audit.warnings:
        findings.append(AuditFinding("warn", "archive_zip", warning, warning, path=str(archive_path)))
    if config.expected_archive_sha256 and zip_audit.sha256 != config.expected_archive_sha256:
        findings.append(
            AuditFinding(
                "error",
                "archive_zip",
                "archive_sha256_mismatch",
                f"downloaded archive sha {zip_audit.sha256} != expected {config.expected_archive_sha256}",
                path=str(archive_path),
            )
        )
    if config.expected_archive_bytes and zip_audit.size_bytes != config.expected_archive_bytes:
        findings.append(
            AuditFinding(
                "error",
                "archive_zip",
                "archive_size_mismatch",
                f"downloaded archive bytes {zip_audit.size_bytes} != expected {config.expected_archive_bytes}",
                path=str(archive_path),
            )
        )
    return zip_audit


def _audit_submission_tree(
    config: PublicSubmissionAuditConfig,
    runner: Runner,
    clone_dir: Path,
    submission_name: str | None,
    archive_path: Path,
    findings: list[AuditFinding],
    checked: list[dict[str, object]],
) -> InflateSmokeProof | None:
    if not submission_name:
        return None
    submission_dir = clone_dir / "submissions" / submission_name
    if not submission_dir.is_dir():
        findings.append(
            AuditFinding(
                "error",
                "submission_tree",
                "submission_dir_missing",
                f"missing submissions/{submission_name}",
                path=str(submission_dir),
            )
        )
        return None
    for rel in ("README.md", "inflate.py", "inflate.sh", "requirements.txt"):
        if not (submission_dir / rel).is_file():
            findings.append(
                AuditFinding(
                    "error",
                    "submission_tree",
                    "required_file_missing",
                    f"missing {rel}",
                    path=f"submissions/{submission_name}/{rel}",
                )
            )
    for rel in ("inflate.sh", "compress.sh"):
        path = submission_dir / rel
        if path.exists():
            mode = path.stat().st_mode
            if not (mode & stat.S_IXUSR):
                findings.append(
                    AuditFinding(
                        "error",
                        "submission_tree",
                        "script_not_executable",
                        f"{rel} is not user-executable",
                        path=str(path),
                    )
                )
    _add_pattern_findings(findings, root=submission_dir, surface="submission_tree")
    tracked = runner(["git", "ls-files", f"submissions/{submission_name}"], clone_dir, config.command_timeout_s)
    if tracked.ok:
        for rel in tracked.stdout.splitlines():
            if "__pycache__" in rel or rel.endswith(".pyc") or rel.endswith(".DS_Store"):
                findings.append(AuditFinding("error", "submission_tree", "tracked_generated_file", rel, path=rel))
            if Path(rel).suffix in BLOB_SUFFIXES:
                findings.append(AuditFinding("error", "submission_tree", "tracked_blob_file", rel, path=rel))
    _run_checked(
        checked,
        findings,
        runner,
        ["python3", "-m", "compileall", "-q", str(submission_dir)],
        cwd=clone_dir,
        surface="submission_tree",
        rule="compileall_failed",
        timeout_s=config.command_timeout_s,
    )
    for rel in ("inflate.sh", "compress.sh"):
        path = submission_dir / rel
        if path.is_file():
            _run_checked(
                checked,
                findings,
                runner,
                ["bash", "-n", str(path)],
                cwd=clone_dir,
                surface="submission_tree",
                rule=f"{rel}_bash_syntax_failed",
                timeout_s=config.command_timeout_s,
            )
    for command in (
        ["bash", str(submission_dir / "compress.sh"), "--help"],
        ["python3", str(submission_dir / "encoder" / "build_pr101_frame_exploit_selector_packet.py"), "--help"],
        ["python3", str(submission_dir / "encoder" / "frame_exploit_segnet_posenet_sweep.py"), "--help"],
    ):
        if Path(command[1]).is_file():
            _run_checked(
                checked,
                findings,
                runner,
                command,
                cwd=clone_dir,
                surface="submission_tree",
                rule="help_command_failed",
                timeout_s=config.command_timeout_s,
                env_pythonpath=submission_dir / "src",
            )
    if config.run_inflate_smoke:
        if config.expected_output_sha256 is None:
            findings.append(
                AuditFinding("error", "inflate_smoke", "expected_output_sha_missing", "inflate smoke requires expected output SHA")
            )
            return InflateSmokeProof(
                video_name=config.video_name,
                expected_output_sha256=None,
                ok=False,
            )
        data_dir = clone_dir / "_audit_archive_data"
        out_dir = clone_dir / "_audit_inflate_out"
        list_path = clone_dir / "_audit_video_names.txt"
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        data_dir.mkdir()
        out_dir.mkdir()
        list_path.write_text(config.video_name + "\n", encoding="utf-8")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
        env_python = config.python_bin or Path("python3")
        old_python_bin = os.environ.get("PACT_PYTHON_BIN")
        os.environ["PACT_PYTHON_BIN"] = str(env_python)
        try:
            inflate_result = _run_checked(
                checked,
                findings,
                runner,
                ["bash", str(submission_dir / "inflate.sh"), str(data_dir), str(out_dir), str(list_path)],
                cwd=clone_dir,
                surface="inflate_smoke",
                rule="inflate_smoke_failed",
                timeout_s=max(config.command_timeout_s, 180.0),
                env_pythonpath=submission_dir / "src",
            )
        finally:
            if old_python_bin is None:
                os.environ.pop("PACT_PYTHON_BIN", None)
            else:
                os.environ["PACT_PYTHON_BIN"] = old_python_bin
        raw_path = out_dir / "0.raw"
        actual: str | None = None
        if raw_path.is_file():
            actual = _sha256_file(raw_path)
            if actual != config.expected_output_sha256:
                findings.append(
                    AuditFinding(
                        "error",
                        "inflate_smoke",
                        "inflate_output_sha_mismatch",
                        f"raw sha {actual} != expected {config.expected_output_sha256}",
                        path=str(raw_path),
                    )
                )
        else:
            findings.append(
                AuditFinding(
                    "error",
                    "inflate_smoke",
                    "inflate_output_missing",
                    "inflate smoke did not produce 0.raw",
                    path=str(raw_path),
                )
            )
        return InflateSmokeProof(
            video_name=config.video_name,
            expected_output_sha256=config.expected_output_sha256,
            raw_sha256=actual,
            raw_path=str(raw_path) if (config.keep_work_dir or config.work_dir is not None) else None,
            ok=bool(inflate_result.ok and actual == config.expected_output_sha256),
        )
    return None


def audit_public_submission_pr(
    config: PublicSubmissionAuditConfig,
    *,
    runner: Runner = default_runner,
) -> PublicSubmissionAuditResult:
    started = _dt.datetime.now(_dt.UTC)
    findings: list[AuditFinding] = []
    checked: list[dict[str, object]] = []
    zip_audit: ZipAudit | None = None
    temp_root_obj: tempfile.TemporaryDirectory[str] | None = None
    if config.work_dir is None:
        temp_root_obj = tempfile.TemporaryDirectory(prefix="public_pr_audit_")
        work_root = Path(temp_root_obj.name)
    else:
        work_root = config.work_dir
        work_root.mkdir(parents=True, exist_ok=True)

    pr_payload: dict[str, Any] = {}
    body = ""
    submission_name = config.submission_name
    archive_url: str | None = None
    archive_bytes = config.expected_archive_bytes
    body_suggestions: tuple[str, ...] = ()
    inflate_smoke: InflateSmokeProof | None = None
    try:
        pr_payload = _fetch_pr_payload(config, runner)
        pr_findings, body, submission_name, archive_url, archive_bytes, body_suggestions = _audit_pr_surface(
            pr_payload,
            config,
        )
        findings.extend(pr_findings)
        archive_path = work_root / DEFAULT_RELEASE_ASSET_NAME
        zip_audit = _audit_release_asset(config, runner, archive_url, archive_path, findings)
        clone_info = _head_repo_clone_url(pr_payload)
        if clone_info is None:
            findings.append(AuditFinding("error", "clone", "head_repo_unavailable", "PR head repo/branch unavailable"))
        else:
            clone_url, branch, _repo_slug = clone_info
            clone_dir = work_root / "head"
            result = runner(
                ["git", "clone", "--depth", "1", "--branch", branch, clone_url, str(clone_dir)],
                None,
                config.command_timeout_s,
            )
            checked.append(
                {
                    "args": ["git", "clone", "--depth", "1", "--branch", branch, clone_url, str(clone_dir)],
                    "cwd": None,
                    "returncode": result.returncode,
                    "stdout_preview": result.stdout[:500],
                    "stderr_preview": result.stderr[:500],
                }
            )
            if not result.ok:
                findings.append(AuditFinding("error", "clone", "git_clone_failed", result.stderr or result.stdout))
            else:
                inflate_smoke = _audit_submission_tree(
                    config,
                    runner,
                    clone_dir,
                    submission_name,
                    archive_path,
                    findings,
                    checked,
                )
        if config.run_encoder_rebuild:
            if not config.encoder_rebuild_command:
                findings.append(
                    AuditFinding("error", "encoder_rebuild", "encoder_rebuild_command_missing", "no rebuild command provided")
                )
            else:
                _run_checked(
                    checked,
                    findings,
                    runner,
                    config.encoder_rebuild_command,
                    cwd=work_root,
                    surface="encoder_rebuild",
                    rule="encoder_rebuild_failed",
                    timeout_s=max(config.command_timeout_s, 300.0),
                )
        elapsed = (_dt.datetime.now(_dt.UTC) - started).total_seconds()
        return PublicSubmissionAuditResult(
            schema_version=PUBLIC_PR_AUDIT_SCHEMA_VERSION,
            target_repo=config.target_repo,
            pr_number=config.pr_number,
            pr_url=pr_payload.get("url") if isinstance(pr_payload.get("url"), str) else None,
            head_sha=pr_payload.get("headRefOid") if isinstance(pr_payload.get("headRefOid"), str) else None,
            submission_name=submission_name,
            archive_url=archive_url,
            archive_sha256=(zip_audit.sha256 if zip_audit else (config.expected_archive_sha256 or extract_archive_sha256(body))),
            archive_bytes=(zip_audit.size_bytes if zip_audit else archive_bytes),
            findings=tuple(findings),
            zip_audit=zip_audit,
            inflate_smoke=inflate_smoke,
            body_patch_suggestions=body_suggestions,
            checked_commands=tuple(checked),
            work_dir=str(work_root) if (config.keep_work_dir or config.work_dir is not None) else None,
            elapsed_seconds=float(elapsed),
            measurement_utc=_utc_now(),
        )
    finally:
        if temp_root_obj is not None and not config.keep_work_dir:
            temp_root_obj.cleanup()


def self_test_result() -> PublicSubmissionAuditResult:
    body = """# submission name:
sample_submission

# upload zipped `archive.zip`
[archive.zip](https://github.com/example/repo/releases/download/v1/archive.zip)
Runtime tree at PR head [`abcdef0`](https://github.com/example/repo/tree/abcdef0123456789abcdef0123456789abcdef01/submissions/sample_submission).

# report.txt
Archive SHA-256: """ + "a" * 64 + """
Archive size bytes: 123
Final score: 0.19 [contest-CPU]

# does your submission require gpu for evaluation (inflation)?
no

# did you include the compression script? and want it to be merged?
yes

# is this submission competitive or innovative? explain why
Competitive: yes. @example #101.
"""
    payload = {
        "state": "OPEN",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "headRefOid": "abcdef0123456789abcdef0123456789abcdef01",
        "body": body,
        "comments": [],
        "url": "https://github.com/example/repo/pull/1",
    }
    config = PublicSubmissionAuditConfig(target_repo=DEFAULT_TARGET_REPO, pr_number=1)
    findings, body_text, submission_name, archive_url, archive_bytes, suggestions = _audit_pr_surface(payload, config)
    return PublicSubmissionAuditResult(
        schema_version=PUBLIC_PR_AUDIT_SCHEMA_VERSION,
        target_repo=DEFAULT_TARGET_REPO,
        pr_number=1,
        pr_url=payload["url"],
        head_sha=payload["headRefOid"],
        submission_name=submission_name,
        archive_url=archive_url,
        archive_sha256=extract_archive_sha256(body_text),
        archive_bytes=archive_bytes,
        findings=tuple(findings),
        body_patch_suggestions=suggestions,
        measurement_utc=_utc_now(),
        elapsed_seconds=0.0,
    )


def format_text_report(result: PublicSubmissionAuditResult) -> str:
    status = "PASS" if result.overall_clean else "FAIL"
    lines = [
        f"public submission PR audit: {status}",
        f"repo/pr: {result.target_repo}#{result.pr_number}",
        f"head: {result.head_sha or 'unknown'}",
        f"submission: {result.submission_name or 'unknown'}",
        f"archive: {result.archive_sha256 or 'unknown'} bytes={result.archive_bytes or 'unknown'}",
        f"errors={result.error_count} warnings={result.warn_count} info={result.info_count}",
    ]
    if result.body_patch_suggestions:
        lines.append("body_patch_suggestions:")
        lines.extend(f"- {item}" for item in result.body_patch_suggestions)
    if result.findings:
        lines.append("findings:")
        lines.extend(
            f"- [{finding.severity}] {finding.surface}:{finding.rule}: {finding.message}"
            for finding in result.findings
        )
    return "\n".join(lines)


def config_from_args(args: argparse.Namespace) -> PublicSubmissionAuditConfig:
    command: tuple[str, ...] = ()
    if getattr(args, "encoder_rebuild_command_json", None):
        payload = json.loads(Path(args.encoder_rebuild_command_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise SystemExit("--encoder-rebuild-command-json must contain a JSON list of strings")
        command = tuple(payload)
    python_bin = None
    if args.python_bin:
        raw_python_bin = Path(args.python_bin)
        python_bin = raw_python_bin if raw_python_bin.is_absolute() else Path.cwd() / raw_python_bin
    return PublicSubmissionAuditConfig(
        target_repo=args.repo,
        pr_number=int(args.pr),
        submission_name=args.submission_name,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        keep_work_dir=bool(args.keep_work_dir),
        expected_archive_sha256=args.expected_archive_sha256,
        expected_archive_bytes=args.expected_archive_bytes,
        run_inflate_smoke=bool(args.inflate_smoke),
        expected_output_sha256=args.expected_output_sha256,
        video_name=args.video_name,
        python_bin=python_bin,
        run_encoder_rebuild=bool(args.encoder_rebuild),
        encoder_rebuild_command=command,
        command_timeout_s=float(args.command_timeout_s),
        network_timeout_s=float(args.network_timeout_s),
    )
