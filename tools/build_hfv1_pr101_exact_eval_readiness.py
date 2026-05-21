#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a paired exact-eval readiness packet for HFV1 PR101 adapter archives.

The archive-surface recode queue identifies HFV1 PR101 adapter archives that
have local runtime/parity evidence and are blocked primarily on missing
``contest_auth_eval.json``. This tool verifies archive custody, binds the
candidate archives to the local runtime tree through the paired Modal dispatcher
plan, and dry-runs lane claims. It never dispatches remote work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tools.dispatch_modal_paired_auth_eval import build_plan  # noqa: E402

DEFAULT_RESULT_ROOT = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex"
)
DEFAULT_SUBMISSION_DIR = DEFAULT_RESULT_ROOT / "submission_dir"
DEFAULT_OUTPUT_ROOT = Path("experiments/results")
DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
DEFAULT_CLAIM_AGENT = "codex:hfv1_pr101_exact_eval_readiness"
DEFAULT_CLASSES = {"hfv1_pr101_adapter"}
MIN_NONTRIVIAL_RECOVERABLE_BYTES = 1000


@dataclass(frozen=True)
class ZipMemberCustody:
    name: str
    file_size: int
    compress_size: int
    crc: int
    compress_type: int
    sha256: str


@dataclass(frozen=True)
class DryRunClaimResult:
    axis: str
    lane_id: str
    instance_job_id: str
    command: list[str]
    returncode: int
    conflict_free: bool
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Hfv1ExactEvalCandidate:
    variant: str
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    queue_rank: int
    estimated_recoverable_zip_bytes: int
    estimated_rate_delta_if_floor_reached: float
    zip_members: list[ZipMemberCustody]
    manifest_match: bool
    manifest_sources: list[str]
    local_runtime_submission_dir: str
    local_inflate_sh_exists: bool
    local_report_exists: bool
    local_archive_manifest_exists: bool
    local_inflate_parity_artifact: str | None
    queue_promotion_blockers: list[str]
    readiness_blockers: list[str]
    dispatch_plan_ready: bool
    paired_dispatch_plan: dict[str, Any] | None
    paired_dispatch_command_plan: list[str]
    paired_dispatch_command_execute: list[str]
    dry_run_claims: list[DryRunClaimResult]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class Hfv1ExactEvalReadinessPacket:
    schema: str
    generated_at_utc: str
    queue_json: str
    submission_dir: str
    rows_seen: int
    candidates_emitted: int
    skipped_rows: list[dict[str, Any]]
    candidates: list[Hfv1ExactEvalCandidate]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    full = _repo_path(path).resolve()
    try:
        return str(full.relative_to(REPO_ROOT))
    except ValueError:
        return str(full)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "candidate"


def _quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_members(path: Path) -> list[ZipMemberCustody]:
    out: list[ZipMemberCustody] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            payload = archive.read(info.filename)
            out.append(
                ZipMemberCustody(
                    name=info.filename,
                    file_size=int(info.file_size),
                    compress_size=int(info.compress_size),
                    crc=int(info.CRC),
                    compress_type=int(info.compress_type),
                    sha256=_sha256_bytes(payload),
                )
            )
    return out


def _latest_queue_json(results_root: Path) -> Path:
    candidates = sorted(
        results_root.glob("archive_surface_recode_queue_*/archive_surface_recode_queue.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no archive_surface_recode_queue_*/archive_surface_recode_queue.json under {results_root}"
        )
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _archive_manifest_index(paths: list[Path]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for path in paths:
        full = _repo_path(path)
        if not full.exists():
            continue
        payload = _load_json(full)
        archive_nodes: list[dict[str, Any]] = []
        archives = payload.get("archives")
        if isinstance(archives, dict):
            archive_nodes.extend(
                value for value in archives.values() if isinstance(value, dict)
            )
        archive = payload.get("archive")
        if isinstance(archive, dict):
            archive_nodes.append(archive)
        source_archive = payload.get("source_pr101_archive")
        if isinstance(source_archive, dict):
            archive_nodes.append(source_archive)

        for node in archive_nodes:
            sha = str(node.get("sha256") or "").strip().lower()
            if not sha:
                continue
            label = f"{_repo_rel(full)}::{node.get('path') or node.get('name') or 'archive'}"
            index.setdefault(sha, []).append(label)
    return index


def _variant_from_archive_path(path: str) -> str:
    parts = Path(path).parts
    for part in reversed(parts):
        if part.startswith("archive_"):
            return part.removeprefix("archive_")
    for idx, part in enumerate(parts):
        if part == "advisory_raw_eval" and idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name


def _plan_command(
    *,
    archive: Path,
    archive_sha256: str,
    submission_dir: Path,
    variant: str,
    run_id: str,
    lane_id_base: str,
    output_root: Path,
) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/dispatch_modal_paired_auth_eval.py",
        "--archive",
        _repo_rel(archive),
        "--submission-dir",
        _repo_rel(submission_dir),
        "--inflate-sh",
        "inflate.sh",
        "--label",
        f"hfv1_pr101_{variant}",
        "--run-id",
        run_id,
        "--pair-group-id",
        run_id,
        "--lane-id-base",
        lane_id_base,
        "--output-root",
        str(output_root),
        "--expected-archive-sha256",
        archive_sha256,
        "--expected-runtime-tree-sha256",
        "auto",
        "--skip-axis-if-promotable-anchor-exists",
    ]


def _dry_run_claim(
    *,
    axis: str,
    lane_id: str,
    instance_job_id: str,
    archive_sha256: str,
    archive_bytes: int,
    pair_group_id: str,
    claim_agent: str,
) -> DryRunClaimResult:
    cmd = [
        sys.executable,
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        lane_id,
        "--platform",
        "modal",
        "--instance-job-id",
        instance_job_id,
        "--agent",
        claim_agent,
        "--status",
        "dry_run_planned_modal_auth_eval",
        "--notes",
        (
            "HFV1 PR101 paired Modal auth-eval readiness dry-run; "
            f"axis={axis}; pair_group_id={pair_group_id}; "
            f"archive_sha={archive_sha256}; bytes={archive_bytes}"
        ),
        "--dry-run",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return DryRunClaimResult(
        axis=axis,
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        command=cmd,
        returncode=int(proc.returncode),
        conflict_free=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _relative_existing(path_text: str | None) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not _repo_path(path).exists():
        return None
    return _repo_rel(path)


def _candidate_from_queue_row(
    row: dict[str, Any],
    *,
    queue_rank: int,
    manifest_index: dict[str, list[str]],
    submission_dir: Path,
    output_root: Path,
    min_recoverable_bytes: int,
    claim_agent: str,
) -> Hfv1ExactEvalCandidate | None:
    archive = _repo_path(Path(str(row["archive_zip_path"]))).resolve()
    if not archive.is_file():
        return None
    archive_sha = _sha256_file(archive)
    archive_bytes = archive.stat().st_size
    queue_sha = str(row["archive_zip_sha256"]).strip().lower()
    if archive_sha != queue_sha:
        raise ValueError(
            f"queue archive sha mismatch for {archive}: queue={queue_sha} actual={archive_sha}"
        )

    estimated_recoverable = int(row["estimated_recoverable_zip_bytes"])
    if estimated_recoverable < min_recoverable_bytes:
        return None

    variant = _safe_slug(_variant_from_archive_path(str(row["archive_zip_path"])))
    short_sha = archive_sha[:12]
    run_id = f"hfv1_pr101_{variant}_{short_sha}"
    lane_id_base = f"hfv1_pr101_exact_eval_{variant}_{short_sha}"
    plan_command = _plan_command(
        archive=archive,
        archive_sha256=archive_sha,
        submission_dir=submission_dir,
        variant=variant,
        run_id=run_id,
        lane_id_base=lane_id_base,
        output_root=output_root,
    )
    execute_command = [*plan_command, "--execute"]

    readiness_blockers: list[str] = []
    local_inflate = submission_dir / "inflate.sh"
    local_report = submission_dir / "report.txt"
    local_manifest = submission_dir / "archive_manifest.json"
    if not local_inflate.is_file():
        readiness_blockers.append("missing_submission_dir_inflate_sh")
    if not local_report.is_file():
        readiness_blockers.append("missing_submission_dir_report_txt")
    if not local_manifest.is_file():
        readiness_blockers.append("missing_submission_dir_archive_manifest")

    manifest_sources = manifest_index.get(archive_sha, [])
    if not manifest_sources:
        readiness_blockers.append("candidate_archive_not_found_in_known_manifests")

    paired_plan: dict[str, Any] | None = None
    dry_run_claims: list[DryRunClaimResult] = []
    if not readiness_blockers:
        paired_plan = build_plan(
            archive=archive,
            submission_dir=str(submission_dir),
            inflate_sh="inflate.sh",
            run_id=run_id,
            pair_group_id=run_id,
            lane_id_base=lane_id_base,
            output_root=output_root,
            modal_bin=".venv/bin/modal",
            gpu="T4",
            claim_agent=claim_agent,
            claim_notes=(
                "HFV1 PR101 adapter exact-eval readiness; archive from "
                "archive-surface recode queue"
            ),
            expected_runtime_tree_sha256="auto",
            expected_archive_sha256=archive_sha,
            skip_axis_if_promotable_anchor_exists=True,
            repo_root=REPO_ROOT,
        )
        for axis in ("contest_cuda", "contest_cpu"):
            dry = _dry_run_claim(
                axis=axis,
                lane_id=str(paired_plan["lanes"][axis]),
                instance_job_id=f"{run_id}_{axis.removeprefix('contest_')}",
                archive_sha256=archive_sha,
                archive_bytes=archive_bytes,
                pair_group_id=run_id,
                claim_agent=claim_agent,
            )
            dry_run_claims.append(dry)
            if not dry.conflict_free:
                readiness_blockers.append(f"active_lane_claim_conflict_{axis}")

    evidence = row.get("evidence_files") if isinstance(row.get("evidence_files"), dict) else {}
    parity = (
        _relative_existing(evidence.get("official_inflate_raw_comparison_json"))
        or _relative_existing(evidence.get("raw_comparison_json"))
        or _relative_existing(evidence.get("inflated_outputs_manifest_json"))
    )
    dispatch_plan_ready = paired_plan is not None and not readiness_blockers

    return Hfv1ExactEvalCandidate(
        variant=variant,
        archive_zip_path=_repo_rel(archive),
        archive_zip_bytes=archive_bytes,
        archive_zip_sha256=archive_sha,
        queue_rank=queue_rank,
        estimated_recoverable_zip_bytes=estimated_recoverable,
        estimated_rate_delta_if_floor_reached=float(
            row["estimated_rate_delta_if_floor_reached"]
        ),
        zip_members=_zip_members(archive),
        manifest_match=bool(manifest_sources),
        manifest_sources=manifest_sources,
        local_runtime_submission_dir=_repo_rel(submission_dir),
        local_inflate_sh_exists=local_inflate.is_file(),
        local_report_exists=local_report.is_file(),
        local_archive_manifest_exists=local_manifest.is_file(),
        local_inflate_parity_artifact=parity,
        queue_promotion_blockers=list(row.get("promotion_blockers") or []),
        readiness_blockers=readiness_blockers,
        dispatch_plan_ready=dispatch_plan_ready,
        paired_dispatch_plan=paired_plan,
        paired_dispatch_command_plan=plan_command,
        paired_dispatch_command_execute=execute_command,
        dry_run_claims=dry_run_claims,
    )


def build_packet(
    *,
    queue_json: Path,
    submission_dir: Path,
    output_root: Path,
    min_recoverable_bytes: int,
    claim_agent: str,
) -> Hfv1ExactEvalReadinessPacket:
    queue_payload = _load_json(queue_json)
    rows = queue_payload.get("queue")
    if not isinstance(rows, list):
        raise ValueError(f"queue JSON has no list-valued queue: {queue_json}")

    manifest_index = _archive_manifest_index(
        [
            DEFAULT_RESULT_ROOT / "composed_archive_manifest.json",
            DEFAULT_RESULT_ROOT / "seed_top16_component_hardpairs_manifest.json",
            DEFAULT_RESULT_ROOT / "submission_dir" / "archive_manifest.json",
        ]
    )
    candidates: list[Hfv1ExactEvalCandidate] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row.get("candidate_class") not in DEFAULT_CLASSES:
            continue
        blockers = list(row.get("promotion_blockers") or [])
        recoverable = int(row.get("estimated_recoverable_zip_bytes") or 0)
        if blockers != ["missing_contest_auth_eval_json"]:
            skipped.append(
                {
                    "archive_zip_path": row.get("archive_zip_path"),
                    "reason": "not_one_blocker_missing_contest_auth_eval_json",
                    "promotion_blockers": blockers,
                    "estimated_recoverable_zip_bytes": recoverable,
                }
            )
            continue
        if recoverable < min_recoverable_bytes:
            skipped.append(
                {
                    "archive_zip_path": row.get("archive_zip_path"),
                    "reason": "below_min_recoverable_bytes",
                    "estimated_recoverable_zip_bytes": recoverable,
                    "min_recoverable_bytes": min_recoverable_bytes,
                }
            )
            continue
        candidate = _candidate_from_queue_row(
            row,
            queue_rank=int(row["rank"]),
            manifest_index=manifest_index,
            submission_dir=_repo_path(submission_dir).resolve(),
            output_root=output_root,
            min_recoverable_bytes=min_recoverable_bytes,
            claim_agent=claim_agent,
        )
        if candidate is not None:
            candidates.append(candidate)

    return Hfv1ExactEvalReadinessPacket(
        schema="hfv1_pr101_exact_eval_readiness_v1",
        generated_at_utc=_utc_iso(),
        queue_json=_repo_rel(queue_json),
        submission_dir=_repo_rel(submission_dir),
        rows_seen=len(rows),
        candidates_emitted=len(candidates),
        skipped_rows=skipped,
        candidates=candidates,
    )


def render_markdown(packet: Hfv1ExactEvalReadinessPacket) -> str:
    lines = [
        "# HFV1 PR101 Exact-Eval Readiness",
        "",
        f"- Generated UTC: {packet.generated_at_utc}",
        f"- Queue JSON: `{packet.queue_json}`",
        f"- Submission dir: `{packet.submission_dir}`",
        f"- Rows seen: {packet.rows_seen}",
        f"- Candidates emitted: {packet.candidates_emitted}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
        "## Candidates",
        "",
        "| variant | archive | bytes | recoverable | manifest | parity | dispatch plan | blockers |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in packet.candidates:
        blockers = ", ".join(row.readiness_blockers) or "none"
        parity = row.local_inflate_parity_artifact or "missing"
        lines.append(
            "| "
            f"`{row.variant}` | "
            f"`{row.archive_zip_path}` | "
            f"{row.archive_zip_bytes} | "
            f"{row.estimated_recoverable_zip_bytes} | "
            f"{'yes' if row.manifest_match else 'no'} | "
            f"`{parity}` | "
            f"{'ready' if row.dispatch_plan_ready else 'blocked'} | "
            f"{blockers} |"
        )

    lines.extend(["", "## Plan Commands", ""])
    for row in packet.candidates:
        lines.append(f"### {row.variant}")
        lines.append("")
        lines.append("Plan only:")
        lines.append("")
        lines.append("```bash")
        lines.append(_quote_command(row.paired_dispatch_command_plan))
        lines.append("```")
        lines.append("")
        lines.append("Execute after operator/agent decision:")
        lines.append("")
        lines.append("```bash")
        lines.append(_quote_command(row.paired_dispatch_command_execute))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/results"),
        help="Used to discover the newest archive-surface recode queue.",
    )
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv1_pr101_exact_eval_readiness_{_utc_stamp()}",
    )
    parser.add_argument(
        "--min-recoverable-bytes",
        type=int,
        default=MIN_NONTRIVIAL_RECOVERABLE_BYTES,
    )
    parser.add_argument("--claim-agent", default=DEFAULT_CLAIM_AGENT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queue_json = args.queue_json or _latest_queue_json(args.results_root)
    packet = build_packet(
        queue_json=queue_json,
        submission_dir=args.submission_dir,
        output_root=args.output_root,
        min_recoverable_bytes=max(0, args.min_recoverable_bytes),
        claim_agent=args.claim_agent,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv1_pr101_exact_eval_readiness.json").write_text(
        payload,
        encoding="utf-8",
    )
    (args.output_dir / "hfv1_pr101_exact_eval_readiness.md").write_text(
        render_markdown(packet),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
