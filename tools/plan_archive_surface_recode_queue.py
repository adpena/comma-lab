#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan a focused archive-surface recode/eval queue.

This consumes ``tools/build_archive_surface_inventory.py`` output and turns the
raw archive census into a smaller operator queue. The queue is intentionally
diagnostic: it ranks existing byte surfaces and records the custody/parity
checks that block promotion, but it never marks a row ready for exact eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = (
    "pr101_null_byte_smoke",
    "hfv1_pr101_adapter",
    "lfv1_lapose_foveation",
    "openpilot_prior_candidate",
    "z7_world_model_candidate",
)

CLASS_PRIORITIES = {
    "pr101_null_byte_smoke": 0,
    "hfv1_pr101_adapter": 1,
    "lfv1_lapose_foveation": 2,
    "openpilot_prior_candidate": 3,
    "z7_world_model_candidate": 4,
    "generic_entropy_headroom": 9,
}

CLASS_RATIONALES = {
    "pr101_null_byte_smoke": (
        "current FEC6/PR101 null-space smoke surface; highest relevance to "
        "master-gradient byte work"
    ),
    "hfv1_pr101_adapter": (
        "recent PR101 HFV1 adapter surface; connects foveation sidecars to "
        "the current frontier archive grammar"
    ),
    "lfv1_lapose_foveation": (
        "recent LFV1/LAPose foveation surface; high nominal entropy headroom "
        "but needs runtime/parity proof"
    ),
    "openpilot_prior_candidate": (
        "openpilot-prior candidate surface; relevant to driving-prior lanes"
    ),
    "z7_world_model_candidate": (
        "Z6/Z7/Z8 world-model-family archive surface; relevant to the "
        "predictive-substrate campaign"
    ),
    "generic_entropy_headroom": (
        "large entropy-headroom surface without a specific current frontier "
        "binding"
    ),
}

EVIDENCE_FILENAMES = {
    "inflate_sh": "inflate.sh",
    "inflate_py": "inflate.py",
    "report_txt": "report.txt",
    "archive_manifest_json": "archive_manifest.json",
    "archive_member_manifest_json": "archive_member_manifest.json",
    "manifest_json": "manifest.json",
    "candidate_json": "candidate.json",
    "readiness_json": "readiness.json",
    "summary_json": "summary.json",
    "contest_auth_eval_json": "contest_auth_eval.json",
    "inflated_outputs_manifest_json": "inflated_outputs_manifest.json",
    "provenance_json": "provenance.json",
    "composed_archive_manifest_json": "composed_archive_manifest.json",
    "official_inflate_raw_comparison_json": "official_inflate_raw_comparison.json",
    "raw_comparison_json": "raw_comparison.json",
    "runtime_consumer_proof_skeleton_json": "runtime_consumer_proof_skeleton.json",
    "adapter_smoke_summary_json": "adapter_smoke_summary.json",
}


@dataclass(frozen=True)
class RecodeQueueRow:
    rank: int
    candidate_class: str
    class_priority: int
    class_rationale: str
    archive_zip_path: str
    archive_zip_bytes: int
    archive_zip_sha256: str
    archive_sha256_verified: bool
    member_count: int
    member_names: list[str]
    submission_shape_hint: str
    estimated_recoverable_zip_bytes: int
    estimated_rate_delta_if_floor_reached: float
    duplicate_archive_sha_count: int
    evidence_files: dict[str, str | None]
    has_inflate_runtime: bool
    has_report: bool
    has_archive_manifest: bool
    has_auth_eval_artifact: bool
    has_inflate_parity_artifact: bool
    byte_closed_candidate: bool
    required_next_checks: list[str]
    promotion_blockers: list[str]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class RecodeQueue:
    schema: str
    generated_at_utc: str
    inventory_json: str
    inventory_generated_at_utc: str | None
    classes_included: list[str]
    rows_seen: int
    rows_after_filter: int
    rows_after_dedup: int
    queue: list[RecodeQueueRow]
    dropped_duplicate_archives: int
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_inventory_json(results_root: Path) -> Path:
    candidates = sorted(
        results_root.glob("archive_surface_inventory_*/archive_surface_inventory.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no archive_surface_inventory_*/archive_surface_inventory.json under {results_root}"
        )
    return candidates[0]


def _candidate_class(archive_path: str) -> str:
    path = archive_path.lower()
    if "pr101_gold_master_gradient_null_byte_removal_smoke" in path:
        return "pr101_null_byte_smoke"
    if (
        "pr101_fec6_lfv1_hfv1_integrated_adapter" in path
        or "hfv1_sidecar_candidates" in path
    ):
        return "hfv1_pr101_adapter"
    if "theoretical_floor_lfv1" in path or "lapose_foveation" in path:
        return "lfv1_lapose_foveation"
    if "categorical_openpilot_payload_candidate" in path:
        return "openpilot_prior_candidate"
    if "z7_mamba" in path or "world_model" in path:
        return "z7_world_model_candidate"
    return "generic_entropy_headroom"


def _ancestor_dirs(path: Path, *, max_levels: int = 3) -> list[Path]:
    dirs: list[Path] = []
    current = path.parent
    for _ in range(max_levels):
        if str(current) in ("", "."):
            break
        if current in dirs:
            break
        dirs.append(current)
        if current.parent == current:
            break
        current = current.parent
    return dirs


def _candidate_evidence_dirs(archive_path: Path) -> list[Path]:
    dirs: list[Path] = []
    for ancestor in _ancestor_dirs(archive_path):
        dirs.append(ancestor)
        for rel in (
            "submission_dir",
            "auth_eval_work",
            "advisory_eval",
            "advisory_eval/submission_dir",
            "advisory_eval/auth_eval_work",
            "adapter_smoke",
            "adapter_smoke/extracted",
            "archive",
            "archive_candidate",
            "official_inflate_control",
        ):
            dirs.append(ancestor / rel)
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in dirs:
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_dir():
            existing.append(path)
    return existing


def _find_named_file(dirs: list[Path], filename: str) -> str | None:
    for directory in dirs:
        direct = directory / filename
        if direct.exists() and direct.is_file():
            return str(direct)
    return None


def _evidence_files(archive_path: Path) -> dict[str, str | None]:
    dirs = _candidate_evidence_dirs(archive_path)
    return {
        key: _find_named_file(dirs, filename)
        for key, filename in EVIDENCE_FILENAMES.items()
    }


def _required_next_checks(
    *,
    evidence: dict[str, str | None],
    archive_sha_verified: bool,
    byte_closed_candidate: bool,
) -> tuple[list[str], list[str]]:
    checks = [
        "recompute archive/member SHA-256 and byte counts",
        "prove runtime consumes the candidate archive bytes",
        "run full-frame inflate parity or same-runtime auth-eval comparison",
        "decompose score axis before promotion: rate / SegNet / PoseNet",
        "run strict pre-submission compliance only after byte closure",
    ]
    blockers: list[str] = []
    if not archive_sha_verified:
        blockers.append("archive_sha_mismatch_or_missing")
    if not (evidence["inflate_sh"] or evidence["inflate_py"]):
        blockers.append("missing_inflate_runtime")
    if not (
        evidence["archive_manifest_json"]
        or evidence["archive_member_manifest_json"]
        or evidence["manifest_json"]
    ):
        blockers.append("missing_archive_or_candidate_manifest")
    if not evidence["report_txt"]:
        blockers.append("missing_report_txt")
    if not evidence["contest_auth_eval_json"]:
        blockers.append("missing_contest_auth_eval_json")
    if not (
        evidence["official_inflate_raw_comparison_json"]
        or evidence["raw_comparison_json"]
        or evidence["inflated_outputs_manifest_json"]
    ):
        blockers.append("missing_inflate_parity_or_output_manifest")
    if not byte_closed_candidate:
        blockers.append("not_byte_closed_submission_candidate")
    return checks, blockers


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    candidate_class = _candidate_class(str(row["archive_zip_path"]))
    return (
        CLASS_PRIORITIES[candidate_class],
        -int(row["estimated_recoverable_zip_bytes"]),
        -int(row["archive_zip_bytes"]),
        str(row["archive_zip_path"]),
    )


def build_queue(
    inventory_json: Path,
    *,
    include_classes: set[str],
    include_all_classes: bool,
    limit: int,
    allow_duplicate_archives: bool,
) -> RecodeQueue:
    payload = json.loads(inventory_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("inventory JSON has no list-valued rows")

    selected: list[dict[str, Any]] = []
    for row in rows:
        candidate_class = _candidate_class(str(row["archive_zip_path"]))
        if include_all_classes or candidate_class in include_classes:
            selected.append(row)

    selected.sort(key=_row_sort_key)

    seen_shas: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dropped = 0
    for row in selected:
        sha = str(row["archive_zip_sha256"])
        if not allow_duplicate_archives and sha in seen_shas:
            dropped += 1
            continue
        seen_shas.add(sha)
        deduped.append(row)

    queue_rows: list[RecodeQueueRow] = []
    for row in deduped[:limit]:
        archive_path = Path(str(row["archive_zip_path"]))
        archive_sha = str(row["archive_zip_sha256"])
        archive_sha_verified = (
            archive_path.exists()
            and archive_path.is_file()
            and _sha256_file(archive_path) == archive_sha
        )
        evidence = _evidence_files(archive_path)
        has_inflate_runtime = bool(evidence["inflate_sh"] or evidence["inflate_py"])
        has_report = bool(evidence["report_txt"])
        has_manifest = bool(
            evidence["archive_manifest_json"]
            or evidence["archive_member_manifest_json"]
            or evidence["manifest_json"]
        )
        has_auth_eval = bool(evidence["contest_auth_eval_json"])
        has_parity = bool(
            evidence["official_inflate_raw_comparison_json"]
            or evidence["raw_comparison_json"]
            or evidence["inflated_outputs_manifest_json"]
        )
        byte_closed = bool(archive_path.exists() and has_inflate_runtime)
        checks, blockers = _required_next_checks(
            evidence=evidence,
            archive_sha_verified=archive_sha_verified,
            byte_closed_candidate=byte_closed,
        )
        candidate_class = _candidate_class(str(row["archive_zip_path"]))
        queue_rows.append(
            RecodeQueueRow(
                rank=len(queue_rows) + 1,
                candidate_class=candidate_class,
                class_priority=CLASS_PRIORITIES[candidate_class],
                class_rationale=CLASS_RATIONALES[candidate_class],
                archive_zip_path=str(row["archive_zip_path"]),
                archive_zip_bytes=int(row["archive_zip_bytes"]),
                archive_zip_sha256=archive_sha,
                archive_sha256_verified=archive_sha_verified,
                member_count=int(row["member_count"]),
                member_names=list(row["member_names"]),
                submission_shape_hint=str(row["submission_shape_hint"]),
                estimated_recoverable_zip_bytes=int(
                    row["estimated_recoverable_zip_bytes"]
                ),
                estimated_rate_delta_if_floor_reached=float(
                    row["estimated_rate_delta_if_floor_reached"]
                ),
                duplicate_archive_sha_count=int(row["duplicate_archive_sha_count"]),
                evidence_files=evidence,
                has_inflate_runtime=has_inflate_runtime,
                has_report=has_report,
                has_archive_manifest=has_manifest,
                has_auth_eval_artifact=has_auth_eval,
                has_inflate_parity_artifact=has_parity,
                byte_closed_candidate=byte_closed,
                required_next_checks=checks,
                promotion_blockers=blockers,
            )
        )

    return RecodeQueue(
        schema="archive_surface_recode_queue_v1",
        generated_at_utc=_utc_iso(),
        inventory_json=str(inventory_json),
        inventory_generated_at_utc=payload.get("generated_at_utc"),
        classes_included=(
            sorted(CLASS_PRIORITIES) if include_all_classes else sorted(include_classes)
        ),
        rows_seen=len(rows),
        rows_after_filter=len(selected),
        rows_after_dedup=len(deduped),
        queue=queue_rows,
        dropped_duplicate_archives=dropped,
    )


def render_markdown(queue: RecodeQueue) -> str:
    lines = [
        "# Archive Surface Recode Queue",
        "",
        f"- Generated UTC: {queue.generated_at_utc}",
        f"- Inventory JSON: `{queue.inventory_json}`",
        f"- Inventory generated UTC: {queue.inventory_generated_at_utc}",
        f"- Rows seen: {queue.rows_seen}",
        f"- Rows after class filter: {queue.rows_after_filter}",
        f"- Rows after duplicate-archive filter: {queue.rows_after_dedup}",
        f"- Dropped duplicate archives: {queue.dropped_duplicate_archives}",
        "- Score claim: false",
        "- Promotion eligible: false",
        "- Ready for exact eval dispatch: false",
        "",
        "## Queue",
        "",
        "| rank | class | archive | bytes | recoverable | rate delta | evidence | blockers |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in queue.queue:
        evidence_flags = []
        if row.archive_sha256_verified:
            evidence_flags.append("sha")
        if row.has_inflate_runtime:
            evidence_flags.append("runtime")
        if row.has_report:
            evidence_flags.append("report")
        if row.has_archive_manifest:
            evidence_flags.append("manifest")
        if row.has_auth_eval_artifact:
            evidence_flags.append("auth-eval")
        if row.has_inflate_parity_artifact:
            evidence_flags.append("parity")
        blockers = ", ".join(row.promotion_blockers) or "none"
        evidence = ", ".join(evidence_flags) or "none"
        lines.append(
            "| "
            f"{row.rank} | "
            f"`{row.candidate_class}` | "
            f"`{row.archive_zip_path}` | "
            f"{row.archive_zip_bytes} | "
            f"{row.estimated_recoverable_zip_bytes} | "
            f"{row.estimated_rate_delta_if_floor_reached:.12g} | "
            f"{evidence} | "
            f"{blockers} |"
        )

    lines.extend(["", "## Required Checks", ""])
    if queue.queue:
        for check in queue.queue[0].required_next_checks:
            lines.append(f"- {check}")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-json",
        type=Path,
        help="Path to archive_surface_inventory.json; defaults to newest scan.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/results"),
        help="Root used to discover the newest inventory if --inventory-json is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"archive_surface_recode_queue_{_utc_stamp()}",
    )
    parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        choices=sorted(CLASS_PRIORITIES),
        help="Candidate class to include. Repeatable. Defaults to current frontier-relevant classes.",
    )
    parser.add_argument("--include-all-classes", action="store_true")
    parser.add_argument("--allow-duplicate-archives", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_json = args.inventory_json or _latest_inventory_json(args.results_root)
    include_classes = set(args.classes or DEFAULT_CLASSES)
    queue = build_queue(
        inventory_json,
        include_classes=include_classes,
        include_all_classes=args.include_all_classes,
        limit=max(0, args.limit),
        allow_duplicate_archives=args.allow_duplicate_archives,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(queue.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "archive_surface_recode_queue.json").write_text(
        payload,
        encoding="utf-8",
    )
    (args.output_dir / "archive_surface_recode_queue.md").write_text(
        render_markdown(queue),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
