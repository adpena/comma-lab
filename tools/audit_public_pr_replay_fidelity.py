#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed audit for public-PR replay fidelity.

This compares a public PR intake archive and public metadata against a local
exact replay result. A mismatched score is a runtime/eval fidelity problem, not
evidence that the public PR family underperformed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402

DEFAULT_SCORE_TOLERANCE = 0.001


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _archive_meta(path: Path) -> dict[str, Any]:
    import hashlib

    blob = path.read_bytes()
    with ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        members = [{"name": info.filename, "file_size": info.file_size} for info in infos]
    return {
        "archive": _repo_rel(path),
        "archive_bytes": len(blob),
        "archive_sha256": hashlib.sha256(blob).hexdigest(),
        "member_count": len(members),
        "members": members,
    }


def _load_adjudication_from_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    prefix = "ADJUDICATION_JSON:"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :].strip())
    return None


def _load_replay_result(eval_dir: Path) -> dict[str, Any]:
    for name in ("contest_auth_eval.adjudicated.json", "contest_auth_eval.json"):
        path = eval_dir / name
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict):
                payload["_result_source"] = _repo_rel(path)
                return payload
    payload = _load_adjudication_from_log(eval_dir / "adjudication.log")
    if payload is not None:
        payload["_result_source"] = _repo_rel(eval_dir / "adjudication.log")
        return payload
    raise ValueError(f"no replay result JSON or ADJUDICATION_JSON log found in {eval_dir}")


def _score(payload: dict[str, Any]) -> float | None:
    for key in ("score_recomputed_from_components", "score_recomputed", "score"):
        value = payload.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _archive_sha(payload: dict[str, Any]) -> str | None:
    for key in ("archive_sha256",):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("archive_sha256"), str):
        return str(provenance["archive_sha256"])
    return None


def _archive_bytes(payload: dict[str, Any]) -> int | None:
    for key in ("archive_size_bytes", "archive_bytes"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def audit_public_pr_replay_fidelity(
    *,
    intake_dir: Path,
    eval_dir: Path,
    score_tolerance: float = DEFAULT_SCORE_TOLERANCE,
) -> AuditReport:
    blockers: list[str] = []
    metadata_path = intake_dir / "pr_metadata.json"
    archive_path = intake_dir / "archive.zip"
    eval_archive_path = eval_dir / "archive.zip"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    if not isinstance(metadata, dict):
        metadata = {}
        blockers.append("public_pr_metadata_not_object")
    if not metadata_path.is_file():
        blockers.append("missing_public_pr_metadata")
    if not archive_path.is_file():
        blockers.append("missing_public_pr_archive")
    if not eval_archive_path.is_file():
        blockers.append("missing_replay_archive")

    public_score = metadata.get("leaderboard_score")
    if not isinstance(public_score, int | float):
        blockers.append("missing_public_leaderboard_score")
        public_score = None

    intake_archive = _archive_meta(archive_path) if archive_path.is_file() else {}
    eval_archive = _archive_meta(eval_archive_path) if eval_archive_path.is_file() else {}
    replay_result: dict[str, Any] = {}
    try:
        replay_result = _load_replay_result(eval_dir)
    except ValueError as exc:
        blockers.append(str(exc))

    replay_score = _score(replay_result)
    if replay_result and replay_score is None:
        blockers.append("missing_replay_score")

    if intake_archive and eval_archive:
        if intake_archive.get("archive_sha256") != eval_archive.get("archive_sha256"):
            blockers.append("archive_sha_mismatch_between_intake_and_replay")
        if intake_archive.get("archive_bytes") != eval_archive.get("archive_bytes"):
            blockers.append("archive_bytes_mismatch_between_intake_and_replay")

    result_sha = _archive_sha(replay_result)
    result_bytes = _archive_bytes(replay_result)
    if result_sha and eval_archive and result_sha != eval_archive.get("archive_sha256"):
        blockers.append("replay_result_archive_sha_mismatch")
    if result_bytes is not None and eval_archive and result_bytes != eval_archive.get("archive_bytes"):
        blockers.append("replay_result_archive_bytes_mismatch")

    score_delta = None
    if isinstance(public_score, int | float) and replay_score is not None:
        score_delta = replay_score - float(public_score)
        if abs(score_delta) > score_tolerance:
            blockers.append("public_leaderboard_score_mismatch")

    summary = {
        "pr_number": metadata.get("pr_number"),
        "leaderboard_name": metadata.get("leaderboard_name"),
        "public_leaderboard_score": float(public_score) if isinstance(public_score, int | float) else None,
        "replay_score": replay_score,
        "score_delta_vs_public": score_delta,
        "score_tolerance": score_tolerance,
        "intake_archive": intake_archive,
        "replay_archive": eval_archive,
        "replay_result_source": replay_result.get("_result_source"),
        "classification": (
            "public_runtime_or_eval_fidelity_mismatch"
            if "public_leaderboard_score_mismatch" in blockers
            else "public_replay_fidelity_closed"
        ),
    }
    return AuditReport(
        audit="public_pr_replay_fidelity",
        readiness_key="replay_fidelity_closed",
        ready=not blockers,
        blockers=tuple(blockers),
        summary=summary,
        score_claim=False,
        dispatch_attempted=False,
        metadata={
            "intake_dir": _repo_rel(intake_dir),
            "eval_dir": _repo_rel(eval_dir),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=DEFAULT_SCORE_TOLERANCE)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    report = audit_public_pr_replay_fidelity(
        intake_dir=args.intake_dir,
        eval_dir=args.eval_dir,
        score_tolerance=args.score_tolerance,
    )
    if args.format == "json":
        print(json_text(report.to_dict()), end="")
    else:
        print(report.render_text())
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
