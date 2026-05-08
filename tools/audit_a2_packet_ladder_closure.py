#!/usr/bin/env python3
"""Audit A2 packet-ladder closure artifacts before exact-eval dispatch.

This is an operator-visible guard for the A2 hardening class: packet builders
and runtime probes may prove byte closure or parse closure, but they must keep
stub/proxy/no-score blockers alive until exact CUDA/CPU auth eval exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

A2_PATHSPECS = (
    ":(glob)experiments/results/**/a2_packet_ladder_manifest.json",
    ":(glob)experiments/results/**/candidate_manifest.json",
    ":(glob)experiments/results/**/a2_runtime_closure*.json",
    ":(glob)experiments/results/**/runtime_closure*.json",
)

LOCAL_A2_GLOBS = (
    "experiments/results/**/a2_packet_ladder_manifest.json",
    "experiments/results/**/candidate_manifest.json",
    "experiments/results/**/a2_runtime_closure*.json",
    "experiments/results/**/runtime_closure*.json",
)


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _tracked_manifest_paths(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", *A2_PATHSPECS],
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return sorted(
        {
            repo_root / raw.decode("utf-8")
            for raw in result.stdout.split(b"\0")
            if raw
        }
    )


def _local_manifest_paths(repo_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in LOCAL_A2_GLOBS:
        paths.update(repo_root.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _looks_a2(payload: dict[str, Any], rel: str) -> bool:
    schema = str(payload.get("schema", "")).lower()
    tool = str(payload.get("tool", "")).lower()
    return (
        "a2" in schema
        or "a2" in tool
        or "a2" in rel.lower()
        or str(payload.get("wire_format", "")).upper() == "A2K1"
    )


def _check_authority_false(payload: dict[str, Any], rel: str, violations: list[str]) -> None:
    for field, expected in FALSE_AUTHORITY_FIELDS.items():
        if payload.get(field) is not expected:
            violations.append(
                f"{rel}: {field} must be {expected!r} for A2 packet/probe artifacts "
                "until exact eval and blocker review close"
            )


def _check_blockers(payload: dict[str, Any], rel: str, violations: list[str]) -> None:
    blockers = payload.get("dispatch_blockers")
    if not isinstance(blockers, list) or not blockers:
        violations.append(f"{rel}: dispatch_blockers must be a non-empty list")
    closure = payload.get("packet_closure") or payload.get("runtime_closure")
    if not isinstance(closure, dict):
        return
    cleared = closure.get("cleared_blockers")
    if cleared is None:
        return
    if not isinstance(cleared, list):
        violations.append(f"{rel}: cleared_blockers must be a list when present")
        return
    evidence = closure.get("cleared_blockers_by_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    for blocker in cleared:
        if blocker in (blockers or []):
            violations.append(
                f"{rel}: blocker {blocker!r} appears in both cleared_blockers "
                "and dispatch_blockers"
            )
        if not evidence.get(blocker):
            violations.append(
                f"{rel}: blocker {blocker!r} cleared without evidence label"
            )


def audit(repo_root: Path = REPO, *, tracked_only: bool = False) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    paths = _tracked_manifest_paths(repo_root)
    if not tracked_only:
        paths = sorted({*paths, *_local_manifest_paths(repo_root)})
    violations: list[str] = []
    scanned: list[str] = []
    for path in paths:
        rel = _repo_rel(path, repo_root)
        payload = _read_json(path)
        if payload is None or not _looks_a2(payload, rel):
            continue
        scanned.append(rel)
        _check_authority_false(payload, rel, violations)
        _check_blockers(payload, rel, violations)
        for index, variant in enumerate(payload.get("variants") or []):
            if not isinstance(variant, dict):
                violations.append(f"{rel}: variants[{index}] must be an object")
                continue
            variant_rel = f"{rel}::variants[{index}]"
            _check_authority_false(variant, variant_rel, violations)
            _check_blockers(variant, variant_rel, violations)
    return {
        "schema": "a2_packet_ladder_closure_audit_v1",
        "scanned_artifacts": scanned,
        "violations": violations,
        "passed": not violations,
        "score_claim": False,
        "dispatch_attempted": False,
        "tracked_only": tracked_only,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Scan only git-tracked A2 manifests. Default also scans ignored local A2 artifacts.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.repo_root, tracked_only=args.tracked_only)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.strict and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
