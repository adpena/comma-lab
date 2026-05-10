#!/usr/bin/env python3
"""Audit A2 packet-ladder closure artifacts before exact-eval dispatch.

This is an operator-visible guard for the A2 hardening class: packet builders
and runtime probes may prove byte closure or parse closure, but they must keep
stub/proxy/no-score blockers alive until exact CUDA/CPU auth eval exists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

REQUIRED_FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
OPTIONAL_FALSE_AUTHORITY_FIELDS = {
    "score_claim_valid": False,
    "dispatch_attempted": False,
}
FALSE_AUTHORITY_FIELDS = {
    **REQUIRED_FALSE_AUTHORITY_FIELDS,
    **OPTIONAL_FALSE_AUTHORITY_FIELDS,
}
PROMOTABLE_EVIDENCE_GRADES = {
    "A",
    "A++",
    "contest-CPU-1to1",
    "contest-CUDA",
    "contest-CPU",
}

PACKET_LADDER_SCHEMAS = {
    "a2_sensitivity_weighted_pr101_packet_ladder.v1",
    "a2_sensitivity_weighted_pr101_packet_variant.v1",
}
RUNTIME_CLOSURE_SCHEMA = "a2_packet_runtime_closure_probe.v1"
SENSITIVITY_AUTHORITY_BLOCKERS = {
    "diagnostic_or_stub_sensitivity_map_not_score_authority",
    "score_sensitivity_artifact_must_be_certified_before_promotion",
}
SENSITIVITY_MARKER_TERMS = (
    "diagnostic",
    "proxy",
    "stub",
    "uniform",
)
SENSITIVITY_BLOCKER_TERMS = (
    "sensitivity",
    "stub",
    "proxy",
    "uniform",
    "cpu_local_allocator_proxy_only",
)
VALID_CLEARANCE_EVIDENCE = {
    "no_byte_closed_runtime_packet_built": {
        "packet_local_parse_smoke",
        "inflate_parity_log",
    },
    "packet_local_inflate_parity_not_run": {"inflate_parity_log"},
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
LOCAL_A2_EXACT_FILENAMES = frozenset(
    {
        "a2_packet_ladder_manifest.json",
        "candidate_manifest.json",
    }
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
    results_root = repo_root / "experiments" / "results"
    if not results_root.exists():
        return []
    paths: set[Path] = set()
    for dirpath, _dirnames, filenames in os.walk(results_root):
        base = Path(dirpath)
        for filename in filenames:
            if filename in LOCAL_A2_EXACT_FILENAMES or (
                filename.endswith(".json")
                and (
                    filename.startswith("a2_runtime_closure")
                    or filename.startswith("runtime_closure")
                )
            ):
                paths.add(base / filename)
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


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _contains_marker(value: Any, markers: tuple[str, ...]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    return any(marker in text for marker in markers)


def _sensitivity_artifacts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for artifact in (
        payload.get("sensitivity_artifact"),
        _nested_dict(payload, "upstream_a2_manifest", "sensitivity_artifact"),
        _nested_dict(payload, "candidate_manifest", "sensitivity_artifact"),
    ):
        if isinstance(artifact, dict) and artifact:
            artifacts.append(artifact)
    return artifacts


def _sensitivity_required_blockers(payload: dict[str, Any]) -> set[str]:
    required: set[str] = set()

    for source in (
        payload,
        _nested_dict(payload, "upstream_a2_manifest"),
        _nested_dict(payload, "candidate_manifest"),
    ):
        for blocker in _as_string_list(source.get("dispatch_blockers")) + _as_string_list(
            source.get("blockers")
        ):
            lowered = blocker.lower()
            if any(term in lowered for term in SENSITIVITY_BLOCKER_TERMS):
                required.add(blocker)

    for artifact in _sensitivity_artifacts(payload):
        metadata = artifact.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata_blockers = _as_string_list(artifact.get("metadata_blockers"))
        required.update(metadata_blockers)
        diagnostic = (
            bool(metadata_blockers)
            or artifact.get("allow_diagnostic_sensitivity") is True
            or _contains_marker(artifact.get("status"), SENSITIVITY_MARKER_TERMS)
            or _contains_marker(artifact.get("path"), SENSITIVITY_MARKER_TERMS)
            or _contains_marker(metadata.get("tag"), SENSITIVITY_MARKER_TERMS)
            or metadata.get("is_stub") is True
            or metadata.get("local_proxy") is True
            or metadata.get("proxy_only") is True
            or _contains_marker(metadata.get("source"), SENSITIVITY_MARKER_TERMS)
            or _contains_marker(metadata.get("role"), SENSITIVITY_MARKER_TERMS)
            or _contains_marker(metadata.get("format"), SENSITIVITY_MARKER_TERMS)
        )
        if diagnostic:
            required.update(SENSITIVITY_AUTHORITY_BLOCKERS)
    return required


def _check_authority_false(payload: dict[str, Any], rel: str, violations: list[str]) -> None:
    for field, expected in REQUIRED_FALSE_AUTHORITY_FIELDS.items():
        if payload.get(field) is not expected:
            violations.append(
                f"{rel}: {field} must be {expected!r} for A2 packet/probe artifacts "
                "until exact eval and blocker review close"
            )
    for field, expected in OPTIONAL_FALSE_AUTHORITY_FIELDS.items():
        if payload.get(field, expected) is not expected:
            violations.append(
                f"{rel}: {field} must be {expected!r} for A2 packet/probe artifacts "
                "until exact eval and blocker review close"
            )
    for field in ("evidence_grade", "score_evidence_grade"):
        grade = payload.get(field)
        if isinstance(grade, str) and grade.strip() in PROMOTABLE_EVIDENCE_GRADES:
            violations.append(
                f"{rel}: {field}={grade!r} is promotable false authority for an A2 "
                "packet/probe artifact while dispatch blockers remain"
            )


def _check_blockers(payload: dict[str, Any], rel: str, violations: list[str]) -> None:
    blockers = payload.get("dispatch_blockers")
    if not isinstance(blockers, list) or not blockers:
        violations.append(f"{rel}: dispatch_blockers must be a non-empty list")
        blockers = []
    schema = str(payload.get("schema", ""))
    if schema in PACKET_LADDER_SCHEMAS and not isinstance(
        payload.get("packet_closure"), dict
    ):
        violations.append(f"{rel}: packet_closure must be present for A2 packet artifacts")
    if schema == RUNTIME_CLOSURE_SCHEMA:
        if not isinstance(payload.get("runtime_closure"), dict):
            violations.append(f"{rel}: runtime_closure must be present for A2 runtime probes")
        candidate_manifest = payload.get("candidate_manifest")
        if not isinstance(candidate_manifest, dict) or not candidate_manifest.get("present"):
            violations.append(
                f"{rel}: runtime probe must preserve candidate_manifest evidence "
                "so inherited sensitivity/proxy blockers cannot be dropped"
            )
    closure = payload.get("packet_closure") or payload.get("runtime_closure")
    if not isinstance(closure, dict):
        return
    cleared = closure.get("cleared_blockers")
    byte_closed = any(
        closure.get(key) is True
        for key in (
            "byte_closed_packet_built",
            "byte_closed_packet_ladder_built",
            "verified",
        )
    )
    if cleared is None:
        if byte_closed:
            violations.append(
                f"{rel}: byte/runtime closure is marked true but cleared_blockers "
                "evidence is missing"
            )
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
        evidence_label = evidence.get(blocker)
        if not evidence_label:
            violations.append(
                f"{rel}: blocker {blocker!r} cleared without evidence label"
            )
            continue
        allowed = VALID_CLEARANCE_EVIDENCE.get(blocker)
        if allowed is not None and evidence_label not in allowed:
            violations.append(
                f"{rel}: blocker {blocker!r} cleared with evidence "
                f"{evidence_label!r}; expected one of {sorted(allowed)}"
            )
            continue
        if blocker == "packet_local_inflate_parity_not_run":
            parity = closure.get("inflate_parity_record")
            if not isinstance(parity, dict) or parity.get("passed") is not True:
                violations.append(
                    f"{rel}: blocker 'packet_local_inflate_parity_not_run' "
                    "cleared without inflate_parity_record.passed=true"
                )


def _check_sensitivity_blockers(
    payload: dict[str, Any],
    rel: str,
    violations: list[str],
    *,
    inherited_required: set[str] | None = None,
) -> set[str]:
    schema = str(payload.get("schema", ""))
    required = set(inherited_required or set()) | _sensitivity_required_blockers(payload)
    if schema in PACKET_LADDER_SCHEMAS and not _sensitivity_artifacts(payload):
        violations.append(
            f"{rel}: A2 packet artifact must preserve sensitivity_artifact "
            "provenance before it can be interpreted for dispatch readiness"
        )
    if not required:
        return required
    blockers = set(_as_string_list(payload.get("dispatch_blockers")))
    missing = sorted(required - blockers)
    if missing:
        violations.append(
            f"{rel}: sensitivity/proxy blockers missing from dispatch_blockers: {missing}"
        )
    closure = payload.get("runtime_closure")
    if isinstance(closure, dict) and isinstance(closure.get("remaining_blockers"), list):
        remaining = set(_as_string_list(closure.get("remaining_blockers")))
        missing_remaining = sorted(required - remaining)
        if missing_remaining:
            violations.append(
                f"{rel}: sensitivity/proxy blockers missing from "
                f"runtime_closure.remaining_blockers: {missing_remaining}"
            )
    return required


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
        inherited_sensitivity_blockers = _check_sensitivity_blockers(
            payload,
            rel,
            violations,
        )
        for index, variant in enumerate(payload.get("variants") or []):
            if not isinstance(variant, dict):
                violations.append(f"{rel}: variants[{index}] must be an object")
                continue
            variant_rel = f"{rel}::variants[{index}]"
            _check_authority_false(variant, variant_rel, violations)
            _check_blockers(variant, variant_rel, violations)
            _check_sensitivity_blockers(
                variant,
                variant_rel,
                violations,
                inherited_required=inherited_sensitivity_blockers,
            )
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
