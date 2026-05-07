#!/usr/bin/env python3
"""Build a deterministic field/meta dispatch-selection report.

The report consumes local candidate-packet manifests from any paradigm. It is a
selection artifact only: it does not claim scores, dispatch remote work, or
promote a candidate. Static candidate readiness is separate from
``ready_for_exact_eval_dispatch``: exact-eval dispatch readiness additionally
requires a matching active Level-2 lane claim for the manifest lane/job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file  # noqa: E402

SCHEMA_VERSION = 3
TOOL = "tools/build_field_meta_dispatch_selection.py"
STRICT_PREFLIGHT = "experiments/preflight_candidate_manifest_dispatch_readiness.py"
HEX_CHARS = set("0123456789abcdef")
PARETO_EPS = 1e-12
PROXY_EVIDENCE_GRADE_MARKERS = (
    "planning",
    "proxy",
    "prediction",
)
PARETO_MINIMIZE_OBJECTIVES = (
    "expected_total_score_delta",
    "byte_delta",
    "expected_seg_dist_delta",
    "expected_pose_dist_delta",
)
PARETO_MAXIMIZE_OBJECTIVES = (
    "confidence",
    "candidate_static_preflight_ready",
)
PARETO_OBJECTIVE_DIRECTIONS = {
    "expected_total_score_delta": "min",
    "byte_delta": "min",
    "expected_seg_dist_delta": "min",
    "expected_pose_dist_delta": "min",
    "confidence": "max",
    "candidate_static_preflight_ready": "max",
}
BLOCKER_CATEGORY_KEYS = (
    "custody",
    "runtime",
    "dispatch_claim",
    "kkt_or_admm",
    "pareto_frontier",
    "interaction_model",
    "proxy_or_planning",
    "dirty_worktree",
    "score_claim",
    "exact_cuda_eval",
    "strict_preflight",
    "other",
)
KKT_PROOF_PASS_STATUSES = {
    "pass",
    "passed",
    "ready",
    "verified",
}
KKT_PROOF_FAIL_STATUSES = {
    "blocked",
    "fail",
    "failed",
    "invalid",
    "missing",
    "not_ready",
}
SELECTION_PENALTIES = {
    "dirty_path_blocked": 10_000.0,
    "non_byte_closed_archive": 1_000.0,
    "non_byte_closed_runtime": 1_000.0,
    "strict_candidate_preflight_blocked": 500.0,
    "missing_dispatch_identity_for_lane_claim": 250.0,
    "missing_active_lane_dispatch_claim": 100.0,
    "planning_or_proxy_packet": 75.0,
    "pareto_ineligible_packet": 50.0,
    "kkt_not_ready_for_field_planning": 25.0,
    "pareto_dominated_packet": 10.0,
}
TERMINAL_CLAIM_STATUS_PREFIXES = (
    "completed_",
    "completed_score=",
    "completed_no_frontier",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)


def _strict_preflight_module(repo_root: Path) -> Any:
    script = repo_root / STRICT_PREFLIGHT
    spec = importlib.util.spec_from_file_location("field_meta_candidate_preflight", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import strict candidate preflight: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in HEX_CHARS for char in text)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _string_list(value):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _ordered_unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_nonempty_string(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> str:
    for path in paths:
        value = _nested(mapping, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_numeric(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]], default: float = 0.0) -> float:
    for path in paths:
        value = _nested(mapping, path)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return default


def _first_bool(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> bool | None:
    for path in paths:
        value = _nested(mapping, path)
        if isinstance(value, bool):
            return value
    return None


def _optional_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _resolve_local_path(value: Any, *, repo_root: Path, manifest_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest_dir / path


def _display_path(path: Path | None, *, repo_root: Path) -> str:
    if path is None:
        return ""
    return repo_relative(path, repo_root)


def _repo_relative_if_possible(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _path_matches(candidate: str, dirty_path: str) -> bool:
    candidate = candidate.strip()
    dirty_path = dirty_path.strip()
    if not candidate or not dirty_path:
        return False
    return (
        dirty_path == candidate
        or dirty_path.startswith(candidate.rstrip("/") + "/")
        or candidate.startswith(dirty_path.rstrip("/") + "/")
    )


def _path_values(value: Any) -> list[str]:
    out: list[str] = []
    for item in _string_list(value):
        text = item.strip()
        if text:
            out.append(text)
    return out


def _candidate_watch_paths(payload: Mapping[str, Any], *, manifest_path: Path, repo_root: Path) -> list[str]:
    paths: list[str] = [_repo_relative_if_possible(manifest_path, repo_root=repo_root)]
    for key in (
        "code_paths",
        "evidence_paths",
        "source_paths",
        "input_paths",
        "artifact_paths",
        "changed_paths",
        "runtime_paths",
    ):
        paths.extend(_path_values(payload.get(key)))
    for section_path, key in (
        (("tool_run_manifest",), "input_paths"),
        (("tool_run_manifest",), "output_paths"),
        (("candidate_bundle",), "code_paths"),
        (("candidate_bundle",), "evidence_paths"),
        (("dispatch_readiness",), "evidence_paths"),
        (("exact_eval_dispatch_gate",), "evidence_paths"),
        (("dispatch_gate",), "evidence_paths"),
    ):
        section = _nested(payload, section_path)
        if isinstance(section, Mapping):
            paths.extend(_path_values(section.get(key)))
    normalized: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_absolute():
            normalized.append(_repo_relative_if_possible(candidate, repo_root=repo_root))
        else:
            normalized.append(path)
    return _ordered_unique_strings(normalized)


def _dirty_paths_for_payload(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
    dirty_paths: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    watch_paths = _candidate_watch_paths(payload, manifest_path=manifest_path, repo_root=repo_root)
    matches: list[str] = []
    for dirty_path in dirty_paths or []:
        if any(_path_matches(path, str(dirty_path)) for path in watch_paths):
            matches.append(str(dirty_path))
    return sorted(set(matches)), watch_paths


def _unsafe_zip_member_name(name: str) -> str | None:
    if not name:
        return "zip_empty_member_name"
    pure = PurePosixPath(name)
    parts = pure.parts
    if pure.is_absolute() or name.startswith("/"):
        return "zip_absolute_member_name"
    if any(part in {"", ".", ".."} for part in parts):
        return "zip_slip_member_name"
    if any(part == "__MACOSX" or part.startswith("._") for part in parts):
        return "zip_resource_fork_member"
    if any(part in {".DS_Store", "Thumbs.db"} or part.startswith(".") for part in parts):
        return "zip_hidden_member"
    return None


def _local_header_name(path: Path, info: zipfile.ZipInfo) -> str | None:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return None
        flag_bits = int.from_bytes(header[6:8], "little")
        name_len = int.from_bytes(header[26:28], "little")
        raw_name = handle.read(name_len)
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw_name.decode(encoding)
    except UnicodeDecodeError:
        return None


def _zip_custody_proof(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "status": "blocked",
            "member_count": 0,
            "member_names": [],
            "blockers": ["archive_file_missing_for_zip_custody"],
        }
    blockers: list[str] = []
    names: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            if not names:
                blockers.append("zip_empty_archive")
            if len(names) != len(set(names)):
                blockers.append("zip_duplicate_members")
            for info in infos:
                central_name = info.filename
                unsafe = _unsafe_zip_member_name(central_name)
                if unsafe:
                    blockers.append(unsafe)
                local_name = _local_header_name(path, info)
                if local_name != central_name:
                    blockers.append("zip_local_header_name_mismatch")
    except (OSError, zipfile.BadZipFile, RuntimeError):
        blockers.append("archive_zip_unreadable")
    blockers = _unique_strings(blockers)
    return {
        "status": "passed" if not blockers else "blocked",
        "member_count": len(names),
        "member_names": names,
        "blockers": blockers,
    }


def _archive_identity_candidates(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    manifest_dir = manifest_path.parent
    archive = payload.get("archive") if isinstance(payload.get("archive"), Mapping) else {}
    archive_identity = (
        payload.get("archive_identity") if isinstance(payload.get("archive_identity"), Mapping) else {}
    )
    submission = payload.get("submission") if isinstance(payload.get("submission"), Mapping) else {}
    submission_archive = (
        submission.get("archive") if isinstance(submission.get("archive"), Mapping) else {}
    )
    artifact_relative = archive.get("artifact_relative_path")
    artifact_path = manifest_dir / str(artifact_relative) if artifact_relative else None
    candidates = [
        {
            "source": "candidate_archive_fields",
            "path_value": payload.get("candidate_archive_path"),
            "sha256": payload.get("candidate_archive_sha256"),
            "bytes": payload.get("candidate_archive_bytes"),
        },
        {
            "source": "archive_identity",
            "path_value": archive_identity.get("path"),
            "sha256": archive_identity.get("sha256"),
            "bytes": archive_identity.get("bytes"),
        },
        {
            "source": "archive_object",
            "path_value": archive.get("path") or (artifact_path.as_posix() if artifact_path else None),
            "sha256": archive.get("sha256"),
            "bytes": archive.get("bytes", archive.get("size_bytes")),
        },
        {
            "source": "submission_archive",
            "path_value": submission_archive.get("path"),
            "sha256": submission_archive.get("sha256"),
            "bytes": submission_archive.get("bytes", submission_archive.get("size_bytes")),
        },
        {
            "source": "root_archive_fields",
            "path_value": payload.get("archive_path"),
            "sha256": payload.get("archive_sha256"),
            "bytes": payload.get("archive_bytes", payload.get("archive_size_bytes")),
        },
    ]
    scored = []
    for index, candidate in enumerate(candidates):
        score = 0
        score += int(bool(candidate["path_value"])) * 4
        score += int(_is_sha256(candidate["sha256"])) * 2
        score += int(_coerce_positive_int(candidate["bytes"]) is not None)
        if score:
            scored.append({**candidate, "sort_key": (-score, index)})
    scored.sort(key=lambda item: item["sort_key"])
    return scored


def _archive_proof(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    candidates = _archive_identity_candidates(payload, manifest_path=manifest_path)
    if not candidates:
        return {
            "status": "blocked",
            "byte_closed": False,
            "source": "",
            "path": "",
            "sha256_expected": "",
            "sha256_actual": "",
            "bytes_expected": None,
            "bytes_actual": None,
            "zip_custody": _zip_custody_proof(None),
            "blockers": ["archive_identity_missing"],
        }
    selected = candidates[0]
    path = _resolve_local_path(
        selected.get("path_value"),
        repo_root=repo_root,
        manifest_dir=manifest_path.parent,
    )
    expected_sha = str(selected.get("sha256") or "")
    expected_bytes = _coerce_positive_int(selected.get("bytes"))
    blockers: list[str] = []
    actual_sha = ""
    actual_bytes: int | None = None
    zip_custody = _zip_custody_proof(None)
    if path is None:
        blockers.append("archive_path_missing")
    elif not path.is_file():
        blockers.append("archive_file_missing")
    if not _is_sha256(expected_sha):
        blockers.append("archive_sha256_missing_or_invalid")
    if expected_bytes is None:
        blockers.append("archive_bytes_missing_or_invalid")
    if path is not None and path.is_file():
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        zip_custody = _zip_custody_proof(path)
        blockers.extend(f"zip:{blocker}" for blocker in zip_custody["blockers"])
        if _is_sha256(expected_sha) and actual_sha != expected_sha:
            blockers.append("archive_sha256_mismatch")
        if expected_bytes is not None and actual_bytes != expected_bytes:
            blockers.append("archive_bytes_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "byte_closed": not blockers,
        "source": selected["source"],
        "path": _display_path(path, repo_root=repo_root),
        "sha256_expected": expected_sha,
        "sha256_actual": actual_sha,
        "bytes_expected": expected_bytes,
        "bytes_actual": actual_bytes,
        "zip_custody": zip_custody,
        "blockers": _unique_strings(blockers),
    }


def _runtime_tree_from_mapping(mapping: Mapping[str, Any]) -> str:
    for path in (
        ("runtime_tree_sha256",),
        ("runtime_manifest", "runtime_tree_sha256"),
        ("inflate_runtime_manifest", "runtime_tree_sha256"),
        ("provenance", "inflate_runtime_manifest", "runtime_tree_sha256"),
        ("provenance", "runtime_tree_sha256"),
        ("validated_fields", "provenance_fields", "inflate_runtime_manifest", "runtime_tree_sha256"),
        ("validated_fields", "provenance_fields", "runtime_tree_sha256"),
    ):
        value = _nested(mapping, path)
        if value:
            return str(value)
    return ""


def _section_ready(section: Mapping[str, Any], *ready_keys: str) -> bool:
    return any(section.get(key) is True for key in ready_keys)


def _runtime_candidate(
    *,
    source: str,
    section: Mapping[str, Any],
    ready: bool,
    blockers: Iterable[Any] = (),
) -> dict[str, Any]:
    runtime_tree = _runtime_tree_from_mapping(section)
    local_blockers = list(blockers)
    if not _is_sha256(runtime_tree):
        local_blockers.append("runtime_tree_sha256_missing_or_invalid")
    if not ready:
        local_blockers.append("runtime_proof_status_not_ready")
    return {
        "source": source,
        "runtime_tree_sha256": runtime_tree,
        "status": "passed" if not local_blockers else "blocked",
        "runtime_closed": not local_blockers,
        "blockers": _unique_strings(local_blockers),
    }


def _load_sibling_json(manifest_path: Path, name: str) -> dict[str, Any] | None:
    path = manifest_path.parent / name
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _runtime_proof_candidates(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    public_preflight = _load_sibling_json(manifest_path, "public_replay_preflight.json")
    if isinstance(public_preflight, Mapping):
        runtime = public_preflight.get("runtime")
        runtime_section = runtime if isinstance(runtime, Mapping) else {}
        candidates.append(
            _runtime_candidate(
                source="public_replay_preflight.runtime",
                section=runtime_section,
                ready=(
                    public_preflight.get("ready_for_exact_eval_dispatch") is True
                    and not public_preflight.get("blockers")
                ),
                blockers=public_preflight.get("blockers") or [],
            )
        )
    exact_runtime = payload.get("exact_eval_runtime_contract")
    if isinstance(exact_runtime, Mapping):
        candidates.append(
            _runtime_candidate(
                source="exact_eval_runtime_contract",
                section=exact_runtime,
                ready=_section_ready(exact_runtime, "ready_for_exact_eval_runtime"),
                blockers=exact_runtime.get("remaining_blockers") or exact_runtime.get("blockers") or [],
            )
        )
    fixed_runtime = payload.get("fixed_runtime_preflight")
    if isinstance(fixed_runtime, Mapping):
        candidates.append(
            _runtime_candidate(
                source="fixed_runtime_preflight",
                section=fixed_runtime,
                ready=_section_ready(
                    fixed_runtime,
                    "ready_for_fixed_runtime_exact_eval",
                    "ready_for_fixed_runtime_exact_eval_readiness",
                ),
                blockers=fixed_runtime.get("remaining_blockers") or fixed_runtime.get("blockers") or [],
            )
        )
    runtime = payload.get("runtime")
    if isinstance(runtime, Mapping):
        candidates.append(
            _runtime_candidate(
                source="manifest.runtime",
                section=runtime,
                ready=not runtime.get("blockers"),
                blockers=runtime.get("blockers") or [],
            )
        )
    for source, section in (
        ("inflate_runtime_manifest", payload.get("inflate_runtime_manifest")),
        ("contest_auth_eval", payload.get("contest_auth_eval")),
        ("provenance", payload.get("provenance")),
    ):
        if isinstance(section, Mapping):
            candidates.append(
                _runtime_candidate(
                    source=source,
                    section=section,
                    ready=True,
                    blockers=[],
                )
            )
    candidates.sort(key=lambda item: (item["status"] != "passed", item["source"]))
    return candidates


def _runtime_proof(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    candidates = _runtime_proof_candidates(payload, manifest_path=manifest_path)
    if not candidates:
        return {
            "status": "blocked",
            "runtime_closed": False,
            "source": "",
            "runtime_tree_sha256": "",
            "candidates": [],
            "blockers": ["runtime_proof_missing"],
        }
    selected = candidates[0]
    blockers = [] if selected["runtime_closed"] else selected["blockers"]
    return {
        "status": selected["status"],
        "runtime_closed": bool(selected["runtime_closed"]),
        "source": selected["source"],
        "runtime_tree_sha256": selected["runtime_tree_sha256"],
        "candidates": candidates,
        "blockers": _unique_strings(blockers),
    }


def _strict_candidate_preflight(
    manifest_path: Path,
    *,
    repo_root: Path,
    claims_path: Path | None,
    now_utc: str | None,
    ttl_hours: float,
) -> dict[str, Any]:
    module = _strict_preflight_module(repo_root)
    try:
        payload = module.build_preflight(
            manifest_path,
            claims_path=claims_path,
            now_utc=now_utc,
            ttl_hours=ttl_hours,
        )
    except SystemExit as exc:
        return {
            "schema": getattr(module, "SCHEMA", "candidate_manifest_dispatch_readiness_preflight_v1"),
            "candidate_static_preflight_ready": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": [{"code": "strict_candidate_preflight_error", "detail": str(exc)}],
            "warnings": [],
        }
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    return {
        "schema": payload.get("schema"),
        "candidate_static_preflight_ready": payload.get("ready_for_exact_eval_dispatch") is True,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "warnings": warnings,
        "lane_claim": payload.get("lane_claim"),
    }


def _parse_utc(value: str) -> dt.datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _is_terminal_claim_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_STATUS_PREFIXES)


def _parse_claim_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[0] in {"timestamp_utc", "---"}:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _manifest_lane_id(payload: Mapping[str, Any]) -> str:
    return _first_nonempty_string(
        payload,
        (
            ("lane_id",),
            ("dispatch_lane_id",),
            ("exact_eval_dispatch_lane_id",),
            ("exact_eval_dispatch_gate", "lane_id"),
            ("exact_eval_dispatch_gate", "claim", "lane_id"),
            ("dispatch_gate", "lane_id"),
            ("dispatch_gate", "claim", "lane_id"),
            ("dispatch", "lane_id"),
            ("dispatch", "claim", "lane_id"),
            ("dispatch_readiness", "lane_id"),
            ("dispatch_readiness", "claim", "lane_id"),
        ),
    )


def _manifest_instance_job_id(payload: Mapping[str, Any]) -> str:
    return _first_nonempty_string(
        payload,
        (
            ("instance_job_id",),
            ("job_name",),
            ("dispatch_job_name",),
            ("exact_eval_job_name",),
            ("exact_eval_instance_job_id",),
            ("exact_eval_dispatch_gate", "instance_job_id"),
            ("exact_eval_dispatch_gate", "job_name"),
            ("exact_eval_dispatch_gate", "claim", "instance_job_id"),
            ("exact_eval_dispatch_gate", "claim", "job_name"),
            ("dispatch_gate", "instance_job_id"),
            ("dispatch_gate", "job_name"),
            ("dispatch_gate", "claim", "instance_job_id"),
            ("dispatch_gate", "claim", "job_name"),
            ("dispatch", "instance_job_id"),
            ("dispatch", "job_name"),
            ("dispatch", "claim", "instance_job_id"),
            ("dispatch", "claim", "job_name"),
            ("dispatch_readiness", "instance_job_id"),
            ("dispatch_readiness", "job_name"),
            ("dispatch_readiness", "claim", "instance_job_id"),
            ("dispatch_readiness", "claim", "job_name"),
        ),
    )


def _dispatch_claim_proof(
    payload: Mapping[str, Any],
    *,
    claims_path: Path | None,
    now_utc: str | None,
    ttl_hours: float,
) -> dict[str, Any]:
    lane_id = _manifest_lane_id(payload)
    instance_job_id = _manifest_instance_job_id(payload)
    blockers: list[str] = []
    checked = claims_path is not None
    if claims_path is None:
        blockers.append("dispatch_claim_check_missing")
    elif not claims_path.is_file():
        blockers.append("dispatch_claims_file_missing")
    if not lane_id:
        blockers.append("dispatch_lane_id_missing")
    if not instance_job_id:
        blockers.append("dispatch_instance_job_id_missing")
    active_claim: dict[str, str] | None = None
    matching_claims: list[dict[str, str]] = []
    parsed_now = None
    if claims_path is not None:
        if now_utc is None:
            raise SystemExit(
                "--now-utc is required with --claims-path so dispatch readiness output is byte-reproducible"
            )
        parsed_now = _parse_utc(now_utc)
        if parsed_now is None:
            raise SystemExit(f"invalid --now-utc: {now_utc}")
    if checked and claims_path is not None and claims_path.is_file() and lane_id and instance_job_id:
        cutoff = parsed_now - dt.timedelta(hours=ttl_hours) if parsed_now is not None else None
        for claim in _parse_claim_rows(claims_path):
            if claim["lane_id"] != lane_id or claim["instance_job_id"] != instance_job_id:
                continue
            timestamp = _parse_utc(claim["timestamp_utc"])
            if timestamp is None or (cutoff is not None and timestamp < cutoff):
                continue
            matching_claims.append(claim)
        if matching_claims:
            matching_claims.sort(
                key=lambda claim: _parse_utc(claim["timestamp_utc"]) or dt.datetime.min.replace(tzinfo=dt.UTC),
                reverse=True,
            )
            latest = matching_claims[0]
            if _is_terminal_claim_status(latest["status"]):
                blockers.append("dispatch_claim_latest_status_terminal")
            else:
                active_claim = latest
        else:
            blockers.append("active_dispatch_claim_missing")
    active = active_claim is not None
    return {
        "status": "passed" if active else "blocked",
        "checked": checked,
        "claims_path": str(claims_path) if claims_path is not None else None,
        "now_utc": parsed_now.strftime("%Y-%m-%dT%H:%M:%SZ") if parsed_now is not None else None,
        "ttl_hours": ttl_hours,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "active_lane_claim": active,
        "active_claim": active_claim,
        "matching_claims": matching_claims,
        "blockers": _unique_strings(blockers),
    }


def _candidate_id(payload: Mapping[str, Any], manifest_path: Path) -> str:
    for key in ("candidate_id", "atom_id", "key", "lane_id", "job_name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return manifest_path.parent.name or manifest_path.stem


def _numeric_first(payload: Mapping[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return default


def _candidate_family(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    family = _first_nonempty_string(
        payload,
        (
            ("family",),
            ("atom_family",),
            ("family_group",),
            ("meta_lagrangian_atom", "family"),
            ("meta_lagrangian_atom", "family_group"),
            ("meta_lagrangian_atom_export", "atom_template", "family"),
            ("meta_lagrangian_atom_export", "atom_template", "family_group"),
            ("selected_target", "family"),
            ("selected_target", "family_group"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "family"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "family_group"),
        ),
    )
    family_group = _first_nonempty_string(
        payload,
        (
            ("family_group",),
            ("family",),
            ("atom_family",),
            ("meta_lagrangian_atom", "family_group"),
            ("meta_lagrangian_atom", "family"),
            ("meta_lagrangian_atom_export", "atom_template", "family_group"),
            ("meta_lagrangian_atom_export", "atom_template", "family"),
            ("selected_target", "family_group"),
            ("selected_target", "family"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "family_group"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "family"),
        ),
    )
    if not family:
        family = "unknown"
    if not family_group:
        family_group = family
    pareto_scope = _first_nonempty_string(
        payload,
        (
            ("pareto_scope",),
            ("meta_lagrangian_atom", "pareto_scope"),
            ("meta_lagrangian_atom_export", "atom_template", "pareto_scope"),
            ("selected_target", "pareto_scope"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "pareto_scope"),
        ),
    )
    if not pareto_scope:
        pareto_scope = family_group
    return family, family_group, pareto_scope


def _candidate_evidence_grade(payload: Mapping[str, Any]) -> str:
    grade = _first_nonempty_string(
        payload,
        (
            ("evidence_grade",),
            ("evidence_semantics",),
            ("meta_lagrangian_atom", "evidence_grade"),
            ("meta_lagrangian_atom_export", "atom_template", "evidence_grade"),
            ("selected_target", "evidence_grade"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "evidence_grade"),
        ),
    )
    return grade or "unknown"


def _candidate_interaction_assumptions(payload: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for path in (
        ("interaction_assumptions",),
        ("meta_lagrangian_atom", "interaction_assumptions"),
        ("meta_lagrangian_atom_export", "atom_template", "interaction_assumptions"),
        ("selected_target", "interaction_assumptions"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "interaction_assumptions"),
    ):
        values.extend(_string_list(_nested(payload, path)))
    return _ordered_unique_strings(values)


def _candidate_conflicts(payload: Mapping[str, Any], field: str) -> list[str]:
    values: list[Any] = []
    for path in (
        (field,),
        ("meta_lagrangian_atom", field),
        ("meta_lagrangian_atom_export", "atom_template", field),
        ("selected_target", field),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", field),
    ):
        values.extend(_string_list(_nested(payload, path)))
    return sorted({str(value) for value in values if str(value)})


def _first_int_field(
    payload: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> tuple[int | None, str, list[str]]:
    for path in paths:
        value = _nested(payload, path)
        source = ".".join(path)
        if value is None:
            continue
        if isinstance(value, bool):
            return None, source, [f"{source}_must_be_integer"]
        if isinstance(value, int):
            if value < 0:
                return None, source, [f"{source}_must_be_non_negative"]
            return value, source, []
        return None, source, [f"{source}_must_be_integer"]
    return None, "", []


def _first_string_field(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> str:
    for path in paths:
        value = _nested(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _candidate_volterra_terms(payload: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for path in (
        ("volterra_terms",),
        ("higher_order_interaction_terms",),
        ("meta_lagrangian_atom", "volterra_terms"),
        ("meta_lagrangian_atom", "higher_order_interaction_terms"),
        ("meta_lagrangian_atom_export", "atom_template", "volterra_terms"),
        ("meta_lagrangian_atom_export", "atom_template", "higher_order_interaction_terms"),
        ("selected_target", "volterra_terms"),
        ("selected_target", "higher_order_interaction_terms"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "volterra_terms"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "higher_order_interaction_terms"),
    ):
        values.extend(_string_list(_nested(payload, path)))
    return sorted({str(value) for value in values if str(value)})


def _field_interaction_contract(
    payload: Mapping[str, Any],
    *,
    assumptions: list[str],
    conflicts_with_families: list[str],
    conflicts_with_atoms: list[str],
) -> dict[str, Any]:
    volterra_order, volterra_order_source, volterra_blockers = _first_int_field(
        payload,
        (
            ("volterra_order",),
            ("interaction_order",),
            ("max_interaction_order",),
            ("meta_lagrangian_atom", "volterra_order"),
            ("meta_lagrangian_atom", "interaction_order"),
            ("meta_lagrangian_atom_export", "atom_template", "volterra_order"),
            ("meta_lagrangian_atom_export", "atom_template", "interaction_order"),
            ("selected_target", "volterra_order"),
            ("selected_target", "interaction_order"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "volterra_order"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "interaction_order"),
        ),
    )
    interaction_model = _first_string_field(
        payload,
        (
            ("interaction_model",),
            ("volterra_model",),
            ("field_interaction_model",),
            ("meta_lagrangian_atom", "interaction_model"),
            ("meta_lagrangian_atom", "volterra_model"),
            ("meta_lagrangian_atom_export", "atom_template", "interaction_model"),
            ("meta_lagrangian_atom_export", "atom_template", "volterra_model"),
            ("selected_target", "interaction_model"),
            ("selected_target", "volterra_model"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "interaction_model"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "volterra_model"),
        ),
    )
    blockers = list(volterra_blockers)
    if not assumptions:
        blockers.append("missing_interaction_assumptions")
    volterra_terms = _candidate_volterra_terms(payload)
    return {
        "schema": "field_interaction_contract_v1",
        "status": "passed" if not blockers else "blocked",
        "assumptions": assumptions,
        "assumption_count": len(assumptions),
        "conflicts_with_families": conflicts_with_families,
        "conflicts_with_atoms": conflicts_with_atoms,
        "interaction_model": interaction_model,
        "volterra_order": volterra_order,
        "volterra_order_source": volterra_order_source,
        "volterra_terms": volterra_terms,
        "volterra_terms_declared": bool(volterra_terms),
        "volterra_scope": (
            "declared_order"
            if volterra_order is not None
            else "undeclared_order_first_order_assumptions_only"
        ),
        "blockers": _unique_strings(blockers),
    }


def _candidate_proxy_row(payload: Mapping[str, Any], evidence_grade: str) -> bool:
    grade = evidence_grade.strip().lower()
    explicit_proxy = _first_bool(
        payload,
        (
            ("proxy_row",),
            ("planning_only",),
            ("meta_lagrangian_atom", "proxy_row"),
            ("meta_lagrangian_atom_export", "atom_template", "proxy_row"),
            ("selected_target", "planning_only"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "proxy_row"),
        ),
    )
    if explicit_proxy is True:
        return True
    return any(marker in grade for marker in PROXY_EVIDENCE_GRADE_MARKERS)


def _candidate_confidence(payload: Mapping[str, Any]) -> float:
    value = _first_numeric(
        payload,
        (
            ("confidence",),
            ("meta_lagrangian_atom", "confidence"),
            ("meta_lagrangian_atom_export", "atom_template", "confidence"),
            ("selected_target", "confidence"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "confidence"),
        ),
        default=1.0,
    )
    return max(0.0, min(1.0, value))


def _proof_status_passed(section: Mapping[str, Any]) -> bool | None:
    status = str(section.get("status") or section.get("state") or "").strip().lower()
    if status in KKT_PROOF_PASS_STATUSES:
        return True
    if status in KKT_PROOF_FAIL_STATUSES:
        return False
    for key in (
        "verified",
        "passed",
        "kkt_ready_for_field_planning",
        "ready_for_field_planning",
        "kkt_waterline_satisfied",
        "waterline_satisfied",
    ):
        value = section.get(key)
        if isinstance(value, bool):
            return value
    return None


def _proof_residual(section: Mapping[str, Any]) -> tuple[float | None, str]:
    for key in (
        "kkt_residual",
        "waterline_kkt_residual",
        "max_kkt_violation",
        "stationarity_residual",
        "primal_residual",
        "dual_residual",
    ):
        value = _optional_numeric(section.get(key))
        if value is not None:
            return value, key
    residuals = section.get("residuals")
    if isinstance(residuals, Mapping):
        numeric_values = [
            float(value)
            for value in residuals.values()
            if _optional_numeric(value) is not None
        ]
        if numeric_values:
            return max(abs(value) for value in numeric_values), "residuals"
    return None, ""


def _proof_tolerance(section: Mapping[str, Any]) -> tuple[float | None, str]:
    for key in (
        "kkt_tolerance",
        "kkt_tol",
        "waterline_kkt_tol",
        "kkt_waterline_tol",
        "max_kkt_violation_tolerance",
        "stationarity_tolerance",
    ):
        value = _optional_numeric(section.get(key))
        if value is not None:
            return value, key
    return None, ""


def _kkt_proof_candidate(source: str, section: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    passed = _proof_status_passed(section)
    residual, residual_source = _proof_residual(section)
    tolerance, tolerance_source = _proof_tolerance(section)
    if passed is False:
        blockers.append("kkt_proof_status_not_passed")
    if residual is None:
        blockers.append("kkt_proof_residual_missing")
    if tolerance is not None and residual is not None and abs(residual) > tolerance:
        blockers.append("kkt_proof_residual_exceeds_tolerance")
    if passed is None and not (residual is not None and tolerance is not None and abs(residual) <= tolerance):
        blockers.append("kkt_proof_pass_status_missing")
    return {
        "source": source,
        "kind": "kkt_proof",
        "status": "passed" if not blockers else "blocked",
        "residual": residual,
        "residual_source": residual_source,
        "tolerance": tolerance,
        "tolerance_source": tolerance_source,
        "blockers": _unique_strings(blockers),
    }


def _admm_result_candidate(
    source: str,
    section: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    config = config or {}
    converged = section.get("converged")
    if converged is not True:
        blockers.append("admm_result_not_converged")
    residual = _optional_numeric(
        section.get("waterline_kkt_residual", section.get("kkt_waterline_residual"))
    )
    tolerance = _optional_numeric(
        section.get(
            "kkt_waterline_tol",
            section.get(
                "waterline_kkt_tol",
                config.get("kkt_waterline_tol", config.get("waterline_kkt_tol")),
            ),
        )
    )
    satisfied = section.get("kkt_waterline_satisfied", section.get("waterline_satisfied"))
    if residual is None:
        blockers.append("admm_result_kkt_residual_missing")
    if satisfied is False:
        blockers.append("admm_result_kkt_waterline_not_satisfied")
    if tolerance is not None and residual is not None and abs(residual) > tolerance:
        blockers.append("admm_result_kkt_residual_exceeds_tolerance")
    if satisfied is not True and not (residual is not None and tolerance is not None and abs(residual) <= tolerance):
        blockers.append("admm_result_kkt_satisfaction_missing")
    return {
        "source": source,
        "kind": "admm_result",
        "status": "passed" if not blockers else "blocked",
        "converged": converged is True,
        "residual": residual,
        "residual_source": "waterline_kkt_residual" if "waterline_kkt_residual" in section else "kkt_waterline_residual",
        "tolerance": tolerance,
        "tolerance_source": "kkt_waterline_tol",
        "blockers": _unique_strings(blockers),
    }


def _mapping_candidates(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> list[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for path in paths:
        value = _nested(payload, path)
        if isinstance(value, Mapping):
            candidates.append((".".join(path), value))
    return candidates


def _kkt_proof(payload: Mapping[str, Any]) -> dict[str, Any]:
    proof_paths = (
        ("kkt_proof",),
        ("field_kkt_proof",),
        ("meta_lagrangian_kkt_proof",),
        ("meta_lagrangian_atom", "kkt_proof"),
        ("meta_lagrangian_atom_export", "kkt_proof"),
        ("meta_lagrangian_atom_export", "atom_template", "kkt_proof"),
        ("selected_target", "kkt_proof"),
        ("selected_target", "meta_lagrangian_atom_export", "kkt_proof"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "kkt_proof"),
    )
    admm_paths = (
        ("admm_result",),
        ("admm", "result"),
        ("joint_admm", "result"),
        ("meta_lagrangian_atom", "admm_result"),
        ("meta_lagrangian_atom_export", "admm_result"),
        ("meta_lagrangian_atom_export", "atom_template", "admm_result"),
        ("selected_target", "admm_result"),
        ("selected_target", "meta_lagrangian_atom_export", "admm_result"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "admm_result"),
    )
    candidates = [
        _kkt_proof_candidate(source, section)
        for source, section in _mapping_candidates(payload, proof_paths)
    ]
    admm_config = payload.get("admm_config")
    config = admm_config if isinstance(admm_config, Mapping) else {}
    candidates.extend(
        _admm_result_candidate(source, section, config=config)
        for source, section in _mapping_candidates(payload, admm_paths)
    )
    candidates.sort(key=lambda item: (item["status"] != "passed", item["source"]))
    if not candidates:
        return {
            "status": "blocked",
            "source": "",
            "kind": "",
            "residual": None,
            "tolerance": None,
            "candidates": [],
            "blockers": ["kkt_proof_or_admm_result_missing"],
        }
    selected = candidates[0]
    return {
        "status": selected["status"],
        "source": selected["source"],
        "kind": selected["kind"],
        "residual": selected.get("residual"),
        "tolerance": selected.get("tolerance"),
        "candidates": candidates,
        "blockers": list(selected["blockers"]),
    }


def _byte_delta(payload: Mapping[str, Any]) -> int:
    for key in (
        "candidate_archive_byte_delta_vs_source_estimate",
        "section_byte_delta",
        "byte_delta",
    ):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for path in (
        ("meta_lagrangian_atom", "byte_delta"),
        ("meta_lagrangian_atom", "estimated_byte_delta"),
        ("meta_lagrangian_atom_export", "atom_template", "byte_delta"),
        ("meta_lagrangian_atom_export", "atom_template", "estimated_byte_delta"),
        ("selected_target", "byte_delta"),
        ("selected_target", "estimated_byte_delta"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "byte_delta"),
        ("selected_target", "meta_lagrangian_atom_export", "atom_template", "estimated_byte_delta"),
    ):
        value = _nested(payload, path)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    archive_bytes = _coerce_positive_int(payload.get("archive_bytes", payload.get("candidate_archive_bytes")))
    source_bytes = _coerce_positive_int(payload.get("source_archive_bytes"))
    if archive_bytes is not None and source_bytes is not None:
        return archive_bytes - source_bytes
    return 0


def _hnerv_lowlevel_total_byte_delta(payload: Mapping[str, Any]) -> int | None:
    audit = payload.get("candidate_diff_audit")
    if isinstance(audit, Mapping):
        value = audit.get("total_byte_delta")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    archive_bytes = _coerce_positive_int(payload.get("candidate_archive_bytes"))
    source_bytes = _coerce_positive_int(payload.get("source_archive_bytes"))
    if archive_bytes is not None and source_bytes is not None:
        return archive_bytes - source_bytes
    return None


def _hnerv_lowlevel_candidate_id(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> str:
    attempts = payload.get("attempts")
    accepted_attempts = (
        [
            attempt
            for attempt in attempts
            if isinstance(attempt, Mapping) and attempt.get("accepted_for_candidate") is True
        ]
        if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes, bytearray))
        else []
    )
    lgblocks = {
        int(attempt["lgblock"])
        for attempt in accepted_attempts
        if isinstance(attempt.get("lgblock"), int) and not isinstance(attempt.get("lgblock"), bool)
    }
    byte_delta = _hnerv_lowlevel_total_byte_delta(payload)
    source_label = str(payload.get("source_label") or "").lower()
    if "pr106x" in source_label and lgblocks == {16} and byte_delta == -1:
        return "pr106x_lgblock16_lowlevel_brotli_1byte"
    return manifest_path.parent.name or manifest_path.stem


def _normalize_hnerv_lowlevel_result_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any] | None:
    if payload.get("tool") != "tac.hnerv_lowlevel_packer.build_lowlevel_brotli_repack_candidate":
        return None
    normalized = dict(payload)
    normalized.setdefault(
        "candidate_id",
        _hnerv_lowlevel_candidate_id(payload, manifest_path=manifest_path),
    )
    normalized.setdefault("family", "hnerv_lowlevel_brotli_repack")
    normalized.setdefault("family_group", "hnerv_lowlevel_brotli_repack")
    normalized.setdefault("pareto_scope", "hnerv_lowlevel_brotli_repack")
    normalized.setdefault(
        "evidence_grade",
        "empirical local archive candidate; raw-equivalence audit only until exact CUDA",
    )
    normalized.setdefault(
        "interaction_assumptions",
        [
            "rate_only_raw_equivalent_brotli_repack",
            "component_deltas_require_exact_cuda_confirmation",
        ],
    )
    byte_delta = _hnerv_lowlevel_total_byte_delta(payload)
    if byte_delta is not None:
        normalized.setdefault("byte_delta", byte_delta)
    audit = payload.get("candidate_diff_audit")
    if isinstance(audit, Mapping):
        score_delta = audit.get("rate_score_delta_if_components_equal")
        if isinstance(score_delta, int | float) and not isinstance(score_delta, bool):
            normalized.setdefault("expected_total_score_delta", float(score_delta))
    normalized.setdefault("expected_seg_dist_delta", 0.0)
    normalized.setdefault("expected_pose_dist_delta", 0.0)
    normalized.setdefault("expected_information_gain_nats", 0.0)
    artifact_paths = [
        *(_path_values(normalized.get("artifact_paths"))),
        str(normalized.get("candidate_archive_path") or ""),
        str(normalized.get("source_archive_path") or ""),
        repo_relative(manifest_path, repo_root),
    ]
    normalized["artifact_paths"] = _ordered_unique_strings(
        path for path in artifact_paths if path
    )
    return normalized


def _normalize_manifest_payload(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    hnerv_lowlevel = _normalize_hnerv_lowlevel_result_manifest(
        payload,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    if hnerv_lowlevel is not None:
        return hnerv_lowlevel
    return dict(payload)


def _operator_next_steps(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("operator_next_steps")
    return value if isinstance(value, Mapping) else {}


def _operator_step_sort_key(step: Mapping[str, Any]) -> tuple[int, str]:
    order = step.get("order")
    if isinstance(order, bool) or not isinstance(order, int):
        order = 10_000
    return order, str(step.get("id") or step.get("purpose") or "")


def _operator_steps(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    section = _operator_next_steps(payload)
    raw_steps = section.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes, bytearray)):
        return []
    steps = [step for step in raw_steps if isinstance(step, Mapping)]
    return sorted(steps, key=_operator_step_sort_key)


def _command_step_summary(
    *,
    source: str,
    step_id: str,
    command: str,
    order: int | None = None,
    purpose: str = "",
    writes_repo_state: bool | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "id": step_id,
        "order": order,
        "purpose": purpose,
        "command": command,
        "copy_safe_command": command,
        "dispatches_remote_gpu": False,
        "writes_repo_state": writes_repo_state,
    }


def _step_command(step: Mapping[str, Any]) -> str:
    for key in ("copy_safe_command", "command", "local_command"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _step_summary(step: Mapping[str, Any]) -> dict[str, Any]:
    order = step.get("order")
    return _command_step_summary(
        source="operator_next_steps.steps",
        step_id=str(step.get("id") or ""),
        order=order if isinstance(order, int) and not isinstance(order, bool) else None,
        purpose=str(step.get("purpose") or ""),
        command=_step_command(step),
        writes_repo_state=(
            step.get("writes_repo_state")
            if isinstance(step.get("writes_repo_state"), bool)
            else None
        ),
    )


def _next_local_non_gpu_action(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    for step in _operator_steps(payload):
        if step.get("dispatches_remote_gpu") is True:
            continue
        command = _step_command(step)
        if command:
            return _step_summary(step)
    for key in ("next_local_non_gpu_command", "next_local_command"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _command_step_summary(
                source=key,
                step_id=key,
                command=value.strip(),
            )
    commands = payload.get("commands")
    if isinstance(commands, Mapping):
        for key in ("preflight", "refresh", "claim", "assert", "build", "harvest"):
            value = commands.get(key)
            if isinstance(value, str) and value.strip():
                return _command_step_summary(
                    source=f"commands.{key}",
                    step_id=key,
                    command=value.strip(),
                )
    return None


def _operator_current_blockers(payload: Mapping[str, Any]) -> list[str]:
    section = _operator_next_steps(payload)
    blockers: list[Any] = []
    for key in (
        "current_blockers",
        "current_submit_blockers",
        "current_dispatch_blockers",
        "blockers",
    ):
        blockers.extend(_string_list(section.get(key)))
    for key in (
        "blockers",
        "dispatch_blockers",
        "approval_blockers",
        "refresh_blockers",
        "static_blockers",
    ):
        blockers.extend(_string_list(payload.get(key)))
    return _unique_strings(blockers)


def _operator_claim_blockers(
    payload: Mapping[str, Any],
    *,
    current_blockers: Sequence[str],
) -> list[str]:
    blockers = [
        blocker
        for blocker in current_blockers
        if any(token in blocker.lower() for token in ("claim", "lane"))
    ]
    lane_claim = payload.get("lane_claim_preflight")
    if isinstance(lane_claim, Mapping):
        if lane_claim.get("active_claim_present") is not True:
            blockers.append("missing_active_lane_dispatch_claim")
        if lane_claim.get("claims_path_exists") is False:
            blockers.append("dispatch_claims_file_missing")
        if lane_claim.get("conflict_present") is True:
            blockers.append("active_lane_dispatch_claim_conflict")
    return _unique_strings(blockers)


def _operator_refresh_blockers(
    payload: Mapping[str, Any],
    *,
    current_blockers: Sequence[str],
) -> list[str]:
    blockers = [
        blocker
        for blocker in current_blockers
        if "refresh" in blocker.lower() or "static" in blocker.lower()
    ]
    if payload.get("static_packet_ready") is False:
        blockers.append("static_packet_ready_false")
    blockers.extend(_string_list(payload.get("static_blockers")))
    blockers.extend(_string_list(payload.get("refresh_blockers")))
    refresh, _ = _static_compliance_refresh(payload)
    if refresh:
        returncode = refresh.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0:
            blockers.append("static_compliance_refresh_failed")
    return _unique_strings(blockers)


def _operator_approval_blockers(
    payload: Mapping[str, Any],
    *,
    current_blockers: Sequence[str],
) -> list[str]:
    blockers = [
        blocker
        for blocker in current_blockers
        if "approval" in blocker.lower() or "approved" in blocker.lower()
    ]
    if payload.get("operator_approved_exact_cuda") is False:
        blockers.append("missing_operator_exact_cuda_approval")
    blockers.extend(_string_list(payload.get("approval_blockers")))
    return _unique_strings(blockers)


def _operator_environment_blockers(
    payload: Mapping[str, Any],
    *,
    current_blockers: Sequence[str],
) -> list[str]:
    blockers = [
        blocker
        for blocker in current_blockers
        if "environment" in blocker.lower() or "env" in blocker.lower()
    ]
    blockers.extend(f"missing_env:{name}" for name in _string_list(payload.get("missing_env")))
    return _unique_strings(blockers)


def _static_compliance_refresh(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    refresh = payload.get("static_compliance_refresh")
    if isinstance(refresh, Mapping):
        return refresh, "static_compliance_refresh"
    refreshes = payload.get("refreshes")
    if isinstance(refreshes, Mapping):
        nested = refreshes.get("static_compliance")
        if isinstance(nested, Mapping):
            return nested, "refreshes.static_compliance"
    return {}, ""


def _static_refresh_status(payload: Mapping[str, Any]) -> str:
    refresh, _ = _static_compliance_refresh(payload)
    if not refresh:
        return "not_recorded"
    returncode = refresh.get("returncode")
    if isinstance(returncode, int) and not isinstance(returncode, bool):
        return "passed" if returncode == 0 else "failed"
    return "unknown"


def _static_refresh_command(payload: Mapping[str, Any]) -> str:
    refresh, _ = _static_compliance_refresh(payload)
    if not refresh:
        return ""
    command = refresh.get("command")
    return command.strip() if isinstance(command, str) else ""


def _static_refresh_schema(payload: Mapping[str, Any]) -> str:
    refresh, _ = _static_compliance_refresh(payload)
    schema = refresh.get("schema") if refresh else ""
    return str(schema or "")


def _static_refresh_source(payload: Mapping[str, Any]) -> str:
    _, source = _static_compliance_refresh(payload)
    return source


def _operator_next_steps_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    section = _operator_next_steps(payload)
    steps = _operator_steps(payload)
    current_blockers = _operator_current_blockers(payload)
    next_local_action = _next_local_non_gpu_action(payload)
    local_step_ids = [
        str(step.get("id") or "")
        for step in steps
        if step.get("dispatches_remote_gpu") is not True and str(step.get("id") or "")
    ]
    remote_step_ids = [
        str(step.get("id") or "")
        for step in steps
        if step.get("dispatches_remote_gpu") is True and str(step.get("id") or "")
    ]
    return {
        "schema": "packet_operator_next_steps_summary_v1",
        "source": "operator_next_steps" if section else "manifest_command_fields",
        "packet_operator_next_steps_schema": str(section.get("schema") or ""),
        "packet_path": str(section.get("packet_path") or ""),
        "copy_safe": section.get("copy_safe") if isinstance(section.get("copy_safe"), bool) else None,
        "must_run_in_order": (
            section.get("must_run_in_order")
            if isinstance(section.get("must_run_in_order"), bool)
            else None
        ),
        "step_count": len(steps),
        "local_non_gpu_step_ids": local_step_ids,
        "remote_gpu_step_ids": remote_step_ids,
        "first_remote_gpu_step": str(section.get("first_remote_gpu_step") or ""),
        "current_blockers": current_blockers,
        "claim_blockers": _operator_claim_blockers(
            payload,
            current_blockers=current_blockers,
        ),
        "refresh_blockers": _operator_refresh_blockers(
            payload,
            current_blockers=current_blockers,
        ),
        "approval_blockers": _operator_approval_blockers(
            payload,
            current_blockers=current_blockers,
        ),
        "environment_blockers": _operator_environment_blockers(
            payload,
            current_blockers=current_blockers,
        ),
        "static_refresh_status": _static_refresh_status(payload),
        "static_refresh_command": _static_refresh_command(payload),
        "static_refresh_schema": _static_refresh_schema(payload),
        "static_refresh_source": _static_refresh_source(payload),
        "next_local_non_gpu_action": next_local_action,
        "next_local_non_gpu_command": (
            str(next_local_action.get("command") or "") if next_local_action else ""
        ),
    }


def _next_required_proofs(blockers: Sequence[str]) -> list[str]:
    out: list[str] = []
    if "dirty_worktree_overlap" in blockers:
        out.append("clean_or_isolate_dirty_candidate_paths_before_selection")
    if "missing_byte_closed_archive_proof" in blockers:
        out.append("local_archive_file_with_matching_sha256_and_bytes")
    if "missing_byte_closed_runtime_proof" in blockers:
        out.append("runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract")
    if "strict_candidate_preflight_not_ready" in blockers:
        out.append(f"passing_{STRICT_PREFLIGHT}")
    if "missing_dispatch_identity_for_lane_claim" in blockers:
        out.append("manifest_lane_id_and_instance_job_id_for_level2_claim")
    if "missing_active_lane_dispatch_claim" in blockers:
        out.append("matching_active_level2_lane_claim_for_manifest_lane_and_job")
    if not out:
        out.append("exact_cuda_auth_eval_on_selected_archive_bytes")
    return out


def _dispatch_identity_proof(claim: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not claim.get("lane_id"):
        blockers.append("dispatch_lane_id_missing")
    if not claim.get("instance_job_id"):
        blockers.append("dispatch_instance_job_id_missing")
    return {
        "status": "passed" if not blockers else "blocked",
        "lane_id": str(claim.get("lane_id") or ""),
        "instance_job_id": str(claim.get("instance_job_id") or ""),
        "blockers": blockers,
    }


def _pareto_eligibility_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("candidate_static_preflight_ready") is not True:
        blockers.append("candidate_static_preflight_not_ready")
    if row.get("proxy_row") is True:
        blockers.append("planning_or_proxy_packet")
    if row.get("dirty_blocked") is True:
        blockers.append("dirty_worktree_overlap")
    return blockers


def _pareto_objectives(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "expected_total_score_delta": float(row["expected_total_score_delta"]),
        "byte_delta": float(row["byte_delta"]),
        "expected_seg_dist_delta": float(row["expected_seg_dist_delta"]),
        "expected_pose_dist_delta": float(row["expected_pose_dist_delta"]),
        "confidence": float(row["confidence"]),
        "candidate_static_preflight_ready": float(bool(row["candidate_static_preflight_ready"])),
    }


def _non_dominated_frontier_reason(row: Mapping[str, Any]) -> dict[str, Any]:
    eligibility_blockers = _pareto_eligibility_blockers(row)
    dominated_by = sorted(str(value) for value in _string_list(row.get("pareto_dominated_by")))
    if eligibility_blockers:
        status = "ineligible"
        reason = "not_on_frontier_until_pareto_eligibility_blockers_clear"
    elif dominated_by:
        status = "dominated"
        reason = "dominated_within_pareto_scope_by_static_ready_non_proxy_candidate"
    elif row.get("pareto_frontier") is True:
        status = "non_dominated"
        reason = "non_dominated_within_pareto_scope"
    else:
        status = "blocked"
        reason = "pareto_frontier_annotation_missing_or_blocked"
    return {
        "schema": "non_dominated_frontier_reason_v1",
        "status": status,
        "reason": reason,
        "scope": str(row.get("pareto_scope") or ""),
        "eligible": bool(row.get("pareto_eligible")),
        "frontier": bool(row.get("pareto_frontier")),
        "dominated_by": dominated_by,
        "eligibility_blockers": eligibility_blockers,
        "objective_directions": dict(PARETO_OBJECTIVE_DIRECTIONS),
        "objectives": dict(row.get("pareto_objectives") or {}),
    }


def _dominates_pareto(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get("pareto_eligible") is not True or b.get("pareto_eligible") is not True:
        return False
    if str(a["pareto_scope"]) != str(b["pareto_scope"]):
        return False
    strictly_better = False
    for objective in PARETO_MINIMIZE_OBJECTIVES:
        av = float(a[objective])
        bv = float(b[objective])
        if av > bv + PARETO_EPS:
            return False
        strictly_better = strictly_better or av < bv - PARETO_EPS
    for objective in PARETO_MAXIMIZE_OBJECTIVES:
        av = float(a[objective])
        bv = float(b[objective])
        if av < bv - PARETO_EPS:
            return False
        strictly_better = strictly_better or av > bv + PARETO_EPS
    return strictly_better


def _annotate_pareto_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frontier_count = 0
    dominated_count = 0
    scope_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        row["pareto_frontier"] = bool(row["pareto_eligible"])
        row["pareto_dominated_by"] = []
        row["pareto_objectives"] = _pareto_objectives(row)
        row["pareto_eligibility_blockers"] = _pareto_eligibility_blockers(row)
    for row in rows:
        scope = str(row["pareto_scope"])
        stats = scope_counts.setdefault(
            scope,
            {"eligible": 0, "frontier": 0, "dominated": 0, "ineligible": 0},
        )
        if row.get("pareto_eligible") is not True:
            row["pareto_frontier"] = False
            stats["ineligible"] += 1
            continue
        stats["eligible"] += 1
        dominators = [
            str(other["candidate_id"])
            for other in rows
            if other is not row and _dominates_pareto(other, row)
        ]
        if dominators:
            row["pareto_frontier"] = False
            row["pareto_dominated_by"] = sorted(dominators)
            dominated_count += 1
            stats["dominated"] += 1
        else:
            frontier_count += 1
            stats["frontier"] += 1
    _annotate_row_explanations(rows)
    return {
        "objective_direction": dict(PARETO_OBJECTIVE_DIRECTIONS),
        "scope_default": "family_group",
        "eligibility": (
            "candidate_static_preflight_ready_and_non_proxy_and_not_dirty; "
            "dispatch readiness still additionally requires an active lane claim"
        ),
        "frontier_count": frontier_count,
        "dominated_count": dominated_count,
        "eligible_count": sum(int(row.get("pareto_eligible") is True) for row in rows),
        "scope_counts": dict(sorted(scope_counts.items())),
    }


def _blocker_categories(blockers: Iterable[Any]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {key: [] for key in BLOCKER_CATEGORY_KEYS}
    for blocker in _unique_strings(blockers):
        lowered = blocker.lower()
        if any(token in lowered for token in ("archive", "byte_closed", "manifest", "custody", "zip")):
            categories["custody"].append(blocker)
        elif "runtime" in lowered:
            categories["runtime"].append(blocker)
        elif "claim" in lowered or "lane" in lowered or "identity" in lowered:
            categories["dispatch_claim"].append(blocker)
        elif "kkt" in lowered or "admm" in lowered:
            categories["kkt_or_admm"].append(blocker)
        elif "pareto" in lowered or "dominated" in lowered or "frontier" in lowered:
            categories["pareto_frontier"].append(blocker)
        elif "interaction" in lowered or "volterra" in lowered or "conflict" in lowered:
            categories["interaction_model"].append(blocker)
        elif "proxy" in lowered or "planning" in lowered:
            categories["proxy_or_planning"].append(blocker)
        elif "dirty" in lowered or "worktree" in lowered:
            categories["dirty_worktree"].append(blocker)
        elif "score_claim" in lowered:
            categories["score_claim"].append(blocker)
        elif "cuda" in lowered or "auth_eval" in lowered or "exact_eval" in lowered:
            categories["exact_cuda_eval"].append(blocker)
        elif "strict" in lowered or "preflight" in lowered:
            categories["strict_preflight"].append(blocker)
        else:
            categories["other"].append(blocker)
    return categories


def _exact_dispatch_blockers(row: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = [
        *row.get("candidate_blockers", []),
        *row.get("kkt_blockers", []),
        *row.get("selection_blockers", []),
    ]
    if row.get("pareto_eligible") is not True:
        blockers.append("pareto_ineligible_for_field_selection")
    elif row.get("pareto_frontier") is not True:
        blockers.append("pareto_dominated_within_scope")
    if row.get("field_interaction_contract", {}).get("status") != "passed":
        blockers.append("field_interaction_contract_blocked")
    blockers = _unique_strings(blockers)
    rigorous_ready = bool(row.get("ready_for_exact_eval_dispatch") and not blockers)
    next_required_proof = list(row.get("next_required_proof") or [])
    if row.get("kkt_ready_for_field_planning") is not True:
        next_required_proof.append("passed_kkt_proof_or_converged_admm_waterline_result")
    if row.get("pareto_eligible") is not True:
        next_required_proof.append("pareto_eligible_static_ready_non_proxy_candidate")
    elif row.get("pareto_frontier") is not True:
        next_required_proof.append("non_dominated_candidate_or_explicit_pareto_scope_override")
    if row.get("field_interaction_contract", {}).get("status") != "passed":
        next_required_proof.append("explicit_interaction_assumptions_or_volterra_scope")
    return {
        "schema": "exact_dispatch_blockers_v1",
        "ready_for_exact_eval_dispatch": rigorous_ready,
        "candidate_preflight_ready_for_exact_eval_dispatch": bool(row.get("ready_for_exact_eval_dispatch")),
        "dispatch_refused": not rigorous_ready,
        "score_claim": False,
        "dispatch_attempted": False,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "blocker_categories": _blocker_categories(blockers),
        "next_required_proof": _unique_strings(next_required_proof),
    }


def _annotate_row_explanations(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        row["pareto_eligibility_blockers"] = _pareto_eligibility_blockers(row)
        row["non_dominated_frontier_reason"] = _non_dominated_frontier_reason(row)
        row["exact_dispatch_blockers"] = _exact_dispatch_blockers(row)
        row["field_selection_ready_for_exact_eval_dispatch"] = bool(
            row["exact_dispatch_blockers"]["ready_for_exact_eval_dispatch"]
        )


def _selection_penalty_terms(row: Mapping[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    if row.get("dirty_blocked") is True:
        terms["dirty_path_blocked"] = SELECTION_PENALTIES["dirty_path_blocked"]
    if row.get("archive_proof", {}).get("byte_closed") is not True:
        terms["non_byte_closed_archive"] = SELECTION_PENALTIES["non_byte_closed_archive"]
    if row.get("runtime_proof", {}).get("runtime_closed") is not True:
        terms["non_byte_closed_runtime"] = SELECTION_PENALTIES["non_byte_closed_runtime"]
    if row.get("strict_candidate_preflight_ready") is not True:
        terms["strict_candidate_preflight_blocked"] = SELECTION_PENALTIES[
            "strict_candidate_preflight_blocked"
        ]
    if row.get("dispatch_identity_proof", {}).get("status") != "passed":
        terms["missing_dispatch_identity_for_lane_claim"] = SELECTION_PENALTIES[
            "missing_dispatch_identity_for_lane_claim"
        ]
    if row.get("candidate_static_preflight_ready") is True and row.get("ready_for_exact_eval_dispatch") is not True:
        terms["missing_active_lane_dispatch_claim"] = SELECTION_PENALTIES[
            "missing_active_lane_dispatch_claim"
        ]
    if row.get("proxy_row") is True:
        terms["planning_or_proxy_packet"] = SELECTION_PENALTIES["planning_or_proxy_packet"]
    if row.get("pareto_eligible") is not True:
        terms["pareto_ineligible_packet"] = SELECTION_PENALTIES["pareto_ineligible_packet"]
    elif row.get("pareto_frontier") is not True:
        terms["pareto_dominated_packet"] = SELECTION_PENALTIES["pareto_dominated_packet"]
    if row.get("kkt_ready_for_field_planning") is not True:
        terms["kkt_not_ready_for_field_planning"] = SELECTION_PENALTIES[
            "kkt_not_ready_for_field_planning"
        ]
    return {key: terms[key] for key in sorted(terms)}


def _selection_decision(row: Mapping[str, Any]) -> str:
    if row.get("ready_for_exact_eval_dispatch") is True:
        return "ready_for_exact_eval_dispatch_after_active_claim"
    if row.get("dirty_blocked") is True:
        return "refused_dirty_worktree_overlap"
    if row.get("candidate_static_preflight_ready") is True:
        return "needs_active_lane_claim_before_dispatch"
    if row.get("candidate_local_preflight_ready") is True:
        return "needs_dispatch_identity_before_lane_claim"
    if row.get("archive_proof", {}).get("byte_closed") is not True:
        return "needs_byte_closed_archive_proof"
    if row.get("runtime_proof", {}).get("runtime_closed") is not True:
        return "needs_byte_closed_runtime_proof"
    if row.get("strict_candidate_preflight_ready") is not True:
        return "strict_candidate_preflight_refused"
    return "blocked_until_required_proofs_land"


def _lexicographic_feasibility_tuple(row: Mapping[str, Any]) -> list[Any]:
    return [
        bool(row["ready_for_exact_eval_dispatch"]),
        bool(row["candidate_static_preflight_ready"]),
        bool(row["archive_proof"]["byte_closed"]),
        bool(row["runtime_proof"]["runtime_closed"]),
        not bool(row["dirty_blocked"]),
        bool(row["pareto_frontier"]),
        bool(row["kkt_ready_for_field_planning"]),
        round(float(row["expected_total_score_delta"]), 12),
        round(float(row["expected_information_gain_nats"]), 12),
    ]


def _lexicographic_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    values = row["lexicographic_feasibility_tuple"]
    return (
        not bool(values[0]),
        not bool(values[1]),
        not bool(values[2]),
        not bool(values[3]),
        not bool(values[4]),
        not bool(values[5]),
        not bool(values[6]),
        float(values[7]),
        -float(values[8]),
        str(row["manifest_path"]),
    )


def _annotate_selection_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        terms = _selection_penalty_terms(row)
        penalty = round(sum(terms.values()), 12)
        row["selection_penalty_terms"] = {key: round(value, 12) for key, value in terms.items()}
        row["selection_penalty_units"] = penalty
        row["field_selection_score_delta"] = round(float(row["expected_total_score_delta"]), 12)
        row["selection_blockers"] = list(terms)
        row["selection_decision"] = _selection_decision(row)
        row["dispatch_refused"] = row.get("ready_for_exact_eval_dispatch") is not True
        row["lexicographic_feasibility_tuple"] = _lexicographic_feasibility_tuple(row)


def _row_for_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    claims_path: Path | None,
    now_utc: str | None,
    ttl_hours: float,
    dirty_paths: Sequence[str] | None,
) -> dict[str, Any]:
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"packet manifest must be a JSON object: {manifest_path}")
    payload = _normalize_manifest_payload(
        payload,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    manifest_sha256 = sha256_file(manifest_path)
    archive = _archive_proof(payload, manifest_path=manifest_path, repo_root=repo_root)
    runtime = _runtime_proof(payload, manifest_path=manifest_path)
    strict = _strict_candidate_preflight(
        manifest_path,
        repo_root=repo_root,
        claims_path=None,
        now_utc=None,
        ttl_hours=ttl_hours,
    )
    claim = _dispatch_claim_proof(
        payload,
        claims_path=claims_path,
        now_utc=now_utc,
        ttl_hours=ttl_hours,
    )
    identity = _dispatch_identity_proof(claim)
    static_blockers: list[str] = []
    if not archive["byte_closed"]:
        static_blockers.append("missing_byte_closed_archive_proof")
        static_blockers.extend(f"archive:{blocker}" for blocker in archive["blockers"])
    if not runtime["runtime_closed"]:
        static_blockers.append("missing_byte_closed_runtime_proof")
        static_blockers.extend(f"runtime:{blocker}" for blocker in runtime["blockers"])
    if strict["candidate_static_preflight_ready"] is not True:
        static_blockers.append("strict_candidate_preflight_not_ready")
        static_blockers.extend(
            f"strict:{blocker.get('code', 'unknown')}"
            for blocker in strict["blockers"]
            if isinstance(blocker, Mapping)
        )
    packet_static_ready = payload.get("static_packet_ready")
    packet_static_blockers = _string_list(payload.get("static_blockers"))
    if packet_static_ready is not None and packet_static_ready is not True:
        static_blockers.append("packet_static_preflight_not_ready")
    static_blockers.extend(f"packet_static:{blocker}" for blocker in packet_static_blockers)
    static_blockers = _unique_strings(static_blockers)
    strict_ready = strict["candidate_static_preflight_ready"] is True
    base_static_ready = bool(strict_ready and archive["byte_closed"] and runtime["runtime_closed"] and not static_blockers)
    if base_static_ready and identity["status"] != "passed":
        static_blockers.append("missing_dispatch_identity_for_lane_claim")
        static_blockers.extend(f"identity:{blocker}" for blocker in identity["blockers"])
        static_blockers = _unique_strings(static_blockers)
    static_ready = bool(base_static_ready and identity["status"] == "passed")
    dispatch_blockers: list[str] = []
    if static_ready and not claim["active_lane_claim"]:
        dispatch_blockers.append("missing_active_lane_dispatch_claim")
        dispatch_blockers.extend(f"claim:{blocker}" for blocker in claim["blockers"])
    dispatch_blockers = _unique_strings(dispatch_blockers)
    dirty_matches, watch_paths = _dirty_paths_for_payload(
        payload,
        manifest_path=manifest_path,
        repo_root=repo_root,
        dirty_paths=dirty_paths,
    )
    if dirty_matches:
        dispatch_blockers.append("dirty_worktree_overlap")
        dispatch_blockers.extend(f"dirty:{path}" for path in dirty_matches)
        dispatch_blockers = _unique_strings(dispatch_blockers)
    blockers = _unique_strings([*static_blockers, *dispatch_blockers])
    ready = bool(static_ready and claim["active_lane_claim"] and not dispatch_blockers)
    family, family_group, pareto_scope = _candidate_family(payload)
    evidence_grade = _candidate_evidence_grade(payload)
    proxy_row = _candidate_proxy_row(payload, evidence_grade)
    interaction_assumptions = _candidate_interaction_assumptions(payload)
    conflicts_with_families = _candidate_conflicts(payload, "conflicts_with_families")
    conflicts_with_atoms = _candidate_conflicts(payload, "conflicts_with_atoms")
    field_interaction_contract = _field_interaction_contract(
        payload,
        assumptions=interaction_assumptions,
        conflicts_with_families=conflicts_with_families,
        conflicts_with_atoms=conflicts_with_atoms,
    )
    kkt_proof = _kkt_proof(payload)
    kkt_blockers: list[str] = []
    candidate_static_ready_after_dirty = bool(static_ready and not dirty_matches)
    if not static_ready:
        kkt_blockers.extend(static_blockers or ["candidate_static_preflight_not_ready"])
    if dirty_matches:
        kkt_blockers.append("dirty_worktree_overlap")
    if proxy_row:
        kkt_blockers.append("planning_or_proxy_packet")
    kkt_blockers.extend(field_interaction_contract["blockers"])
    if kkt_proof["status"] != "passed":
        kkt_blockers.extend(f"kkt:{blocker}" for blocker in kkt_proof["blockers"])
    kkt_blockers = _unique_strings(kkt_blockers)
    kkt_ready = not kkt_blockers
    expected_seg_delta = _first_numeric(
        payload,
        (
            ("expected_seg_dist_delta",),
            ("meta_lagrangian_atom", "expected_seg_dist_delta"),
            ("meta_lagrangian_atom_export", "atom_template", "expected_seg_dist_delta"),
            ("selected_target", "expected_seg_dist_delta"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "expected_seg_dist_delta"),
        ),
    )
    expected_pose_delta = _first_numeric(
        payload,
        (
            ("expected_pose_dist_delta",),
            ("meta_lagrangian_atom", "expected_pose_dist_delta"),
            ("meta_lagrangian_atom_export", "atom_template", "expected_pose_dist_delta"),
            ("selected_target", "expected_pose_dist_delta"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "expected_pose_dist_delta"),
        ),
    )
    expected_info_gain = _first_numeric(
        payload,
        (
            ("expected_information_gain_nats",),
            ("information_gain_nats",),
            ("meta_lagrangian_atom", "expected_information_gain_nats"),
            ("meta_lagrangian_atom_export", "atom_template", "expected_information_gain_nats"),
            ("selected_target", "expected_information_gain_nats"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "expected_information_gain_nats"),
        ),
    )
    expected_score_variance = _first_numeric(
        payload,
        (
            ("expected_score_variance",),
            ("predicted_score_variance",),
            ("score_variance",),
            ("meta_lagrangian_atom", "expected_score_variance"),
            ("meta_lagrangian_atom_export", "atom_template", "expected_score_variance"),
            ("selected_target", "expected_score_variance"),
            ("selected_target", "meta_lagrangian_atom_export", "atom_template", "expected_score_variance"),
        ),
    )
    operator_summary = _operator_next_steps_summary(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": repo_relative(manifest_path, repo_root),
        "manifest_sha256": manifest_sha256,
        "candidate_id": _candidate_id(payload, manifest_path),
        "packet_kind": str(payload.get("packet_kind") or payload.get("schema") or payload.get("schema_version") or ""),
        "family": family,
        "family_group": family_group,
        "pareto_scope": pareto_scope,
        "conflicts_with_families": conflicts_with_families,
        "conflicts_with_atoms": conflicts_with_atoms,
        "interaction_assumptions": interaction_assumptions,
        "field_interaction_contract": field_interaction_contract,
        "evidence_grade": evidence_grade,
        "proxy_row": proxy_row,
        "confidence": _candidate_confidence(payload),
        "score_claim": False,
        "dispatch_attempted": False,
        "operator_next_steps_summary": operator_summary,
        "operator_current_blockers": operator_summary["current_blockers"],
        "operator_claim_blockers": operator_summary["claim_blockers"],
        "operator_refresh_blockers": operator_summary["refresh_blockers"],
        "operator_approval_blockers": operator_summary["approval_blockers"],
        "operator_environment_blockers": operator_summary["environment_blockers"],
        "next_local_non_gpu_action": operator_summary["next_local_non_gpu_action"],
        "next_local_non_gpu_command": operator_summary["next_local_non_gpu_command"],
        "ready_for_exact_eval_dispatch": ready,
        "candidate_local_preflight_ready": base_static_ready,
        "candidate_static_preflight_ready": candidate_static_ready_after_dirty,
        "candidate_static_preflight_ready_before_dirty": static_ready,
        "static_candidate_blockers": static_blockers,
        "dispatch_identity_proof": identity,
        "dispatch_claim_proof": claim,
        "strict_candidate_preflight_ready": strict_ready,
        "strict_candidate_static_preflight_ready": strict_ready,
        "strict_candidate_preflight": strict,
        "archive_proof": archive,
        "runtime_proof": runtime,
        "dirty_path_blockers": dirty_matches,
        "dirty_blocked": bool(dirty_matches),
        "candidate_watch_paths": watch_paths,
        "byte_delta": _byte_delta(payload),
        "expected_total_score_delta": _numeric_first(
            payload,
            ("expected_total_score_delta", "rate_score_delta_vs_source_estimate", "score_delta"),
        ),
        "expected_seg_dist_delta": expected_seg_delta,
        "expected_pose_dist_delta": expected_pose_delta,
        "expected_information_gain_nats": expected_info_gain,
        "expected_score_variance": expected_score_variance,
        "kkt_ready_for_field_planning": kkt_ready,
        "kkt_proof": kkt_proof,
        "kkt_blockers": kkt_blockers,
        "pareto_eligible": bool(candidate_static_ready_after_dirty and not proxy_row),
        "pareto_frontier": False,
        "pareto_dominated_by": [],
        "pareto_eligibility_blockers": [],
        "non_dominated_frontier_reason": {},
        "pareto_objectives": {},
        "candidate_blockers": blockers,
        "next_required_proof": _next_required_proofs(blockers),
        "exact_dispatch_blockers": {},
    }


def _expand_manifest_inputs(
    *,
    repo_root: Path,
    manifest_paths: Sequence[Path] | None,
    manifest_globs: Sequence[str] | None,
) -> list[Path]:
    paths: list[Path] = []
    for path in manifest_paths or []:
        paths.append(path if path.is_absolute() else repo_root / path)
    for pattern in manifest_globs or []:
        paths.extend(repo_root.glob(pattern))
    out: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def build_selection_report(
    *,
    repo_root: Path,
    manifest_paths: Sequence[Path] | None = None,
    manifest_globs: Sequence[str] | None = None,
    claims_path: Path | None = None,
    now_utc: str | None = None,
    ttl_hours: float = 24.0,
    dirty_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifests = _expand_manifest_inputs(
        repo_root=repo_root,
        manifest_paths=manifest_paths,
        manifest_globs=manifest_globs,
    )
    rows = [
        _row_for_manifest(
            path,
            repo_root=repo_root,
            claims_path=claims_path,
            now_utc=now_utc,
            ttl_hours=ttl_hours,
            dirty_paths=dirty_paths,
        )
        for path in manifests
    ]
    pareto_summary = _annotate_pareto_frontier(rows)
    _annotate_selection_scores(rows)
    _annotate_row_explanations(rows)
    rows.sort(
        key=lambda row: (
            *_lexicographic_sort_key(row),
            not bool(row["candidate_local_preflight_ready"]),
            not bool(row["pareto_eligible"]),
            len(row["candidate_blockers"]),
            int(row["byte_delta"]),
        )
    )
    selected = rows[0] if rows else None
    ready_count = sum(int(row["ready_for_exact_eval_dispatch"]) for row in rows)
    field_selection_ready_count = sum(
        int(row["field_selection_ready_for_exact_eval_dispatch"]) for row in rows
    )
    local_ready_count = sum(int(row["candidate_local_preflight_ready"]) for row in rows)
    static_ready_count = sum(int(row["candidate_static_preflight_ready"]) for row in rows)
    dirty_blocked_count = sum(int(row["dirty_blocked"]) for row in rows)
    kkt_ready_count = sum(int(row["kkt_ready_for_field_planning"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "strict_candidate_preflight": STRICT_PREFLIGHT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": bool(selected and selected["ready_for_exact_eval_dispatch"]),
        "candidate_local_preflight_ready": bool(selected and selected["candidate_local_preflight_ready"]),
        "candidate_static_preflight_ready": bool(selected and selected["candidate_static_preflight_ready"]),
        "candidate_count": len(rows),
        "candidate_local_preflight_ready_count": local_ready_count,
        "candidate_static_preflight_ready_count": static_ready_count,
        "ready_candidate_count": ready_count,
        "field_selection_ready_for_exact_eval_dispatch": bool(
            selected and selected["field_selection_ready_for_exact_eval_dispatch"]
        ),
        "field_selection_ready_for_exact_eval_dispatch_count": field_selection_ready_count,
        "dirty_blocked_candidate_count": dirty_blocked_count,
        "kkt_ready_for_field_planning_count": kkt_ready_count,
        "pareto_summary": pareto_summary,
        "selection_penalties": dict(sorted(SELECTION_PENALTIES.items())),
        "lexicographic_feasibility_order": [
            "ready_for_exact_eval_dispatch desc",
            "candidate_static_preflight_ready desc",
            "archive_proof.byte_closed desc",
            "runtime_proof.runtime_closed desc",
            "clean_worktree_overlap desc",
            "pareto_frontier desc",
            "kkt_ready_for_field_planning desc",
            "expected_total_score_delta asc",
            "expected_information_gain_nats desc",
        ],
        "selection_policy": (
            "field packet selection is planning-only: exact-eval-ready rows sort first; "
            "otherwise the explicit lexicographic feasibility tuple orders byte-closed, "
            "clean, Pareto-frontier, KKT-proof-backed rows before pure expected-score "
            "deltas, with expected-information-gain as a tie-breaker; penalty terms are "
            "diagnostic only"
        ),
        "selected_candidate": selected,
        "rows": rows,
        "dispatch_blockers": [
            "selection_report_only_no_dispatch",
            "requires_lane_dispatch_claim_before_remote_gpu_submit",
            "requires_exact_cuda_auth_eval_for_score_claim",
        ],
        "report_blockers": [] if rows else ["no_candidate_packet_manifests_supplied"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--manifest-glob", action="append", default=[])
    parser.add_argument("--claims-path", type=Path)
    parser.add_argument("--now-utc")
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--dirty-path", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_selection_report(
        repo_root=args.repo_root,
        manifest_paths=args.manifest,
        manifest_globs=args.manifest_glob,
        claims_path=args.claims_path,
        now_utc=args.now_utc,
        ttl_hours=args.ttl_hours,
        dirty_paths=args.dirty_path,
    )
    text = json_text(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
