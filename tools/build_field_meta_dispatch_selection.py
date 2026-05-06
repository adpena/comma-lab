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

SCHEMA_VERSION = 1
TOOL = "tools/build_field_meta_dispatch_selection.py"
STRICT_PREFLIGHT = "experiments/preflight_candidate_manifest_dispatch_readiness.py"
HEX_CHARS = set("0123456789abcdef")
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


def _byte_delta(payload: Mapping[str, Any]) -> int:
    for key in (
        "candidate_archive_byte_delta_vs_source_estimate",
        "section_byte_delta",
        "byte_delta",
    ):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    archive_bytes = _coerce_positive_int(payload.get("archive_bytes", payload.get("candidate_archive_bytes")))
    source_bytes = _coerce_positive_int(payload.get("source_archive_bytes"))
    if archive_bytes is not None and source_bytes is not None:
        return archive_bytes - source_bytes
    return 0


def _next_required_proofs(blockers: Sequence[str]) -> list[str]:
    out: list[str] = []
    if "missing_byte_closed_archive_proof" in blockers:
        out.append("local_archive_file_with_matching_sha256_and_bytes")
    if "missing_byte_closed_runtime_proof" in blockers:
        out.append("runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract")
    if "strict_candidate_preflight_not_ready" in blockers:
        out.append(f"passing_{STRICT_PREFLIGHT}")
    if "missing_active_lane_dispatch_claim" in blockers:
        out.append("matching_active_level2_lane_claim_for_manifest_lane_and_job")
    if not out:
        out.append("exact_cuda_auth_eval_on_selected_archive_bytes")
    return out


def _row_for_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    claims_path: Path | None,
    now_utc: str | None,
    ttl_hours: float,
) -> dict[str, Any]:
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"packet manifest must be a JSON object: {manifest_path}")
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
    static_blockers = _unique_strings(static_blockers)
    strict_ready = strict["candidate_static_preflight_ready"] is True
    static_ready = bool(strict_ready and archive["byte_closed"] and runtime["runtime_closed"] and not static_blockers)
    dispatch_blockers: list[str] = []
    if static_ready and not claim["active_lane_claim"]:
        dispatch_blockers.append("missing_active_lane_dispatch_claim")
        dispatch_blockers.extend(f"claim:{blocker}" for blocker in claim["blockers"])
    dispatch_blockers = _unique_strings(dispatch_blockers)
    blockers = _unique_strings([*static_blockers, *dispatch_blockers])
    ready = bool(static_ready and claim["active_lane_claim"] and not dispatch_blockers)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": repo_relative(manifest_path, repo_root),
        "manifest_sha256": manifest_sha256,
        "candidate_id": _candidate_id(payload, manifest_path),
        "packet_kind": str(payload.get("packet_kind") or payload.get("schema") or payload.get("schema_version") or ""),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": ready,
        "candidate_static_preflight_ready": static_ready,
        "static_candidate_blockers": static_blockers,
        "dispatch_claim_proof": claim,
        "strict_candidate_preflight_ready": strict_ready,
        "strict_candidate_static_preflight_ready": strict_ready,
        "strict_candidate_preflight": strict,
        "archive_proof": archive,
        "runtime_proof": runtime,
        "byte_delta": _byte_delta(payload),
        "expected_total_score_delta": _numeric_first(
            payload,
            ("expected_total_score_delta", "rate_score_delta_vs_source_estimate", "score_delta"),
        ),
        "expected_information_gain_nats": _numeric_first(
            payload,
            ("expected_information_gain_nats", "information_gain_nats"),
        ),
        "candidate_blockers": blockers,
        "next_required_proof": _next_required_proofs(blockers),
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
        )
        for path in manifests
    ]
    rows.sort(
        key=lambda row: (
            not bool(row["ready_for_exact_eval_dispatch"]),
            not bool(row["candidate_static_preflight_ready"]),
            len(row["candidate_blockers"]),
            float(row["expected_total_score_delta"]),
            int(row["byte_delta"]),
            -float(row["expected_information_gain_nats"]),
            str(row["manifest_path"]),
        )
    )
    selected = rows[0] if rows else None
    ready_count = sum(int(row["ready_for_exact_eval_dispatch"]) for row in rows)
    static_ready_count = sum(int(row["candidate_static_preflight_ready"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "strict_candidate_preflight": STRICT_PREFLIGHT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": bool(selected and selected["ready_for_exact_eval_dispatch"]),
        "candidate_static_preflight_ready": bool(selected and selected["candidate_static_preflight_ready"]),
        "candidate_count": len(rows),
        "candidate_static_preflight_ready_count": static_ready_count,
        "ready_candidate_count": ready_count,
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
