# SPDX-License-Identifier: MIT
"""Reusable Modal auth-eval dispatch and recovery helpers.

The experiment entry points own provider-specific images and archive upload.
This module owns the provider-agnostic custody shape: dispatch claims, detached
Modal call metadata, artifact harvest, and result JSON materialization.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.auth_eval_result import parse_auth_eval_score_claim
from tac.deploy.claims import (
    DispatchClaimSpec,
    active_claim_row,
    predicted_eta,
    record_dispatch_claim,
    terminal_dispatch_claim,
    utc_now,
)
from tac.repo_io import read_json, write_json

SPAWN_SCHEMA = "modal_auth_eval_spawn_v1"
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
SKIPPED_RUNTIME_UPLOAD_FILENAMES = {".DS_Store", ".gitignore", ".gitattributes"}
SENSITIVE_RUNTIME_UPLOAD_NAMES = {
    ".env",
    ".env.local",
    ".netrc",
    "authorized_keys",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS = (
    "apikey",
    "api_key",
    "credential",
    "private_key",
    "secret",
    "token",
)


class UnsafeModalArtifactPath(ValueError):
    """Raised when a Modal artifact key would escape its recovery directory."""


class ModalArtifactWriteError(RuntimeError):
    """Raised when Modal artifact materialization cannot be completed safely."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(f"failed to materialize {len(errors)} Modal artifact(s)")


class ModalAuthEvalPairingError(ValueError):
    """Raised when a Modal auth-eval dispatch lacks paired-axis custody."""


@dataclass(frozen=True)
class PreparedModalAuthEvalRequest:
    """Canonical local upload payload for Modal auth-eval entry points."""

    archive_path: Path
    archive_bytes: bytes
    archive_sha256: str
    archive_size_bytes: int
    inflate_sh_rel: str
    submission_dir_path: Path | None
    submission_dir_zip: bytes | None
    submission_dir_zip_sha256: str | None
    output_dir: Path


@dataclass(frozen=True)
class ClaimSpec(DispatchClaimSpec):
    """Dispatch-claim metadata for a Modal auth-eval job."""

    platform: str = "modal"


def validate_modal_auth_eval_pairing(
    *,
    axis: str,
    pair_group_id: str,
    single_axis_waiver_reason: str,
) -> dict[str, Any]:
    """Validate the paired-by-default Modal auth-eval contract.

    Modal CPU and CUDA auth evals can disagree materially. New exact-eval
    dispatches must therefore belong to a CPU/CUDA pair group unless the
    operator gives an explicit single-axis waiver reason. This helper only
    validates dispatch metadata; pair completion is adjudicated from harvested
    artifacts.
    """

    normalized_axis = str(axis or "").strip().lower()
    if normalized_axis not in {"contest_cuda", "contest_cpu", "diagnostic_cuda", "diagnostic_cpu"}:
        raise ModalAuthEvalPairingError(f"unsupported auth-eval axis: {axis!r}")
    pair = str(pair_group_id or "").strip()
    waiver = str(single_axis_waiver_reason or "").strip()
    if pair and waiver:
        raise ModalAuthEvalPairingError(
            "set either pair_group_id or single_axis_waiver_reason, not both"
        )
    if not pair and not waiver:
        raise ModalAuthEvalPairingError(
            "Modal auth eval is paired-by-default. Use the paired launcher or "
            "pass --pair-group-id shared by the CPU/CUDA sibling jobs. "
            "Single-axis runs require --single-axis-waiver-reason."
        )
    return {
        "paired_axis_required": True,
        "pair_group_id": pair or None,
        "single_axis_waiver_reason": waiver or None,
        "single_axis_waiver_used": bool(waiver),
        "axis": normalized_axis,
        "required_pair_axes": ["contest_cuda", "contest_cpu"],
    }


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for in-memory custody payloads."""

    return hashlib.sha256(data).hexdigest()


def complete_false_authority_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Force false-authority fields on non-authoritative Modal payloads."""

    out = dict(payload)
    for key, value in FALSE_AUTHORITY_FIELDS.items():
        out[key] = value
    return out


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def modal_uploaded_submission_dir_runtime_manifest(
    local_runtime_manifest: dict[str, Any],
    *,
    remote_submission_dir: str = "/tmp/modal_auth_eval/submission_dir",
) -> dict[str, Any]:
    """Project a local contest runtime manifest to Modal's uploaded tree shape.

    ``experiments/modal_auth_eval.py`` extracts ``--submission-dir`` transport
    zips under ``/tmp/modal_auth_eval/submission_dir`` on the remote Linux host
    before invoking ``experiments/contest_auth_eval.py``. The contest runtime
    tree hash includes the runtime root name and each out-of-repo absolute file
    path. Packet builders use this helper so exact-eval claim notes bind the
    same runtime-tree hash the Modal evaluator will record after extraction.
    """

    files = []
    remote_root = str(remote_submission_dir).rstrip("/")
    for row in local_runtime_manifest.get("files", []):
        if not isinstance(row, dict):
            continue
        relative_path = row.get("relative_path")
        rewritten = dict(row)
        if isinstance(relative_path, str):
            rewritten["repo_relative_path"] = f"{remote_root}/{relative_path}"
        files.append(rewritten)

    repo_local_tac = dict(local_runtime_manifest.get("repo_local_tac_import_manifest") or {})
    repo_local_tac["runtime_root_name"] = "submission_dir"
    tree_payload = {
        "runtime_root_name": "submission_dir",
        "files": files,
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": local_runtime_manifest.get("upstream_evaluate_py"),
    }
    content_payload = {
        "files": [
            {
                "relative_path": row.get("relative_path"),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            }
            for row in files
        ],
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {
            key: value for key, value in repo_local_tac.items() if key != "runtime_root_name"
        },
        "upstream_evaluate_py": local_runtime_manifest.get("upstream_evaluate_py"),
    }
    return {
        "schema": "contest_auth_eval_runtime_dependency_manifest_v1",
        "runtime_root": remote_root,
        "runtime_file_count": len(files),
        "runtime_tree_sha256": _canonical_json_sha256(tree_payload),
        "runtime_content_tree_sha256": _canonical_json_sha256(content_payload),
        "files": files,
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": local_runtime_manifest.get("upstream_evaluate_py"),
    }


def safe_artifact_label(value: str) -> str:
    """Return a filesystem-safe label for Modal auth-eval artifact directories."""

    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "archive"


def safe_modal_artifact_path(artifacts_dir: Path, relpath: str) -> Path:
    """Return a safe local target for an untrusted Modal artifact key."""

    raw = str(relpath).replace("\\", "/").strip()
    path = Path(raw)
    if (
        not raw
        or raw in {".", ".."}
        or path.is_absolute()
        or any(part in {"", ".."} for part in path.parts)
    ):
        raise UnsafeModalArtifactPath(f"unsafe Modal artifact path: {relpath!r}")
    root = artifacts_dir.resolve(strict=False)
    target = (artifacts_dir / path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise UnsafeModalArtifactPath(
            f"Modal artifact path escapes recovery root: {relpath!r}"
        ) from exc
    return target


def materialize_modal_artifacts(
    *,
    out_dir: Path,
    artifacts: dict[str, Any],
) -> list[str]:
    """Safely write every Modal artifact and return its relative names.

    The function validates all paths and byte payload types before writing any
    file, so unsafe or malformed provider results cannot leave a partial local
    recovery tree that later looks harvested.
    """

    planned: list[tuple[str, Path, bytes]] = []
    errors: list[dict[str, Any]] = []
    for name, data in sorted(artifacts.items(), key=lambda item: str(item[0])):
        try:
            if not isinstance(name, str):
                raise TypeError(f"artifact key must be str, got {type(name).__name__}")
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError(
                    "artifact payload must be bytes-like, "
                    f"got {type(data).__name__}"
                )
            planned.append((name, safe_modal_artifact_path(out_dir, name), bytes(data)))
        except Exception as exc:
            errors.append(
                {
                    "relative_path": str(name),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:1000],
                }
            )
    if errors:
        raise ModalArtifactWriteError(errors)

    written: list[Path] = []
    for _name, target, data in planned:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written.append(target)
        except Exception as exc:  # pragma: no cover - filesystem edge
            for path in written:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise ModalArtifactWriteError(
                [
                    {
                        "relative_path": _name,
                        "target": str(target),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                ]
            ) from exc
    return [name for name, _target, _data in planned]


def function_call_id(call: Any) -> str:
    """Extract a Modal FunctionCall id from SDK-version-dependent handles."""

    call_id = getattr(call, "object_id", None) or getattr(call, "function_call_id", None)
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError("Modal .spawn() returned no function call id")
    return call_id


def runtime_upload_skip_reason(rel: str) -> str | None:
    """Return the reason a runtime-file path is omitted from transport zips."""

    path = Path(rel)
    if path.name in SKIPPED_RUNTIME_UPLOAD_FILENAMES:
        return "ignored host metadata"
    if path.suffix == ".pyc" or "__pycache__" in path.parts:
        return "ignored python bytecode cache"
    return None


def validate_runtime_upload_file(path: Path, rel: str) -> None:
    """Fail closed on runtime files that are unsafe to upload to Modal."""

    rel_path = Path(rel)
    if path.is_symlink():
        raise ValueError(f"refusing symlink in uploaded runtime tree: {rel}")
    for part in rel_path.parts:
        if part.startswith("."):
            raise ValueError(f"refusing hidden file or directory in uploaded runtime tree: {rel}")
    lowered_parts = {part.lower() for part in rel_path.parts}
    if lowered_parts & SENSITIVE_RUNTIME_UPLOAD_NAMES:
        raise ValueError(f"refusing secret-looking file in uploaded runtime tree: {rel}")
    lowered_rel = rel.lower()
    if any(marker in lowered_rel for marker in SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS):
        raise ValueError(f"refusing secret-looking file in uploaded runtime tree: {rel}")


def submission_dir_zip_bytes(submission_dir: Path) -> bytes:
    """Return a deterministic transport zip for a Modal runtime tree."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in submission_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(submission_dir).as_posix()
            if runtime_upload_skip_reason(rel):
                continue
            validate_runtime_upload_file(path, rel)
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def prepare_modal_auth_eval_request(
    *,
    archive: str | Path,
    output_dir: str | Path,
    inflate_sh: str | Path,
    submission_dir: str | Path,
    default_output_root: Path,
    cwd: Path | None = None,
) -> PreparedModalAuthEvalRequest:
    """Build the shared local request shape for Modal CPU/CUDA auth eval.

    This function intentionally owns path traversal checks, deterministic
    runtime-tree transport zipping, archive hashing, and default artifact
    directory naming for both Modal auth-eval entry points. Keeping that
    normalization in one place prevents CPU/CUDA wrapper drift.
    """

    root = (cwd or Path.cwd()).resolve()
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise SystemExit(f"FATAL: archive not found: {archive_path}")

    archive_bytes = archive_path.read_bytes()
    archive_sha256 = sha256_bytes(archive_bytes)
    archive_size_bytes = len(archive_bytes)

    submission_dir_path = Path(submission_dir).resolve() if str(submission_dir or "") else None
    inflate_sh_path = Path(inflate_sh)
    transport_zip: bytes | None = None
    transport_zip_sha256: str | None = None

    if submission_dir_path is not None:
        if not submission_dir_path.is_dir():
            raise SystemExit(f"FATAL: --submission-dir is not a directory: {submission_dir_path}")
        if inflate_sh_path.is_absolute():
            try:
                inflate_sh_rel = str(inflate_sh_path.resolve().relative_to(submission_dir_path))
            except ValueError as exc:
                raise SystemExit(
                    "FATAL: absolute --inflate-sh must be inside --submission-dir "
                    f"when uploading a runtime tree: {inflate_sh_path}"
                ) from exc
        else:
            inflate_sh_rel = str(inflate_sh_path)
        if ".." in Path(inflate_sh_rel).parts:
            raise SystemExit(
                f"FATAL: --inflate-sh must not contain parent traversal: {inflate_sh_rel}"
            )
        if not (submission_dir_path / inflate_sh_rel).is_file():
            raise SystemExit(
                f"FATAL: --inflate-sh {inflate_sh_rel!r} not found under --submission-dir "
                f"{submission_dir_path}"
            )
        transport_zip = submission_dir_zip_bytes(submission_dir_path)
        transport_zip_sha256 = sha256_bytes(transport_zip)
    else:
        if inflate_sh_path.is_absolute():
            try:
                inflate_sh_rel = str(inflate_sh_path.resolve().relative_to(root))
            except ValueError as exc:
                raise SystemExit(
                    "FATAL: --inflate-sh must be relative to repo root or inside it: "
                    f"{inflate_sh_path}"
                ) from exc
        else:
            inflate_sh_rel = str(inflate_sh_path)
        if ".." in Path(inflate_sh_rel).parts:
            raise SystemExit(
                f"FATAL: --inflate-sh must not contain parent traversal: {inflate_sh_rel}"
            )

    label = safe_artifact_label(archive_path.stem)
    out_dir = (
        Path(output_dir).resolve()
        if str(output_dir or "")
        else (root / default_output_root / f"{label}_{archive_sha256[:12]}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    return PreparedModalAuthEvalRequest(
        archive_path=archive_path,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_path=submission_dir_path,
        submission_dir_zip=transport_zip,
        submission_dir_zip_sha256=transport_zip_sha256,
        output_dir=out_dir,
    )


def claim_modal_auth_eval_dispatch(
    *,
    repo_root: Path,
    spec: ClaimSpec,
    status: str,
) -> None:
    """Record the required Level-2 dispatch claim before provider submission."""

    if not spec.lane_id or not spec.instance_job_id:
        raise SystemExit(
            "FATAL: Modal auth eval dispatch requires --lane-id and "
            "--instance-job-id before provider work starts"
        )
    record_dispatch_claim(
        repo_root=repo_root,
        spec=spec,
        status=status,
        default_notes="Modal auth eval dispatch; score_claim=false until recovered",
    )


def require_active_modal_auth_eval_claim(
    *,
    repo_root: Path,
    spec: ClaimSpec,
) -> dict[str, str]:
    """Validate that a queue-owned claim already exists before Modal spend."""

    if not spec.lane_id or not spec.instance_job_id:
        raise SystemExit(
            "FATAL: Modal auth eval dispatch requires --lane-id and "
            "--instance-job-id before provider work starts"
        )
    claims_path = repo_root / ".omx" / "state" / "active_lane_dispatch_claims.md"
    try:
        row = active_claim_row(
            claims_path,
            lane_id=spec.lane_id,
            instance_job_id=spec.instance_job_id,
        )
    except ValueError as exc:
        raise SystemExit(
            "FATAL: Modal auth eval --claim-policy require_active could not "
            f"find an active lane claim: {exc}"
        ) from exc
    if row.get("platform") != spec.platform:
        raise SystemExit(
            "FATAL: Modal auth eval active claim platform mismatch: "
            f"expected={spec.platform} actual={row.get('platform')}"
        )
    return row


def terminal_modal_auth_eval_claim(
    *,
    repo_root: Path,
    spec: ClaimSpec,
    status: str,
    notes: str,
) -> None:
    """Close or update a Modal auth-eval claim with a terminal status."""

    terminal_dispatch_claim(repo_root=repo_root, spec=spec, status=status, notes=notes)


def write_spawn_metadata(
    *,
    out_dir: Path,
    tool: str,
    app: str,
    axis: str,
    call_id: str,
    local_request: dict[str, Any],
    result_json_name: str,
    recover_tool: str = "tools/recover_modal_auth_eval.py",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write detached Modal call metadata and the call-id sentinel."""

    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SPAWN_SCHEMA,
        "tool": tool,
        "app": app,
        "axis": axis,
        "call_id": call_id,
        "dispatched_at_utc": utc_now(),
        "result_json_name": result_json_name,
        "score_claim": False,
        "promotion_eligible": False,
        "adjudication_required": True,
        "recover_command": f".venv/bin/python {recover_tool} --output-dir {out_dir}",
        "local_request": local_request,
    }
    if extra:
        payload.update(extra)
    payload = complete_false_authority_fields(payload)
    path = out_dir / "modal_auth_eval_spawn.json"
    write_json(path, payload)
    (out_dir / "modal_call_id.txt").write_text(call_id + "\n", encoding="utf-8")
    return path


def fail_closed_remote_exception_result(
    *,
    out_dir: Path,
    work_dir: Path,
    validation_path: Path,
    canonical_path: str,
    exc: BaseException,
    collect_artifacts: Callable[[Path, Path], dict[str, bytes]],
) -> dict[str, Any]:
    """Return a structured, non-promotable result for unexpected remote errors.

    Modal can surface remote exceptions as a provider-level ``RemoteError`` with
    little or no message. Wrapping the remote body with this helper preserves the
    traceback as a normal artifact-bearing result, so recovery tooling can close
    dispatch claims without fabricating a score.
    """

    validation: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "returncode": 98,
        "canonical_path": canonical_path,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "score_claim": False,
        "promotion_eligible": False,
        "adjudication_required": True,
        "allowed_use": ["debug", "no_score_claim", "no_promotion"],
    }
    validation = complete_false_authority_fields(validation)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(validation_path, validation)
    except Exception as write_exc:  # pragma: no cover - defensive remote diagnostic path
        validation["artifact_write_error"] = repr(write_exc)
    try:
        artifacts = collect_artifacts(out_dir, work_dir)
    except Exception as collect_exc:  # pragma: no cover - defensive remote diagnostic path
        validation["artifact_collection_error"] = repr(collect_exc)
        artifacts = {}
    return {**validation, "artifacts": artifacts}


def read_spawn_metadata(out_dir: Path) -> dict[str, Any]:
    """Read detached Modal auth-eval metadata."""

    path = out_dir / "modal_auth_eval_spawn.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != SPAWN_SCHEMA:
        raise ValueError(f"{path} has unsupported schema: {payload.get('schema_version')!r}")
    return payload


def _function_call_from_id(modal_module: Any, call_id: str) -> Any:
    functions = getattr(modal_module, "functions", None)
    function_call = getattr(functions, "FunctionCall", None) if functions else None
    if function_call is None:
        function_call = getattr(modal_module, "FunctionCall", None)
    if function_call is None:
        raise RuntimeError("Modal SDK has no FunctionCall interface")
    return function_call.from_id(call_id)


def _truthy(value: Any) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}
    )


def _recovered_claim_flags(
    *,
    out_dir: Path,
    result_without_artifacts: dict[str, Any],
    score_axis: str,
) -> dict[str, Any]:
    """Return claim flags, preferring canonical auth-eval artifact custody."""

    flags: dict[str, Any] = {
        "score_claim": False,
        "promotion_eligible": False,
        "artifact_required": True,
    }
    artifact = out_dir / "contest_auth_eval.json"
    if not artifact.is_file():
        flags["diagnostic_blockers"] = ["missing_canonical_contest_auth_eval_json"]
        return flags
    try:
        payload = read_json(artifact)
    except Exception:
        flags["diagnostic_blockers"] = ["invalid_canonical_contest_auth_eval_json"]
        return flags
    if not isinstance(payload, dict):
        flags["diagnostic_blockers"] = ["invalid_canonical_contest_auth_eval_json"]
        return flags

    if score_axis == "contest_cuda":
        claim = parse_auth_eval_score_claim(payload, required_score_axis=score_axis)
        if claim is not None:
            flags["score_claim"] = True
            flags["promotion_eligible"] = _truthy(payload.get("promotion_eligible"))
        else:
            flags["diagnostic_blockers"] = [
                f"canonical_auth_eval_not_valid_{score_axis}_score_claim"
            ]
    elif score_axis == "contest_cpu":
        if (
            payload.get("score_axis") != "contest_cpu"
            or payload.get("evidence_grade") != "contest-CPU"
        ):
            flags["diagnostic_blockers"] = [
                "canonical_auth_eval_not_valid_contest_cpu_leaderboard_artifact"
            ]
        else:
            flags["score_claim"] = True
            flags["promotion_eligible"] = False
    else:
        flags["diagnostic_blockers"] = [
            f"unsupported_modal_auth_eval_recovery_axis:{score_axis or '<missing>'}"
        ]
    for key in (
        "score_axis",
        "evidence_grade",
        "diagnostic_blockers",
        "inflate_device_policy",
    ):
        if key in payload:
            if key == "diagnostic_blockers":
                existing = flags.get("diagnostic_blockers")
                existing_list = existing if isinstance(existing, list) else []
                payload_blockers = payload.get(key)
                payload_list = payload_blockers if isinstance(payload_blockers, list) else []
                flags[key] = [*existing_list, *payload_list]
            else:
                flags[key] = payload.get(key)
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        for key in ("inflate_device_policy", "inflate_env_override_mode"):
            if key in provenance:
                flags[key] = provenance.get(key)
    return flags


def recover_modal_auth_eval(
    *,
    out_dir: Path,
    call_id: str | None = None,
    timeout_s: float = 0.0,
    result_json_name: str | None = None,
    modal_module: Any | None = None,
) -> dict[str, Any]:
    """Recover a detached Modal auth-eval result into ``out_dir``.

    The returned summary is JSON-serializable and does not contain raw artifact
    bytes. Artifact bytes are written to ``out_dir`` exactly as returned by the
    Modal function.
    """

    metadata = read_spawn_metadata(out_dir)
    resolved_call_id = call_id or str(metadata.get("call_id") or "").strip()
    if not resolved_call_id:
        raise ValueError(f"{out_dir}: missing Modal call_id")
    if modal_module is None:
        import modal as modal_module  # type: ignore[no-redef]

    try:
        result = _function_call_from_id(modal_module, resolved_call_id).get(
            timeout=float(timeout_s)
        )
    except TimeoutError:
        summary = {
            "schema_version": "modal_auth_eval_recover_summary_v1",
            "status": "pending",
            "call_id": resolved_call_id,
            "output_dir": str(out_dir),
            "recovered_at_utc": utc_now(),
            "score_claim": False,
            "promotion_eligible": False,
        }
        summary = complete_false_authority_fields(summary)
        write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
        return summary
    except Exception as exc:
        summary = {
            "schema_version": "modal_auth_eval_recover_summary_v1",
            "status": "remote_error",
            "call_id": resolved_call_id,
            "output_dir": str(out_dir),
            "recovered_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "score_claim": False,
            "promotion_eligible": False,
        }
        summary = complete_false_authority_fields(summary)
        write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
        return summary

    if not isinstance(result, dict):
        summary = {
            "schema_version": "modal_auth_eval_recover_summary_v1",
            "status": "invalid_result",
            "call_id": resolved_call_id,
            "output_dir": str(out_dir),
            "recovered_at_utc": utc_now(),
            "error": f"Modal result must be a dict, got {type(result).__name__}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        summary = complete_false_authority_fields(summary)
        write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
        return summary

    artifacts = result.get("artifacts")
    artifact_names: list[str] = []
    if isinstance(artifacts, dict):
        try:
            artifact_names = materialize_modal_artifacts(
                out_dir=out_dir,
                artifacts=artifacts,
            )
        except ModalArtifactWriteError as exc:
            summary = {
                "schema_version": "modal_auth_eval_recover_summary_v1",
                "status": "invalid_artifacts",
                "call_id": resolved_call_id,
                "output_dir": str(out_dir),
                "recovered_at_utc": utc_now(),
                "artifact_write_errors": exc.errors,
                "score_claim": False,
                "promotion_eligible": False,
            }
            summary = complete_false_authority_fields(summary)
            write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
            return summary

    result_without_artifacts = {k: v for k, v in result.items() if k != "artifacts"}
    result_name = result_json_name or str(
        metadata.get("result_json_name") or "modal_auth_eval_result.json"
    )
    write_json(out_dir / result_name, result_without_artifacts)
    claim_flags = _recovered_claim_flags(
        out_dir=out_dir,
        result_without_artifacts=result_without_artifacts,
        score_axis=str(metadata.get("axis") or ""),
    )
    diagnostic_blockers = claim_flags.get("diagnostic_blockers")
    diagnostic_blocker_list = (
        diagnostic_blockers if isinstance(diagnostic_blockers, list) else []
    )
    canonical_artifact_blockers = {
        "missing_canonical_contest_auth_eval_json",
        "invalid_canonical_contest_auth_eval_json",
    }
    canonical_artifact_failed = bool(
        canonical_artifact_blockers.intersection(str(item) for item in diagnostic_blocker_list)
    )
    recovered_status = "recovered"
    recovered_passed = bool(result_without_artifacts.get("passed"))
    recovered_returncode = result_without_artifacts.get("returncode")
    if recovered_passed and canonical_artifact_failed:
        recovered_passed = False
        recovered_status = "recovered_missing_canonical_auth_eval_artifact"
        if "invalid_canonical_contest_auth_eval_json" in diagnostic_blocker_list:
            recovered_status = "recovered_invalid_canonical_auth_eval_artifact"
        recovered_returncode = 97

    summary = {
        "schema_version": "modal_auth_eval_recover_summary_v1",
        "status": recovered_status,
        "call_id": resolved_call_id,
        "output_dir": str(out_dir),
        "recovered_at_utc": utc_now(),
        "result_json": str(out_dir / result_name),
        "artifact_names": artifact_names,
        "passed": recovered_passed,
        "returncode": recovered_returncode,
        "score_claim": claim_flags["score_claim"],
        "promotion_eligible": claim_flags["promotion_eligible"],
        "score_recomputed_from_components": result_without_artifacts.get(
            "score_recomputed_from_components"
        ),
        "avg_posenet_dist": result_without_artifacts.get("avg_posenet_dist"),
        "avg_segnet_dist": result_without_artifacts.get("avg_segnet_dist"),
    }
    for key in (
        "score_axis",
        "evidence_grade",
        "diagnostic_blockers",
        "inflate_device_policy",
        "inflate_env_override_mode",
    ):
        if key in claim_flags:
            summary[key] = claim_flags[key]
    write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
    return summary


__all__ = [
    "ClaimSpec",
    "ModalArtifactWriteError",
    "ModalAuthEvalPairingError",
    "PreparedModalAuthEvalRequest",
    "UnsafeModalArtifactPath",
    "claim_modal_auth_eval_dispatch",
    "complete_false_authority_fields",
    "fail_closed_remote_exception_result",
    "function_call_id",
    "materialize_modal_artifacts",
    "modal_uploaded_submission_dir_runtime_manifest",
    "predicted_eta",
    "prepare_modal_auth_eval_request",
    "read_spawn_metadata",
    "recover_modal_auth_eval",
    "require_active_modal_auth_eval_claim",
    "runtime_upload_skip_reason",
    "safe_modal_artifact_path",
    "submission_dir_zip_bytes",
    "terminal_modal_auth_eval_claim",
    "utc_now",
    "validate_modal_auth_eval_pairing",
    "validate_runtime_upload_file",
    "write_spawn_metadata",
]
