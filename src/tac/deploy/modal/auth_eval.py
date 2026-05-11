"""Reusable Modal auth-eval dispatch and recovery helpers.

The experiment entry points own provider-specific images and archive upload.
This module owns the provider-agnostic custody shape: dispatch claims, detached
Modal call metadata, artifact harvest, and result JSON materialization.
"""
from __future__ import annotations

import subprocess
import io
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tac.repo_io import read_json, write_json

SPAWN_SCHEMA = "modal_auth_eval_spawn_v1"
SKIPPED_RUNTIME_UPLOAD_FILENAMES = {".DS_Store"}
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


def utc_now() -> str:
    """Return a compact UTC timestamp for custody metadata."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def predicted_eta(hours: float = 3.0) -> str:
    """Return a conservative UTC ETA for claim rows."""

    return (
        datetime.now(UTC).replace(microsecond=0) + timedelta(hours=float(hours))
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


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


@dataclass(frozen=True)
class ClaimSpec:
    """Dispatch-claim metadata for a Modal auth-eval job."""

    lane_id: str
    instance_job_id: str
    agent: str
    platform: str = "modal"
    predicted_eta_utc: str = ""
    force: bool = False
    notes: str = ""


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
    cmd = [
        ".venv/bin/python",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--lane-id",
        spec.lane_id,
        "--platform",
        spec.platform,
        "--instance-job-id",
        spec.instance_job_id,
        "--agent",
        spec.agent,
        "--predicted-eta-utc",
        spec.predicted_eta_utc or predicted_eta(),
        "--status",
        status,
        "--notes",
        spec.notes or "Modal auth eval dispatch; score_claim=false until recovered",
    ]
    if spec.force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=repo_root, text=True, check=False)
    if proc.returncode:
        raise SystemExit(
            f"FATAL: dispatch claim failed rc={proc.returncode}; aborting before Modal spend"
        )


def terminal_modal_auth_eval_claim(
    *,
    repo_root: Path,
    spec: ClaimSpec,
    status: str,
    notes: str,
) -> None:
    """Close or update a Modal auth-eval claim with a terminal status."""

    terminal = ClaimSpec(
        lane_id=spec.lane_id,
        instance_job_id=spec.instance_job_id,
        agent=spec.agent,
        platform=spec.platform,
        predicted_eta_utc=utc_now(),
        force=True,
        notes=notes,
    )
    claim_modal_auth_eval_dispatch(repo_root=repo_root, spec=terminal, status=status)


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
    path = out_dir / "modal_auth_eval_spawn.json"
    write_json(path, payload)
    (out_dir / "modal_call_id.txt").write_text(call_id + "\n", encoding="utf-8")
    return path


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
        write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
        return summary

    if not isinstance(result, dict):
        raise TypeError(f"Modal result must be a dict, got {type(result).__name__}")

    artifacts = result.get("artifacts")
    artifact_names: list[str] = []
    if isinstance(artifacts, dict):
        for name, data in sorted(artifacts.items()):
            if not isinstance(name, str) or not isinstance(data, (bytes, bytearray)):
                continue
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(data))
            artifact_names.append(name)

    result_without_artifacts = {k: v for k, v in result.items() if k != "artifacts"}
    result_name = result_json_name or str(
        metadata.get("result_json_name") or "modal_auth_eval_result.json"
    )
    write_json(out_dir / result_name, result_without_artifacts)

    summary = {
        "schema_version": "modal_auth_eval_recover_summary_v1",
        "status": "recovered",
        "call_id": resolved_call_id,
        "output_dir": str(out_dir),
        "recovered_at_utc": utc_now(),
        "result_json": str(out_dir / result_name),
        "artifact_names": artifact_names,
        "passed": bool(result_without_artifacts.get("passed")),
        "returncode": result_without_artifacts.get("returncode"),
        "score_claim": bool(result_without_artifacts.get("score_claim")),
        "promotion_eligible": bool(result_without_artifacts.get("promotion_eligible")),
        "score_recomputed_from_components": result_without_artifacts.get(
            "score_recomputed_from_components"
        ),
        "avg_posenet_dist": result_without_artifacts.get("avg_posenet_dist"),
        "avg_segnet_dist": result_without_artifacts.get("avg_segnet_dist"),
    }
    write_json(out_dir / "modal_auth_eval_recover_summary.json", summary)
    return summary


__all__ = [
    "ClaimSpec",
    "claim_modal_auth_eval_dispatch",
    "function_call_id",
    "predicted_eta",
    "read_spawn_metadata",
    "recover_modal_auth_eval",
    "terminal_modal_auth_eval_claim",
    "runtime_upload_skip_reason",
    "submission_dir_zip_bytes",
    "utc_now",
    "validate_runtime_upload_file",
    "write_spawn_metadata",
]
