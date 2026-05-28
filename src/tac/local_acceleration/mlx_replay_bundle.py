# SPDX-License-Identifier: MIT
"""Deterministic replay bundles for MLX-local advisory rows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import sha256_file, tree_sha256

MLX_LOCAL_REPLAY_BUNDLE_SCHEMA = "mlx_local_replay_bundle.v1"
SENSITIVE_ENV_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "COOKIE",
    "CREDENTIAL",
    "PRIVATE",
    "API_KEY",
    "AUTH",
)


class MlxReplayBundleError(ValueError):
    """Raised when a replay bundle request is malformed."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.resolve(strict=False).as_posix()


def _resolve(path: str | Path, repo_root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else repo_root / value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_text(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_schema_fields(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
        return {}
    if path.suffix.lower() == ".jsonl":
        return {"json_kind": "jsonl"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"json_kind": "unreadable_json"}
    if not isinstance(payload, Mapping):
        return {"json_kind": type(payload).__name__}
    receiver_verification = payload.get("receiver_verification")
    return {
        "json_kind": "object",
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "score_claim": payload.get("score_claim"),
        "promotion_eligible": payload.get("promotion_eligible"),
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch"),
        "byte_closed_candidate_emitted": payload.get("byte_closed_candidate_emitted"),
        "receiver_contract_satisfied": payload.get("receiver_contract_satisfied"),
        "full_frame_inflate_parity_satisfied": payload.get(
            "full_frame_inflate_parity_satisfied"
        ),
        "receiver_proof_sha256": receiver_verification.get("proof_sha256")
        if isinstance(receiver_verification, Mapping)
        else None,
    }


def artifact_manifest_entry(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Return a stable manifest row for a replay input or output artifact."""

    root = Path(repo_root)
    resolved = _resolve(path, root)
    row: dict[str, Any] = {
        "path": _repo_rel(resolved, root),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        row["missing"] = True
        return row
    if resolved.is_file():
        row.update(
            {
                "kind": "file",
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
        row.update(_json_schema_fields(resolved))
        return row
    if resolved.is_dir():
        row.update(
            {
                "kind": "directory",
                "tree_sha256": tree_sha256(resolved),
                "entry_count": sum(1 for _ in resolved.rglob("*")),
            }
        )
        return row
    row["kind"] = "other"
    return row


def _capture_environment() -> dict[str, Any]:
    captured: dict[str, str] = {}
    redacted: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if any(marker in key.upper() for marker in SENSITIVE_ENV_MARKERS):
            redacted[key] = f"<redacted:sha256={_sha256_text(value)}>"
        else:
            captured[key] = value
    return {
        "capture_mode": "full_env_with_secret_values_redacted_and_hashed",
        "key_count": len(os.environ),
        "full_env_sha256": _sha256_text(json.dumps(dict(sorted(os.environ.items())), sort_keys=True)),
        "values": captured,
        "redacted_values": redacted,
        "redacted_key_count": len(redacted),
    }


def _capture_packages() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("mlx", "numpy", "torch"):
        try:
            module = __import__(name)
        except Exception:
            packages[name] = None
        else:
            packages[name] = getattr(module, "__version__", "unknown")
    return packages


def _capture_git(repo_root: Path) -> dict[str, Any]:
    status = _git_text(repo_root, "status", "--porcelain=v1") or ""
    return {
        "head": _git_text(repo_root, "rev-parse", "HEAD"),
        "branch": _git_text(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status.strip()),
        "status_porcelain_sha256": _sha256_text(status),
        "status_porcelain_line_count": 0 if not status else len(status.splitlines()),
    }


def build_mlx_local_replay_bundle(
    *,
    repo_root: str | Path,
    bundle_id: str,
    axis: str,
    commands: Sequence[Sequence[str]],
    artifact_paths: Sequence[str | Path],
    input_artifact_paths: Sequence[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed deterministic replay bundle for MLX advisory work."""

    if not bundle_id.strip():
        raise MlxReplayBundleError("bundle_id must be non-empty")
    if "[macOS-MLX research-signal]" not in axis and "MLX" not in axis:
        raise MlxReplayBundleError("axis must identify MLX-local advisory evidence")
    if not commands:
        raise MlxReplayBundleError("at least one replay command is required")
    root = Path(repo_root)
    output_artifacts = [
        artifact_manifest_entry(path, repo_root=root) for path in artifact_paths
    ]
    input_artifacts = [
        artifact_manifest_entry(path, repo_root=root) for path in input_artifact_paths
    ]
    missing = [
        row["path"]
        for row in [*input_artifacts, *output_artifacts]
        if row.get("missing") is True
    ]
    env = _capture_environment()
    byte_closed_receiver_proof_present = any(
        row.get("byte_closed_candidate_emitted") is True
        and row.get("receiver_contract_satisfied") is True
        and row.get("full_frame_inflate_parity_satisfied") is True
        and bool(row.get("receiver_proof_sha256"))
        for row in output_artifacts
    )
    blockers = [
        "macos_mlx_research_signal_has_no_score_authority",
        "contest_cpu_or_cuda_exact_eval_required_before_promotion",
    ]
    if not byte_closed_receiver_proof_present:
        blockers.append("byte_closed_archive_and_receiver_runtime_proof_required_before_dispatch")
    if missing:
        blockers.append("missing_replay_artifacts")
    if env["redacted_key_count"]:
        blockers.append("secret_env_values_redacted_exact_replay_requires_local_context")
    payload: dict[str, Any] = {
        "schema": MLX_LOCAL_REPLAY_BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "generated_at_utc": _utc_now(),
        "axis": axis,
        "repo": _capture_git(root),
        "platform": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": platform.uname()._asdict(),
        },
        "packages": _capture_packages(),
        "environment": env,
        "argv": list(argv) if argv is not None else list(sys.argv),
        "replay_commands": [list(command) for command in commands],
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
        "missing_artifacts": missing,
        "metadata": dict(metadata or {}),
        "replay_readiness": {
            "local_replay_ready": not missing,
            "contest_exact_eval_ready": False,
            "byte_closed_receiver_proof_present": byte_closed_receiver_proof_present,
            "exact_eval_blockers": blockers,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="mlx_local_replay_bundle")
    return payload


__all__ = [
    "MLX_LOCAL_REPLAY_BUNDLE_SCHEMA",
    "MlxReplayBundleError",
    "artifact_manifest_entry",
    "build_mlx_local_replay_bundle",
]
