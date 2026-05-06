"""Deterministic run manifests for repository-local JSON tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file


def _input_record(path: Path, repo_root: Path) -> dict[str, Any]:
    """Return deterministic custody metadata for one input file."""

    return {
        "path": repo_relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def attach_tool_run_manifest(
    payload: dict[str, Any],
    *,
    tool: str,
    argv: Sequence[str],
    input_paths: Sequence[Path],
    repo_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Attach non-promotable tool-run custody without changing payload hashes.

    The recorded canonical payload hash excludes ``tool_run_manifest`` itself
    so repeated runs can compare the scientific JSON payload independently of
    output path, argv spelling, or wrapper custody metadata.
    """

    without_tool_manifest = dict(payload)
    without_tool_manifest.pop("tool_run_manifest", None)
    payload_hash = sha256_bytes(json_text(without_tool_manifest).encode("utf-8"))

    result = dict(without_tool_manifest)
    result["tool_run_manifest"] = {
        "schema_version": 1,
        "tool": tool,
        "argv": list(argv),
        "input_files": [_input_record(Path(path), repo_root) for path in input_paths],
        "output_path": repo_relative(output_path, repo_root) if output_path else "",
        "canonical_payload_without_tool_manifest_sha256": payload_hash,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
    }
    result.setdefault("score_claim", False)
    result.setdefault("dispatch_attempted", False)
    result.setdefault("ready_for_exact_eval_dispatch", False)
    return result


__all__ = ["attach_tool_run_manifest"]
