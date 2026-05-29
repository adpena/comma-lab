#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build strict runtime-consumption proof for a monolithic packet candidate.

This tool does not run inflate or score anything. It converts runtime evidence
that already exists on disk into the strict `tac_runtime_consumption_proof_v1`
shape consumed by `tac.monolithic_packet_candidate`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

SCHEMA = "tac_runtime_consumption_proof_v1"
MONOLITHIC_MANIFEST_SCHEMA = "tac_monolithic_packet_candidate_v1"


class RuntimeConsumptionProofError(ValueError):
    """Raised when proof inputs are malformed."""


def build_runtime_consumption_proof(
    *,
    candidate_manifest_path: Path,
    command_text: str | None = None,
    command_file: Path | None = None,
    runtime_log: Path,
) -> dict[str, Any]:
    """Return strict runtime-consumption proof for one monolithic manifest."""

    manifest = _load_json_object(candidate_manifest_path)
    if manifest.get("schema") != MONOLITHIC_MANIFEST_SCHEMA:
        raise RuntimeConsumptionProofError("candidate manifest schema is not tac_monolithic_packet_candidate_v1")
    if manifest.get("score_claim") is not False:
        raise RuntimeConsumptionProofError("candidate manifest must not claim score")

    command = _load_command(command_text=command_text, command_file=command_file)
    log_text = runtime_log.read_text(encoding="utf-8", errors="replace")
    candidate_archive = _require_mapping(manifest.get("candidate_archive"), "candidate_archive")
    layout = _require_mapping(manifest.get("monolithic_layout"), "monolithic_layout")
    replacements = _require_list(manifest.get("replacements"), "replacements")
    candidate_sha = _require_sha(candidate_archive.get("sha256"), "candidate_archive.sha256")
    new_member_sha = _require_sha(layout.get("new_member_sha256"), "monolithic_layout.new_member_sha256")

    changed_sections: list[dict[str, Any]] = []
    blockers: list[str] = []
    required_tokens = {
        "candidate_archive_sha256": candidate_sha,
        "new_member_sha256": new_member_sha,
    }
    for name, token in required_tokens.items():
        if token not in log_text:
            blockers.append(f"runtime_log_missing_{name}")

    for replacement in replacements:
        if not isinstance(replacement, Mapping):
            raise RuntimeConsumptionProofError("replacement entry is not an object")
        section_name = replacement.get("section_name")
        section_sha = replacement.get("new_sha256")
        if not isinstance(section_name, str) or not section_name:
            raise RuntimeConsumptionProofError("replacement section_name missing")
        section_sha = _require_sha(section_sha, f"replacement:{section_name}.new_sha256")
        token_found = section_sha in log_text
        if not token_found:
            blockers.append(f"runtime_log_missing_changed_section_sha:{section_name}")
        changed_sections.append(
            {
                "section_name": section_name,
                "new_sha256": section_sha,
                "runtime_log_token_found": token_found,
            }
        )

    if not command.strip():
        blockers.append("runtime_command_empty")
    if not log_text.strip():
        blockers.append("runtime_log_empty")

    payload = {
        "schema": SCHEMA,
        "candidate_id": manifest.get("candidate_id", ""),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
        "candidate_archive_path": str(candidate_archive.get("path", "")),
        "candidate_archive_bytes": candidate_archive.get("bytes"),
        "candidate_archive_sha256": candidate_sha,
        "rebuilt_member_sha256": new_member_sha,
        "new_member_sha256": new_member_sha,
        "changed_sections": changed_sections,
        "command_sha256": _sha256_text(command),
        "command_source": str(command_file) if command_file is not None else "inline",
        "log_path": str(runtime_log),
        "log_sha256": _sha256_file(runtime_log),
        "ready_for_exact_eval_runtime": not blockers,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    return payload


def _load_command(*, command_text: str | None, command_file: Path | None) -> str:
    if command_text is not None and command_file is not None:
        raise RuntimeConsumptionProofError("use either command_text or command_file, not both")
    if command_file is not None:
        return command_file.read_text(encoding="utf-8")
    if command_text is not None:
        return command_text
    raise RuntimeConsumptionProofError("command_text or command_file is required")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeConsumptionProofError(f"{path} must contain a JSON object")
    return payload


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConsumptionProofError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeConsumptionProofError(f"{name} must be a list")
    if not value:
        raise RuntimeConsumptionProofError(f"{name} must not be empty")
    return value


def _require_sha(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    ):
        raise RuntimeConsumptionProofError(f"{name} must be a SHA-256 hex string")
    return value.lower()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps_json(payload: dict[str, Any]) -> str:
    """Return stable pretty JSON."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--command-text")
    command_group.add_argument("--command-file", type=Path)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    parser.add_argument(
        "--allow-overwrite-existing-historical-provenance",
        action="store_true",
        help=(
            "Opt-in to overwriting an existing .omx/research/<dir>/ that already "
            "contains canonical HISTORICAL_PROVENANCE JSON files. Per Catalog #113 + "
            "anti-pattern "
            "research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1, "
            "the default behavior is fail-closed; requires --overwrite-rationale."
        ),
    )
    parser.add_argument(
        "--overwrite-rationale",
        type=str,
        default=None,
        help=(
            "Substantive operator rationale (>=4 chars; non-placeholder per "
            "Catalog #287) required when --allow-overwrite-existing-historical-provenance "
            "is set."
        ),
    )
    args = parser.parse_args(argv)

    # Canonical HISTORICAL_PROVENANCE safety per Catalog #113 + anti-pattern
    # `research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1`
    # (registered 2026-05-28). Validates the parent dir of --json-out when set;
    # tool also supports stdout-only mode where no canonical file write happens.
    # Slot F canonical helper extension 2026-05-29.
    if args.json_out is not None:
        from tac.research_pipeline_output_dir_safety import (
            OutputDirSafetyError,
            enforce_research_pipeline_output_dir,
        )

        try:
            enforce_research_pipeline_output_dir(
                args.json_out.parent,
                repo_root=REPO_ROOT,
                allow_overwrite_existing_historical_provenance=(
                    args.allow_overwrite_existing_historical_provenance
                ),
                waiver_rationale=args.overwrite_rationale,
            )
        except OutputDirSafetyError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 3

    try:
        payload = build_runtime_consumption_proof(
            candidate_manifest_path=args.candidate_manifest,
            command_text=args.command_text,
            command_file=args.command_file,
            runtime_log=args.runtime_log,
        )
    except (RuntimeConsumptionProofError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"runtime consumption proof failed: {exc}") from None

    text = dumps_json(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_not_ready and payload["ready_for_exact_eval_runtime"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
