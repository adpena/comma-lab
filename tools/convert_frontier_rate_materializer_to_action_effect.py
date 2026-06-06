#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert frontier-rate materializer manifests into ActionEffect JSONL rows.

This is a thin adapter around ``ActionEffect.from_frontier_rate_materializer``:
it does not score archives, run inflate, or mint authority.  It loads an
existing materializer manifest plus optional exact auth-eval rows, verifies the
runtime proof hash when the manifest cites one, appends one canonical
``tac.action_effect.v1`` row, and emits the same validation surface consumed by
the commutator ledger and launch gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import (  # noqa: E402
    ActionEffect,
    append_action_effect,
    validate_action_effect_payload,
)
from tac.repo_io import sha256_file  # noqa: E402


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _with_artifact_refs(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    row = dict(manifest)
    row.setdefault("manifest_path", manifest_path.as_posix())
    row.setdefault("artifact_ref", manifest_path.as_posix())
    proof_path_value = row.get("runtime_consumption_proof_path")
    proof_sha_value = row.get("runtime_consumption_proof_sha256")
    if isinstance(proof_path_value, str) and proof_path_value.strip():
        proof_path = _resolve_manifest_relative_path(proof_path_value, manifest_path=manifest_path)
        if not proof_path.is_file():
            raise ValueError(f"runtime proof path missing: {proof_path}")
        actual_sha = sha256_file(proof_path)
        if isinstance(proof_sha_value, str) and proof_sha_value.strip() and proof_sha_value != actual_sha:
            raise ValueError(
                "runtime proof sha256 mismatch: "
                f"manifest={proof_sha_value} actual={actual_sha} path={proof_path}"
            )
        proof = _load_json_object(proof_path, label="runtime proof")
        row.setdefault("runtime_consumption_proof_passed", proof.get("runtime_consumption_proof_passed"))
        row.setdefault("restore_state_pass", proof.get("passed"))
        row.setdefault("runtime_consumption_proof_path", proof_path.as_posix())
        row.setdefault("runtime_consumption_proof_sha256", actual_sha)
    return row


def _resolve_manifest_relative_path(value: str, *, manifest_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    nearby = (manifest_path.parent / candidate).resolve(strict=False)
    if nearby.exists():
        return nearby
    return (REPO_ROOT / candidate).resolve(strict=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Frontier-rate materializer manifest JSON.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="ActionEffect JSONL ledger to append.")
    parser.add_argument("--auth-eval", type=Path, default=None, help="Optional candidate exact auth-eval JSON.")
    parser.add_argument("--source-auth-eval", type=Path, default=None, help="Optional source exact auth-eval JSON.")
    parser.add_argument("--consumer", default="action_effect_commutator_ledger")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest_path = args.manifest.resolve(strict=False)
        manifest = _with_artifact_refs(_load_json_object(manifest_path, label="manifest"), manifest_path)
        auth_eval = _load_json_object(args.auth_eval, label="auth-eval") if args.auth_eval is not None else None
        source_auth_eval = (
            _load_json_object(args.source_auth_eval, label="source-auth-eval")
            if args.source_auth_eval is not None
            else None
        )
        effect = ActionEffect.from_frontier_rate_materializer(
            manifest,
            auth_eval=auth_eval,
            source_auth_eval=source_auth_eval,
            consumer=args.consumer,
        )
        record = append_action_effect(effect, args.output_jsonl)
        validation = validate_action_effect_payload(record)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: could not convert frontier-rate materializer to ActionEffect: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": "tac.frontier_rate_materializer_action_effect_conversion.v1",
        "manifest_path": args.manifest.as_posix(),
        "output_jsonl": args.output_jsonl.as_posix(),
        "action_id": record["action_id"],
        "authority": record["authority"],
        "delta_score_total": record["delta_score_total"],
        "delta_bytes": record["delta_bytes"],
        "value_per_byte": record["value_per_byte"],
        "validation": validation,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")
    return 0 if validation.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
