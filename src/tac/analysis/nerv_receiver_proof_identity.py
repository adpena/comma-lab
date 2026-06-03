# SPDX-License-Identifier: MIT
"""File-backed receiver-proof identity binding for NeRV ladder rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

RECEIVER_PROOF_PATH_KEYS = (
    "proof_path",
    "receiver_proof_path",
    "runtime_consumption_proof_path",
)
RECEIVER_PROOF_SHA_KEYS = (
    "proof_sha256",
    "receiver_proof_sha256",
    "runtime_consumption_proof_sha256",
)
RECEIVER_PROOF_TRUTHY_KEYS = (
    "receiver_archive_replay_verified",
    "receiver_contract_satisfied",
    "runtime_consumption_proof_ready",
    "receiver_matches_direct",
    "receiver_proof_passed",
    "runtime_consumption_proof_passed",
    "receiver_closed",
    "byte_closed_receiver_proof",
)


def receiver_proof_identity_binding(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: str | Path,
    schema: str = "nerv_receiver_proof_identity_binding.v1",
    path_keys: Sequence[Any] = RECEIVER_PROOF_PATH_KEYS,
    sha_keys: Sequence[Any] = RECEIVER_PROOF_SHA_KEYS,
    archive_bytes: int | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind a receiver proof to a real file and expected SHA-256."""

    root = Path(repo_root).expanduser().resolve()
    proof_path_raw = _first_string(*rows, keys=path_keys)
    expected_sha = _first_string(*rows, keys=sha_keys)
    claimed_proof_passed = any(
        _truthy(_lookup(row, key)) for row in rows for key in RECEIVER_PROOF_TRUTHY_KEYS
    )
    blockers: list[str] = []
    proof_path: Path | None = None
    actual_sha: str | None = None
    proof_file_claimed_pass = False
    proof_file_blockers: list[str] = []
    if proof_path_raw is None:
        blockers.append("receiver_proof_path_missing")
    else:
        proof_path = _resolve_path(proof_path_raw, repo_root=root)
        if proof_path is None or not proof_path.is_file():
            blockers.append("receiver_proof_path_not_file")
        else:
            actual_sha = sha256_file(proof_path)
            proof_file_claimed_pass, proof_file_blockers = _proof_file_claims_pass(
                proof_path
            )
    if not is_sha256_hex(expected_sha):
        blockers.append("receiver_proof_sha256_missing_or_invalid")
    elif actual_sha is not None and actual_sha != str(expected_sha).strip().lower():
        blockers.append("receiver_proof_sha256_mismatch")
    if proof_path is not None and actual_sha is not None and not proof_file_claimed_pass:
        blockers.extend(proof_file_blockers or ["receiver_proof_file_content_not_passed"])
    blockers = _ordered_unique(blockers)
    bound = not blockers
    return {
        "schema": str(schema),
        "bound": bound,
        "proof_path": proof_path.as_posix() if proof_path is not None else None,
        "proof_sha256": actual_sha,
        "expected_proof_sha256": (
            str(expected_sha).strip().lower() if expected_sha is not None else None
        ),
        "proof_passed": bool(
            bound and claimed_proof_passed and proof_file_claimed_pass
        ),
        "claimed_receiver_proof_pass": bool(claimed_proof_passed),
        "proof_file_claimed_receiver_proof_pass": bool(proof_file_claimed_pass),
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "archive_sha256": (
            str(archive_sha256).strip().lower() if archive_sha256 is not None else None
        ),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def bind_nerv_receiver_proof_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: str | Path,
    archive_bytes: int | None = None,
    archive_sha256: str | None = None,
    schema: str = "nerv_trained_ladder_receiver_proof_identity_binding.v1",
) -> dict[str, Any]:
    """Compatibility wrapper for trained-row and ladder callers."""

    return receiver_proof_identity_binding(
        rows,
        repo_root=repo_root,
        schema=schema,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
    )


def is_sha256_hex(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _first_string(*rows: Mapping[str, Any], keys: Sequence[Any]) -> str | None:
    for row in rows:
        for key in keys:
            value = _lookup(row, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _lookup(row: Mapping[str, Any], key: Any) -> Any:
    if isinstance(key, tuple):
        cur: Any = row
        for part in key:
            if not isinstance(cur, Mapping):
                return None
            cur = cur.get(part)
        return cur
    return row.get(key)


def _proof_file_claims_pass(path: Path) -> tuple[bool, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False, ["receiver_proof_json_unreadable"]
    if not isinstance(payload, Mapping):
        return False, ["receiver_proof_payload_not_object"]
    blockers: list[str] = []
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema.strip():
        blockers.append("receiver_proof_schema_missing")
    pass_flag = any(_truthy(_lookup(payload, key)) for key in RECEIVER_PROOF_TRUTHY_KEYS)
    if not pass_flag:
        blockers.append("receiver_proof_payload_pass_flag_missing")
    return not blockers, blockers


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _resolve_path(value: str | Path | None, *, repo_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "FALSE_AUTHORITY",
    "RECEIVER_PROOF_PATH_KEYS",
    "RECEIVER_PROOF_SHA_KEYS",
    "RECEIVER_PROOF_TRUTHY_KEYS",
    "bind_nerv_receiver_proof_identity",
    "is_sha256_hex",
    "receiver_proof_identity_binding",
]
