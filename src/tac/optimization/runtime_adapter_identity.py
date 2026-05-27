# SPDX-License-Identifier: MIT
"""Runtime-adapter identity validation for queue and harvest boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file, tree_sha256

RUNTIME_READY_FIELDS = (
    "runtime_adapter_ready",
    "candidate_runtime_adapter_blocker_cleared",
)
RUNTIME_DIR_FIELDS = (
    "candidate_runtime_dir",
    "runtime_dir",
    "inflate_runtime_dir",
    "runtime_adapter_dir",
)
RUNTIME_TREE_SHA_FIELDS = (
    "candidate_runtime_tree_sha256",
    "runtime_tree_sha256",
    "inflate_runtime_tree_sha256",
    "runtime_adapter_tree_sha256",
)
RUNTIME_EXPECTED_TREE_SHA_FIELDS = (
    "expected_runtime_tree_sha256",
    "expected_inflate_runtime_tree_sha256",
    "expected_candidate_runtime_tree_sha256",
    "modal_expected_runtime_tree_sha256",
)
RUNTIME_ADAPTER_FILE_PATH_FIELDS = (
    "path",
    "runtime_adapter_path",
    "source_runtime_unpacker_path",
)
RUNTIME_ADAPTER_FILE_SHA_FIELDS = (
    "sha256",
    "runtime_adapter_sha256",
    "source_sha256",
)
RUNTIME_NESTED_MAPPING_FIELDS = (
    "runtime_adapter_manifest",
    "receiver_verification",
    "runtime_manifest",
    "candidate_runtime",
    "packet_member_merge_receiver_runtime",
    "tensor_factorize_receiver_runtime",
    "renderer_payload_dfl1_receiver_runtime",
    "full_frame_inflate_parity_verification",
)
_SHA256_HEX = frozenset("0123456789abcdef")


def runtime_adapter_identity_claimed(payload: Mapping[str, Any]) -> bool:
    """Return whether ``payload`` claims an executable runtime adapter is ready."""

    return any(
        mapping.get(key) is True for mapping in _iter_runtime_identity_mappings(payload) for key in RUNTIME_READY_FIELDS
    )


def runtime_adapter_tree_sha256_from_mapping(mapping: Mapping[str, Any]) -> str | None:
    """Return the first explicit runtime-tree SHA-256 in ``mapping``."""

    for candidate in _iter_runtime_identity_mappings(mapping):
        for key in RUNTIME_TREE_SHA_FIELDS:
            value = _sha256_or_none(candidate.get(key))
            if value is not None:
                return value
    return None


def runtime_adapter_identity_blockers(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
    context: str = "runtime_adapter",
    require_claimed: bool = False,
) -> list[str]:
    """Return fail-closed blockers for claimed runtime-adapter identity.

    A ready runtime adapter must be anchored to a live directory tree with an
    explicit tree hash, or to a concrete adapter file with a matching file hash
    for source-native receiver adapters.

    ``require_claimed`` is for authority boundaries where runtime identity is a
    mandatory postcondition; missing readiness claims must fail closed there.
    """

    if not runtime_adapter_identity_claimed(payload):
        if require_claimed:
            return [f"{context}_runtime_adapter_identity_claim_missing"]
        return []

    repo = Path(repo_root)
    blockers: list[str] = []
    runtime_dirs = _runtime_dir_paths(payload, repo_root=repo)
    if runtime_dirs:
        observed_tree_sha = runtime_adapter_tree_sha256_from_mapping(payload)
        expected_tree_sha = _runtime_adapter_expected_tree_sha256_from_mapping(payload)
        if expected_tree_sha is None:
            blockers.append(f"{context}_expected_runtime_tree_sha256_missing")
        if observed_tree_sha is None and expected_tree_sha is None:
            blockers.append(f"{context}_runtime_tree_sha256_missing")
        blockers.extend(
            _expected_runtime_tree_identity_blockers(
                observed_tree_sha=observed_tree_sha,
                expected_tree_sha=expected_tree_sha,
                context=context,
            )
        )
        live_tree_checked = False
        for runtime_dir in runtime_dirs:
            if runtime_dir.is_symlink():
                blockers.append(f"{context}_runtime_dir_is_symlink")
                continue
            if not runtime_dir.is_dir():
                blockers.append(f"{context}_runtime_dir_missing")
                continue
            if not (runtime_dir / "inflate.sh").is_file():
                blockers.append(f"{context}_runtime_dir_missing_inflate_sh")
                continue
            live_tree_checked = True
            actual_tree_sha = tree_sha256(runtime_dir).lower()
            if expected_tree_sha is not None and actual_tree_sha != expected_tree_sha:
                blockers.append(f"{context}_expected_runtime_tree_sha256_mismatch")
            if observed_tree_sha is not None and actual_tree_sha != observed_tree_sha:
                blockers.append(f"{context}_runtime_tree_sha256_mismatch")
        if not live_tree_checked:
            blockers.append(f"{context}_runtime_tree_live_identity_unverified")
        return _ordered_unique(blockers)

    adapter_file_blockers, adapter_file_checked = _runtime_adapter_file_blockers(
        payload,
        repo_root=repo,
        context=context,
    )
    if adapter_file_checked:
        return _ordered_unique(adapter_file_blockers)

    blockers.append(f"{context}_runtime_dir_missing_for_tree_identity")
    return blockers


def _runtime_adapter_expected_tree_sha256_from_mapping(
    mapping: Mapping[str, Any],
) -> str | None:
    """Return the first explicit expected runtime-tree SHA-256 in ``mapping``."""

    for candidate in _iter_runtime_identity_mappings(mapping):
        for key in RUNTIME_EXPECTED_TREE_SHA_FIELDS:
            value = _sha256_or_none(candidate.get(key))
            if value is not None:
                return value
    return None


def _expected_runtime_tree_identity_blockers(
    *,
    observed_tree_sha: str | None,
    expected_tree_sha: str | None,
    context: str,
) -> list[str]:
    if observed_tree_sha is None or expected_tree_sha is None:
        return []
    if observed_tree_sha == expected_tree_sha:
        return []
    return [f"{context}_expected_runtime_tree_sha256_mismatch"]


def _runtime_dir_paths(payload: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for mapping in _iter_runtime_identity_mappings(payload):
        for key in RUNTIME_DIR_FIELDS:
            value = _nonempty_string(mapping.get(key))
            if value is None:
                continue
            path = _resolve_path(value, repo_root=repo_root)
            marker = path.as_posix()
            if marker in seen:
                continue
            seen.add(marker)
            out.append(path)
    return out


def _runtime_adapter_file_blockers(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
    context: str,
) -> tuple[list[str], bool]:
    blockers: list[str] = []
    checked = False
    for mapping in _iter_runtime_identity_mappings(payload):
        path_value = next(
            (
                _nonempty_string(mapping.get(key))
                for key in RUNTIME_ADAPTER_FILE_PATH_FIELDS
                if _nonempty_string(mapping.get(key)) is not None
            ),
            None,
        )
        if path_value is None:
            continue
        expected_sha = next(
            (
                _sha256_or_none(mapping.get(key))
                for key in RUNTIME_ADAPTER_FILE_SHA_FIELDS
                if _sha256_or_none(mapping.get(key)) is not None
            ),
            None,
        )
        checked = True
        path = _resolve_path(path_value, repo_root=repo_root)
        if expected_sha is None:
            blockers.append(f"{context}_runtime_adapter_file_sha256_missing")
            continue
        if path.is_symlink():
            blockers.append(f"{context}_runtime_adapter_file_is_symlink")
            continue
        if not path.is_file():
            blockers.append(f"{context}_runtime_adapter_file_missing")
            continue
        if sha256_file(path) != expected_sha:
            blockers.append(f"{context}_runtime_adapter_file_sha256_mismatch")
    return blockers, checked


def _iter_runtime_identity_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    yield value
    for key in RUNTIME_NESTED_MAPPING_FIELDS:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            yield from _iter_runtime_identity_mappings(nested)
    for nested in value.values():
        if isinstance(nested, list | tuple):
            for item in nested:
                if isinstance(item, Mapping):
                    yield from _iter_runtime_identity_mappings(item)


def _resolve_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) == 64 and all(ch in _SHA256_HEX for ch in text):
        return text
    return None


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = [
    "RUNTIME_DIR_FIELDS",
    "RUNTIME_EXPECTED_TREE_SHA_FIELDS",
    "RUNTIME_READY_FIELDS",
    "RUNTIME_TREE_SHA_FIELDS",
    "runtime_adapter_identity_blockers",
    "runtime_adapter_identity_claimed",
    "runtime_adapter_tree_sha256_from_mapping",
]
