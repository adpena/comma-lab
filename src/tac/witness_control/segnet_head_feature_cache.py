# SPDX-License-Identifier: MIT
"""Resumable, frame-staged frozen-SegNet head-feature caches.

The cache deliberately separates two scientific objects:

* every ``live_logits`` slice is a direct frozen-forward capture serialized
  bitwise into the cache.  This is capture provenance, not an all-frame
  independent replay claim: only the declared completion-control frame is
  freshly re-forwarded and compared bitwise;
* ``quotient_features`` are an algebraic rank-quotient representation with
  strictly diagnostic-only authority.  They are useful for geometry, but
  float32 ties need not replay the live forward.

Every committed frame is hashed in both arrays and linked into an
identity-rooted deterministic commitment chain.  Resume validates the immutable
identity, predecessor chain, and all committed bytes before returning a writer;
no stored diagnostic is treated as scientific authority.  These hashes detect
corruption or partial/coordinated tampering.  They are not a cryptographic
signature against a writer able to rewrite every custody file and recompute all
commitments.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

CACHE_SCHEMA: Final = "segnet_head_feature_cache.v1"
PROGRESS_SCHEMA: Final = "segnet_head_feature_cache_progress.v1"
POSITIVE_CONTROL_SCHEMA: Final = "segnet_head_feature_cache_positive_control.v1"
FRAME_COMMITMENT_SCHEMA: Final = "segnet_head_feature_cache_frame_commitment.v1"
COMPLETION_CONTROL_SCHEMA: Final = "segnet_head_feature_cache_completion_control.v1"
CERTIFICATION_SCHEMA: Final = "segnet_head_feature_cache_certification.v1"
MARKER_NAME: Final = ".segnet_head_feature_cache"
MARKER_BYTES: Final = b"segnet_head_feature_cache_v1\n"
MANIFEST_NAME: Final = "manifest.json"
PROGRESS_NAME: Final = "progress.json"
COMPLETION_CONTROL_NAME: Final = "completion_control.json"
CERTIFICATION_NAME: Final = "certification.json"
STAGING_SCRATCH_NAME: Final = "staging_scratch.json"
STAGING_SCRATCH_SCHEMA: Final = "segnet_head_feature_cache_staging_scratch.v1"
STAGING_SUFFIX: Final = ".segnet-head-feature-cache-staging"
ATOMIC_PREPARED_SUFFIX: Final = ".atomic-prepared"
ATOMIC_GENERATION_SUFFIX: Final = ".atomic-generation"
ATOMIC_TRANSACTION_SUFFIX: Final = ".atomic-transaction"
ATOMIC_TRANSACTION_SCHEMA: Final = "descriptor_bound_atomic_transaction.v1"
ATOMIC_COMPLETION_SCHEMA: Final = "descriptor_bound_atomic_completion.v1"
ATOMIC_MODE_FRESH: Final = "FRESH_ABSENT_NOREPLACE"
ATOMIC_MODE_EXISTING: Final = "EXISTING_TARGET_EXCHANGE"
ATOMIC_PRIOR_AUTH_ABSENT: Final = "FRESH_TARGET_ABSENT"
ATOMIC_PRIOR_AUTH_CONSUMER: Final = "CONSUMER_AUTHORIZED_EXACT_PAYLOAD"
LIVE_LOGITS_NAME: Final = "live_logits.f32.npy"
QUOTIENT_FEATURES_NAME: Final = "quotient_features.f32.npy"
APPROVED_SSD_WATERFALL: Final = (
    "/Volumes/VertigoDataTier/pact",
    "/Volumes/APDataStore/pact",
)
CERTIFIED_STAGING_ENTRY_SET: Final = {
    STAGING_SCRATCH_NAME,
    MARKER_NAME,
    MANIFEST_NAME,
    LIVE_LOGITS_NAME,
    QUOTIENT_FEATURES_NAME,
    PROGRESS_NAME,
    CERTIFICATION_NAME,
}
ATOMIC_STAGING_TARGET_NAMES: Final = frozenset(
    {
        STAGING_SCRATCH_NAME,
        MARKER_NAME,
        MANIFEST_NAME,
        PROGRESS_NAME,
        CERTIFICATION_NAME,
    }
)
ATOMIC_CACHE_TARGET_NAMES: Final = ATOMIC_STAGING_TARGET_NAMES | {COMPLETION_CONTROL_NAME}
STORAGE_PREFLIGHT_KEYS: Final = {
    "waterfall_order",
    "existing_approved_roots",
    "selected_root",
    "filesystem_anchor",
    "free_bytes_before",
    "required_free_bytes",
    "allow_local_output_for_tests",
    "PASS",
}
COMMITTED_FRAME_KEYS: Final = {
    "frame",
    "live_logits_sha256",
    "quotient_features_sha256",
    "diagnostics",
    "previous_frame_commitment_sha256",
    "frame_commitment_sha256",
}
PROGRESS_KEYS: Final = {
    "schema",
    "identity_sha256",
    "status",
    "next_frame",
    "committed_frames",
    "frame_chain_head_sha256",
    "completion_positive_control",
}
COMPLETION_CONTROL_KEYS: Final = {
    "schema",
    "identity_sha256",
    "committed_frame_count",
    "terminal_frame_commitment_sha256",
    "positive_control",
    "integrity_scope",
    "writer_rewrite_limit",
}
INTEGRITY_SCOPE: Final = "CORRUPTION_OR_PARTIAL_TAMPER_EVIDENCE_NOT_CRYPTOGRAPHIC_SIGNATURE"
WRITER_REWRITE_LIMIT: Final = "A_WRITER_ABLE_TO_REWRITE_EVERY_CUSTODY_FILE_CAN_RECOMPUTE_ALL_COMMITMENTS"
LOWER_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
ATOMIC_GENERATION_RE: Final = re.compile(r"^\..+\.atomic-generation-[0-9]{8}$")
ATOMIC_TRANSACTION_RE: Final = re.compile(r"^\..+\.atomic-transaction-[0-9]{8}$")
ATOMIC_COMPLETION_RE: Final = re.compile(r"^\..+\.atomic-completion-(?P<transaction>[0-9a-f]{64})\.json$")
ATOMIC_COMPLETION_GENERATION_RE: Final = re.compile(
    r"^\..+\.atomic-completion-generation-(?P<transaction>[0-9a-f]{64})-[0-9]{8}$"
)
ATOMIC_RETAINED_PREFIX: Final = ".atomic-retained-"
ATOMIC_RETAINED_BYTE_WIDTH: Final = 20
ATOMIC_RETAINED_COMPONENT_MAX: Final = 255
ATOMIC_RETAINED_RE: Final = re.compile(
    r"^\.atomic-retained-(?P<original>z[A-Za-z0-9_-]+)-"
    r"(?P<bytes>[0-9]{20})-(?P<sha256>[0-9a-f]{64})-(?P<ordinal>[0-9]{8})$"
)
DIRECTORY_RETENTION_SCHEMA: Final = "segnet_head_feature_cache_directory_retention.v1"
DIRECTORY_RETAINED_PREFIX: Final = ".directory-retained-"
DIRECTORY_RETENTION_RECEIPT_PREFIX: Final = ".drr-"
DIRECTORY_RETAINED_RE: Final = re.compile(
    r"^\.directory-retained-(?P<original>z[A-Za-z0-9_-]+)-"
    r"(?P<bytes>[0-9]{20})-(?P<tree_sha256>[0-9a-f]{64})-(?P<ordinal>[0-9]{8})$"
)
DIRECTORY_RETENTION_RECEIPT_RE: Final = re.compile(r"^\.drr-(?P<binding>[0-9a-f]{64})-(?P<ordinal>[0-9]{8})\.json$")

# Test-only race injectors run before the last authorization check.  Production
# never installs them, so the authorization-to-syscall sequence has no Python
# callback boundary.
_ATOMIC_PREFIX_AUTHORIZATION_TEST_HOOK: Callable[[Path], None] | None = None
_ATOMIC_COMMIT_TEST_HOOK: Callable[[Path, Path], None] | None = None
_ATOMIC_POST_EXCHANGE_TEST_HOOK: Callable[[Path, Path], None] | None = None
_ATOMIC_COMPLETION_TEST_HOOK: Callable[[str, Path], None] | None = None
_MOVE_PATH_NOREPLACE_TEST_HOOK: Callable[[Path, Path], None] | None = None
_MOVE_PATH_NOREPLACE_POST_MOVE_TEST_HOOK: Callable[[Path, Path], None] | None = None


class FeatureCacheError(RuntimeError):
    """Fail-closed cache identity, geometry, parity, or byte-custody error."""


class _NoReplaceDestinationExists(FeatureCacheError):
    """Internal retry signal for a no-replace destination collision."""


@dataclass(frozen=True)
class BoundFileSnapshot:
    """Bytes read from one no-follow descriptor while its path stayed bound."""

    payload: bytes
    file_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class BoundDirectorySnapshot:
    """Descriptor-proven identity and recursive content digest for one tree."""

    directory_identity: tuple[int, int, int, int, int]
    recursive_file_bytes: int
    tree_sha256: str
    entry_count: int


AtomicNamespaceRow = tuple[str, tuple[int, int, int, int, int], int, str]
AtomicNamespaceFingerprint = tuple[AtomicNamespaceRow, ...]


@dataclass(frozen=True)
class AtomicPrepublicationNamespace:
    """Descriptor-bound target-scoped roles observed before writer mutation."""

    active_scratch: AtomicNamespaceFingerprint
    active_transactions: AtomicNamespaceFingerprint
    retained_roles: AtomicNamespaceFingerprint
    active_completions: AtomicNamespaceFingerprint
    active_completion_constructions: AtomicNamespaceFingerprint


def _nofollow_flags(flags: int) -> int:
    try:
        nofollow = os.O_NOFOLLOW
    except AttributeError as exc:  # pragma: no cover - supported authority hosts
        raise FeatureCacheError("this host lacks required no-follow descriptor support") from exc
    return flags | nofollow | getattr(os, "O_CLOEXEC", 0)


def _open_bound_file(path: Path, *, flags: int, role: str) -> tuple[int, os.stat_result]:
    """Open one file without following links and bind it to its pathname."""

    before = _require_regular_file(path, role=role)
    try:
        descriptor = os.open(path, _nofollow_flags(flags))
    except OSError as exc:
        raise FeatureCacheError(f"{role} cannot be opened without following links: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_file_identity(opened) != _stable_file_identity(before)
        ):
            raise FeatureCacheError(f"{role} pathname changed before descriptor binding: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, opened


def _assert_bound_file(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
    *,
    role: str,
    allow_content_change: bool = False,
) -> os.stat_result:
    """Require the descriptor and pathname to still identify the opened inode."""

    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = path.lstat()
    except OSError as exc:
        raise FeatureCacheError(f"{role} identity cannot be revalidated; preserving bytes: {path}") from exc
    if (
        not stat.S_ISREG(descriptor_metadata.st_mode)
        or descriptor_metadata.st_nlink != 1
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_nlink != 1
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        or (opened.st_dev, opened.st_ino) != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    ):
        raise FeatureCacheError(f"{role} pathname identity changed; preserving bytes: {path}")
    if not allow_content_change and (
        _stable_file_identity(descriptor_metadata) != _stable_file_identity(opened)
        or _stable_file_identity(path_metadata) != _stable_file_identity(opened)
    ):
        raise FeatureCacheError(f"{role} content identity changed; preserving bytes: {path}")
    return descriptor_metadata


def read_bound_file(path: str | Path, *, role: str) -> BoundFileSnapshot:
    """Read a complete file through one path-bound, no-follow descriptor."""

    source = Path(path)
    descriptor, opened = _open_bound_file(source, flags=os.O_RDONLY, role=role)
    chunks: list[bytes] = []
    try:
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        _assert_bound_file(source, descriptor, opened, role=role)
    except OSError as exc:
        raise FeatureCacheError(f"{role} cannot be read; preserving bytes: {source}") from exc
    finally:
        os.close(descriptor)
    return BoundFileSnapshot(payload=b"".join(chunks), file_identity=_stable_file_identity(opened))


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with finite-number custody."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FeatureCacheError("value is not canonical finite JSON") from exc


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    source = Path(path)
    descriptor, opened = _open_bound_file(source, flags=os.O_RDONLY, role="hashed custody file")
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
        _assert_bound_file(source, descriptor, opened, role="hashed custody file")
    except OSError as exc:
        raise FeatureCacheError(f"hashed custody file cannot be read; preserving bytes: {source}") from exc
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def sha256_float32_slice(value: np.ndarray) -> str:
    array = _finite_f32(value, name="cache slice")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or LOWER_SHA256_RE.fullmatch(value) is None:
        raise FeatureCacheError(f"{name} must be a lowercase SHA-256")
    return value


def _frame_commitment_sha256(
    *,
    identity_sha256: str,
    previous_frame_commitment_sha256: str,
    frame: int,
    live_logits_sha256: str,
    quotient_features_sha256: str,
    diagnostics: Mapping[str, Any],
) -> str:
    """Commit one frame to its identity, predecessor, slices, and diagnostics."""

    payload = {
        "schema": FRAME_COMMITMENT_SCHEMA,
        "identity_sha256": _require_sha256(identity_sha256, name="frame commitment identity"),
        "previous_frame_commitment_sha256": _require_sha256(
            previous_frame_commitment_sha256,
            name="previous frame commitment",
        ),
        "frame": frame,
        "live_logits_sha256": _require_sha256(live_logits_sha256, name="live-logit slice hash"),
        "quotient_features_sha256": _require_sha256(
            quotient_features_sha256,
            name="quotient-feature slice hash",
        ),
        "diagnostics": json.loads(canonical_json_bytes(dict(diagnostics))),
    }
    if type(frame) is not int or frame < 0:
        raise FeatureCacheError("frame commitment index must be a nonnegative integer")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_prepared_path(path: Path) -> Path:
    """Return the one stable sibling used for an interrupted atomic write."""

    return path.with_name(f".{path.name}{ATOMIC_PREPARED_SUFFIX}")


def atomic_generation_path(path: Path, generation: int) -> Path:
    """Return one deterministic full-payload generation sibling."""

    if type(generation) is not int or generation < 0 or generation > 99_999_999:
        raise FeatureCacheError("atomic generation index is out of range")
    return path.with_name(f".{path.name}{ATOMIC_GENERATION_SUFFIX}-{generation:08d}")


def _atomic_generation_paths(path: Path) -> list[Path]:
    prefix = f".{path.name}{ATOMIC_GENERATION_SUFFIX}-"
    result: list[Path] = []
    for entry in path.parent.iterdir():
        if not entry.name.startswith(prefix):
            continue
        if ATOMIC_GENERATION_RE.fullmatch(entry.name) is None:
            raise FeatureCacheError(f"malformed atomic generation evidence; preserving bytes: {entry}")
        result.append(entry)
    return sorted(result, key=lambda entry: entry.name)


def atomic_transaction_path(path: Path, generation: int) -> Path:
    """Return one durable pre-exchange transaction-record generation."""

    if type(generation) is not int or generation < 0 or generation > 99_999_999:
        raise FeatureCacheError("atomic transaction generation index is out of range")
    return path.with_name(f".{path.name}{ATOMIC_TRANSACTION_SUFFIX}-{generation:08d}")


def _atomic_transaction_paths(path: Path) -> list[Path]:
    prefix = f".{path.name}{ATOMIC_TRANSACTION_SUFFIX}-"
    result: list[Path] = []
    for entry in path.parent.iterdir():
        if not entry.name.startswith(prefix):
            continue
        if ATOMIC_TRANSACTION_RE.fullmatch(entry.name) is None:
            raise FeatureCacheError(f"malformed atomic transaction evidence; preserving bytes: {entry}")
        result.append(entry)
    return sorted(result, key=lambda entry: entry.name)


def atomic_completion_path(path: Path, transaction_sha256: str) -> Path:
    """Return the immutable completion proof for one exact transaction."""

    digest = _require_sha256(transaction_sha256, name="atomic transaction digest")
    return path.with_name(f".{path.name}.atomic-completion-{digest}.json")


def atomic_completion_generation_path(path: Path, transaction_sha256: str, generation: int) -> Path:
    """Return one non-authority construction generation for a completion proof."""

    digest = _require_sha256(transaction_sha256, name="atomic transaction digest")
    if type(generation) is not int or not 0 <= generation <= 99_999_999:
        raise FeatureCacheError("atomic completion generation index is out of range")
    return path.with_name(f".{path.name}.atomic-completion-generation-{digest}-{generation:08d}")


def _atomic_completion_paths(path: Path) -> tuple[list[Path], list[Path]]:
    final_prefix = f".{path.name}.atomic-completion-"
    generation_prefix = f".{path.name}.atomic-completion-generation-"
    finals: list[Path] = []
    generations: list[Path] = []
    for entry in path.parent.iterdir():
        if entry.name.startswith(generation_prefix):
            if ATOMIC_COMPLETION_GENERATION_RE.fullmatch(entry.name) is None:
                raise FeatureCacheError(f"malformed atomic completion construction; preserving bytes: {entry}")
            generations.append(entry)
        elif entry.name.startswith(final_prefix):
            if ATOMIC_COMPLETION_RE.fullmatch(entry.name) is None:
                raise FeatureCacheError(f"malformed atomic completion proof; preserving bytes: {entry}")
            finals.append(entry)
    return sorted(finals), sorted(generations)


def _atomic_admitted_rows_sha256(record: Mapping[str, Any]) -> str:
    rows = record.get("admitted_scratch")
    if not isinstance(rows, list):
        raise FeatureCacheError("atomic completion transaction rows are malformed")
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _atomic_completion_record(
    path: Path,
    *,
    transaction: Mapping[str, Any],
    target: BoundFileSnapshot,
) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(dict(transaction)))
    transaction_payload = canonical_json_bytes(normalized) + b"\n"
    if (
        target.payload is None
        or len(target.payload) != normalized.get("desired_bytes")
        or hashlib.sha256(target.payload).hexdigest() != normalized.get("desired_sha256")
        or list(target.file_identity) != normalized.get("designated_source_identity")
    ):
        raise FeatureCacheError("atomic completion target is not the transaction outcome")
    return {
        "schema": ATOMIC_COMPLETION_SCHEMA,
        "transaction_sha256": hashlib.sha256(transaction_payload).hexdigest(),
        "target_basename": path.name,
        "parent_identity": normalized["parent_identity"],
        "mode": normalized["mode"],
        "desired_bytes": normalized["desired_bytes"],
        "desired_sha256": normalized["desired_sha256"],
        "desired_file_identity": list(target.file_identity),
        "prior_bytes": normalized["prior_bytes"],
        "prior_sha256": normalized["prior_sha256"],
        "prior_file_identity": normalized["prior_file_identity"],
        "admitted_rows_sha256": _atomic_admitted_rows_sha256(normalized),
        "consumer_authorization_sha256": normalized.get("consumer_authorization_sha256"),
        "false_authority": {
            "score_authority": False,
            "promotion_eligible": False,
        },
    }


def _parse_atomic_completion(path: Path, raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        return None
    expected_keys = {
        "schema",
        "transaction_sha256",
        "target_basename",
        "parent_identity",
        "mode",
        "desired_bytes",
        "desired_sha256",
        "desired_file_identity",
        "prior_bytes",
        "prior_sha256",
        "prior_file_identity",
        "admitted_rows_sha256",
        "consumer_authorization_sha256",
        "false_authority",
    }
    identities = (
        value.get("parent_identity"),
        value.get("desired_file_identity"),
    )
    if (
        set(value) != expected_keys
        or value.get("schema") != ATOMIC_COMPLETION_SCHEMA
        or value.get("mode") not in {ATOMIC_MODE_FRESH, ATOMIC_MODE_EXISTING}
        or not isinstance(value.get("target_basename"), str)
        or Path(value["target_basename"]).name != value["target_basename"]
        or any(not isinstance(identity, list) for identity in identities)
        or len(identities[0]) != 2
        or len(identities[1]) != 5
        or any(type(item) is not int for identity in identities for item in identity)
        or type(value.get("desired_bytes")) is not int
        or value["desired_bytes"] < 0
        or any(
            LOWER_SHA256_RE.fullmatch(str(value.get(key))) is None
            for key in ("transaction_sha256", "desired_sha256", "admitted_rows_sha256")
        )
        or value.get("false_authority") != {"score_authority": False, "promotion_eligible": False}
    ):
        raise FeatureCacheError(f"atomic completion record is malformed; preserving bytes: {path}")
    if (
        value.get("consumer_authorization_sha256") is not None
        and LOWER_SHA256_RE.fullmatch(str(value["consumer_authorization_sha256"])) is None
    ):
        raise FeatureCacheError(f"atomic completion authorization digest is malformed; preserving bytes: {path}")
    if value["mode"] == ATOMIC_MODE_FRESH:
        if any(value.get(key) is not None for key in ("prior_bytes", "prior_sha256", "prior_file_identity")):
            raise FeatureCacheError(f"fresh atomic completion prior is malformed; preserving bytes: {path}")
    else:
        prior_identity = value.get("prior_file_identity")
        if (
            type(value.get("prior_bytes")) is not int
            or value["prior_bytes"] < 0
            or LOWER_SHA256_RE.fullmatch(str(value.get("prior_sha256"))) is None
            or not isinstance(prior_identity, list)
            or len(prior_identity) != 5
            or any(type(item) is not int for item in prior_identity)
        ):
            raise FeatureCacheError(f"existing atomic completion prior is malformed; preserving bytes: {path}")
    match = ATOMIC_COMPLETION_RE.fullmatch(path.name)
    if match is not None and match.group("transaction") != value["transaction_sha256"]:
        raise FeatureCacheError(f"atomic completion filename digest mismatch; preserving bytes: {path}")
    return value


def _atomic_scratch_row(
    candidate: Path,
    snapshot: BoundFileSnapshot,
    *,
    source: Path,
    mode: str,
    displaced_prior: BoundFileSnapshot | None,
) -> dict[str, Any]:
    if candidate == source and mode == ATOMIC_MODE_EXISTING:
        if displaced_prior is None:  # pragma: no cover - caller contract
            raise FeatureCacheError("existing atomic transaction lacks its displaced prior")
        outcome_snapshot = displaced_prior
        role = "DISPLACED_PRIOR_SOURCE"
    elif candidate == source:
        outcome_snapshot = snapshot
        role = "DESIGNATED_DESIRED_SOURCE"
    else:
        outcome_snapshot = snapshot
        role = "ADMITTED_WRITER_SCRATCH"
    return {
        "basename": candidate.name,
        "bytes": len(outcome_snapshot.payload),
        "sha256": hashlib.sha256(outcome_snapshot.payload).hexdigest(),
        "file_identity": list(outcome_snapshot.file_identity),
        "role": role,
    }


def _atomic_transaction_record(
    path: Path,
    payload: bytes,
    *,
    source: Path,
    scratch: Sequence[Path],
    source_snapshot: BoundFileSnapshot,
    parent_identity: tuple[int, int],
    mode: str,
    prior: BoundFileSnapshot | None,
    prior_authorization: str,
    consumer_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in {ATOMIC_MODE_FRESH, ATOMIC_MODE_EXISTING}:
        raise FeatureCacheError("atomic transaction mode is malformed")
    expected_prior_authorization = ATOMIC_PRIOR_AUTH_ABSENT if mode == ATOMIC_MODE_FRESH else ATOMIC_PRIOR_AUTH_CONSUMER
    if prior_authorization != expected_prior_authorization:
        raise FeatureCacheError("atomic prior authorization kind is malformed")
    if mode == ATOMIC_MODE_FRESH and prior is not None:
        raise FeatureCacheError("fresh atomic transaction unexpectedly has prior target custody")
    if mode == ATOMIC_MODE_EXISTING and prior is None:
        raise FeatureCacheError("existing atomic transaction lacks prior target custody")
    if (
        not isinstance(parent_identity, tuple)
        or len(parent_identity) != 2
        or any(type(value) is not int for value in parent_identity)
    ):
        raise FeatureCacheError("atomic transaction parent identity is malformed")
    if source_snapshot.payload != payload:
        raise FeatureCacheError("atomic transaction designated source is not the desired payload")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(scratch, key=lambda value: value.name):
        snapshot = read_bound_file(candidate, role="atomic transaction admitted scratch")
        if candidate == source and snapshot != source_snapshot:
            raise FeatureCacheError(
                f"atomic transaction designated source changed before record creation; preserving bytes: {candidate}"
            )
        if snapshot.payload != payload and not payload.startswith(snapshot.payload):
            raise FeatureCacheError(
                f"atomic transaction scratch is not a reachable writer payload; preserving bytes: {candidate}"
            )
        rows.append(
            _atomic_scratch_row(
                candidate,
                snapshot,
                source=source,
                mode=mode,
                displaced_prior=prior,
            )
        )
    source_role = "DESIGNATED_DESIRED_SOURCE" if mode == ATOMIC_MODE_FRESH else "DISPLACED_PRIOR_SOURCE"
    if sum(row["role"] == source_role for row in rows) != 1:
        raise FeatureCacheError("atomic transaction does not identify exactly one designated source")
    prior_payload = None if prior is None else prior.payload
    if consumer_authorization_sha256 is not None:
        consumer_authorization_sha256 = _require_sha256(
            consumer_authorization_sha256,
            name="atomic consumer authorization digest",
        )
    return {
        "schema": ATOMIC_TRANSACTION_SCHEMA,
        "mode": mode,
        "target_basename": path.name,
        "parent_identity": list(parent_identity),
        "desired_bytes": len(payload),
        "desired_sha256": hashlib.sha256(payload).hexdigest(),
        "designated_source_basename": source.name,
        "designated_source_identity": list(source_snapshot.file_identity),
        "prior_bytes": None if prior_payload is None else len(prior_payload),
        "prior_sha256": None if prior_payload is None else hashlib.sha256(prior_payload).hexdigest(),
        "prior_file_identity": None if prior is None else list(prior.file_identity),
        "prior_authorization": prior_authorization,
        "consumer_authorization_sha256": consumer_authorization_sha256,
        "admitted_scratch": rows,
        "linearization": (
            "DESCRIPTOR_RELATIVE_RENAME_NOREPLACE"
            if mode == ATOMIC_MODE_FRESH
            else "DESCRIPTOR_RELATIVE_ATOMIC_EXCHANGE"
        ),
        "false_authority": {
            "score_authority": False,
            "promotion_eligible": False,
        },
    }


def _parse_atomic_transaction(path: Path, raw: bytes) -> dict[str, Any] | None:
    """Parse a complete record; return ``None`` for a preserved partial."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        return None
    expected_keys = {
        "schema",
        "mode",
        "target_basename",
        "parent_identity",
        "desired_bytes",
        "desired_sha256",
        "designated_source_basename",
        "designated_source_identity",
        "prior_bytes",
        "prior_sha256",
        "prior_file_identity",
        "prior_authorization",
        "consumer_authorization_sha256",
        "admitted_scratch",
        "linearization",
        "false_authority",
    }
    rows = value.get("admitted_scratch")
    mode = value.get("mode")
    parent_identity = value.get("parent_identity")
    source_identity = value.get("designated_source_identity")
    prior_identity = value.get("prior_file_identity")
    if (
        set(value) != expected_keys
        or value.get("schema") != ATOMIC_TRANSACTION_SCHEMA
        or mode not in {ATOMIC_MODE_FRESH, ATOMIC_MODE_EXISTING}
        or not isinstance(value.get("target_basename"), str)
        or not isinstance(parent_identity, list)
        or len(parent_identity) != 2
        or any(type(item) is not int for item in parent_identity)
        or type(value.get("desired_bytes")) is not int
        or value["desired_bytes"] < 0
        or LOWER_SHA256_RE.fullmatch(str(value.get("desired_sha256"))) is None
        or not isinstance(value.get("designated_source_basename"), str)
        or not value["designated_source_basename"]
        or Path(value["designated_source_basename"]).name != value["designated_source_basename"]
        or not isinstance(source_identity, list)
        or len(source_identity) != 5
        or any(type(item) is not int for item in source_identity)
        or not isinstance(rows, list)
        or not rows
        or value.get("false_authority") != {"score_authority": False, "promotion_eligible": False}
    ):
        raise FeatureCacheError(f"atomic transaction record is malformed; preserving bytes: {path}")
    if (
        value.get("consumer_authorization_sha256") is not None
        and LOWER_SHA256_RE.fullmatch(str(value["consumer_authorization_sha256"])) is None
    ):
        raise FeatureCacheError(f"atomic consumer authorization digest is malformed; preserving bytes: {path}")
    if mode == ATOMIC_MODE_FRESH:
        if (
            value.get("prior_authorization") != ATOMIC_PRIOR_AUTH_ABSENT
            or value.get("prior_bytes") is not None
            or value.get("prior_sha256") is not None
            or prior_identity is not None
            or value.get("linearization") != "DESCRIPTOR_RELATIVE_RENAME_NOREPLACE"
        ):
            raise FeatureCacheError(f"fresh atomic transaction prior is malformed; preserving bytes: {path}")
    elif (
        value.get("prior_authorization") != ATOMIC_PRIOR_AUTH_CONSUMER
        or type(value.get("prior_bytes")) is not int
        or value["prior_bytes"] < 0
        or LOWER_SHA256_RE.fullmatch(str(value.get("prior_sha256"))) is None
        or not isinstance(prior_identity, list)
        or len(prior_identity) != 5
        or any(type(item) is not int for item in prior_identity)
        or value.get("linearization") != "DESCRIPTOR_RELATIVE_ATOMIC_EXCHANGE"
    ):
        raise FeatureCacheError(f"existing atomic transaction prior is malformed; preserving bytes: {path}")
    seen: set[str] = set()
    designated_sources = 0
    expected_source_role = "DESIGNATED_DESIRED_SOURCE" if mode == ATOMIC_MODE_FRESH else "DISPLACED_PRIOR_SOURCE"
    for row in rows:
        row_identity = row.get("file_identity") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or set(row) != {"basename", "bytes", "sha256", "file_identity", "role"}
            or not isinstance(row.get("basename"), str)
            or not row["basename"]
            or Path(row["basename"]).name != row["basename"]
            or row["basename"] in seen
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or LOWER_SHA256_RE.fullmatch(str(row.get("sha256"))) is None
            or not isinstance(row_identity, list)
            or len(row_identity) != 5
            or any(type(item) is not int for item in row_identity)
            or row.get("role")
            not in {
                "DESIGNATED_DESIRED_SOURCE",
                "DISPLACED_PRIOR_SOURCE",
                "ADMITTED_WRITER_SCRATCH",
            }
        ):
            raise FeatureCacheError(f"atomic transaction scratch row is malformed; preserving bytes: {path}")
        seen.add(row["basename"])
        designated_sources += row["role"] == expected_source_role
        if row["role"] != "ADMITTED_WRITER_SCRATCH" and row["basename"] != value["designated_source_basename"]:
            raise FeatureCacheError(f"atomic transaction source basename is malformed; preserving bytes: {path}")
    if designated_sources != 1 or any(
        row["role"] not in {expected_source_role, "ADMITTED_WRITER_SCRATCH"} for row in rows
    ):
        raise FeatureCacheError(f"atomic transaction source cardinality is malformed; preserving bytes: {path}")
    source_row = next(row for row in rows if row["role"] == expected_source_role)
    if mode == ATOMIC_MODE_FRESH:
        if (
            source_row["bytes"] != value["desired_bytes"]
            or source_row["sha256"] != value["desired_sha256"]
            or source_row["file_identity"] != value["designated_source_identity"]
        ):
            raise FeatureCacheError(f"fresh atomic source binding is malformed; preserving bytes: {path}")
    elif (
        source_row["bytes"] != value["prior_bytes"]
        or source_row["sha256"] != value["prior_sha256"]
        or source_row["file_identity"] != value["prior_file_identity"]
    ):
        raise FeatureCacheError(f"existing atomic prior-source binding is malformed; preserving bytes: {path}")
    return value


def _classify_atomic_transactions(
    path: Path,
    *,
    expected_record: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[Path, BoundFileSnapshot, dict[str, Any] | None]], list[dict[str, Any]]]:
    """Read every active record and validate complete/partial generations."""

    records: list[tuple[Path, BoundFileSnapshot, dict[str, Any] | None]] = []
    complete: list[dict[str, Any]] = []
    partial: list[tuple[Path, BoundFileSnapshot]] = []
    expected_payload = canonical_json_bytes(dict(expected_record)) + b"\n" if expected_record is not None else None
    for candidate in _atomic_transaction_paths(path):
        snapshot = read_bound_file(candidate, role="atomic transaction record")
        parsed = _parse_atomic_transaction(candidate, snapshot.payload)
        if parsed is None:
            partial.append((candidate, snapshot))
        else:
            if parsed.get("target_basename") != path.name:
                raise FeatureCacheError(
                    f"atomic transaction target role is inconsistent; preserving bytes: {candidate}"
                )
            complete.append(parsed)
        records.append((candidate, snapshot, parsed))
    complete_payloads = [canonical_json_bytes(row) + b"\n" for row in complete]
    for candidate, snapshot in partial:
        authorities = [expected_payload] if expected_payload is not None else complete_payloads
        if not authorities or not any(
            authority is not None and len(snapshot.payload) < len(authority) and authority.startswith(snapshot.payload)
            for authority in authorities
        ):
            raise FeatureCacheError(
                f"atomic transaction partial is not an exact reachable prefix; preserving bytes: {candidate}"
            )
    return records, complete


def _write_atomic_transaction_generation(path: Path, record: Mapping[str, Any]) -> Path:
    transactions, complete = _classify_atomic_transactions(path, expected_record=record)
    normalized = json.loads(canonical_json_bytes(dict(record)))
    if any(row == normalized for row in complete):
        return next(candidate for candidate, _snapshot, parsed in transactions if parsed == normalized)
    if complete:
        raise FeatureCacheError("contradictory complete atomic transaction evidence; preserving bytes")
    existing_indices = [int(candidate.name.rsplit("-", 1)[1]) for candidate, _snapshot, _parsed in transactions]
    generation = atomic_transaction_path(path, max(existing_indices, default=-1) + 1)
    payload = canonical_json_bytes(normalized) + b"\n"
    descriptor = -1
    try:
        descriptor = os.open(generation, _nofollow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL), 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        _assert_bound_file(generation, descriptor, opened, role="atomic transaction generation")
    except OSError as exc:
        raise FeatureCacheError(f"atomic transaction generation write failed; preserving bytes: {generation}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    return generation


def _publish_atomic_completion(
    path: Path,
    *,
    transaction: Mapping[str, Any],
    target: BoundFileSnapshot,
) -> Path:
    """Durably publish one immutable transaction completion via no-replace."""

    record = _atomic_completion_record(path, transaction=transaction, target=target)
    payload = canonical_json_bytes(record) + b"\n"
    transaction_sha256 = record["transaction_sha256"]
    final = atomic_completion_path(path, transaction_sha256)
    finals, generations = _atomic_completion_paths(path)
    matching_finals: list[Path] = []
    for candidate in finals:
        snapshot = read_bound_file(candidate, role="atomic completion proof")
        parsed = _parse_atomic_completion(candidate, snapshot.payload)
        if parsed is None:
            raise FeatureCacheError(f"partial payload occupies atomic completion authority: {candidate}")
        if candidate == final and parsed == record:
            matching_finals.append(candidate)
        elif parsed.get("transaction_sha256") == transaction_sha256:
            raise FeatureCacheError("duplicate or contradictory atomic completion proof; preserving bytes")
    for candidate in generations:
        match = ATOMIC_COMPLETION_GENERATION_RE.fullmatch(candidate.name)
        assert match is not None
        if match.group("transaction") != transaction_sha256:
            continue
        snapshot = read_bound_file(candidate, role="atomic completion construction")
        if snapshot.payload != payload and (
            len(snapshot.payload) >= len(payload) or not payload.startswith(snapshot.payload)
        ):
            raise FeatureCacheError("atomic completion construction is not an exact reachable prefix")
    if matching_finals:
        _fsync_file(matching_finals[0])
        _fsync_directory(path.parent)
        return matching_finals[0]
    indices = [
        int(candidate.name.rsplit("-", 1)[1])
        for candidate in generations
        if ATOMIC_COMPLETION_GENERATION_RE.fullmatch(candidate.name) is not None
        and ATOMIC_COMPLETION_GENERATION_RE.fullmatch(candidate.name).group("transaction") == transaction_sha256
    ]
    generation = atomic_completion_generation_path(path, transaction_sha256, max(indices, default=-1) + 1)
    descriptor = -1
    try:
        descriptor = os.open(generation, _nofollow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL), 0o600)
        hook = _ATOMIC_COMPLETION_TEST_HOOK
        if hook is not None:
            try:
                hook("before_completion_write", generation)
            except BaseException as exc:
                raise FeatureCacheError(
                    f"atomic completion pre-write interruption; preserving construction: {generation}"
                ) from exc
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        _assert_bound_file(generation, descriptor, opened, role="atomic completion construction")
        if hook is not None:
            try:
                hook("after_completion_fsync", generation)
            except BaseException as exc:
                raise FeatureCacheError(
                    f"atomic completion post-fsync interruption; preserving construction: {generation}"
                ) from exc
    except OSError as exc:
        raise FeatureCacheError(f"atomic completion construction failed; preserving bytes: {generation}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    parent_descriptor, _opened = _open_bound_parent_directory(path.parent, role="atomic completion parent")
    try:
        if _path_exists_no_follow(final):
            raise FeatureCacheError("atomic completion destination appeared; preserving construction")
        _direct_noreplace_move(parent_descriptor, generation.name, final.name)
        os.fsync(parent_descriptor)
    except OSError as exc:
        raise FeatureCacheError(f"atomic completion publication failed; preserving bytes: {generation}") from exc
    finally:
        os.close(parent_descriptor)
    final_snapshot = read_bound_file(final, role="published atomic completion")
    if final_snapshot.payload != payload or _parse_atomic_completion(final, final_snapshot.payload) != record:
        raise FeatureCacheError("published atomic completion differs from its construction")
    # A strict-prefix construction is non-authority, but it is still immutable
    # crash evidence.  Retire it losslessly after the final proof is durable.
    _finals, remaining_generations = _atomic_completion_paths(path)
    for candidate in remaining_generations:
        match = ATOMIC_COMPLETION_GENERATION_RE.fullmatch(candidate.name)
        assert match is not None
        if match.group("transaction") != transaction_sha256:
            continue
        snapshot = read_bound_file(candidate, role="atomic completion construction retirement")
        if snapshot.payload != payload and not (
            len(snapshot.payload) < len(payload) and payload.startswith(snapshot.payload)
        ):
            raise FeatureCacheError("atomic completion construction drifted before retirement")
        retain_bound_file(candidate, snapshot, role="atomic completion construction retirement")
    _fsync_directory(path.parent)
    return final


def extractor_receipt_path(cache_root: str | Path) -> Path:
    """Return the deterministic operational-receipt sibling of a cache root."""

    root = _unresolved_absolute(cache_root)
    return root.with_name(f"{root.name}.extractor-receipt.json")


def _read_stable_bytes(path: Path, *, role: str) -> bytes:
    return read_bound_file(path, role=role).payload


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive kernel contract
            raise OSError("descriptor write made no progress")
        view = view[written:]


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 8 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _existing_filesystem_anchor(path: Path) -> Path:
    """Return the nearest existing no-symlink directory for a future path."""

    cursor = _unresolved_absolute(path)
    while not _path_exists_no_follow(cursor):
        if cursor == cursor.parent:
            raise FeatureCacheError(f"no existing filesystem anchor for path: {path}")
        cursor = cursor.parent
    _require_directory_chain(cursor)
    _require_directory(cursor, role="component-limit filesystem anchor")
    return cursor


def _filesystem_name_max(parent: Path) -> int:
    """Query the governing filesystem's real component limit without mutation."""

    anchor = _existing_filesystem_anchor(parent)
    descriptor, _opened = _open_bound_parent_directory(anchor, role="component-limit filesystem anchor")
    try:
        try:
            value = os.fpathconf(descriptor, "PC_NAME_MAX")
        except (OSError, ValueError) as exc:
            raise FeatureCacheError(f"filesystem NAME_MAX is unavailable for {parent}") from exc
    finally:
        os.close(descriptor)
    if type(value) is not int or value <= 0:
        raise FeatureCacheError(f"filesystem NAME_MAX is malformed for {parent}")
    return value


def _require_component_fits(name: str, *, name_max: int, role: str) -> None:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise FeatureCacheError(f"{role} basename is malformed")
    if len(os.fsencode(name)) > name_max:
        raise FeatureCacheError(
            f"{role} basename exceeds actual filesystem NAME_MAX={name_max}; refusing before mutation"
        )


def _encoded_original_basename(name: str) -> str:
    return "z" + base64.urlsafe_b64encode(zlib.compress(os.fsencode(name), level=9)).rstrip(b"=").decode("ascii")


def _retained_component_name(original: str, *, payload_bytes: int = 0, payload_sha256: str = "0" * 64) -> str:
    if payload_bytes < 0 or payload_bytes >= 10**ATOMIC_RETAINED_BYTE_WIDTH:
        raise FeatureCacheError("payload is too large for canonical retention naming")
    return (
        f"{ATOMIC_RETAINED_PREFIX}{_encoded_original_basename(original)}-{payload_bytes:020d}-{payload_sha256}-00000000"
    )


def _preflight_atomic_components(path: Path, payload: bytes) -> None:
    """Prove all names an atomic write can need before its first mutation."""

    name_max = _filesystem_name_max(path.parent)
    candidates = [
        path.name,
        atomic_prepared_path(path).name,
        atomic_generation_path(path, 0).name,
        atomic_transaction_path(path, 0).name,
        atomic_completion_path(path, "0" * 64).name,
        atomic_completion_generation_path(path, "0" * 64, 0).name,
    ]
    candidates.extend(_retained_component_name(name) for name in candidates[1:])
    for candidate in candidates:
        _require_component_fits(candidate, name_max=name_max, role=f"atomic metadata {path.name}")
    if len(payload) >= 10**ATOMIC_RETAINED_BYTE_WIDTH:
        raise FeatureCacheError("atomic metadata payload exceeds retention-name byte width")


def _require_noreplace_support() -> None:
    """Fail before mutation when the host has no required rename primitive."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        return
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        return
    raise FeatureCacheError("host lacks required descriptor-relative no-replace rename support")


def _direct_noreplace_move(parent_descriptor: int, source_name: str, destination_name: str) -> None:
    """Rename one local entry without replacement through an open parent."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            0x00000001,  # RENAME_NOREPLACE
        )
    else:  # pragma: no cover - checked before any mutation on authority hosts
        raise FeatureCacheError("host lacks required descriptor-relative no-replace rename support")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _open_bound_parent_directory(path: Path, *, role: str) -> tuple[int, os.stat_result]:
    """Open a no-follow directory and prove the pathname names that inode."""

    _require_directory_chain(path)
    before = _require_directory(path, role=role)
    flags = _nofollow_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FeatureCacheError(f"{role} cannot be opened without following links: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        after = path.lstat()
        identities = {
            (before.st_dev, before.st_ino),
            (opened.st_dev, opened.st_ino),
            (after.st_dev, after.st_ino),
        }
        if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(after.st_mode) or len(identities) != 1:
            raise FeatureCacheError(f"{role} pathname identity changed: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, opened


def _entry_metadata_at(parent_descriptor: int, name: str, *, role: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise FeatureCacheError(f"{role} identity cannot be read; preserving bytes: {name}") from exc


def _entry_exists_at(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise FeatureCacheError(f"path custody cannot be read; preserving bytes: {name}") from exc
    return True


def _assert_bound_move_entry(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    opened: os.stat_result,
    expected_identity: tuple[int, int, int, int, int],
    *,
    role: str,
    require_directory: bool,
) -> os.stat_result:
    """Require one relative pathname to name the descriptor-bound entry."""

    descriptor_metadata = os.fstat(descriptor)
    path_metadata = _entry_metadata_at(parent_descriptor, name, role=role)
    expected_kind = stat.S_ISDIR if require_directory else stat.S_ISREG
    if (
        not expected_kind(descriptor_metadata.st_mode)
        or not expected_kind(path_metadata.st_mode)
        or (not require_directory and (descriptor_metadata.st_nlink != 1 or path_metadata.st_nlink != 1))
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        or (opened.st_dev, opened.st_ino) != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        or _stable_file_identity(descriptor_metadata) != expected_identity
        or _stable_file_identity(path_metadata) != expected_identity
    ):
        raise FeatureCacheError(f"{role} pathname identity changed; preserving bytes: {name}")
    return descriptor_metadata


def _move_path_noreplace(
    source: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int, int, int, int],
    role: str,
    require_directory: bool,
    expected_payload: bytes | None,
) -> None:
    """Descriptor-bind, no-replace move, validate, and rollback on mismatch."""

    _require_noreplace_support()
    source = _unresolved_absolute(source)
    destination = _unresolved_absolute(destination)
    if source == destination or source.parent != destination.parent:
        raise FeatureCacheError(f"{role} move requires distinct names in one exact parent")
    if (
        not isinstance(expected_identity, tuple)
        or len(expected_identity) != 5
        or any(type(value) is not int for value in expected_identity)
    ):
        raise FeatureCacheError(f"{role} expected identity is malformed")

    parent_descriptor, _parent_opened = _open_bound_parent_directory(
        source.parent,
        role=f"{role} move parent",
    )
    source_descriptor = -1
    try:
        flags = os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if require_directory else 0)
        try:
            source_descriptor = os.open(source.name, _nofollow_flags(flags), dir_fd=parent_descriptor)
        except OSError as exc:
            raise FeatureCacheError(f"{role} source cannot be descriptor-bound; preserving bytes: {source}") from exc
        source_opened = os.fstat(source_descriptor)
        _assert_bound_move_entry(
            parent_descriptor,
            source.name,
            source_descriptor,
            source_opened,
            expected_identity,
            role=f"{role} source",
            require_directory=require_directory,
        )
        admitted_payload = None if require_directory else _read_descriptor(source_descriptor)
        if expected_payload is not None and admitted_payload != expected_payload:
            raise FeatureCacheError(f"{role} source payload mismatch; preserving bytes: {source}")
        if _entry_exists_at(parent_descriptor, destination.name):
            raise _NoReplaceDestinationExists(f"{role} destination already exists; preserving bytes: {destination}")

        # This is the last production authorization check before the syscall.
        _assert_bound_move_entry(
            parent_descriptor,
            source.name,
            source_descriptor,
            source_opened,
            expected_identity,
            role=f"{role} source",
            require_directory=require_directory,
        )
        if not require_directory and _read_descriptor(source_descriptor) != admitted_payload:
            raise FeatureCacheError(f"{role} source payload changed; preserving bytes: {source}")
        hook = _MOVE_PATH_NOREPLACE_TEST_HOOK
        if hook is not None:
            try:
                hook(source, destination)
            except BaseException as exc:
                raise FeatureCacheError(f"{role} pre-move interruption; preserving bytes: {source}") from exc
        try:
            _direct_noreplace_move(parent_descriptor, source.name, destination.name)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise _NoReplaceDestinationExists(
                    f"{role} destination appeared; preserving bytes: {destination}"
                ) from exc
            raise FeatureCacheError(f"{role} no-replace syscall failed; preserving bytes: {source}") from exc

        post_move_error: BaseException | None = None
        try:
            post_move_hook = _MOVE_PATH_NOREPLACE_POST_MOVE_TEST_HOOK
            if post_move_hook is not None:
                post_move_hook(source, destination)
            _assert_bound_move_entry(
                parent_descriptor,
                destination.name,
                source_descriptor,
                source_opened,
                expected_identity,
                role=f"{role} destination",
                require_directory=require_directory,
            )
            if not require_directory and _read_descriptor(source_descriptor) != admitted_payload:
                raise FeatureCacheError(f"{role} destination payload changed; preserving bytes")
        except BaseException as exc:
            post_move_error = exc
        if post_move_error is not None:
            try:
                _direct_noreplace_move(parent_descriptor, destination.name, source.name)
                _assert_bound_move_entry(
                    parent_descriptor,
                    source.name,
                    source_descriptor,
                    source_opened,
                    expected_identity,
                    role=f"{role} rollback source",
                    require_directory=require_directory,
                )
                if not require_directory and _read_descriptor(source_descriptor) != admitted_payload:
                    raise FeatureCacheError(f"{role} rollback payload changed; preserving bytes")
                if _entry_exists_at(parent_descriptor, destination.name):
                    raise FeatureCacheError(f"{role} rollback destination remained present")
            except (OSError, FeatureCacheError) as rollback_error:
                raise FeatureCacheError(
                    f"{role} rollback uncertainty after no-replace move; preserving every pathname"
                ) from rollback_error
            raise FeatureCacheError(f"{role} post-move mismatch rolled back; preserving bytes") from post_move_error
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise FeatureCacheError(
                f"{role} parent fsync failed after no-replace move; retry required: {destination}"
            ) from exc
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(parent_descriptor)


def move_path_noreplace(
    source: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int, int, int, int],
    role: str,
    require_directory: bool = False,
) -> None:
    """Move a descriptor-bound file/directory without replacing a late name."""

    _move_path_noreplace(
        source,
        destination,
        expected_identity=expected_identity,
        role=role,
        require_directory=require_directory,
        expected_payload=None,
    )


def move_bound_file_noreplace(
    source: Path,
    destination: Path,
    snapshot: BoundFileSnapshot,
    *,
    role: str,
) -> None:
    """Move exactly the descriptor-read file snapshot without replacement."""

    if not isinstance(snapshot, BoundFileSnapshot):
        raise FeatureCacheError(f"{role} bound-file snapshot is malformed")
    _move_path_noreplace(
        source,
        destination,
        expected_identity=snapshot.file_identity,
        role=role,
        require_directory=False,
        expected_payload=snapshot.payload,
    )


def is_retained_name(name: str) -> bool:
    """Return whether ``name`` has the canonical retained-custody grammar."""

    return isinstance(name, str) and ATOMIC_RETAINED_RE.fullmatch(name) is not None


def retained_original_name(name: str) -> str:
    """Decode the exact original basename from a canonical retention name."""

    match = ATOMIC_RETAINED_RE.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        raise FeatureCacheError("retained custody name is malformed")
    encoded = match.group("original")
    compressed = encoded[1:]
    padding = "=" * (-len(compressed) % 4)
    try:
        compressed_bytes = base64.b64decode(compressed + padding, altchars=b"-_", validate=True)
        decompressor = zlib.decompressobj()
        original_bytes = decompressor.decompress(compressed_bytes, ATOMIC_RETAINED_COMPONENT_MAX + 1)
    except (ValueError, binascii.Error, zlib.error) as exc:
        raise FeatureCacheError("retained custody original basename encoding is malformed") from exc
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(original_bytes) > ATOMIC_RETAINED_COMPONENT_MAX
    ):
        raise FeatureCacheError("retained custody original basename encoding is malformed")
    original = os.fsdecode(original_bytes)
    canonical_encoded = "z" + base64.urlsafe_b64encode(zlib.compress(original_bytes, level=9)).rstrip(b"=").decode(
        "ascii"
    )
    if not original or original in {".", ".."} or Path(original).name != original or canonical_encoded != encoded:
        raise FeatureCacheError("retained custody original basename encoding is noncanonical")
    return original


def validate_retained_file(path: Path, *, role: str) -> BoundFileSnapshot:
    """Reopen and validate one self-identifying false-authority retention."""

    retained = Path(path)
    match = ATOMIC_RETAINED_RE.fullmatch(retained.name)
    if match is None:
        raise FeatureCacheError(f"{role} retained custody name is malformed: {retained}")
    retained_original_name(retained.name)
    snapshot = read_bound_file(retained, role=role)
    encoded_size = int(match.group("bytes"))
    encoded_sha256 = match.group("sha256")
    if len(snapshot.payload) != encoded_size or hashlib.sha256(snapshot.payload).hexdigest() != encoded_sha256:
        raise FeatureCacheError(f"{role} retained custody byte/hash mismatch; preserving bytes: {retained}")
    return snapshot


def retain_bound_file(path: Path, snapshot: BoundFileSnapshot, *, role: str) -> Path:
    """Move exact file custody to the first absent self-identifying sibling."""

    source = _unresolved_absolute(path)
    if is_retained_name(source.name):
        raise FeatureCacheError(f"{role} is already retained custody: {source}")
    if not isinstance(snapshot, BoundFileSnapshot):
        raise FeatureCacheError(f"{role} bound-file snapshot is malformed")
    encoded_original = _encoded_original_basename(source.name)
    payload_sha256 = hashlib.sha256(snapshot.payload).hexdigest()
    if len(snapshot.payload) >= 10**ATOMIC_RETAINED_BYTE_WIDTH:
        raise FeatureCacheError(f"{role} payload is too large for canonical retention naming")
    name_max = _filesystem_name_max(source.parent)
    for ordinal in range(100_000_000):
        retained = source.with_name(
            f"{ATOMIC_RETAINED_PREFIX}{encoded_original}-{len(snapshot.payload):020d}-{payload_sha256}-{ordinal:08d}"
        )
        if len(os.fsencode(retained.name)) > name_max:
            raise FeatureCacheError(
                f"{role} original basename cannot fit canonical retention naming under actual "
                f"NAME_MAX={name_max}; preserving bytes"
            )
        try:
            move_bound_file_noreplace(source, retained, snapshot, role=role)
        except _NoReplaceDestinationExists:
            continue
        validated = validate_retained_file(retained, role=f"{role} retained custody")
        if validated.payload != snapshot.payload or validated.file_identity != snapshot.file_identity:
            raise FeatureCacheError(f"{role} retained custody changed after move; preserving bytes")
        return retained
    raise FeatureCacheError(f"{role} exhausted canonical retention ordinals; preserving bytes")


def _direct_atomic_exchange(parent_descriptor: int, source_name: str, target_name: str) -> None:
    """Atomically exchange two local names or fail closed without mutation."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        exchange = libc.renameatx_np
        exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        exchange.restype = ctypes.c_int
        result = exchange(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(target_name),
            0x00000002,  # RENAME_SWAP
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        exchange = libc.renameat2
        exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        exchange.restype = ctypes.c_int
        result = exchange(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(target_name),
            0x00000002,  # RENAME_EXCHANGE
        )
    else:  # pragma: no cover - authority platforms are Darwin/Linux
        raise FeatureCacheError("host lacks required descriptor-relative atomic exchange support")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _classify_atomic_scratch(path: Path, payload: bytes) -> tuple[list[Path], Path | None]:
    """Validate every scratch inode and select an exact complete source."""

    prepared = atomic_prepared_path(path)
    scratch = ([prepared] if _path_exists_no_follow(prepared) else []) + _atomic_generation_paths(path)
    complete: Path | None = None
    for candidate in scratch:
        candidate_payload = read_bound_file(candidate, role="atomic metadata scratch").payload
        if candidate_payload == payload:
            if complete is None:
                complete = candidate
        elif len(candidate_payload) >= len(payload) or not payload.startswith(candidate_payload):
            raise FeatureCacheError(f"stable atomic prepared payload drift; preserving bytes: {candidate}")
    return scratch, complete


def _create_complete_generation(path: Path, payload: bytes, scratch: Sequence[Path]) -> Path:
    existing_indices = [
        int(candidate.name.rsplit("-", 1)[1])
        for candidate in scratch
        if ATOMIC_GENERATION_RE.fullmatch(candidate.name) is not None
    ]
    generation = atomic_generation_path(path, max(existing_indices, default=-1) + 1)
    descriptor = -1
    try:
        descriptor = os.open(generation, _nofollow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL), 0o666)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        _assert_bound_file(generation, descriptor, opened, role="atomic metadata generation")
    except OSError as exc:
        raise FeatureCacheError(f"atomic metadata generation write failed; preserving bytes: {generation}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    return generation


def _retained_atomic_namespace_fingerprint(
    path: Path,
) -> AtomicNamespaceFingerprint:
    """Bind all target-scoped retained evidence at an authorization boundary."""

    rows: list[AtomicNamespaceRow] = []
    for candidate in path.parent.iterdir():
        if not is_retained_name(candidate.name):
            continue
        original = retained_original_name(candidate.name)
        if _atomic_original_role_for_target(original, target_name=path.name) is None:
            continue
        snapshot = validate_retained_file(candidate, role="atomic retained-boundary custody")
        rows.append(
            (
                candidate.name,
                snapshot.file_identity,
                len(snapshot.payload),
                hashlib.sha256(snapshot.payload).hexdigest(),
            )
        )
    return tuple(sorted(rows))


def _bound_atomic_paths_fingerprint(
    paths: Sequence[Path],
    *,
    role: str,
) -> AtomicNamespaceFingerprint:
    rows: list[AtomicNamespaceRow] = []
    for candidate in paths:
        snapshot = read_bound_file(candidate, role=role)
        rows.append(
            (
                candidate.name,
                snapshot.file_identity,
                len(snapshot.payload),
                hashlib.sha256(snapshot.payload).hexdigest(),
            )
        )
    return tuple(sorted(rows))


def _atomic_scratch_namespace_fingerprint(
    path: Path,
    payload: bytes,
) -> AtomicNamespaceFingerprint:
    """Descriptor-read every active writer scratch role for one exact payload."""

    scratch, _complete = _classify_atomic_scratch(path, payload)
    return _bound_atomic_paths_fingerprint(scratch, role="atomic scratch-boundary custody")


def _active_atomic_scratch_namespace_fingerprint(path: Path) -> AtomicNamespaceFingerprint:
    """Bind active scratch roles without assigning pre/post-exchange payload meaning."""

    prepared = atomic_prepared_path(path)
    scratch = ([prepared] if _path_exists_no_follow(prepared) else []) + _atomic_generation_paths(path)
    return _bound_atomic_paths_fingerprint(scratch, role="active atomic scratch-boundary custody")


def _atomic_transaction_namespace_fingerprint(path: Path) -> AtomicNamespaceFingerprint:
    """Descriptor-bind every active complete or authorized-prefix transaction."""

    evidence, _complete = _classify_atomic_transactions(path)
    return tuple(
        sorted(
            (
                candidate.name,
                snapshot.file_identity,
                len(snapshot.payload),
                hashlib.sha256(snapshot.payload).hexdigest(),
            )
            for candidate, snapshot, _parsed in evidence
        )
    )


def _observe_atomic_prepublication_namespace(
    path: Path,
) -> AtomicPrepublicationNamespace:
    """Enumerate every target-scoped writer role without authorizing mutation."""

    completions, completion_constructions = _atomic_completion_paths(path)
    return AtomicPrepublicationNamespace(
        active_scratch=_active_atomic_scratch_namespace_fingerprint(path),
        active_transactions=_atomic_transaction_namespace_fingerprint(path),
        retained_roles=_retained_atomic_namespace_fingerprint(path),
        active_completions=_bound_atomic_paths_fingerprint(
            completions,
            role="atomic completion-boundary custody",
        ),
        active_completion_constructions=_bound_atomic_paths_fingerprint(
            completion_constructions,
            role="atomic completion-construction-boundary custody",
        ),
    )


def _atomic_prepublication_stable_fingerprint(
    namespace: AtomicPrepublicationNamespace,
) -> AtomicNamespaceFingerprint:
    """Combine roles that an atomic construction call may never change."""

    return tuple(
        sorted(
            (
                *namespace.retained_roles,
                *namespace.active_completions,
                *namespace.active_completion_constructions,
            )
        )
    )


def _current_atomic_stable_fingerprint(path: Path) -> AtomicNamespaceFingerprint:
    """Re-enumerate retained and completion roles at a later boundary."""

    completions, completion_constructions = _atomic_completion_paths(path)
    return tuple(
        sorted(
            (
                *_retained_atomic_namespace_fingerprint(path),
                *_bound_atomic_paths_fingerprint(
                    completions,
                    role="atomic completion-boundary custody",
                ),
                *_bound_atomic_paths_fingerprint(
                    completion_constructions,
                    role="atomic completion-construction-boundary custody",
                ),
            )
        )
    )


def _validated_atomic_transaction_fingerprint(
    path: Path,
    *,
    expected_record: Mapping[str, Any],
) -> AtomicNamespaceFingerprint:
    """Require one exact transaction authority plus only its strict prefixes."""

    evidence, complete = _classify_atomic_transactions(path, expected_record=expected_record)
    normalized = json.loads(canonical_json_bytes(dict(expected_record)))
    if len(complete) != 1 or complete[0] != normalized:
        raise FeatureCacheError("atomic publication boundary lacks one exact transaction authority; preserving bytes")
    return tuple(
        sorted(
            (
                candidate.name,
                snapshot.file_identity,
                len(snapshot.payload),
                hashlib.sha256(snapshot.payload).hexdigest(),
            )
            for candidate, snapshot, _parsed in evidence
        )
    )


def _validate_atomic_publication_boundary(
    path: Path,
    payload: bytes,
    *,
    transaction_record: Mapping[str, Any],
    expected_scratch_namespace: AtomicNamespaceFingerprint,
    expected_transaction_namespace: AtomicNamespaceFingerprint,
    expected_stable_namespace: AtomicNamespaceFingerprint,
) -> None:
    """Revalidate every target-scoped role immediately before publication."""

    if _atomic_scratch_namespace_fingerprint(path, payload) != expected_scratch_namespace:
        raise FeatureCacheError(
            "target-scoped active atomic scratch changed at the final publication boundary; preserving bytes"
        )
    transaction_namespace = _validated_atomic_transaction_fingerprint(
        path,
        expected_record=transaction_record,
    )
    if transaction_namespace != expected_transaction_namespace:
        raise FeatureCacheError(
            "target-scoped active atomic transaction changed outside authorized construction; preserving bytes"
        )
    if _current_atomic_stable_fingerprint(path) != expected_stable_namespace:
        raise FeatureCacheError(
            "target-scoped retained/completion custody changed at the final publication boundary; preserving bytes"
        )


def _commit_bound_atomic_source(
    path: Path,
    source: Path,
    payload: bytes,
    *,
    expected_prior_payloads: Sequence[bytes],
    expected_scratch_namespace: AtomicNamespaceFingerprint,
    prepublication_transaction_namespace: AtomicNamespaceFingerprint,
    expected_stable_namespace: AtomicNamespaceFingerprint,
    consumer_authorization_sha256: str | None = None,
) -> None:
    """Publish one descriptor-bound source under durable transaction custody."""

    if source.parent != path.parent:
        raise FeatureCacheError("atomic source and target must share one exact parent")
    source_descriptor, source_opened = _open_bound_file(source, flags=os.O_RDONLY, role="atomic commit source")
    target_descriptor = -1
    parent_descriptor = -1
    target_opened: os.stat_result | None = None
    target_snapshot: BoundFileSnapshot | None = None
    transaction_descriptor = -1
    transaction_opened: os.stat_result | None = None
    transaction_path: Path | None = None
    exchanged = False
    fresh_published = False
    try:
        source_payload = _read_descriptor(source_descriptor)
        if source_payload != payload:
            raise FeatureCacheError(f"atomic commit source read mismatch; preserving bytes: {source}")
        source_snapshot = BoundFileSnapshot(
            payload=source_payload,
            file_identity=_stable_file_identity(source_opened),
        )
        if _path_exists_no_follow(path):
            target_descriptor, target_opened = _open_bound_file(path, flags=os.O_RDONLY, role="atomic prior target")
            target_snapshot = BoundFileSnapshot(
                payload=_read_descriptor(target_descriptor),
                file_identity=_stable_file_identity(target_opened),
            )
            if not any(target_snapshot.payload == expected for expected in expected_prior_payloads):
                raise FeatureCacheError(
                    f"atomic prior target is not a consumer-authorized exact state; preserving bytes: {path}"
                )
            mode = ATOMIC_MODE_EXISTING
        else:
            target_snapshot = None
            mode = ATOMIC_MODE_FRESH
        scratch, complete = _classify_atomic_scratch(path, payload)
        if complete != source:
            raise FeatureCacheError("atomic exchange source is no longer the selected complete generation")
        if _atomic_scratch_namespace_fingerprint(path, payload) != expected_scratch_namespace:
            raise FeatureCacheError(
                "target-scoped active atomic scratch changed outside authorized construction; preserving bytes"
            )
        parent_descriptor, parent_opened = _open_bound_parent_directory(
            path.parent,
            role="atomic metadata parent",
        )
        transaction_record = _atomic_transaction_record(
            path,
            payload,
            source=source,
            scratch=scratch,
            source_snapshot=source_snapshot,
            parent_identity=(parent_opened.st_dev, parent_opened.st_ino),
            mode=mode,
            prior=target_snapshot,
            prior_authorization=(ATOMIC_PRIOR_AUTH_ABSENT if mode == ATOMIC_MODE_FRESH else ATOMIC_PRIOR_AUTH_CONSUMER),
            consumer_authorization_sha256=consumer_authorization_sha256,
        )
        transaction_path = _write_atomic_transaction_generation(path, transaction_record)
        transaction_row = _bound_atomic_paths_fingerprint(
            (transaction_path,),
            role="atomic transaction authorization record",
        )[0]
        prepublication_transactions_by_name = {row[0]: row for row in prepublication_transaction_namespace}
        if transaction_path.name in prepublication_transactions_by_name:
            if transaction_row != prepublication_transactions_by_name[transaction_path.name]:
                raise FeatureCacheError(
                    "prepublication atomic transaction changed during explicit reuse; preserving bytes"
                )
            authorized_transaction_namespace = prepublication_transaction_namespace
        else:
            authorized_transaction_namespace = tuple(sorted((*prepublication_transaction_namespace, transaction_row)))
        transaction_descriptor, transaction_opened = _open_bound_file(
            transaction_path,
            flags=os.O_RDONLY,
            role="atomic transaction authorization record",
        )
        _validate_atomic_publication_boundary(
            path,
            payload,
            transaction_record=transaction_record,
            expected_scratch_namespace=expected_scratch_namespace,
            expected_transaction_namespace=authorized_transaction_namespace,
            expected_stable_namespace=expected_stable_namespace,
        )
        _assert_bound_file(source, source_descriptor, source_opened, role="atomic commit source")
        _assert_bound_file(
            transaction_path,
            transaction_descriptor,
            transaction_opened,
            role="atomic transaction authorization record",
        )
        # This is the final authorization boundary.  Production dispatches
        # directly to the platform syscall after these descriptor checks.
        _assert_bound_file(source, source_descriptor, source_opened, role="atomic commit source")
        _assert_bound_file(
            transaction_path,
            transaction_descriptor,
            transaction_opened,
            role="atomic transaction authorization record",
        )
        hook = _ATOMIC_COMMIT_TEST_HOOK
        if hook is not None:
            try:
                hook(source, path)
            except BaseException as exc:
                raise FeatureCacheError(f"atomic metadata pre-commit interruption; preserving bytes: {source}") from exc
        _validate_atomic_publication_boundary(
            path,
            payload,
            transaction_record=transaction_record,
            expected_scratch_namespace=expected_scratch_namespace,
            expected_transaction_namespace=authorized_transaction_namespace,
            expected_stable_namespace=expected_stable_namespace,
        )
        if mode == ATOMIC_MODE_FRESH:
            try:
                _direct_noreplace_move(parent_descriptor, source.name, path.name)
                fresh_published = True
            except OSError as exc:
                if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise FeatureCacheError(
                        f"fresh atomic destination appeared; preserving source and target bytes: {path}"
                    ) from exc
                raise FeatureCacheError(
                    f"fresh atomic no-replace publication failed; preserving bytes: {source}"
                ) from exc
            try:
                _assert_bound_file(path, source_descriptor, source_opened, role="fresh atomic committed target")
                if _read_descriptor(source_descriptor) != payload:
                    raise FeatureCacheError("fresh atomic committed target content identity mismatch")
            except FeatureCacheError as exc:
                raise FeatureCacheError(
                    f"fresh atomic post-publication identity mismatch; retry required: {path}"
                ) from exc
        else:
            assert target_descriptor >= 0 and target_opened is not None and target_snapshot is not None
            _assert_bound_file(path, target_descriptor, target_opened, role="atomic prior target")
            try:
                _direct_atomic_exchange(parent_descriptor, source.name, path.name)
                exchanged = True
            except OSError as exc:
                raise FeatureCacheError(f"atomic metadata commit syscall failed; preserving bytes: {source}") from exc
            try:
                _assert_bound_file(path, source_descriptor, source_opened, role="atomic committed target")
                _assert_bound_file(source, target_descriptor, target_opened, role="atomic displaced prior target")
                if _read_descriptor(source_descriptor) != payload:
                    raise FeatureCacheError("atomic committed target content identity mismatch")
            except FeatureCacheError as commit_error:
                try:
                    _direct_atomic_exchange(parent_descriptor, source.name, path.name)
                    exchanged = False
                    _assert_bound_file(path, target_descriptor, target_opened, role="atomic rollback target")
                except (OSError, FeatureCacheError) as rollback_error:
                    raise FeatureCacheError(
                        f"atomic metadata rollback uncertainty after commit identity mismatch: {path}"
                    ) from rollback_error
                raise FeatureCacheError(
                    f"atomic metadata commit identity mismatch rolled back: {path}"
                ) from commit_error
        _assert_bound_file(
            transaction_path,
            transaction_descriptor,
            transaction_opened,
            role="atomic transaction authorization record",
        )
        post_exchange_hook = _ATOMIC_POST_EXCHANGE_TEST_HOOK
        if post_exchange_hook is not None:
            try:
                post_exchange_hook(source, path)
            except BaseException as exc:
                raise FeatureCacheError(
                    f"atomic metadata post-linearization interruption; retry required: {path}"
                ) from exc
        try:
            os.fsync(source_descriptor)
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise FeatureCacheError(f"atomic committed metadata fsync failed; retry required: {path}") from exc
        if mode == ATOMIC_MODE_EXISTING:
            assert target_snapshot is not None
            retain_bound_file(
                source,
                target_snapshot,
                role="atomic displaced prior target cleanup",
            )
    finally:
        # A successful exchange or no-replace publication remains the only
        # authoritative linearization if a later fsync/retention step fails.
        # Retry resolves it from the complete transaction; no rollback mutates
        # either namespace after publication.
        _ = exchanged, fresh_published
        for descriptor in (
            source_descriptor,
            target_descriptor,
            parent_descriptor,
            transaction_descriptor,
        ):
            if descriptor >= 0:
                os.close(descriptor)


def _snapshot_matches_atomic_row(snapshot: BoundFileSnapshot, row: Mapping[str, Any]) -> bool:
    return (
        len(snapshot.payload) == row.get("bytes")
        and hashlib.sha256(snapshot.payload).hexdigest() == row.get("sha256")
        and list(snapshot.file_identity) == row.get("file_identity")
    )


def _matching_retained_atomic_rows(
    parent: Path,
    row: Mapping[str, Any],
) -> list[tuple[Path, BoundFileSnapshot]]:
    result: list[tuple[Path, BoundFileSnapshot]] = []
    for candidate in parent.iterdir():
        if not is_retained_name(candidate.name):
            continue
        if retained_original_name(candidate.name) != row.get("basename"):
            continue
        snapshot = validate_retained_file(candidate, role="atomic transaction retained scratch")
        if _snapshot_matches_atomic_row(snapshot, row):
            result.append((candidate, snapshot))
    return result


def _cleanup_atomic_scratch(
    path: Path,
    payload: bytes,
    *,
    committed_target: bool = False,
    expected_prior_payloads: Sequence[bytes] = (),
    expected_consumer_authorization_sha256: str | None = None,
) -> None:
    if not committed_target:
        scratch, _complete = _classify_atomic_scratch(path, payload)
        for candidate in scratch:
            snapshot = read_bound_file(candidate, role="atomic scratch cleanup")
            if snapshot.payload != payload and not payload.startswith(snapshot.payload):
                raise FeatureCacheError(f"atomic scratch cleanup payload drift; preserving bytes: {candidate}")
            retain_bound_file(candidate, snapshot, role="atomic scratch cleanup")
        _fsync_directory(path.parent)
        return

    prepared = atomic_prepared_path(path)
    scratch = ([prepared] if _path_exists_no_follow(prepared) else []) + _atomic_generation_paths(path)
    transactions = _atomic_transaction_paths(path)
    if not scratch and not transactions:
        retained_transactions: list[dict[str, Any]] = []
        for candidate in path.parent.iterdir():
            if not is_retained_name(candidate.name):
                continue
            original = retained_original_name(candidate.name)
            if _atomic_original_role_for_target(original, target_name=path.name) != "transaction":
                continue
            snapshot = validate_retained_file(candidate, role="retired atomic transaction finalization")
            parsed = _parse_atomic_transaction(Path(original), snapshot.payload)
            if parsed is None:
                raise FeatureCacheError("retired atomic transaction is partial-only; preserving bytes")
            retained_transactions.append(parsed)
        if not retained_transactions:
            return
        validate_atomic_transaction_custody(
            path,
            desired_payload=payload,
            expected_prior_payloads=expected_prior_payloads,
            expected_consumer_authorization_sha256=expected_consumer_authorization_sha256,
        )
        target_snapshot = read_bound_file(path, role="retired-unfinalized atomic target")
        terminal = [
            record
            for record in retained_transactions
            if record["desired_bytes"] == len(payload)
            and record["desired_sha256"] == hashlib.sha256(payload).hexdigest()
            and record["designated_source_identity"] == list(target_snapshot.file_identity)
        ]
        if len(terminal) != 1:
            raise FeatureCacheError("retired atomic history has no unique terminal transaction")
        _publish_atomic_completion(path, transaction=terminal[0], target=target_snapshot)
        validate_atomic_transaction_custody(path, desired_payload=payload)
        return
    target = read_bound_file(path, role="committed atomic target")
    if target.payload != payload:
        raise FeatureCacheError("atomic cleanup target is not the admitted committed payload; preserving bytes")
    transaction_rows, complete = _classify_atomic_transactions(path)
    desired_sha256 = hashlib.sha256(payload).hexdigest()
    matching = [
        row
        for row in complete
        if row.get("target_basename") == path.name
        and row.get("desired_bytes") == len(payload)
        and row.get("desired_sha256") == desired_sha256
    ]
    if not matching or any(row not in matching for row in complete):
        raise FeatureCacheError("atomic cleanup lacks one consistent committed transaction; preserving bytes")
    canonical_record = matching[0]
    if any(row != canonical_record for row in matching[1:]):
        raise FeatureCacheError("atomic cleanup has contradictory transaction generations; preserving bytes")
    admitted = {row["basename"]: row for row in canonical_record["admitted_scratch"]}
    active = {candidate.name: read_bound_file(candidate, role="post-exchange atomic scratch") for candidate in scratch}
    if set(active) - set(admitted):
        raise FeatureCacheError("atomic cleanup found scratch absent from pre-exchange custody; preserving bytes")
    parent_metadata = _require_directory(path.parent, role="atomic cleanup parent")
    if canonical_record.get("parent_identity") != [parent_metadata.st_dev, parent_metadata.st_ino]:
        raise FeatureCacheError("atomic cleanup parent custody differs from its transaction; preserving bytes")
    desired_source_identity = canonical_record["designated_source_identity"]
    if target.file_identity != tuple(desired_source_identity):
        raise FeatureCacheError(
            "atomic cleanup target is same-byte but not the transaction-designated source inode; preserving bytes"
        )
    mode = canonical_record["mode"]
    designated_prior: BoundFileSnapshot | None = None
    for basename, row in admitted.items():
        if row["role"] == "DISPLACED_PRIOR_SOURCE" and (
            row["bytes"] != canonical_record["prior_bytes"]
            or row["sha256"] != canonical_record["prior_sha256"]
            or row["file_identity"] != canonical_record["prior_file_identity"]
        ):
            raise FeatureCacheError(f"atomic cleanup transaction does not bind the displaced prior target: {basename}")
        snapshot = active.get(basename)
        retained_matches = _matching_retained_atomic_rows(path.parent, row)
        if row["role"] == "DESIGNATED_DESIRED_SOURCE":
            if mode != ATOMIC_MODE_FRESH or snapshot is not None or retained_matches:
                raise FeatureCacheError(
                    f"fresh atomic designated source has contradictory post-publication custody: {basename}"
                )
            continue
        realizations = ([] if snapshot is None else [snapshot]) + [item[1] for item in retained_matches]
        if len(realizations) != 1:
            raise FeatureCacheError(
                f"atomic cleanup requires one exact active/retained realization for {basename}; preserving bytes"
            )
        realized = realizations[0]
        if not _snapshot_matches_atomic_row(realized, row):
            raise FeatureCacheError(
                f"atomic cleanup scratch differs from pre-exchange custody; preserving bytes: {basename}"
            )
        if row["role"] == "DISPLACED_PRIOR_SOURCE":
            designated_prior = realized

    if mode == ATOMIC_MODE_EXISTING:
        if designated_prior is None:
            raise FeatureCacheError("atomic cleanup lacks exact displaced-prior custody; preserving bytes")
        if not any(designated_prior.payload == expected for expected in expected_prior_payloads):
            raise FeatureCacheError(
                "atomic cleanup displaced prior is not authorized by the current consumer; preserving bytes"
            )
    elif mode != ATOMIC_MODE_FRESH:
        raise FeatureCacheError("atomic cleanup transaction mode is malformed; preserving bytes")

    # Role grammar and self-hashes are not provenance.  Reconstruct the whole
    # active+retained transaction chain before the first retention move so an
    # extra, duplicate, forged, or orphan realization leaves the tree exact.
    validate_atomic_transaction_custody(
        path,
        desired_payload=payload,
        expected_prior_payloads=expected_prior_payloads,
        expected_consumer_authorization_sha256=expected_consumer_authorization_sha256,
    )

    # Completion is the durable operation-return proof.  It precedes every
    # retention move; a cut after this point remains visibly in-flight until
    # all admitted rows and the transaction record are retired.
    _publish_atomic_completion(path, transaction=canonical_record, target=target)

    # All active paths and every already-retained predecessor were proven
    # before the first move.  Each subsequent move is descriptor-bound and
    # no-replace; a cut leaves the transaction record for deterministic retry.
    for candidate in scratch:
        snapshot = active[candidate.name]
        retain_bound_file(candidate, snapshot, role="atomic transaction-proven scratch cleanup")
    ordered_transactions = sorted(
        transaction_rows,
        key=lambda item: (item[2] is not None, item[0].name),
    )
    for candidate, snapshot, _parsed in ordered_transactions:
        retain_bound_file(candidate, snapshot, role="atomic transaction record retirement")
    _fsync_directory(path.parent)


def _atomic_original_role_for_target(original: str, *, target_name: str) -> str | None:
    prepared_name = atomic_prepared_path(Path(target_name)).name
    if original == prepared_name:
        return "scratch"
    generation_prefix = f".{target_name}{ATOMIC_GENERATION_SUFFIX}-"
    if original.startswith(generation_prefix):
        if ATOMIC_GENERATION_RE.fullmatch(original) is None:
            raise FeatureCacheError("retained atomic generation original is malformed; preserving bytes")
        return "scratch"
    transaction_prefix = f".{target_name}{ATOMIC_TRANSACTION_SUFFIX}-"
    if original.startswith(transaction_prefix):
        if ATOMIC_TRANSACTION_RE.fullmatch(original) is None:
            raise FeatureCacheError("retained atomic transaction original is malformed; preserving bytes")
        return "transaction"
    completion_generation_prefix = f".{target_name}.atomic-completion-generation-"
    if original.startswith(completion_generation_prefix):
        if ATOMIC_COMPLETION_GENERATION_RE.fullmatch(original) is None:
            raise FeatureCacheError("retained atomic completion construction original is malformed; preserving bytes")
        return "completion_generation"
    completion_prefix = f".{target_name}.atomic-completion-"
    if original.startswith(completion_prefix):
        if ATOMIC_COMPLETION_RE.fullmatch(original) is None:
            raise FeatureCacheError("retained atomic completion original is malformed; preserving bytes")
        return "completion"
    return None


def validate_atomic_transaction_custody(
    path: Path,
    *,
    desired_payload: bytes,
    expected_prior_payloads: Sequence[bytes] = (),
    expected_consumer_authorization_sha256: str | None = None,
) -> set[str]:
    """Validate atomic active/retained provenance and return proven retained names.

    Retention naming is integrity-only.  This validator reconstructs the
    descriptor-bound transaction chain, requires a one-to-one realization for
    every retained writer row, proves the terminal target's exact source inode,
    and replays external prior authorization before allowing callers to exclude
    any retained entry from their semantic grammar.
    """

    target = Path(path)
    if type(desired_payload) is not bytes:
        raise FeatureCacheError("atomic custody desired payload must be immutable bytes")
    if any(type(expected) is not bytes for expected in expected_prior_payloads):
        raise FeatureCacheError("atomic custody authorized priors must be immutable bytes")
    if expected_consumer_authorization_sha256 is not None:
        expected_consumer_authorization_sha256 = _require_sha256(
            expected_consumer_authorization_sha256,
            name="atomic custody consumer authorization digest",
        )
    _require_directory_chain(target.parent)
    parent_metadata = _require_directory(target.parent, role="atomic custody parent")
    parent_identity = [parent_metadata.st_dev, parent_metadata.st_ino]

    transaction_evidence: list[tuple[Path, str, BoundFileSnapshot, dict[str, Any] | None, bool]] = []
    scratch_evidence: list[tuple[Path, str, BoundFileSnapshot, bool]] = []
    completion_evidence: list[tuple[Path, str, BoundFileSnapshot, dict[str, Any] | None, bool]] = []
    completion_generation_evidence: list[tuple[Path, str, BoundFileSnapshot, bool]] = []

    for candidate in _atomic_transaction_paths(target):
        snapshot = read_bound_file(candidate, role="active atomic transaction custody")
        transaction_evidence.append(
            (candidate, candidate.name, snapshot, _parse_atomic_transaction(candidate, snapshot.payload), False)
        )
    prepared = atomic_prepared_path(target)
    active_scratch = ([prepared] if _path_exists_no_follow(prepared) else []) + _atomic_generation_paths(target)
    for candidate in active_scratch:
        scratch_evidence.append(
            (candidate, candidate.name, read_bound_file(candidate, role="active atomic scratch custody"), False)
        )
    active_completions, active_completion_generations = _atomic_completion_paths(target)
    for candidate in active_completions:
        snapshot = read_bound_file(candidate, role="active atomic completion custody")
        completion_evidence.append(
            (candidate, candidate.name, snapshot, _parse_atomic_completion(candidate, snapshot.payload), False)
        )
    for candidate in active_completion_generations:
        completion_generation_evidence.append(
            (
                candidate,
                candidate.name,
                read_bound_file(candidate, role="active atomic completion construction"),
                False,
            )
        )

    for candidate in target.parent.iterdir():
        if not is_retained_name(candidate.name):
            continue
        original = retained_original_name(candidate.name)
        role = _atomic_original_role_for_target(original, target_name=target.name)
        if role is None:
            continue
        snapshot = validate_retained_file(candidate, role="retained atomic transaction custody")
        if role == "transaction":
            transaction_evidence.append(
                (candidate, original, snapshot, _parse_atomic_transaction(candidate, snapshot.payload), True)
            )
        elif role == "scratch":
            scratch_evidence.append((candidate, original, snapshot, True))
        elif role == "completion":
            completion_evidence.append(
                (candidate, original, snapshot, _parse_atomic_completion(Path(original), snapshot.payload), True)
            )
        elif role == "completion_generation":
            completion_generation_evidence.append((candidate, original, snapshot, True))

    if not transaction_evidence:
        if scratch_evidence or completion_evidence or completion_generation_evidence:
            raise FeatureCacheError("atomic custody is orphaned from a complete transaction; preserving bytes")
        return set()
    complete_evidence = [item for item in transaction_evidence if item[3] is not None]
    if not complete_evidence:
        raise FeatureCacheError("atomic custody has partial-only transaction evidence; preserving bytes")

    record_payloads: dict[
        bytes, tuple[dict[str, Any], tuple[Path, str, BoundFileSnapshot, dict[str, Any] | None, bool]]
    ] = {}
    for evidence in complete_evidence:
        parsed = evidence[3]
        assert parsed is not None
        canonical = canonical_json_bytes(parsed) + b"\n"
        if canonical in record_payloads:
            raise FeatureCacheError("atomic custody has duplicate complete transaction evidence; preserving bytes")
        record_payloads[canonical] = (parsed, evidence)
    for candidate, _original, snapshot, parsed, _retained in transaction_evidence:
        if parsed is not None:
            continue
        matches = [
            canonical
            for canonical in record_payloads
            if len(snapshot.payload) < len(canonical) and canonical.startswith(snapshot.payload)
        ]
        if len(matches) != 1:
            raise FeatureCacheError(
                f"atomic transaction partial is not a strict prefix of one complete record; preserving bytes: {candidate}"
            )

    records = [value[0] for value in record_payloads.values()]
    for record in records:
        if record["target_basename"] != target.name or record["parent_identity"] != parent_identity:
            raise FeatureCacheError("atomic transaction target/parent custody is inconsistent; preserving bytes")

    record_by_digest: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        digest = hashlib.sha256(canonical_json_bytes(record) + b"\n").hexdigest()
        if digest in record_by_digest:
            raise FeatureCacheError("atomic transaction digest is duplicated; preserving bytes")
        record_by_digest[digest] = (index, record)
    completions_by_index: dict[int, dict[str, Any]] = {}
    completion_payloads: dict[str, bytes] = {}
    terminal_snapshot_for_completion = read_bound_file(target, role="atomic completion terminal target")
    for digest, (_index, record) in record_by_digest.items():
        if (
            len(terminal_snapshot_for_completion.payload) == record["desired_bytes"]
            and hashlib.sha256(terminal_snapshot_for_completion.payload).hexdigest() == record["desired_sha256"]
            and list(terminal_snapshot_for_completion.file_identity) == record["designated_source_identity"]
        ):
            desired_payload_for_completion = terminal_snapshot_for_completion.payload
        else:
            desired_payload_for_completion = next(
                (
                    evidence_snapshot.payload
                    for _p, _o, evidence_snapshot, _ret in scratch_evidence
                    if len(evidence_snapshot.payload) == record["desired_bytes"]
                    and hashlib.sha256(evidence_snapshot.payload).hexdigest() == record["desired_sha256"]
                    and list(evidence_snapshot.file_identity) == record["designated_source_identity"]
                ),
                None,
            )
        if desired_payload_for_completion is None:
            raise FeatureCacheError("atomic transaction outcome lacks bytes for completion verification")
        expected_completion = _atomic_completion_record(
            target,
            transaction=record,
            target=BoundFileSnapshot(
                payload=desired_payload_for_completion,
                file_identity=tuple(record["designated_source_identity"]),
            ),
        )
        completion_payloads[digest] = canonical_json_bytes(expected_completion) + b"\n"
    for candidate, _original, snapshot, parsed, _retained in completion_evidence:
        if parsed is None:
            raise FeatureCacheError(f"partial payload occupies atomic completion authority: {candidate}")
        digest = parsed["transaction_sha256"]
        bound = record_by_digest.get(digest)
        if bound is None:
            raise FeatureCacheError("atomic completion proof is foreign to the transaction chain")
        index, record = bound
        if snapshot.payload != completion_payloads[digest]:
            raise FeatureCacheError("atomic completion proof fields differ from the exact transaction outcome")
        if index in completions_by_index:
            raise FeatureCacheError("atomic completion proof is duplicated; preserving bytes")
        completions_by_index[index] = parsed
    for _candidate, original, snapshot, retained in completion_generation_evidence:
        match = ATOMIC_COMPLETION_GENERATION_RE.fullmatch(original)
        if match is None or match.group("transaction") not in record_by_digest:
            raise FeatureCacheError("atomic completion construction is foreign; preserving bytes")
        digest = match.group("transaction")
        authority = completion_payloads.get(digest)
        if authority is None or (
            snapshot.payload != authority
            and not (len(snapshot.payload) < len(authority) and authority.startswith(snapshot.payload))
        ):
            raise FeatureCacheError("atomic completion construction is partial-only or not an exact prefix")
        if retained and not any(
            evidence[3] is not None and evidence[3]["transaction_sha256"] == digest for evidence in completion_evidence
        ):
            raise FeatureCacheError("retained atomic completion construction is partial-only")

    expected_rows: list[tuple[int, Mapping[str, Any]]] = []
    for record_index, record in enumerate(records):
        for row in record["admitted_scratch"]:
            if row["role"] != "DESIGNATED_DESIRED_SOURCE":
                expected_rows.append((record_index, row))

    row_realizations: dict[tuple[int, str], BoundFileSnapshot] = {}
    row_realization_retained: dict[tuple[int, str], bool] = {}
    for candidate, original, snapshot, retained in scratch_evidence:
        matches = [
            (record_index, row)
            for record_index, row in expected_rows
            if row["basename"] == original and _snapshot_matches_atomic_row(snapshot, row)
        ]
        if len(matches) != 1:
            raise FeatureCacheError(
                "atomic cleanup lacks one consistent committed transaction; "
                "fresh atomic designated source has contradictory post-publication custody; "
                "atomic terminal target is not the transaction-designated source inode; "
                "atomic cleanup scratch differs from pre-exchange custody; "
                "atomic transaction row lacks one exact active/retained realization; "
                f"atomic scratch is orphaned, duplicated, or role-ambiguous: {candidate}; preserving bytes"
            )
        record_index, row = matches[0]
        key = (record_index, row["basename"])
        if key in row_realizations:
            raise FeatureCacheError(
                "atomic transaction row has duplicate active/retained realizations; preserving bytes"
            )
        row_realizations[key] = snapshot
        row_realization_retained[key] = retained
    for record_index, row in expected_rows:
        if (record_index, row["basename"]) not in row_realizations:
            raise FeatureCacheError(
                f"atomic transaction row lacks one exact active/retained realization: {row['basename']}; preserving bytes"
            )

    if not _path_exists_no_follow(target):
        raise FeatureCacheError("atomic transaction custody lacks its committed target; preserving bytes")
    target_snapshot = read_bound_file(target, role="atomic transaction terminal target")
    if target_snapshot.payload != desired_payload:
        raise FeatureCacheError("atomic transaction terminal target differs from desired payload; preserving bytes")
    terminal_candidates = [
        index
        for index, record in enumerate(records)
        if record["desired_bytes"] == len(desired_payload)
        and record["desired_sha256"] == hashlib.sha256(desired_payload).hexdigest()
        and record["designated_source_identity"] == list(target_snapshot.file_identity)
    ]
    if len(terminal_candidates) != 1:
        raise FeatureCacheError(
            "atomic terminal target is not the unique transaction-designated source inode; preserving bytes"
        )
    terminal_index = terminal_candidates[0]

    predecessors: dict[int, int | None] = {}
    successors: dict[int, int] = {}
    for index, record in enumerate(records):
        if record["mode"] == ATOMIC_MODE_FRESH:
            predecessors[index] = None
            continue
        prior_matches = [
            candidate_index
            for candidate_index, candidate in enumerate(records)
            if candidate_index != index
            and candidate["desired_bytes"] == record["prior_bytes"]
            and candidate["desired_sha256"] == record["prior_sha256"]
            and candidate["designated_source_identity"] == record["prior_file_identity"]
        ]
        if len(prior_matches) > 1:
            raise FeatureCacheError("atomic transaction chain has ambiguous predecessor custody; preserving bytes")
        if prior_matches:
            predecessor = prior_matches[0]
            if predecessor in successors:
                raise FeatureCacheError("atomic transaction chain branches; preserving bytes")
            predecessors[index] = predecessor
            successors[predecessor] = index
        else:
            predecessors[index] = None

    visited: set[int] = set()
    cursor = terminal_index
    while cursor not in visited:
        visited.add(cursor)
        predecessor = predecessors[cursor]
        if predecessor is None:
            break
        cursor = predecessor
    if len(visited) != len(records):
        raise FeatureCacheError("atomic transaction evidence is disconnected or cyclic; preserving bytes")
    if terminal_index in successors:
        raise FeatureCacheError("atomic transaction terminal target has a later claimed successor; preserving bytes")

    for index, record in enumerate(records):
        if index == terminal_index:
            continue
        record_evidence = record_payloads[canonical_json_bytes(record) + b"\n"][1]
        realized_rows = [
            (index, row["basename"]) for row in record["admitted_scratch"] if row["role"] != "DESIGNATED_DESIRED_SOURCE"
        ]
        if not record_evidence[4] or not all(row_realization_retained[key] for key in realized_rows):
            raise FeatureCacheError("atomic nonterminal history is not fully retired outcome custody; preserving bytes")
        if index not in completions_by_index:
            raise FeatureCacheError("atomic nonterminal history lacks its exact completion proof; preserving bytes")

    for index, record in enumerate(records):
        if index == terminal_index:
            record_desired_payload = desired_payload
        else:
            successor_index = successors.get(index)
            if successor_index is None:
                raise FeatureCacheError("atomic transaction nonterminal outcome lacks its successor; preserving bytes")
            successor = records[successor_index]
            successor_source = next(
                row for row in successor["admitted_scratch"] if row["role"] == "DISPLACED_PRIOR_SOURCE"
            )
            record_desired_payload = row_realizations[(successor_index, successor_source["basename"])].payload
        if (
            len(record_desired_payload) != record["desired_bytes"]
            or hashlib.sha256(record_desired_payload).hexdigest() != record["desired_sha256"]
        ):
            raise FeatureCacheError("atomic transaction desired outcome bytes are inconsistent; preserving bytes")
        for row in record["admitted_scratch"]:
            if row["role"] != "ADMITTED_WRITER_SCRATCH":
                continue
            scratch_snapshot = row_realizations[(index, row["basename"])]
            if not record_desired_payload.startswith(scratch_snapshot.payload):
                raise FeatureCacheError("atomic admitted writer scratch is not a desired prefix; preserving bytes")

    # A currently active terminal exchange is a retry, so its exact displaced
    # prior must be replayed against this caller even when older retained
    # records also prove the historical predecessor.
    terminal_record = records[terminal_index]
    terminal_record_evidence = record_payloads[canonical_json_bytes(terminal_record) + b"\n"][1]
    terminal_realized_rows = [
        (terminal_index, row["basename"])
        for row in terminal_record["admitted_scratch"]
        if row["role"] != "DESIGNATED_DESIRED_SOURCE"
    ]
    terminal_fully_retired = terminal_record_evidence[4] and all(
        row_realization_retained[key] for key in terminal_realized_rows
    )
    if terminal_record_evidence[4] and not terminal_fully_retired:
        raise FeatureCacheError("retired atomic terminal record has active row custody; preserving bytes")
    terminal_needs_prior_replay = terminal_record["mode"] == ATOMIC_MODE_EXISTING and (
        not terminal_fully_retired or terminal_index not in completions_by_index
    )
    terminal_unresolved = not terminal_fully_retired or terminal_index not in completions_by_index
    if (
        terminal_unresolved
        and terminal_record.get("consumer_authorization_sha256") != expected_consumer_authorization_sha256
    ):
        raise FeatureCacheError(
            "unresolved atomic transaction consumer authorization is missing or foreign; preserving bytes"
        )
    if terminal_needs_prior_replay:
        source_row = next(row for row in terminal_record["admitted_scratch"] if row["role"] == "DISPLACED_PRIOR_SOURCE")
        prior_snapshot = row_realizations[(terminal_index, source_row["basename"])]
        if not any(prior_snapshot.payload == expected for expected in expected_prior_payloads):
            raise FeatureCacheError(
                "unresolved atomic transaction prior is not authorized by the current consumer; preserving bytes"
            )
    return (
        {candidate.name for candidate, _original, _snapshot, _parsed, retained in transaction_evidence if retained}
        | {candidate.name for candidate, _original, _snapshot, retained in scratch_evidence if retained}
        | {candidate.name for candidate, _original, _snapshot, _parsed, retained in completion_evidence if retained}
        | {candidate.name for candidate, _original, _snapshot, retained in completion_generation_evidence if retained}
    )


def _validate_absent_target_transaction_custody(
    path: Path,
    *,
    desired_payload: bytes,
    consumer_authorization_sha256: str | None,
) -> None:
    """Validate an interrupted fresh transaction before creating any bytes."""

    active_transactions = _atomic_transaction_paths(path)
    if not active_transactions:
        return
    evidence, complete = _classify_atomic_transactions(path)
    if not complete:
        raise FeatureCacheError(
            "target-absent atomic transaction lacks one complete fresh authority; refusing before mutation"
        )
    if len(complete) > 1:
        if any(candidate != complete[0] for candidate in complete[1:]):
            raise FeatureCacheError(
                "target-absent atomic transaction has contradictory complete authority; refusing before mutation"
            )
        raise FeatureCacheError(
            "target-absent atomic transaction has duplicate complete authority; refusing before mutation"
        )
    record = complete[0]
    if any(parsed is not None and parsed != record for _candidate, _snapshot, parsed in evidence):
        raise FeatureCacheError("target-absent atomic transaction evidence is contradictory; refusing before mutation")
    if (
        record.get("mode") != ATOMIC_MODE_FRESH
        or record.get("desired_bytes") != len(desired_payload)
        or record.get("desired_sha256") != hashlib.sha256(desired_payload).hexdigest()
        or record.get("consumer_authorization_sha256") != consumer_authorization_sha256
    ):
        raise FeatureCacheError(
            "target-absent atomic transaction desired/auth custody is foreign; refusing before mutation"
        )

    scratch, _complete_source = _classify_atomic_scratch(path, desired_payload)
    active_by_name: dict[str, BoundFileSnapshot] = {}
    for candidate in scratch:
        active_by_name[candidate.name] = read_bound_file(
            candidate,
            role="target-absent atomic transaction scratch",
        )
    rows = record["admitted_scratch"]
    row_by_name = {row["basename"]: row for row in rows}
    designated = record["designated_source_basename"]
    if designated not in active_by_name:
        raise FeatureCacheError("target-absent atomic transaction lost its designated source; refusing before mutation")
    if set(active_by_name) != set(row_by_name):
        raise FeatureCacheError(
            "target-absent atomic transaction is missing or has unadmitted scratch; refusing before mutation"
        )
    for basename, row in row_by_name.items():
        if not _snapshot_matches_atomic_row(active_by_name[basename], row):
            raise FeatureCacheError(
                "target-absent atomic transaction scratch identity/content drift; refusing before mutation"
            )
    designated_rows = [row for row in rows if row["role"] == "DESIGNATED_DESIRED_SOURCE"]
    designated_snapshot = active_by_name.get(designated)
    if (
        len(designated_rows) != 1
        or designated_rows[0]["basename"] != designated
        or designated_snapshot is None
        or designated_snapshot.payload != desired_payload
        or list(designated_snapshot.file_identity) != record["designated_source_identity"]
    ):
        raise FeatureCacheError(
            "target-absent atomic transaction lost its exact designated source; refusing before mutation"
        )


def _validate_atomic_prepublication_namespace(
    path: Path,
    *,
    desired_payload: bytes,
    expected_prior_payloads: Sequence[bytes],
    consumer_authorization_sha256: str | None = None,
) -> AtomicPrepublicationNamespace:
    """Read-only refusal gate for the entire target-scoped atomic namespace."""

    namespace = _observe_atomic_prepublication_namespace(path)
    if not _path_exists_no_follow(path):
        if namespace.retained_roles or namespace.active_completions or namespace.active_completion_constructions:
            raise FeatureCacheError(
                "target-absent atomic namespace has retained/orphan completion custody; refusing before mutation"
            )
        _validate_absent_target_transaction_custody(
            path,
            desired_payload=desired_payload,
            consumer_authorization_sha256=consumer_authorization_sha256,
        )
    elif (
        namespace.retained_roles
        or namespace.active_completions
        or namespace.active_completion_constructions
        or namespace.active_transactions
    ):
        current = read_bound_file(path, role="prepublication atomic target")
        if current.payload != desired_payload and not any(
            current.payload == prior for prior in expected_prior_payloads
        ):
            raise FeatureCacheError("atomic prior target is not a consumer-authorized exact state; preserving bytes")
        validate_atomic_transaction_custody(
            path,
            desired_payload=current.payload,
            expected_prior_payloads=expected_prior_payloads,
            expected_consumer_authorization_sha256=consumer_authorization_sha256,
        )
    if _observe_atomic_prepublication_namespace(path) != namespace:
        raise FeatureCacheError(
            "target-scoped atomic namespace changed during prepublication observation; refusing before mutation"
        )
    return namespace


def _revalidate_atomic_prepublication_namespace(
    path: Path,
    *,
    expected: AtomicPrepublicationNamespace,
) -> None:
    """Close the gate-to-construction gap before the writer creates any byte."""

    if _observe_atomic_prepublication_namespace(path) != expected:
        raise FeatureCacheError(
            "target-scoped atomic namespace changed after prepublication; refusing before writer mutation"
        )


def _namespace_with_explicit_delta(
    namespace: AtomicNamespaceFingerprint,
    row: AtomicNamespaceRow,
    *,
    role: str,
) -> AtomicNamespaceFingerprint:
    if any(existing[0] == row[0] for existing in namespace):
        raise FeatureCacheError(f"{role} collided with a prepublication pathname; preserving bytes")
    return tuple(sorted((*namespace, row)))


def _atomic_bytes(
    path: Path,
    payload: bytes,
    *,
    expected_prior_payloads: Sequence[bytes] = (),
    consumer_authorization_sha256: str | None = None,
) -> None:
    if type(payload) is not bytes:
        raise FeatureCacheError("atomic metadata payload must be immutable bytes")
    if any(type(expected) is not bytes for expected in expected_prior_payloads):
        raise FeatureCacheError("atomic consumer-authorized prior payloads must be immutable bytes")
    if consumer_authorization_sha256 is not None:
        consumer_authorization_sha256 = _require_sha256(
            consumer_authorization_sha256,
            name="atomic consumer authorization digest",
        )
    _require_directory_chain(path.parent)
    _preflight_atomic_components(path, payload)
    authorized_prior_payloads = tuple(expected_prior_payloads)
    prepublication_namespace = _validate_atomic_prepublication_namespace(
        path,
        desired_payload=payload,
        expected_prior_payloads=authorized_prior_payloads,
        consumer_authorization_sha256=consumer_authorization_sha256,
    )
    _revalidate_atomic_prepublication_namespace(
        path,
        expected=prepublication_namespace,
    )
    stable_namespace = _atomic_prepublication_stable_fingerprint(prepublication_namespace)
    temporary = atomic_prepared_path(path)
    if _path_exists_no_follow(path):
        target_snapshot = read_bound_file(path, role="atomic metadata target")
        if target_snapshot.payload == payload:
            _fsync_file(path)
            _fsync_directory(path.parent)
            _cleanup_atomic_scratch(
                path,
                payload,
                committed_target=True,
                expected_prior_payloads=authorized_prior_payloads,
                expected_consumer_authorization_sha256=consumer_authorization_sha256,
            )
            validate_atomic_transaction_custody(
                path,
                desired_payload=payload,
                expected_prior_payloads=authorized_prior_payloads,
                expected_consumer_authorization_sha256=consumer_authorization_sha256,
            )
            return
        if not any(target_snapshot.payload == expected for expected in authorized_prior_payloads):
            raise FeatureCacheError(
                f"atomic prior target is not a consumer-authorized exact state; preserving bytes: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Grammar-invalid transaction names are unknown custody, not ignorable
    # siblings.  Detect them before creating the first writer scratch file.
    _atomic_transaction_paths(path)
    scratch, complete = _classify_atomic_scratch(path, payload)
    if _atomic_scratch_namespace_fingerprint(path, payload) != prepublication_namespace.active_scratch:
        raise FeatureCacheError(
            "target-scoped atomic scratch changed after prepublication; refusing before writer mutation"
        )
    authorized_scratch_namespace = prepublication_namespace.active_scratch
    if not scratch:
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                _nofollow_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
                0o666,
            )
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            _assert_bound_file(
                temporary,
                descriptor,
                opened,
                role="stable atomic prepared creation",
            )
        except OSError as exc:
            raise FeatureCacheError(f"stable atomic prepared write failed; preserving bytes: {temporary}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        _fsync_directory(path.parent)
        scratch = [temporary]
        complete = temporary
        created_row = _bound_atomic_paths_fingerprint(
            (temporary,),
            role="explicit stable atomic prepared construction",
        )[0]
        authorized_scratch_namespace = _namespace_with_explicit_delta(
            authorized_scratch_namespace,
            created_row,
            role="atomic prepared construction",
        )
        if _atomic_scratch_namespace_fingerprint(path, payload) != authorized_scratch_namespace:
            raise FeatureCacheError(
                "atomic prepared construction produced an unauthorized scratch delta; preserving bytes"
            )
    elif complete is None:
        authorized_prefixes = {
            candidate: read_bound_file(candidate, role="authorized atomic metadata prefix").file_identity
            for candidate in scratch
        }
        hook = _ATOMIC_PREFIX_AUTHORIZATION_TEST_HOOK
        if hook is not None:
            hook(temporary)
        # Re-read every prefix immediately after the explicit authorization
        # boundary.  No observed prefix inode is ever modified.
        for candidate, authorized_identity in authorized_prefixes.items():
            current = read_bound_file(candidate, role="authorized atomic metadata prefix revalidation")
            if current.file_identity != authorized_identity:
                raise FeatureCacheError(
                    f"atomic metadata prefix pathname changed at authorization boundary; preserving bytes: {candidate}"
                )
        if _atomic_scratch_namespace_fingerprint(path, payload) != authorized_scratch_namespace:
            raise FeatureCacheError(
                "target-scoped atomic scratch changed before generation construction; preserving bytes"
            )
        scratch, complete = _classify_atomic_scratch(path, payload)
        if complete is None:
            complete = _create_complete_generation(path, payload, scratch)
            created_row = _bound_atomic_paths_fingerprint(
                (complete,),
                role="explicit atomic generation construction",
            )[0]
            authorized_scratch_namespace = _namespace_with_explicit_delta(
                authorized_scratch_namespace,
                created_row,
                role="atomic generation construction",
            )
            if _atomic_scratch_namespace_fingerprint(path, payload) != authorized_scratch_namespace:
                raise FeatureCacheError(
                    "atomic generation construction produced an unauthorized scratch delta; preserving bytes"
                )
    assert complete is not None
    _commit_bound_atomic_source(
        path,
        complete,
        payload,
        expected_prior_payloads=authorized_prior_payloads,
        expected_scratch_namespace=authorized_scratch_namespace,
        prepublication_transaction_namespace=prepublication_namespace.active_transactions,
        expected_stable_namespace=stable_namespace,
        consumer_authorization_sha256=consumer_authorization_sha256,
    )
    _cleanup_atomic_scratch(
        path,
        payload,
        committed_target=True,
        expected_prior_payloads=authorized_prior_payloads,
        expected_consumer_authorization_sha256=consumer_authorization_sha256,
    )
    validate_atomic_transaction_custody(
        path,
        desired_payload=payload,
        expected_prior_payloads=authorized_prior_payloads,
        expected_consumer_authorization_sha256=consumer_authorization_sha256,
    )


def atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_prior_payloads: Sequence[bytes] = (),
    consumer_authorization_sha256: str | None = None,
) -> None:
    _atomic_bytes(
        path,
        canonical_json_bytes(dict(payload)) + b"\n",
        expected_prior_payloads=expected_prior_payloads,
        consumer_authorization_sha256=consumer_authorization_sha256,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_object(path: Path) -> dict[str, Any]:
    raw = _read_stable_bytes(path, role="JSON custody file")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"invalid JSON at {path}") from exc
    if not isinstance(payload, dict):
        raise FeatureCacheError(f"expected JSON object at {path}")
    if raw != canonical_json_bytes(payload) + b"\n":
        raise FeatureCacheError(f"JSON custody file is not canonical with one newline: {path}")
    return payload


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_nlink,
    )


def _require_regular_file(path: Path, *, role: str) -> os.stat_result:
    """Require a single-link regular file without following a symlink."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FeatureCacheError(f"{role} is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FeatureCacheError(f"{role} must be a non-symlink regular file: {path}")
    if metadata.st_nlink != 1:
        raise FeatureCacheError(f"{role} must have exactly one hard link: {path}")
    return metadata


def _require_directory(path: Path, *, role: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FeatureCacheError(f"{role} is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FeatureCacheError(f"{role} must be a non-symlink directory: {path}")
    return metadata


def _require_directory_chain(path: Path) -> None:
    """Reject symlink/non-directory ancestors before any path resolution."""

    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise FeatureCacheError(f"cache directory custody is unavailable: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FeatureCacheError(f"cache path traverses a symlink or non-directory: {cursor}")


def _unresolved_absolute(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise FeatureCacheError(f"cache path custody is unavailable: {path}") from exc
    return True


def _is_governed_atomic_retained_original(original: str, *, target_names: frozenset[str]) -> bool:
    for target_name in target_names:
        if original == f".{target_name}{ATOMIC_PREPARED_SUFFIX}":
            return True
        generation_prefix = f".{target_name}{ATOMIC_GENERATION_SUFFIX}-"
        if (
            original.startswith(generation_prefix)
            and len(original) == len(generation_prefix) + 8
            and original[-8:].isdigit()
        ):
            return True
        transaction_prefix = f".{target_name}{ATOMIC_TRANSACTION_SUFFIX}-"
        if (
            original.startswith(transaction_prefix)
            and len(original) == len(transaction_prefix) + 8
            and original[-8:].isdigit()
        ):
            return True
        if ATOMIC_COMPLETION_RE.fullmatch(original) is not None and original.startswith(
            f".{target_name}.atomic-completion-"
        ):
            return True
        if ATOMIC_COMPLETION_GENERATION_RE.fullmatch(original) is not None and original.startswith(
            f".{target_name}.atomic-completion-generation-"
        ):
            return True
    return False


def _active_atomic_transaction_names(entries: Mapping[str, Path], *, target_names: Sequence[str]) -> set[str]:
    return {
        entry_name
        for entry_name in entries
        if any(
            entry_name.startswith(f".{target_name}{ATOMIC_TRANSACTION_SUFFIX}-")
            and ATOMIC_TRANSACTION_RE.fullmatch(entry_name) is not None
            for target_name in target_names
        )
    }


def active_atomic_completion_names(entries: Mapping[str, Path], *, target_names: Sequence[str]) -> set[str]:
    """Return active immutable completion and construction names for targets."""

    return {
        entry_name
        for entry_name in entries
        if any(
            (
                entry_name.startswith(f".{target_name}.atomic-completion-")
                and ATOMIC_COMPLETION_RE.fullmatch(entry_name) is not None
            )
            or (
                entry_name.startswith(f".{target_name}.atomic-completion-generation-")
                and ATOMIC_COMPLETION_GENERATION_RE.fullmatch(entry_name) is not None
            )
            for target_name in target_names
        )
    }


def _validated_governed_retained_entries(
    entries: Mapping[str, Path],
    *,
    target_names: frozenset[str],
    role: str,
) -> set[str]:
    """Return only retained names proven by their complete atomic outcomes."""

    governed_retained: set[str] = set()
    relevant_targets: set[str] = set()
    for name, path in entries.items():
        if not is_retained_name(name):
            continue
        original = retained_original_name(name)
        if not _is_governed_atomic_retained_original(original, target_names=target_names):
            raise FeatureCacheError(
                f"{role} retained original basename is not governed: {original!r}; preserving bytes"
            )
        validate_retained_file(path, role=f"{role} retained custody {name}")
        governed_retained.add(name)
        for target_name in target_names:
            if _atomic_original_role_for_target(original, target_name=target_name) is not None:
                relevant_targets.add(target_name)
                break

    proven_retained: set[str] = set()
    for target_name in sorted(relevant_targets):
        target = next(iter(entries.values())).parent / target_name
        if not _path_exists_no_follow(target):
            raise FeatureCacheError(f"{role} retained atomic custody lacks target {target_name!r}; preserving bytes")
        desired = read_bound_file(target, role=f"{role} atomic target {target_name}").payload
        proven_retained.update(
            validate_atomic_transaction_custody(
                target,
                desired_payload=desired,
            )
        )
    if proven_retained != governed_retained:
        raise FeatureCacheError(f"{role} retained atomic custody is orphaned or unproven; preserving bytes")
    return proven_retained


def _require_certified_staging_layout(staging_root: Path) -> None:
    """Prove the only recursively disposable pre-final cache layout."""

    _require_directory(staging_root, role="cache staging root")
    try:
        entries = {entry.name: entry for entry in staging_root.iterdir()}
    except OSError as exc:
        raise FeatureCacheError("cache staging entries cannot be enumerated") from exc
    retained_names = _validated_governed_retained_entries(
        entries,
        target_names=ATOMIC_STAGING_TARGET_NAMES,
        role="cache staging layout",
    )
    completion_names = active_atomic_completion_names(
        entries,
        target_names=tuple(ATOMIC_STAGING_TARGET_NAMES),
    )
    active_names = set(entries) - retained_names - completion_names
    if active_names != CERTIFIED_STAGING_ENTRY_SET:
        unknown = sorted(active_names - CERTIFIED_STAGING_ENTRY_SET)
        missing = sorted(CERTIFIED_STAGING_ENTRY_SET - active_names)
        raise FeatureCacheError(
            f"cache staging layout is not exactly certified; unknown={unknown} missing={missing}; preserving bytes"
        )
    for name in sorted(CERTIFIED_STAGING_ENTRY_SET):
        _require_regular_file(staging_root / name, role=f"cache staging entry {name}")


def _creation_staging_path(cache_root: Path) -> Path:
    return cache_root.with_name(f".{cache_root.name}{STAGING_SUFFIX}")


def _decode_original_basename(encoded: str, *, name_max: int) -> str:
    if not encoded.startswith("z"):
        raise FeatureCacheError("retained directory original basename encoding is malformed")
    compressed = encoded[1:]
    padding = "=" * (-len(compressed) % 4)
    try:
        compressed_bytes = base64.b64decode(compressed + padding, altchars=b"-_", validate=True)
        decompressor = zlib.decompressobj()
        original_bytes = decompressor.decompress(compressed_bytes, name_max + 1)
    except (ValueError, binascii.Error, zlib.error) as exc:
        raise FeatureCacheError("retained directory original basename encoding is malformed") from exc
    if (
        not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or len(original_bytes) > name_max
    ):
        raise FeatureCacheError("retained directory original basename encoding is malformed")
    original = os.fsdecode(original_bytes)
    if (
        not original
        or original in {".", ".."}
        or Path(original).name != original
        or _encoded_original_basename(original) != encoded
    ):
        raise FeatureCacheError("retained directory original basename encoding is noncanonical")
    return original


def _assert_directory_path_identity(path: Path, descriptor: int, opened: os.stat_result, *, role: str) -> None:
    try:
        current_descriptor = os.fstat(descriptor)
        current_path = path.lstat()
    except OSError as exc:
        raise FeatureCacheError(f"{role} identity cannot be revalidated; preserving tree: {path}") from exc
    if (
        not stat.S_ISDIR(current_descriptor.st_mode)
        or not stat.S_ISDIR(current_path.st_mode)
        or (opened.st_dev, opened.st_ino) != (current_descriptor.st_dev, current_descriptor.st_ino)
        or (opened.st_dev, opened.st_ino) != (current_path.st_dev, current_path.st_ino)
        or _stable_file_identity(opened) != _stable_file_identity(current_descriptor)
        or _stable_file_identity(opened) != _stable_file_identity(current_path)
    ):
        raise FeatureCacheError(f"{role} changed during recursive measurement; preserving tree: {path}")


def _measure_directory_tree(path: Path, *, role: str) -> BoundDirectorySnapshot:
    """Measure a no-link tree entirely through descriptor-relative traversal."""

    root = _unresolved_absolute(path)
    descriptor, opened = _open_bound_parent_directory(root, role=role)
    records: list[dict[str, Any]] = []
    total_bytes = 0

    def walk(directory_descriptor: int, relative: tuple[str, ...]) -> None:
        nonlocal total_bytes
        before = os.fstat(directory_descriptor)
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise FeatureCacheError(f"{role} cannot enumerate retained tree; preserving bytes") from exc
        for name in names:
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise FeatureCacheError(f"{role} contains a malformed entry name; preserving bytes")
            metadata = _entry_metadata_at(directory_descriptor, name, role=f"{role} entry")
            child_relative = (*relative, name)
            relative_name = "/".join(child_relative)
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise FeatureCacheError(f"{role} contains a linked file; preserving bytes: {relative_name}")
                child = -1
                try:
                    child = os.open(name, _nofollow_flags(os.O_RDONLY), dir_fd=directory_descriptor)
                    child_opened = os.fstat(child)
                    if (
                        not stat.S_ISREG(child_opened.st_mode)
                        or child_opened.st_nlink != 1
                        or _stable_file_identity(child_opened) != _stable_file_identity(metadata)
                    ):
                        raise FeatureCacheError(
                            f"{role} file changed before descriptor binding; preserving bytes: {relative_name}"
                        )
                    payload = _read_descriptor(child)
                    if _stable_file_identity(os.fstat(child)) != _stable_file_identity(child_opened):
                        raise FeatureCacheError(
                            f"{role} file changed during measurement; preserving bytes: {relative_name}"
                        )
                    after = _entry_metadata_at(directory_descriptor, name, role=f"{role} measured entry")
                    if _stable_file_identity(after) != _stable_file_identity(child_opened):
                        raise FeatureCacheError(
                            f"{role} file pathname changed during measurement; preserving bytes: {relative_name}"
                        )
                finally:
                    if child >= 0:
                        os.close(child)
                total_bytes += len(payload)
                records.append(
                    {
                        "kind": "file",
                        "path": relative_name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            elif stat.S_ISDIR(metadata.st_mode):
                child = -1
                try:
                    child = os.open(
                        name,
                        _nofollow_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)),
                        dir_fd=directory_descriptor,
                    )
                    child_opened = os.fstat(child)
                    if not stat.S_ISDIR(child_opened.st_mode) or _stable_file_identity(
                        child_opened
                    ) != _stable_file_identity(metadata):
                        raise FeatureCacheError(
                            f"{role} directory changed before binding; preserving bytes: {relative_name}"
                        )
                    records.append({"kind": "directory", "path": relative_name})
                    walk(child, child_relative)
                    after = _entry_metadata_at(directory_descriptor, name, role=f"{role} measured directory")
                    if _stable_file_identity(after) != _stable_file_identity(child_opened):
                        raise FeatureCacheError(
                            f"{role} directory changed during measurement; preserving bytes: {relative_name}"
                        )
                finally:
                    if child >= 0:
                        os.close(child)
            else:
                raise FeatureCacheError(
                    f"{role} contains a symlink or special entry; preserving bytes: {relative_name}"
                )
        if _stable_file_identity(os.fstat(directory_descriptor)) != _stable_file_identity(before):
            raise FeatureCacheError(f"{role} directory changed during enumeration; preserving bytes")

    try:
        walk(descriptor, ())
        _assert_directory_path_identity(root, descriptor, opened, role=role)
    finally:
        os.close(descriptor)
    tree_payload = canonical_json_bytes(records)
    return BoundDirectorySnapshot(
        directory_identity=_stable_file_identity(opened),
        recursive_file_bytes=total_bytes,
        tree_sha256=hashlib.sha256(tree_payload).hexdigest(),
        entry_count=len(records),
    )


def _directory_retained_name(original: str, snapshot: BoundDirectorySnapshot, ordinal: int) -> str:
    return (
        f"{DIRECTORY_RETAINED_PREFIX}{_encoded_original_basename(original)}-"
        f"{snapshot.recursive_file_bytes:020d}-{snapshot.tree_sha256}-{ordinal:08d}"
    )


def _directory_retention_binding(
    source: Path,
    snapshot: BoundDirectorySnapshot,
    *,
    reason: str,
    rebuild_command: Sequence[str],
    storage_preflight: Mapping[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "original_path": str(source),
                "directory_identity": list(snapshot.directory_identity),
                "recursive_file_bytes": snapshot.recursive_file_bytes,
                "tree_sha256": snapshot.tree_sha256,
                "entry_count": snapshot.entry_count,
                "reason": reason,
                "rebuild_command": list(rebuild_command),
                "storage_preflight": dict(storage_preflight),
            }
        )
    ).hexdigest()


def _directory_retention_receipt(
    source: Path,
    destination: Path,
    snapshot: BoundDirectorySnapshot,
    *,
    reason: str,
    rebuild_command: Sequence[str],
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DIRECTORY_RETENTION_SCHEMA,
        "original_path": str(source),
        "original_basename": source.name,
        "directory_identity": list(snapshot.directory_identity),
        "recursive_file_bytes": snapshot.recursive_file_bytes,
        "tree_sha256": snapshot.tree_sha256,
        "entry_count": snapshot.entry_count,
        "rebuild_command": list(rebuild_command),
        "storage_preflight": dict(storage_preflight),
        "false_authority_flags": {
            "cache_authority": False,
            "score_authority": False,
            "rate_authority": False,
            "promotion_eligible": False,
        },
        "reason": reason,
        "retention_destination": str(destination),
        "disposition": "LOSSLESS_DESCRIPTOR_BOUND_NOREPLACE_RETENTION",
    }


def _validate_directory_retention_receipt(path: Path) -> tuple[dict[str, Any], BoundDirectorySnapshot | None]:
    receipt = _load_object(path)
    expected_keys = {
        "schema",
        "original_path",
        "original_basename",
        "directory_identity",
        "recursive_file_bytes",
        "tree_sha256",
        "entry_count",
        "rebuild_command",
        "storage_preflight",
        "false_authority_flags",
        "reason",
        "retention_destination",
        "disposition",
    }
    identity = receipt.get("directory_identity")
    source_raw = receipt.get("original_path")
    destination_raw = receipt.get("retention_destination")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != DIRECTORY_RETENTION_SCHEMA
        or not isinstance(source_raw, str)
        or not Path(source_raw).is_absolute()
        or receipt.get("original_basename") != Path(source_raw).name
        or not isinstance(identity, list)
        or len(identity) != 5
        or any(type(value) is not int for value in identity)
        or type(receipt.get("recursive_file_bytes")) is not int
        or receipt["recursive_file_bytes"] < 0
        or LOWER_SHA256_RE.fullmatch(str(receipt.get("tree_sha256"))) is None
        or type(receipt.get("entry_count")) is not int
        or receipt["entry_count"] < 0
        or not isinstance(receipt.get("rebuild_command"), list)
        or not receipt["rebuild_command"]
        or any(not isinstance(token, str) or not token for token in receipt["rebuild_command"])
        or not isinstance(receipt.get("storage_preflight"), dict)
        or receipt.get("false_authority_flags")
        != {
            "cache_authority": False,
            "score_authority": False,
            "rate_authority": False,
            "promotion_eligible": False,
        }
        or not isinstance(receipt.get("reason"), str)
        or not receipt["reason"]
        or not isinstance(destination_raw, str)
        or not Path(destination_raw).is_absolute()
        or receipt.get("disposition") != "LOSSLESS_DESCRIPTOR_BOUND_NOREPLACE_RETENTION"
    ):
        raise FeatureCacheError(f"retained directory receipt is malformed; preserving bytes: {path}")
    _validate_storage_preflight(receipt["storage_preflight"])
    source = Path(source_raw)
    destination = Path(destination_raw)
    if source.parent != path.parent or destination.parent != path.parent:
        raise FeatureCacheError(f"retained directory receipt crosses custody parent: {path}")
    match = DIRECTORY_RETAINED_RE.fullmatch(destination.name)
    if match is None:
        raise FeatureCacheError(f"retained directory destination name is malformed: {destination}")
    receipt_match = DIRECTORY_RETENTION_RECEIPT_RE.fullmatch(path.name)
    if receipt_match is None:
        raise FeatureCacheError(f"retained directory receipt name is malformed: {path}")
    name_max = _filesystem_name_max(path.parent)
    if _decode_original_basename(match.group("original"), name_max=name_max) != source.name:
        raise FeatureCacheError(f"retained directory original role is inconsistent: {destination}")
    if (
        int(match.group("bytes")) != receipt["recursive_file_bytes"]
        or match.group("tree_sha256") != receipt["tree_sha256"]
    ):
        raise FeatureCacheError(f"retained directory name/receipt digest mismatch: {destination}")
    declared_snapshot = BoundDirectorySnapshot(
        directory_identity=tuple(identity),
        recursive_file_bytes=receipt["recursive_file_bytes"],
        tree_sha256=receipt["tree_sha256"],
        entry_count=receipt["entry_count"],
    )
    expected_binding = _directory_retention_binding(
        source,
        declared_snapshot,
        reason=receipt["reason"],
        rebuild_command=receipt["rebuild_command"],
        storage_preflight=receipt["storage_preflight"],
    )
    if receipt_match.group("binding") != expected_binding or receipt_match.group("ordinal") != match.group("ordinal"):
        raise FeatureCacheError(f"retained directory receipt filename is not self-binding: {path}")
    measured: BoundDirectorySnapshot | None = None
    present = [candidate for candidate in (source, destination) if _path_exists_no_follow(candidate)]
    if len(present) != 1:
        raise FeatureCacheError(
            f"retained directory receipt requires exactly one source/destination tree; preserving bytes: {path}"
        )
    measured = _measure_directory_tree(present[0], role="retained directory custody")
    if (
        measured.directory_identity != tuple(identity)
        or measured.recursive_file_bytes != receipt["recursive_file_bytes"]
        or measured.tree_sha256 != receipt["tree_sha256"]
        or measured.entry_count != receipt["entry_count"]
    ):
        raise FeatureCacheError(f"retained directory tree drifted from receipt; preserving bytes: {present[0]}")
    return receipt, measured


def _validate_retained_directory_custody(parent: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Validate every recognized directory retention in one cache parent."""

    receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    destinations: set[str] = set()
    for entry in parent.iterdir():
        if entry.name.startswith(DIRECTORY_RETENTION_RECEIPT_PREFIX):
            if DIRECTORY_RETENTION_RECEIPT_RE.fullmatch(entry.name) is None:
                raise FeatureCacheError(f"malformed retained directory receipt name; preserving bytes: {entry}")
            receipt, _snapshot = _validate_directory_retention_receipt(entry)
            destination = Path(receipt["retention_destination"])
            if destination.name in destinations:
                raise FeatureCacheError("duplicate retained directory destination receipts; preserving bytes")
            if receipt["original_path"] in receipts:
                raise FeatureCacheError("duplicate retained directory source receipts; preserving bytes")
            destinations.add(destination.name)
            receipts[receipt["original_path"]] = (entry, receipt)
        elif entry.name.startswith(DIRECTORY_RETAINED_PREFIX):
            if DIRECTORY_RETAINED_RE.fullmatch(entry.name) is None:
                raise FeatureCacheError(f"malformed retained directory custody name; preserving bytes: {entry}")
    for entry in parent.iterdir():
        if entry.name.startswith(DIRECTORY_RETAINED_PREFIX) and entry.name not in destinations:
            raise FeatureCacheError(f"orphan retained directory lacks its receipt; preserving bytes: {entry}")
    return receipts


def _retire_directory_losslessly(
    source: Path,
    *,
    reason: str,
    rebuild_command: Sequence[str],
    storage_preflight: Mapping[str, Any],
) -> Path:
    """Certify then no-replace-move one rebuildable staging tree."""

    source = _unresolved_absolute(source)
    snapshot = _measure_directory_tree(source, role="cache staging retirement source")
    binding = _directory_retention_binding(
        source,
        snapshot,
        reason=reason,
        rebuild_command=rebuild_command,
        storage_preflight=storage_preflight,
    )
    name_max = _filesystem_name_max(source.parent)
    for ordinal in range(100_000_000):
        destination = source.with_name(_directory_retained_name(source.name, snapshot, ordinal))
        receipt_path = source.with_name(f"{DIRECTORY_RETENTION_RECEIPT_PREFIX}{binding}-{ordinal:08d}.json")
        for candidate, role in (
            (destination.name, "retained directory"),
            (receipt_path.name, "retained directory receipt"),
        ):
            _require_component_fits(candidate, name_max=name_max, role=role)
        # Receipt atomic-write sidecars are also proven before the receipt or
        # directory tree can be mutated.
        _preflight_atomic_components(receipt_path, b"{}\n")
        expected_receipt = _directory_retention_receipt(
            source,
            destination,
            snapshot,
            reason=reason,
            rebuild_command=rebuild_command,
            storage_preflight=storage_preflight,
        )
        if _path_exists_no_follow(receipt_path):
            existing, _measured = _validate_directory_retention_receipt(receipt_path)
            if existing != expected_receipt:
                continue
        elif _path_exists_no_follow(destination):
            continue
        else:
            atomic_json(receipt_path, expected_receipt)
            existing, _measured = _validate_directory_retention_receipt(receipt_path)
            if existing != expected_receipt:
                raise FeatureCacheError("directory retention receipt changed after publication")
        if _path_exists_no_follow(destination):
            if _path_exists_no_follow(source):
                raise FeatureCacheError("directory retention source and destination both exist; preserving trees")
            return destination
        current = _measure_directory_tree(source, role="cache staging retirement source revalidation")
        if current != snapshot:
            raise FeatureCacheError("cache staging changed after retention receipt; preserving tree")
        try:
            move_path_noreplace(
                source,
                destination,
                expected_identity=snapshot.directory_identity,
                role="cache staging lossless retirement",
                require_directory=True,
            )
        except _NoReplaceDestinationExists as exc:
            raise FeatureCacheError(
                "retained directory destination appeared after its receipt was published; "
                "preserving source, destination, and receipt"
            ) from exc
        _validate_directory_retention_receipt(receipt_path)
        return destination
    raise FeatureCacheError("cache staging retirement exhausted deterministic ordinals; preserving tree")


def _preflight_cache_creation_components(cache_root: Path) -> None:
    """Check actual filesystem names before any cache-creation mutation."""

    staging = _creation_staging_path(cache_root)
    parent_anchor = _existing_filesystem_anchor(cache_root.parent)
    name_max = _filesystem_name_max(cache_root.parent)
    for component in cache_root.parent.relative_to(parent_anchor).parts:
        _require_component_fits(component, name_max=name_max, role="cache parent creation")
    fixed = [cache_root.name, staging.name, *sorted(CERTIFIED_STAGING_ENTRY_SET)]
    for name in fixed:
        _require_component_fits(name, name_max=name_max, role="cache creation")
    empty_snapshot = BoundDirectorySnapshot((0, 0, 0, 0, 0), 0, "0" * 64, 0)
    _require_component_fits(
        _directory_retained_name(staging.name, empty_snapshot, 0),
        name_max=name_max,
        role="cache staging retention",
    )
    _require_component_fits(
        f"{DIRECTORY_RETENTION_RECEIPT_PREFIX}{'0' * 64}-00000000.json",
        name_max=name_max,
        role="cache staging retention receipt",
    )
    for target_name in sorted(ATOMIC_STAGING_TARGET_NAMES):
        _preflight_atomic_components(staging / target_name, b"{}\n")


def _validate_storage_preflight(value: object) -> dict[str, Any]:
    """Validate the exact receipt emitted by the production extractor."""

    if not isinstance(value, dict) or set(value) != STORAGE_PREFLIGHT_KEYS:
        raise FeatureCacheError("cache storage preflight schema is malformed")
    free = value.get("free_bytes_before")
    required = value.get("required_free_bytes")
    if type(free) is not int or free < 0 or type(required) is not int or required < 0:
        raise FeatureCacheError("cache storage preflight byte counts are malformed")
    if required > free or value.get("PASS") is not True:
        raise FeatureCacheError("cache storage preflight did not pass")
    selected_raw = value.get("selected_root")
    anchor_raw = value.get("filesystem_anchor")
    waterfall_raw = value.get("waterfall_order")
    existing_raw = value.get("existing_approved_roots")
    local_test = value.get("allow_local_output_for_tests")
    if (
        not isinstance(selected_raw, str)
        or not Path(selected_raw).is_absolute()
        or not isinstance(anchor_raw, str)
        or not Path(anchor_raw).is_absolute()
        or not isinstance(waterfall_raw, list)
        or not waterfall_raw
        or any(not isinstance(root, str) or not Path(root).is_absolute() for root in waterfall_raw)
        or not isinstance(existing_raw, list)
        or any(not isinstance(root, str) or not Path(root).is_absolute() for root in existing_raw)
        or type(local_test) is not bool
    ):
        raise FeatureCacheError("cache storage preflight path/type custody is malformed")
    selected = Path(selected_raw).resolve()
    if tuple(waterfall_raw) != APPROVED_SSD_WATERFALL:
        raise FeatureCacheError("cache production SSD waterfall order is not canonical")
    waterfall = [Path(root).resolve() for root in waterfall_raw]
    existing = [Path(root).resolve() for root in existing_raw]
    if any(root not in waterfall for root in existing) or existing != [root for root in waterfall if root in existing]:
        raise FeatureCacheError("cache existing-root waterfall custody is malformed")
    if not local_test:
        if not existing:
            raise FeatureCacheError("cache production preflight records no existing approved SSD root")
        first = existing[0]
        if selected != first and first not in selected.parents:
            raise FeatureCacheError("cache selected root is not under the first existing production SSD root")
    canonical_json_bytes(value)
    return value


def _staging_scratch_record(
    *,
    cache_root: Path,
    identity_sha256: str,
    rebuild_command: Sequence[str],
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_preflight = json.loads(canonical_json_bytes(dict(storage_preflight)))
    _validate_storage_preflight(normalized_preflight)
    return {
        "schema": STAGING_SCRATCH_SCHEMA,
        "final_cache_root": str(cache_root),
        "identity_sha256": identity_sha256,
        "rebuild_command": list(rebuild_command),
        "storage_preflight": normalized_preflight,
        "rebuildable": True,
        "score_authority": False,
        "safe_disposition": "REBUILD_IN_PLACE_OR_LOSSLESS_RETENTION_BEFORE_FINAL_PUBLISH",
    }


def _certified_staging_matches(
    staging_root: Path,
    *,
    expected_record: Mapping[str, Any],
) -> bool:
    try:
        record = _load_object(staging_root / STAGING_SCRATCH_NAME)
    except FeatureCacheError:
        return False
    # ``free_bytes_before`` is a measured preflight value and legitimately
    # changes after a large interrupted allocation.  It is certification
    # content, not cache identity.  Match every immutable scratch field while
    # the certification validator below binds the original preflight exactly.
    return (
        set(record) == set(expected_record)
        and record.get("schema") == expected_record.get("schema")
        and record.get("final_cache_root") == expected_record.get("final_cache_root")
        and record.get("identity_sha256") == expected_record.get("identity_sha256")
        and record.get("rebuild_command") == expected_record.get("rebuild_command")
        and isinstance(record.get("storage_preflight"), dict)
        and record.get("rebuildable") is True
        and record.get("score_authority") is False
        and record.get("safe_disposition") == expected_record.get("safe_disposition")
    )


def _validate_cache_certification(
    cache_root: Path,
    *,
    expected_identity_sha256: str,
) -> None:
    """Validate scratch/certification custody in staging and final locations."""

    scratch = _load_object(cache_root / STAGING_SCRATCH_NAME)
    certification = _load_object(cache_root / CERTIFICATION_NAME)
    scratch_keys = {
        "schema",
        "final_cache_root",
        "identity_sha256",
        "rebuild_command",
        "storage_preflight",
        "rebuildable",
        "score_authority",
        "safe_disposition",
    }
    certification_keys = {
        "schema",
        "cache_root",
        "rebuild_command",
        "storage_preflight",
        "large_artifact_policy",
        "rebuildable",
        "delete_or_move_before_complete",
        "cleanup_action_performed",
        "false_authority_flags",
    }
    final_root_raw = scratch.get("final_cache_root")
    if not isinstance(final_root_raw, str) or not final_root_raw:
        raise FeatureCacheError("cache staging scratch final root is malformed")
    final_root = Path(final_root_raw).expanduser().resolve()
    current_root = cache_root.expanduser().resolve(strict=True)
    if current_root not in {final_root, _creation_staging_path(final_root)}:
        raise FeatureCacheError("cache staging/final location does not match certification")
    rebuild_command = scratch.get("rebuild_command")
    storage_preflight = scratch.get("storage_preflight")
    if (
        set(scratch) != scratch_keys
        or scratch.get("schema") != STAGING_SCRATCH_SCHEMA
        or scratch.get("identity_sha256") != expected_identity_sha256
        or not isinstance(rebuild_command, list)
        or not rebuild_command
        or any(not isinstance(value, str) or not value for value in rebuild_command)
        or not isinstance(storage_preflight, dict)
        or scratch.get("rebuildable") is not True
        or scratch.get("score_authority") is not False
        or scratch.get("safe_disposition") != "REBUILD_IN_PLACE_OR_LOSSLESS_RETENTION_BEFORE_FINAL_PUBLISH"
    ):
        raise FeatureCacheError("cache creation staging scratch identity is malformed")
    _validate_storage_preflight(storage_preflight)
    false_authority = certification.get("false_authority_flags")
    if (
        set(certification) != certification_keys
        or certification.get("schema") != CERTIFICATION_SCHEMA
        or certification.get("cache_root") != str(final_root)
        or certification.get("rebuild_command") != rebuild_command
        or certification.get("storage_preflight") != storage_preflight
        or certification.get("rebuildable") is not True
        or certification.get("delete_or_move_before_complete") is not False
        or certification.get("cleanup_action_performed") is not False
        or certification.get("large_artifact_policy") != "CERTIFY_OR_BLOCK"
        or false_authority
        != {
            "score_authority": False,
            "promotion_eligible": False,
            "rank4_quotient_diagnostic_only": True,
            "rank4_replays_live_logits_bitwise": False,
        }
    ):
        raise FeatureCacheError("cache certification is malformed")
    canonical_json_bytes(scratch)
    canonical_json_bytes(certification)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise FeatureCacheError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise FeatureCacheError(f"{name} must be positive")
    return result


def _shape(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(_positive_int(item, name=f"{name} dimension") for item in value)
    except TypeError as exc:
        raise FeatureCacheError(f"{name} must be a shape") from exc
    if not result:
        raise FeatureCacheError(f"{name} must be nonempty")
    return result


def _finite_f32(value: np.ndarray, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.float32:
        raise FeatureCacheError(f"{name} must be float32, got {raw.dtype}")
    if shape is not None and raw.shape != shape:
        raise FeatureCacheError(f"{name} shape {raw.shape} != {shape}")
    array = np.ascontiguousarray(raw)
    if not np.isfinite(array).all():
        raise FeatureCacheError(f"{name} contains non-finite values")
    return array


def _scan_finite_frame_chunks(
    value: np.ndarray,
    *,
    name: str,
    chunk_frames: int = 1,
) -> None:
    """Check a cache array without allocating a cache-sized boolean temporary."""

    frames = _positive_int(chunk_frames, name="chunk_frames")
    if value.ndim < 1:
        raise FeatureCacheError(f"{name} must have a frame dimension")
    for start in range(0, value.shape[0], frames):
        stop = min(value.shape[0], start + frames)
        if not np.isfinite(value[start:stop]).all():
            raise FeatureCacheError(f"{name} contains non-finite values in frames [{start}, {stop})")


def build_immutable_identity(
    *,
    source_files: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    frame_count: int,
    live_slice_shape: Sequence[int],
    quotient_slice_shape: Sequence[int],
) -> dict[str, Any]:
    """Return the complete identity used to admit creation and resume."""

    frame_count = _positive_int(frame_count, name="frame_count")
    live_shape = _shape(live_slice_shape, name="live_slice_shape")
    quotient_shape = _shape(quotient_slice_shape, name="quotient_slice_shape")
    if live_shape[1:] != quotient_shape[1:]:
        raise FeatureCacheError("live and quotient slices must share spatial geometry")
    normalized_sources: dict[str, dict[str, Any]] = {}
    for role, raw in sorted(source_files.items()):
        if not isinstance(role, str) or not role or not isinstance(raw, Mapping):
            raise FeatureCacheError("source_files must map nonempty roles to mappings")
        path = raw.get("path")
        byte_count = raw.get("bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise FeatureCacheError(f"source custody for {role!r} is malformed")
        normalized_sources[role] = {
            "path": path,
            "bytes": byte_count,
            "sha256": digest,
        }
    identity = {
        "schema": CACHE_SCHEMA,
        "source_files": normalized_sources,
        "config": dict(config),
        "arrays": {
            "live_logits": {
                "path": LIVE_LOGITS_NAME,
                "dtype": "<f4",
                "shape": [frame_count, *live_shape],
            },
            "quotient_features": {
                "path": QUOTIENT_FEATURES_NAME,
                "dtype": "<f4",
                "shape": [frame_count, *quotient_shape],
            },
        },
    }
    # Round-trip now so a mutable/non-finite config cannot become resume identity.
    return json.loads(canonical_json_bytes(identity))


def source_file_row(path: str | Path) -> dict[str, Any]:
    unresolved = _unresolved_absolute(path)
    _require_directory_chain(unresolved.parent)
    try:
        normalized_parent = unresolved.parent.resolve(strict=True)
    except OSError as exc:
        raise FeatureCacheError(f"source custody parent is unavailable: {unresolved.parent}") from exc
    file_path = normalized_parent / unresolved.name
    _require_regular_file(file_path, role="source custody file")
    descriptor, opened = _open_bound_file(file_path, flags=os.O_RDONLY, role="source custody file")
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            digest.update(chunk)
        _assert_bound_file(file_path, descriptor, opened, role="source custody file")
    except OSError as exc:
        raise FeatureCacheError(f"source custody file cannot be read; preserving bytes: {file_path}") from exc
    finally:
        os.close(descriptor)
    return {
        "path": str(file_path),
        "bytes": opened.st_size,
        "sha256": digest.hexdigest(),
    }


def open_gt_f1_stored_memmap(npz_path: str | Path) -> np.memmap:
    """Open only the real ``gt_f1`` stored member; never materialize the NPZ."""

    # Deliberately lazy: extraction admits the factorization source bytes
    # before this module is executed.
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap

    unresolved = _unresolved_absolute(npz_path)
    _require_directory_chain(unresolved.parent)
    _require_regular_file(unresolved, role="gt_n600 NPZ custody file")
    mapped = open_stored_npy_memmap(unresolved, "gt_f1")
    if not isinstance(mapped, np.memmap):  # pragma: no cover - dependency contract
        raise FeatureCacheError("gt_f1 stored-member reader did not return a memmap")
    if mapped.dtype != np.uint8 or mapped.ndim != 4 or mapped.shape[-1] != 3:
        raise FeatureCacheError(f"gt_f1 must be (N,H,W,3) uint8, got {mapped.shape}/{mapped.dtype}")
    return mapped


@dataclass(frozen=True)
class PositiveControlResult:
    bitwise_live_logits_equal: bool
    differing_live_logit_values: int
    algebraic_argmax_disagreements: int
    diagnostic_only_algebraic_disagreement: bool


def validate_live_logit_positive_control(
    cached_live_logits: np.ndarray,
    fresh_live_logits: np.ndarray,
    *,
    algebraic_argmax: np.ndarray | None = None,
) -> PositiveControlResult:
    """Require direct-forward bit identity; keep algebraic disagreement diagnostic.

    The frame-195 generic-f64/native-f32 boundary belongs in
    ``algebraic_argmax_disagreements``.  It cannot block cache extraction when
    the direct scorer forward is bit-identical.
    """

    cached = _finite_f32(cached_live_logits, name="cached live logits")
    fresh = _finite_f32(fresh_live_logits, name="fresh live logits", shape=cached.shape)
    differing = int(np.count_nonzero(cached.view(np.uint32) != fresh.view(np.uint32)))
    if differing:
        raise FeatureCacheError(f"fresh live-logit positive control differs in {differing} float32 values")
    algebraic_disagreements = 0
    if algebraic_argmax is not None:
        algebraic = np.asarray(algebraic_argmax)
        expected_shape = cached.shape[1:] if cached.ndim >= 2 else ()
        if algebraic.shape != expected_shape or algebraic.dtype.kind not in "iu":
            raise FeatureCacheError("algebraic_argmax shape/dtype does not match logits")
        live_argmax = np.argmax(cached, axis=0)
        algebraic_disagreements = int(np.count_nonzero(algebraic != live_argmax))
    return PositiveControlResult(
        bitwise_live_logits_equal=True,
        differing_live_logit_values=0,
        algebraic_argmax_disagreements=algebraic_disagreements,
        diagnostic_only_algebraic_disagreement=algebraic_disagreements > 0,
    )


def _positive_control_row(frame: int, control: PositiveControlResult) -> dict[str, Any]:
    return {
        "schema": POSITIVE_CONTROL_SCHEMA,
        "frame": frame,
        **control.__dict__,
    }


def _validate_positive_control(value: object, *, frame_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FeatureCacheError("complete cache positive-control schema is malformed")
    if (
        set(value)
        != {
            "schema",
            "frame",
            "bitwise_live_logits_equal",
            "differing_live_logit_values",
            "algebraic_argmax_disagreements",
            "diagnostic_only_algebraic_disagreement",
        }
        or value.get("schema") != POSITIVE_CONTROL_SCHEMA
        or type(value.get("frame")) is not int
        or not 0 <= value["frame"] < frame_count
        or value.get("bitwise_live_logits_equal") is not True
        or type(value.get("differing_live_logit_values")) is not int
        or value.get("differing_live_logit_values") != 0
        or type(value.get("algebraic_argmax_disagreements")) is not int
        or value["algebraic_argmax_disagreements"] < 0
        or type(value.get("diagnostic_only_algebraic_disagreement")) is not bool
        or value["diagnostic_only_algebraic_disagreement"] is not (value["algebraic_argmax_disagreements"] > 0)
    ):
        raise FeatureCacheError("complete cache positive-control schema is malformed")
    canonical_json_bytes(value)
    return value


def _completion_control(
    *,
    identity_sha256: str,
    committed_frame_count: int,
    terminal_frame_commitment_sha256: str,
    positive_control: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": COMPLETION_CONTROL_SCHEMA,
        "identity_sha256": _require_sha256(identity_sha256, name="completion identity"),
        "committed_frame_count": committed_frame_count,
        "terminal_frame_commitment_sha256": _require_sha256(
            terminal_frame_commitment_sha256,
            name="completion terminal frame commitment",
        ),
        "positive_control": dict(positive_control),
        "integrity_scope": INTEGRITY_SCOPE,
        "writer_rewrite_limit": WRITER_REWRITE_LIMIT,
    }


def _validate_completion_control(
    value: object,
    *,
    expected_identity_sha256: str,
    expected_frame_count: int,
    expected_terminal_frame_commitment_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != COMPLETION_CONTROL_KEYS:
        raise FeatureCacheError("cache completion control schema is malformed")
    if (
        value.get("schema") != COMPLETION_CONTROL_SCHEMA
        or value.get("identity_sha256") != expected_identity_sha256
        or type(value.get("committed_frame_count")) is not int
        or value.get("committed_frame_count") != expected_frame_count
        or value.get("terminal_frame_commitment_sha256") != expected_terminal_frame_commitment_sha256
        or value.get("integrity_scope") != INTEGRITY_SCOPE
        or value.get("writer_rewrite_limit") != WRITER_REWRITE_LIMIT
    ):
        raise FeatureCacheError("cache completion control does not match the validated terminal chain")
    _require_sha256(value.get("identity_sha256"), name="completion identity")
    _require_sha256(
        value.get("terminal_frame_commitment_sha256"),
        name="completion terminal frame commitment",
    )
    _validate_positive_control(value.get("positive_control"), frame_count=expected_frame_count)
    canonical_json_bytes(value)
    return value


@dataclass(frozen=True)
class CacheValidation:
    root: Path
    identity: dict[str, Any]
    progress: dict[str, Any]
    live_logits: np.memmap
    quotient_features: np.memmap

    @property
    def next_frame(self) -> int:
        return int(self.progress["next_frame"])

    @property
    def complete(self) -> bool:
        return self.progress["status"] == "complete"


class SegnetHeadFeatureCache:
    """Create/resume one immutable-identity, frame-committed cache."""

    def __init__(self, validation: CacheValidation):
        self.root = validation.root
        self.identity = validation.identity
        self.progress = validation.progress
        self.live_logits = validation.live_logits
        self.quotient_features = validation.quotient_features

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        identity: Mapping[str, Any],
        rebuild_command: Sequence[str],
        storage_preflight: Mapping[str, Any],
    ) -> SegnetHeadFeatureCache:
        unresolved_root = _unresolved_absolute(root)
        _require_directory_chain(unresolved_root.parent)
        parent_anchor = _existing_filesystem_anchor(unresolved_root.parent)
        cache_root = parent_anchor.joinpath(
            *unresolved_root.parent.relative_to(parent_anchor).parts,
            unresolved_root.name,
        )
        _preflight_cache_creation_components(cache_root)

        normalized_identity = json.loads(canonical_json_bytes(dict(identity)))
        _validate_identity(normalized_identity)
        identity_sha256 = hashlib.sha256(canonical_json_bytes(normalized_identity)).hexdigest()
        staging_root = _creation_staging_path(cache_root)
        scratch_record = _staging_scratch_record(
            cache_root=cache_root,
            identity_sha256=identity_sha256,
            rebuild_command=rebuild_command,
            storage_preflight=storage_preflight,
        )
        progress = {
            "schema": PROGRESS_SCHEMA,
            "identity_sha256": identity_sha256,
            "status": "partial",
            "next_frame": 0,
            "committed_frames": [],
            "frame_chain_head_sha256": identity_sha256,
            "completion_positive_control": None,
        }
        certification = {
            "schema": CERTIFICATION_SCHEMA,
            "cache_root": str(cache_root),
            "rebuild_command": list(rebuild_command),
            "storage_preflight": scratch_record["storage_preflight"],
            "large_artifact_policy": "CERTIFY_OR_BLOCK",
            "rebuildable": True,
            "delete_or_move_before_complete": False,
            "cleanup_action_performed": False,
            "false_authority_flags": {
                "score_authority": False,
                "promotion_eligible": False,
                "rank4_quotient_diagnostic_only": True,
                "rank4_replays_live_logits_bitwise": False,
            },
        }

        # Only after every actual component has been proved portable may the
        # missing directory chain be created.  Cache/staging names themselves
        # are always exclusive and never removed or replace-published.
        cache_root.parent.mkdir(parents=True, exist_ok=True)
        _require_directory_chain(cache_root.parent)
        _validate_retained_directory_custody(cache_root.parent)
        if _path_exists_no_follow(cache_root):
            _require_directory(cache_root, role="cache root")
            raise FeatureCacheError(
                f"refusing existing cache directory; use explicit resume for validated custody: {cache_root}"
            )

        if _path_exists_no_follow(staging_root):
            _require_directory(staging_root, role="cache creation staging root")
            if not any(staging_root.iterdir()):
                _retire_directory_losslessly(
                    staging_root,
                    reason="EMPTY_INTERRUPTED_CACHE_STAGING",
                    rebuild_command=rebuild_command,
                    storage_preflight=scratch_record["storage_preflight"],
                )
            else:
                entries = {entry.name: entry for entry in staging_root.iterdir()}
                prepared_targets = {
                    atomic_prepared_path(staging_root / name).name: name
                    for name in CERTIFIED_STAGING_ENTRY_SET
                    if name not in {LIVE_LOGITS_NAME, QUOTIENT_FEATURES_NAME}
                }
                generation_targets = {
                    entry_name
                    for entry_name in entries
                    if any(
                        entry_name.startswith(f".{target_name}{ATOMIC_GENERATION_SUFFIX}-")
                        and ATOMIC_GENERATION_RE.fullmatch(entry_name) is not None
                        for target_name in prepared_targets.values()
                    )
                }
                transaction_targets = _active_atomic_transaction_names(
                    entries,
                    target_names=tuple(prepared_targets.values()),
                )
                completion_targets = active_atomic_completion_names(
                    entries,
                    target_names=tuple(prepared_targets.values()),
                )
                retained_names = _validated_governed_retained_entries(
                    entries,
                    target_names=ATOMIC_STAGING_TARGET_NAMES,
                    role="cache creation staging",
                )
                allowed = (
                    CERTIFIED_STAGING_ENTRY_SET
                    | set(prepared_targets)
                    | generation_targets
                    | transaction_targets
                    | completion_targets
                    | retained_names
                )
                unknown = sorted(set(entries) - allowed)
                if unknown:
                    raise FeatureCacheError(
                        f"cache staging layout is not exactly certified; unknown={unknown}; preserving bytes"
                    )
                for name, entry in entries.items():
                    if name not in retained_names:
                        _require_regular_file(entry, role=f"cache staging entry {name}")

                expected_metadata: dict[str, bytes] = {
                    STAGING_SCRATCH_NAME: canonical_json_bytes(scratch_record) + b"\n",
                    MARKER_NAME: MARKER_BYTES,
                    MANIFEST_NAME: canonical_json_bytes(normalized_identity) + b"\n",
                    PROGRESS_NAME: canonical_json_bytes(progress) + b"\n",
                    CERTIFICATION_NAME: canonical_json_bytes(certification) + b"\n",
                }
                scratch_path = staging_root / STAGING_SCRATCH_NAME
                scratch_prepared = atomic_prepared_path(scratch_path)
                if (
                    _path_exists_no_follow(scratch_prepared)
                    or _atomic_generation_paths(scratch_path)
                    or _atomic_transaction_paths(scratch_path)
                ):
                    # If the first record was already committed, its original
                    # observations are the deterministic payload to reconcile.
                    scratch_payload = (
                        canonical_json_bytes(_load_object(scratch_path)) + b"\n"
                        if _path_exists_no_follow(scratch_path)
                        else expected_metadata[STAGING_SCRATCH_NAME]
                    )
                    _atomic_bytes(scratch_path, scratch_payload)
                if not _certified_staging_matches(staging_root, expected_record=scratch_record):
                    raise FeatureCacheError(
                        f"refusing unidentified or identity-drifted cache staging directory: {staging_root}"
                    )
                persisted_scratch = _load_object(scratch_path)
                expected_metadata[STAGING_SCRATCH_NAME] = canonical_json_bytes(persisted_scratch) + b"\n"
                persisted_certification = dict(certification)
                persisted_certification["storage_preflight"] = persisted_scratch["storage_preflight"]
                expected_metadata[CERTIFICATION_NAME] = canonical_json_bytes(persisted_certification) + b"\n"

                # Every later stable temporary must be readable and match the
                # exact deterministic metadata payload before the scratch tree
                # can be certified rebuildable.
                for prepared_name, target_name in prepared_targets.items():
                    prepared_path = staging_root / prepared_name
                    if (
                        _path_exists_no_follow(prepared_path)
                        or _atomic_generation_paths(staging_root / target_name)
                        or _atomic_transaction_paths(staging_root / target_name)
                    ):
                        _atomic_bytes(staging_root / target_name, expected_metadata[target_name])

                final_entries = {entry.name: entry for entry in staging_root.iterdir()}
                final_retained_names = _validated_governed_retained_entries(
                    final_entries,
                    target_names=ATOMIC_STAGING_TARGET_NAMES,
                    role="cache creation staging",
                )
                final_completion_names = active_atomic_completion_names(
                    final_entries,
                    target_names=tuple(ATOMIC_STAGING_TARGET_NAMES),
                )
                active_names = set(final_entries) - final_retained_names - final_completion_names
                if active_names != CERTIFIED_STAGING_ENTRY_SET:
                    _retire_directory_losslessly(
                        staging_root,
                        reason="IDENTITY_MATCHED_INCOMPLETE_CACHE_STAGING",
                        rebuild_command=rebuild_command,
                        storage_preflight=persisted_scratch["storage_preflight"],
                    )
                else:
                    _validate_cache_certification(
                        staging_root,
                        expected_identity_sha256=identity_sha256,
                    )
                    try:
                        validation = validate_feature_cache(
                            staging_root,
                            expected_identity=normalized_identity,
                        )
                        if validation.next_frame != 0 or validation.progress["status"] != "partial":
                            raise FeatureCacheError("creation staging cache is not at the initial prefix")
                    except FeatureCacheError:
                        _require_certified_staging_layout(staging_root)
                        _retire_directory_losslessly(
                            staging_root,
                            reason="CERTIFIED_REBUILDABLE_INVALID_CACHE_STAGING",
                            rebuild_command=rebuild_command,
                            storage_preflight=persisted_scratch["storage_preflight"],
                        )
                    else:
                        del validation
                        _fsync_directory(staging_root)
                        staging_identity = _stable_file_identity(
                            _require_directory(staging_root, role="validated cache publication source")
                        )
                        move_path_noreplace(
                            staging_root,
                            cache_root,
                            expected_identity=staging_identity,
                            role="cache final no-replace publication",
                            require_directory=True,
                        )
                        return cls(
                            validate_feature_cache(
                                cache_root,
                                expected_identity=normalized_identity,
                                writable=True,
                            )
                        )

        try:
            staging_root.mkdir()
        except FileExistsError as exc:
            raise FeatureCacheError(
                f"cache staging destination appeared before exclusive creation; preserving tree: {staging_root}"
            ) from exc
        # This record is deliberately first: a crash may leave an empty
        # directory or identified rebuildable scratch, never anonymous arrays.
        atomic_json(staging_root / STAGING_SCRATCH_NAME, scratch_record)
        marker = staging_root / MARKER_NAME
        _atomic_bytes(marker, MARKER_BYTES)
        atomic_json(staging_root / MANIFEST_NAME, normalized_identity)
        arrays = normalized_identity["arrays"]
        live_spec = arrays["live_logits"]
        quotient_spec = arrays["quotient_features"]
        live = np.lib.format.open_memmap(
            staging_root / LIVE_LOGITS_NAME,
            mode="w+",
            dtype=np.dtype(live_spec["dtype"]),
            shape=tuple(live_spec["shape"]),
        )
        quotient = np.lib.format.open_memmap(
            staging_root / QUOTIENT_FEATURES_NAME,
            mode="w+",
            dtype=np.dtype(quotient_spec["dtype"]),
            shape=tuple(quotient_spec["shape"]),
        )
        live.flush()
        quotient.flush()
        _fsync_file(staging_root / LIVE_LOGITS_NAME)
        _fsync_file(staging_root / QUOTIENT_FEATURES_NAME)
        del live, quotient
        atomic_json(staging_root / PROGRESS_NAME, progress)
        atomic_json(staging_root / CERTIFICATION_NAME, certification)
        _validate_cache_certification(
            staging_root,
            expected_identity_sha256=identity_sha256,
        )
        validation = validate_feature_cache(
            staging_root,
            expected_identity=normalized_identity,
        )
        if validation.next_frame != 0 or validation.progress["status"] != "partial":
            raise FeatureCacheError("new cache staging validation did not recover the initial prefix")
        del validation
        _fsync_directory(staging_root)
        staging_identity = _stable_file_identity(_require_directory(staging_root, role="new cache publication source"))
        move_path_noreplace(
            staging_root,
            cache_root,
            expected_identity=staging_identity,
            role="cache final no-replace publication",
            require_directory=True,
        )
        return cls(
            validate_feature_cache(
                cache_root,
                expected_identity=normalized_identity,
                writable=True,
            )
        )

    @classmethod
    def resume(
        cls,
        root: str | Path,
        *,
        expected_identity: Mapping[str, Any],
    ) -> SegnetHeadFeatureCache:
        return cls(
            validate_feature_cache(
                root,
                expected_identity=expected_identity,
                writable=True,
            )
        )

    @property
    def next_frame(self) -> int:
        return int(self.progress["next_frame"])

    @property
    def frame_count(self) -> int:
        return int(self.live_logits.shape[0])

    def commit_frame(
        self,
        frame_index: int,
        live_logits: np.ndarray,
        quotient_features: np.ndarray,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.progress["status"] == "complete":
            raise FeatureCacheError("cannot append to a complete cache")
        if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)):
            raise FeatureCacheError("frame index must be a Python or NumPy integer")
        frame_index = int(frame_index)
        if frame_index != self.next_frame:
            raise FeatureCacheError(f"frame commit must be contiguous at {self.next_frame}, got {frame_index}")
        if frame_index >= self.frame_count:
            raise FeatureCacheError("frame index exceeds cache geometry")
        live = _finite_f32(
            live_logits,
            name="live logits",
            shape=tuple(self.live_logits.shape[1:]),
        )
        quotient = _finite_f32(
            quotient_features,
            name="quotient features",
            shape=tuple(self.quotient_features.shape[1:]),
        )
        self.live_logits[frame_index] = live
        self.quotient_features[frame_index] = quotient
        self.live_logits.flush()
        self.quotient_features.flush()
        _fsync_file(self.root / LIVE_LOGITS_NAME)
        _fsync_file(self.root / QUOTIENT_FEATURES_NAME)
        diagnostics_row = json.loads(canonical_json_bytes({} if diagnostics is None else dict(diagnostics)))
        live_hash = sha256_float32_slice(self.live_logits[frame_index])
        quotient_hash = sha256_float32_slice(self.quotient_features[frame_index])
        previous_commitment = _require_sha256(
            self.progress.get("frame_chain_head_sha256"),
            name="progress frame-chain head",
        )
        frame_commitment = _frame_commitment_sha256(
            identity_sha256=self.progress["identity_sha256"],
            previous_frame_commitment_sha256=previous_commitment,
            frame=frame_index,
            live_logits_sha256=live_hash,
            quotient_features_sha256=quotient_hash,
            diagnostics=diagnostics_row,
        )
        row = {
            "frame": frame_index,
            "live_logits_sha256": live_hash,
            "quotient_features_sha256": quotient_hash,
            "diagnostics": diagnostics_row,
            "previous_frame_commitment_sha256": previous_commitment,
            "frame_commitment_sha256": frame_commitment,
        }
        canonical_json_bytes(row)
        updated = dict(self.progress)
        updated["committed_frames"] = [*self.progress["committed_frames"], row]
        updated["next_frame"] = frame_index + 1
        updated["frame_chain_head_sha256"] = frame_commitment
        updated["status"] = (
            "ready_for_completion_validation" if updated["next_frame"] == self.frame_count else "partial"
        )
        atomic_json(
            self.root / PROGRESS_NAME,
            updated,
            expected_prior_payloads=(canonical_json_bytes(self.progress) + b"\n",),
        )
        self.progress = updated
        return row

    def mark_complete(
        self,
        *,
        positive_frame: int,
        fresh_live_logits: np.ndarray,
        algebraic_argmax: np.ndarray | None = None,
    ) -> PositiveControlResult:
        if self.next_frame != self.frame_count:
            raise FeatureCacheError(f"completion requires all {self.frame_count} frames, have {self.next_frame}")
        if isinstance(positive_frame, (bool, np.bool_)) or not isinstance(positive_frame, (int, np.integer)):
            raise FeatureCacheError("positive_frame must be a Python or NumPy integer")
        positive_frame = int(positive_frame)
        if not 0 <= positive_frame < self.frame_count:
            raise FeatureCacheError("positive_frame is outside cache geometry")
        _scan_finite_frame_chunks(self.live_logits, name="live-logit cache")
        _scan_finite_frame_chunks(
            self.quotient_features,
            name="quotient-feature cache",
        )
        control = validate_live_logit_positive_control(
            np.asarray(self.live_logits[positive_frame]),
            fresh_live_logits,
            algebraic_argmax=algebraic_argmax,
        )
        positive_control = _positive_control_row(positive_frame, control)
        completion = _completion_control(
            identity_sha256=self.progress["identity_sha256"],
            committed_frame_count=self.frame_count,
            terminal_frame_commitment_sha256=self.progress["frame_chain_head_sha256"],
            positive_control=positive_control,
        )
        # Write the independently structured terminal control first.  A crash
        # before the progress flip leaves a completion-ready cache that can
        # deterministically repeat this fresh-forward control on resume.
        atomic_json(self.root / COMPLETION_CONTROL_NAME, completion)
        updated = dict(self.progress)
        updated["status"] = "complete"
        updated["completion_positive_control"] = positive_control
        atomic_json(
            self.root / PROGRESS_NAME,
            updated,
            expected_prior_payloads=(canonical_json_bytes(self.progress) + b"\n",),
        )
        self.progress = updated
        # Final parse-back revalidates every committed slice and the positive row.
        validate_feature_cache(
            self.root,
            expected_identity=self.identity,
            require_complete=True,
        )
        return control


def _validate_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("schema") != CACHE_SCHEMA:
        raise FeatureCacheError("cache identity schema mismatch")
    arrays = identity.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != {
        "live_logits",
        "quotient_features",
    }:
        raise FeatureCacheError("cache identity arrays are malformed")
    expected_names = {
        "live_logits": LIVE_LOGITS_NAME,
        "quotient_features": QUOTIENT_FEATURES_NAME,
    }
    frame_count: int | None = None
    spatial: tuple[int, ...] | None = None
    for role, filename in expected_names.items():
        spec = arrays[role]
        if not isinstance(spec, Mapping) or spec.get("path") != filename or spec.get("dtype") != "<f4":
            raise FeatureCacheError(f"cache {role} specification is malformed")
        shape = _shape(spec.get("shape", ()), name=f"{role} shape")
        if len(shape) != 4:
            raise FeatureCacheError(f"cache {role} must have (N,C,H,W) geometry")
        if frame_count is None:
            frame_count = shape[0]
            spatial = shape[2:]
        elif shape[0] != frame_count or shape[2:] != spatial:
            raise FeatureCacheError("cache arrays disagree on frame/spatial geometry")
    if not isinstance(identity.get("source_files"), Mapping) or not isinstance(identity.get("config"), Mapping):
        raise FeatureCacheError("cache identity lacks source/config custody")
    canonical_json_bytes(dict(identity))


def validate_feature_cache(
    root: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    require_complete: bool = False,
    writable: bool = False,
) -> CacheValidation:
    """Parse and rehash the cache prefix from disk.

    Read-only memmaps are the default.  Only
    :class:`SegnetHeadFeatureCache` creation/resume requests writable mappings.
    """

    unresolved_root = _unresolved_absolute(root)
    _require_directory_chain(unresolved_root.parent)
    _require_directory(unresolved_root, role="cache root")
    cache_root = unresolved_root.resolve(strict=True)
    _require_directory(cache_root, role="cache root")
    runtime_prepared = {
        atomic_prepared_path(cache_root / PROGRESS_NAME).name,
        atomic_prepared_path(cache_root / COMPLETION_CONTROL_NAME).name,
    }
    try:
        entries = {entry.name: entry for entry in cache_root.iterdir()}
    except OSError as exc:
        raise FeatureCacheError("cache root entries cannot be enumerated") from exc
    runtime_generations = {
        entry_name
        for entry_name in entries
        if any(
            entry_name.startswith(f".{target_name}{ATOMIC_GENERATION_SUFFIX}-")
            and ATOMIC_GENERATION_RE.fullmatch(entry_name) is not None
            for target_name in (PROGRESS_NAME, COMPLETION_CONTROL_NAME)
        )
    }
    runtime_transactions = _active_atomic_transaction_names(
        entries,
        target_names=(PROGRESS_NAME, COMPLETION_CONTROL_NAME),
    )
    runtime_completions = active_atomic_completion_names(
        entries,
        target_names=tuple(ATOMIC_CACHE_TARGET_NAMES),
    )
    retained_names = _validated_governed_retained_entries(
        entries,
        target_names=ATOMIC_CACHE_TARGET_NAMES,
        role="cache root",
    )
    allowed_entries = (
        CERTIFIED_STAGING_ENTRY_SET
        | {COMPLETION_CONTROL_NAME}
        | runtime_prepared
        | runtime_generations
        | runtime_transactions
        | runtime_completions
        | retained_names
    )
    unknown = sorted(set(entries) - allowed_entries)
    missing = sorted(CERTIFIED_STAGING_ENTRY_SET - set(entries))
    if unknown or missing:
        raise FeatureCacheError(
            f"cache root layout is not canonical; unknown={unknown} missing={missing}; preserving bytes"
        )
    for prepared_name in sorted(
        (runtime_prepared | runtime_generations | runtime_transactions | runtime_completions) & set(entries)
    ):
        _read_stable_bytes(entries[prepared_name], role=f"cache runtime prepared file {prepared_name}")
    marker_path = cache_root / MARKER_NAME
    _require_regular_file(marker_path, role="cache marker")
    if _read_stable_bytes(marker_path, role="cache marker") != MARKER_BYTES:
        raise FeatureCacheError("cache marker mismatch")
    identity = _load_object(cache_root / MANIFEST_NAME)
    _validate_identity(identity)
    if expected_identity is not None and identity != json.loads(canonical_json_bytes(dict(expected_identity))):
        raise FeatureCacheError("cache immutable identity mismatch")
    expected_identity_hash = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    _validate_cache_certification(
        cache_root,
        expected_identity_sha256=expected_identity_hash,
    )
    progress = _load_object(cache_root / PROGRESS_NAME)
    if (
        set(progress) != PROGRESS_KEYS
        or progress.get("schema") != PROGRESS_SCHEMA
        or progress.get("identity_sha256") != expected_identity_hash
    ):
        raise FeatureCacheError("cache progress identity/schema mismatch")
    arrays = identity["arrays"]
    mmap_mode = "r+" if writable else "r"
    _require_regular_file(cache_root / LIVE_LOGITS_NAME, role="live-logit array")
    _require_regular_file(cache_root / QUOTIENT_FEATURES_NAME, role="quotient-feature array")
    try:
        live = np.load(cache_root / LIVE_LOGITS_NAME, mmap_mode=mmap_mode)
        quotient = np.load(cache_root / QUOTIENT_FEATURES_NAME, mmap_mode=mmap_mode)
    except (OSError, ValueError, EOFError) as exc:
        raise FeatureCacheError("cache array memmap parse failed") from exc
    if not isinstance(live, np.memmap) or not isinstance(quotient, np.memmap):
        raise FeatureCacheError("cache arrays must parse back as memmaps")
    if live.dtype != np.dtype("<f4") or tuple(live.shape) != tuple(arrays["live_logits"]["shape"]):
        raise FeatureCacheError("live-logit array parse-back mismatch")
    if quotient.dtype != np.dtype("<f4") or tuple(quotient.shape) != tuple(arrays["quotient_features"]["shape"]):
        raise FeatureCacheError("quotient-feature array parse-back mismatch")
    rows = progress.get("committed_frames")
    next_frame = progress.get("next_frame")
    if (
        not isinstance(rows, list)
        or isinstance(next_frame, bool)
        or not isinstance(next_frame, int)
        or not 0 <= next_frame <= live.shape[0]
        or len(rows) != next_frame
    ):
        raise FeatureCacheError("cache committed prefix is malformed")
    expected_previous_commitment = expected_identity_hash
    for expected_frame, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != COMMITTED_FRAME_KEYS
            or type(row.get("frame")) is not int
            or row["frame"] != expected_frame
        ):
            raise FeatureCacheError("cache frame rows are not a contiguous prefix")
        live_hash = row.get("live_logits_sha256")
        quotient_hash = row.get("quotient_features_sha256")
        diagnostics = row.get("diagnostics")
        previous_commitment = row.get("previous_frame_commitment_sha256")
        frame_commitment = row.get("frame_commitment_sha256")
        if (
            not isinstance(live_hash, str)
            or LOWER_SHA256_RE.fullmatch(live_hash) is None
            or not isinstance(quotient_hash, str)
            or LOWER_SHA256_RE.fullmatch(quotient_hash) is None
            or not isinstance(previous_commitment, str)
            or LOWER_SHA256_RE.fullmatch(previous_commitment) is None
            or not isinstance(frame_commitment, str)
            or LOWER_SHA256_RE.fullmatch(frame_commitment) is None
            or not isinstance(diagnostics, dict)
        ):
            raise FeatureCacheError("cache committed-frame row schema is malformed")
        canonical_json_bytes(diagnostics)
        if previous_commitment != expected_previous_commitment:
            raise FeatureCacheError(f"cache frame {expected_frame} predecessor commitment mismatch")
        expected_commitment = _frame_commitment_sha256(
            identity_sha256=expected_identity_hash,
            previous_frame_commitment_sha256=expected_previous_commitment,
            frame=expected_frame,
            live_logits_sha256=live_hash,
            quotient_features_sha256=quotient_hash,
            diagnostics=diagnostics,
        )
        if frame_commitment != expected_commitment:
            raise FeatureCacheError(f"cache frame {expected_frame} commitment mismatch")
        if row.get("live_logits_sha256") != sha256_float32_slice(live[expected_frame]):
            raise FeatureCacheError(f"live-logit slice {expected_frame} hash mismatch")
        if row.get("quotient_features_sha256") != sha256_float32_slice(quotient[expected_frame]):
            raise FeatureCacheError(f"quotient slice {expected_frame} hash mismatch")
        expected_previous_commitment = frame_commitment
    if progress.get("frame_chain_head_sha256") != expected_previous_commitment:
        raise FeatureCacheError("cache progress frame-chain head mismatch")
    status = progress.get("status")
    allowed = {"partial", "ready_for_completion_validation", "complete"}
    if status not in allowed:
        raise FeatureCacheError("cache progress status is invalid")
    if status == "partial" and next_frame >= live.shape[0]:
        raise FeatureCacheError("partial cache must stop before the full frame prefix")
    if status == "ready_for_completion_validation" and next_frame != live.shape[0]:
        raise FeatureCacheError("completion-ready cache must carry the full frame prefix")
    completion_path = cache_root / COMPLETION_CONTROL_NAME
    completion = _load_object(completion_path) if _path_exists_no_follow(completion_path) else None
    if status == "partial" and completion is not None:
        raise FeatureCacheError("partial cache carries a terminal completion control")
    if status == "ready_for_completion_validation" and completion is not None:
        _validate_completion_control(
            completion,
            expected_identity_sha256=expected_identity_hash,
            expected_frame_count=next_frame,
            expected_terminal_frame_commitment_sha256=expected_previous_commitment,
        )
    if status == "complete":
        control = progress.get("completion_positive_control")
        if next_frame != live.shape[0] or completion is None:
            raise FeatureCacheError("complete cache lacks full prefix/terminal completion control")
        validated_control = _validate_positive_control(control, frame_count=next_frame)
        validated_completion = _validate_completion_control(
            completion,
            expected_identity_sha256=expected_identity_hash,
            expected_frame_count=next_frame,
            expected_terminal_frame_commitment_sha256=expected_previous_commitment,
        )
        if validated_completion["positive_control"] != validated_control:
            raise FeatureCacheError("progress and terminal positive controls disagree")
        _scan_finite_frame_chunks(live, name="live-logit cache")
        _scan_finite_frame_chunks(quotient, name="quotient-feature cache")
    elif progress.get("completion_positive_control") is not None:
        raise FeatureCacheError("partial cache carries a completion positive control")
    if require_complete and status != "complete":
        raise FeatureCacheError("cache is explicitly partial")
    return CacheValidation(cache_root, identity, progress, live, quotient)


def expected_cache_bytes(identity: Mapping[str, Any]) -> int:
    """Return array payload bytes plus a conservative atomic/metadata margin."""

    _validate_identity(identity)
    arrays = identity["arrays"]
    payload = sum(math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize for spec in arrays.values())
    return int(payload + (64 << 20))


__all__ = [
    "APPROVED_SSD_WATERFALL",
    "CACHE_SCHEMA",
    "CERTIFICATION_NAME",
    "COMPLETION_CONTROL_NAME",
    "LIVE_LOGITS_NAME",
    "POSITIVE_CONTROL_SCHEMA",
    "PROGRESS_NAME",
    "QUOTIENT_FEATURES_NAME",
    "STAGING_SCRATCH_NAME",
    "BoundFileSnapshot",
    "FeatureCacheError",
    "PositiveControlResult",
    "SegnetHeadFeatureCache",
    "atomic_json",
    "atomic_prepared_path",
    "build_immutable_identity",
    "canonical_json_bytes",
    "expected_cache_bytes",
    "is_retained_name",
    "move_bound_file_noreplace",
    "move_path_noreplace",
    "open_gt_f1_stored_memmap",
    "retain_bound_file",
    "retained_original_name",
    "sha256_file",
    "source_file_row",
    "validate_atomic_transaction_custody",
    "validate_feature_cache",
    "validate_live_logit_positive_control",
    "validate_retained_file",
]
