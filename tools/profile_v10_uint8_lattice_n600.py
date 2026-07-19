#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable real-cache v10 uint8-lattice profiler.

The default all-n600 path verifies the original uint8 blocks as exact source
witnesses and derives vectorized cardinality bounds without Python DFS.  Exact
or bounded candidate enumeration remains explicit ``enumerated_subset`` mode.
Every frame persists a canonical receipt in an identity-rooted hash chain;
``--max-frames`` is an explicitly labeled, hash-valid prefix control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import stat
import struct
import subprocess
import sys
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import brotli
import numpy as np
import torch

try:
    import tools.tool_bootstrap as tool_bootstrap_module
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import tool_bootstrap as tool_bootstrap_module
    from tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import tac.admission_guard as admission_guard_module  # noqa: E402
import tac.boundary_math.power_diagram_witness as stored_npz_module  # noqa: E402
import tac.governed_profile_admission as governed_profile_admission_module  # noqa: E402
import tac.optimization.uint8_lattice_feasibility as feasibility_module  # noqa: E402
import tac.optimization.uint8_lattice_profile as profile_module  # noqa: E402
import tac.witness_control.segnet_head_feature_cache as feature_cache_module  # noqa: E402
from tac.admission_guard import (  # noqa: E402
    ADMISSION_ENFORCE_ENV,
    BYPASS_OVERRIDE_ENV,
    assert_governed_admission,
)
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.governed_profile_admission import (  # noqa: E402
    GovernedAdmissionError,
    attest_safe_run_parent,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tac.optimization.uint8_lattice_profile import (  # noqa: E402
    BlockProfileResult,
    LatticeProfileError,
    NoOpPosePlugin,
    SignedResidualCostModel,
    StreamingProfileAggregator,
    build_rd_row,
    decode_candidate_stream,
    encode_candidate_stream,
    noncorner_positive_control,
    profile_cache_key,
    profile_integer_block,
    vectorized_source_witness_bounds,
)
from tac.witness_control.segnet_head_feature_cache import (  # noqa: E402
    ATOMIC_GENERATION_RE,
    ATOMIC_GENERATION_SUFFIX,
    ATOMIC_TRANSACTION_RE,
    ATOMIC_TRANSACTION_SUFFIX,
    BoundFileSnapshot,
    atomic_json,
    atomic_prepared_path,
    canonical_json_bytes,
    read_bound_file,
    sha256_file,
    source_file_row,
    validate_feature_cache,
)

EXPECTED_PAIRS: Final = 600
EXPECTED_CAMERA_HW: Final = (874, 1164)
EXPECTED_SEG_HW: Final = (384, 512)
EXPECTED_CLASSES: Final = 5
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
SACRED_RESULT_ROOT: Final = Path(
    "/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z"
)
PROGRESS_SCHEMA: Final = "v10_uint8_lattice_profile_progress.v4"
STAGE_RECEIPT_SCHEMA: Final = "v10_uint8_lattice_profile_frame_stage.v5"
PARTITION_CUSTODY_SCHEMA: Final = "v10_uint8_lattice_partition_custody.v1"
FINAL_RECEIPT_SCHEMA: Final = "v10_uint8_lattice_n600_profile_receipt.v3"
RECEIPT_AUTHORIZATION_SCHEMA: Final = "profiler_receipt_transition_authorization.v1"
TIMING_SCHEMA: Final = "v10_uint8_lattice_profile_timing.v1"
TIMING_SUMMARY_SCHEMA: Final = "v10_uint8_lattice_profile_timing_summary.v1"
OUTPUT_CERTIFICATION_SCHEMA: Final = "v10_uint8_lattice_profile_certification.v1"
STAGING_SCRATCH_SCHEMA: Final = "v10_uint8_lattice_profile_staging_scratch.v1"
STAGE_MAGIC: Final = b"V10U8ST5"
STREAM_MAGIC: Final = b"V10U8N600RD1"
IDENTITY_NAME: Final = "identity.json"
PROGRESS_NAME: Final = "progress.json"
RECEIPT_NAME: Final = "receipt.json"
RECEIPT_AUTHORIZATION_ROOT_NAME: Final = "receipt-authorizations"
OUTPUT_CERTIFICATION_NAME: Final = "certification.json"
STAGING_SCRATCH_NAME: Final = "staging_scratch.json"
CREATION_STAGING_SUFFIX: Final = ".v10-uint8-lattice-profile-staging"
CREATION_PREPARED_SUFFIX: Final = ".creation-prepared"
TIMING_CUSTODY_LABEL: Final = "MEASURED_PROCESS_OBSERVATION_STAGE_CHAIN_CUSTODY_NOT_SEMANTICALLY_REPLAYABLE"
BOUNDS_MODE: Final = "bounds_only_source_witness"
ENUMERATED_MODE: Final = "enumerated_subset"
LOWER_BOUND_METHOD: Final = "MAX_DISJOINT_PAIR_NULL_FIBER_PRODUCT"
COUNTER_NAMES: Final = (
    "selected_blocks",
    "total_blocks",
    "exhaustive_selected_blocks",
    "bounded_selected_blocks",
    "omitted_blocks",
    "segnet_mismatches",
    "segnet_pixels",
)
CANONICAL_CLASS_KEYS: Final = tuple(str(index) for index in range(EXPECTED_CLASSES))
CANONICAL_STRATA: Final = ("boundary_annulus", "degenerate", "fragile")
BUCKET_STATE_KEYS: Final = {
    "scorer_pixels",
    "channel_blocks",
    "exact_blocks",
    "bounded_blocks",
    "lower",
    "upper",
}
COMPACT_STATE_KEYS: Final = {
    "bin_width",
    "bins",
    "count",
    "zero_count",
    "total",
    "minimum",
    "maximum",
}
PREPARED_STAGE_SUFFIX: Final = ".prepared"
STAGE_INTENT_RE: Final = re.compile(r"^\.(frame_(\d{4})\.bin)\.intent-attempt-([0-9]{8})-([0-9]+)-([0-9a-f]{64})$")
RECOVERY_ROOT_NAME: Final = "recovery"
RECOVERY_MANIFEST_NAME: Final = "manifest.json"
RECOVERY_PAYLOAD_NAME: Final = "interrupted-stage.bin"
STAGE_SUCCESS_NAME: Final = "success.json"
STAGE_ATTEMPT_NAME: Final = "attempt.json"
STAGE_ATTEMPT_TRANSACTION_SCHEMA: Final = "profiler_stage_attempt_transaction.v1"
STAGE_ATTEMPT_SUCCESS_SCHEMA: Final = "profiler_stage_attempt_success.v1"
RECOVERY_MANIFEST_SCHEMA: Final = "v10_uint8_lattice_stage_recovery.v2"
RECOVERY_TRANSACTION_RE: Final = re.compile(r"^frame_([0-9]{4})-attempt_([0-9]{8})$")
STAGE_ATTEMPT_FALSE_AUTHORITY_FLAGS: Final = {
    "stage_chain_member": False,
    "rate_stream_member": False,
    "score_authority": False,
    "promotion_eligible": False,
}
RECEIPT_AUTHORIZATION_FALSE_AUTHORITY_FLAGS: Final = {
    "stage_chain_member": False,
    "rate_stream_member": False,
    "score_authority": False,
    "promotion_eligible": False,
}
RECEIPT_AUTHORIZATION_KEYS: Final = {
    "schema",
    "profile_identity_sha256",
    "exact_rebuild_argv",
    "validated_stage_chain_head_sha256",
    "validated_frame_count",
    "semantic_validation_sha256",
    "target_basename",
    "prior",
    "desired_receipt",
    "false_authority_flags",
}
STAGE_ATTEMPT_TRANSACTION_KEYS: Final = {
    "schema",
    "identity_sha256",
    "frame",
    "attempt",
    "final_basename",
    "prepared_basename",
    "intent_basename",
    "intended_bytes",
    "intended_sha256",
    "exact_rebuild_argv",
    "false_authority_flags",
}
STAGE_ATTEMPT_SUCCESS_KEYS: Final = {
    "schema",
    "identity_sha256",
    "frame",
    "attempt",
    "stage_attempt_transaction_sha256",
    "final_basename",
    "final_bytes",
    "final_sha256",
    "exact_rebuild_argv",
    "false_authority_flags",
}
PROGRESS_KEYS: Final = {
    "schema",
    "identity_sha256",
    "status",
    "next_frame",
    "stage_chain_head_sha256",
    "storage_preflight",
    "exact_argv",
}
PARTITION_MASK_KEYS: Final = {"dtype", "shape", "sha256", "true_count"}
SELECTION_CUSTODY_KEYS: Final = {
    "selection_label",
    "selection_globally_exact",
    "exact_count_claim",
    "min_description_claim",
    "seed_source_witness",
    "receiver_non_closure",
    "pose_bank_wired",
    "factor10_solved",
    "scope_extrapolation",
    "per_block_selector_minimum_proved",
    "global_compressed_stream_minimum_claim",
}
SELECTION_CUSTODY_BOOL_KEYS: Final = SELECTION_CUSTODY_KEYS - {
    "selection_label",
    "scope_extrapolation",
}
TIMING_KEYS: Final = {
    "schema",
    "wall_seconds",
    "blocks_per_second",
    "peak_rss_bytes",
    "custody_label",
}
FINAL_RECEIPT_KEYS: Final = {
    "schema",
    "status",
    "scope_label",
    "mode",
    "lower_bound_method",
    "derivation",
    "frames_profiled",
    "expected_frames",
    "profiled_frame_indices",
    "scope_extrapolation",
    "aggregate",
    "rd_row",
    "counters_rebuilt_from_hashed_stage_receipts",
    "feature_cache_binding",
    "claims",
    "positive_control",
    "authority",
    "identity_sha256",
    "git_head",
    "exact_rebuild_argv",
    "requested_outer_governor_limits",
    "custody",
    "timing_summary",
}
TERMINAL_CUSTODY_KEYS: Final = {
    "identity_sha256",
    "identity_json_sha256",
    "identity_json_bytes",
    "output_certification_sha256",
    "exact_rebuild_argv",
    "ordered_stage_count",
    "terminal_stage_chain_sha256",
    "progress_pointer",
    "progress_pointer_sha256",
    "stream_accounting",
    "recovery_transactions",
}
STORAGE_PREFLIGHT_KEYS: Final = {
    "waterfall_order",
    "existing_approved_roots",
    "selection_scope",
    "selected_root",
    "filesystem_anchor",
    "free_bytes_before",
    "required_free_bytes",
    "allow_local_output_for_tests",
    "PASS",
}
CREATION_STORAGE_IDENTITY_SCHEMA: Final = "v10_uint8_lattice_creation_storage_identity.v1"
CREATION_STORAGE_IDENTITY_KEYS: Final = {
    "schema",
    "stable_preflight",
    "volatile_fields_excluded",
}
VOLATILE_STORAGE_PREFLIGHT_KEYS: Final = frozenset(
    {
        "filesystem_anchor",
        "free_bytes_before",
    }
)
STABLE_STORAGE_PREFLIGHT_KEYS: Final = STORAGE_PREFLIGHT_KEYS - VOLATILE_STORAGE_PREFLIGHT_KEYS
FRESH_STORAGE_SELECTION_SCOPE: Final = "FRESH_FIRST_EXISTING_TIER"
RESUME_STORAGE_SELECTION_SCOPE: Final = "RESUME_EXISTING_SACRED_TIER"
LOWER_SHA256_CHARS: Final = frozenset("0123456789abcdef")
ZLIB_STREAM_SETTINGS: Final = {
    "level": 9,
    "method": zlib.DEFLATED,
    "wbits": zlib.MAX_WBITS,
    "mem_level": zlib.DEF_MEM_LEVEL,
    "strategy": zlib.Z_DEFAULT_STRATEGY,
}
BROTLI_STREAM_SETTINGS: Final = {
    "mode": brotli.MODE_GENERIC,
    "quality": 11,
    "lgwin": 22,
    "lgblock": 0,
}
OUTSIDE_SUPPORT_FILL_VALUE: Final = 0
OUTSIDE_SUPPORT_FILL_LABEL: Final = "CONSTANT_UINT8_ZERO_OUTSIDE_EXACT_RESIZE_SUPPORT_UNION"
ORDER0_ESTIMATE_LABEL: Final = "ORDER0_IID_PLUGIN_IDEAL_LENGTH_ESTIMATE_NOT_UNIVERSAL_LOWER_BOUND"
CODEC_BYTE_COUNT_LABEL: Final = "DETERMINISTIC_ENCODER_BYTE_COUNT_NOT_GLOBAL_COMPRESSED_STREAM_MINIMUM"


class ProfilerError(RuntimeError):
    """Fail-closed cache, storage, resume, scorer, or stream error."""


@dataclass(frozen=True)
class _FrozenScorerSnapshot:
    rows: dict[str, dict[str, Any]]
    segnet_payload: bytes


class _MalformedCanonicalJSONError(ProfilerError):
    """Positively identified malformed content, never a custody-read error."""

    parsed_value: dict[str, Any] | None = None
    file_snapshot: BoundFileSnapshot | None = None


@dataclass(frozen=True)
class _FrameSemanticReplay:
    """Scientific stage state independently derived from immutable inputs."""

    partition_custody: dict[str, Any]
    aggregate_state: dict[str, Any]
    candidate_payload: bytes
    counters: dict[str, int]
    selection_custody: dict[str, Any]


@dataclass(frozen=True)
class _ValidatedProgressSnapshot(Mapping[str, Any]):
    """Immutable canonical pointer bytes plus the validated path identity."""

    canonical_payload: bytes
    file_identity: tuple[int, int, int, int, int]

    def _value(self) -> dict[str, Any]:
        value = json.loads(self.canonical_payload)
        assert isinstance(value, dict)
        return value

    def __getitem__(self, key: str) -> Any:
        return self._value()[key]

    def __iter__(self):
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())


@dataclass(frozen=True)
class _ValidatedStageAttemptCustody:
    """Role-proven stage retention and complete immutable attempt outcomes."""

    outcomes: tuple[dict[str, Any], ...]
    proven_stage_retained_names: frozenset[str]
    intent_building: tuple[tuple[int, int], ...] = ()


def _stat_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return (device, inode, size, mtime_ns, link_count) for bound custody."""

    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_nlink)


@dataclass(frozen=True)
class _DecodedFramePayload:
    """Strict receiver parse-back of one frame's wrapped row streams."""

    selected_frame: np.ndarray | None
    selected_blocks: int
    total_blocks: int
    receiver_closed: bool


@dataclass(frozen=True)
class _FrameProfileArtifacts:
    """One deterministic frame computation shared by writing and replay."""

    replay: _FrameSemanticReplay
    aggregate: StreamingProfileAggregator
    counters: dict[str, int]
    candidate_payload: bytes
    lower_bound_method: str | None
    labels: np.ndarray
    selected_frame: np.ndarray | None
    decoded_frame: np.ndarray | None
    receiver_closed: bool


def _assert_real_governed_admission(
    *,
    allow_local_output_for_tests: bool,
    env: dict[str, str] | None = None,
) -> None:
    """Require the real child marker; local-output policy never bypasses it."""

    if type(allow_local_output_for_tests) is not bool:
        raise ProfilerError("allow_local_output_for_tests must be boolean")
    strict_env = dict(os.environ if env is None else env)
    strict_env[ADMISSION_ENFORCE_ENV] = "1"
    strict_env.pop(BYPASS_OVERRIDE_ENV, None)
    try:
        assert_governed_admission(
            "profile_v10_uint8_lattice_n600",
            env=strict_env,
            on_refuse="raise",
        )
    except PermissionError as exc:
        raise ProfilerError("production profiling requires the real governed-admission child marker") from exc


def _assert_local_test_scope(*, allow_local_output_for_tests: bool, max_frames: int) -> None:
    """Keep the public local-output switch confined to a real tiny pytest prefix."""

    if not allow_local_output_for_tests:
        return
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    # ``python -m pytest`` legitimately exposes ``__main__.py`` as argv[0].
    # The live per-test marker plus the imported pytest module identifies the
    # active harness; direct safe-run parent custody is enforced separately.
    if "pytest" not in sys.modules or not current_test:
        raise ProfilerError("local profiler output is permitted only inside an actual pytest process")
    if max_frames > 2:
        raise ProfilerError("local pytest output is limited to a tiny prefix and can never complete n600")


def _git_head() -> str:
    """Return the exact source-freeze commit without asserting tree cleanliness."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProfilerError("git HEAD custody is unavailable") from exc
    head = completed.stdout.strip()
    if len(head) not in (40, 64) or any(character not in LOWER_SHA256_CHARS for character in head):
        raise ProfilerError("git HEAD must be a lowercase Git object id")
    return head


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_exact_module_file(module: Any, expected: Path, *, role: str) -> Path:
    loaded_name = getattr(module, "__file__", None)
    if not isinstance(loaded_name, str):
        raise ProfilerError(f"executed {role} module has no source path")
    loaded = Path(loaded_name).resolve(strict=True)
    exact = expected.resolve(strict=True)
    if loaded != exact:
        raise ProfilerError(f"executed {role} path {loaded} != bound source {exact}")
    return loaded


def _resolve_without_symlink_components(path: Path, *, name: str) -> Path:
    """Return an absolute path only after rejecting every existing symlink component."""

    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    absolute = Path(os.path.abspath(absolute))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ProfilerError(f"{name} path custody is unavailable: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProfilerError(f"{name} may not traverse a symlink: {cursor}")
        if cursor != absolute and not stat.S_ISDIR(metadata.st_mode):
            raise ProfilerError(f"{name} traverses a non-directory: {cursor}")
    return absolute.resolve()


def _existing_approved_ssd_roots() -> list[Path]:
    existing: list[Path] = []
    for root in SSD_ROOTS:
        if not os.path.lexists(root):
            continue
        resolved = _resolve_without_symlink_components(root, name="approved SSD root")
        try:
            metadata = root.lstat()
        except OSError as exc:
            raise ProfilerError(f"approved SSD root custody is unavailable: {root}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ProfilerError(f"approved SSD root is not a non-symlink directory: {root}")
        existing.append(resolved)
    return existing


def _safe_output_root(
    path: Path,
    *,
    allow_local_output_for_tests: bool,
    resume: bool = False,
) -> Path:
    resolved = _resolve_without_symlink_components(path, name="profiler output")
    if _is_relative_to(resolved, SACRED_RESULT_ROOT.resolve()):
        raise ProfilerError("profiler output may not address the sacred result tree")
    existing_roots = _existing_approved_ssd_roots()
    if not allow_local_output_for_tests:
        if not existing_roots:
            raise ProfilerError("no approved SSD root exists for the profiler output")
        selected_roots = existing_roots if resume else existing_roots[:1]
        if not any(_is_relative_to(resolved, root) for root in selected_roots):
            requirement = "an existing sacred SSD tier" if resume else f"first existing SSD root {existing_roots[0]}"
            raise ProfilerError(f"profiler output must use {requirement}; lower tiers are refused for fresh creation")
    return resolved


def _storage_preflight(
    root: Path,
    *,
    max_frames: int,
    allow_local_output_for_tests: bool,
    resume: bool = False,
) -> dict[str, Any]:
    existing_roots = _existing_approved_ssd_roots()
    if not allow_local_output_for_tests:
        selected_roots = existing_roots if resume else existing_roots[:1]
        if not selected_roots or not any(_is_relative_to(root, tier) for tier in selected_roots):
            raise ProfilerError("storage preflight requires the selected approved SSD tier")
    anchor = root
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free
    # One raw selected camera frame per stage plus atomic-write and receipt margin.
    required = max_frames * (np.prod(EXPECTED_CAMERA_HW) * 3 + (4 << 20)) + (64 << 20)
    if free < required:
        raise ProfilerError(f"storage preflight refused: free={free} < required={required}")
    return {
        "waterfall_order": [str(path) for path in SSD_ROOTS],
        "existing_approved_roots": [str(path) for path in existing_roots],
        "selection_scope": (RESUME_STORAGE_SELECTION_SCOPE if resume else FRESH_STORAGE_SELECTION_SCOPE),
        "selected_root": str(root),
        "filesystem_anchor": str(anchor),
        "free_bytes_before": int(free),
        "required_free_bytes": int(required),
        "allow_local_output_for_tests": allow_local_output_for_tests,
        "PASS": True,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_local_regular_file(path: Path, *, name: str) -> os.stat_result:
    """Reject symlinked, hard-linked, or non-regular custody bytes."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProfilerError(f"{name} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProfilerError(f"{name} must be one local regular file with link count one")
    return metadata


def _require_local_directory(path: Path, *, name: str) -> os.stat_result:
    """Reject symlinked/non-directory custody roots without following them."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProfilerError(f"{name} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProfilerError(f"{name} must be one local directory")
    return metadata


def _load_canonical_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        snapshot = read_bound_file(path, role=name)
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError(f"{name} custody read failed") from exc
    raw = snapshot.payload
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        error = _MalformedCanonicalJSONError(f"{name} is malformed canonical JSON")
        error.file_snapshot = snapshot
        raise error from exc
    if not isinstance(value, dict):
        error = _MalformedCanonicalJSONError(f"{name} is not a canonical JSON object")
        error.file_snapshot = snapshot
        raise error
    try:
        canonical = canonical_json_bytes(value)
    except feature_cache_module.FeatureCacheError as exc:
        error = _MalformedCanonicalJSONError(f"{name} is not finite canonical JSON")
        error.file_snapshot = snapshot
        raise error from exc
    if raw != canonical + b"\n":
        error = _MalformedCanonicalJSONError(f"{name} is not canonical JSON")
        error.parsed_value = value
        error.file_snapshot = snapshot
        raise error
    return value


def _read_bound_bytes(path: Path, *, name: str) -> BoundFileSnapshot:
    try:
        return read_bound_file(path, role=name)
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError(f"{name} custody read failed") from exc


def _assert_snapshot_names_path(path: Path, snapshot: BoundFileSnapshot, *, name: str) -> None:
    current = _require_local_regular_file(path, name=name)
    if _stat_file_identity(current) != snapshot.file_identity:
        raise ProfilerError(f"{name} path identity changed; preserving bytes")


def _unlink_bound(path: Path, snapshot: BoundFileSnapshot, *, name: str) -> None:
    """Retire exact custody without a destructive pathname unlink."""

    try:
        feature_cache_module.retain_bound_file(path, snapshot, role=name)
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError(f"{name} retention failed; preserving bytes") from exc


def _replace_bound(source: Path, destination: Path, snapshot: BoundFileSnapshot, *, name: str) -> None:
    """Move exact custody to an absent destination without overwrite semantics."""

    try:
        feature_cache_module.move_bound_file_noreplace(
            source,
            destination,
            snapshot,
            role=name,
        )
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError(f"{name} no-replace move failed; preserving bytes") from exc


def _move_directory_noreplace(source: Path, destination: Path, *, name: str) -> None:
    """Move a stable local directory without replacing late-arriving custody."""

    metadata = _require_local_directory(source, name=name)
    try:
        feature_cache_module.move_path_noreplace(
            source,
            destination,
            expected_identity=_stat_file_identity(metadata),
            role=name,
            require_directory=True,
        )
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError(f"{name} no-replace move failed; preserving bytes") from exc


def _is_atomic_scratch_original(name: str, *, targets: Iterable[str]) -> bool:
    for target in targets:
        if name == atomic_prepared_path(Path(target)).name:
            return True
        if (
            name.startswith(f".{target}{ATOMIC_GENERATION_SUFFIX}-")
            and ATOMIC_GENERATION_RE.fullmatch(name) is not None
        ):
            return True
        if (
            name.startswith(f".{target}{ATOMIC_TRANSACTION_SUFFIX}-")
            and ATOMIC_TRANSACTION_RE.fullmatch(name) is not None
        ):
            return True
        if name.startswith(f".{target}.atomic-completion-") and (
            feature_cache_module.ATOMIC_COMPLETION_RE.fullmatch(name) is not None
            or feature_cache_module.ATOMIC_COMPLETION_GENERATION_RE.fullmatch(name) is not None
        ):
            return True
    return False


def _is_creation_prepared_original(name: str) -> bool:
    return name in {
        _creation_prepared_path(Path(target)).name
        for target in (STAGING_SCRATCH_NAME, IDENTITY_NAME, PROGRESS_NAME, OUTPUT_CERTIFICATION_NAME)
    }


def _is_output_retained_original(name: str) -> bool:
    return _is_creation_prepared_original(name) or _is_atomic_scratch_original(
        name,
        targets=(PROGRESS_NAME, RECEIPT_NAME),
    )


def _is_stage_retained_original(name: str) -> bool:
    return (
        STAGE_INTENT_RE.fullmatch(name) is not None
        or re.fullmatch(r"\.frame_[0-9]{4}\.bin\.prepared", name) is not None
    )


def _is_recovery_retained_original(name: str) -> bool:
    return _is_atomic_scratch_original(name, targets=(RECOVERY_MANIFEST_NAME,))


def _validated_retained_names(
    paths: Iterable[Path],
    *,
    name: str,
    allowed_original: Callable[[str], bool],
) -> set[str]:
    """Integrity-check retained names without granting grammar authority."""

    result: set[str] = set()
    for path in paths:
        if not feature_cache_module.is_retained_name(path.name):
            continue
        try:
            original = feature_cache_module.retained_original_name(path.name)
            if not allowed_original(original):
                raise feature_cache_module.FeatureCacheError(
                    f"{name} retained custody names an impossible original: {original}"
                )
            feature_cache_module.validate_retained_file(path, role=f"{name} {path.name}")
        except feature_cache_module.FeatureCacheError as exc:
            raise ProfilerError(f"{name} retained custody is malformed; preserving bytes") from exc
        result.add(path.name)
    return result


def _validate_atomic_retained_names(
    path: Path,
    *,
    desired_payload: bytes,
    expected_prior_payloads: Iterable[bytes] = (),
    expected_consumer_authorization_sha256: str | None = None,
    name: str,
) -> set[str]:
    """Return only retained names proven by the generic atomic transaction chain."""

    try:
        return feature_cache_module.validate_atomic_transaction_custody(
            path,
            desired_payload=desired_payload,
            expected_prior_payloads=tuple(expected_prior_payloads),
            expected_consumer_authorization_sha256=expected_consumer_authorization_sha256,
        )
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError(f"{name} atomic transaction custody is malformed; preserving bytes") from exc


def _validate_creation_retained_names(
    paths: Iterable[Path],
    *,
    expected_payloads_by_original: Mapping[str, bytes],
    name: str,
) -> set[str]:
    """Prove retained creation scratch against its deterministic certified payload."""

    expected = dict(expected_payloads_by_original)
    if any(
        not isinstance(original, str)
        or not _is_creation_prepared_original(original)
        or type(payload) is not bytes
        or not payload
        for original, payload in expected.items()
    ):
        raise ProfilerError(f"{name} creation-retention expectation is malformed")
    proven: set[str] = set()
    for path in paths:
        if not feature_cache_module.is_retained_name(path.name):
            continue
        try:
            original = feature_cache_module.retained_original_name(path.name)
            if not _is_creation_prepared_original(original):
                continue
            snapshot = feature_cache_module.validate_retained_file(
                path,
                role=f"{name} retained creation {path.name}",
            )
        except feature_cache_module.FeatureCacheError as exc:
            raise ProfilerError(f"{name} retained creation custody is malformed; preserving bytes") from exc
        target_payload = expected.get(original)
        if target_payload is None or not (
            snapshot.payload == target_payload
            or (len(snapshot.payload) < len(target_payload) and target_payload.startswith(snapshot.payload))
        ):
            raise ProfilerError(f"{name} retained creation payload lacks certified target/prefix provenance")
        proven.add(path.name)
    return proven


def _active_atomic_scratch(path: Path) -> list[Path]:
    prepared = atomic_prepared_path(path)
    result = [prepared] if os.path.lexists(prepared) else []
    result.extend(
        sorted(
            entry
            for entry in path.parent.iterdir()
            if entry.name.startswith(f".{path.name}{ATOMIC_GENERATION_SUFFIX}-")
            and ATOMIC_GENERATION_RE.fullmatch(entry.name) is not None
        )
    )
    result.extend(
        sorted(
            entry
            for entry in path.parent.iterdir()
            if entry.name.startswith(f".{path.name}{ATOMIC_TRANSACTION_SUFFIX}-")
            and ATOMIC_TRANSACTION_RE.fullmatch(entry.name) is not None
        )
    )
    return result


def _active_atomic_completion(path: Path) -> list[Path]:
    entries = {entry.name: entry for entry in path.parent.iterdir()}
    names = feature_cache_module.active_atomic_completion_names(
        entries,
        target_names=(path.name,),
    )
    return [entries[name] for name in sorted(names)]


def _reconcile_committed_atomic_json(
    path: Path,
    *,
    name: str,
    expected_prior_payloads: Iterable[bytes],
    expected_consumer_authorization_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Converge target-present post-exchange scratch before strict layout checks."""

    scratch = _active_atomic_scratch(path)
    if not os.path.lexists(path):
        return None
    value = _load_canonical_object(path, name=name)
    if scratch:
        try:
            atomic_json(
                path,
                value,
                expected_prior_payloads=tuple(expected_prior_payloads),
                consumer_authorization_sha256=expected_consumer_authorization_sha256,
            )
        except feature_cache_module.FeatureCacheError as exc:
            raise ProfilerError(f"{name} atomic transaction custody is malformed; preserving bytes") from exc
    return value


def _receipt_semantic_validation_sha256(
    *,
    desired_receipt_payload: bytes,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    terminal_stage_chain_sha256: str,
    frame_count: int,
) -> str:
    if type(desired_receipt_payload) is not bytes or not desired_receipt_payload:
        raise ProfilerError("receipt semantic validation requires complete desired bytes")
    if type(frame_count) is not int or not 0 <= frame_count <= EXPECTED_PAIRS:
        raise ProfilerError("receipt semantic validation frame count is malformed")
    value = {
        "schema": "profiler_receipt_semantic_validation.v1",
        "profile_identity_sha256": _canonical_sha256(identity_sha256, name="receipt semantic identity"),
        "exact_rebuild_argv": _normalized_argv(exact_rebuild_argv, name="receipt semantic argv"),
        "terminal_stage_chain_sha256": _canonical_sha256(
            terminal_stage_chain_sha256,
            name="receipt semantic stage-chain head",
        ),
        "validated_frame_count": frame_count,
        "desired_receipt_bytes": len(desired_receipt_payload),
        "desired_receipt_sha256": _sha256_bytes(desired_receipt_payload),
        "target_basename": RECEIPT_NAME,
    }
    return _sha256_bytes(canonical_json_bytes(value) + b"\n")


def _receipt_transition_authorization(
    *,
    desired_receipt: Mapping[str, Any],
    prior_snapshot: BoundFileSnapshot | None,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    terminal_stage_chain_sha256: str,
    frame_count: int,
) -> dict[str, Any]:
    desired_payload = canonical_json_bytes(dict(desired_receipt)) + b"\n"
    prior = (
        {
            "state": "ABSENT",
            "payload_hex": None,
            "bytes": 0,
            "sha256": None,
            "file_identity": None,
        }
        if prior_snapshot is None
        else {
            "state": "PRESENT",
            "payload_hex": prior_snapshot.payload.hex(),
            "bytes": len(prior_snapshot.payload),
            "sha256": _sha256_bytes(prior_snapshot.payload),
            "file_identity": list(prior_snapshot.file_identity),
        }
    )
    return {
        "schema": RECEIPT_AUTHORIZATION_SCHEMA,
        "profile_identity_sha256": _canonical_sha256(identity_sha256, name="receipt authorization identity"),
        "exact_rebuild_argv": _normalized_argv(exact_rebuild_argv, name="receipt authorization argv"),
        "validated_stage_chain_head_sha256": _canonical_sha256(
            terminal_stage_chain_sha256,
            name="receipt authorization stage-chain head",
        ),
        "validated_frame_count": frame_count,
        "semantic_validation_sha256": _receipt_semantic_validation_sha256(
            desired_receipt_payload=desired_payload,
            identity_sha256=identity_sha256,
            exact_rebuild_argv=exact_rebuild_argv,
            terminal_stage_chain_sha256=terminal_stage_chain_sha256,
            frame_count=frame_count,
        ),
        "target_basename": RECEIPT_NAME,
        "prior": prior,
        "desired_receipt": {
            "bytes": len(desired_payload),
            "sha256": _sha256_bytes(desired_payload),
        },
        "false_authority_flags": dict(RECEIPT_AUTHORIZATION_FALSE_AUTHORITY_FLAGS),
    }


def _validate_receipt_transition_authorization(value: object) -> tuple[dict[str, Any], bytes | None]:
    if (
        not isinstance(value, dict)
        or set(value) != RECEIPT_AUTHORIZATION_KEYS
        or value.get("schema") != RECEIPT_AUTHORIZATION_SCHEMA
        or value.get("target_basename") != RECEIPT_NAME
        or value.get("false_authority_flags") != RECEIPT_AUTHORIZATION_FALSE_AUTHORITY_FLAGS
    ):
        raise ProfilerError("receipt transition authorization schema is malformed")
    identity_sha256 = _canonical_sha256(value.get("profile_identity_sha256"), name="receipt authorization identity")
    argv = _normalized_argv(value.get("exact_rebuild_argv", ()), name="receipt authorization argv")
    stage_head = _canonical_sha256(
        value.get("validated_stage_chain_head_sha256"),
        name="receipt authorization stage-chain head",
    )
    frame_count = value.get("validated_frame_count")
    if type(frame_count) is not int or not 0 <= frame_count <= EXPECTED_PAIRS:
        raise ProfilerError("receipt transition authorization frame count is malformed")
    semantic_digest = _canonical_sha256(
        value.get("semantic_validation_sha256"),
        name="receipt authorization semantic digest",
    )
    desired = value.get("desired_receipt")
    if (
        not isinstance(desired, dict)
        or set(desired) != {"bytes", "sha256"}
        or type(desired.get("bytes")) is not int
        or desired["bytes"] <= 0
    ):
        raise ProfilerError("receipt transition authorization desired custody is malformed")
    desired_sha256 = _canonical_sha256(desired.get("sha256"), name="authorized desired receipt")
    prior = value.get("prior")
    if not isinstance(prior, dict) or set(prior) != {
        "state",
        "payload_hex",
        "bytes",
        "sha256",
        "file_identity",
    }:
        raise ProfilerError("receipt transition authorization prior custody is malformed")
    prior_payload: bytes | None
    if prior.get("state") == "ABSENT":
        if (
            any(prior.get(key) is not None for key in ("payload_hex", "sha256", "file_identity"))
            or prior.get("bytes") != 0
        ):
            raise ProfilerError("ABSENT receipt authorization claims prior bytes")
        prior_payload = None
    elif prior.get("state") == "PRESENT":
        encoded = prior.get("payload_hex")
        identity = prior.get("file_identity")
        if (
            not isinstance(encoded, str)
            or type(prior.get("bytes")) is not int
            or prior["bytes"] <= 0
            or not isinstance(identity, list)
            or len(identity) != 5
            or any(type(item) is not int or item < 0 for item in identity)
        ):
            raise ProfilerError("PRESENT receipt authorization prior custody is malformed")
        try:
            prior_payload = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ProfilerError("receipt authorization prior encoding is malformed") from exc
        if (
            len(prior_payload) != prior["bytes"]
            or _sha256_bytes(prior_payload) != _canonical_sha256(prior.get("sha256"), name="authorized prior receipt")
            or identity[2] != len(prior_payload)
            or identity[4] != 1
        ):
            raise ProfilerError("receipt authorization prior bytes/identity disagree")
    else:
        raise ProfilerError("receipt transition authorization prior state is malformed")
    normalized = json.loads(canonical_json_bytes(value))
    normalized["profile_identity_sha256"] = identity_sha256
    normalized["exact_rebuild_argv"] = argv
    normalized["validated_stage_chain_head_sha256"] = stage_head
    normalized["semantic_validation_sha256"] = semantic_digest
    normalized["desired_receipt"]["sha256"] = desired_sha256
    return normalized, prior_payload


def _receipt_authorization_root(output_root: Path) -> Path:
    return output_root / RECEIPT_AUTHORIZATION_ROOT_NAME


def _receipt_authorization_path(output_root: Path, authorization: Mapping[str, Any]) -> Path:
    payload = canonical_json_bytes(dict(authorization)) + b"\n"
    return _receipt_authorization_root(output_root) / f"{_sha256_bytes(payload)}.json"


def _load_receipt_authorizations(output_root: Path) -> list[tuple[dict[str, Any], str, bytes | None]]:
    root = _receipt_authorization_root(output_root)
    if not os.path.lexists(root):
        return []
    _require_local_directory(root, name="receipt authorization root")
    entries = {entry.name: entry for entry in root.iterdir()}
    target_re = re.compile(r"^(?P<digest>[0-9a-f]{64})\.json$")
    role_re = re.compile(r"^\.(?P<target>[0-9a-f]{64}\.json)\.(?:prepared|atomic-.+)$")
    targets: set[str] = set()
    for name, _path in entries.items():
        original = (
            feature_cache_module.retained_original_name(name) if feature_cache_module.is_retained_name(name) else name
        )
        match = target_re.fullmatch(original)
        if match is not None:
            targets.add(original)
            continue
        role = role_re.fullmatch(original)
        if role is None or not _is_atomic_scratch_original(original, targets=(role.group("target"),)):
            raise ProfilerError("receipt authorization root contains unidentified custody")
        targets.add(role.group("target"))

    records: list[tuple[dict[str, Any], str, bytes | None]] = []
    building = 0
    for target_name in sorted(targets):
        path = root / target_name
        related = [
            entry
            for entry in entries.values()
            if (
                entry.name == target_name
                or (
                    feature_cache_module.is_retained_name(entry.name)
                    and _is_atomic_scratch_original(
                        feature_cache_module.retained_original_name(entry.name),
                        targets=(target_name,),
                    )
                )
                or _is_atomic_scratch_original(entry.name, targets=(target_name,))
            )
        ]
        if not os.path.lexists(path):
            building += 1
            if building > 1 or any(
                feature_cache_module.is_retained_name(entry.name)
                or feature_cache_module.ATOMIC_COMPLETION_RE.fullmatch(entry.name) is not None
                or feature_cache_module.ATOMIC_COMPLETION_GENERATION_RE.fullmatch(entry.name) is not None
                for entry in related
            ):
                raise ProfilerError("receipt authorization construction is orphaned or duplicated")
            for entry in related:
                _read_bound_bytes(entry, name="receipt authorization construction")
            continue
        value = _load_canonical_object(path, name="receipt transition authorization")
        normalized, prior_payload = _validate_receipt_transition_authorization(value)
        payload = canonical_json_bytes(normalized) + b"\n"
        digest = _sha256_bytes(payload)
        if target_name != f"{digest}.json":
            raise ProfilerError("receipt transition authorization pathname disagrees with its digest")
        try:
            feature_cache_module.validate_atomic_transaction_custody(
                path,
                desired_payload=payload,
            )
        except feature_cache_module.FeatureCacheError as exc:
            raise ProfilerError("receipt transition authorization atomic custody is malformed") from exc
        records.append((normalized, digest, prior_payload))
    return records


def _resolve_receipt_transition_authorization(
    output_root: Path,
    *,
    desired_receipt_payload: bytes,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    terminal_stage_chain_sha256: str,
    frame_count: int,
) -> tuple[tuple[bytes, ...], str | None]:
    records = _load_receipt_authorizations(output_root)
    receipt_path = output_root / RECEIPT_NAME
    atomic_evidence = bool(_active_atomic_scratch(receipt_path) or _active_atomic_completion(receipt_path))
    if not atomic_evidence:
        return (), None
    try:
        desired_value = json.loads(desired_receipt_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfilerError("receipt authorization target is malformed canonical JSON") from exc
    declared_custody = desired_value.get("custody") if isinstance(desired_value, dict) else None
    if not isinstance(declared_custody, dict):
        raise ProfilerError("receipt authorization target lacks declared terminal custody")
    authorized_stage_head = _canonical_sha256(
        declared_custody.get("terminal_stage_chain_sha256"),
        name="receipt authorization target stage-chain head",
    )
    authorized_frame_count = declared_custody.get("ordered_stage_count")
    if type(authorized_frame_count) is not int or not 0 <= authorized_frame_count <= EXPECTED_PAIRS:
        raise ProfilerError("receipt authorization target frame count is malformed")
    _canonical_sha256(terminal_stage_chain_sha256, name="observed receipt stage-chain head")
    if type(frame_count) is not int or not 0 <= frame_count <= EXPECTED_PAIRS:
        raise ProfilerError("observed receipt frame count is malformed")
    desired_sha256 = _sha256_bytes(desired_receipt_payload)
    semantic_sha256 = _receipt_semantic_validation_sha256(
        desired_receipt_payload=desired_receipt_payload,
        identity_sha256=identity_sha256,
        exact_rebuild_argv=exact_rebuild_argv,
        terminal_stage_chain_sha256=authorized_stage_head,
        frame_count=authorized_frame_count,
    )
    rebuild_argv = _normalized_argv(exact_rebuild_argv, name="receipt authorization resolution argv")
    matches = [
        (record, digest, prior)
        for record, digest, prior in records
        if record["profile_identity_sha256"] == identity_sha256
        and record["exact_rebuild_argv"] == rebuild_argv
        and record["validated_stage_chain_head_sha256"] == authorized_stage_head
        and record["validated_frame_count"] == authorized_frame_count
        and record["semantic_validation_sha256"] == semantic_sha256
        and record["desired_receipt"] == {"bytes": len(desired_receipt_payload), "sha256": desired_sha256}
    ]
    if len(matches) != 1:
        raise ProfilerError("receipt atomic custody lacks one exact semantic prior authorization")
    _record, digest, prior = matches[0]
    return (() if prior is None else (prior,)), digest


def _persist_receipt_transition_authorization(
    output_root: Path,
    *,
    authorization: Mapping[str, Any],
    authorize_mutation: Callable[[], None],
) -> tuple[Path, str]:
    normalized, _prior = _validate_receipt_transition_authorization(dict(authorization))
    path = _receipt_authorization_path(output_root, normalized)
    digest = path.stem
    root = path.parent
    if os.path.lexists(root):
        _require_local_directory(root, name="receipt authorization root")
    else:
        authorize_mutation()
        root.mkdir()
        _fsync_stage_directory(root.parent)
    authorize_mutation()
    atomic_json(path, normalized)
    if _active_atomic_scratch(path):
        raise ProfilerError("receipt authorization did not reach durable completion")
    _load_receipt_authorizations(output_root)
    return path, digest


def _write_exclusive_bytes(path: Path, payload: bytes, *, name: str) -> BoundFileSnapshot:
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    except AttributeError as exc:  # pragma: no cover - supported authority hosts
        raise ProfilerError("this host lacks required no-follow descriptor support") from exc
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o666)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive kernel contract
                raise OSError("descriptor write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise ProfilerError(f"{name} exclusive write failed; preserving bytes") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    snapshot = _read_bound_bytes(path, name=name)
    if snapshot.payload != payload:
        raise ProfilerError(f"{name} changed after exclusive write; preserving bytes")
    return snapshot


def _creation_staging_path(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}{CREATION_STAGING_SUFFIX}")


def _creation_prepared_path(path: Path) -> Path:
    return path.with_name(f".{path.name}{CREATION_PREPARED_SUFFIX}")


def _write_exclusive_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write stable-name creation custody without PID-named orphan files."""

    payload = canonical_json_bytes(dict(value)) + b"\n"
    _write_exclusive_bytes(path, payload, name=f"exclusive creation {path.name}")


def _materialize_creation_json(path: Path, value: Mapping[str, Any]) -> None:
    """Recover or commit one certified stable prepared initialization file."""

    expected = json.loads(canonical_json_bytes(dict(value)))
    prepared = _creation_prepared_path(path)
    if os.path.lexists(path):
        if _load_canonical_object(path, name=f"creation {path.name}") != expected:
            raise ProfilerError(f"certified creation file drift: {path.name}")
        if os.path.lexists(prepared):
            try:
                prepared_value = _load_canonical_object(
                    prepared,
                    name=f"prepared creation {path.name}",
                )
            except _MalformedCanonicalJSONError as exc:
                # A regular, link-count-one partial prepared file is certified
                # rebuildable pre-stage scratch by staging_scratch.json.
                if exc.file_snapshot is None:  # pragma: no cover - defensive
                    raise
                _unlink_bound(prepared, exc.file_snapshot, name=f"partial prepared creation {path.name}")
            else:
                if prepared_value != expected:
                    raise ProfilerError(f"prepared certified creation file drift: {path.name}")
                snapshot = _read_bound_bytes(prepared, name=f"prepared creation {path.name}")
                _unlink_bound(prepared, snapshot, name=f"prepared creation {path.name}")
        return
    if os.path.lexists(prepared):
        try:
            prepared_value = _load_canonical_object(
                prepared,
                name=f"prepared creation {path.name}",
            )
        except _MalformedCanonicalJSONError as exc:
            if exc.file_snapshot is None:  # pragma: no cover - defensive
                raise
            _unlink_bound(prepared, exc.file_snapshot, name=f"partial prepared creation {path.name}")
            _write_exclusive_canonical_json(prepared, expected)
        else:
            if prepared_value != expected:
                raise ProfilerError(f"prepared certified creation file drift: {path.name}")
    else:
        _write_exclusive_canonical_json(prepared, expected)
    prepared_snapshot = _read_bound_bytes(prepared, name=f"prepared creation {path.name}")
    if prepared_snapshot.payload != canonical_json_bytes(expected) + b"\n":
        raise ProfilerError(f"prepared certified creation file drift: {path.name}")
    _replace_bound(prepared, path, prepared_snapshot, name=f"prepared creation {path.name}")
    _fsync_stage_directory(path.parent)


def _validate_storage_preflight(
    value: object,
    *,
    expected_output_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != STORAGE_PREFLIGHT_KEYS:
        raise ProfilerError("profile storage preflight schema is malformed")
    free = value.get("free_bytes_before")
    required = value.get("required_free_bytes")
    if type(free) is not int or free < 0 or type(required) is not int or required < 0:
        raise ProfilerError("profile storage preflight byte counts are malformed")
    if required > free or value.get("PASS") is not True:
        raise ProfilerError("profile storage preflight did not pass")
    selected_raw = value.get("selected_root")
    anchor_raw = value.get("filesystem_anchor")
    waterfall_raw = value.get("waterfall_order")
    existing_raw = value.get("existing_approved_roots")
    selection_scope = value.get("selection_scope")
    local_test = value.get("allow_local_output_for_tests")
    expected_waterfall = [str(path) for path in SSD_ROOTS]
    if (
        not isinstance(selected_raw, str)
        or not Path(selected_raw).is_absolute()
        or Path(selected_raw) != expected_output_root
        or not isinstance(anchor_raw, str)
        or not Path(anchor_raw).is_absolute()
        or not isinstance(waterfall_raw, list)
        or waterfall_raw != expected_waterfall
        or not isinstance(existing_raw, list)
        or any(not isinstance(root, str) or not Path(root).is_absolute() for root in existing_raw)
        or selection_scope not in {FRESH_STORAGE_SELECTION_SCOPE, RESUME_STORAGE_SELECTION_SCOPE}
        or type(local_test) is not bool
    ):
        raise ProfilerError("profile storage preflight path/type custody is malformed")
    selected = Path(selected_raw)
    waterfall = [Path(root).resolve() for root in waterfall_raw]
    existing = [Path(root).resolve() for root in existing_raw]
    waterfall_positions = {root: index for index, root in enumerate(waterfall)}
    try:
        positions = [waterfall_positions[root] for root in existing]
    except KeyError as exc:
        raise ProfilerError("profile existing SSD roots are outside the canonical waterfall") from exc
    if len(existing) != len(set(existing)) or positions != sorted(positions):
        raise ProfilerError("profile existing SSD roots are duplicated or out of order")
    if not local_test:
        selected_roots = existing if selection_scope == RESUME_STORAGE_SELECTION_SCOPE else existing[:1]
        if not selected_roots or not any(_is_relative_to(selected, root) for root in selected_roots):
            raise ProfilerError("profile selected root violates its persisted SSD selection scope")
    canonical_json_bytes(value)
    return value


def _creation_storage_identity_from_preflight(
    value: Mapping[str, Any],
    *,
    expected_output_root: Path,
) -> dict[str, Any]:
    """Project a complete creation preflight onto immutable custody fields.

    Free-space bytes and the nearest-existing sampling path are observations,
    not experiment identity.  The latter can change merely because this
    transaction created the output parent.  The complete first observation is
    still preserved in scratch/progress/certification; only this explicit
    projection roots the stage chain.
    """

    normalized = json.loads(canonical_json_bytes(dict(value)))
    _validate_storage_preflight(normalized, expected_output_root=expected_output_root)
    stable = {key: normalized[key] for key in sorted(STABLE_STORAGE_PREFLIGHT_KEYS)}
    identity = {
        "schema": CREATION_STORAGE_IDENTITY_SCHEMA,
        "stable_preflight": stable,
        "volatile_fields_excluded": sorted(VOLATILE_STORAGE_PREFLIGHT_KEYS),
    }
    canonical_json_bytes(identity)
    return identity


def _validate_creation_storage_identity(
    value: object,
    *,
    expected_output_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CREATION_STORAGE_IDENTITY_KEYS:
        raise ProfilerError("profile creation storage identity schema is malformed")
    if value.get("schema") != CREATION_STORAGE_IDENTITY_SCHEMA or value.get("volatile_fields_excluded") != sorted(
        VOLATILE_STORAGE_PREFLIGHT_KEYS
    ):
        raise ProfilerError("profile creation storage identity exclusions are malformed")
    stable = value.get("stable_preflight")
    if not isinstance(stable, dict) or set(stable) != STABLE_STORAGE_PREFLIGHT_KEYS:
        raise ProfilerError("profile creation storage stable preflight is malformed")
    reconstructed = {
        **stable,
        # These synthetic observation values are used only to reuse the full
        # preflight schema validator.  They never enter the returned identity.
        "filesystem_anchor": stable.get("selected_root"),
        "free_bytes_before": stable.get("required_free_bytes"),
    }
    _validate_storage_preflight(reconstructed, expected_output_root=expected_output_root)
    if stable.get("selection_scope") != FRESH_STORAGE_SELECTION_SCOPE:
        raise ProfilerError("profile creation storage identity is not fresh-first")
    normalized = json.loads(canonical_json_bytes(value))
    canonical_json_bytes(normalized)
    return normalized


def _normalized_argv(value: Iterable[str], *, name: str) -> list[str]:
    result = list(value)
    if not result or any(not isinstance(argument, str) or not argument for argument in result):
        raise ProfilerError(f"{name} must be a nonempty string argv")
    canonical_json_bytes(result)
    return result


def _attested_argument_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    return value


def _attest_exact_argv(args: argparse.Namespace, exact_argv: Iterable[str]) -> list[str]:
    """Prove the persisted command reparses to the effective profiler request."""

    argv = _normalized_argv(exact_argv, name="exact argv")
    if len(argv) < 2:
        raise ProfilerError("exact argv must name the Python executable and profiler tool")
    try:
        executable = Path(argv[0]).expanduser().resolve(strict=True)
        tool = Path(argv[1]).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ProfilerError("exact argv executable/tool custody is unavailable") from exc
    if executable != Path(sys.executable).resolve(strict=True) or tool != Path(__file__).resolve(strict=True):
        raise ProfilerError("exact argv does not name this Python runtime and profiler tool")
    try:
        reparsed = _parse_args(argv[2:])
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ProfilerError("exact argv does not parse as a profiler invocation") from exc
    effective = vars(args)
    reconstructed = vars(reparsed)
    if set(effective) != set(reconstructed):
        raise ProfilerError("exact argv/effective argument schema mismatch")
    effective_attestation = {key: _attested_argument_value(value) for key, value in sorted(effective.items())}
    reconstructed_attestation = {key: _attested_argument_value(value) for key, value in sorted(reconstructed.items())}
    if effective_attestation != reconstructed_attestation:
        raise ProfilerError("exact argv does not reproduce the effective profiler arguments")
    canonical_json_bytes(effective_attestation)
    return argv


def _canonical_rebuild_argv(exact_argv: Iterable[str], *, resume: bool) -> list[str]:
    argv = _normalized_argv(exact_argv, name="exact argv")
    resume_count = argv.count("--resume")
    if resume:
        if resume_count != 1:
            raise ProfilerError("resume argv must contain exactly one --resume token")
        argv.remove("--resume")
    elif resume_count:
        raise ProfilerError("fresh profile argv unexpectedly contains --resume")
    return argv


def _staging_scratch_record(
    *,
    output_root: Path,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_preflight = json.loads(canonical_json_bytes(dict(storage_preflight)))
    _validate_storage_preflight(normalized_preflight, expected_output_root=output_root)
    return {
        "schema": STAGING_SCRATCH_SCHEMA,
        "final_output_root": str(output_root),
        "identity_sha256": _canonical_sha256(identity_sha256, name="staging identity hash"),
        "exact_rebuild_argv": _normalized_argv(exact_rebuild_argv, name="staging rebuild argv"),
        "storage_preflight": normalized_preflight,
        "rebuildable": True,
        "score_authority": False,
        "safe_disposition": "REBUILD_IN_PLACE_OR_DELETE_ONLY_CERTIFIED_PRE_FINAL_STAGING",
    }


def _staging_scratch_records_match(
    record: Mapping[str, Any],
    expected_record: Mapping[str, Any],
) -> bool:
    stored_preflight = record.get("storage_preflight")
    expected_preflight = expected_record.get("storage_preflight")
    return (
        set(record) == set(expected_record)
        and record.get("schema") == expected_record.get("schema")
        and record.get("final_output_root") == expected_record.get("final_output_root")
        and record.get("identity_sha256") == expected_record.get("identity_sha256")
        and record.get("exact_rebuild_argv") == expected_record.get("exact_rebuild_argv")
        and isinstance(stored_preflight, dict)
        and isinstance(expected_preflight, Mapping)
        and {key: value for key, value in stored_preflight.items() if key not in VOLATILE_STORAGE_PREFLIGHT_KEYS}
        == {key: value for key, value in expected_preflight.items() if key not in VOLATILE_STORAGE_PREFLIGHT_KEYS}
        and record.get("rebuildable") is True
        and record.get("score_authority") is False
        and record.get("safe_disposition") == expected_record.get("safe_disposition")
    )


def _load_complete_staging_scratch(staging_root: Path) -> dict[str, Any] | None:
    """Load final or fully prepared first-scratch custody without hiding links.

    A malformed regular link-count-one prepared file is the certified partial
    write case handled by ``_materialize_creation_json`` and returns ``None``.
    Link/path violations and malformed final scratch records remain fatal.
    """

    scratch_path = staging_root / STAGING_SCRATCH_NAME
    if os.path.lexists(scratch_path):
        return _load_canonical_object(scratch_path, name="pre-final staging scratch")
    prepared = _creation_prepared_path(scratch_path)
    if not os.path.lexists(prepared):
        return None
    metadata = _require_local_regular_file(prepared, name="prepared pre-final staging scratch")
    try:
        return _load_canonical_object(prepared, name="prepared pre-final staging scratch")
    except _MalformedCanonicalJSONError:
        if metadata.st_nlink != 1:  # defensive; helper already enforces this
            raise
        return None


def _certified_staging_matches(
    staging_root: Path,
    *,
    expected_record: Mapping[str, Any],
) -> bool:
    try:
        record = _load_canonical_object(staging_root / STAGING_SCRATCH_NAME, name="staging scratch")
    except ProfilerError:
        return False
    return _staging_scratch_records_match(record, expected_record)


def _output_certification_record(
    *,
    output_root: Path,
    identity_sha256: str,
    identity_json_sha256: str,
    exact_rebuild_argv: Iterable[str],
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": OUTPUT_CERTIFICATION_SCHEMA,
        "output_root": str(output_root),
        "identity_sha256": _canonical_sha256(identity_sha256, name="certification identity hash"),
        "identity_json_sha256": _canonical_sha256(
            identity_json_sha256,
            name="certification identity-file hash",
        ),
        "exact_rebuild_argv": _normalized_argv(exact_rebuild_argv, name="certification rebuild argv"),
        "storage_preflight": json.loads(canonical_json_bytes(dict(storage_preflight))),
        "large_artifact_policy": "CERTIFY_OR_BLOCK",
        "rebuildable": True,
        "retention_policy": "DURABLE_STAGE_CHAIN_PRESERVED",
        "delete_or_move_before_complete": False,
        "cleanup_action_performed": False,
        "cleanup_disposition": "PRESERVE_UNTIL_EXPLICIT_CERTIFIED_COLD_STORE_OR_REBUILDABLE_CLEANUP",
        "false_authority_flags": {
            "score_authority": False,
            "promotion_eligible": False,
            "global_compressed_stream_minimum_claim": False,
            "pose_bank_wired": False,
            "factor10_solved": False,
        },
    }


def _validate_output_certification(
    current_root: Path,
    *,
    final_output_root: Path,
    expected_identity: Mapping[str, Any],
    expected_rebuild_argv: Iterable[str],
) -> dict[str, Any]:
    normalized_identity = json.loads(canonical_json_bytes(dict(expected_identity)))
    identity_sha256 = _sha256_bytes(canonical_json_bytes(normalized_identity))
    unexpanded_current = current_root.expanduser()
    _require_local_directory(unexpanded_current, name="profile certification root")
    resolved_current = unexpanded_current.resolve(strict=True)
    resolved_final = final_output_root.expanduser().resolve()
    bound_storage_identity = _validate_creation_storage_identity(
        normalized_identity.get("creation_storage_identity"),
        expected_output_root=resolved_final,
    )
    if resolved_current not in {resolved_final, _creation_staging_path(resolved_final)}:
        raise ProfilerError("profile staging/final location does not match certification")
    scratch = _load_canonical_object(resolved_current / STAGING_SCRATCH_NAME, name="staging scratch")
    scratch_preflight = scratch.get("storage_preflight")
    if not isinstance(scratch_preflight, dict):
        raise ProfilerError("profile staging scratch lacks creation storage custody")
    _validate_storage_preflight(scratch_preflight, expected_output_root=resolved_final)
    if (
        _creation_storage_identity_from_preflight(
            scratch_preflight,
            expected_output_root=resolved_final,
        )
        != bound_storage_identity
    ):
        raise ProfilerError("profile staging scratch differs from identity-bound creation storage")
    certification = _load_canonical_object(
        resolved_current / OUTPUT_CERTIFICATION_NAME,
        name="output certification",
    )
    identity_path = resolved_current / IDENTITY_NAME
    stored_identity = _load_canonical_object(identity_path, name="profile identity")
    if stored_identity != normalized_identity:
        raise ProfilerError("persisted profile identity does not match freshly derived identity")
    if _sha256_bytes(canonical_json_bytes(stored_identity)) != identity_sha256:
        raise ProfilerError("persisted profile identity hash drift")
    identity_json_sha256 = sha256_file(identity_path)
    expected_argv = _normalized_argv(expected_rebuild_argv, name="expected rebuild argv")
    expected_scratch = _staging_scratch_record(
        output_root=resolved_final,
        identity_sha256=identity_sha256,
        exact_rebuild_argv=expected_argv,
        storage_preflight=scratch_preflight,
    )
    if scratch != expected_scratch:
        raise ProfilerError("profile staging scratch identity is malformed")
    expected_certification = _output_certification_record(
        output_root=resolved_final,
        identity_sha256=identity_sha256,
        identity_json_sha256=identity_json_sha256,
        exact_rebuild_argv=expected_argv,
        storage_preflight=scratch_preflight,
    )
    if certification != expected_certification:
        raise ProfilerError("profile output certification is malformed")
    return certification


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _atomic_stage(
    path: Path,
    payload: bytes,
    *,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    authorize_mutation: Callable[[], None] = lambda: None,
) -> None:
    """Write one deterministic recoverable stage without replacing custody."""

    identity_hash = _canonical_sha256(identity_sha256, name="stage-attempt identity")
    rebuild_argv = _normalized_argv(exact_rebuild_argv, name="stage-attempt rebuild argv")
    final_match = re.fullmatch(r"frame_([0-9]{4})\.bin", path.name)
    if final_match is None:
        raise ProfilerError("stage commit requires a canonical final-stage name")
    frame = int(final_match.group(1))
    prepared = _prepared_stage_path(path)
    stage_entries = list(path.parent.iterdir())
    attempt_custody = _validate_stage_attempt_custody(
        path.parent.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=rebuild_argv,
        terminal=False,
    )
    if attempt_custody.intent_building:
        if len(attempt_custody.intent_building) != 1 or attempt_custody.intent_building[0][0] != frame:
            raise ProfilerError("stage commit conflicts with another pending intent construction")
        attempt = attempt_custody.intent_building[0][1]
    else:
        attempt = _next_stage_attempt(path)
    transaction_root = _ensure_stage_attempt_directory(
        path.parent,
        identity_sha256=identity_hash,
        frame=frame,
        attempt=attempt,
        authorize_mutation=authorize_mutation,
    )
    intent = transaction_root / STAGE_ATTEMPT_NAME
    transaction = _stage_attempt_transaction(
        final_path=path,
        intent_path=intent,
        identity_sha256=identity_hash,
        frame=frame,
        attempt=attempt,
        intended_payload=payload,
        exact_rebuild_argv=rebuild_argv,
    )
    retained_names = set(attempt_custody.proven_stage_retained_names)
    active_entries = [entry for entry in stage_entries if entry.name not in retained_names]
    final_entries = sorted(
        entry for entry in active_entries if re.fullmatch(r"frame_[0-9]{4}\.bin", entry.name) is not None
    )
    expected_final_names = [f"frame_{index:04d}.bin" for index in range(len(final_entries))]
    if (
        [entry.name for entry in final_entries] != expected_final_names
        or path.name != f"frame_{len(final_entries):04d}.bin"
        or any(entry not in final_entries for entry in active_entries)
    ):
        raise ProfilerError("stage commit root contains unidentified, non-prefix, or active scratch custody")
    competing_intents = [entry for entry in stage_entries if STAGE_INTENT_RE.fullmatch(entry.name) is not None]
    if os.path.lexists(path) or os.path.lexists(prepared) or competing_intents:
        raise ProfilerError(f"refusing conflicting final/prepared stage bytes for {path.name}")
    authorize_mutation()
    atomic_json(intent, transaction)
    intent_snapshot = _read_bound_bytes(intent, name=f"stage-attempt transaction for {path.name}")
    _fsync_stage_directory(transaction_root)
    # ``xb`` makes a second writer fail without overwriting the first writer's
    # evidence.  A write/rename interruption deliberately leaves the prepared
    # bytes in place for fail-closed resume reconciliation.
    authorize_mutation()
    prepared_snapshot = _write_exclusive_bytes(prepared, payload, name=f"prepared stage {path.name}")
    if os.path.lexists(path):
        raise ProfilerError(f"final stage appeared while preparing {path.name}")
    authorize_mutation()
    _replace_bound(prepared, path, prepared_snapshot, name=f"prepared stage {path.name}")
    _fsync_stage_directory(path.parent)
    _finalize_successful_stage_attempt(
        stages=path.parent,
        final_path=path,
        intent_path=intent,
        intent_snapshot=intent_snapshot,
        identity_sha256=identity_hash,
        exact_rebuild_argv=rebuild_argv,
        authorize_mutation=authorize_mutation,
    )


def _prepared_stage_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}{PREPARED_STAGE_SUFFIX}")


def _stage_intent_path(
    final_path: Path,
    *,
    identity_sha256: str,
    payload_bytes: int,
    payload_sha256: str,
    attempt: int | None = None,
) -> Path:
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise ProfilerError("stage intent payload size must be a nonnegative integer")
    _canonical_sha256(payload_sha256, name="stage intent payload hash")
    if attempt is None:
        attempt = _next_stage_attempt(final_path)
    if type(attempt) is not int or not 0 <= attempt <= 99_999_999:
        raise ProfilerError("stage attempt ordinal must be an eight-digit nonnegative integer")
    frame_match = re.fullmatch(r"frame_([0-9]{4})\.bin", final_path.name)
    if frame_match is None:
        raise ProfilerError("stage attempt intent requires a canonical final-stage name")
    return (
        _stage_attempt_directory(
            final_path.parent,
            identity_sha256=_canonical_sha256(identity_sha256, name="stage intent identity"),
            frame=int(frame_match.group(1)),
            attempt=attempt,
        )
        / STAGE_ATTEMPT_NAME
    )


def _stage_attempt_transaction(
    *,
    final_path: Path,
    intent_path: Path,
    identity_sha256: str,
    frame: int,
    attempt: int,
    intended_payload: bytes,
    exact_rebuild_argv: Iterable[str],
) -> dict[str, Any]:
    """Build the immutable authority record persisted as the stage intent."""

    identity_hash = _canonical_sha256(identity_sha256, name="stage-attempt identity")
    rebuild_argv = _normalized_argv(exact_rebuild_argv, name="stage-attempt rebuild argv")
    if type(frame) is not int or not 0 <= frame < EXPECTED_PAIRS:
        raise ProfilerError("stage-attempt frame is malformed")
    if type(attempt) is not int or not 0 <= attempt <= 99_999_999:
        raise ProfilerError("stage-attempt ordinal is malformed")
    if type(intended_payload) is not bytes or not intended_payload:
        raise ProfilerError("stage-attempt intended payload must be nonempty immutable bytes")
    prepared_path = _prepared_stage_path(final_path)
    expected_intent = _stage_intent_path(
        final_path,
        identity_sha256=identity_hash,
        payload_bytes=len(intended_payload),
        payload_sha256=_sha256_bytes(intended_payload),
        attempt=attempt,
    )
    if intent_path != expected_intent or final_path.name != f"frame_{frame:04d}.bin":
        raise ProfilerError("stage-attempt path binding is malformed")
    return {
        "schema": STAGE_ATTEMPT_TRANSACTION_SCHEMA,
        "identity_sha256": identity_hash,
        "frame": frame,
        "attempt": attempt,
        "final_basename": final_path.name,
        "prepared_basename": prepared_path.name,
        "intent_basename": intent_path.name,
        "intended_bytes": len(intended_payload),
        "intended_sha256": _sha256_bytes(intended_payload),
        "exact_rebuild_argv": rebuild_argv,
        "false_authority_flags": dict(STAGE_ATTEMPT_FALSE_AUTHORITY_FLAGS),
    }


def _read_stage_attempt_transaction(
    path: Path,
    *,
    expected_identity_sha256: str | None = None,
    expected_rebuild_argv: Iterable[str] | None = None,
) -> tuple[dict[str, Any], str, BoundFileSnapshot]:
    """Read one active/retained intent as an exact stage-attempt transaction."""

    original = (
        feature_cache_module.retained_original_name(path.name)
        if feature_cache_module.is_retained_name(path.name)
        else path.name
    )
    legacy_match = STAGE_INTENT_RE.fullmatch(original)
    new_match = RECOVERY_TRANSACTION_RE.fullmatch(path.parent.name) if original == STAGE_ATTEMPT_NAME else None
    if legacy_match is None and new_match is None:
        raise ProfilerError(f"malformed stage-attempt transaction name: {original}")
    try:
        snapshot = (
            feature_cache_module.validate_retained_file(path, role=f"retained stage-attempt {path.name}")
            if feature_cache_module.is_retained_name(path.name)
            else _read_bound_bytes(path, name=f"stage-attempt transaction {path.name}")
        )
    except feature_cache_module.FeatureCacheError as exc:
        raise ProfilerError("stage-attempt retained custody is malformed; preserving bytes") from exc
    try:
        value = json.loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfilerError("stage-attempt transaction payload is malformed canonical JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != STAGE_ATTEMPT_TRANSACTION_KEYS
        or snapshot.payload != canonical_json_bytes(value) + b"\n"
    ):
        raise ProfilerError("stage-attempt transaction schema/canonical encoding is malformed")
    frame = int(legacy_match.group(2) if legacy_match is not None else new_match.group(1))
    attempt = int(legacy_match.group(3) if legacy_match is not None else new_match.group(2))
    intended_bytes = int(legacy_match.group(4)) if legacy_match is not None else value.get("intended_bytes")
    intended_sha256 = legacy_match.group(5) if legacy_match is not None else value.get("intended_sha256")
    expected_final = legacy_match.group(1) if legacy_match is not None else f"frame_{frame:04d}.bin"
    if (
        value.get("schema") != STAGE_ATTEMPT_TRANSACTION_SCHEMA
        or value.get("frame") != frame
        or value.get("attempt") != attempt
        or value.get("final_basename") != expected_final
        or value.get("prepared_basename") != _prepared_stage_path(Path(expected_final)).name
        or value.get("intent_basename") != original
        or value.get("intended_bytes") != intended_bytes
        or value.get("intended_sha256") != intended_sha256
        or value.get("false_authority_flags") != STAGE_ATTEMPT_FALSE_AUTHORITY_FLAGS
    ):
        raise ProfilerError("stage-attempt transaction disagrees with its durable role/name")
    identity_hash = _canonical_sha256(value.get("identity_sha256"), name="stage-attempt transaction identity")
    rebuild_argv = _normalized_argv(value.get("exact_rebuild_argv", ()), name="stage-attempt transaction argv")
    if expected_identity_sha256 is not None and identity_hash != _canonical_sha256(
        expected_identity_sha256,
        name="expected stage-attempt identity",
    ):
        raise ProfilerError("stage-attempt transaction identity differs from profile custody")
    if new_match is not None and path.parent.parent.name != identity_hash:
        raise ProfilerError("stage-attempt directory identity differs from its transaction")
    if expected_rebuild_argv is not None and rebuild_argv != _normalized_argv(
        expected_rebuild_argv,
        name="expected stage-attempt argv",
    ):
        raise ProfilerError("stage-attempt transaction argv differs from profile custody")
    return value, _sha256_bytes(snapshot.payload), snapshot


def _stage_attempt_directory(stages: Path, *, identity_sha256: str, frame: int, attempt: int) -> Path:
    return _recovery_transaction_path(
        stages,
        identity_hash=identity_sha256,
        frame=frame,
        attempt=attempt,
    )


def _ensure_stage_attempt_directory(
    stages: Path,
    *,
    identity_sha256: str,
    frame: int,
    attempt: int,
    authorize_mutation: Callable[[], None],
) -> Path:
    transaction = _stage_attempt_directory(
        stages,
        identity_sha256=identity_sha256,
        frame=frame,
        attempt=attempt,
    )
    for directory in (transaction.parent.parent, transaction.parent, transaction):
        if os.path.lexists(directory):
            _require_local_directory(directory, name=f"stage-attempt directory {directory.name}")
        else:
            authorize_mutation()
            directory.mkdir()
            _fsync_stage_directory(directory.parent)
    return transaction


def _stage_success_outcome(
    *,
    transaction: Mapping[str, Any],
    transaction_sha256: str,
    final_payload: bytes,
) -> dict[str, Any]:
    _canonical_sha256(transaction_sha256, name="successful stage-attempt transaction hash")
    return {
        "schema": STAGE_ATTEMPT_SUCCESS_SCHEMA,
        "identity_sha256": transaction["identity_sha256"],
        "frame": transaction["frame"],
        "attempt": transaction["attempt"],
        "stage_attempt_transaction_sha256": transaction_sha256,
        "final_basename": transaction["final_basename"],
        "final_bytes": len(final_payload),
        "final_sha256": _sha256_bytes(final_payload),
        "exact_rebuild_argv": transaction["exact_rebuild_argv"],
        "false_authority_flags": dict(STAGE_ATTEMPT_FALSE_AUTHORITY_FLAGS),
    }


def _finalize_successful_stage_attempt(
    *,
    stages: Path,
    final_path: Path,
    intent_path: Path,
    intent_snapshot: BoundFileSnapshot,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    authorize_mutation: Callable[[], None],
) -> dict[str, Any]:
    """Durably bind final bytes to their attempt before retiring the intent."""

    transaction, transaction_sha256, current_intent = _read_stage_attempt_transaction(
        intent_path,
        expected_identity_sha256=identity_sha256,
        expected_rebuild_argv=exact_rebuild_argv,
    )
    if (
        current_intent.file_identity != intent_snapshot.file_identity
        or current_intent.payload != intent_snapshot.payload
    ):
        raise ProfilerError("stage-attempt transaction changed before success finalization")
    final_snapshot = _read_bound_bytes(final_path, name="successful stage-attempt final")
    if (
        final_path.name != transaction["final_basename"]
        or len(final_snapshot.payload) != transaction["intended_bytes"]
        or _sha256_bytes(final_snapshot.payload) != transaction["intended_sha256"]
    ):
        raise ProfilerError("successful final stage differs from its stage-attempt transaction")
    transaction_root = _ensure_stage_attempt_directory(
        stages,
        identity_sha256=identity_sha256,
        frame=transaction["frame"],
        attempt=transaction["attempt"],
        authorize_mutation=authorize_mutation,
    )
    if os.path.lexists(transaction_root / RECOVERY_MANIFEST_NAME) or os.path.lexists(
        transaction_root / RECOVERY_PAYLOAD_NAME
    ):
        raise ProfilerError("stage attempt has both success and recovery outcomes")
    success = _stage_success_outcome(
        transaction=transaction,
        transaction_sha256=transaction_sha256,
        final_payload=final_snapshot.payload,
    )
    success_path = transaction_root / STAGE_SUCCESS_NAME
    success_payload = canonical_json_bytes(success) + b"\n"
    if os.path.lexists(success_path):
        if _load_canonical_object(success_path, name="stage-attempt success outcome") != success:
            raise ProfilerError("stage-attempt success outcome drift")
        authorize_mutation()
        atomic_json(success_path, success, expected_prior_payloads=(success_payload,))
    else:
        authorize_mutation()
        atomic_json(success_path, success)
    _fsync_stage_directory(transaction_root)
    # The canonical attempt record remains immutable beside its bound outcome.
    _fsync_stage_directory(transaction_root)
    return success


def _next_stage_attempt(final_path: Path) -> int:
    frame_match = re.fullmatch(r"\.?frame_([0-9]{4})\.bin(?:\.prepared)?", final_path.name)
    if frame_match is None:
        raise ProfilerError("stage attempt allocation requires a canonical final-stage name")
    frame = int(frame_match.group(1))
    recovery_root = final_path.parent.parent / RECOVERY_ROOT_NAME
    attempts: list[int] = []
    if os.path.lexists(recovery_root):
        _require_local_directory(recovery_root, name="stage attempt recovery root")
        for identity_root in recovery_root.iterdir():
            _require_local_directory(identity_root, name="stage attempt identity root")
            if re.fullmatch(r"[0-9a-f]{64}", identity_root.name) is None:
                raise ProfilerError("stage attempt recovery root contains an unknown identity name")
            for transaction in identity_root.iterdir():
                match = RECOVERY_TRANSACTION_RE.fullmatch(transaction.name)
                if match is None:
                    raise ProfilerError("stage attempt recovery root contains a malformed transaction")
                if int(match.group(1)) == frame:
                    attempts.append(int(match.group(2)))
    return max(attempts, default=-1) + 1


def _parse_stage_intent(path: Path) -> tuple[Path, int, int, int, str, BoundFileSnapshot]:
    transaction, _transaction_sha256, snapshot = _read_stage_attempt_transaction(path)
    stages = path.parent if STAGE_INTENT_RE.fullmatch(path.name) is not None else path.parents[3] / "stages"
    return (
        stages / transaction["final_basename"],
        transaction["frame"],
        transaction["attempt"],
        transaction["intended_bytes"],
        transaction["intended_sha256"],
        snapshot,
    )


def _pending_stage_attempt_paths(
    output_root: Path,
    *,
    identity_sha256: str,
    custody: _ValidatedStageAttemptCustody,
) -> list[Path]:
    completed: set[tuple[int, int]] = set()
    for outcome in custody.outcomes:
        record = outcome.get("success") if outcome.get("outcome") == "success" else outcome.get("manifest")
        if isinstance(record, dict):
            completed.add((record["frame"], record["attempt"]))
    identity_root = output_root / RECOVERY_ROOT_NAME / identity_sha256
    if not os.path.lexists(identity_root):
        return []
    result: list[Path] = []
    for directory in sorted(identity_root.iterdir(), key=lambda item: item.name):
        match = RECOVERY_TRANSACTION_RE.fullmatch(directory.name)
        if match is None:
            raise ProfilerError("stage attempt recovery root contains a malformed transaction")
        key = (int(match.group(1)), int(match.group(2)))
        attempt_path = directory / STAGE_ATTEMPT_NAME
        if key not in completed and os.path.lexists(attempt_path):
            result.append(attempt_path)
    return result


def _recovery_transaction_path(stages: Path, *, identity_hash: str, frame: int, attempt: int) -> Path:
    return stages.parent / RECOVERY_ROOT_NAME / identity_hash / f"frame_{frame:04d}-attempt_{attempt:08d}"


def _recovery_manifest(
    *,
    original: Path,
    destination: Path,
    actual_payload: bytes,
    intended_bytes: int,
    intended_sha256: str,
    identity_hash: str,
    frame: int,
    attempt: int,
    stage_attempt_transaction_sha256: str,
    prepared_source_present: bool,
    prepared_source_file_identity: tuple[int, int, int, int, int] | None,
    exact_rebuild_argv: Iterable[str],
) -> dict[str, Any]:
    if type(prepared_source_present) is not bool:
        raise ProfilerError("recovery prepared-source presence must be boolean")
    if prepared_source_present:
        if (
            type(prepared_source_file_identity) is not tuple
            or len(prepared_source_file_identity) != 5
            or any(type(value) is not int or value < 0 for value in prepared_source_file_identity)
            or prepared_source_file_identity[4] != 1
        ):
            raise ProfilerError("recovery prepared-source file identity is malformed")
        normalized_prepared_identity: list[int] | None = list(prepared_source_file_identity)
    else:
        if prepared_source_file_identity is not None:
            raise ProfilerError("missing recovery prepared source cannot claim a file identity")
        normalized_prepared_identity = None
    return {
        "schema": RECOVERY_MANIFEST_SCHEMA,
        "original_path": str(original.resolve(strict=False)),
        "destination_path": str(destination.resolve(strict=False)),
        "actual_bytes": len(actual_payload),
        "actual_sha256": _sha256_bytes(actual_payload),
        "intended_bytes": intended_bytes,
        "intended_sha256": _canonical_sha256(intended_sha256, name="recovery intended hash"),
        "identity_sha256": _canonical_sha256(identity_hash, name="recovery identity hash"),
        "frame": frame,
        "attempt": attempt,
        "transaction": f"frame_{frame:04d}-attempt_{attempt:08d}",
        "stage_attempt_transaction_sha256": _canonical_sha256(
            stage_attempt_transaction_sha256,
            name="recovery stage-attempt transaction hash",
        ),
        "prepared_source_present": prepared_source_present,
        "prepared_source_file_identity": normalized_prepared_identity,
        "exact_rebuild_argv": _normalized_argv(exact_rebuild_argv, name="recovery rebuild argv"),
        "rebuildability_reason": "INTENT_BOUND_MISSING_OR_SIZE_SHORT_INTERRUPTED_STAGE_SCRATCH",
        "false_authority_flags": {
            "stage_chain_member": False,
            "rate_stream_member": False,
            "score_authority": False,
            "promotion_eligible": False,
        },
    }


def _recover_interrupted_stage(
    *,
    stages: Path,
    final_path: Path,
    prepared_path: Path,
    intent_path: Path,
    frame: int,
    attempt: int,
    intended_bytes: int,
    intended_sha256: str,
    identity_hash: str,
    exact_rebuild_argv: Iterable[str],
    authorize_mutation: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Move intent-proven interrupted bytes into an idempotent recovery txn."""

    attempt_transaction, attempt_transaction_sha256, intent_snapshot = _read_stage_attempt_transaction(
        intent_path,
        expected_identity_sha256=identity_hash,
        expected_rebuild_argv=exact_rebuild_argv,
    )
    if (
        attempt_transaction["frame"] != frame
        or attempt_transaction["attempt"] != attempt
        or attempt_transaction["final_basename"] != final_path.name
        or attempt_transaction["prepared_basename"] != prepared_path.name
        or attempt_transaction["intended_bytes"] != intended_bytes
        or attempt_transaction["intended_sha256"] != intended_sha256
    ):
        raise ProfilerError("interrupted stage differs from its stage-attempt transaction")
    transaction = _recovery_transaction_path(
        stages,
        identity_hash=identity_hash,
        frame=frame,
        attempt=attempt,
    )
    recovery_root = transaction.parent.parent
    identity_root = transaction.parent
    for directory in (recovery_root, identity_root, transaction):
        if os.path.lexists(directory):
            _require_local_directory(directory, name=f"stage recovery directory {directory.name}")
        else:
            authorize_mutation()
            directory.mkdir()
            _fsync_stage_directory(directory.parent)
    destination = transaction / RECOVERY_PAYLOAD_NAME
    manifest_path = transaction / RECOVERY_MANIFEST_NAME
    if os.path.lexists(transaction / STAGE_SUCCESS_NAME):
        raise ProfilerError("stage attempt has both recovery and success outcomes")
    source_exists = os.path.lexists(prepared_path)
    destination_exists = os.path.lexists(destination)
    source_snapshot = (
        _read_bound_bytes(prepared_path, name="interrupted stage recovery source") if source_exists else None
    )
    destination_snapshot = (
        _read_bound_bytes(destination, name="interrupted stage recovery destination") if destination_exists else None
    )
    if source_snapshot is not None and destination_snapshot is not None:
        if source_snapshot.payload != destination_snapshot.payload:
            raise ProfilerError("interrupted stage source/destination conflict; preserving both byte sets")
        actual_payload = source_snapshot.payload
    elif source_snapshot is not None:
        actual_payload = source_snapshot.payload
    elif destination_snapshot is not None:
        actual_payload = destination_snapshot.payload
    else:
        actual_payload = b""
    if len(actual_payload) >= intended_bytes:
        raise ProfilerError("stage recovery is restricted to missing or size-short payloads")
    existing_manifest = (
        _load_canonical_object(manifest_path, name="stage recovery manifest")
        if os.path.lexists(manifest_path)
        else None
    )
    if existing_manifest is None:
        prepared_source_present = source_snapshot is not None
        prepared_source_file_identity = source_snapshot.file_identity if source_snapshot is not None else None
    else:
        prepared_source_present = existing_manifest.get("prepared_source_present")
        stored_file_identity = existing_manifest.get("prepared_source_file_identity")
        if type(prepared_source_present) is not bool or (
            stored_file_identity is not None
            and (
                not isinstance(stored_file_identity, list)
                or len(stored_file_identity) != 5
                or any(type(value) is not int or value < 0 for value in stored_file_identity)
            )
        ):
            raise ProfilerError("stage recovery manifest prepared-source custody is malformed")
        prepared_source_file_identity = tuple(stored_file_identity) if stored_file_identity is not None else None
        if source_snapshot is not None and (
            not prepared_source_present or source_snapshot.file_identity != prepared_source_file_identity
        ):
            raise ProfilerError("stage recovery source differs from its manifest-bound descriptor identity")
    manifest = _recovery_manifest(
        original=prepared_path,
        destination=destination,
        actual_payload=actual_payload,
        intended_bytes=intended_bytes,
        intended_sha256=intended_sha256,
        identity_hash=identity_hash,
        frame=frame,
        attempt=attempt,
        stage_attempt_transaction_sha256=attempt_transaction_sha256,
        prepared_source_present=prepared_source_present,
        prepared_source_file_identity=prepared_source_file_identity,
        exact_rebuild_argv=exact_rebuild_argv,
    )
    if existing_manifest is not None:
        if existing_manifest != manifest:
            raise ProfilerError("stage recovery manifest drift")
        # Also finishes a descriptor-exchange crash after linearization but
        # before the atomic writer removed its displaced-target scratch.
        authorize_mutation()
        atomic_json(
            manifest_path,
            manifest,
            expected_prior_payloads=(canonical_json_bytes(manifest) + b"\n",),
        )
    else:
        authorize_mutation()
        atomic_json(manifest_path, manifest)
    _fsync_stage_directory(transaction)
    if source_snapshot is not None:
        source_snapshot = _read_bound_bytes(prepared_path, name="interrupted stage recovery source")
        if source_snapshot.payload != actual_payload:
            raise ProfilerError("interrupted stage recovery source changed before retention")
        if destination_snapshot is None:
            authorize_mutation()
            destination_snapshot = _write_exclusive_bytes(
                destination,
                actual_payload,
                name="interrupted stage recovery payload copy",
            )
        _fsync_stage_directory(transaction)
        authorize_mutation()
        _unlink_bound(
            prepared_path,
            source_snapshot,
            name="interrupted stage recovery source",
        )
        _fsync_stage_directory(stages)
    elif destination_snapshot is None:
        authorize_mutation()
        destination_snapshot = _write_exclusive_bytes(
            destination,
            b"",
            name="missing interrupted stage recovery payload",
        )
        _fsync_stage_directory(transaction)
    destination_snapshot = _read_bound_bytes(destination, name="recovered interrupted stage payload")
    if destination_snapshot.payload != actual_payload:
        raise ProfilerError("recovered interrupted stage payload changed during transaction")
    if prepared_source_present:
        retained_source_matches: list[BoundFileSnapshot] = []
        for entry in stages.iterdir():
            if not feature_cache_module.is_retained_name(entry.name):
                continue
            if feature_cache_module.retained_original_name(entry.name) != prepared_path.name:
                continue
            try:
                retained_snapshot = feature_cache_module.validate_retained_file(
                    entry,
                    role="recovered prepared-stage descriptor custody",
                )
            except feature_cache_module.FeatureCacheError as exc:
                raise ProfilerError("recovered prepared-stage retention is malformed") from exc
            if retained_snapshot.file_identity == prepared_source_file_identity:
                retained_source_matches.append(retained_snapshot)
        if len(retained_source_matches) != 1 or retained_source_matches[0].payload != actual_payload:
            raise ProfilerError("recovery outcome lacks its exact retained prepared-stage descriptor")
    if os.path.lexists(final_path):
        raise ProfilerError("final stage appeared during interrupted-stage recovery")
    current_transaction, current_transaction_sha256, intent_snapshot = _read_stage_attempt_transaction(
        intent_path,
        expected_identity_sha256=identity_hash,
        expected_rebuild_argv=exact_rebuild_argv,
    )
    if current_transaction != attempt_transaction or current_transaction_sha256 != attempt_transaction_sha256:
        raise ProfilerError("recovered stage-attempt transaction changed before retention")
    # The canonical attempt record remains immutable beside its bound outcome.
    _fsync_stage_directory(intent_path.parent)
    return manifest


def _fsync_stage_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_stage_attempt_custody(
    output_root: Path,
    *,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    terminal: bool,
) -> _ValidatedStageAttemptCustody:
    """Prove every stage attempt, outcome, and retained object bijectively."""

    identity_hash = _canonical_sha256(identity_sha256, name="stage-attempt custody identity")
    rebuild_argv = _normalized_argv(exact_rebuild_argv, name="stage-attempt custody rebuild argv")
    stages = output_root / "stages"
    _require_local_directory(stages, name="stage-attempt custody stage root")
    intents: dict[tuple[int, int], tuple[Path, dict[str, Any], str, BoundFileSnapshot, bool]] = {}
    prepared_active: dict[int, tuple[Path, BoundFileSnapshot]] = {}
    prepared_retained: dict[int, list[tuple[Path, BoundFileSnapshot, int]]] = {}
    final_payloads: dict[int, tuple[Path, bytes]] = {}
    active_intent_count = 0
    for entry in stages.iterdir():
        retained = feature_cache_module.is_retained_name(entry.name)
        original = feature_cache_module.retained_original_name(entry.name) if retained else entry.name
        if STAGE_INTENT_RE.fullmatch(original) is not None:
            transaction, transaction_sha256, snapshot = _read_stage_attempt_transaction(
                entry,
                expected_identity_sha256=identity_hash,
                expected_rebuild_argv=rebuild_argv,
            )
            key = (transaction["frame"], transaction["attempt"])
            if key in intents:
                raise ProfilerError("stage-attempt custody contains duplicate transaction intents")
            intents[key] = (entry, transaction, transaction_sha256, snapshot, retained)
            if not retained:
                active_intent_count += 1
            continue
        prepared_match = re.fullmatch(r"\.frame_([0-9]{4})\.bin\.prepared", original)
        if prepared_match is not None:
            frame = int(prepared_match.group(1))
            if retained:
                try:
                    snapshot = feature_cache_module.validate_retained_file(
                        entry,
                        role=f"retained stage-attempt prepared {entry.name}",
                    )
                except feature_cache_module.FeatureCacheError as exc:
                    raise ProfilerError("retained prepared-stage custody is malformed; preserving bytes") from exc
                retained_match = feature_cache_module.ATOMIC_RETAINED_RE.fullmatch(entry.name)
                assert retained_match is not None
                prepared_retained.setdefault(frame, []).append((entry, snapshot, int(retained_match.group("ordinal"))))
            else:
                if frame in prepared_active:
                    raise ProfilerError("stage-attempt custody contains duplicate active prepared stages")
                prepared_active[frame] = (
                    entry,
                    _read_bound_bytes(entry, name=f"active prepared stage {entry.name}"),
                )
            continue
        final_match = re.fullmatch(r"frame_([0-9]{4})\.bin", original)
        if final_match is not None and not retained:
            frame = int(final_match.group(1))
            if frame in final_payloads:
                raise ProfilerError("stage-attempt custody contains duplicate final stages")
            final_payloads[frame] = (
                entry,
                _read_bound_bytes(entry, name=f"stage-attempt final {entry.name}").payload,
            )
            continue
        raise ProfilerError("stage-attempt root contains unidentified or role-unproven custody")
    if active_intent_count > 1 or len(prepared_active) > 1:
        raise ProfilerError("stage-attempt root contains multiple in-flight writers")

    recovery_root = output_root / RECOVERY_ROOT_NAME
    if not os.path.lexists(recovery_root):
        retained_intents = any(evidence[4] for evidence in intents.values())
        if retained_intents or prepared_retained:
            raise ProfilerError("retained stage custody lacks its immutable attempt outcome root")
        if terminal and (intents or prepared_active or final_payloads):
            raise ProfilerError("terminal stage custody lacks immutable attempt outcomes")
        active_intents = {key: evidence for key, evidence in intents.items() if not evidence[4]}
        if prepared_active and any(not any(key[0] == frame for key in active_intents) for frame in prepared_active):
            raise ProfilerError("active prepared stage lacks its exact live transaction")
        if final_payloads:
            active_final_frames = {key[0] for key in active_intents}
            if set(final_payloads) != active_final_frames or len(final_payloads) != len(active_intents):
                raise ProfilerError("committed stage lacks its exact live success transaction")
            for (frame, _attempt), (_intent_path, record, _digest, _snapshot, _retained) in active_intents.items():
                final_path, final_payload = final_payloads[frame]
                if (
                    frame in prepared_active
                    or final_path.name != record["final_basename"]
                    or len(final_payload) != record["intended_bytes"]
                    or _sha256_bytes(final_payload) != record["intended_sha256"]
                ):
                    raise ProfilerError("committed stage differs from its exact live success transaction")
        return _ValidatedStageAttemptCustody(outcomes=(), proven_stage_retained_names=frozenset())
    _require_local_directory(recovery_root, name="stage-attempt outcome root")
    root_entries = list(recovery_root.iterdir())
    if not root_entries:
        if terminal or any(evidence[4] for evidence in intents.values()):
            raise ProfilerError("empty stage-attempt outcome root lacks its exact active transaction")
        return _ValidatedStageAttemptCustody(outcomes=(), proven_stage_retained_names=frozenset())
    if len(root_entries) != 1 or root_entries[0].name != identity_hash:
        raise ProfilerError("stage-attempt outcome root contains an unknown identity")
    identity_root = root_entries[0]
    _require_local_directory(identity_root, name="stage-attempt outcome identity root")
    identity_entries = list(identity_root.iterdir())
    if not identity_entries:
        if terminal or any(evidence[4] for evidence in intents.values()):
            raise ProfilerError("empty stage-attempt identity root lacks its exact active transaction")
        return _ValidatedStageAttemptCustody(outcomes=(), proven_stage_retained_names=frozenset())

    outcomes: list[dict[str, Any]] = []
    complete_keys: set[tuple[int, int]] = set()
    success_by_frame: dict[int, int] = {}
    pending_keys: set[tuple[int, int]] = set()
    intent_building: set[tuple[int, int]] = set()
    proven_stage_retained: set[str] = set()
    consumed_retained_prepared: set[str] = set()
    transaction_keys: set[tuple[int, int]] = set()
    for outcome_root in sorted(identity_entries, key=lambda path: path.name):
        _require_local_directory(outcome_root, name=f"stage-attempt outcome {outcome_root.name}")
        match = RECOVERY_TRANSACTION_RE.fullmatch(outcome_root.name)
        if match is None:
            raise ProfilerError("stage-attempt outcome directory name is malformed")
        frame = int(match.group(1))
        attempt = int(match.group(2))
        key = (frame, attempt)
        if key in transaction_keys:
            raise ProfilerError("stage-attempt outcome directory is duplicated")
        transaction_keys.add(key)
        entries = list(outcome_root.iterdir())
        active_names: set[str] = set()
        retained_by_target: dict[str, set[str]] = {
            STAGE_ATTEMPT_NAME: set(),
            STAGE_SUCCESS_NAME: set(),
            RECOVERY_MANIFEST_NAME: set(),
        }
        for entry in entries:
            if feature_cache_module.is_retained_name(entry.name):
                original = feature_cache_module.retained_original_name(entry.name)
                matches = [
                    target for target in retained_by_target if _is_atomic_scratch_original(original, targets=(target,))
                ]
                if len(matches) != 1:
                    raise ProfilerError("stage-attempt outcome contains role-unproven retained bytes")
                retained_by_target[matches[0]].add(entry.name)
            else:
                active_names.add(entry.name)
        success_path = outcome_root / STAGE_SUCCESS_NAME
        manifest_path = outcome_root / RECOVERY_MANIFEST_NAME
        payload_path = outcome_root / RECOVERY_PAYLOAD_NAME
        attempt_path = outcome_root / STAGE_ATTEMPT_NAME
        attempt_scratch = {path.name for path in _active_atomic_scratch(attempt_path)}
        attempt_completion = {path.name for path in _active_atomic_completion(attempt_path)}
        success_scratch = {path.name for path in _active_atomic_scratch(success_path)}
        success_completion = {path.name for path in _active_atomic_completion(success_path)}
        manifest_scratch = {path.name for path in _active_atomic_scratch(manifest_path)}
        manifest_completion = {path.name for path in _active_atomic_completion(manifest_path)}
        allowed_active = {
            STAGE_ATTEMPT_NAME,
            STAGE_SUCCESS_NAME,
            RECOVERY_MANIFEST_NAME,
            RECOVERY_PAYLOAD_NAME,
            *attempt_scratch,
            *attempt_completion,
            *success_scratch,
            *success_completion,
            *manifest_scratch,
            *manifest_completion,
        }
        if not active_names.issubset(allowed_active):
            raise ProfilerError("stage-attempt outcome contains unidentified active bytes")
        if os.path.lexists(attempt_path):
            attempt_record, attempt_sha256, attempt_snapshot = _read_stage_attempt_transaction(
                attempt_path,
                expected_identity_sha256=identity_hash,
                expected_rebuild_argv=rebuild_argv,
            )
            if (attempt_record["frame"], attempt_record["attempt"]) != key:
                raise ProfilerError("stage-attempt record differs from its recovery directory")
            attempt_payload = canonical_json_bytes(attempt_record) + b"\n"
            proven_attempt_atomic = _validate_atomic_retained_names(
                attempt_path,
                desired_payload=attempt_payload,
                name="stage-attempt transaction",
            )
            if proven_attempt_atomic != retained_by_target[STAGE_ATTEMPT_NAME]:
                raise ProfilerError("stage-attempt transaction has orphaned atomic custody")
            if key in intents:
                raise ProfilerError("stage-attempt custody contains duplicate transaction intents")
            intents[key] = (attempt_path, attempt_record, attempt_sha256, attempt_snapshot, False)
        elif attempt_scratch or attempt_completion or retained_by_target[STAGE_ATTEMPT_NAME]:
            if attempt_completion or retained_by_target[STAGE_ATTEMPT_NAME]:
                raise ProfilerError("stage-attempt construction has orphaned authority evidence")
            if any(name not in attempt_scratch for name in active_names):
                raise ProfilerError("stage-attempt construction is mixed with semantic outcome bytes")
            intent_building.add(key)
            pending_keys.add(key)
            continue
        elif not entries:
            intent_building.add(key)
            pending_keys.add(key)
            continue
        intent = intents.get(key)
        if intent is None:
            raise ProfilerError("stage-attempt outcome lacks its exact immutable transaction")
        intent_path, attempt_record, attempt_sha256, _intent_snapshot, intent_retained = intent
        has_success = bool(
            ({STAGE_SUCCESS_NAME} | success_scratch | success_completion) & active_names
            or retained_by_target[STAGE_SUCCESS_NAME]
        )
        has_recovery = bool(
            ({RECOVERY_MANIFEST_NAME, RECOVERY_PAYLOAD_NAME} | manifest_scratch | manifest_completion) & active_names
            or retained_by_target[RECOVERY_MANIFEST_NAME]
        )
        if has_success and has_recovery:
            raise ProfilerError("stage attempt has both success and recovery outcomes")
        if not has_success and not has_recovery:
            non_intent_active = active_names - {
                STAGE_ATTEMPT_NAME,
                *attempt_scratch,
                *attempt_completion,
            }
            non_intent_retained = set().union(
                retained_by_target[STAGE_SUCCESS_NAME],
                retained_by_target[RECOVERY_MANIFEST_NAME],
            )
            if terminal or intent_retained or non_intent_active or non_intent_retained:
                raise ProfilerError("stage-attempt outcome directory is incomplete or orphaned")
            pending_keys.add(key)
            continue

        intended_bytes = attempt_record["intended_bytes"]
        intended_sha256 = attempt_record["intended_sha256"]
        prepared_stage = stages / f".frame_{frame:04d}.bin{PREPARED_STAGE_SUFFIX}"
        if has_success:
            final = final_payloads.get(frame)
            if final is None:
                raise ProfilerError("successful stage attempt lacks its exact committed final")
            final_path, final_payload = final
            if (
                final_path.name != attempt_record["final_basename"]
                or len(final_payload) != intended_bytes
                or _sha256_bytes(final_payload) != intended_sha256
            ):
                raise ProfilerError("successful stage bytes differ from their attempt transaction")
            expected_success = _stage_success_outcome(
                transaction=attempt_record,
                transaction_sha256=attempt_sha256,
                final_payload=final_payload,
            )
            expected_success_payload = canonical_json_bytes(expected_success) + b"\n"
            success_present = os.path.lexists(success_path)
            if (
                success_present
                and _load_canonical_object(
                    success_path,
                    name="stage-attempt success outcome",
                )
                != expected_success
            ):
                raise ProfilerError("stage-attempt success outcome custody mismatch")
            if success_present:
                proven_atomic = _validate_atomic_retained_names(
                    success_path,
                    desired_payload=expected_success_payload,
                    expected_prior_payloads=(expected_success_payload,),
                    name="stage-attempt success outcome",
                )
                if proven_atomic != retained_by_target[STAGE_SUCCESS_NAME]:
                    raise ProfilerError("stage-attempt success has orphaned retained atomic custody")
            elif retained_by_target[STAGE_SUCCESS_NAME]:
                raise ProfilerError("retained success atomic custody lacks its committed target")
            complete = (
                success_present
                and not success_scratch
                and not any(
                    feature_cache_module.ATOMIC_COMPLETION_GENERATION_RE.fullmatch(name) is not None
                    for name in success_completion
                )
            )
            if terminal and not complete:
                raise ProfilerError("terminal successful stage attempt is incomplete")
            if not complete:
                if intent_retained:
                    raise ProfilerError("incomplete success outcome has an already-retained transaction intent")
                pending_keys.add(key)
                continue
            if frame in success_by_frame:
                raise ProfilerError("stage frame has multiple successful attempt outcomes")
            success_by_frame[frame] = attempt
            outcome = {
                "outcome": "success",
                "success": expected_success,
                "success_sha256": _sha256_bytes(expected_success_payload),
            }
        else:
            payload = (
                _read_bound_bytes(payload_path, name="stage-attempt recovery payload").payload
                if os.path.lexists(payload_path)
                else None
            )
            candidate_active_prepared = prepared_active.get(frame) if not intent_retained else None
            if payload is not None:
                actual_payload = payload
            elif candidate_active_prepared is not None:
                actual_payload = candidate_active_prepared[1].payload
            else:
                actual_payload = b""
            if len(actual_payload) >= intended_bytes:
                raise ProfilerError("stage recovery admits only missing or size-short attempts")
            manifest_present = os.path.lexists(manifest_path)
            manifest_value = (
                _load_canonical_object(manifest_path, name="stage-attempt recovery manifest")
                if manifest_present
                else None
            )
            if manifest_value is None:
                active_prepared = candidate_active_prepared
                prepared_source_present = active_prepared is not None
                prepared_source_file_identity = (
                    active_prepared[1].file_identity if active_prepared is not None else None
                )
            else:
                prepared_source_present = manifest_value.get("prepared_source_present")
                stored_file_identity = manifest_value.get("prepared_source_file_identity")
                if type(prepared_source_present) is not bool or (
                    stored_file_identity is not None
                    and (
                        not isinstance(stored_file_identity, list)
                        or len(stored_file_identity) != 5
                        or any(type(value) is not int or value < 0 for value in stored_file_identity)
                    )
                ):
                    raise ProfilerError("recovery manifest prepared-source custody is malformed")
                prepared_source_file_identity = (
                    tuple(stored_file_identity) if stored_file_identity is not None else None
                )
                active_prepared = (
                    candidate_active_prepared
                    if candidate_active_prepared is not None
                    and candidate_active_prepared[1].file_identity == prepared_source_file_identity
                    else None
                )
                if payload is None and active_prepared is None:
                    actual_payload = b""
            expected_manifest = _recovery_manifest(
                original=prepared_stage,
                destination=payload_path,
                actual_payload=actual_payload,
                intended_bytes=intended_bytes,
                intended_sha256=intended_sha256,
                identity_hash=identity_hash,
                frame=frame,
                attempt=attempt,
                stage_attempt_transaction_sha256=attempt_sha256,
                prepared_source_present=prepared_source_present,
                prepared_source_file_identity=prepared_source_file_identity,
                exact_rebuild_argv=rebuild_argv,
            )
            expected_manifest_payload = canonical_json_bytes(expected_manifest) + b"\n"
            if manifest_value is not None and manifest_value != expected_manifest:
                raise ProfilerError("recovery manifest/payload/transaction custody mismatch")
            if manifest_present:
                proven_atomic = _validate_atomic_retained_names(
                    manifest_path,
                    desired_payload=expected_manifest_payload,
                    expected_prior_payloads=(expected_manifest_payload,),
                    name="stage-attempt recovery manifest",
                )
                if proven_atomic != retained_by_target[RECOVERY_MANIFEST_NAME]:
                    raise ProfilerError("stage recovery has orphaned retained atomic custody")
            elif retained_by_target[RECOVERY_MANIFEST_NAME]:
                raise ProfilerError("retained recovery atomic custody lacks its committed target")
            matching_retained_prepared = [
                (path, snapshot)
                for path, snapshot, _ordinal in prepared_retained.get(frame, ())
                if snapshot.file_identity == prepared_source_file_identity
            ]
            if prepared_source_present:
                if active_prepared is not None:
                    if (
                        active_prepared[1].file_identity != prepared_source_file_identity
                        or active_prepared[1].payload != actual_payload
                        or matching_retained_prepared
                    ):
                        raise ProfilerError("active prepared stage differs from its recovery descriptor custody")
                elif len(matching_retained_prepared) != 1 or matching_retained_prepared[0][1].payload != actual_payload:
                    raise ProfilerError("recovery attempt lacks its exact retained prepared-stage descriptor")
                else:
                    consumed_retained_prepared.add(matching_retained_prepared[0][0].name)
            elif active_prepared is not None or matching_retained_prepared:
                raise ProfilerError("missing-source recovery claims unexpected prepared-stage custody")
            complete = (
                manifest_present
                and payload is not None
                and not manifest_scratch
                and not any(
                    feature_cache_module.ATOMIC_COMPLETION_GENERATION_RE.fullmatch(name) is not None
                    for name in manifest_completion
                )
                and active_prepared is None
            )
            if terminal and not complete:
                raise ProfilerError("terminal recovery stage attempt is incomplete")
            if not complete:
                if intent_retained:
                    raise ProfilerError("incomplete recovery outcome has an already-retained transaction intent")
                pending_keys.add(key)
                continue
            outcome = {
                "outcome": "recovery",
                "manifest": expected_manifest,
                "manifest_sha256": _sha256_bytes(expected_manifest_payload),
                "payload_bytes": len(payload),
                "payload_sha256": _sha256_bytes(payload),
            }

        complete_keys.add(key)
        outcomes.append(outcome)
        if intent_retained:
            proven_stage_retained.add(intent_path.name)

    for key, (intent_path, _record, _digest, _snapshot, retained) in intents.items():
        if retained and key not in complete_keys:
            raise ProfilerError("retained stage intent is orphaned from its exact outcome")
        if not retained and key not in complete_keys and key not in pending_keys:
            pending_keys.add(key)
        if retained and intent_path.name not in proven_stage_retained:
            raise ProfilerError("retained stage intent was not consumed by exactly one outcome")
    if len(pending_keys) > 1:
        raise ProfilerError("stage-attempt custody contains multiple incomplete outcomes")
    all_retained_prepared = {path.name for rows in prepared_retained.values() for path, _snapshot, _ordinal in rows}
    if all_retained_prepared != consumed_retained_prepared:
        raise ProfilerError("retained prepared-stage custody is orphaned from its exact recovery outcome")
    proven_stage_retained.update(consumed_retained_prepared)
    for frame in prepared_active:
        if not any(key[0] == frame and not intents[key][4] for key in pending_keys):
            raise ProfilerError("active prepared stage lacks its exact incomplete transaction outcome")
    for frame in {key[0] for key in intents}:
        attempts = sorted(key[1] for key in intents if key[0] == frame)
        if attempts != list(range(len(attempts))):
            raise ProfilerError(f"stage attempts for frame {frame} are not a contiguous durable sequence")
        success_attempt = success_by_frame.get(frame)
        if success_attempt is not None and success_attempt != attempts[-1]:
            raise ProfilerError("successful stage outcome is not the final attempt for its frame")
    for frame in final_payloads:
        if frame not in success_by_frame and not any(key[0] == frame for key in pending_keys):
            raise ProfilerError("committed final stage lacks its exact success attempt outcome")
    if any(frame not in final_payloads for frame in success_by_frame):
        raise ProfilerError("successful attempt outcome has no committed final stage")
    if terminal and pending_keys:
        raise ProfilerError("terminal stage-attempt custody contains an incomplete outcome")
    return _ValidatedStageAttemptCustody(
        outcomes=tuple(outcomes),
        proven_stage_retained_names=frozenset(proven_stage_retained),
        intent_building=tuple(sorted(intent_building)),
    )


def _validate_recovery_grammar(
    output_root: Path,
    *,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
    terminal: bool,
) -> list[dict[str, Any]]:
    """Compatibility view over fully validated failed-attempt outcomes."""

    custody = _validate_stage_attempt_custody(
        output_root,
        identity_sha256=identity_sha256,
        exact_rebuild_argv=exact_rebuild_argv,
        terminal=terminal,
    )
    return [dict(outcome) for outcome in custody.outcomes if outcome["outcome"] == "recovery"]


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    mask[:, 1:] |= horizontal
    mask[:, :-1] |= horizontal
    mask[1:, :] |= vertical
    mask[:-1, :] |= vertical
    return mask


def _frame_strata(live_logits: np.ndarray, labels: np.ndarray, fragile_margin: float) -> tuple[np.ndarray, np.ndarray]:
    sorted_logits = np.partition(live_logits, -2, axis=0)
    margins = sorted_logits[-1] - sorted_logits[-2]
    return _boundary_mask(labels), margins <= fragile_margin


def _stage_payload(receipt: dict[str, Any], candidate_payload: bytes) -> bytes:
    receipt_bytes = canonical_json_bytes(receipt)
    return STAGE_MAGIC + struct.pack("<I", len(receipt_bytes)) + receipt_bytes + candidate_payload


def _parse_stage_payload(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if not payload.startswith(STAGE_MAGIC) or len(payload) < len(STAGE_MAGIC) + 4:
        raise ProfilerError("stage payload header is malformed")
    offset = len(STAGE_MAGIC)
    receipt_bytes = struct.unpack_from("<I", payload, offset)[0]
    start = offset + 4
    stop = start + receipt_bytes
    if stop > len(payload):
        raise ProfilerError("stage receipt length exceeds payload")
    try:
        receipt = json.loads(payload[start:stop])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfilerError("stage receipt is not canonical JSON") from exc
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) != payload[start:stop]:
        raise ProfilerError("stage receipt is not canonical")
    candidate_payload = payload[stop:]
    if receipt.get("candidate_payload_bytes") != len(candidate_payload) or receipt.get(
        "candidate_payload_sha256"
    ) != _sha256_bytes(candidate_payload):
        raise ProfilerError("stage candidate payload custody mismatch")
    return receipt, candidate_payload


def _nonnegative_receipt_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ProfilerError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ProfilerError(f"{name} must be a nonnegative integer")
    return result


def _finite_receipt_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ProfilerError(f"{name} must be finite numeric state")
    result = float(value)
    if not math.isfinite(result):
        raise ProfilerError(f"{name} must be finite numeric state")
    return result


def _stage_timing(*, wall_seconds: float, total_blocks: int, peak_rss_bytes: int) -> dict[str, Any]:
    wall = _finite_receipt_float(wall_seconds, name="stage wall_seconds")
    blocks = _nonnegative_receipt_int(total_blocks, name="stage timing total_blocks")
    rss = _nonnegative_receipt_int(peak_rss_bytes, name="stage timing peak_rss_bytes")
    if wall <= 0 or blocks <= 0:
        raise ProfilerError("stage timing requires positive wall time and block count")
    return {
        "schema": TIMING_SCHEMA,
        "wall_seconds": wall,
        "blocks_per_second": blocks / wall,
        "peak_rss_bytes": rss,
        "custody_label": TIMING_CUSTODY_LABEL,
    }


def _validate_stage_timing(value: object, *, total_blocks: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != TIMING_KEYS or value.get("schema") != TIMING_SCHEMA:
        raise ProfilerError("stage timing schema is malformed")
    wall = _finite_receipt_float(value.get("wall_seconds"), name="stage wall_seconds")
    rate = _finite_receipt_float(value.get("blocks_per_second"), name="stage blocks_per_second")
    blocks = _nonnegative_receipt_int(total_blocks, name="stage timing total_blocks")
    rss = _nonnegative_receipt_int(value.get("peak_rss_bytes"), name="stage peak_rss_bytes")
    if wall <= 0 or rate <= 0 or blocks <= 0 or rate != blocks / wall:
        raise ProfilerError("stage timing rate relation is malformed")
    if value.get("custody_label") != TIMING_CUSTODY_LABEL:
        raise ProfilerError("stage timing custody label is malformed")
    return {
        "schema": TIMING_SCHEMA,
        "wall_seconds": wall,
        "blocks_per_second": rate,
        "peak_rss_bytes": rss,
        "custody_label": TIMING_CUSTODY_LABEL,
    }


def _timing_summary(
    receipts: Iterable[Mapping[str, Any]],
    *,
    terminal_stage_chain_sha256: str,
) -> dict[str, Any]:
    terminal = _canonical_sha256(terminal_stage_chain_sha256, name="timing terminal stage root")
    rows: list[dict[str, Any]] = []
    total_blocks = 0
    for expected_frame, receipt in enumerate(receipts):
        if type(receipt.get("frame")) is not int or receipt.get("frame") != expected_frame:
            raise ProfilerError("timing summary frame order is malformed")
        counters = receipt.get("counters")
        if not isinstance(counters, Mapping):
            raise ProfilerError("timing summary counters are malformed")
        blocks = _nonnegative_receipt_int(counters.get("total_blocks"), name="timing frame blocks")
        timing = _validate_stage_timing(receipt.get("timing"), total_blocks=blocks)
        rows.append({"frame": expected_frame, **timing})
        total_blocks += blocks
    if not rows or total_blocks <= 0:
        raise ProfilerError("timing summary requires at least one measured frame")
    total_wall = math.fsum(row["wall_seconds"] for row in rows)
    return {
        "schema": TIMING_SUMMARY_SCHEMA,
        "custody_label": TIMING_CUSTODY_LABEL,
        "semantically_replayable": False,
        "frame_count": len(rows),
        "terminal_stage_chain_sha256": terminal,
        "total_wall_seconds": total_wall,
        "total_blocks": total_blocks,
        "aggregate_blocks_per_second": total_blocks / total_wall,
        "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "frames": rows,
    }


def _float_state_close(left: float, right: float) -> bool:
    """Narrow deterministic allowance for alternate float64 summation order."""

    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def _compact_bin_interval(index: int, width: float) -> tuple[float, float]:
    lower = index * width
    # Four uint8 values admit at most 256**4 points.  Bin 128 is therefore
    # clipped to the exact log2(K)=32 ceiling, not the nominal [32, 32.25).
    upper = 32.0 if index == 128 else lower + width
    return lower, upper


def _inhabits_compact_bin(value: float, index: int, width: float) -> bool:
    lower, upper = _compact_bin_interval(index, width)
    if index == 128:
        return value == lower
    return lower <= value < upper


def _validate_compact_state(state: object, *, name: str, expected_count: int) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != COMPACT_STATE_KEYS:
        raise ProfilerError(f"{name} compact state schema is malformed")
    width = _finite_receipt_float(state["bin_width"], name=f"{name} bin_width")
    if width != 0.25:
        raise ProfilerError(f"{name} compact histogram geometry drift")
    bins_value = state["bins"]
    if not isinstance(bins_value, list) or len(bins_value) != 129:
        raise ProfilerError(f"{name} compact histogram geometry drift")
    bins = [
        _nonnegative_receipt_int(value, name=f"{name} histogram bin {index}") for index, value in enumerate(bins_value)
    ]
    count = _nonnegative_receipt_int(state["count"], name=f"{name} count")
    zero_count = _nonnegative_receipt_int(state["zero_count"], name=f"{name} zero_count")
    if count != expected_count or zero_count > count or sum(bins) != count - zero_count:
        raise ProfilerError(f"{name} compact count/histogram accounting mismatch")
    total = _finite_receipt_float(state["total"], name=f"{name} total")
    if total < 0:
        raise ProfilerError(f"{name} compact total must be nonnegative")
    minimum_raw = state["minimum"]
    maximum_raw = state["maximum"]
    nonzero = count - zero_count
    if nonzero == 0:
        if minimum_raw is not None or maximum_raw is not None or total != 0.0:
            raise ProfilerError(f"{name} empty compact extrema are inconsistent")
        minimum = maximum = None
    else:
        minimum = _finite_receipt_float(minimum_raw, name=f"{name} minimum")
        maximum = _finite_receipt_float(maximum_raw, name=f"{name} maximum")
        occupied = [index for index, value in enumerate(bins) if value]
        first = occupied[0]
        last = occupied[-1]
        if (
            minimum < 0
            or maximum < minimum
            or maximum > 32.0
            or not _inhabits_compact_bin(minimum, first, width)
            or not _inhabits_compact_bin(maximum, last, width)
        ):
            raise ProfilerError(f"{name} compact extrema are inconsistent")
        if minimum == maximum:
            lower_total = upper_total = nonzero * minimum
        else:
            lower_total = 0.0
            upper_total = 0.0
            for index, bin_count in enumerate(bins):
                bin_lower, bin_upper = _compact_bin_interval(index, width)
                lower_total += bin_count * bin_lower
                upper_total += bin_count * bin_upper
            first_lower, first_upper = _compact_bin_interval(first, width)
            last_lower, last_upper = _compact_bin_interval(last, width)
            # min and max are distinct observations when minimum < maximum,
            # including when both occupy the same histogram bin.
            lower_total += (minimum - first_lower) + (maximum - last_lower)
            upper_total += (minimum - first_upper) + (maximum - last_upper)
        if (total < lower_total and not _float_state_close(total, lower_total)) or (
            total > upper_total and not _float_state_close(total, upper_total)
        ):
            raise ProfilerError(f"{name} compact total/extrema are inconsistent")
    return {
        "count": count,
        "zero_count": zero_count,
        "total": total,
        "minimum": minimum,
        "maximum": maximum,
        "bins": bins,
    }


def _validate_bucket_state(state: object, *, name: str) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != BUCKET_STATE_KEYS:
        raise ProfilerError(f"{name} bucket schema is malformed")
    counts = {
        field: _nonnegative_receipt_int(state[field], name=f"{name} {field}")
        for field in ("scorer_pixels", "channel_blocks", "exact_blocks", "bounded_blocks")
    }
    if counts["exact_blocks"] + counts["bounded_blocks"] != counts["channel_blocks"]:
        raise ProfilerError(f"{name} exact/bounded channel accounting mismatch")
    if counts["channel_blocks"] != 3 * counts["scorer_pixels"]:
        raise ProfilerError(f"{name} RGB-channel/scorer-pixel accounting mismatch")
    lower = _validate_compact_state(
        state["lower"],
        name=f"{name} lower",
        expected_count=counts["channel_blocks"],
    )
    upper = _validate_compact_state(
        state["upper"],
        name=f"{name} upper",
        expected_count=counts["channel_blocks"],
    )
    if lower["zero_count"] < upper["zero_count"] or (
        lower["total"] > upper["total"] and not _float_state_close(lower["total"], upper["total"])
    ):
        raise ProfilerError(f"{name} lower/upper compact stats are inconsistent")
    for bound in ("minimum", "maximum"):
        lower_value = lower[bound]
        upper_value = upper[bound]
        if lower_value is not None and upper_value is not None and lower_value > upper_value:
            raise ProfilerError(f"{name} lower/upper {bound} is inconsistent")
    return {**counts, "lower": lower, "upper": upper}


def _validate_aggregate_state(state: object) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != {"n_classes", "global", "per_class", "strata"}:
        raise ProfilerError("stage aggregate state schema is malformed")
    if _nonnegative_receipt_int(state["n_classes"], name="aggregate n_classes") != EXPECTED_CLASSES:
        raise ProfilerError("stage aggregate class geometry drift")
    per_class_raw = state["per_class"]
    strata_raw = state["strata"]
    if not isinstance(per_class_raw, dict) or set(per_class_raw) != set(CANONICAL_CLASS_KEYS):
        raise ProfilerError("stage aggregate class key set is noncanonical")
    if not isinstance(strata_raw, dict) or set(strata_raw) != set(CANONICAL_STRATA):
        raise ProfilerError("stage aggregate named-stratum key set is noncanonical")
    global_bucket = _validate_bucket_state(state["global"], name="aggregate global")
    classes = [
        _validate_bucket_state(per_class_raw[key], name=f"aggregate class {key}") for key in CANONICAL_CLASS_KEYS
    ]
    strata = {key: _validate_bucket_state(strata_raw[key], name=f"aggregate stratum {key}") for key in CANONICAL_STRATA}
    for field in ("scorer_pixels", "channel_blocks", "exact_blocks", "bounded_blocks"):
        if sum(bucket[field] for bucket in classes) != global_bucket[field]:
            raise ProfilerError(f"per-class {field} does not partition global")
    for stats_name in ("lower", "upper"):
        global_stats = global_bucket[stats_name]
        for field in ("count", "zero_count"):
            if sum(bucket[stats_name][field] for bucket in classes) != global_stats[field]:
                raise ProfilerError(f"per-class {stats_name} {field} does not partition global")
        for index, count in enumerate(global_stats["bins"]):
            if sum(bucket[stats_name]["bins"][index] for bucket in classes) != count:
                raise ProfilerError(f"per-class {stats_name} histogram does not partition global")
        if not _float_state_close(
            math.fsum(bucket[stats_name]["total"] for bucket in classes),
            global_stats["total"],
        ):
            raise ProfilerError(f"per-class {stats_name} totals do not partition global")
        nonempty = [bucket[stats_name] for bucket in classes if bucket[stats_name]["minimum"] is not None]
        expected_min = None if not nonempty else min(bucket["minimum"] for bucket in nonempty)
        expected_max = None if not nonempty else max(bucket["maximum"] for bucket in nonempty)
        if expected_min != global_stats["minimum"] or expected_max != global_stats["maximum"]:
            raise ProfilerError(f"per-class {stats_name} extrema do not partition global")
    for stratum_name, bucket in strata.items():
        for field in ("scorer_pixels", "channel_blocks", "exact_blocks", "bounded_blocks"):
            if bucket[field] > global_bucket[field]:
                raise ProfilerError(f"stratum {stratum_name} {field} exceeds global")
        for stats_name in ("lower", "upper"):
            stats = bucket[stats_name]
            global_stats = global_bucket[stats_name]
            if stats["count"] > global_stats["count"] or stats["zero_count"] > global_stats["zero_count"]:
                raise ProfilerError(f"stratum {stratum_name} {stats_name} counts exceed global")
            if any(left > right for left, right in zip(stats["bins"], global_stats["bins"], strict=True)):
                raise ProfilerError(f"stratum {stratum_name} {stats_name} histogram exceeds global")
            if stats["total"] > global_stats["total"] and not _float_state_close(stats["total"], global_stats["total"]):
                raise ProfilerError(f"stratum {stratum_name} {stats_name} total exceeds global")
            if stats["minimum"] is not None and (
                global_stats["minimum"] is None
                or stats["minimum"] < global_stats["minimum"]
                or stats["maximum"] > global_stats["maximum"]
            ):
                raise ProfilerError(f"stratum {stratum_name} {stats_name} extrema exceed global")
    return global_bucket


def _canonical_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in LOWER_SHA256_CHARS for character in value)
    ):
        raise ProfilerError(f"{name} must be a lowercase SHA-256")
    return value


def _packed_bool_row(mask: np.ndarray) -> dict[str, Any]:
    value = np.asarray(mask)
    if value.dtype != np.bool_ or value.ndim != 2:
        raise ProfilerError("partition masks must be two-dimensional boolean arrays")
    packed = np.packbits(np.ascontiguousarray(value).reshape(-1), bitorder="little")
    return {
        "dtype": "packed-bool-little-bitorder",
        "shape": list(value.shape),
        "sha256": _sha256_bytes(packed.tobytes()),
        "true_count": int(np.count_nonzero(value)),
    }


def _degenerate_partition_mask(operator: DisjointResizeOperator) -> np.ndarray:
    row_sizes = {len(support.numerators) for support in operator.row_supports}
    column_sizes = {len(support.numerators) for support in operator.col_supports}
    if len(row_sizes) == 1 and len(column_sizes) == 1:
        rows = np.asarray([support.numerators for support in operator.row_supports], dtype=np.int64)
        columns = np.asarray([support.numerators for support in operator.col_supports], dtype=np.int64)
        coefficients = (rows[:, None, :, None] * columns[None, :, None, :]).reshape(
            operator.scorer_h,
            operator.scorer_w,
            -1,
        )
        return (coefficients.shape[-1] < 4) | (np.gcd.reduce(coefficients, axis=-1) > 1)
    mask = np.zeros((operator.scorer_h, operator.scorer_w), dtype=bool)
    for row, row_support in enumerate(operator.row_supports):
        for column, column_support in enumerate(operator.col_supports):
            coefficients = np.outer(row_support.numerators, column_support.numerators).reshape(-1)
            mask[row, column] = coefficients.size < 4 or int(np.gcd.reduce(coefficients)) > 1
    return mask


def _derive_partition_custody(
    frame: int,
    *,
    live_logits: np.ndarray,
    source_frame: np.ndarray,
    operator: DisjointResizeOperator,
    fragile_margin: float,
    degenerate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Re-derive semantic stage custody from immutable production inputs."""

    frame = _nonnegative_receipt_int(frame, name="partition frame")
    logits = np.asarray(live_logits)
    source = np.asarray(source_frame)
    if logits.dtype != np.float32 or logits.shape != (EXPECTED_CLASSES, operator.scorer_h, operator.scorer_w):
        raise ProfilerError("partition live-logit geometry/dtype drift")
    if source.dtype != np.uint8 or source.shape != (operator.camera_h, operator.camera_w, 3):
        raise ProfilerError("partition source-frame geometry/dtype drift")
    if not np.isfinite(logits).all():
        raise ProfilerError("partition live logits contain non-finite values")
    labels = np.argmax(logits, axis=0).astype(np.uint8)
    boundary, fragile = _frame_strata(logits, labels, fragile_margin)
    degenerate = _degenerate_partition_mask(operator) if degenerate_mask is None else np.asarray(degenerate_mask)
    if degenerate.dtype != np.bool_ or degenerate.shape != labels.shape:
        raise ProfilerError("partition degenerate-mask geometry/dtype drift")
    masks = {
        "boundary_annulus": boundary,
        "degenerate": degenerate,
        "fragile": fragile,
    }
    return {
        "schema": PARTITION_CUSTODY_SCHEMA,
        "frame": frame,
        "class_labels": {
            "dtype": "uint8",
            "shape": list(labels.shape),
            "sha256": _sha256_bytes(np.ascontiguousarray(labels).tobytes()),
            "counts": {key: int(np.count_nonzero(labels == int(key))) for key in CANONICAL_CLASS_KEYS},
        },
        "masks": {key: _packed_bool_row(masks[key]) for key in CANONICAL_STRATA},
    }


def _validate_partition_custody(value: object, *, frame: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "frame", "class_labels", "masks"}:
        raise ProfilerError("partition custody schema is malformed")
    if (
        value.get("schema") != PARTITION_CUSTODY_SCHEMA
        or type(value.get("frame")) is not int
        or value["frame"] != frame
    ):
        raise ProfilerError("partition custody frame/schema mismatch")
    labels = value.get("class_labels")
    if not isinstance(labels, dict) or set(labels) != {"dtype", "shape", "sha256", "counts"}:
        raise ProfilerError("partition class-label custody is malformed")
    shape = labels.get("shape")
    counts = labels.get("counts")
    if (
        labels.get("dtype") != "uint8"
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        or not isinstance(counts, dict)
        or set(counts) != set(CANONICAL_CLASS_KEYS)
    ):
        raise ProfilerError("partition class-label geometry/count schema is malformed")
    normalized_counts = {
        key: _nonnegative_receipt_int(counts[key], name=f"partition class {key} count") for key in CANONICAL_CLASS_KEYS
    }
    if sum(normalized_counts.values()) != math.prod(shape):
        raise ProfilerError("partition class counts do not cover label geometry")
    _canonical_sha256(labels.get("sha256"), name="partition class-label hash")
    masks = value.get("masks")
    if not isinstance(masks, dict) or set(masks) != set(CANONICAL_STRATA):
        raise ProfilerError("partition mask key set is noncanonical")
    for key in CANONICAL_STRATA:
        row = masks[key]
        if (
            not isinstance(row, dict)
            or set(row) != PARTITION_MASK_KEYS
            or row.get("dtype") != "packed-bool-little-bitorder"
            or row.get("shape") != shape
        ):
            raise ProfilerError(f"partition {key} mask geometry is malformed")
        _canonical_sha256(row.get("sha256"), name=f"partition {key} mask hash")
        true_count = _nonnegative_receipt_int(row.get("true_count"), name=f"partition {key} true_count")
        if true_count > math.prod(shape):
            raise ProfilerError(f"partition {key} true_count exceeds geometry")
    canonical_json_bytes(value)
    return value


def _validate_stage_receipt(
    receipt: dict[str, Any],
    candidate_payload: bytes,
    *,
    expected_partition_custody: Mapping[str, Any],
    expected_aggregate_state: Mapping[str, Any],
    expected_candidate_payload: bytes,
    expected_counters: Mapping[str, int],
    expected_selection_custody: Mapping[str, Any],
) -> StreamingProfileAggregator:
    """Validate one immutable stage's semantic and byte-accounting invariants."""

    if receipt.get("schema") != STAGE_RECEIPT_SCHEMA:
        raise ProfilerError("stage receipt schema mismatch")
    mode = receipt.get("mode")
    if mode not in (BOUNDS_MODE, ENUMERATED_MODE):
        raise ProfilerError("stage receipt mode is not a typed profiler mode")
    frame = _nonnegative_receipt_int(receipt.get("frame"), name="stage frame")
    stored_partition = _validate_partition_custody(receipt.get("partition_custody"), frame=frame)
    expected_partition = _validate_partition_custody(dict(expected_partition_custody), frame=frame)
    if stored_partition != expected_partition:
        raise ProfilerError("stage partition custody does not match immutable inputs")
    scope = receipt.get("scope")
    if not isinstance(scope, dict) or scope.get("frame_indices") != [frame]:
        raise ProfilerError("stage scope frame indices are malformed")
    scope_blocks = _nonnegative_receipt_int(
        scope.get("rgb_channel_blocks"),
        name="scope rgb_channel_blocks",
    )
    scope_pixels = _nonnegative_receipt_int(
        scope.get("scorer_pixels"),
        name="scope scorer_pixels",
    )

    counters = receipt.get("counters")
    if not isinstance(counters, dict) or set(counters) != set(COUNTER_NAMES):
        raise ProfilerError("stage counters are malformed")
    normalized_counters = {
        name: _nonnegative_receipt_int(counters[name], name=f"counter {name}") for name in COUNTER_NAMES
    }
    if not isinstance(expected_counters, Mapping) or set(expected_counters) != set(COUNTER_NAMES):
        raise ProfilerError("expected semantic replay counters are malformed")
    normalized_expected_counters = {
        name: _nonnegative_receipt_int(expected_counters[name], name=f"expected counter {name}")
        for name in COUNTER_NAMES
    }
    if normalized_counters != normalized_expected_counters:
        raise ProfilerError("stage counters do not match immutable-input semantic replay")
    selected = normalized_counters["selected_blocks"]
    total = normalized_counters["total_blocks"]
    exhaustive_selected = normalized_counters["exhaustive_selected_blocks"]
    bounded_selected = normalized_counters["bounded_selected_blocks"]
    omitted = normalized_counters["omitted_blocks"]
    if selected != exhaustive_selected + bounded_selected or selected + omitted != total:
        raise ProfilerError("stage selection/counter arithmetic mismatch")
    _validate_stage_timing(receipt.get("timing"), total_blocks=total)

    aggregate_state = receipt.get("aggregate_delta_state")
    aggregate_counts = _validate_aggregate_state(aggregate_state)
    expected_aggregate = dict(expected_aggregate_state)
    _validate_aggregate_state(expected_aggregate)
    if aggregate_state != expected_aggregate:
        raise ProfilerError("stage semantic aggregate does not match immutable-input replay")
    per_class_state = aggregate_state["per_class"]
    strata_state = aggregate_state["strata"]
    label_counts = expected_partition["class_labels"]["counts"]
    for key in CANONICAL_CLASS_KEYS:
        if per_class_state[key]["scorer_pixels"] != label_counts[key]:
            raise ProfilerError(f"stage class {key} scorer-pixel count mismatches partition custody")
    for key in CANONICAL_STRATA:
        if strata_state[key]["scorer_pixels"] != expected_partition["masks"][key]["true_count"]:
            raise ProfilerError(f"stage stratum {key} scorer-pixel count mismatches partition custody")
    if (
        aggregate_counts["channel_blocks"] != total
        or aggregate_counts["channel_blocks"] != scope_blocks
        or aggregate_counts["scorer_pixels"] != scope_pixels
        or aggregate_counts["exact_blocks"] + aggregate_counts["bounded_blocks"] != total
    ):
        raise ProfilerError("stage counter/aggregate/scope mismatch")
    if exhaustive_selected > aggregate_counts["exact_blocks"] or bounded_selected > aggregate_counts["bounded_blocks"]:
        raise ProfilerError("stage selected exactness accounting mismatch")

    mismatches = normalized_counters["segnet_mismatches"]
    scorer_pixels = normalized_counters["segnet_pixels"]
    if mismatches > scorer_pixels or scorer_pixels not in (0, scope_pixels):
        raise ProfilerError("stage scorer mismatch/pixel arithmetic is invalid")
    if scorer_pixels and selected != total:
        raise ProfilerError("stage scorer custody requires a fully selected frame")

    payload_bytes = _nonnegative_receipt_int(
        receipt.get("candidate_payload_bytes"),
        name="candidate payload bytes",
    )
    if payload_bytes != len(candidate_payload) or receipt.get("candidate_payload_sha256") != _sha256_bytes(
        candidate_payload
    ):
        raise ProfilerError("stage candidate payload custody mismatch")
    if not isinstance(expected_candidate_payload, bytes) or candidate_payload != expected_candidate_payload:
        raise ProfilerError("stage candidate payload does not match immutable-input semantic replay")
    selection_custody = receipt.get("selection_custody")
    expected_selection = dict(expected_selection_custody)
    for name, value in (("stored", selection_custody), ("expected", expected_selection)):
        if not isinstance(value, dict) or set(value) != SELECTION_CUSTODY_KEYS:
            raise ProfilerError(f"{name} selection custody schema is malformed")
        if any(type(value[key]) is not bool for key in SELECTION_CUSTODY_BOOL_KEYS):
            raise ProfilerError(f"{name} selection custody boolean fields are malformed")
        if not isinstance(value["selection_label"], str) or not value["selection_label"]:
            raise ProfilerError(f"{name} selection custody label is malformed")
        if value["scope_extrapolation"] != "NONE_EXACT_FRAME_INDICES_ONLY":
            raise ProfilerError(f"{name} selection custody extrapolation is malformed")
        if value["pose_bank_wired"] or value["factor10_solved"]:
            raise ProfilerError(f"{name} selection custody falsely claims pose/factor10 authority")
        if value["global_compressed_stream_minimum_claim"] or value["min_description_claim"]:
            raise ProfilerError(f"{name} selection custody falsely claims global compressed-stream optimality")
    if selection_custody != expected_selection:
        raise ProfilerError("stage selection custody does not match immutable-input semantic replay")
    expected_derivation = _expected_derivation(
        mode=mode,
        seed_source_witness=selection_custody["seed_source_witness"],
    )
    if receipt.get("derivation") != expected_derivation:
        raise ProfilerError("stage derivation does not match semantic replay")
    expected_lower_method = LOWER_BOUND_METHOD if mode == BOUNDS_MODE else None
    if receipt.get("lower_bound_method") != expected_lower_method:
        raise ProfilerError("stage lower-bound method does not match profiler mode")
    if mode == BOUNDS_MODE and (
        selected != 0
        or exhaustive_selected != 0
        or bounded_selected != 0
        or aggregate_counts["exact_blocks"] != 0
        or aggregate_counts["bounded_blocks"] != total
        or omitted != total
        or candidate_payload
        or scorer_pixels != 0
        or mismatches != 0
    ):
        raise ProfilerError("bounds-only stage violates mode-specific invariants")

    try:
        aggregate = StreamingProfileAggregator.from_state(aggregate_state)
    except (KeyError, TypeError, ValueError, OverflowError, LatticeProfileError) as exc:
        raise ProfilerError("stage aggregate receipt is not reconstructable") from exc
    summary = aggregate.summary()["global"]
    if (
        summary["rgb_channel_blocks"] != total
        or summary["scorer_pixels"] != scope_pixels
        or summary["exact_blocks"] != aggregate_counts["exact_blocks"]
        or summary["bounded_blocks"] != aggregate_counts["bounded_blocks"]
    ):
        raise ProfilerError("stage reconstructed aggregate summary mismatch")
    return aggregate


def _initialize_fresh_output(
    output_root: Path,
    *,
    identity_hash: str,
    storage_preflight: dict[str, Any],
    exact_argv: list[str],
    identity: Mapping[str, Any] | None = None,
    allow_uncertified_test_output: bool = False,
) -> tuple[Path, Path]:
    """Create a frame-zero pointer through stable certified staging."""

    if type(allow_uncertified_test_output) is not bool:
        raise ProfilerError("allow_uncertified_test_output must be boolean")
    output_root = output_root.expanduser().resolve()
    _canonical_sha256(identity_hash, name="fresh identity hash")
    rebuild_argv = _normalized_argv(exact_argv, name="fresh rebuild argv")
    normalized_preflight = json.loads(canonical_json_bytes(dict(storage_preflight)))

    def progress_for(creation_preflight: Mapping[str, Any]) -> dict[str, Any]:
        value = {
            "schema": PROGRESS_SCHEMA,
            "identity_sha256": identity_hash,
            "status": "partial",
            "next_frame": 0,
            "stage_chain_head_sha256": identity_hash,
            "storage_preflight": json.loads(canonical_json_bytes(dict(creation_preflight))),
            "exact_argv": rebuild_argv,
        }
        _validate_progress_pointer(value, identity_hash=identity_hash, max_frames=EXPECTED_PAIRS)
        return value

    progress = progress_for(normalized_preflight)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if allow_uncertified_test_output:
        if identity is not None:
            raise ProfilerError("uncertified test initialization cannot accept a production identity")
        if output_root.exists() and any(output_root.iterdir()):
            raise ProfilerError("refusing to overwrite existing profiler output; pass --resume")
        output_root.mkdir(exist_ok=True)
        progress_path = output_root / PROGRESS_NAME
        atomic_json(progress_path, progress)
        stages = output_root / "stages"
        stages.mkdir()
        return stages, progress_path

    if identity is None:
        raise ProfilerError("certified profile initialization requires the full identity")
    normalized_identity = json.loads(canonical_json_bytes(dict(identity)))
    if _sha256_bytes(canonical_json_bytes(normalized_identity)) != identity_hash:
        raise ProfilerError("fresh profile identity object/hash mismatch")
    _validate_storage_preflight(normalized_preflight, expected_output_root=output_root)
    bound_storage_identity = _validate_creation_storage_identity(
        normalized_identity.get("creation_storage_identity"),
        expected_output_root=output_root,
    )
    if (
        _creation_storage_identity_from_preflight(
            normalized_preflight,
            expected_output_root=output_root,
        )
        != bound_storage_identity
    ):
        raise ProfilerError("fresh profile preflight differs from identity-bound storage custody")
    if os.path.lexists(output_root):
        _require_local_directory(output_root, name="fresh output root")
        raise ProfilerError("refusing an existing profiler output path; pass --resume")
    staging_root = _creation_staging_path(output_root)
    scratch = _staging_scratch_record(
        output_root=output_root,
        identity_sha256=identity_hash,
        exact_rebuild_argv=rebuild_argv,
        storage_preflight=normalized_preflight,
    )
    if os.path.lexists(staging_root):
        _require_local_directory(staging_root, name="profile creation staging")
        persisted_scratch = _load_complete_staging_scratch(staging_root)
        if persisted_scratch is not None:
            if not _staging_scratch_records_match(persisted_scratch, scratch):
                raise ProfilerError(f"refusing identity-drifted prepared profile staging: {staging_root}")
            persisted_preflight = persisted_scratch.get("storage_preflight")
            if not isinstance(persisted_preflight, dict):
                raise ProfilerError("prepared profile staging lacks creation storage custody")
            _validate_storage_preflight(
                persisted_preflight,
                expected_output_root=output_root,
            )
            if (
                _creation_storage_identity_from_preflight(
                    persisted_preflight,
                    expected_output_root=output_root,
                )
                != bound_storage_identity
            ):
                raise ProfilerError("prepared profile staging differs from identity-bound storage")
            # Preserve the first complete preflight observations.  The live
            # current-capacity gate already ran before this adoption.
            scratch = persisted_scratch

    def creation_preflight_from_scratch(root: Path) -> dict[str, Any]:
        record = _load_canonical_object(root / STAGING_SCRATCH_NAME, name="staging scratch")
        stored_preflight = record.get("storage_preflight")
        if not isinstance(stored_preflight, dict):
            raise ProfilerError("certified creation scratch lacks storage preflight")
        _validate_storage_preflight(stored_preflight, expected_output_root=output_root)
        if (
            _creation_storage_identity_from_preflight(
                stored_preflight,
                expected_output_root=output_root,
            )
            != bound_storage_identity
        ):
            raise ProfilerError("certified creation scratch differs from identity-bound storage")
        return stored_preflight

    def creation_retained_expectations(creation_preflight: Mapping[str, Any]) -> dict[str, bytes]:
        initial_progress = progress_for(creation_preflight)
        identity_payload = canonical_json_bytes(normalized_identity) + b"\n"
        certification = _output_certification_record(
            output_root=output_root,
            identity_sha256=identity_hash,
            identity_json_sha256=_sha256_bytes(identity_payload),
            exact_rebuild_argv=rebuild_argv,
            storage_preflight=creation_preflight,
        )
        values = {
            STAGING_SCRATCH_NAME: scratch,
            IDENTITY_NAME: normalized_identity,
            PROGRESS_NAME: initial_progress,
            OUTPUT_CERTIFICATION_NAME: certification,
        }
        return {
            _creation_prepared_path(Path(target)).name: canonical_json_bytes(value) + b"\n"
            for target, value in values.items()
        }

    def validate_initial_root(root: Path) -> None:
        _require_local_directory(root, name="certified creation staging")
        _validate_output_certification(
            root,
            final_output_root=output_root,
            expected_identity=normalized_identity,
            expected_rebuild_argv=rebuild_argv,
        )
        allowed = {
            STAGING_SCRATCH_NAME,
            IDENTITY_NAME,
            PROGRESS_NAME,
            OUTPUT_CERTIFICATION_NAME,
            "stages",
        }
        root_entries = list(root.iterdir())
        retained = _validate_creation_retained_names(
            root_entries,
            name="certified creation staging",
            expected_payloads_by_original=creation_retained_expectations(creation_preflight_from_scratch(root)),
        )
        if {path.name for path in root_entries} - retained != allowed:
            raise ProfilerError("certified creation staging contains unidentified bytes")
        stages_path = root / "stages"
        _require_local_directory(stages_path, name="certified creation stage root")
        if any(stages_path.iterdir()):
            raise ProfilerError("certified creation staging is not at the empty initial prefix")
        stored_progress = _load_canonical_object(root / PROGRESS_NAME, name="initial progress")
        _validate_progress_pointer(stored_progress, identity_hash=identity_hash, max_frames=EXPECTED_PAIRS)
        if stored_progress != progress_for(creation_preflight_from_scratch(root)):
            raise ProfilerError("certified creation staging progress drift")

    def finish_certified_staging(root: Path) -> None:
        target_names = (IDENTITY_NAME, PROGRESS_NAME, OUTPUT_CERTIFICATION_NAME)
        allowed_during_creation = {
            STAGING_SCRATCH_NAME,
            "stages",
            *target_names,
            *(_creation_prepared_path(root / name).name for name in target_names),
        }
        root_entries = list(root.iterdir())
        retained = _validate_creation_retained_names(
            root_entries,
            name="certified creation staging",
            expected_payloads_by_original=creation_retained_expectations(creation_preflight_from_scratch(root)),
        )
        if any(path.name not in allowed_during_creation and path.name not in retained for path in root_entries):
            raise ProfilerError("certified creation staging contains unidentified bytes")
        persisted_creation_preflight = creation_preflight_from_scratch(root)
        _materialize_creation_json(root / IDENTITY_NAME, normalized_identity)
        _materialize_creation_json(
            root / PROGRESS_NAME,
            progress_for(persisted_creation_preflight),
        )
        stages_path = root / "stages"
        if os.path.lexists(stages_path):
            _require_local_directory(stages_path, name="certified creation stage root")
            if any(stages_path.iterdir()):
                raise ProfilerError("certified creation staging contains committed stage bytes")
        else:
            stages_path.mkdir()
            _fsync_stage_directory(root)
        certification = _output_certification_record(
            output_root=output_root,
            identity_sha256=identity_hash,
            identity_json_sha256=sha256_file(root / IDENTITY_NAME),
            exact_rebuild_argv=rebuild_argv,
            storage_preflight=persisted_creation_preflight,
        )
        _materialize_creation_json(root / OUTPUT_CERTIFICATION_NAME, certification)
        validate_initial_root(root)

    if os.path.lexists(staging_root):
        _require_local_directory(staging_root, name="profile creation staging")
        if any(staging_root.iterdir()):
            scratch_path = staging_root / STAGING_SCRATCH_NAME
            scratch_prepared = _creation_prepared_path(scratch_path)
            staging_entries = list(staging_root.iterdir())
            names = {path.name for path in staging_entries}
            retained = _validate_creation_retained_names(
                staging_entries,
                name="profile creation staging",
                expected_payloads_by_original=creation_retained_expectations(
                    scratch["storage_preflight"]
                    if isinstance(scratch.get("storage_preflight"), dict)
                    else normalized_preflight
                ),
            )
            active_names = names - retained
            if not active_names:
                # A cut after lossless retirement of an incomplete first
                # scratch but before its deterministic rebuild is reachable.
                _materialize_creation_json(scratch_path, scratch)
            elif active_names == {scratch_prepared.name}:
                # The first record is small deterministic metadata, so a
                # partial stable prepared write is safe to rebuild in place.
                _materialize_creation_json(scratch_path, scratch)
            if not _certified_staging_matches(staging_root, expected_record=scratch):
                raise ProfilerError(f"refusing unidentified or identity-drifted profile staging: {staging_root}")
            finish_certified_staging(staging_root)
            _fsync_stage_directory(staging_root)
            _move_directory_noreplace(
                staging_root,
                output_root,
                name="certified profile staging finalization",
            )
            _fsync_stage_directory(output_root.parent)
            return output_root / "stages", output_root / PROGRESS_NAME
    else:
        staging_root.mkdir()
    # Scratch certification is first, so later bytes can never become
    # unidentified rebuildable bulk after an interrupted creation.
    _materialize_creation_json(staging_root / STAGING_SCRATCH_NAME, scratch)
    finish_certified_staging(staging_root)
    _fsync_stage_directory(staging_root)
    _move_directory_noreplace(
        staging_root,
        output_root,
        name="certified profile staging finalization",
    )
    _fsync_stage_directory(output_root.parent)
    return output_root / "stages", output_root / PROGRESS_NAME


def _validate_progress_pointer(
    value: object,
    *,
    identity_hash: str,
    max_frames: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PROGRESS_KEYS:
        raise ProfilerError("progress must use the exact pointer schema")
    _canonical_sha256(identity_hash, name="expected identity hash")
    _canonical_sha256(value.get("identity_sha256"), name="progress identity hash")
    _canonical_sha256(value.get("stage_chain_head_sha256"), name="progress stage-chain hash")
    next_frame = value.get("next_frame")
    if (
        value.get("schema") != PROGRESS_SCHEMA
        or value.get("identity_sha256") != identity_hash
        or type(next_frame) is not int
        or not 0 <= next_frame <= EXPECTED_PAIRS
        or next_frame > max_frames
    ):
        raise ProfilerError("resume identity/schema/pointer mismatch")
    status = value.get("status")
    if status not in ("partial", "complete") or (status == "complete") is not (next_frame == EXPECTED_PAIRS):
        raise ProfilerError("resume progress status/prefix invariant is malformed")
    preflight = value.get("storage_preflight")
    if not isinstance(preflight, dict):
        raise ProfilerError("resume progress storage preflight is malformed")
    exact_argv = value.get("exact_argv")
    if (
        not isinstance(exact_argv, list)
        or not exact_argv
        or any(not isinstance(argument, str) or not argument for argument in exact_argv)
    ):
        raise ProfilerError("resume progress argv custody is malformed")
    canonical_json_bytes(value)
    return value


def _capture_validated_progress_snapshot(
    progress_path: Path,
    *,
    identity_hash: str,
    max_frames: int,
) -> _ValidatedProgressSnapshot:
    snapshot = _read_bound_bytes(progress_path, name="progress authorization snapshot")
    try:
        value = json.loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfilerError("progress authorization snapshot is malformed JSON") from exc
    progress = _validate_progress_pointer(value, identity_hash=identity_hash, max_frames=max_frames)
    canonical = canonical_json_bytes(progress)
    if snapshot.payload != canonical + b"\n":
        raise ProfilerError("progress authorization snapshot is not canonical JSON")
    return _ValidatedProgressSnapshot(
        canonical_payload=canonical,
        file_identity=snapshot.file_identity,
    )


def _validate_resume_root(
    output_root: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_rebuild_argv: Iterable[str],
    max_frames: int,
) -> tuple[Path, Path, dict[str, Any], _ValidatedProgressSnapshot]:
    """Validate whole-object identity and certified root before resume."""

    unexpanded = output_root.expanduser()
    _require_local_directory(unexpanded, name="resume output root")
    try:
        resolved = unexpanded.resolve(strict=True)
    except OSError as exc:
        raise ProfilerError("resume output root is unavailable") from exc
    normalized_identity = json.loads(canonical_json_bytes(dict(expected_identity)))
    identity_hash = _sha256_bytes(canonical_json_bytes(normalized_identity))
    rebuild_argv = _normalized_argv(expected_rebuild_argv, name="resume rebuild argv")
    certification = _validate_output_certification(
        resolved,
        final_output_root=resolved,
        expected_identity=normalized_identity,
        expected_rebuild_argv=rebuild_argv,
    )
    root_entries = list(resolved.iterdir())
    names = {path.name for path in root_entries}
    retained_names = {path.name for path in root_entries if feature_cache_module.is_retained_name(path.name)}
    metadata_generations = {
        name
        for name in names
        if any(
            name.startswith(f".{target}{ATOMIC_GENERATION_SUFFIX}-")
            and ATOMIC_GENERATION_RE.fullmatch(name) is not None
            for target in (PROGRESS_NAME, RECEIPT_NAME)
        )
    }
    metadata_transactions = {
        name
        for name in names
        if any(
            name.startswith(f".{target}{ATOMIC_TRANSACTION_SUFFIX}-")
            and ATOMIC_TRANSACTION_RE.fullmatch(name) is not None
            for target in (PROGRESS_NAME, RECEIPT_NAME)
        )
    }
    metadata_completions = feature_cache_module.active_atomic_completion_names(
        {path.name: path for path in root_entries},
        target_names=(PROGRESS_NAME, RECEIPT_NAME),
    )
    allowed = {
        STAGING_SCRATCH_NAME,
        IDENTITY_NAME,
        PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
        "stages",
        RECEIPT_NAME,
        RECEIPT_AUTHORIZATION_ROOT_NAME,
        RECOVERY_ROOT_NAME,
        atomic_prepared_path(resolved / PROGRESS_NAME).name,
        atomic_prepared_path(resolved / RECEIPT_NAME).name,
        *metadata_generations,
        *metadata_transactions,
        *metadata_completions,
        *retained_names,
    }
    if not names.issubset(allowed) or not {
        STAGING_SCRATCH_NAME,
        IDENTITY_NAME,
        PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
        "stages",
    }.issubset(names):
        raise ProfilerError("resume output root contains unidentified or missing custody bytes")
    _load_receipt_authorizations(resolved)
    progress_path = resolved / PROGRESS_NAME
    progress_before = _stat_file_identity(_require_local_regular_file(progress_path, name="pre-read resume progress"))
    progress = _load_canonical_object(progress_path, name="resume progress")
    loaded_progress_identity = _stat_file_identity(
        _require_local_regular_file(progress_path, name="loaded resume progress")
    )
    if loaded_progress_identity != progress_before:
        raise ProfilerError("resume progress path changed during canonical validation")
    progress_replaced = False
    _validate_progress_pointer(progress, identity_hash=identity_hash, max_frames=max_frames)
    if progress["exact_argv"] != rebuild_argv:
        raise ProfilerError("resume progress rebuild argv differs from certified creation argv")
    if progress["storage_preflight"] != certification["storage_preflight"]:
        raise ProfilerError("resume progress storage preflight differs from output certification")
    stages = resolved / "stages"
    _require_local_directory(stages, name="resume stage root")
    # Recovery is part of the resume grammar, not an ignorable side tree.
    # Validate it before reconciling any prepared pointer or adopting a stage.
    attempt_custody = _validate_stage_attempt_custody(
        resolved,
        identity_sha256=identity_hash,
        exact_rebuild_argv=rebuild_argv,
        terminal=False,
    )
    resume_stage_paths, _resume_stage_receipts, resume_stage_head = _ordered_stage_chain(
        stages,
        identity_sha256=identity_hash,
        proven_retained_names=attempt_custody.proven_stage_retained_names,
    )
    progress_prepared = atomic_prepared_path(progress_path)
    progress_generations = sorted(
        resolved / name
        for name in metadata_generations
        if name.startswith(f".{PROGRESS_NAME}{ATOMIC_GENERATION_SUFFIX}-")
    )
    progress_transactions = sorted(
        resolved / name
        for name in metadata_transactions
        if name.startswith(f".{PROGRESS_NAME}{ATOMIC_TRANSACTION_SUFFIX}-")
    )
    progress_payload_scratch = (
        [progress_prepared] if os.path.lexists(progress_prepared) else []
    ) + progress_generations
    current_bytes = canonical_json_bytes(progress) + b"\n"
    authorized_progress_priors = [current_bytes]
    cleanup_current = False
    if progress_payload_scratch or progress_transactions:
        final_stages = sorted(
            entry for entry in stages.iterdir() if re.fullmatch(r"frame_[0-9]{4}\.bin", entry.name) is not None
        )
        candidates = [current_bytes]
        if len(final_stages) > progress["next_frame"] and len(final_stages) <= max_frames:
            orphan = dict(progress)
            orphan.update(
                {
                    "status": "complete" if len(final_stages) == EXPECTED_PAIRS else "partial",
                    "next_frame": len(final_stages),
                    "stage_chain_head_sha256": sha256_file(final_stages[-1]),
                }
            )
            candidates.append(canonical_json_bytes(orphan) + b"\n")
            authorized_progress_priors.append(candidates[-1])
        cleanup_current = True
        for scratch_path in progress_payload_scratch:
            prepared_bytes = _read_bound_bytes(
                scratch_path,
                name="prepared resume progress",
            ).payload
            matching = [
                candidate
                for candidate in candidates
                if candidate == prepared_bytes or candidate.startswith(prepared_bytes)
            ]
            if matching:
                cleanup_current = cleanup_current and len(matching) == 1 and matching[0] == current_bytes
                continue
            try:
                prior = json.loads(prepared_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProfilerError("prepared resume progress payload drift; preserving bytes") from exc
            if not isinstance(prior, dict) or canonical_json_bytes(prior) + b"\n" != prepared_bytes:
                raise ProfilerError("prepared resume progress payload drift; preserving bytes")
            _validate_progress_pointer(prior, identity_hash=identity_hash, max_frames=max_frames)
            prior_next = prior["next_frame"]
            expected_prior_head = (
                identity_hash if prior_next == 0 else sha256_file(stages / f"frame_{prior_next - 1:04d}.bin")
            )
            if (
                prior["exact_argv"] != rebuild_argv
                or prior["storage_preflight"] != certification["storage_preflight"]
                or prior_next > progress["next_frame"]
                or prior["stage_chain_head_sha256"] != expected_prior_head
            ):
                raise ProfilerError("prepared resume progress payload drift; preserving bytes")
            authorized_progress_priors.append(prepared_bytes)
    receipt_path = resolved / RECEIPT_NAME
    receipt_payload = (
        _read_bound_bytes(receipt_path, name="resume receipt retention target").payload
        if os.path.lexists(receipt_path)
        else None
    )
    receipt_prior_payloads, receipt_authorization_sha256 = (
        _resolve_receipt_transition_authorization(
            resolved,
            desired_receipt_payload=receipt_payload,
            identity_sha256=identity_hash,
            exact_rebuild_argv=rebuild_argv,
            terminal_stage_chain_sha256=resume_stage_head,
            frame_count=len(resume_stage_paths),
        )
        if receipt_payload is not None
        else ((), None)
    )
    _validate_output_retained_custody(
        resolved,
        expected_identity=normalized_identity,
        certification=certification,
        exact_rebuild_argv=rebuild_argv,
        progress_prior_payloads=authorized_progress_priors,
        receipt_prior_payloads=receipt_prior_payloads,
        receipt_authorization_sha256=receipt_authorization_sha256,
    )
    if cleanup_current:
        current_progress = _require_local_regular_file(
            progress_path,
            name="prepared-reconciliation progress",
        )
        if _stat_file_identity(current_progress) != loaded_progress_identity:
            raise ProfilerError("resume progress path changed before prepared reconciliation")
        atomic_json(
            progress_path,
            progress,
            expected_prior_payloads=tuple(authorized_progress_priors),
        )
        progress_replaced = True
    if receipt_payload is not None and _active_atomic_scratch(receipt_path):
        atomic_json(
            receipt_path,
            _load_canonical_object(receipt_path, name="resume receipt reconciliation"),
            expected_prior_payloads=receipt_prior_payloads,
            consumer_authorization_sha256=receipt_authorization_sha256,
        )
    receipt_prepared = atomic_prepared_path(resolved / RECEIPT_NAME)
    if os.path.lexists(receipt_prepared):
        _read_bound_bytes(receipt_prepared, name="prepared final receipt")
    progress_metadata = _require_local_regular_file(progress_path, name="validated resume progress")
    if not progress_replaced and _stat_file_identity(progress_metadata) != loaded_progress_identity:
        raise ProfilerError("resume progress path changed after canonical validation")
    _validate_output_retained_custody(
        resolved,
        expected_identity=normalized_identity,
        certification=certification,
        exact_rebuild_argv=rebuild_argv,
        progress_prior_payloads=(canonical_json_bytes(progress) + b"\n",),
        receipt_prior_payloads=receipt_prior_payloads,
        receipt_authorization_sha256=receipt_authorization_sha256,
    )
    return (
        stages,
        progress_path,
        certification,
        _ValidatedProgressSnapshot(
            canonical_payload=canonical_json_bytes(progress),
            file_identity=_stat_file_identity(progress_metadata),
        ),
    )


def _load_identity_bound_creation_storage_identity(output_root: Path) -> dict[str, Any]:
    """Load the stable creation anchor needed to re-derive a resumable identity.

    This is not an authority grant by itself.  The caller must subsequently
    validate the complete stored identity, certification, and stage-chain root.
    """

    _require_local_directory(output_root, name="resume output root")
    stored_identity = _load_canonical_object(
        output_root / IDENTITY_NAME,
        name="resume profile identity",
    )
    return _validate_creation_storage_identity(
        stored_identity.get("creation_storage_identity"),
        expected_output_root=output_root,
    )


def _fresh_creation_storage_identity(
    output_root: Path,
    *,
    current_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Bootstrap stable custody from certified pre-final staging when present.

    ``current_preflight`` has already enforced live free-space admission.  A
    complete staging scratch record preserves the first creation observation;
    using its stable projection prevents ordinary observation drift from
    changing the frame-zero identity.  Stable selection drift still blocks.
    """

    current_identity = _creation_storage_identity_from_preflight(
        current_preflight,
        expected_output_root=output_root,
    )
    staging_root = _creation_staging_path(output_root)
    if os.path.lexists(staging_root):
        _require_local_directory(staging_root, name="pre-final creation staging")
    scratch = _load_complete_staging_scratch(staging_root)
    if scratch is None:
        return current_identity
    stored_preflight = scratch.get("storage_preflight")
    if not isinstance(stored_preflight, dict):
        raise ProfilerError("pre-final staging scratch lacks creation storage custody")
    stored_identity = _creation_storage_identity_from_preflight(
        stored_preflight,
        expected_output_root=output_root,
    )
    if stored_identity != current_identity:
        raise ProfilerError("pre-final staging stable storage custody differs from current preflight")
    return stored_identity


def _stream_accounting(stage_paths: list[Path]) -> dict[str, Any]:
    raw_hash = hashlib.sha256()
    zlib_hash = hashlib.sha256()
    brotli_hash = hashlib.sha256()
    zlib_parseback_hash = hashlib.sha256()
    brotli_parseback_hash = hashlib.sha256()
    raw_count = zlib_count = brotli_count = 0
    zlib_parseback_count = brotli_parseback_count = 0
    symbols = np.zeros(256, dtype=np.int64)
    zcompressor = zlib.compressobj(
        level=ZLIB_STREAM_SETTINGS["level"],
        method=ZLIB_STREAM_SETTINGS["method"],
        wbits=ZLIB_STREAM_SETTINGS["wbits"],
        memLevel=ZLIB_STREAM_SETTINGS["mem_level"],
        strategy=ZLIB_STREAM_SETTINGS["strategy"],
    )
    bcompressor = brotli.Compressor(**BROTLI_STREAM_SETTINGS)
    zdecompressor = zlib.decompressobj(wbits=ZLIB_STREAM_SETTINGS["wbits"])
    bdecompressor = brotli.Decompressor()

    def parse_compressed(zchunk: bytes, bchunk: bytes) -> None:
        nonlocal zlib_parseback_count, brotli_parseback_count
        if zchunk:
            decoded = zdecompressor.decompress(zchunk)
            zlib_parseback_hash.update(decoded)
            zlib_parseback_count += len(decoded)
        if bchunk:
            decoded = bdecompressor.process(bchunk)
            brotli_parseback_hash.update(decoded)
            brotli_parseback_count += len(decoded)

    def consume(chunk: bytes) -> None:
        nonlocal raw_count, zlib_count, brotli_count
        raw_hash.update(chunk)
        raw_count += len(chunk)
        symbols[:] += np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
        zchunk = zcompressor.compress(chunk)
        zlib_hash.update(zchunk)
        zlib_count += len(zchunk)
        bchunk = bcompressor.process(chunk)
        brotli_hash.update(bchunk)
        brotli_count += len(bchunk)
        parse_compressed(zchunk, bchunk)

    consume(STREAM_MAGIC + struct.pack("<I", len(stage_paths)))
    for stage in stage_paths:
        _receipt, candidate_payload = _parse_stage_payload(
            _read_bound_bytes(stage, name="stream-accounting stage").payload
        )
        consume(struct.pack("<Q", len(candidate_payload)))
        consume(candidate_payload)
    ztail = zcompressor.flush()
    zlib_hash.update(ztail)
    zlib_count += len(ztail)
    btail = bcompressor.finish()
    brotli_hash.update(btail)
    brotli_count += len(btail)
    parse_compressed(ztail, btail)
    zfinal = zdecompressor.flush()
    zlib_parseback_hash.update(zfinal)
    zlib_parseback_count += len(zfinal)
    if (
        not zdecompressor.eof
        or zdecompressor.unused_data
        or zdecompressor.unconsumed_tail
        or not bdecompressor.is_finished()
    ):
        raise ProfilerError("streaming codec parse-back did not reach exact stream termination")
    raw_digest = raw_hash.hexdigest()
    if (
        zlib_parseback_count != raw_count
        or brotli_parseback_count != raw_count
        or zlib_parseback_hash.hexdigest() != raw_digest
        or brotli_parseback_hash.hexdigest() != raw_digest
    ):
        raise ProfilerError("streaming zlib/Brotli parse-back differs from the raw receiver stream")
    nonzero = symbols[symbols > 0].astype(np.float64)
    probabilities = nonzero / raw_count
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "raw": {
            "label": "MEASURED_ACTUAL_RAW_STREAM_BYTES",
            "bytes": raw_count,
            "sha256": raw_digest,
        },
        "zlib_level9": {
            "label": CODEC_BYTE_COUNT_LABEL,
            "codec": "zlib_level9",
            "bytes": zlib_count,
            "sha256": zlib_hash.hexdigest(),
            "codec_parseback_identical_raw": True,
            "decompressed_raw_bytes": zlib_parseback_count,
            "decompressed_raw_sha256": zlib_parseback_hash.hexdigest(),
        },
        "brotli_quality11": {
            "label": CODEC_BYTE_COUNT_LABEL,
            "codec": "brotli_quality11",
            "bytes": brotli_count,
            "sha256": brotli_hash.hexdigest(),
            "codec_parseback_identical_raw": True,
            "decompressed_raw_bytes": brotli_parseback_count,
            "decompressed_raw_sha256": brotli_parseback_hash.hexdigest(),
        },
        "headers_and_termination_included": True,
        "order0_entropy": {
            "label": ORDER0_ESTIMATE_LABEL,
            "assumptions": (
                "empirical order-0 IID PMF and model are free; no header or "
                "termination charge; not a bound for context or grammar coders"
            ),
            "bits_per_byte_symbol": entropy,
            "rounded_up_bytes": int(np.ceil(entropy * raw_count / 8.0)),
        },
    }


def _ordered_stage_chain(
    stages: Path,
    *,
    identity_sha256: str,
    proven_retained_names: Iterable[str] = (),
) -> tuple[list[Path], list[dict[str, Any]], str]:
    """Read the exact ordered committed stage chain without semantic replay."""

    identity_hash = _canonical_sha256(identity_sha256, name="stage-chain identity hash")
    _require_local_directory(stages, name="terminal stage root")
    stage_entries = sorted(stages.iterdir())
    retained_names = set(proven_retained_names)
    actual_retained_names = {path.name for path in stage_entries if feature_cache_module.is_retained_name(path.name)}
    if retained_names != actual_retained_names:
        raise ProfilerError("terminal stage root contains role-unproven retained custody")
    paths = [path for path in stage_entries if path.name not in retained_names]
    expected_names = [f"frame_{index:04d}.bin" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names:
        raise ProfilerError("terminal stage root contains missing, prepared, or unidentified bytes")
    previous = identity_hash
    receipts: list[dict[str, Any]] = []
    for frame, path in enumerate(paths):
        _require_local_regular_file(path, name=f"terminal stage {path.name}")
        payload = _read_bound_bytes(path, name=f"terminal stage {path.name}").payload
        receipt, _candidate_payload = _parse_stage_payload(payload)
        if (
            receipt.get("schema") != STAGE_RECEIPT_SCHEMA
            or receipt.get("identity_sha256") != identity_hash
            or receipt.get("previous_stage_sha256") != previous
            or type(receipt.get("frame")) is not int
            or receipt.get("frame") != frame
        ):
            raise ProfilerError("terminal stage receipt chain binding is malformed")
        counters = receipt.get("counters")
        if not isinstance(counters, dict) or set(counters) != set(COUNTER_NAMES):
            raise ProfilerError("terminal stage counters are malformed")
        _validate_stage_timing(receipt.get("timing"), total_blocks=counters["total_blocks"])
        receipts.append(receipt)
        previous = _sha256_bytes(payload)
    return paths, receipts, previous


def _validated_recovery_transactions(
    output_root: Path,
    *,
    identity_sha256: str,
    exact_rebuild_argv: Iterable[str],
) -> list[dict[str, Any]]:
    return _validate_recovery_grammar(
        output_root,
        identity_sha256=identity_sha256,
        exact_rebuild_argv=exact_rebuild_argv,
        terminal=True,
    )


def _creation_retained_expectations_for_output(
    output_root: Path,
    *,
    expected_identity: Mapping[str, Any],
    certification: Mapping[str, Any],
    exact_rebuild_argv: Iterable[str],
) -> dict[str, bytes]:
    """Re-derive every deterministic creation-prepared payload from custody."""

    normalized_identity = json.loads(canonical_json_bytes(dict(expected_identity)))
    identity_hash = _sha256_bytes(canonical_json_bytes(normalized_identity))
    rebuild_argv = _normalized_argv(exact_rebuild_argv, name="creation-retention rebuild argv")
    storage_preflight = certification.get("storage_preflight")
    if not isinstance(storage_preflight, dict):
        raise ProfilerError("creation-retention certification lacks storage preflight")
    scratch = _staging_scratch_record(
        output_root=output_root,
        identity_sha256=identity_hash,
        exact_rebuild_argv=rebuild_argv,
        storage_preflight=storage_preflight,
    )
    if _load_canonical_object(output_root / STAGING_SCRATCH_NAME, name="creation-retention scratch") != scratch:
        raise ProfilerError("creation-retention scratch differs from certified creation")
    initial_progress = {
        "schema": PROGRESS_SCHEMA,
        "identity_sha256": identity_hash,
        "status": "partial",
        "next_frame": 0,
        "stage_chain_head_sha256": identity_hash,
        "storage_preflight": json.loads(canonical_json_bytes(storage_preflight)),
        "exact_argv": rebuild_argv,
    }
    _validate_progress_pointer(initial_progress, identity_hash=identity_hash, max_frames=EXPECTED_PAIRS)
    expected_certification = _output_certification_record(
        output_root=output_root,
        identity_sha256=identity_hash,
        identity_json_sha256=sha256_file(output_root / IDENTITY_NAME),
        exact_rebuild_argv=rebuild_argv,
        storage_preflight=storage_preflight,
    )
    if dict(certification) != expected_certification:
        raise ProfilerError("creation-retention certification differs from deterministic custody")
    values = {
        STAGING_SCRATCH_NAME: scratch,
        IDENTITY_NAME: normalized_identity,
        PROGRESS_NAME: initial_progress,
        OUTPUT_CERTIFICATION_NAME: expected_certification,
    }
    return {
        _creation_prepared_path(Path(target)).name: canonical_json_bytes(value) + b"\n"
        for target, value in values.items()
    }


def _validate_output_retained_custody(
    output_root: Path,
    *,
    expected_identity: Mapping[str, Any],
    certification: Mapping[str, Any],
    exact_rebuild_argv: Iterable[str],
    progress_prior_payloads: Iterable[bytes] = (),
    receipt_prior_payloads: Iterable[bytes] = (),
    receipt_authorization_sha256: str | None = None,
) -> set[str]:
    """Prove creation and generic-atomic retention before grammar exclusion."""

    entries = list(output_root.iterdir())
    actual_retained = {entry.name for entry in entries if feature_cache_module.is_retained_name(entry.name)}
    proven = _validate_creation_retained_names(
        entries,
        expected_payloads_by_original=_creation_retained_expectations_for_output(
            output_root,
            expected_identity=expected_identity,
            certification=certification,
            exact_rebuild_argv=exact_rebuild_argv,
        ),
        name="profile output root",
    )
    for target_name, prior_payloads in (
        (PROGRESS_NAME, tuple(progress_prior_payloads)),
        (RECEIPT_NAME, tuple(receipt_prior_payloads)),
    ):
        target = output_root / target_name
        related_retained = {
            entry.name
            for entry in entries
            if feature_cache_module.is_retained_name(entry.name)
            and _is_atomic_scratch_original(
                feature_cache_module.retained_original_name(entry.name),
                targets=(target_name,),
            )
        }
        if not os.path.lexists(target):
            if related_retained:
                raise ProfilerError(f"retained {target_name} atomic custody lacks its committed target")
            continue
        desired_payload = _read_bound_bytes(target, name=f"retained-custody target {target_name}").payload
        target_proven = _validate_atomic_retained_names(
            target,
            desired_payload=desired_payload,
            expected_prior_payloads=prior_payloads,
            expected_consumer_authorization_sha256=(
                receipt_authorization_sha256 if target_name == RECEIPT_NAME else None
            ),
            name=f"profile output {target_name}",
        )
        if target_proven != related_retained:
            raise ProfilerError(f"profile output {target_name} has orphaned retained atomic custody")
        proven.update(target_proven)
    if proven != actual_retained:
        raise ProfilerError("profile output root contains role-unproven retained custody")
    return proven


def _validate_terminal_output_layout(
    output_root: Path,
    *,
    proven_retained_names: Iterable[str] = (),
    allow_receipt_scratch: bool = False,
) -> None:
    """Require the exact terminal root grammar including role-bound retention."""

    _require_local_directory(output_root, name="terminal output root")
    entries = list(output_root.iterdir())
    retained = set(proven_retained_names)
    actual_retained = {path.name for path in entries if feature_cache_module.is_retained_name(path.name)}
    if retained != actual_retained:
        raise ProfilerError("terminal output root contains role-unproven retained custody")
    active_names = {path.name for path in entries} - retained
    receipt_scratch = {path.name for path in _active_atomic_scratch(output_root / RECEIPT_NAME)}
    metadata_completions = feature_cache_module.active_atomic_completion_names(
        {path.name: path for path in entries},
        target_names=(PROGRESS_NAME, RECEIPT_NAME),
    )
    if receipt_scratch:
        for path in entries:
            if path.name in receipt_scratch:
                _read_bound_bytes(path, name="terminal receipt recovery scratch")
        if not allow_receipt_scratch:
            raise ProfilerError("terminal output root contains active receipt scratch custody")
    allowed = {
        STAGING_SCRATCH_NAME,
        IDENTITY_NAME,
        PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
        "stages",
        RECEIPT_NAME,
        RECEIPT_AUTHORIZATION_ROOT_NAME,
        RECOVERY_ROOT_NAME,
        *receipt_scratch,
        *metadata_completions,
    }
    required = {
        STAGING_SCRATCH_NAME,
        IDENTITY_NAME,
        PROGRESS_NAME,
        OUTPUT_CERTIFICATION_NAME,
        "stages",
    }
    if not required.issubset(active_names) or not active_names.issubset(allowed):
        raise ProfilerError("terminal output root contains unidentified or active scratch custody")


def _terminal_custody(
    output_root: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_rebuild_argv: Iterable[str],
    allow_receipt_scratch: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized_identity = json.loads(canonical_json_bytes(dict(expected_identity)))
    identity_sha256 = _sha256_bytes(canonical_json_bytes(normalized_identity))
    rebuild_argv = _normalized_argv(expected_rebuild_argv, name="terminal rebuild argv")
    certification = _validate_output_certification(
        output_root,
        final_output_root=output_root,
        expected_identity=normalized_identity,
        expected_rebuild_argv=rebuild_argv,
    )
    progress_path = output_root / PROGRESS_NAME
    progress = _load_canonical_object(progress_path, name="terminal progress")
    max_frames = int(normalized_identity["config"]["profiled_frame_limit"])
    _validate_progress_pointer(progress, identity_hash=identity_sha256, max_frames=max_frames)
    if progress["exact_argv"] != rebuild_argv or progress["storage_preflight"] != certification["storage_preflight"]:
        raise ProfilerError("terminal progress differs from certified creation custody")
    progress_payload = canonical_json_bytes(progress) + b"\n"
    attempt_custody = _validate_stage_attempt_custody(
        output_root,
        identity_sha256=identity_sha256,
        exact_rebuild_argv=rebuild_argv,
        terminal=True,
    )
    stage_paths, receipts, terminal_hash = _ordered_stage_chain(
        output_root / "stages",
        identity_sha256=identity_sha256,
        proven_retained_names=attempt_custody.proven_stage_retained_names,
    )
    receipt_path = output_root / RECEIPT_NAME
    receipt_payload = (
        _read_bound_bytes(receipt_path, name="terminal receipt retention target").payload
        if os.path.lexists(receipt_path)
        else None
    )
    receipt_prior_payloads, receipt_authorization_sha256 = (
        _resolve_receipt_transition_authorization(
            output_root,
            desired_receipt_payload=receipt_payload,
            identity_sha256=identity_sha256,
            exact_rebuild_argv=rebuild_argv,
            terminal_stage_chain_sha256=terminal_hash,
            frame_count=len(stage_paths),
        )
        if receipt_payload is not None
        else ((), None)
    )
    proven_output_retained = _validate_output_retained_custody(
        output_root,
        expected_identity=normalized_identity,
        certification=certification,
        exact_rebuild_argv=rebuild_argv,
        progress_prior_payloads=(progress_payload,),
        receipt_prior_payloads=receipt_prior_payloads,
        receipt_authorization_sha256=receipt_authorization_sha256,
    )
    _validate_terminal_output_layout(
        output_root,
        proven_retained_names=proven_output_retained,
        allow_receipt_scratch=allow_receipt_scratch,
    )
    if progress["next_frame"] != len(stage_paths) or progress["stage_chain_head_sha256"] != terminal_hash:
        raise ProfilerError("terminal progress does not bind the ordered stage root")
    mode = normalized_identity["config"]["mode"]
    stream_accounting = _stream_accounting(stage_paths) if mode == ENUMERATED_MODE else None
    identity_path = output_root / IDENTITY_NAME
    _validate_terminal_output_layout(
        output_root,
        proven_retained_names=proven_output_retained,
        allow_receipt_scratch=allow_receipt_scratch,
    )
    return (
        {
            "identity_sha256": identity_sha256,
            "identity_json_sha256": sha256_file(identity_path),
            "identity_json_bytes": identity_path.stat().st_size,
            "output_certification_sha256": sha256_file(output_root / OUTPUT_CERTIFICATION_NAME),
            "exact_rebuild_argv": rebuild_argv,
            "ordered_stage_count": len(stage_paths),
            "terminal_stage_chain_sha256": terminal_hash,
            "progress_pointer": progress,
            "progress_pointer_sha256": sha256_file(progress_path),
            "stream_accounting": stream_accounting,
            "recovery_transactions": [
                dict(outcome) for outcome in attempt_custody.outcomes if outcome["outcome"] == "recovery"
            ],
        },
        receipts,
    )


def _validate_terminal_semantic_replay(
    output_root: Path,
    stage_receipts: list[dict[str, Any]],
    *,
    semantic_replay_provider: Callable[[int], _FrameSemanticReplay],
) -> None:
    """Re-derive every committed stage before accepting terminal custody."""

    if not callable(semantic_replay_provider):
        raise ProfilerError("final receipt validation requires a semantic replay provider")
    stages = output_root / "stages"
    _require_local_directory(stages, name="semantic replay stage root")
    for frame, stored_receipt in enumerate(stage_receipts):
        path = stages / f"frame_{frame:04d}.bin"
        _require_local_regular_file(path, name=f"semantic replay stage {path.name}")
        parsed_receipt, candidate_payload = _parse_stage_payload(
            _read_bound_bytes(path, name=f"final receipt stage {path.name}").payload
        )
        if parsed_receipt != stored_receipt:
            raise ProfilerError("semantic replay stage receipt differs from terminal chain read")
        expected = semantic_replay_provider(frame)
        if not isinstance(expected, _FrameSemanticReplay):
            raise ProfilerError("semantic replay provider returned a malformed result")
        _validate_stage_receipt(
            parsed_receipt,
            candidate_payload,
            expected_partition_custody=expected.partition_custody,
            expected_aggregate_state=expected.aggregate_state,
            expected_candidate_payload=expected.candidate_payload,
            expected_counters=expected.counters,
            expected_selection_custody=expected.selection_custody,
        )


def _reconstruct_rd_row(
    *,
    totals: Mapping[str, int],
    stage_receipts: list[dict[str, Any]],
    config: Mapping[str, Any],
    stream_accounting: Mapping[str, Any] | None,
    feature_cache_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Rebuild the complete RD row without trusting persisted result prose."""

    if config.get("mode") == BOUNDS_MODE:
        if stream_accounting is not None:
            raise ProfilerError("bounds-only RD reconstruction received a stream")
        return None
    if config.get("mode") != ENUMERATED_MODE or not isinstance(stream_accounting, Mapping):
        raise ProfilerError("enumerated RD reconstruction requires typed stream custody")
    rate_frames = tuple(range(len(stage_receipts)))
    scorer_frames = tuple(
        index for index, receipt in enumerate(stage_receipts) if receipt["counters"]["segnet_pixels"] > 0
    )
    scored = config.get("score_segnet") is True and scorer_frames == rate_frames and bool(rate_frames)
    committed_prefix = feature_cache_binding.get("committed_prefix_sha256")
    _canonical_sha256(committed_prefix, name="RD feature-cache prefix hash")
    try:
        row = build_rd_row(
            selected_block_count=totals["selected_blocks"],
            total_block_count=totals["total_blocks"],
            stream_accounting=stream_accounting,
            axis="[macOS-CPU advisory]" if scored else "NO_VERDICT_SCORER_CUSTODY",
            cache_scope=f"feature_prefix_{committed_prefix}",
            receiver_scope="strict_signed_residual_zigzag_uleb128_support_union_parseback.v3",
            mismatch_count=totals["segnet_mismatches"] if scored else None,
            scorer_pixel_count=totals["segnet_pixels"] if scored else None,
            rate_scope_frames=rate_frames,
            scorer_scope_frames=scorer_frames,
        )
    except (KeyError, TypeError, ValueError, LatticeProfileError) as exc:
        raise ProfilerError("RD row cannot be reconstructed from semantic stage custody") from exc
    row.update(
        {
            "exactness_insistence_rule": (
                "every selected source seed satisfies the exact integer resize equation; bounded "
                "search reports only cheapest-seen non-global selection"
                if config.get("seed_source_witness") is True
                else row["exactness_insistence_rule"]
            ),
            "globally_exhaustive_selection_count": totals["exhaustive_selected_blocks"],
            "cheapest_seen_bounded_selection_count": totals["bounded_selected_blocks"],
            "omitted_or_fallback_blocks": totals["omitted_blocks"],
            **_selection_custody(
                mode=ENUMERATED_MODE,
                counters=totals,
                seed_source_witness=config.get("seed_source_witness"),
                receiver_closed=(
                    bool(stage_receipts)
                    and all(
                        receipt["selection_custody"].get("receiver_non_closure") is False for receipt in stage_receipts
                    )
                ),
            ),
            "scorer_coverage_frames": list(scorer_frames),
            "rate_stream_coverage_frames": list(rate_frames),
            "node_cap": config.get("max_nodes"),
            "profiled_frame_indices": list(rate_frames),
        }
    )
    return json.loads(canonical_json_bytes(row))


def _validate_final_receipt(
    value: object,
    *,
    output_root: Path,
    expected_identity: Mapping[str, Any],
    expected_rebuild_argv: Iterable[str],
    semantic_replay_provider: Callable[[int], _FrameSemanticReplay],
) -> dict[str, Any]:
    """Clean-room validation of terminal identity, chain, stream, and timing custody."""

    if not isinstance(value, dict) or set(value) != FINAL_RECEIPT_KEYS:
        raise ProfilerError("final profile receipt schema is malformed")
    normalized_identity = json.loads(canonical_json_bytes(dict(expected_identity)))
    identity_sha256 = _sha256_bytes(canonical_json_bytes(normalized_identity))
    config = normalized_identity.get("config")
    repository = normalized_identity.get("repository")
    if not isinstance(config, dict) or not isinstance(repository, dict):
        raise ProfilerError("expected identity config/repository custody is malformed")
    custody, stage_receipts = _terminal_custody(
        output_root,
        expected_identity=normalized_identity,
        expected_rebuild_argv=expected_rebuild_argv,
    )
    _validate_terminal_semantic_replay(
        output_root,
        stage_receipts,
        semantic_replay_provider=semantic_replay_provider,
    )
    stage_count = len(stage_receipts)
    status = "complete" if stage_count == EXPECTED_PAIRS else "partial_prefix"
    scope_label = "FULL_N600" if stage_count == EXPECTED_PAIRS else "HASH_VALID_EXPLICIT_PREFIX"
    if (
        value.get("schema") != FINAL_RECEIPT_SCHEMA
        or value.get("identity_sha256") != identity_sha256
        or value.get("git_head") != repository.get("git_head")
        or value.get("exact_rebuild_argv") != custody["exact_rebuild_argv"]
        or value.get("requested_outer_governor_limits") != config.get("requested_outer_governor_limits")
        or value.get("custody") != custody
        or value.get("status") != status
        or value.get("scope_label") != scope_label
        or value.get("mode") != config.get("mode")
        or value.get("frames_profiled") != stage_count
        or value.get("expected_frames") != EXPECTED_PAIRS
        or value.get("profiled_frame_indices") != list(range(stage_count))
        or value.get("scope_extrapolation") != "NONE_EXACT_FRAME_INDICES_ONLY"
        or value.get("feature_cache_binding") != normalized_identity.get("feature_cache_binding")
    ):
        raise ProfilerError("final profile receipt identity/scope/custody mismatch")
    expected_timing = _timing_summary(
        stage_receipts,
        terminal_stage_chain_sha256=custody["terminal_stage_chain_sha256"],
    )
    if value.get("timing_summary") != expected_timing:
        raise ProfilerError("final profile receipt timing summary mismatch")

    aggregate = StreamingProfileAggregator(
        n_classes=EXPECTED_CLASSES,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    totals = dict.fromkeys(COUNTER_NAMES, 0)
    for receipt in stage_receipts:
        _validate_aggregate_state(receipt.get("aggregate_delta_state"))
        aggregate.merge(StreamingProfileAggregator.from_state(receipt["aggregate_delta_state"]))
        for name in COUNTER_NAMES:
            totals[name] += _nonnegative_receipt_int(receipt["counters"][name], name=f"terminal {name}")
    if (
        value.get("aggregate") != aggregate.summary()
        or value.get("counters_rebuilt_from_hashed_stage_receipts") != totals
    ):
        raise ProfilerError("final profile receipt aggregate/counter reconstruction mismatch")
    rd_row = value.get("rd_row")
    expected_rd_row = _reconstruct_rd_row(
        totals=totals,
        stage_receipts=stage_receipts,
        config=config,
        stream_accounting=custody["stream_accounting"],
        feature_cache_binding=normalized_identity.get("feature_cache_binding", {}),
    )
    expected_derivation = _expected_derivation(
        mode=config.get("mode"),
        seed_source_witness=config.get("seed_source_witness"),
    )
    expected_positive_control = _expected_positive_control(mode=config.get("mode"))
    if (
        rd_row != expected_rd_row
        or value.get("lower_bound_method") != (LOWER_BOUND_METHOD if config.get("mode") == BOUNDS_MODE else None)
        or value.get("derivation") != expected_derivation
        or value.get("positive_control") != expected_positive_control
    ):
        raise ProfilerError("final profile receipt RD/derivation/positive-control reconstruction mismatch")
    claims = value.get("claims")
    authority = value.get("authority")
    all_receivers_closed = bool(stage_receipts) and all(
        isinstance(receipt.get("selection_custody"), dict)
        and receipt["selection_custody"].get("receiver_non_closure") is False
        for receipt in stage_receipts
    )
    expected_selection = _selection_custody(
        mode=config["mode"],
        counters=totals,
        seed_source_witness=config["seed_source_witness"],
        receiver_closed=all_receivers_closed,
    )
    expected_claims = {
        "exact_count_claim": expected_selection["exact_count_claim"],
        "min_description_claim": False,
        "selection_globally_exact": expected_selection["selection_globally_exact"],
        "selection_label": expected_selection["selection_label"],
        "d_seg_claim": bool(isinstance(expected_rd_row, dict) and expected_rd_row.get("d_seg") is not None),
        "candidate_stream_emitted": config["mode"] == ENUMERATED_MODE,
        "receiver_non_closure": expected_selection["receiver_non_closure"],
        "per_block_selector_minimum_proved": expected_selection["per_block_selector_minimum_proved"],
        "global_compressed_stream_minimum_claim": False,
    }
    expected_authority = {
        "score_authority": False,
        "promotion_eligible": False,
        "pose_bank_wired": False,
        "factor10_solved": False,
        "global_compressed_stream_minimum_claim": False,
    }
    if claims != expected_claims or authority != expected_authority:
        raise ProfilerError("final profile receipt false-authority flags are malformed")
    canonical_json_bytes(value)
    return value


def _feature_binding(
    feature: Any,
    gt_cache: Path,
    *,
    prefix_frames: int,
    scorer_sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    declared_sources = feature.identity.get("source_files", {})
    if not isinstance(declared_sources, Mapping):
        raise ProfilerError("feature cache source bindings are malformed")
    cache_config = feature.identity.get("config")
    if not isinstance(cache_config, Mapping):
        raise ProfilerError("feature cache runtime contract is malformed")
    expected_runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    expected_determinism = {
        "torch_deterministic_algorithms": True,
        "torch_threads_effective": 1,
        "torch_interop_threads_effective": 1,
    }
    if (
        cache_config.get("authority_mode") != "deterministic_cpu_float32_batch_one"
        or cache_config.get("batch_size") != 1
        or cache_config.get("runtime") != expected_runtime
        or cache_config.get("determinism") != expected_determinism
    ):
        raise ProfilerError("feature cache runtime/batch-one determinism contract is stale or foreign")
    declared = declared_sources.get("gt_n600_npz")
    actual = source_file_row(gt_cache)
    if declared != actual:
        raise ProfilerError("--gt-cache does not match feature manifest gt_n600_npz path/bytes/SHA-256")
    if feature.next_frame < prefix_frames:
        raise ProfilerError("feature cache does not cover requested profiler prefix")
    committed = feature.progress.get("committed_frames")
    if not isinstance(committed, list) or len(committed) < prefix_frames:
        raise ProfilerError("feature cache committed prefix is malformed")
    current_cache_sources = {
        "extractor_tool": source_file_row(REPO_ROOT / "tools/extract_segnet_head_features_n600.py"),
        "cache_module": source_file_row(REPO_ROOT / "src/tac/witness_control/segnet_head_feature_cache.py"),
    }
    if scorer_sources is not None:
        current_cache_sources.update({key: dict(row) for key, row in scorer_sources.items()})
    for role, row in current_cache_sources.items():
        if declared_sources.get(role) != row:
            raise ProfilerError(f"feature cache {role} source binding is stale or foreign")
    return {
        "manifest_identity_sha256": _sha256_bytes(canonical_json_bytes(feature.identity)),
        "progress_identity_sha256": feature.progress.get("identity_sha256"),
        "committed_prefix_frames": prefix_frames,
        "committed_prefix_sha256": _sha256_bytes(canonical_json_bytes(committed[:prefix_frames])),
        "completion_positive_control_sha256": (
            _sha256_bytes(canonical_json_bytes(feature.progress["completion_positive_control"]))
            if prefix_frames == EXPECTED_PAIRS
            else None
        ),
        "gt_n600_npz": actual,
        "current_execution_source_binding_sha256": _sha256_bytes(canonical_json_bytes(current_cache_sources)),
        "current_execution_runtime_contract_sha256": _sha256_bytes(
            canonical_json_bytes(
                {
                    "authority_mode": cache_config["authority_mode"],
                    "batch_size": cache_config["batch_size"],
                    "runtime": expected_runtime,
                    "determinism": expected_determinism,
                }
            )
        ),
    }


def _scorer_source_bindings(upstream_root: Path) -> _FrozenScorerSnapshot:
    rows = {
        "executed_modules_py": source_file_row(upstream_root / "modules.py"),
        "executed_frame_utils_py": source_file_row(upstream_root / "frame_utils.py"),
        "executed_tac_scorer_py": source_file_row(REPO_ROOT / "src/tac/scorer.py"),
    }
    weights_path = upstream_root / "models" / "segnet.safetensors"
    admitted = _read_bound_bytes(weights_path, name="admitted SegNet weights")
    rows["segnet_weights"] = {
        "path": str(weights_path),
        "bytes": len(admitted.payload),
        "sha256": _sha256_bytes(admitted.payload),
    }
    return _FrozenScorerSnapshot(rows=rows, segnet_payload=admitted.payload)


def _require_equal_scorer_source_snapshots(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized_before = json.loads(canonical_json_bytes({key: dict(value) for key, value in before.items()}))
    normalized_after = json.loads(canonical_json_bytes({key: dict(value) for key, value in after.items()}))
    if normalized_before != normalized_after:
        changed = sorted(
            role
            for role in set(normalized_before) | set(normalized_after)
            if normalized_before.get(role) != normalized_after.get(role)
        )
        raise ProfilerError(f"frozen scorer source snapshot changed during scorer load: {changed}")
    return normalized_before


def _stable_admission_contract(
    admission_attestation: Mapping[str, Any],
    *,
    canonical_fresh_rebuild_argv: Iterable[str],
) -> dict[str, Any]:
    """Derive resume-stable admission custody from a live parent attestation.

    The direct parent PID and its exact invocation necessarily change after a
    crash.  They remain in the per-invocation receipt, but cannot participate
    in the immutable experiment identity.  This contract retains the launcher
    sources, resource caps, runtime, and canonical fresh child command while
    proving that the current child is exactly that command or its one-token
    ``--resume`` continuation.
    """

    normalized = json.loads(canonical_json_bytes(dict(admission_attestation)))
    fresh_argv = _normalized_argv(
        canonical_fresh_rebuild_argv,
        name="canonical fresh admission argv",
    )
    child_argv = _normalized_argv(
        normalized.get("child_exact_argv", ()),
        name="attested child argv",
    )
    child_without_resume = list(child_argv)
    resume_count = child_without_resume.count("--resume")
    if resume_count > 1:
        raise ProfilerError("attested child argv contains multiple --resume tokens")
    if resume_count == 1:
        child_without_resume.remove("--resume")
    if child_without_resume != fresh_argv:
        raise ProfilerError("attested child argv differs from the canonical fresh rebuild argv")

    try:
        fresh_python = Path(fresh_argv[0]).expanduser().resolve(strict=True)
        parent_python = Path(normalized.get("parent_python_executable", "")).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ProfilerError("safe-run Python runtime custody is unavailable") from exc
    if parent_python != fresh_python:
        raise ProfilerError("safe-run Python runtime differs from the canonical child runtime")

    outer_caps = normalized.get("outer_resource_caps")
    source_custody = normalized.get("source_custody")
    if (
        normalized.get("schema") != "governed_safe_run_parent_attestation.v1"
        or normalized.get("attestation_scope") != "DIRECT_PARENT_COMMAND_AT_CHILD_START_NOT_COMPLETED_STATUS"
        or normalized.get("governed_marker_present") is not True
        or normalized.get("admission_bypass_present") is not False
        or normalized.get("completed_safe_run_status_receipt") is not None
        or not isinstance(outer_caps, dict)
        or not isinstance(source_custody, dict)
    ):
        raise ProfilerError("safe-run parent attestation is malformed")

    return {
        "schema": "governed_safe_run_stable_admission_contract.v1",
        "derived_from_attestation_schema": normalized["schema"],
        "attestation_requirement": "FRESH_DIRECT_PARENT_AT_EVERY_INVOCATION",
        "canonical_fresh_child_argv": fresh_argv,
        "launcher_python_executable": str(fresh_python),
        "outer_resource_caps": outer_caps,
        "governed_marker_required": True,
        "admission_bypass_forbidden": True,
        "completed_safe_run_status_receipt": None,
        "source_custody": source_custody,
        "volatile_per_invocation_fields_excluded": [
            "parent_pid",
            "parent_exact_argv",
            "child_exact_argv",
        ],
    }


def _identity(
    args: argparse.Namespace,
    gt_cache: Path,
    feature_root: Path,
    feature_binding: dict[str, Any],
    scorer_sources: dict[str, dict[str, Any]] | None,
    admission_attestation: Mapping[str, Any],
    canonical_fresh_rebuild_argv: Iterable[str],
    creation_storage_identity: Mapping[str, Any],
) -> dict[str, Any]:
    selector = SignedResidualCostModel()
    plugin = NoOpPosePlugin()
    brotli_version = getattr(brotli, "__version__", None)
    if not isinstance(brotli_version, str) or not brotli_version:
        raise ProfilerError("Brotli runtime version custody is unavailable")
    canonical_operator = DisjointResizeOperator.build(
        camera_h=EXPECTED_CAMERA_HW[0],
        camera_w=EXPECTED_CAMERA_HW[1],
        scorer_h=EXPECTED_SEG_HW[0],
        scorer_w=EXPECTED_SEG_HW[1],
    )
    support_union = _receiver_support_union(canonical_operator)
    feasibility_path = _require_exact_module_file(
        feasibility_module,
        REPO_ROOT / "src/tac/optimization/uint8_lattice_feasibility.py",
        role="tac.optimization.uint8_lattice_feasibility",
    )
    profile_path = _require_exact_module_file(
        profile_module,
        REPO_ROOT / "src/tac/optimization/uint8_lattice_profile.py",
        role="tac.optimization.uint8_lattice_profile",
    )
    feature_cache_path = _require_exact_module_file(
        feature_cache_module,
        REPO_ROOT / "src/tac/witness_control/segnet_head_feature_cache.py",
        role="tac.witness_control.segnet_head_feature_cache",
    )
    stored_npz_path = _require_exact_module_file(
        stored_npz_module,
        REPO_ROOT / "src/tac/boundary_math/power_diagram_witness.py",
        role="tac.boundary_math.power_diagram_witness",
    )
    admission_guard_path = _require_exact_module_file(
        admission_guard_module,
        REPO_ROOT / "src/tac/admission_guard.py",
        role="tac.admission_guard",
    )
    governed_admission_path = _require_exact_module_file(
        governed_profile_admission_module,
        REPO_ROOT / "src/tac/governed_profile_admission.py",
        role="tac.governed_profile_admission",
    )
    tool_bootstrap_path = _require_exact_module_file(
        tool_bootstrap_module,
        REPO_ROOT / "tools/tool_bootstrap.py",
        role="tools.tool_bootstrap",
    )
    stable_admission = _stable_admission_contract(
        admission_attestation,
        canonical_fresh_rebuild_argv=canonical_fresh_rebuild_argv,
    )
    source_custody = stable_admission.get("source_custody")
    expected_operational_sources = {
        "governed_profile_admission": source_file_row(governed_admission_path),
        "safe_run": source_file_row(REPO_ROOT / "tools/safe_run.py"),
        "admission_guard": source_file_row(admission_guard_path),
    }
    if source_custody != expected_operational_sources or stable_admission.get("outer_resource_caps") != {
        "rss_cap_mb": args.rss_cap_mb,
        "timeout_seconds": float(args.timeout_seconds),
    }:
        raise ProfilerError("safe-run parent attestation/source custody is malformed or stale")
    normalized_creation_storage_identity = json.loads(canonical_json_bytes(dict(creation_storage_identity)))
    stable_preflight = normalized_creation_storage_identity.get("stable_preflight")
    selected_root = stable_preflight.get("selected_root") if isinstance(stable_preflight, dict) else None
    if not isinstance(selected_root, str) or not Path(selected_root).is_absolute():
        raise ProfilerError("identity-bound creation storage root is malformed")
    _validate_creation_storage_identity(
        normalized_creation_storage_identity,
        expected_output_root=Path(selected_root),
    )
    git_head = _git_head()
    value = {
        "schema": PROGRESS_SCHEMA,
        "repository": {
            "git_head": git_head,
            "worktree_cleanliness_required": False,
            "source_byte_hashes_primary": True,
        },
        "sources": {
            "gt_cache": source_file_row(gt_cache),
            "feature_manifest": source_file_row(feature_root / "manifest.json"),
            "profiler_tool": source_file_row(Path(__file__).resolve()),
            "executed_uint8_lattice_feasibility_module": source_file_row(feasibility_path),
            "executed_uint8_lattice_profile_module": source_file_row(profile_path),
            "executed_feature_cache_module": source_file_row(feature_cache_path),
            "executed_feature_extractor_tool": source_file_row(
                REPO_ROOT / "tools/extract_segnet_head_features_n600.py"
            ),
            "executed_stored_npz_module": source_file_row(stored_npz_path),
            "executed_admission_guard_module": source_file_row(admission_guard_path),
            "operational_custody_governed_profile_admission": expected_operational_sources[
                "governed_profile_admission"
            ],
            "operational_custody_safe_run": expected_operational_sources["safe_run"],
            "operational_custody_admission_guard": expected_operational_sources["admission_guard"],
            "executed_tool_bootstrap_module": source_file_row(tool_bootstrap_path),
            **({} if scorer_sources is None else scorer_sources),
        },
        "feature_cache_binding": feature_binding,
        "creation_storage_identity": normalized_creation_storage_identity,
        "resource_custody": {
            "stable_admission_contract": stable_admission,
            "completed_safe_run_status_receipt": None,
            "status_receipt_scope": "PARENT_EMITS_ONLY_AFTER_CHILD_EXIT_SEPARATE_FROM_THIS_RECEIPT",
        },
        "config": {
            "camera_hw": list(EXPECTED_CAMERA_HW),
            "seg_hw": list(EXPECTED_SEG_HW),
            "expected_pairs": EXPECTED_PAIRS,
            "profiled_frame_limit": args.max_frames,
            "mode": args.mode,
            "seed_source_witness": args.seed_source_witness,
            "max_nodes": args.max_nodes,
            "time_limit_seconds_per_block": args.time_limit_seconds_per_block,
            "selector_identity": selector.identity,
            "pose_plugin_identity": plugin.identity,
            "fragile_margin": args.fragile_margin,
            "score_segnet": args.score_segnet,
            "axis": "[macOS-CPU advisory]" if args.score_segnet else "NO_VERDICT_SCORER_CUSTODY",
            "requested_outer_governor_limits": {
                "rss_cap_mb": args.rss_cap_mb,
                "timeout_seconds": args.timeout_seconds,
                "profiler_self_enforced": False,
                "scope": "REQUESTED_OUTER_GOVERNOR_LIMITS_METADATA_ONLY_NOT_ENFORCEMENT_RECEIPT",
                "required_launcher": "tools/safe_run.py process-group/system-memory governor",
            },
            "receiver_stream_codecs": {
                "frame_magic_hex": STREAM_MAGIC.hex(),
                "frame_count_encoding": "uint32_little_endian",
                "candidate_length_encoding": "uint64_little_endian",
                "zlib": dict(ZLIB_STREAM_SETTINGS),
                "brotli": dict(BROTLI_STREAM_SETTINGS),
                "headers_and_termination_included": True,
            },
            "receiver_support": {
                "definition": "UNION_OF_ALL_EXACT_INTEGER_RESIZE_TAP_SUPPORTS",
                "camera_rgb_shape": list(support_union.shape),
                "support_value_count": int(np.count_nonzero(support_union)),
                "outside_support_value_count": int(support_union.size - np.count_nonzero(support_union)),
                "packed_support_sha256": _sha256_bytes(
                    np.packbits(support_union.reshape(-1), bitorder="little").tobytes()
                ),
                "coverage_requirement": "EXACTLY_S_NO_OVERLAP_NO_MISSING_NO_EXCESS",
                "outside_support_fill": {
                    "label": OUTSIDE_SUPPORT_FILL_LABEL,
                    "value": OUTSIDE_SUPPORT_FILL_VALUE,
                },
                "source_seed_equality_scope": "SOURCE_EQUALITY_ON_S_PLUS_EXACT_RESIZE_NUMERATORS",
            },
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": str(Path(sys.executable).resolve()),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "zlib_build": zlib.ZLIB_VERSION,
                "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
                "brotli": brotli_version,
            },
            "determinism": {
                "torch_threads": 1,
                "torch_interop_threads": 1,
                "torch_deterministic_algorithms": True,
            },
        },
    }
    return json.loads(canonical_json_bytes(value))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", required=True, type=Path)
    parser.add_argument("--feature-cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mode",
        choices=(BOUNDS_MODE, ENUMERATED_MODE),
        default=BOUNDS_MODE,
    )
    parser.add_argument("--max-frames", type=int, default=EXPECTED_PAIRS)
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--time-limit-seconds-per-block", type=float)
    parser.add_argument("--seed-source-witness", action="store_true")
    parser.add_argument("--reuse-cache-entries", type=int, default=200_000)
    parser.add_argument("--fragile-margin", type=float, default=1e-5)
    parser.add_argument("--score-segnet", action="store_true")
    parser.add_argument("--rss-cap-mb", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--allow-local-output-for-tests", action="store_true")
    return parser.parse_args(argv)


def _validate_feature_for_request(feature_root: Path, *, max_frames: int) -> Any:
    """Enforce complete-cache authority for the canonical all-n600 request."""

    return validate_feature_cache(
        feature_root,
        require_complete=max_frames == EXPECTED_PAIRS,
    )


def _load_profile_source_module(
    module_name: str,
    source_row: Mapping[str, Any],
    *,
    role: str,
) -> ModuleType:
    path_value = source_row.get("path")
    expected_bytes = source_row.get("bytes")
    expected_sha256 = source_row.get("sha256")
    if not isinstance(path_value, str) or type(expected_bytes) is not int or not isinstance(expected_sha256, str):
        raise ProfilerError(f"{role} first source snapshot row is malformed")
    snapshot = _read_bound_bytes(Path(path_value), name=f"executed {role} source")
    if len(snapshot.payload) != expected_bytes or _sha256_bytes(snapshot.payload) != expected_sha256:
        raise ProfilerError(f"{role} source changed before admitted execution")
    module = ModuleType(module_name)
    module.__file__ = path_value
    module.__package__ = module_name.rpartition(".")[0]
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(snapshot.payload, path_value, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _load_bound_scorer(
    upstream: Path,
    source_snapshot: Mapping[str, Mapping[str, Any]],
    segnet_payload: bytes,
) -> torch.nn.Module:
    expected_paths = {
        "modules": upstream / "modules.py",
        "frame_utils": upstream / "frame_utils.py",
    }
    prepend_paths(upstream)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    frame_utils_module = _load_profile_source_module(
        "frame_utils",
        source_snapshot["executed_frame_utils_py"],
        role="frame_utils",
    )
    modules_module = _load_profile_source_module(
        "modules",
        source_snapshot["executed_modules_py"],
        role="modules",
    )
    scorer_module = _load_profile_source_module(
        "tac.scorer",
        source_snapshot["executed_tac_scorer_py"],
        role="tac.scorer",
    )
    if type(segnet_payload) is not bytes:
        raise ProfilerError("admitted SegNet weight payload must be immutable bytes")
    try:
        from safetensors.torch import load

        admitted_state = load(segnet_payload)
        constructor = getattr(modules_module, "SegNet", None)
        if not callable(constructor):
            raise ProfilerError("admitted modules.py has no callable SegNet constructor")
        scorer = constructor()
        if not isinstance(scorer, torch.nn.Module):
            raise ProfilerError("admitted SegNet constructor returned a non-module")
        scorer.load_state_dict(admitted_state, strict=True)
    except ProfilerError:
        raise
    except BaseException as exc:
        raise ProfilerError("admitted SegNet payload cannot construct the frozen scorer") from exc
    scorer = scorer.to(torch.device("cpu")).eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    realized = scorer.state_dict()
    if set(realized) != set(admitted_state):
        raise ProfilerError("SegNet realized state keys differ from admitted payload")
    for name, expected_tensor in admitted_state.items():
        actual = realized[name].detach().cpu()
        expected_cpu = expected_tensor.detach().cpu()
        if (
            actual.dtype != expected_cpu.dtype
            or actual.shape != expected_cpu.shape
            or not torch.equal(actual, expected_cpu)
        ):
            raise ProfilerError(f"SegNet realized state differs from admitted payload: {name}")
    if scorer.training or any(parameter.requires_grad for parameter in scorer.parameters()):
        raise ProfilerError("byte-fed SegNet did not preserve eval/frozen semantics")
    _require_exact_module_file(modules_module, expected_paths["modules"], role="modules")
    _require_exact_module_file(frame_utils_module, expected_paths["frame_utils"], role="frame_utils")
    _require_exact_module_file(
        scorer_module,
        REPO_ROOT / "src/tac/scorer.py",
        role="tac.scorer",
    )
    return scorer


def _source_block_geometry(
    operator: DisjointResizeOperator,
    source_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_sizes = {len(support.indices) for support in operator.row_supports}
    col_sizes = {len(support.indices) for support in operator.col_supports}
    if len(row_sizes) != 1 or len(col_sizes) != 1:
        raise ProfilerError("bounds mode requires uniform disjoint resize supports")
    row_indices = np.asarray([support.indices for support in operator.row_supports], dtype=np.intp)
    col_indices = np.asarray([support.indices for support in operator.col_supports], dtype=np.intp)
    row_coefficients = np.asarray(
        [support.numerators for support in operator.row_supports],
        dtype=np.int64,
    )
    col_coefficients = np.asarray(
        [support.numerators for support in operator.col_supports],
        dtype=np.int64,
    )
    coefficients = (row_coefficients[:, None, :, None] * col_coefficients[None, :, None, :]).reshape(
        operator.scorer_h, operator.scorer_w, 1, -1
    )
    blocks = source_frame[
        row_indices[:, None, :, None],
        col_indices[None, :, None, :],
        :,
    ]
    witnesses = np.moveaxis(blocks, -1, 2).reshape(
        operator.scorer_h,
        operator.scorer_w,
        source_frame.shape[-1],
        -1,
    )
    targets, _denominator = operator.apply_numerators(source_frame)
    return coefficients, np.asarray(targets, dtype=np.int64), witnesses


def _source_seed_candidate(
    source_frame: np.ndarray,
    *,
    row_indices: Iterable[int],
    column_indices: Iterable[int],
    channel: int,
) -> tuple[int, ...]:
    """Extract one source witness in the canonical support/flatten order."""

    source = np.asarray(source_frame)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 3:
        raise ProfilerError("source seed frame must be HWC uint8 RGB")
    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < 3:
        raise ProfilerError("source seed channel must be an RGB channel index")
    rows = tuple(int(value) for value in row_indices)
    columns = tuple(int(value) for value in column_indices)
    if not rows or not columns:
        raise ProfilerError("source seed support must be nonempty")
    if any(value < 0 or value >= source.shape[0] for value in rows) or any(
        value < 0 or value >= source.shape[1] for value in columns
    ):
        raise ProfilerError("source seed support is outside the camera frame")
    block = source[np.ix_(rows, columns, (channel,))][:, :, 0]
    return tuple(int(value) for value in block.reshape(-1))


def _receiver_support_union(operator: DisjointResizeOperator) -> np.ndarray:
    """Return the exact, non-overlapping camera-byte support union ``S``.

    The disjoint integer resize consumes only these bytes.  A receiver is
    closed when it reconstructs every byte in ``S`` exactly once; camera bytes
    outside ``S`` are deterministically zero-filled and are not source
    residuals.
    """

    coverage_count = np.zeros((operator.camera_h, operator.camera_w, 3), dtype=np.uint16)
    for scorer_row, row_support in enumerate(operator.row_supports):
        rows = tuple(int(value) for value in row_support.indices)
        if not rows or len(set(rows)) != len(rows) or any(not 0 <= value < operator.camera_h for value in rows):
            raise ProfilerError(f"resize row support {scorer_row} is empty, duplicated, or out of range")
        for scorer_col, column_support in enumerate(operator.col_supports):
            columns = tuple(int(value) for value in column_support.indices)
            if (
                not columns
                or len(set(columns)) != len(columns)
                or any(not 0 <= value < operator.camera_w for value in columns)
            ):
                raise ProfilerError(f"resize column support {scorer_col} is empty, duplicated, or out of range")
            index = np.ix_(rows, columns, range(3))
            coverage_count[index] += 1
    if np.any(coverage_count > 1):
        raise ProfilerError("canonical resize supports overlap")
    support = coverage_count == 1
    if not np.any(support):
        raise ProfilerError("canonical resize support union is empty")
    return support


def _validate_receiver_frame(
    frame: np.ndarray,
    *,
    coverage: np.ndarray,
    operator: DisjointResizeOperator,
) -> np.ndarray:
    """Require coverage exactly equal to ``S`` and deterministic fill outside it."""

    decoded = np.asarray(frame)
    observed = np.asarray(coverage)
    expected = _receiver_support_union(operator)
    if decoded.dtype != np.uint8 or decoded.shape != expected.shape:
        raise ProfilerError("receiver frame geometry/dtype drift")
    if observed.dtype != np.bool_ or observed.shape != expected.shape:
        raise ProfilerError("receiver coverage geometry/dtype drift")
    if not np.array_equal(observed, expected):
        raise ProfilerError("receiver coverage is not exactly the canonical resize support union S")
    if np.any(decoded[~expected] != OUTSIDE_SUPPORT_FILL_VALUE):
        raise ProfilerError("receiver bytes outside support S violate deterministic zero fill")
    return expected


def _validate_source_seed_receiver(
    source_frame: np.ndarray,
    decoded_frame: np.ndarray,
    *,
    operator: DisjointResizeOperator,
) -> None:
    """Prove the source-seeded receiver on ``S`` and exact resize numerators."""

    source = np.asarray(source_frame)
    decoded = np.asarray(decoded_frame)
    support = _receiver_support_union(operator)
    if source.dtype != np.uint8 or source.shape != support.shape:
        raise ProfilerError("source-seed validation source geometry/dtype drift")
    synthetic_coverage = support.copy()
    _validate_receiver_frame(decoded, coverage=synthetic_coverage, operator=operator)
    if not np.array_equal(decoded[support], source[support]):
        raise ProfilerError("node-cap-1 source seed differs from source on resize support S")
    source_numerators, source_denominator = operator.apply_numerators(source)
    decoded_numerators, decoded_denominator = operator.apply_numerators(decoded)
    if source_denominator != decoded_denominator or not np.array_equal(source_numerators, decoded_numerators):
        raise ProfilerError("node-cap-1 source seed does not preserve exact resize numerators")


def _decode_frame_candidate_payload(
    candidate_payload: bytes | bytearray | memoryview,
    *,
    operator: DisjointResizeOperator,
    cost_model: SignedResidualCostModel,
) -> _DecodedFramePayload:
    """Strictly parse row wrappers and reassemble canonical camera bytes."""

    if not isinstance(candidate_payload, (bytes, bytearray, memoryview)):
        raise ProfilerError("frame candidate payload must be bytes-like")
    raw = bytes(candidate_payload)
    if len(raw) < 4:
        raise ProfilerError("frame candidate payload row-count header is truncated")
    (row_count,) = struct.unpack_from("<I", raw, 0)
    if row_count != operator.scorer_h:
        raise ProfilerError("frame candidate payload row count mismatches scorer geometry")
    offset = 4
    selected_frame = np.full(
        (operator.camera_h, operator.camera_w, 3),
        OUTSIDE_SUPPORT_FILL_VALUE,
        dtype=np.uint8,
    )
    coverage = np.zeros_like(selected_frame, dtype=bool)
    selected_blocks = 0
    total_blocks = operator.scorer_h * operator.scorer_w * 3
    for scorer_row, row_support in enumerate(operator.row_supports):
        if offset + 4 > len(raw):
            raise ProfilerError(f"frame candidate row {scorer_row} length header is truncated")
        (row_bytes,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        row_end = offset + row_bytes
        if row_end > len(raw):
            raise ProfilerError(f"frame candidate row {scorer_row} exceeds payload")
        try:
            candidates = decode_candidate_stream(raw[offset:row_end], cost_model=cost_model)
        except LatticeProfileError as exc:
            raise ProfilerError(f"frame candidate row {scorer_row} stream is malformed") from exc
        expected_candidates = operator.scorer_w * 3
        if len(candidates) != expected_candidates:
            raise ProfilerError(f"frame candidate row {scorer_row} count mismatches scorer geometry")
        for scorer_col, column_support in enumerate(operator.col_supports):
            expected_arity = len(row_support.indices) * len(column_support.indices)
            for channel in range(3):
                candidate = candidates[scorer_col * 3 + channel]
                if candidate is None:
                    continue
                if len(candidate) != expected_arity:
                    raise ProfilerError(
                        f"frame candidate ({scorer_row},{scorer_col},{channel}) arity mismatches support"
                    )
                index = np.ix_(row_support.indices, column_support.indices, (channel,))
                if np.any(coverage[index]):
                    raise ProfilerError("frame candidate supports overlap during reassembly")
                block = np.asarray(candidate, dtype=np.uint8).reshape(
                    len(row_support.indices),
                    len(column_support.indices),
                )
                selected_frame[index] = block[:, :, None]
                coverage[index] = True
                selected_blocks += 1
        offset = row_end
    if offset != len(raw):
        raise ProfilerError("frame candidate payload has trailing bytes")
    receiver_closed = False
    if selected_blocks == total_blocks:
        _validate_receiver_frame(selected_frame, coverage=coverage, operator=operator)
        receiver_closed = True
    return _DecodedFramePayload(
        selected_frame=selected_frame if receiver_closed else None,
        selected_blocks=selected_blocks,
        total_blocks=total_blocks,
        receiver_closed=receiver_closed,
    )


def _selection_custody(
    *,
    mode: str,
    counters: Mapping[str, int],
    seed_source_witness: bool,
    receiver_closed: bool,
) -> dict[str, Any]:
    """Derive conservative selection/receiver authority labels from counters."""

    if mode not in (BOUNDS_MODE, ENUMERATED_MODE):
        raise ProfilerError("selection custody mode is invalid")
    if type(seed_source_witness) is not bool or type(receiver_closed) is not bool:
        raise ProfilerError("selection custody flags must be boolean")
    if seed_source_witness and mode != ENUMERATED_MODE:
        raise ProfilerError("source-witness seeding is valid only in enumerated_subset mode")
    normalized = {
        name: _nonnegative_receipt_int(counters.get(name), name=f"selection counter {name}") for name in COUNTER_NAMES
    }
    bounded = normalized["bounded_selected_blocks"]
    omitted = normalized["omitted_blocks"]
    selected = normalized["selected_blocks"]
    total = normalized["total_blocks"]
    if mode == BOUNDS_MODE:
        label = "NO_CANDIDATE_STREAM_BOUNDS_ONLY"
        globally_exact = False
    elif bounded:
        label = (
            "KNOWN_SOURCE_WITNESS_SEEDED_CHEAPEST_SEEN_NON_GLOBAL"
            if seed_source_witness
            else "CHEAPEST_SEEN_NON_GLOBAL"
        )
        globally_exact = False
    elif omitted or selected != total:
        label = "INCOMPLETE_CANDIDATE_SELECTION_NO_GLOBAL_CLAIM"
        globally_exact = False
    else:
        label = "PER_BLOCK_RECEIVER_PUBLIC_SELECTOR_MINIMUM_EXACT"
        globally_exact = True
    closed = mode == ENUMERATED_MODE and receiver_closed and omitted == 0 and selected == total
    return {
        "selection_label": label,
        "selection_globally_exact": globally_exact,
        "exact_count_claim": globally_exact,
        "min_description_claim": False,
        "seed_source_witness": seed_source_witness,
        "receiver_non_closure": not closed,
        "pose_bank_wired": False,
        "factor10_solved": False,
        "scope_extrapolation": "NONE_EXACT_FRAME_INDICES_ONLY",
        "per_block_selector_minimum_proved": globally_exact,
        "global_compressed_stream_minimum_claim": False,
    }


def _expected_derivation(*, mode: str, seed_source_witness: bool) -> str:
    if mode == BOUNDS_MODE:
        return "DERIVED_BOUNDS_FROM_REAL_N600_SOURCE_WITNESS"
    if mode != ENUMERATED_MODE or type(seed_source_witness) is not bool:
        raise ProfilerError("derivation requires a typed profiler mode/source-seed flag")
    return "KNOWN_SOURCE_WITNESS_SEEDED_CHEAPEST_SEEN" if seed_source_witness else "EXACT_OR_BOUNDED_ENUMERATED_SUBSET"


def _expected_positive_control(*, mode: str) -> dict[str, Any] | None:
    if mode == BOUNDS_MODE:
        return None
    if mode != ENUMERATED_MODE:
        raise ProfilerError("positive-control reconstruction requires a typed profiler mode")
    positive = noncorner_positive_control()
    if not positive["witness_satisfies"] or positive["any_corner_satisfies"]:
        raise ProfilerError("named noncorner positive control drift")
    result = profile_integer_block(
        positive["coefficients"],
        sum(positive["coefficients"]),
        positive["target_integer"],
        cost_model=SignedResidualCostModel(),
        pose_plugin=NoOpPosePlugin(),
        max_nodes=1000,
    )
    if not result.exhaustive or result.cardinality_lower_bound == 0:
        raise ProfilerError("named noncorner positive control failed")
    return json.loads(canonical_json_bytes({**positive, "profile": result.__dict__}))


def _score_frame_artifacts(
    artifacts: _FrameProfileArtifacts,
    *,
    scorer: torch.nn.Module | None,
    source_frame: np.ndarray | None = None,
) -> dict[str, int]:
    """Score decoded bytes only after a fresh current-scorer source forward."""

    counters = dict(artifacts.counters)
    if scorer is None:
        return counters
    if not artifacts.receiver_closed or artifacts.decoded_frame is None:
        raise ProfilerError("SegNet scoring requires receiver-closed decoded frame bytes")
    source = np.asarray(source_frame)
    if source.dtype != np.uint8 or source.shape != artifacts.decoded_frame.shape:
        raise ProfilerError("SegNet scoring requires the current uint8 source frame")
    # The cache contract is batch-one.  Re-forward source first, then candidate
    # under the same batch-one kernel geometry so batch-dependent reductions
    # cannot counterfeit either cache equality or d_seg.
    predictions: list[np.ndarray] = []
    for frame in (source, artifacts.decoded_frame):
        batch = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        with torch.inference_mode():
            prediction = scorer(scorer.preprocess_input(batch)).argmax(dim=1).cpu().numpy()
        if prediction.shape != (1, *artifacts.labels.shape):
            raise ProfilerError("SegNet scorer output geometry mismatches cached labels")
        predictions.append(prediction[0])
    source_prediction, candidate_prediction = predictions
    if not np.array_equal(source_prediction, artifacts.labels):
        raise ProfilerError("feature-cache labels differ from fresh current frozen-source SegNet argmax")
    counters["segnet_mismatches"] = int(np.count_nonzero(candidate_prediction != source_prediction))
    counters["segnet_pixels"] = int(artifacts.labels.size)
    return counters


def _profile_frame_semantics(
    frame_index: int,
    *,
    source_frame: np.ndarray,
    live_logits: np.ndarray,
    operator: DisjointResizeOperator,
    mode: str,
    seed_source_witness: bool,
    max_nodes: int,
    time_limit_seconds_per_block: float | None,
    fragile_margin: float,
    selector: SignedResidualCostModel,
    selector_identity: str,
    pose_plugin: NoOpPosePlugin,
    pose_plugin_identity: str,
    reuse_cache_entries: int,
    reuse: OrderedDict[str, BlockProfileResult],
    build_selected_frame: bool,
) -> _FrameProfileArtifacts:
    """Derive one frame's scientific state without consulting stage bytes.

    The same helper is the only bounds/enumeration implementation used by the
    normal writer and resume validation.  A node cap is deterministic; a
    wall-clock cap is deliberately refused because it cannot be replayed
    exactly across process and host scheduling changes.
    """

    frame = _nonnegative_receipt_int(frame_index, name="semantic replay frame")
    if mode not in (BOUNDS_MODE, ENUMERATED_MODE):
        raise ProfilerError("semantic replay mode is not a typed profiler mode")
    if type(seed_source_witness) is not bool:
        raise ProfilerError("semantic replay seed-source-witness flag must be boolean")
    if seed_source_witness and mode != ENUMERATED_MODE:
        raise ProfilerError("source-witness seeding is valid only in enumerated_subset mode")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ProfilerError("semantic replay max_nodes must be a positive integer")
    if isinstance(reuse_cache_entries, bool) or not isinstance(reuse_cache_entries, int) or reuse_cache_entries <= 0:
        raise ProfilerError("semantic replay reuse_cache_entries must be a positive integer")
    if time_limit_seconds_per_block is not None:
        raise ProfilerError("nondeterministic wall-clock profile caps are forbidden; use the deterministic node cap")
    if not np.isfinite(fragile_margin) or fragile_margin < 0:
        raise ProfilerError("semantic replay fragile_margin must be finite and nonnegative")
    if type(build_selected_frame) is not bool:
        raise ProfilerError("semantic replay selected-frame request must be boolean")
    if not isinstance(reuse, OrderedDict):
        raise ProfilerError("semantic replay reuse cache must be an OrderedDict")
    if selector.identity != selector_identity:
        raise ProfilerError("semantic replay selector identity does not match the active selector")
    if pose_plugin.identity != pose_plugin_identity:
        raise ProfilerError("semantic replay pose-plugin identity does not match the active plugin")

    source = np.asarray(source_frame)
    logits = np.asarray(live_logits)
    if source.dtype != np.uint8 or source.shape != (operator.camera_h, operator.camera_w, 3):
        raise ProfilerError("semantic replay source-frame geometry/dtype drift")
    if logits.dtype != np.float32 or logits.shape != (
        EXPECTED_CLASSES,
        operator.scorer_h,
        operator.scorer_w,
    ):
        raise ProfilerError("semantic replay live-logit geometry/dtype drift")
    if not np.isfinite(logits).all():
        raise ProfilerError("semantic replay live logits contain non-finite values")

    target_numerators, denominator = operator.apply_numerators(source)
    labels = np.argmax(logits, axis=0).astype(np.int64)
    boundary, fragile = _frame_strata(logits, labels, fragile_margin)
    degenerate = _degenerate_partition_mask(operator)
    partition_custody = _derive_partition_custody(
        frame,
        live_logits=logits,
        source_frame=source,
        operator=operator,
        fragile_margin=fragile_margin,
        degenerate_mask=degenerate,
    )
    aggregate = StreamingProfileAggregator(
        n_classes=EXPECTED_CLASSES,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    counters = dict.fromkeys(COUNTER_NAMES, 0)
    candidate_payload = b""
    lower_bound_method: str | None = None
    selected_frame = np.zeros_like(source) if build_selected_frame and mode == ENUMERATED_MODE else None
    decoded_frame: np.ndarray | None = None
    receiver_closed = False

    if mode == BOUNDS_MODE:
        coefficients, vector_targets, witnesses = _source_block_geometry(operator, source)
        if not np.array_equal(vector_targets, target_numerators):
            raise ProfilerError("vectorized source geometry target drift")
        bounds = vectorized_source_witness_bounds(
            coefficients,
            vector_targets,
            witnesses,
        )
        aggregate.add_bounds_batch(
            target_classes=labels,
            lower_bounds=bounds.cardinality_lower_bound,
            upper_bounds=bounds.cardinality_upper_bound,
            strata={
                "boundary_annulus": boundary,
                "fragile": fragile,
                "degenerate": degenerate,
            },
        )
        counters["total_blocks"] = int(vector_targets.size)
        counters["omitted_blocks"] = int(vector_targets.size)
        if bounds.witness_verified_blocks != counters["total_blocks"]:
            raise ProfilerError("source witness did not count every channel block exactly once")
        lower_bound_method = bounds.lower_bound_method
    else:
        row_streams: list[bytes] = []
        for scorer_row, row_support in enumerate(operator.row_supports):
            encoded_candidates: list[tuple[int, ...] | None] = []
            for scorer_col, col_support in enumerate(operator.col_supports):
                coefficients = tuple(
                    int(value)
                    for value in np.outer(
                        row_support.numerators,
                        col_support.numerators,
                    ).reshape(-1)
                )
                strata: list[str] = []
                if boundary[scorer_row, scorer_col]:
                    strata.append("boundary_annulus")
                if fragile[scorer_row, scorer_col]:
                    strata.append("fragile")
                if degenerate[scorer_row, scorer_col]:
                    strata.append("degenerate")
                channel_results: list[BlockProfileResult] = []
                for channel in range(3):
                    target_integer = int(target_numerators[scorer_row, scorer_col, channel])
                    seed_candidate = (
                        _source_seed_candidate(
                            source,
                            row_indices=row_support.indices,
                            column_indices=col_support.indices,
                            channel=channel,
                        )
                        if seed_source_witness
                        else None
                    )
                    key = profile_cache_key(
                        coefficients=coefficients,
                        denominator=denominator,
                        target_integer=target_integer,
                        selector_identity=selector_identity,
                        pose_plugin_identity=pose_plugin_identity,
                        seed_candidate=seed_candidate,
                    )
                    result = None if seed_source_witness else reuse.get(key)
                    if result is None:
                        result = profile_integer_block(
                            coefficients,
                            denominator,
                            target_integer,
                            cost_model=selector,
                            pose_plugin=pose_plugin,
                            seed_candidate=seed_candidate,
                            max_nodes=max_nodes,
                            time_limit_seconds=None,
                        )
                        if not seed_source_witness:
                            reuse[key] = result
                            if len(reuse) > reuse_cache_entries:
                                reuse.popitem(last=False)
                    else:
                        reuse.move_to_end(key)
                    if seed_source_witness and max_nodes == 1 and result.selected_candidate != seed_candidate:
                        raise ProfilerError("node-cap-1 source seed was not preserved as the selected candidate")
                    channel_results.append(result)
                    encoded_candidates.append(result.selected_candidate)
                    counters["total_blocks"] += 1
                    if result.selected_candidate is None:
                        counters["omitted_blocks"] += 1
                        continue
                    counters["selected_blocks"] += 1
                    if result.exhaustive:
                        counters["exhaustive_selected_blocks"] += 1
                    else:
                        counters["bounded_selected_blocks"] += 1
                    if selected_frame is not None:
                        block = np.asarray(result.selected_candidate, dtype=np.uint8).reshape(
                            len(row_support.indices),
                            len(col_support.indices),
                        )
                        selected_frame[np.ix_(row_support.indices, col_support.indices, (channel,))] = block[:, :, None]
                aggregate.add_pixel(
                    target_class=int(labels[scorer_row, scorer_col]),
                    channel_results=channel_results,
                    strata=strata,
                )
            row_streams.append(encode_candidate_stream(encoded_candidates, cost_model=selector))
        candidate_out = bytearray(struct.pack("<I", len(row_streams)))
        for row_stream in row_streams:
            candidate_out.extend(struct.pack("<I", len(row_stream)))
            candidate_out.extend(row_stream)
        candidate_payload = bytes(candidate_out)
        decoded = _decode_frame_candidate_payload(
            candidate_payload,
            operator=operator,
            cost_model=selector,
        )
        if decoded.selected_blocks != counters["selected_blocks"] or decoded.total_blocks != counters["total_blocks"]:
            raise ProfilerError("decoded candidate counters disagree with in-memory selection")
        receiver_closed = decoded.receiver_closed
        decoded_frame = decoded.selected_frame
        if selected_frame is not None:
            if decoded_frame is None:
                if counters["selected_blocks"] == counters["total_blocks"]:
                    raise ProfilerError("complete in-memory selection did not close through receiver parse-back")
            elif not np.array_equal(decoded_frame, selected_frame):
                raise ProfilerError("receiver-decoded frame differs from in-memory selected frame")
        if seed_source_witness and max_nodes == 1 and (not receiver_closed or decoded_frame is None):
            raise ProfilerError("node-cap-1 source-seeded parse-back is not receiver-closed")
        if seed_source_witness and max_nodes == 1 and decoded_frame is not None:
            _validate_source_seed_receiver(source, decoded_frame, operator=operator)

    aggregate_state = aggregate.state()
    selection_custody = _selection_custody(
        mode=mode,
        counters=counters,
        seed_source_witness=seed_source_witness,
        receiver_closed=receiver_closed,
    )
    return _FrameProfileArtifacts(
        replay=_FrameSemanticReplay(
            partition_custody=partition_custody,
            aggregate_state=aggregate_state,
            candidate_payload=candidate_payload,
            counters=dict(counters),
            selection_custody=selection_custody,
        ),
        aggregate=aggregate,
        counters=counters,
        candidate_payload=candidate_payload,
        lower_bound_method=lower_bound_method,
        labels=labels,
        selected_frame=selected_frame,
        decoded_frame=decoded_frame,
        receiver_closed=receiver_closed,
    )


def _resume_from_stage_chain(
    stages: Path,
    progress_path: Path,
    *,
    identity_hash: str,
    semantic_replay_provider: Callable[[int], _FrameSemanticReplay],
    max_frames: int = EXPECTED_PAIRS,
    progress_snapshot: _ValidatedProgressSnapshot | None = None,
) -> tuple[StreamingProfileAggregator, list[dict[str, Any]], dict[str, int]]:
    if progress_snapshot is not None and type(progress_snapshot) is not _ValidatedProgressSnapshot:
        raise ProfilerError("progress_snapshot must be the exact validated snapshot type or None")
    effective_snapshot = progress_snapshot

    def assert_snapshot_path_unchanged() -> None:
        if effective_snapshot is not None:
            current = _require_local_regular_file(progress_path, name="snapshotted resume progress")
            if _stat_file_identity(current) != effective_snapshot.file_identity:
                raise ProfilerError("resume progress path changed after validated snapshot")

    assert_snapshot_path_unchanged()
    if progress_snapshot is None:
        direct_before = _stat_file_identity(_require_local_regular_file(progress_path, name="direct resume progress"))
        try:
            progress = _load_canonical_object(progress_path, name="resume progress")
        except _MalformedCanonicalJSONError as exc:
            if exc.parsed_value is not None:
                _validate_progress_pointer(
                    exc.parsed_value,
                    identity_hash=identity_hash,
                    max_frames=max_frames,
                )
            raise
        progress_metadata = _require_local_regular_file(progress_path, name="direct resume progress")
        if _stat_file_identity(progress_metadata) != direct_before:
            raise ProfilerError("resume progress path changed during direct canonical read")
        effective_snapshot = _ValidatedProgressSnapshot(
            canonical_payload=canonical_json_bytes(progress),
            file_identity=_stat_file_identity(progress_metadata),
        )
    else:
        progress = progress_snapshot._value()
    progress = _validate_progress_pointer(
        progress,
        identity_hash=identity_hash,
        max_frames=max_frames,
    )
    attempt_custody = _validate_stage_attempt_custody(
        stages.parent,
        identity_sha256=identity_hash,
        exact_rebuild_argv=progress["exact_argv"],
        terminal=False,
    )
    next_frame = progress["next_frame"]
    all_entries = sorted(stages.iterdir()) if stages.exists() else []
    retained_names = set(attempt_custody.proven_stage_retained_names)
    entries = [path for path in all_entries if path.name not in retained_names]
    for entry in entries:
        _require_local_regular_file(entry, name=f"resume stage entry {entry.name}")
    prepared_entries = [
        path for path in entries if re.fullmatch(r"\.frame_[0-9]{4}\.bin\.prepared", path.name) is not None
    ]
    legacy_intent_entries = [path for path in entries if STAGE_INTENT_RE.fullmatch(path.name) is not None]
    intent_entries = legacy_intent_entries + _pending_stage_attempt_paths(
        stages.parent,
        identity_sha256=identity_hash,
        custody=attempt_custody,
    )
    final_entries = [path for path in entries if re.fullmatch(r"frame_[0-9]{4}\.bin", path.name) is not None]
    recognized = set(prepared_entries) | set(legacy_intent_entries) | set(final_entries)
    if set(entries) != recognized:
        raise ProfilerError("resume stage root contains an unidentified or malformed entry")
    if len(intent_entries) > 1:
        raise ProfilerError("resume stage root contains duplicate stage intents")
    paths: list[Path] = []
    for index, path in enumerate(final_entries):
        expected = stages / f"frame_{index:04d}.bin"
        if path != expected:
            raise ProfilerError("resume stage files have a hole or wrong name")
        paths.append(path)
    if len(paths) < next_frame:
        raise ProfilerError("resume stage prefix is shorter than the progress pointer")
    if len(paths) > max_frames:
        raise ProfilerError("recovered stage prefix exceeds requested max_frames")
    aggregate = StreamingProfileAggregator(
        n_classes=EXPECTED_CLASSES,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    receipts: list[dict[str, Any]] = []
    counters = dict.fromkeys(COUNTER_NAMES, 0)
    previous_hash = identity_hash
    pointer_head = identity_hash
    stage_snapshots: list[tuple[Path, BoundFileSnapshot]] = []
    for frame_index, path in enumerate(paths):
        stage_snapshot = _read_bound_bytes(path, name=f"resume stage {path.name}")
        stage_snapshots.append((path, stage_snapshot))
        payload = stage_snapshot.payload
        receipt, candidate_payload = _parse_stage_payload(payload)
        if (
            receipt.get("schema") != STAGE_RECEIPT_SCHEMA
            or receipt.get("identity_sha256") != identity_hash
            or receipt.get("frame") != frame_index
            or receipt.get("previous_stage_sha256") != previous_hash
        ):
            raise ProfilerError(f"resume stage {frame_index} chain mismatch")
        expected_replay = semantic_replay_provider(frame_index)
        assert_snapshot_path_unchanged()
        if not isinstance(expected_replay, _FrameSemanticReplay):
            raise ProfilerError("semantic replay provider returned a malformed result")
        frame_aggregate = _validate_stage_receipt(
            receipt,
            candidate_payload,
            expected_partition_custody=expected_replay.partition_custody,
            expected_aggregate_state=expected_replay.aggregate_state,
            expected_candidate_payload=expected_replay.candidate_payload,
            expected_counters=expected_replay.counters,
            expected_selection_custody=expected_replay.selection_custody,
        )
        assert_snapshot_path_unchanged()
        _assert_snapshot_names_path(path, stage_snapshot, name=f"semantically replayed stage {path.name}")
        aggregate.merge(frame_aggregate)
        frame_counters = receipt["counters"]
        for name, value in frame_counters.items():
            counters[name] += value
        previous_hash = sha256_file(path)
        if frame_index + 1 == next_frame:
            pointer_head = previous_hash
        receipts.append(receipt)
    if progress.get("stage_chain_head_sha256") != pointer_head:
        raise ProfilerError("progress stage-chain head mismatch")
    if intent_entries and not prepared_entries:
        intent_final, intent_frame, intent_attempt, intended_bytes, intended_sha256, intent_snapshot = (
            _parse_stage_intent(intent_entries[0])
        )
        if intent_frame < len(paths):
            if intent_final != paths[intent_frame]:
                raise ProfilerError("stage intent final-path binding mismatch")
            final_payload = _read_bound_bytes(
                paths[intent_frame],
                name="intent-bound committed stage",
            ).payload
            if len(final_payload) != intended_bytes or _sha256_bytes(final_payload) != intended_sha256:
                raise ProfilerError("committed stage differs from its surviving intent")
            assert_snapshot_path_unchanged()
            _finalize_successful_stage_attempt(
                stages=stages,
                final_path=paths[intent_frame],
                intent_path=intent_entries[0],
                intent_snapshot=intent_snapshot,
                identity_sha256=identity_hash,
                exact_rebuild_argv=progress["exact_argv"],
                authorize_mutation=assert_snapshot_path_unchanged,
            )
            intent_entries = []
        else:
            expected_final = stages / f"frame_{next_frame:04d}.bin"
            if (
                intent_final != expected_final
                or intent_frame != next_frame
                or len(paths) != next_frame
                or next_frame >= max_frames
            ):
                raise ProfilerError("orphan stage intent conflicts with the validated prefix/pointer")
            _recover_interrupted_stage(
                stages=stages,
                final_path=expected_final,
                prepared_path=_prepared_stage_path(expected_final),
                intent_path=intent_entries[0],
                frame=next_frame,
                attempt=intent_attempt,
                intended_bytes=intended_bytes,
                intended_sha256=intended_sha256,
                identity_hash=identity_hash,
                exact_rebuild_argv=progress["exact_argv"],
                authorize_mutation=assert_snapshot_path_unchanged,
            )
            if os.path.lexists(atomic_prepared_path(progress_path)):
                assert_snapshot_path_unchanged()
                atomic_json(
                    progress_path,
                    progress,
                    expected_prior_payloads=(canonical_json_bytes(progress) + b"\n",),
                )
            return aggregate, receipts, counters
    if prepared_entries:
        expected_final = stages / f"frame_{next_frame:04d}.bin"
        expected_prepared = _prepared_stage_path(expected_final)
        if (
            len(prepared_entries) != 1
            or prepared_entries[0] != expected_prepared
            or len(paths) != next_frame
            or next_frame >= max_frames
        ):
            raise ProfilerError("prepared stage conflicts with the validated final prefix/pointer")
        prepared_snapshot = _read_bound_bytes(expected_prepared, name="prepared resume stage")
        prepared_payload = prepared_snapshot.payload
        if intent_entries:
            intent_final, intent_frame, intent_attempt, intended_bytes, intended_sha256, intent_snapshot = (
                _parse_stage_intent(intent_entries[0])
            )
            if intent_final != expected_final or intent_frame != next_frame:
                raise ProfilerError("prepared stage intent conflicts with the validated prefix/pointer")
            if len(prepared_payload) < intended_bytes:
                _recover_interrupted_stage(
                    stages=stages,
                    final_path=expected_final,
                    prepared_path=expected_prepared,
                    intent_path=intent_entries[0],
                    frame=next_frame,
                    attempt=intent_attempt,
                    intended_bytes=intended_bytes,
                    intended_sha256=intended_sha256,
                    identity_hash=identity_hash,
                    exact_rebuild_argv=progress["exact_argv"],
                    authorize_mutation=assert_snapshot_path_unchanged,
                )
                if os.path.lexists(atomic_prepared_path(progress_path)):
                    assert_snapshot_path_unchanged()
                    atomic_json(
                        progress_path,
                        progress,
                        expected_prior_payloads=(canonical_json_bytes(progress) + b"\n",),
                    )
                return aggregate, receipts, counters
            if len(prepared_payload) > intended_bytes or _sha256_bytes(prepared_payload) != intended_sha256:
                raise ProfilerError("prepared stage payload differs from its durable intent")
        prepared_receipt, prepared_candidates = _parse_stage_payload(prepared_payload)
        if (
            prepared_receipt.get("schema") != STAGE_RECEIPT_SCHEMA
            or prepared_receipt.get("identity_sha256") != identity_hash
            or prepared_receipt.get("frame") != next_frame
            or prepared_receipt.get("previous_stage_sha256") != previous_hash
        ):
            raise ProfilerError("prepared stage identity/frame/chain mismatch")
        expected_replay = semantic_replay_provider(next_frame)
        assert_snapshot_path_unchanged()
        if not isinstance(expected_replay, _FrameSemanticReplay):
            raise ProfilerError("semantic replay provider returned a malformed result")
        _validate_stage_receipt(
            prepared_receipt,
            prepared_candidates,
            expected_partition_custody=expected_replay.partition_custody,
            expected_aggregate_state=expected_replay.aggregate_state,
            expected_candidate_payload=expected_replay.candidate_payload,
            expected_counters=expected_replay.counters,
            expected_selection_custody=expected_replay.selection_custody,
        )
        if not intent_entries:
            raise ProfilerError("prepared stage lacks its durable write intent")
        if expected_final.exists():
            raise ProfilerError("prepared stage conflicts with an existing final stage")
        assert_snapshot_path_unchanged()
        _replace_bound(
            expected_prepared,
            expected_final,
            prepared_snapshot,
            name="validated prepared stage",
        )
        _fsync_stage_directory(stages)
        assert_snapshot_path_unchanged()
        _finalize_successful_stage_attempt(
            stages=stages,
            final_path=expected_final,
            intent_path=intent_entries[0],
            intent_snapshot=intent_snapshot,
            identity_sha256=identity_hash,
            exact_rebuild_argv=progress["exact_argv"],
            authorize_mutation=assert_snapshot_path_unchanged,
        )
        # Re-enter the ordinary orphan-adoption path after the durable rename.
        return _resume_from_stage_chain(
            stages,
            progress_path,
            identity_hash=identity_hash,
            semantic_replay_provider=semantic_replay_provider,
            max_frames=max_frames,
            progress_snapshot=effective_snapshot,
        )
    if len(paths) > next_frame:
        updated_progress = dict(progress)
        updated_progress.update(
            {
                "status": "complete" if len(paths) == EXPECTED_PAIRS else "partial",
                "next_frame": len(paths),
                "stage_chain_head_sha256": previous_hash,
            }
        )
        assert_snapshot_path_unchanged()
        for stage_path, stage_snapshot in stage_snapshots:
            _assert_snapshot_names_path(
                stage_path,
                stage_snapshot,
                name=f"progress-authorizing stage {stage_path.name}",
            )
        atomic_json(
            progress_path,
            updated_progress,
            expected_prior_payloads=(canonical_json_bytes(progress) + b"\n",),
        )
    elif os.path.lexists(atomic_prepared_path(progress_path)):
        assert_snapshot_path_unchanged()
        atomic_json(
            progress_path,
            progress,
            expected_prior_payloads=(canonical_json_bytes(progress) + b"\n",),
        )
    return aggregate, receipts, counters


def run_profile(args: argparse.Namespace, *, exact_argv: list[str]) -> dict[str, Any]:
    _assert_real_governed_admission(
        allow_local_output_for_tests=getattr(args, "allow_local_output_for_tests", None),
    )
    for name in ("max_frames", "max_nodes", "reuse_cache_entries", "rss_cap_mb", "timeout_seconds"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProfilerError(f"{name} must be a positive integer")
    if type(args.resume) is not bool:
        raise ProfilerError("resume must be boolean")
    if args.max_frames > EXPECTED_PAIRS:
        raise ProfilerError("max_frames exceeds canonical n600 geometry")
    _assert_local_test_scope(
        allow_local_output_for_tests=args.allow_local_output_for_tests,
        max_frames=args.max_frames,
    )
    if not np.isfinite(args.fragile_margin) or args.fragile_margin < 0:
        raise ProfilerError("fragile_margin must be finite and nonnegative")
    if args.time_limit_seconds_per_block is not None:
        raise ProfilerError("nondeterministic wall-clock profile caps are forbidden; use the deterministic node cap")
    if type(args.seed_source_witness) is not bool:
        raise ProfilerError("seed_source_witness must be boolean")
    if args.seed_source_witness and args.mode != ENUMERATED_MODE:
        raise ProfilerError("--seed-source-witness is valid only in enumerated_subset mode")
    if args.score_segnet and args.mode != ENUMERATED_MODE:
        raise ProfilerError("--score-segnet is valid only in enumerated_subset mode")
    attested_argv = _attest_exact_argv(args, exact_argv)
    try:
        parent_attestation = attest_safe_run_parent(
            exact_child_argv=attested_argv,
            rss_cap_mb=args.rss_cap_mb,
            timeout_seconds=args.timeout_seconds,
            repo_root=REPO_ROOT,
        )
    except GovernedAdmissionError as exc:
        raise ProfilerError("production profiling requires exact direct-parent safe-run custody") from exc
    rebuild_argv = _canonical_rebuild_argv(attested_argv, resume=args.resume)
    output_root = _safe_output_root(
        args.output_root,
        allow_local_output_for_tests=args.allow_local_output_for_tests,
        resume=args.resume,
    )
    gt_cache = _resolve_without_symlink_components(args.gt_cache, name="GT cache")
    _require_local_regular_file(gt_cache, name="GT cache")
    feature_root = _resolve_without_symlink_components(
        args.feature_cache_root,
        name="feature cache root",
    )
    _require_local_directory(feature_root, name="feature cache root")
    feature = _validate_feature_for_request(feature_root, max_frames=args.max_frames)
    if feature.live_logits.shape != (EXPECTED_PAIRS, EXPECTED_CLASSES, *EXPECTED_SEG_HW):
        raise ProfilerError("live-logit feature cache geometry drift")
    scorer_sources: dict[str, dict[str, Any]] | None = None
    frozen_scorer_before: _FrozenScorerSnapshot | None = None
    upstream: Path | None = None
    if args.score_segnet:
        if args.upstream_root is None:
            raise ProfilerError("--score-segnet requires --upstream-root")
        upstream = _resolve_without_symlink_components(
            args.upstream_root,
            name="scorer upstream root",
        )
        _require_local_directory(upstream, name="scorer upstream root")
        frozen_scorer_before = _scorer_source_bindings(upstream)
        scorer_sources = frozen_scorer_before.rows
    gt_f1 = open_stored_npy_memmap(gt_cache, "gt_f1")
    if gt_f1.shape != (EXPECTED_PAIRS, *EXPECTED_CAMERA_HW, 3) or gt_f1.dtype != np.uint8:
        raise ProfilerError("gt_f1 real-cache geometry/dtype drift")

    operator = DisjointResizeOperator.build(
        camera_h=EXPECTED_CAMERA_HW[0],
        camera_w=EXPECTED_CAMERA_HW[1],
        scorer_h=EXPECTED_SEG_HW[0],
        scorer_w=EXPECTED_SEG_HW[1],
    )
    selector = SignedResidualCostModel()
    plugin = NoOpPosePlugin()
    reuse: OrderedDict[str, BlockProfileResult] = OrderedDict()
    scorer: torch.nn.Module | None = None

    def semantic_replay_provider(frame_index: int) -> _FrameSemanticReplay:
        artifacts = _profile_frame_semantics(
            frame_index,
            source_frame=np.asarray(gt_f1[frame_index]),
            live_logits=np.asarray(feature.live_logits[frame_index]),
            operator=operator,
            mode=args.mode,
            seed_source_witness=args.seed_source_witness,
            max_nodes=args.max_nodes,
            time_limit_seconds_per_block=args.time_limit_seconds_per_block,
            fragile_margin=args.fragile_margin,
            selector=selector,
            selector_identity=selector.identity,
            pose_plugin=plugin,
            pose_plugin_identity=plugin.identity,
            reuse_cache_entries=args.reuse_cache_entries,
            reuse=reuse,
            build_selected_frame=args.mode == ENUMERATED_MODE,
        )
        replay_counters = _score_frame_artifacts(
            artifacts,
            scorer=scorer,
            source_frame=np.asarray(gt_f1[frame_index]),
        )
        return _FrameSemanticReplay(
            partition_custody=artifacts.replay.partition_custody,
            aggregate_state=artifacts.replay.aggregate_state,
            candidate_payload=artifacts.candidate_payload,
            counters=replay_counters,
            selection_custody=artifacts.replay.selection_custody,
        )

    if args.score_segnet:
        assert upstream is not None
        assert scorer_sources is not None
        assert frozen_scorer_before is not None
        scorer = _load_bound_scorer(upstream, scorer_sources, frozen_scorer_before.segnet_payload)
        frozen_scorer_after = _scorer_source_bindings(upstream)
        scorer_sources = _require_equal_scorer_source_snapshots(
            scorer_sources,
            frozen_scorer_after.rows,
        )
        if frozen_scorer_before.segnet_payload != frozen_scorer_after.segnet_payload:
            raise ProfilerError("frozen SegNet weight payload changed during byte-fed construction")
    # No output preflight, initialization, resume reconciliation, stage write,
    # or receipt mutation is reachable until scorer execution bytes agree.
    current_preflight = _storage_preflight(
        output_root,
        max_frames=args.max_frames,
        allow_local_output_for_tests=args.allow_local_output_for_tests,
        resume=args.resume,
    )
    creation_storage_identity = (
        _load_identity_bound_creation_storage_identity(output_root)
        if args.resume
        else _fresh_creation_storage_identity(
            output_root,
            current_preflight=current_preflight,
        )
    )
    feature_binding = _feature_binding(
        feature,
        gt_cache,
        prefix_frames=args.max_frames,
        scorer_sources=scorer_sources,
    )
    identity = _identity(
        args,
        gt_cache,
        feature_root,
        feature_binding,
        scorer_sources,
        parent_attestation,
        rebuild_argv,
        creation_storage_identity,
    )
    identity_hash = _sha256_bytes(canonical_json_bytes(identity))
    existing_receipt: dict[str, Any] | None = None
    existing_receipt_snapshot: BoundFileSnapshot | None = None
    if args.resume:
        stages, progress_path, certification, progress_snapshot = _validate_resume_root(
            output_root,
            expected_identity=identity,
            expected_rebuild_argv=rebuild_argv,
            max_frames=args.max_frames,
        )
        custody_preflight = certification["storage_preflight"]
    else:
        stages, progress_path = _initialize_fresh_output(
            output_root,
            identity_hash=identity_hash,
            storage_preflight=current_preflight,
            exact_argv=rebuild_argv,
            identity=identity,
        )
        certification = _validate_output_certification(
            output_root,
            final_output_root=output_root,
            expected_identity=identity,
            expected_rebuild_argv=rebuild_argv,
        )
        custody_preflight = certification["storage_preflight"]
        progress_snapshot = _capture_validated_progress_snapshot(
            progress_path,
            identity_hash=identity_hash,
            max_frames=args.max_frames,
        )
    aggregator = StreamingProfileAggregator(
        n_classes=EXPECTED_CLASSES,
        named_strata=("boundary_annulus", "fragile", "degenerate"),
    )
    receipts: list[dict[str, Any]] = []
    totals = dict.fromkeys(COUNTER_NAMES, 0)
    if args.resume:
        aggregator, receipts, totals = _resume_from_stage_chain(
            stages,
            progress_path,
            identity_hash=identity_hash,
            semantic_replay_provider=semantic_replay_provider,
            max_frames=args.max_frames,
            progress_snapshot=progress_snapshot,
        )
        existing_receipt_path = output_root / RECEIPT_NAME
        existing_receipt_prior_payloads = (
            (_read_bound_bytes(existing_receipt_path, name="existing receipt prior").payload,)
            if os.path.lexists(existing_receipt_path)
            else ()
        )
        existing_receipt = _reconcile_committed_atomic_json(
            existing_receipt_path,
            name="existing final receipt",
            expected_prior_payloads=existing_receipt_prior_payloads,
        )
        if existing_receipt is not None:
            _validate_final_receipt(
                existing_receipt,
                output_root=output_root,
                expected_identity=identity,
                expected_rebuild_argv=rebuild_argv,
                semantic_replay_provider=semantic_replay_provider,
            )
            existing_receipt_snapshot = _read_bound_bytes(
                existing_receipt_path,
                name="validated existing receipt authorization prior",
            )

    progress_snapshot = _capture_validated_progress_snapshot(
        progress_path,
        identity_hash=identity_hash,
        max_frames=args.max_frames,
    )

    def authorize_progress_mutation() -> None:
        current = _require_local_regular_file(progress_path, name="active progress authorization")
        if _stat_file_identity(current) != progress_snapshot.file_identity:
            raise ProfilerError("active progress path changed before profile mutation")

    positive_control = _expected_positive_control(mode=args.mode)

    previous_stage_hash = identity_hash if not receipts else sha256_file(stages / f"frame_{len(receipts) - 1:04d}.bin")

    for frame_index in range(len(receipts), args.max_frames):
        started = time.perf_counter()
        frame_artifacts = _profile_frame_semantics(
            frame_index,
            source_frame=np.asarray(gt_f1[frame_index]),
            live_logits=np.asarray(feature.live_logits[frame_index]),
            operator=operator,
            mode=args.mode,
            seed_source_witness=args.seed_source_witness,
            max_nodes=args.max_nodes,
            time_limit_seconds_per_block=args.time_limit_seconds_per_block,
            fragile_margin=args.fragile_margin,
            selector=selector,
            selector_identity=selector.identity,
            pose_plugin=plugin,
            pose_plugin_identity=plugin.identity,
            reuse_cache_entries=args.reuse_cache_entries,
            reuse=reuse,
            build_selected_frame=args.mode == ENUMERATED_MODE,
        )
        frame_partition_custody = frame_artifacts.replay.partition_custody
        frame_aggregate = frame_artifacts.aggregate
        frame_counters = _score_frame_artifacts(
            frame_artifacts,
            scorer=scorer,
            source_frame=np.asarray(gt_f1[frame_index]),
        )
        candidate_payload = frame_artifacts.candidate_payload
        lower_bound_method = frame_artifacts.lower_bound_method
        labels = frame_artifacts.labels
        selection_custody = frame_artifacts.replay.selection_custody

        elapsed = time.perf_counter() - started
        frame_receipt = {
            "schema": STAGE_RECEIPT_SCHEMA,
            "identity_sha256": identity_hash,
            "previous_stage_sha256": previous_stage_hash,
            "frame": frame_index,
            "partition_custody": frame_partition_custody,
            "mode": args.mode,
            "lower_bound_method": lower_bound_method,
            "derivation": _expected_derivation(
                mode=args.mode,
                seed_source_witness=args.seed_source_witness,
            ),
            "aggregate_delta_state": frame_artifacts.replay.aggregate_state,
            "counters": frame_counters,
            "selection_custody": selection_custody,
            "candidate_payload_bytes": len(candidate_payload),
            "candidate_payload_sha256": _sha256_bytes(candidate_payload),
            "timing": _stage_timing(
                wall_seconds=elapsed,
                total_blocks=frame_counters["total_blocks"],
                peak_rss_bytes=_peak_rss_bytes(),
            ),
            "scope": {
                "frame_indices": [frame_index],
                "rgb_channel_blocks": frame_counters["total_blocks"],
                "scorer_pixels": int(labels.size),
                "node_cap": args.max_nodes if args.mode == ENUMERATED_MODE else None,
                "selection_label": selection_custody["selection_label"],
                "receiver_non_closure": selection_custody["receiver_non_closure"],
                "scope_extrapolation": selection_custody["scope_extrapolation"],
            },
        }
        _validate_stage_receipt(
            frame_receipt,
            candidate_payload,
            expected_partition_custody=frame_partition_custody,
            expected_aggregate_state=frame_artifacts.replay.aggregate_state,
            expected_candidate_payload=frame_artifacts.candidate_payload,
            expected_counters=frame_counters,
            expected_selection_custody=selection_custody,
        )
        stage_payload = _stage_payload(frame_receipt, candidate_payload)
        stage_path = stages / f"frame_{frame_index:04d}.bin"
        _atomic_stage(
            stage_path,
            stage_payload,
            identity_sha256=identity_hash,
            exact_rebuild_argv=rebuild_argv,
            authorize_mutation=authorize_progress_mutation,
        )
        previous_stage_hash = sha256_file(stage_path)
        aggregator.merge(frame_aggregate)
        receipts.append(frame_receipt)
        for name, value in frame_counters.items():
            totals[name] += value
        authorize_progress_mutation()
        atomic_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "identity_sha256": identity_hash,
                "status": "complete" if len(receipts) == EXPECTED_PAIRS else "partial",
                "next_frame": len(receipts),
                "stage_chain_head_sha256": previous_stage_hash,
                "storage_preflight": custody_preflight,
                "exact_argv": rebuild_argv,
            },
            expected_prior_payloads=(progress_snapshot.canonical_payload + b"\n",),
        )
        progress_snapshot = _capture_validated_progress_snapshot(
            progress_path,
            identity_hash=identity_hash,
            max_frames=args.max_frames,
        )

    stage_paths = [stages / f"frame_{index:04d}.bin" for index in range(len(receipts))]
    all_receivers_closed = bool(receipts) and all(
        not row["selection_custody"]["receiver_non_closure"] for row in receipts
    )
    final_selection_custody = _selection_custody(
        mode=args.mode,
        counters=totals,
        seed_source_witness=args.seed_source_witness,
        receiver_closed=all_receivers_closed,
    )
    rd_row: dict[str, Any] | None = None
    if args.mode == ENUMERATED_MODE:
        accounting = _stream_accounting(stage_paths)
        rd_row = _reconstruct_rd_row(
            totals=totals,
            stage_receipts=receipts,
            config=identity["config"],
            stream_accounting=accounting,
            feature_cache_binding=feature_binding,
        )
    terminal_custody, terminal_receipts = _terminal_custody(
        output_root,
        expected_identity=identity,
        expected_rebuild_argv=rebuild_argv,
        allow_receipt_scratch=True,
    )
    if terminal_receipts != receipts:
        raise ProfilerError("terminal stage receipts differ from in-memory validated receipts")
    timing_summary = _timing_summary(
        terminal_receipts,
        terminal_stage_chain_sha256=terminal_custody["terminal_stage_chain_sha256"],
    )
    receipt = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "status": "complete" if len(receipts) == EXPECTED_PAIRS else "partial_prefix",
        "scope_label": ("FULL_N600" if len(receipts) == EXPECTED_PAIRS else "HASH_VALID_EXPLICIT_PREFIX"),
        "mode": args.mode,
        "lower_bound_method": LOWER_BOUND_METHOD if args.mode == BOUNDS_MODE else None,
        "derivation": _expected_derivation(
            mode=args.mode,
            seed_source_witness=args.seed_source_witness,
        ),
        "frames_profiled": len(receipts),
        "expected_frames": EXPECTED_PAIRS,
        "profiled_frame_indices": list(range(len(receipts))),
        "scope_extrapolation": "NONE_EXACT_FRAME_INDICES_ONLY",
        "aggregate": aggregator.summary(),
        "rd_row": rd_row,
        "counters_rebuilt_from_hashed_stage_receipts": totals,
        "feature_cache_binding": feature_binding,
        "identity_sha256": identity_hash,
        "git_head": identity["repository"]["git_head"],
        "exact_rebuild_argv": rebuild_argv,
        "requested_outer_governor_limits": identity["config"]["requested_outer_governor_limits"],
        "custody": terminal_custody,
        "timing_summary": timing_summary,
        "claims": {
            "exact_count_claim": final_selection_custody["exact_count_claim"],
            "min_description_claim": final_selection_custody["min_description_claim"],
            "selection_globally_exact": final_selection_custody["selection_globally_exact"],
            "selection_label": final_selection_custody["selection_label"],
            "d_seg_claim": bool(rd_row is not None and rd_row.get("d_seg") is not None),
            "candidate_stream_emitted": args.mode == ENUMERATED_MODE,
            "receiver_non_closure": final_selection_custody["receiver_non_closure"],
            "per_block_selector_minimum_proved": final_selection_custody["per_block_selector_minimum_proved"],
            "global_compressed_stream_minimum_claim": False,
        },
        "positive_control": positive_control,
        "authority": {
            "score_authority": False,
            "promotion_eligible": False,
            "pose_bank_wired": False,
            "factor10_solved": False,
            "global_compressed_stream_minimum_claim": False,
        },
    }
    receipt_path = output_root / RECEIPT_NAME
    if existing_receipt is not None and existing_receipt == receipt:
        return existing_receipt
    authorization = _receipt_transition_authorization(
        desired_receipt=receipt,
        prior_snapshot=existing_receipt_snapshot,
        identity_sha256=identity_hash,
        exact_rebuild_argv=rebuild_argv,
        terminal_stage_chain_sha256=terminal_custody["terminal_stage_chain_sha256"],
        frame_count=len(receipts),
    )
    _authorization_path, authorization_sha256 = _persist_receipt_transition_authorization(
        output_root,
        authorization=authorization,
        authorize_mutation=authorize_progress_mutation,
    )
    authorize_progress_mutation()
    atomic_json(
        receipt_path,
        receipt,
        expected_prior_payloads=(() if existing_receipt_snapshot is None else (existing_receipt_snapshot.payload,)),
        consumer_authorization_sha256=authorization_sha256,
    )
    stored_receipt = _load_canonical_object(receipt_path, name="final receipt")
    _validate_final_receipt(
        stored_receipt,
        output_root=output_root,
        expected_identity=identity,
        expected_rebuild_argv=rebuild_argv,
        semantic_replay_provider=semantic_replay_provider,
    )
    return stored_receipt


def gcd_many(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = int(np.gcd(result, value))
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exact_argv = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    receipt = run_profile(args, exact_argv=exact_argv)
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
