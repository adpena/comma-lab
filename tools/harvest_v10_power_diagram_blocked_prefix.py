#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest the immutable v10 blocked prefix without resuming extraction.

The receipt is advisory post-hoc feature-pullback evidence for canonical frames
0 through 194 only.  It is not n600 evidence, a through-R measurement, an RGB
receiver result, an equivalent-rate comparison, or a contest-score claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import tools.v10_power_diagram_blocked_evidence as evidence  # noqa: E402
from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    open_stored_npy_memmap,
    power_assign,
    sha256_file,
)

SCHEMA: Final = "v10_power_diagram_blocked_prefix_harvest.v1"
RECEIPT_STATUS: Final = "BLOCKED_WITH_POSTHOC_PREFIX_DIAGNOSTIC"
AUTHORITY_LABEL: Final = "ADVISORY_POSTHOC_PREFIX_0_194_OF_600"
NARROW_VERDICT: Final = "FROZEN_HEAD_FLOAT32_POWER_TARGET_POSITIVE_CONTROL_BLOCKED_AT_FRAME_195"
EXPECTED_MEASUREMENT_TOOL_SHA256: Final = "be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9"
EXPECTED_HISTORICAL_CONTAINER_SHA256: Final = "ee13d263b51f210fe7fd7bbfc6a21099260189573fce80715c0d69df0f2ef329"
EXPECTED_TOMBSTONE_SHA256: Final = "fb7114017c735c3ad38f4e4b81a60653910a9ed49dafa47ac47dd42fce05ce76"
EXPECTED_CHECKPOINT_SHA256: Final = "58656d231af5c63b12b3594d8eeeeccf0b2d0f25c09154ef3ef6da759e1fce4b"
EXPECTED_FEATURE_CACHE_SHA256: Final = "59e96781aa1bac153bc8bb277cecdbd4b4e98fdfd41f50aa2294537b90390944"
EXPECTED_PREFIX_FRAMES: Final = 195
EXPECTED_OBSERVED_FRAMES: Final = 196
EXPECTED_PREFIX_SAMPLES: Final = 38_338_560
EXPECTED_BLOCKED_REASON: Final = (
    "positive-control blocker at canonical frame 195: frozen-head power mismatches=1, CPU-Torch forward mismatches=0"
)
RATE_REFERENCES: Final = {
    "optimistic_shared_edge_mdl_contour": 228_764,
    "optimistic_contour_plus_xi": 235_974,
    "strict_sub_0_15_threshold": 225_272,
}


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int

    def receipt_row(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _absolute(raw: Path, *, option: str, must_exist: bool) -> Path:
    if not raw.is_absolute():
        raise ValueError(f"--{option} must be an explicit absolute path: {raw}")
    return raw.resolve(strict=must_exist)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_durable_output(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError(f"receipt path must be absolute: {path}")
    if path.suffix != ".json":
        raise ValueError(f"receipt path must have exact .json suffix: {path}")
    if path.is_symlink():
        raise ValueError(f"receipt path must not be a symlink: {path}")
    try:
        research_root = (REPO_ROOT / ".omx/research").resolve(strict=True)
        resolved_parent = path.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"receipt parent must already exist: {path.parent}") from exc
    if not research_root.is_dir() or not resolved_parent.is_dir():
        raise ValueError("receipt root and parent must be existing directories")
    resolved = resolved_parent / path.name
    if not _is_relative_to(resolved, research_root):
        raise ValueError(f"receipt path must stay beneath the resolved repository research tree: {resolved}")
    if resolved.is_symlink():
        raise ValueError(f"receipt path must not resolve to a symlink: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite an existing durable receipt: {resolved}")
    return resolved


def _verify_file(path: Path, *, expected_sha256: str, role: str) -> VerifiedFile:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} must be a regular non-symlink file: {path}")
    before = path.stat()
    actual_sha256 = sha256_file(path)
    after = path.stat()
    before_fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_fingerprint = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_fingerprint != after_fingerprint:
        raise RuntimeError(f"{role} changed while it was being verified")
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{role} SHA-256 mismatch: actual={actual_sha256}, expected={expected_sha256}")
    return VerifiedFile(
        path=path,
        bytes=after.st_size,
        sha256=actual_sha256,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
    )


def _assert_unchanged(verified: VerifiedFile, *, role: str, verify_hash: bool = False) -> None:
    current = verified.path.stat()
    fingerprint = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    expected = (verified.device, verified.inode, verified.bytes, verified.mtime_ns)
    if fingerprint != expected:
        raise RuntimeError(f"{role} changed after immutable verification")
    if verify_hash and sha256_file(verified.path) != verified.sha256:
        raise RuntimeError(f"{role} content hash changed after immutable verification")


def _current_execution_source_paths() -> dict[str, Path]:
    canonical = {
        "harvester": (REPO_ROOT / "tools/harvest_v10_power_diagram_blocked_prefix.py").resolve(strict=True),
        "blocked_evidence_helper": (REPO_ROOT / "tools/v10_power_diagram_blocked_evidence.py").resolve(strict=True),
    }
    loaded = {
        "harvester": Path(__file__).resolve(strict=True),
        "blocked_evidence_helper": Path(evidence.__file__).resolve(strict=True),
    }
    if loaded != canonical:
        raise RuntimeError(f"current execution source path drift: loaded={loaded}, canonical={canonical}")
    return canonical


def _capture_current_execution_sources() -> dict[str, VerifiedFile]:
    captured: dict[str, VerifiedFile] = {}
    for role, path in _current_execution_source_paths().items():
        captured[role] = _verify_file(
            path,
            expected_sha256=sha256_file(path),
            role=f"current_execution_{role}",
        )
    return captured


def _assert_current_execution_sources_unchanged(captured: dict[str, VerifiedFile]) -> None:
    expected_roles = set(_current_execution_source_paths())
    if set(captured) != expected_roles:
        raise RuntimeError("current execution source capture is incomplete")
    for role, verified in captured.items():
        _assert_unchanged(
            verified,
            role=f"current_execution_{role}",
            verify_hash=True,
        )


def _current_runtime_custody() -> dict[str, Any]:
    return {
        "python": {
            "executable": str(Path(sys.executable).resolve(strict=True)),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
        },
        "platform": {
            "descriptor": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "numpy": {
            "version": np.__version__,
            "float64_dtype": np.dtype(np.float64).str,
            "int64_dtype": np.dtype(np.int64).str,
        },
    }


def _validate_current_execution_custody(receipt: dict[str, Any]) -> None:
    current = receipt.get("current_execution_custody")
    if not isinstance(current, dict) or set(current) != {"source_files", "runtime"}:
        raise ValueError("current execution custody is missing or noncanonical")
    source_rows = current["source_files"]
    canonical_paths = _current_execution_source_paths()
    if not isinstance(source_rows, dict) or set(source_rows) != set(canonical_paths):
        raise ValueError("current execution source custody is missing or noncanonical")
    expected_row_keys = {"path", "bytes", "sha256", "device", "inode", "mtime_ns"}
    for role, expected_path in canonical_paths.items():
        row = source_rows[role]
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError(f"current execution {role} source custody row is noncanonical")
        if row.get("path") != str(expected_path):
            raise ValueError(f"current execution {role} source path drift")
        if (
            type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or type(row.get("device")) is not int
            or row["device"] < 0
            or type(row.get("inode")) is not int
            or row["inode"] < 0
            or type(row.get("mtime_ns")) is not int
            or row["mtime_ns"] < 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])
        ):
            raise ValueError(f"current execution {role} source metadata is noncanonical")
        try:
            verified = _verify_file(
                expected_path,
                expected_sha256=row["sha256"],
                role=f"current_execution_{role}",
            )
        except RuntimeError as exc:
            raise ValueError(f"current execution {role} source custody drift") from exc
        if row != verified.receipt_row():
            raise ValueError(f"current execution {role} source metadata drift")
    expected_runtime = _current_runtime_custody()
    if current["runtime"] != expected_runtime:
        raise ValueError("current execution runtime custody drift")


def _verify_recorded_file(
    row: Any,
    *,
    expected_path: Path,
    expected_sha256: str,
    role: str,
) -> VerifiedFile:
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"checkpoint {role} custody row is noncanonical")
    recorded_path = Path(row["path"])
    if not recorded_path.is_absolute() or recorded_path.resolve(strict=True) != expected_path:
        raise ValueError(f"checkpoint {role} path drift")
    if row["sha256"] != expected_sha256:
        raise ValueError(f"checkpoint {role} hash drift")
    verified = _verify_file(expected_path, expected_sha256=expected_sha256, role=role)
    if isinstance(row["bytes"], bool) or row["bytes"] != verified.bytes:
        raise ValueError(f"checkpoint {role} byte-count drift")
    return verified


def validate_historical_lineage(
    *,
    historical_container: Path,
    historical_manifest: Path,
    current_tombstone: Path,
) -> tuple[dict[str, Any], dict[str, VerifiedFile]]:
    manifest_file = _verify_file(
        historical_manifest,
        expected_sha256=sha256_file(historical_manifest),
        role="historical_manifest",
    )
    try:
        manifest = json.loads(historical_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("historical source manifest is unreadable") from exc
    expected_keys = {
        "schema",
        "historical_checkpoint_path",
        "container_repo_path",
        "container_bytes",
        "container_sha256",
        "compression",
        "decompressed_bytes",
        "decompressed_sha256",
        "reason",
        "status",
        "executable",
        "authorizing",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("historical source manifest keys are noncanonical")
    if manifest["schema"] != "v10_power_diagram_historical_source_container.v2":
        raise ValueError("historical source manifest schema drift")
    recorded_historical_path = manifest["historical_checkpoint_path"]
    if not isinstance(recorded_historical_path, str) or not Path(recorded_historical_path).is_absolute():
        raise ValueError("manifest historical checkpoint path must be an absolute historical string")
    repo_path = Path(manifest["container_repo_path"])
    if repo_path.is_absolute() or ".." in repo_path.parts:
        raise ValueError("manifest container repo path is unsafe")
    if historical_container != (REPO_ROOT / repo_path).resolve(strict=True):
        raise ValueError("historical container path does not match manifest repo path")
    if historical_container.suffix != ".gz" or historical_container.stat().st_mode & 0o111:
        raise ValueError("historical container must be non-source gzip and non-executable")
    expected_compression = {
        "format": "gzip",
        "level": 9,
        "mtime": 0,
        "original_filename_embedded": False,
        "validation": "IN_MEMORY_DECOMPRESSION_ONLY",
    }
    if (
        manifest["container_sha256"] != EXPECTED_HISTORICAL_CONTAINER_SHA256
        or manifest["container_bytes"] != historical_container.stat().st_size
        or manifest["compression"] != expected_compression
        or manifest["decompressed_bytes"] != 62_907
        or manifest["decompressed_sha256"] != EXPECTED_MEASUREMENT_TOOL_SHA256
        or manifest["status"] != "INERT_NON_SOURCE_GZIP_HISTORICAL_EVIDENCE_NON_AUTHORIZING"
        or manifest["executable"] is not False
        or manifest["authorizing"] is not False
        or not isinstance(manifest["reason"], str)
        or not manifest["reason"]
    ):
        raise ValueError("historical container manifest custody drift")
    container_file = _verify_file(
        historical_container,
        expected_sha256=EXPECTED_HISTORICAL_CONTAINER_SHA256,
        role="historical_container",
    )
    container_bytes = historical_container.read_bytes()
    if container_bytes[:4] != b"\x1f\x8b\x08\x00" or container_bytes[4:8] != b"\x00\x00\x00\x00":
        raise ValueError("historical container is not deterministic filename-free gzip")
    try:
        decompressed = gzip.decompress(container_bytes)
    except (EOFError, OSError) as exc:
        raise ValueError("historical gzip container is invalid") from exc
    if len(decompressed) != 62_907 or hashlib.sha256(decompressed).hexdigest() != EXPECTED_MEASUREMENT_TOOL_SHA256:
        raise ValueError("historical container decompressed source custody drift")
    _assert_unchanged(container_file, role="historical_container", verify_hash=True)
    canonical_tombstone = (REPO_ROOT / "tools/measure_v10_power_diagram_generator_byteclose.py").resolve(strict=True)
    if current_tombstone != canonical_tombstone:
        raise ValueError("current tombstone path must be the canonical live tool path")
    tombstone_file = _verify_file(
        current_tombstone,
        expected_sha256=EXPECTED_TOMBSTONE_SHA256,
        role="current_tombstone",
    )
    tombstone_text = current_tombstone.read_text(encoding="utf-8")
    if (
        "RETIRED_UNSAFE_CLEANUP_CERTIFICATE_FAIL_CLOSED" not in tombstone_text
        or "def cleanup_certified_scratch" not in tombstone_text
        or "return refuse" not in tombstone_text
    ):
        raise ValueError("current live tool is not the reviewed fail-closed tombstone")
    return manifest, {
        "historical_manifest": manifest_file,
        "historical_container": container_file,
        "current_tombstone": tombstone_file,
    }


def _validate_identity(
    identity: Any,
    *,
    upstream_root: Path,
    gt_cache: Path,
    historical_manifest: dict[str, Any],
    lineage_files: dict[str, VerifiedFile],
) -> dict[str, VerifiedFile]:
    if not isinstance(identity, dict) or set(identity) != {
        "custody_derivation",
        "custody",
        "geometry",
        "config",
        "implementation",
    }:
        raise ValueError("checkpoint immutable identity keys are noncanonical")
    if identity["custody_derivation"] != evidence.CUSTODY_DERIVATION:
        raise ValueError("checkpoint custody derivation drift")
    expected_geometry = {
        "expected_pairs": 600,
        "seg_hw": [384, 512],
        "camera_hwc": [874, 1164, 3],
        "n_classes": 5,
        "head_rank": 4,
    }
    if identity["geometry"] != expected_geometry:
        raise ValueError("checkpoint immutable geometry drift")
    expected_config = {
        "ridge": 1e-6,
        "batch_size": 1,
        "device": "cpu",
        "dtype": "torch.float32",
        "deterministic_algorithms": True,
        "torch_threads_requested": 6,
        "torch_threads_effective": 6,
        "torch_interop_threads_requested": 18,
        "torch_interop_threads_effective": 18,
    }
    if identity["config"] != expected_config:
        raise ValueError("checkpoint immutable config drift")

    custody = identity["custody"]
    if not isinstance(custody, dict) or set(custody) != {
        "gt_cache",
        "segnet_model",
        "upstream_modules",
        "upstream_frame_utils",
    }:
        raise ValueError("checkpoint input custody keys are noncanonical")
    verified: dict[str, VerifiedFile] = {
        "gt_cache": _verify_recorded_file(
            custody["gt_cache"],
            expected_path=gt_cache,
            expected_sha256=evidence.PINNED_GT_CACHE_SHA256,
            role="gt_cache",
        ),
        "segnet_model": _verify_recorded_file(
            custody["segnet_model"],
            expected_path=(upstream_root / "models" / "segnet.safetensors").resolve(strict=True),
            expected_sha256=evidence.PINNED_SEGNET_SHA256,
            role="segnet_model",
        ),
        "upstream_modules": _verify_recorded_file(
            custody["upstream_modules"],
            expected_path=(upstream_root / "modules.py").resolve(strict=True),
            expected_sha256=evidence.PINNED_MODULES_SHA256,
            role="upstream_modules",
        ),
        "upstream_frame_utils": _verify_recorded_file(
            custody["upstream_frame_utils"],
            expected_path=(upstream_root / "frame_utils.py").resolve(strict=True),
            expected_sha256=evidence.PINNED_FRAME_UTILS_SHA256,
            role="upstream_frame_utils",
        ),
    }

    implementation = identity["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != {
        "tool",
        "power_diagram_witness",
        "factorized_features_loader",
    }:
        raise ValueError("checkpoint implementation custody keys are noncanonical")
    tool_row = implementation["tool"]
    if not isinstance(tool_row, dict) or set(tool_row) != {"path", "sha256"}:
        raise ValueError("checkpoint measurement-tool implementation row is noncanonical")
    if tool_row["path"] != historical_manifest["historical_checkpoint_path"]:
        raise ValueError("checkpoint historical measurement-tool path lineage drift")
    if tool_row["sha256"] != EXPECTED_MEASUREMENT_TOOL_SHA256:
        raise ValueError("checkpoint measurement-tool immutable implementation hash drift")
    verified.update(lineage_files)
    current_implementations = {
        "power_diagram_witness": REPO_ROOT / "src/tac/boundary_math/power_diagram_witness.py",
        "factorized_features_loader": REPO_ROOT / "src/tac/witness_control/factorized_features.py",
    }
    for role, current_path_raw in current_implementations.items():
        row = implementation[role]
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"checkpoint {role} implementation row is noncanonical")
        if not isinstance(row["path"], str) or not Path(row["path"]).is_absolute():
            raise ValueError(f"checkpoint {role} historical path is noncanonical")
        current_path = current_path_raw.resolve(strict=True)
        verified[role] = _verify_file(
            current_path,
            expected_sha256=row["sha256"],
            role=role,
        )
    return verified


def validate_blocked_checkpoint(
    payload: Any,
    *,
    upstream_root: Path,
    gt_cache: Path,
    historical_manifest: dict[str, Any],
    lineage_files: dict[str, VerifiedFile],
) -> tuple[evidence.ExtractionState, dict[str, VerifiedFile]]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint JSON must be an object")
    identity = payload.get("immutable_identity")
    verified = _validate_identity(
        identity,
        upstream_root=upstream_root,
        gt_cache=gt_cache,
        historical_manifest=historical_manifest,
        lineage_files=lineage_files,
    )
    state = evidence.validate_extraction_checkpoint(payload, expected_identity=identity)
    if state.status != "blocked":
        raise ValueError("checkpoint must preserve blocked status")
    if state.next_frame != EXPECTED_PREFIX_FRAMES:
        raise ValueError("checkpoint canonical prefix length drift")
    if state.statistics.sample_count != EXPECTED_PREFIX_SAMPLES:
        raise ValueError("checkpoint prefix sample-count drift")
    if state.positive_power_mismatches != 1 or state.positive_forward_mismatches != 0:
        raise ValueError("checkpoint positive-control mismatch counts drift")
    if state.blocked_reason != EXPECTED_BLOCKED_REASON:
        raise ValueError("checkpoint blocked reason drift")
    return state, verified


def open_and_validate_feature_cache(path: Path) -> np.memmap:
    cache = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(cache, np.memmap):
        raise RuntimeError("feature cache is not a real memory map")
    expected_shape = (600, 384, 512, 4)
    if cache.shape != expected_shape or cache.dtype != np.dtype("<f4"):
        raise ValueError(f"feature-cache geometry drift: {cache.shape}/{cache.dtype}")
    expected_file_bytes = int(cache.offset) + int(cache.nbytes)
    if path.stat().st_size != expected_file_bytes:
        raise ValueError(f"feature-cache byte geometry drift: {path.stat().st_size} != {expected_file_bytes}")
    return cache


def fit_prefix_target(state: evidence.ExtractionState) -> Any:
    weight, bias = state.statistics.solve(1e-6)
    return evidence.affine_scores_to_power_target(
        weight,
        bias,
        adjacency=tuple(sorted(state.adjacency)),
    )


def scan_prefix(
    feature_cache: np.ndarray,
    labels: np.ndarray,
    target: Any,
    *,
    prefix_frames: int = EXPECTED_PREFIX_FRAMES,
    seg_hw: tuple[int, int] = (384, 512),
    head_rank: int = 4,
    n_classes: int = 5,
) -> dict[str, int]:
    if prefix_frames < 1:
        raise ValueError("prefix_frames must be positive")
    expected_feature_shape = (600, *seg_hw, head_rank)
    expected_label_shape = (600, *seg_hw)
    if tuple(feature_cache.shape) != expected_feature_shape:
        raise ValueError("feature-cache shape drift before prefix scan")
    if tuple(labels.shape) != expected_label_shape or labels.dtype.kind not in "iu":
        raise ValueError("GT label cache shape/dtype drift before prefix scan")
    mismatches = 0
    scanned_samples = 0
    for frame_index in range(prefix_frames):
        features = np.asarray(feature_cache[frame_index], dtype=np.float64).reshape(-1, head_rank)
        frame_labels = np.asarray(labels[frame_index]).reshape(-1)
        if not np.isfinite(features).all():
            raise ValueError(f"feature cache has non-finite committed data at frame {frame_index}")
        if np.any(frame_labels < 0) or np.any(frame_labels >= n_classes):
            raise ValueError(f"GT labels are out of range at frame {frame_index}")
        assignments = power_assign(features, target)
        mismatches += int(np.count_nonzero(assignments != frame_labels))
        scanned_samples += int(frame_labels.size)
    expected_samples = prefix_frames * math.prod(seg_hw)
    if scanned_samples != expected_samples:
        raise RuntimeError(f"prefix scan denominator drift: {scanned_samples} != {expected_samples}")
    return {
        "first_frame": 0,
        "last_frame": prefix_frames - 1,
        "frame_count": prefix_frames,
        "sample_count": scanned_samples,
        "mismatch_count": mismatches,
    }


def _rate_comparisons(brotli_bytes: int) -> dict[str, Any]:
    return {
        name: {
            "label": "NON_EQUIVALENT_TARGET_PAYLOAD_VS_FULL_REALIZATION_REFERENCE",
            "reference_bytes": reference,
            "posthoc_prefix_generator_brotli_minus_reference_bytes": brotli_bytes - reference,
            "scope": (
                "post-hoc prefix PDW1 target payload only; spatial quotient field, receiver, "
                "through-R realization, and equivalent-rate authority absent"
            ),
            "equivalent_rate_comparison": False,
        }
        for name, reference in RATE_REFERENCES.items()
    }


def validate_receipt_authority(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != RECEIPT_STATUS:
        raise ValueError("receipt schema/status drift")
    _validate_current_execution_custody(receipt)
    authority = receipt.get("authority")
    required_authority = {
        "evidence_label": AUTHORITY_LABEL,
        "posthoc_prefix_only": True,
        "feature_pullback_only": True,
        "n600_authority": False,
        "through_r_authority": False,
        "rgb_receiver_authority": False,
        "receiver_arithmetic_specified": False,
        "contest_score_authority": False,
        "promotion_eligible": False,
    }
    if not isinstance(authority, dict):
        raise ValueError("receipt authority is missing")
    for key, expected in required_authority.items():
        actual = authority.get(key)
        if (isinstance(expected, bool) and actual is not expected) or (
            not isinstance(expected, bool) and actual != expected
        ):
            raise ValueError(f"receipt authority field {key!r} must be {expected!r}")
    positive_control = receipt.get("positive_control_exposure")
    expected_positive_control = {
        "label": "MEASURED_PRESERVED_BLOCKED_STATE",
        "observed_first_frame": 0,
        "observed_last_frame": 195,
        "observed_frame_count": EXPECTED_OBSERVED_FRAMES,
        "fit_excluded_frame": 195,
        "blocked_reason": EXPECTED_BLOCKED_REASON,
        "power_target_mismatch_count": 1,
        "cpu_torch_forward_mismatch_count": 0,
    }
    positive_integer_fields = (
        "observed_first_frame",
        "observed_last_frame",
        "observed_frame_count",
        "fit_excluded_frame",
        "power_target_mismatch_count",
        "cpu_torch_forward_mismatch_count",
    )
    if (
        not isinstance(positive_control, dict)
        or any(type(positive_control.get(key)) is not int for key in positive_integer_fields)
        or positive_control != expected_positive_control
    ):
        raise ValueError("positive-control exposure custody drift")
    prefix = receipt.get("prefix_measurement", {})
    if prefix.get("label") != AUTHORITY_LABEL:
        raise ValueError("prefix measurement label drift")
    scan = prefix.get("scan", {})
    if type(scan.get("first_frame")) is not int or type(scan.get("last_frame")) is not int:
        raise ValueError("prefix scan frame range drift")
    if scan.get("first_frame") != 0 or scan.get("last_frame") != 194:
        raise ValueError("prefix scan frame range drift")
    if type(scan.get("frame_count")) is not int or type(scan.get("sample_count")) is not int:
        raise ValueError("prefix scan denominator drift")
    if scan.get("frame_count") != 195 or scan.get("sample_count") != EXPECTED_PREFIX_SAMPLES:
        raise ValueError("prefix scan denominator drift")
    scan_mismatches = scan.get("mismatch_count")
    if (
        isinstance(scan_mismatches, bool)
        or not isinstance(scan_mismatches, int)
        or not 0 <= scan_mismatches <= EXPECTED_PREFIX_SAMPLES
    ):
        raise ValueError("prefix scan mismatch count drift")
    statistics = prefix.get("streaming_statistics", {})
    if type(statistics.get("sample_count")) is not int or statistics.get("sample_count") != EXPECTED_PREFIX_SAMPLES:
        raise ValueError("streaming-stat prefix denominator drift")
    mismatch = prefix.get("fitted_feature_pullback_mismatch", {})
    if mismatch.get("label") != "MEASURED_ADVISORY_POSTHOC_PREFIX_FEATURE_PULLBACK":
        raise ValueError("fitted prefix mismatch label drift")
    if type(mismatch.get("denominator")) is not int or mismatch.get("denominator") != EXPECTED_PREFIX_SAMPLES:
        raise ValueError("fitted prefix mismatch denominator drift")
    if type(mismatch.get("numerator")) is not int or mismatch.get("numerator") != scan_mismatches:
        raise ValueError("fitted mismatch numerator disagrees with prefix scan")
    expected_fraction = scan_mismatches / EXPECTED_PREFIX_SAMPLES
    fraction = mismatch.get("fraction")
    if not isinstance(fraction, float) or not math.isfinite(fraction) or fraction != expected_fraction:
        raise ValueError("fitted mismatch fraction arithmetic drift")
    verdict = receipt.get("verdict", {})
    required_verdict = {
        "narrow_verdict": NARROW_VERDICT,
        "family_open": True,
        "paradigm_open": True,
        "equivalent_rate_win_claimed": False,
        "factor_6_complete": False,
        "score_gap_closed": False,
        "score_pointer_move_authorized": False,
        "equation_registry_registration_authorized": False,
        "cleanup_performed": False,
    }
    for key, expected in required_verdict.items():
        actual = verdict.get(key)
        if (isinstance(expected, bool) and actual is not expected) or (
            not isinstance(expected, bool) and actual != expected
        ):
            raise ValueError(f"receipt verdict field {key!r} must be {expected!r}")
    generator = receipt.get("generator", {})
    if generator.get("strict_parseback_byte_identical") is not True:
        raise ValueError("PDW1 strict parse-back is required")
    if generator.get("raw", {}).get("label") != "MEASURED_ACTUAL_PDW1_BYTES":
        raise ValueError("PDW1 raw-byte label drift")
    if generator.get("brotli_quality11", {}).get("label") != ("MEASURED_ACTUAL_BROTLI_QUALITY11_BYTES"):
        raise ValueError("PDW1 Brotli byte label drift")
    if "order0_arithmetic_lower_bound" in generator:
        raise ValueError("retired order-0 lower-bound semantics are forbidden")
    order0 = generator.get("order0_ideal_entropy_estimate", {})
    if (
        order0.get("label") != "DERIVED_OPTIMISTIC_ROUNDED_UP_IDEAL_ENTROPY_BYTES"
        or order0.get("assumptions") != "empirical PMF free; no model/header/termination overhead"
        or type(order0.get("rounded_up_ideal_entropy_bytes")) is not int
        or order0["rounded_up_ideal_entropy_bytes"] < 0
        or "estimated_bytes_ceiling" in order0
    ):
        raise ValueError("PDW1 order-0 ideal entropy semantics drift")
    comparisons = receipt.get("rate_comparison")
    if not isinstance(comparisons, dict) or set(comparisons) != set(RATE_REFERENCES):
        raise ValueError("rate comparison reference set drift")
    brotli_bytes = generator.get("brotli_quality11", {}).get("bytes")
    if isinstance(brotli_bytes, bool) or not isinstance(brotli_bytes, int) or brotli_bytes <= 0:
        raise ValueError("PDW1 Brotli byte count drift")
    for name, reference in RATE_REFERENCES.items():
        row = comparisons[name]
        delta = row.get("posthoc_prefix_generator_brotli_minus_reference_bytes")
        if (
            row.get("label") != "NON_EQUIVALENT_TARGET_PAYLOAD_VS_FULL_REALIZATION_REFERENCE"
            or row.get("reference_bytes") != reference
            or type(delta) is not int
            or delta != brotli_bytes - reference
            or row.get("equivalent_rate_comparison") is not False
        ):
            raise ValueError(f"rate comparison authority drift for {name}")


def atomic_write_json_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    path = validate_durable_output(path)
    fd, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite an existing durable receipt: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--gt-cache", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--historical-container", required=True, type=Path)
    parser.add_argument("--historical-manifest", required=True, type=Path)
    parser.add_argument("--current-tombstone", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def run_harvest(args: argparse.Namespace, *, exact_argv: list[str]) -> dict[str, Any]:
    output = validate_durable_output(args.output)
    current_execution_sources = _capture_current_execution_sources()
    current_runtime = _current_runtime_custody()
    checkpoint = _absolute(args.checkpoint, option="checkpoint", must_exist=True)
    feature_cache_path = _absolute(args.feature_cache, option="feature-cache", must_exist=True)
    gt_cache = _absolute(args.gt_cache, option="gt-cache", must_exist=True)
    upstream_root = _absolute(args.upstream_root, option="upstream-root", must_exist=True)
    historical_container = _absolute(args.historical_container, option="historical-container", must_exist=True)
    historical_manifest_path = _absolute(args.historical_manifest, option="historical-manifest", must_exist=True)
    current_tombstone = _absolute(args.current_tombstone, option="current-tombstone", must_exist=True)
    if not upstream_root.is_dir():
        raise ValueError("--upstream-root must be a directory")
    if checkpoint.parent != feature_cache_path.parent:
        raise ValueError("checkpoint and feature cache must share the preserved scratch directory")
    if checkpoint.name != evidence.PROGRESS_CHECKPOINT_NAME:
        raise ValueError("checkpoint filename is noncanonical")
    if feature_cache_path.name != evidence.FEATURE_CACHE_NAME:
        raise ValueError("feature-cache filename is noncanonical")
    marker = checkpoint.parent / evidence.SCRATCH_MARKER_NAME
    if marker.is_symlink() or not marker.is_file() or marker.read_bytes() != evidence.SCRATCH_MARKER_BYTES:
        raise RuntimeError("preserved scratch marker is absent or invalid")

    historical_manifest, lineage_files = validate_historical_lineage(
        historical_container=historical_container,
        historical_manifest=historical_manifest_path,
        current_tombstone=current_tombstone,
    )

    verified_checkpoint = _verify_file(checkpoint, expected_sha256=EXPECTED_CHECKPOINT_SHA256, role="checkpoint")
    verified_cache = _verify_file(
        feature_cache_path,
        expected_sha256=EXPECTED_FEATURE_CACHE_SHA256,
        role="feature_cache",
    )
    try:
        checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("preserved checkpoint is unreadable") from exc
    state, identity_files = validate_blocked_checkpoint(
        checkpoint_payload,
        upstream_root=upstream_root,
        gt_cache=gt_cache,
        historical_manifest=historical_manifest,
        lineage_files=lineage_files,
    )
    cache = open_and_validate_feature_cache(feature_cache_path)
    labels = open_stored_npy_memmap(gt_cache, "lstars")
    target = fit_prefix_target(state)
    scan = scan_prefix(cache, labels, target)
    del cache
    del labels
    _assert_unchanged(verified_checkpoint, role="checkpoint", verify_hash=True)
    _assert_unchanged(verified_cache, role="feature_cache", verify_hash=True)
    for role, verified in identity_files.items():
        _assert_unchanged(verified, role=role)

    compression = evidence.compression_accounting(target)
    mismatch_count = scan["mismatch_count"]
    _assert_current_execution_sources_unchanged(current_execution_sources)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": RECEIPT_STATUS,
        "created_utc": _utc_now(),
        "exact_argv": exact_argv,
        "paths": {
            "checkpoint": str(checkpoint),
            "feature_cache": str(feature_cache_path),
            "gt_cache": str(gt_cache),
            "upstream_root": str(upstream_root),
            "historical_container": str(historical_container),
            "historical_manifest": str(historical_manifest_path),
            "current_tombstone": str(current_tombstone),
            "output": str(output),
        },
        "current_execution_custody": {
            "source_files": {
                role: verified.receipt_row() for role, verified in sorted(current_execution_sources.items())
            },
            "runtime": current_runtime,
        },
        "authority": {
            "evidence_label": AUTHORITY_LABEL,
            "posthoc_prefix_only": True,
            "feature_pullback_only": True,
            "n600_authority": False,
            "through_r_authority": False,
            "rgb_receiver_authority": False,
            "receiver_arithmetic_specified": False,
            "contest_score_authority": False,
            "promotion_eligible": False,
            "scope_note": (
                "only committed quotient features and labels for frames 0..194 are fitted and "
                "scanned; frame 195 is positive-control exposure only"
            ),
        },
        "custody": {
            "checkpoint": verified_checkpoint.receipt_row(),
            "feature_cache": {
                **verified_cache.receipt_row(),
                "allocation_scope": "preallocated_for_600_frames",
                "committed_scope": "frames_0_through_194_only",
            },
            "scratch_marker": {
                "path": str(marker),
                "bytes_hex": evidence.SCRATCH_MARKER_BYTES.hex(),
            },
            "immutable_identity": checkpoint_payload["immutable_identity"],
            "verified_identity_files": {
                role: verified.receipt_row() for role, verified in sorted(identity_files.items())
            },
        },
        "positive_control_exposure": {
            "label": "MEASURED_PRESERVED_BLOCKED_STATE",
            "observed_first_frame": 0,
            "observed_last_frame": 195,
            "observed_frame_count": EXPECTED_OBSERVED_FRAMES,
            "fit_excluded_frame": 195,
            "blocked_reason": EXPECTED_BLOCKED_REASON,
            "power_target_mismatch_count": 1,
            "cpu_torch_forward_mismatch_count": 0,
        },
        "prefix_measurement": {
            "label": AUTHORITY_LABEL,
            "scan": scan,
            "streaming_statistics": {
                "label": "PRESERVED_FLOAT64_PREFIX_SUFFICIENT_STATISTICS",
                "feature_dim": state.statistics.feature_dim,
                "n_classes": state.statistics.n_classes,
                "sample_count": state.statistics.sample_count,
                "label_counts": state.statistics.label_counts.tolist(),
                "ridge": 1e-6,
                "fit_frames": "0..194",
            },
            "fitted_feature_pullback_mismatch": {
                "label": "MEASURED_ADVISORY_POSTHOC_PREFIX_FEATURE_PULLBACK",
                "numerator": mismatch_count,
                "denominator": EXPECTED_PREFIX_SAMPLES,
                "fraction": mismatch_count / EXPECTED_PREFIX_SAMPLES,
            },
        },
        "generator": compression,
        "rate_comparison": _rate_comparisons(compression["brotli_quality11"]["bytes"]),
        "verdict": {
            "narrow_verdict": NARROW_VERDICT,
            "family_open": True,
            "paradigm_open": True,
            "equivalent_rate_win_claimed": False,
            "factor_6_complete": False,
            "score_gap_closed": False,
            "score_pointer_move_authorized": False,
            "equation_registry_registration_authorized": False,
            "cleanup_performed": False,
        },
    }
    validate_receipt_authority(receipt)
    atomic_write_json_no_overwrite(output, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exact_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
    ]
    run_harvest(args, exact_argv=exact_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
