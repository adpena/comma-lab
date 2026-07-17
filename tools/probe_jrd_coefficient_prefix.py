#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact n=1 receiver oracle for nested LVLS1 coefficient prefixes.

This is a local, no-training rate probe.  It mutates the already-counted
signed-int8 coefficient streams of one sealed fixture, rebuilds the full
``archive.zip`` for exact byte accounting, and evaluates pair 0 through the
shipped receiver, canonical ``R``, frozen CPU SegNet, and frozen CPU PoseNet.
The full archive always retains all 600 pair codes; the receiver-only pair cap
copies the first two code rows and preserves the original code scale exactly.

The machine-oriented stopping idea is cited to Wuyuan Xie et al. (2026), *The
Last Byte: Learning Just Enough for Machine-Oriented Image Compression*, DOI
10.1609/aaai.v40i19.38635.  The analytic Laplace dead-zone family is cited at
its derivation in ``tac.packet_compiler.jrd_coefficient_prefix`` to Shaohui Li
et al. (2023), DOI 10.1109/TCSVT.2022.3229701.  No MVR-Net or paper code is
copied; both scalar maps are clean-room NumPy.

Authority boundary: ``[macOS-CPU advisory]``, ``score_claim=false``, and
``promotion_eligible=false``.  A V9/v8 family verdict requires a sealed V9/v8
payload and full early/boundary/late saved-regime replay; this fixture is only
the frozen v7.5.2 checkpoint staged by the V9 apply-pass dry run.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "upstream", REPO / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.contest_eval_contract import build_upstream_eval_contract  # noqa: E402
from tac.contest_score import compute_contest_score  # noqa: E402
from tac.packet_compiler.jrd_coefficient_prefix import (  # noqa: E402
    MAX_INT8_PREFIX_PLANES,
    PREFIX_FAMILIES,
    CoefficientSection,
    PrefixMeasurement,
    coefficient_sections,
    fit_laplace_histogram,
    generate_prefix_chain,
    read_section,
    replace_section,
    select_best_byte_safe,
    select_last_safe_plane,
)
from tac.witness_control.resume_registry import (  # noqa: E402
    RESUME_REGISTRY_MANIFEST_KEY,
    ResumeIntegrityError,
    ResumeRegistry,
)

SCHEMA_VERSION = "jrd_coefficient_prefix_probe_v1"
FIXTURE_PATH = REPO / (
    "experiments/results/apply_pass_dryrun_20260711T213329Z/frozen_ckpt/"
    "levelset_witness_ema_BEST.npz"
)
FIXTURE_SHA256 = "bbd0567439298c7c6eac236aa7215e39d5190703540eeaf4a71fc4839af931b0"
FIXTURE_BASELINE_BLOB_SHA256 = (
    "ec290072f773859a1129165aec90ec08314aca04c404bd31a949375910e75194"
)
FIXTURE_BASELINE_ARCHIVE_SHA256 = (
    "6bde3b1749c0a790f747e24254d308c45a063bcad7ace1376d4463c29f5ec10f"
)
FIXTURE_BASELINE_ARCHIVE_BYTES = 83_905
SOURCE_VIDEO = REPO / "upstream/videos/0.mkv"
SOURCE_VIDEO_SHA256 = "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
SOURCE_VIDEO_BYTES = 37_545_489
SEGNET_PATH = REPO / "upstream/models/segnet.safetensors"
POSENET_PATH = REPO / "upstream/models/posenet.safetensors"
EVAL_PAIRS = 1
SEG_TOLERANCE = 0.0
POSE_TOLERANCE = 0.0
STATE_PREFIX = "__jrd_coefficient_prefix_probe_"
EXECUTED_STACK_FILES = (
    "tools/levelset_byte_close_and_eval.py",
    "experiments/train_witness_realized_through_R_mlx.py",
    "src/tac/boundary_math/lever_b_levelset_generator.py",
    "src/tac/boundary_math/analytic_lane_render_band.py",
    "src/tac/boundary_math/seg_core.py",
    "src/tac/optimization/frame1_seg_repair_atoms.py",
    "src/tac/local_acceleration/torch_levelset_inflate.py",
    "upstream/modules.py",
    "upstream/frame_utils.py",
)
RECEIVER_ENV_KEYS = (
    "TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS",
    "TAC_DECODE_MEMORY_TIER",
    "TAC_LEVELSET_INFLATE_WORKERS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "TAC_GOVERNED_ADMISSION",
)
INVENTORY_ROOTS = (
    REPO / "experiments/results",
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
PROTECTED_LIVE_RUN = REPO / "experiments/results/v9_cgauge_432_coherent_arm_20260711"


@dataclass(frozen=True)
class BlobParts:
    manifest: dict[str, Any]
    base_raw: bytes
    code_raw: bytes
    pose: bytes
    lane_band: bytes | None
    pose_carrier: bytes | None


@dataclass
class JrdProbeResumeController:
    """Cursor/fingerprint state registered with Pact's canonical resume apparatus."""

    expected_run_fingerprint: str
    stage: str = ""
    cursor: int = 0
    inventory_sha256: str = ""
    event_mode: bool = True

    def state_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        if not self.stage or not self.inventory_sha256:
            raise ResumeIntegrityError("JRD probe cannot persist incomplete resume state")
        return {
            prefix + "schema": np.asarray("jrd_coefficient_prefix_resume.v1"),
            prefix + "run_fingerprint": np.asarray(self.expected_run_fingerprint),
            prefix + "stage": np.asarray(self.stage),
            prefix + "cursor": np.asarray(self.cursor, np.int64),
            prefix + "inventory_sha256": np.asarray(self.inventory_sha256),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict[str, Any]) -> bool:
        names = ("schema", "run_fingerprint", "stage", "cursor", "inventory_sha256")
        present = [prefix + name in cfg for name in names]
        if not any(present):
            return False
        if not all(present):
            raise ResumeIntegrityError("JRD probe resume state is partial or truncated")
        if str(cfg[prefix + "schema"]) != "jrd_coefficient_prefix_resume.v1":
            raise ResumeIntegrityError("JRD probe resume schema changed")
        if str(cfg[prefix + "run_fingerprint"]) != self.expected_run_fingerprint:
            raise ResumeIntegrityError("JRD probe resume fingerprint changed")
        self.stage = str(cfg[prefix + "stage"])
        self.cursor = int(cfg[prefix + "cursor"])
        self.inventory_sha256 = str(cfg[prefix + "inventory_sha256"])
        return True


def _resume_registry(controller: JrdProbeResumeController) -> ResumeRegistry:
    registry = ResumeRegistry()
    registry.register("jrd_coefficient_prefix_probe", STATE_PREFIX, controller)
    return registry


def utc_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def tree_custody(path: Path) -> dict[str, Any]:
    """Content-address every regular file in a durable artifact tree."""

    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise RuntimeError(f"artifact custody refuses symlink: {item}")
        if item.is_file():
            files.append(
                {
                    "relative_path": item.relative_to(path).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return {
        "root": str(path),
        "files": files,
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def local_cpu_axis() -> dict[str, str]:
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        tag = "[macOS-CPU advisory] NON-PROMOTABLE"
        evidence_tag = "[macOS-CPU advisory only]"
    else:
        tag = f"[{system}-{machine}-CPU local advisory] NON-PROMOTABLE"
        evidence_tag = "[advisory only]"
    return {
        "tag": tag,
        "evidence_tag": evidence_tag,
        "system": system,
        "machine": machine,
        "hardware_substrate": f"{system.lower()}_{machine.lower()}_cpu",
    }


def process_command(pid: int) -> dict[str, Any]:
    """Capture a durable best-effort argv receipt for the governed parent."""

    try:
        completed = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "pid_at_capture": pid,
            "command": None,
            "capture_status": "unavailable",
            "capture_error_class": type(exc).__name__,
            "capture_errno": exc.errno,
            "review_status": "UNKNOWN_command_capture_denied",
        }
    command = completed.stdout.strip()
    return {
        "pid_at_capture": pid,
        "command": command or None,
        "capture_status": "measured" if command else "unavailable",
        "capture_returncode": completed.returncode,
        "review_status": "MEASURED_current_process_table" if command else "UNKNOWN",
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def success_only_candidate_scratch(scratch: Path):
    """Retain failed candidate trees; delete only proven-success scratch."""

    temp_root = Path(tempfile.mkdtemp(prefix="candidate_", dir=scratch))
    try:
        yield temp_root
    except BaseException:
        raise
    else:
        shutil.rmtree(temp_root)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_inventory(
    out_dir: Path,
    *,
    run_fingerprint: str,
    include_relative_paths: set[str] | None = None,
) -> dict[str, Any]:
    records = []
    found_paths: set[str] = set()
    for directory in ("candidates", "combined"):
        root = out_dir / directory
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            relative_path = path.relative_to(out_dir).as_posix()
            if (
                include_relative_paths is not None
                and relative_path not in include_relative_paths
            ):
                continue
            row = load_checked_checkpoint(path, run_fingerprint=run_fingerprint)
            found_paths.add(relative_path)
            records.append(
                {
                    "relative_path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "label": row.get("label"),
                }
            )
    if include_relative_paths is not None and found_paths != include_relative_paths:
        missing = sorted(include_relative_paths - found_paths)
        extra = sorted(found_paths - include_relative_paths)
        raise ResumeIntegrityError(
            f"registered checkpoint selection changed; missing={missing}, extra={extra}"
        )
    return {
        "records": records,
        "count": len(records),
        "sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_npz_state(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except Exception as exc:
        raise ResumeIntegrityError(f"cannot load JRD resume sidecar {path}: {exc}") from exc


def _state_cfg(state: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: value.item() if np.asarray(value).shape == () else value
        for key, value in state.items()
    }


def _validated_resume_payload(
    state: dict[str, np.ndarray], *, run_fingerprint: str
) -> tuple[JrdProbeResumeController, dict[str, Any]]:
    cfg = _state_cfg(state)
    controller = JrdProbeResumeController(expected_run_fingerprint=run_fingerprint)
    report = _resume_registry(controller).restore(cfg)
    if not report.restored.get("jrd_coefficient_prefix_probe", False):
        raise ResumeIntegrityError("JRD controller state did not restore")
    raw_inventory = cfg.get("__jrd_checkpoint_inventory_json")
    if not isinstance(raw_inventory, str):
        raise ResumeIntegrityError("JRD resume state lacks its checkpoint inventory payload")
    try:
        inventory = json.loads(raw_inventory)
    except json.JSONDecodeError as exc:
        raise ResumeIntegrityError("JRD checkpoint inventory JSON is invalid") from exc
    records = inventory.get("records")
    if not isinstance(records, list):
        raise ResumeIntegrityError("JRD checkpoint inventory records are missing")
    expected_sha = sha256_bytes(canonical_json_bytes(records))
    if (
        inventory.get("sha256") != expected_sha
        or inventory.get("count") != len(records)
        or controller.inventory_sha256 != expected_sha
        or controller.cursor != len(records)
    ):
        raise ResumeIntegrityError("JRD resume inventory digest/count is internally inconsistent")
    return controller, inventory


def write_resume_state(
    out_dir: Path,
    *,
    run_fingerprint: str,
    stage: str,
    preserve_as: str | None = None,
    verified_checkpoint_paths: set[str] | None = None,
) -> dict[str, Any]:
    inventory = _checkpoint_inventory(
        out_dir,
        run_fingerprint=run_fingerprint,
        include_relative_paths=verified_checkpoint_paths,
    )
    controller = JrdProbeResumeController(
        expected_run_fingerprint=run_fingerprint,
        stage=stage,
        cursor=int(inventory["count"]),
        inventory_sha256=str(inventory["sha256"]),
    )
    arrays = {
        "__jrd_checkpoint_inventory_json": np.asarray(
            canonical_json_bytes(inventory).decode("utf-8")
        )
    }
    arrays.update(_resume_registry(controller).state_arrays())
    if RESUME_REGISTRY_MANIFEST_KEY not in arrays:
        raise ResumeIntegrityError("JRD event-mode state lacks canonical resume manifest")
    latest = out_dir / "resume" / "jrd_probe_latest.npz"
    _atomic_write_npz(latest, arrays)
    if preserve_as is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", preserve_as):
            raise ValueError(f"unsafe preserved resume name: {preserve_as!r}")
        preserved = out_dir / "resume" / f"{preserve_as}.npz"
        if preserved.exists():
            existing = _load_npz_state(preserved)
            existing_controller, _inventory = _validated_resume_payload(
                existing, run_fingerprint=run_fingerprint
            )
            if (
                existing_controller.stage != stage
                or existing_controller.expected_run_fingerprint != run_fingerprint
            ):
                raise ResumeIntegrityError(f"preserved stage checkpoint changed: {preserved}")
        else:
            _atomic_write_npz(preserved, arrays)
    return inventory


def restore_or_initialize_resume(out_dir: Path, *, run_fingerprint: str) -> dict[str, Any]:
    latest = out_dir / "resume" / "jrd_probe_latest.npz"
    actual = _checkpoint_inventory(out_dir, run_fingerprint=run_fingerprint)
    if not latest.exists():
        if actual["count"]:
            raise ResumeIntegrityError(
                "candidate/combined checkpoints exist without the canonical resume sidecar"
            )
        write_resume_state(
            out_dir,
            run_fingerprint=run_fingerprint,
            stage="initialized",
        )
        return {
            "restored": False,
            "recovered": False,
            "recovery_pending": False,
            "registered_checkpoint_paths": [],
            "pending_checkpoint_paths": [],
            **actual,
        }

    state = _load_npz_state(latest)
    controller, stored_inventory = _validated_resume_payload(
        state, run_fingerprint=run_fingerprint
    )
    if controller.cursor < 0:
        raise ResumeIntegrityError("JRD resume cursor is negative")
    pending_paths: list[str] = []
    if (
        controller.inventory_sha256 != actual["sha256"]
        or controller.cursor != actual["count"]
    ):
        stored_by_path = {
            str(row.get("relative_path")): row for row in stored_inventory["records"]
        }
        actual_by_path = {
            str(row.get("relative_path")): row for row in actual["records"]
        }
        if any(actual_by_path.get(path) != row for path, row in stored_by_path.items()):
            raise ResumeIntegrityError(
                "JRD checkpoint inventory mutated or lost a previously registered row"
            )
        if len(actual_by_path) <= len(stored_by_path):
            raise ResumeIntegrityError(
                "JRD checkpoint inventory changed without a complete append-only checkpoint"
            )
        pending_paths = sorted(set(actual_by_path) - set(stored_by_path))
        atomic_write_json(
            out_dir / "resume" / "resume_recovery.json",
            {
                "status": "pending_exact_reverification",
                "reason": (
                    "atomic candidate/combined JSON committed after the preceding registry sidecar; "
                    "each unregistered row remains non-authoritative until its exact receiver and "
                    "scorer measurement is rerun"
                ),
                "run_fingerprint": run_fingerprint,
                "pending_checkpoint_paths": pending_paths,
                "stored": {
                    "stage": controller.stage,
                    "cursor": controller.cursor,
                    "inventory_sha256": controller.inventory_sha256,
                },
                "actual_inventory": actual,
            },
        )
    registered_paths = [
        str(row["relative_path"]) for row in stored_inventory["records"]
    ]
    return {
        "restored": True,
        "recovered": False,
        "recovery_pending": bool(pending_paths),
        "registered_checkpoint_paths": registered_paths,
        "pending_checkpoint_paths": pending_paths,
        "stage": controller.stage,
        **actual,
    }


def _acquire_output_lock(out_dir: Path) -> int:
    lock_path = out_dir / "run.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"another JRD probe invocation owns {lock_path}") from exc
    os.ftruncate(fd, 0)
    os.write(fd, canonical_json_bytes({"pid": os.getpid(), "argv": sys.argv}))
    os.fsync(fd)
    return fd


def refuse_transient_evidence_path(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp")):
        try:
            resolved.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(f"durable receipt path must not be under {forbidden}: {resolved}")
    try:
        resolved.relative_to(PROTECTED_LIVE_RUN.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"receipt path must not touch the protected live run: {resolved}")
    try:
        resolved.relative_to(REPO / "experiments/results")
    except ValueError as exc:
        raise ValueError(
            f"receipt must live under {REPO / 'experiments/results'}; got {resolved}"
        ) from exc


def _family_marker(path: Path) -> str | None:
    normalized = path.as_posix().lower()
    match = re.search(r"(?:^|[/_.-])(v8|v9)(?:$|[/_.-])", normalized)
    return None if match is None else match.group(1)


def _typed_vehicle_provenance(path: Path, *, payload_sha256: str, root: Path) -> dict[str, Any] | None:
    """Resolve only an exact, hash-bound vehicle manifest; path tokens remain advisory."""

    current = path.parent
    root_resolved = root.resolve()
    while True:
        manifest_path = current / "vehicle_provenance.json"
        if manifest_path.is_symlink():
            return None
        if manifest_path.is_file():
            manifest_resolved = manifest_path.resolve()
            try:
                manifest_resolved.relative_to(root_resolved)
            except ValueError:
                return None
            try:
                manifest_resolved.relative_to(PROTECTED_LIVE_RUN.resolve())
            except ValueError:
                pass
            else:
                return None
            payload = load_json(manifest_path)
            family = payload.get("vehicle_family") if isinstance(payload, dict) else None
            if (
                family in {"v8", "v9"}
                and payload.get("payload_sha256") == payload_sha256
                and payload.get("byte_closed") is True
            ):
                return {
                    "vehicle_family": family,
                    "authority": "typed_hash_bound_vehicle_provenance",
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": sha256_file(manifest_path),
                }
        if current.resolve() == root_resolved or current.parent == current:
            break
        current = current.parent
    return None


def measure_negative_control(
    lbc: Any,
    *,
    negative_path: Path,
    zero_blob: bytes,
    run_fingerprint: str,
    out_dir: Path,
    gt: Any,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    """Remeasure the meter canary on every invocation; its JSON has no authority."""

    negative = measure_blob(
        lbc,
        full_blob=zero_blob,
        label="negative_control_all_coefficients_zero",
        run_fingerprint=run_fingerprint,
        out_dir=out_dir,
        gt=gt,
        segnet=segnet,
        posenet=posenet,
    )
    atomic_write_json(negative_path, negative)
    return negative


def validate_negative_control(
    baseline: dict[str, Any], negative: dict[str, Any]
) -> None:
    """Require the destructive canary to move raw bytes and both scorer components."""

    if (
        negative["raw_sha256_local_host"] == baseline["raw_sha256_local_host"]
        or negative["d_seg"] == baseline["d_seg"]
        or negative["d_pose"] == baseline["d_pose"]
    ):
        raise RuntimeError(
            "negative control did not separate raw output plus both scorer components"
        )


def build_payload_inventory(roots: tuple[Path, ...] = INVENTORY_ROOTS) -> dict[str, Any]:
    """Scan byte-closed LVLS1 candidates without reading the protected live run."""

    records: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    skipped_symlink_candidates: list[dict[str, str]] = []
    skipped_names = {".git", ".venv", "node_modules", "__pycache__", "scratch"}
    protected = PROTECTED_LIVE_RUN.resolve()
    for root in roots:
        if not root.is_dir():
            root_rows.append({"path": str(root), "exists": False})
            continue
        root_rows.append({"path": str(root.resolve()), "exists": True})
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if current.resolve() == protected:
                dirnames[:] = []
                continue
            dirnames[:] = [
                name
                for name in dirnames
                if name not in skipped_names
                and (current / name).resolve() != protected
            ]
            marker = _family_marker(current)
            for name in sorted(filenames):
                path = current / name
                is_candidate_name = name in {"0.bin", "archive.zip"} or (
                    marker is not None
                    and name.startswith("levelset_witness_")
                    and name.endswith(".npz")
                )
                if path.is_symlink():
                    if is_candidate_name:
                        skipped_symlink_candidates.append(
                            {
                                "path": str(path.absolute()),
                                "reason": "candidate symlink refused without resolving or reading target",
                            }
                        )
                    continue
                kind = None
                lvls1 = False
                payload_sha = None
                if name == "0.bin":
                    kind = "lvls1_blob"
                    with path.open("rb") as handle:
                        lvls1 = handle.read(6) == b"LVLS1\x00"
                    payload_sha = sha256_file(path)
                elif name == "archive.zip":
                    kind = "archive_zip"
                    try:
                        with zipfile.ZipFile(path) as archive:
                            member = next(
                                (item for item in archive.namelist() if Path(item).name == "0.bin"),
                                None,
                            )
                            lvls1 = bool(
                                member is not None and archive.read(member)[:6] == b"LVLS1\x00"
                            )
                    except (OSError, zipfile.BadZipFile):
                        lvls1 = False
                    payload_sha = sha256_file(path)
                elif (
                    marker is not None
                    and name.startswith("levelset_witness_")
                    and name.endswith(".npz")
                ):
                    kind = "checkpoint_not_byte_closed"
                    payload_sha = sha256_file(path)
                if kind is None:
                    continue
                typed_provenance = (
                    None
                    if payload_sha is None
                    else _typed_vehicle_provenance(
                        path, payload_sha256=payload_sha, root=root
                    )
                )
                records.append(
                    {
                        "path": str(path.resolve()),
                        "path_family_marker": marker,
                        "path_family_marker_review_status": (
                            None if marker is None else "INFERRED_UNREVIEWED_advisory_only"
                        ),
                        "typed_vehicle_provenance": typed_provenance,
                        "kind": kind,
                        "bytes": path.stat().st_size,
                        "sha256": payload_sha,
                        "lvls1_magic": lvls1,
                        "eligible_nonlive_byte_closed": bool(
                            lvls1
                            and kind in {"lvls1_blob", "archive_zip"}
                            and typed_provenance is not None
                        ),
                    }
                )
    payload = {
        "schema": "jrd_v9_v8_payload_inventory.v1",
        "stores_consulted": (
            "filesystem experiments/results, /Volumes/VertigoDataTier/pact, and "
            "/Volumes/APDataStore/pact"
        ),
        "deliberately_not_loaded": [
            {
                "path": str(PROTECTED_LIVE_RUN),
                "reason": "operator-protected live V9 run",
            }
        ],
        "roots": root_rows,
        "skipped_symlink_candidates": sorted(
            skipped_symlink_candidates, key=lambda row: row["path"]
        ),
        "records": sorted(records, key=lambda row: row["path"]),
    }
    payload["eligible_count"] = sum(
        bool(row["eligible_nonlive_byte_closed"]) for row in payload["records"]
    )
    payload["unclassified_lvls1_count"] = sum(
        bool(row["lvls1_magic"] and row["typed_vehicle_provenance"] is None)
        for row in payload["records"]
    )
    payload["inventory_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def load_byte_close_module():
    import levelset_byte_close_and_eval as lbc

    return lbc


def parse_blob(lbc: Any, blob: bytes) -> BlobParts:
    import brotli

    manifest, base_b, code_b, pose, lane_band, pose_carrier, _chart = lbc._read_blob_bytes(blob)  # (#497) 7th chart block: mechanical unpack update
    return BlobParts(
        manifest=manifest,
        base_raw=brotli.decompress(base_b),
        code_raw=brotli.decompress(code_b),
        pose=pose,
        lane_band=lane_band,
        pose_carrier=pose_carrier,
    )


def repack_blob(lbc: Any, parts: BlobParts, *, base_raw: bytes, code_raw: bytes) -> bytes:
    """Repack through the canonical LVLS1 grammar with unchanged manifest/scales."""

    import brotli

    manifest_json = json.dumps(parts.manifest, separators=(",", ":")).encode("utf-8")
    return lbc._io_pack(
        manifest_json,
        brotli.compress(base_raw, quality=11),
        brotli.compress(code_raw, quality=11),
        parts.pose,
        parts.lane_band,
        parts.pose_carrier,
    )


def exact_pair_cap_blob(lbc: Any, full_blob: bytes, *, eval_pairs: int = EVAL_PAIRS) -> tuple[bytes, dict[str, Any]]:
    """Cap pair count without requantizing code or changing ``code_scale``.

    The canonical search harness's generic ``run_inflate(max_pairs=...)``
    re-quantizes the sliced code and derives a new scale.  That is appropriate
    for a speed preview but is not the first-pair receiver of the full archive.
    Here the first ``2*eval_pairs`` signed-int8 code rows are copied byte for
    byte and the original scale is retained, so the capped receiver emits the
    same pair-local render as the full payload.
    """

    import brotli

    manifest, base_b, code_b, pose, lane_b, pcar_b, _chart = lbc._read_blob_bytes(full_blob)
    original_manifest = copy.deepcopy(manifest)
    n_pairs = int(manifest["n_pairs"])
    if eval_pairs <= 0 or eval_pairs > n_pairs:
        raise ValueError(f"eval_pairs must be in [1,{n_pairs}]; got {eval_pairs}")
    code_shape = tuple(int(x) for x in manifest["code_shape"])
    if len(code_shape) != 2 or code_shape[0] != 2 * n_pairs:
        raise ValueError(
            f"code_shape {code_shape} is not the required (2*n_pairs, mod_dim) layout"
        )
    code_raw = brotli.decompress(code_b)
    expected = int(np.prod(code_shape))
    if len(code_raw) != expected:
        raise ValueError(f"code stream has {len(code_raw)} int8 values; manifest requires {expected}")
    cap_count = 2 * eval_pairs * code_shape[1]
    code_cap_raw = code_raw[:cap_count]
    manifest = copy.deepcopy(manifest)
    manifest["n_pairs"] = eval_pairs
    manifest["code_shape"] = [2 * eval_pairs, code_shape[1]]

    lane_cap = lane_b
    if lane_b is not None:
        if original_manifest.get("lane_render_band") is None:
            raise ValueError("lane bytes exist without lane_render_band manifest")
        lane_pairs, lane_header = lbc.deserialize_lane_band_any(brotli.decompress(lane_b))
        lane_cfg = lbc.render_config_from_header(
            {**original_manifest["lane_render_band"], "pairs": []}
        )
        lane_cap = brotli.compress(
            lbc.serialize_lane_band_any(lane_pairs[:eval_pairs], lane_cfg, lane_header),
            quality=11,
        )

    pcar_cap = pcar_b
    if pcar_b is not None:
        if original_manifest.get("pose_carrier") is None:
            raise ValueError("pose-carrier bytes exist without pose_carrier manifest")
        pcar_cap = lbc._cap_pose_carrier(pcar_b, eval_pairs)
        manifest["pose_carrier"] = {
            **original_manifest["pose_carrier"],
            "n_pairs": eval_pairs,
        }

    manifest_json = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    capped = lbc._io_pack(
        manifest_json,
        base_b,
        brotli.compress(code_cap_raw, quality=11),
        pose,
        lane_cap,
        pcar_cap,
    )
    reparsed, _base2, code2, _pose2, _lane2, _pcar2, _chart2 = lbc._read_blob_bytes(capped)
    roundtrip_code_raw = brotli.decompress(code2)
    proof = {
        "eval_pairs": eval_pairs,
        "source_pairs": n_pairs,
        "source_code_shape": list(code_shape),
        "capped_code_shape": reparsed["code_shape"],
        "code_scale_before": float(original_manifest["code_scale"]),
        "code_scale_after": float(reparsed["code_scale"]),
        "code_scale_unchanged": bool(
            float(original_manifest["code_scale"]) == float(reparsed["code_scale"])
        ),
        "code_prefix_sha256_expected": sha256_bytes(code_cap_raw),
        "code_prefix_sha256_roundtrip": sha256_bytes(roundtrip_code_raw),
        "code_prefix_exact": bool(code_cap_raw == roundtrip_code_raw),
        "pose_sidecar_sha256_before_after": [sha256_bytes(pose), sha256_bytes(pose)],
        "lane_band_present": lane_b is not None,
        "lane_band_sha256_before": None if lane_b is None else sha256_bytes(lane_b),
        "lane_band_sha256_after": None if lane_cap is None else sha256_bytes(lane_cap),
        "pose_carrier_present": pcar_b is not None,
        "pose_carrier_sha256_before": None if pcar_b is None else sha256_bytes(pcar_b),
        "pose_carrier_sha256_after": None if pcar_cap is None else sha256_bytes(pcar_cap),
        "pair_locality_basis": (
            "LVLS1 receiver indexes code[2*pi+frame]; base weights are shared and optional "
            "pair tables are sliced to the same prefix"
        ),
    }
    if not proof["code_scale_unchanged"] or not proof["code_prefix_exact"]:
        raise RuntimeError("pair-cap proof failed: code scale or signed-int8 prefix changed")
    return capped, proof


def read_pair_raw(path: Path, lbc: Any) -> tuple[np.ndarray, np.ndarray, str]:
    expected = 2 * lbc.CAMERA_H * lbc.CAMERA_W * 3
    payload = path.read_bytes()
    if len(payload) != expected:
        raise ValueError(f"receiver raw has {len(payload)} bytes; expected {expected}")
    frames = np.frombuffer(payload, dtype=np.uint8).reshape(
        2, lbc.CAMERA_H, lbc.CAMERA_W, 3
    )
    return frames[0].copy(), frames[1].copy(), sha256_bytes(payload)


def scorer_measurement(lbc: Any, *, raw_path: Path, gt: Any, segnet: Any, posenet: Any) -> dict[str, float | str]:
    f0, f1, raw_sha = read_pair_raw(raw_path, lbc)
    d_seg = float(lbc.twr.cpu_verdict_d_seg_batch(segnet, [f1], [gt.lstars[0]])[0])
    d_pose = float(
        lbc.twr.cpu_verdict_d_pose_batch(posenet, [f0], [f1], [gt.gt_poses[0]])[0]
    )
    if not np.isfinite(d_seg) or not np.isfinite(d_pose):
        raise RuntimeError("frozen scorer returned a nonfinite component")
    return {"d_seg": d_seg, "d_pose": d_pose, "raw_sha256_local_host": raw_sha}


def oracle_receiver_check(lbc: Any, *, capped_blob: bytes, raw_path: Path) -> dict[str, Any]:
    manifest, params, code, lane_pairs, pose_carrier, _chart = lbc._dequant_blob(capped_blob)
    reference, _argmax = lbc.numpy_oracle_reference_frames(
        params,
        code,
        manifest,
        EVAL_PAIRS,
        lane_pairs,
        pose_carrier=pose_carrier,
    )
    f0, f1, _sha = read_pair_raw(raw_path, lbc)
    equals = [bool(np.array_equal(f0, reference[0])), bool(np.array_equal(f1, reference[1]))]
    max_abs = [
        int(np.max(np.abs(f0.astype(np.int16) - reference[0].astype(np.int16)))),
        int(np.max(np.abs(f1.astype(np.int16) - reference[1].astype(np.int16)))),
    ]
    result = {
        "frame_equal": equals,
        "all_equal": bool(all(equals)),
        "max_abs_uint8": max_abs,
        "authority": "canonical numpy-fp32 receiver plus canonical R",
    }
    if not result["all_equal"]:
        raise RuntimeError(f"shipped receiver differs from canonical oracle: {result}")
    return result


def cleanup_scratch(out_dir: Path, *, run_fingerprint: str) -> None:
    """Refuse crash-left scratch; normal scratch is success-cleaned by its context manager."""

    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    contract = scratch / "cleanup_contract.json"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "original_path": str(scratch),
        "reason_rebuildable": (
            "candidate packet/raw scratch is deleted only by TemporaryDirectory on normal scope "
            "exit; crash-left bytes are retained and block until separately certified"
        ),
        "destructive_scope": "none at startup or recovery",
    }
    if contract.exists() and load_json(contract) != expected:
        raise RuntimeError("scratch cleanup contract does not match this run; refusing deletion")
    atomic_write_json(contract, expected)
    orphans = [child for child in sorted(scratch.glob("candidate_*")) if child.is_dir()]
    if orphans:
        blocker = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "blocker": "uncertified_crash_left_candidate_scratch",
            "run_fingerprint": run_fingerprint,
            "cwd": str(REPO),
            "argv": list(sys.argv),
            "relevant_env": {key: os.environ.get(key) for key in RECEIVER_ENV_KEYS},
            "retained_artifacts": [tree_custody(child) for child in orphans],
            "reason": (
                "the creating invocation did not atomically certify these trees as completed; "
                "certify or cold-store them before retrying"
            ),
        }
        atomic_write_json(scratch / "orphan_blocker.json", blocker)
        raise RuntimeError(
            "uncertified crash-left candidate scratch retained; see scratch/orphan_blocker.json"
        )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def run_fingerprint_payload(
    lbc: Any | None = None, *, upstream_contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    resolved_lbc = load_byte_close_module() if lbc is None else lbc
    contract = (
        build_upstream_eval_contract(repo_root=REPO)
        if upstream_contract is None
        else upstream_contract
    )
    if not contract.get("contract_valid"):
        raise RuntimeError(f"upstream evaluator contract invalid: {contract.get('blockers')}")
    if os.environ.get("TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS", ""):
        raise RuntimeError("unsafe archive-group bypass environment is set; refusing exact probe")
    stack_hashes = {}
    for relative in EXECUTED_STACK_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(f"executed receiver/scorer dependency missing: {path}")
        stack_hashes[relative] = sha256_file(path)
    import av

    av_library_versions = {
        name: list(version) if isinstance(version, tuple) else version
        for name, version in av.library_versions.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "prefix_core_sha256": sha256_file(
            REPO / "src/tac/packet_compiler/jrd_coefficient_prefix.py"
        ),
        "fixture_path": str(FIXTURE_PATH.relative_to(REPO)),
        "fixture_sha256": FIXTURE_SHA256,
        "source_video_path": str(SOURCE_VIDEO.relative_to(REPO)),
        "source_video_sha256": SOURCE_VIDEO_SHA256,
        "segnet_sha256": sha256_file(SEGNET_PATH),
        "posenet_sha256": sha256_file(POSENET_PATH),
        "executed_stack_sha256": stack_hashes,
        "shipped_inflate_source_sha256": sha256_bytes(
            str(resolved_lbc._INFLATE_PY).encode("utf-8")
        ),
        "upstream_eval_contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "torch_distribution": _package_version("torch"),
            "timm_distribution": _package_version("timm"),
            "segmentation_models_pytorch_distribution": _package_version(
                "segmentation-models-pytorch"
            ),
            "einops_distribution": _package_version("einops"),
            "safetensors_distribution": _package_version("safetensors"),
            "torchvision_distribution": _package_version("torchvision"),
            "brotli_distribution": _package_version("brotli"),
            "av_distribution": _package_version("av"),
            "av_library_versions": av_library_versions,
            "relevant_env": {key: os.environ.get(key) for key in RECEIVER_ENV_KEYS},
        },
        "eval_pairs": EVAL_PAIRS,
        "seg_tolerance": SEG_TOLERANCE,
        "pose_tolerance": POSE_TOLERANCE,
        "families": list(PREFIX_FAMILIES),
        "planes": list(range(MAX_INT8_PREFIX_PLANES + 1)),
        "self_orient_defaults": {
            "freq_across": 32.0,
            "freq_along": 4.0,
            "tau": 4.0,
            "iters": 4,
        },
    }


def candidate_key(*, section: str, family: str, bits_removed: int) -> str:
    digest = hashlib.sha256(section.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{family}_k{bits_removed}"


def candidate_checkpoint_path(
    out_dir: Path, *, section: str, family: str, bits_removed: int
) -> Path:
    return out_dir / "candidates" / f"{candidate_key(section=section, family=family, bits_removed=bits_removed)}.json"


def load_checked_checkpoint(
    path: Path,
    *,
    run_fingerprint: str,
    expected_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = load_json(path)
    if row.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"checkpoint fingerprint mismatch: {path}")
    if row.get("status") != "complete":
        raise RuntimeError(f"checkpoint is not complete: {path}")
    for key, expected in (expected_fields or {}).items():
        if row.get(key) != expected:
            raise RuntimeError(
                f"checkpoint identity mismatch for {key}: {path} has {row.get(key)!r}, "
                f"expected {expected!r}"
            )
    return row


def load_registered_checkpoint_or_none(
    path: Path,
    *,
    out_dir: Path,
    verified_checkpoint_paths: set[str],
    run_fingerprint: str,
    expected_fields: dict[str, Any],
) -> dict[str, Any] | None:
    """Return only a registry-authenticated row; crash extras require remeasurement."""

    relative_path = path.relative_to(out_dir).as_posix()
    if relative_path not in verified_checkpoint_paths:
        return None
    if not path.is_file():
        raise ResumeIntegrityError(
            f"registered checkpoint disappeared before reuse: {relative_path}"
        )
    return load_checked_checkpoint(
        path,
        run_fingerprint=run_fingerprint,
        expected_fields=expected_fields,
    )


def combined_step_gate(
    measured: dict[str, Any],
    *,
    baseline: dict[str, Any],
    current_archive_bytes: int,
) -> dict[str, bool]:
    """Apply both noncompensating component safety and strict ZIP shrinkage."""

    values = {
        "measured.d_seg": float(measured["d_seg"]),
        "measured.d_pose": float(measured["d_pose"]),
        "baseline.d_seg": float(baseline["d_seg"]),
        "baseline.d_pose": float(baseline["d_pose"]),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"combined-step component values must be finite/non-negative: {values}")
    measured_bytes_value = measured["archive_zip_bytes"]
    if (
        isinstance(measured_bytes_value, bool)
        or not isinstance(measured_bytes_value, int)
        or isinstance(current_archive_bytes, bool)
        or not isinstance(current_archive_bytes, int)
    ):
        raise ValueError("combined-step archive byte counts must be integers")
    measured_bytes = measured_bytes_value
    if measured_bytes < 0 or current_archive_bytes < 0:
        raise ValueError("combined-step archive byte counts must be non-negative")
    safe = bool(
        values["measured.d_seg"] <= values["baseline.d_seg"] + SEG_TOLERANCE
        and values["measured.d_pose"]
        <= values["baseline.d_pose"] + POSE_TOLERANCE
    )
    improves_rate = measured_bytes < current_archive_bytes
    return {
        "safe_vs_sealed_baseline": safe,
        "improves_vs_current_combined_bytes": improves_rate,
        "accepted": bool(safe and improves_rate),
    }


def measure_blob(
    lbc: Any,
    *,
    full_blob: bytes,
    label: str,
    run_fingerprint: str,
    out_dir: Path,
    gt: Any,
    segnet: Any,
    posenet: Any,
    oracle_check: bool = False,
    preserve_packet_as: str | None = None,
    force_receiver_replay: bool = False,
) -> dict[str, Any]:
    """Measure exact full ZIP bytes and exact pair-0 receiver components."""

    start = time.perf_counter()
    capped_blob, cap_proof = exact_pair_cap_blob(lbc, full_blob)
    capped_sha = sha256_bytes(capped_blob)
    cache_key = hashlib.sha256(f"{run_fingerprint}:{capped_sha}".encode()).hexdigest()
    cache_path = out_dir / "receiver_cache" / f"{cache_key}.json"
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with success_only_candidate_scratch(scratch) as temp_root:
        full_packet = temp_root / "full_packet"
        archive_path, archive_bytes = lbc.assemble_packet(full_blob, full_packet)
        archive_sha = sha256_file(archive_path)
        cache_preexisted = cache_path.exists()
        cap_packet = temp_root / "cap_packet"
        lbc.assemble_packet(capped_blob, cap_packet)
        inflate = lbc.run_inflate(cap_packet, EVAL_PAIRS, None)
        raw_path = Path(inflate["raw_path"])
        metrics = scorer_measurement(
            lbc, raw_path=raw_path, gt=gt, segnet=segnet, posenet=posenet
        )
        oracle_result = (
            oracle_receiver_check(lbc, capped_blob=capped_blob, raw_path=raw_path)
            if oracle_check
            else None
        )
        receiver_cache = (
            "forced_receiver_replay_cache_never_authoritative"
            if force_receiver_replay
            else "receiver_replayed_cache_never_authoritative"
        )
        cache_payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "capped_blob_sha256": capped_sha,
            "cache_preexisted": cache_preexisted,
            "authority": "none; every decision reruns receiver and scorer",
            "metrics": metrics,
            "oracle_receiver_check": oracle_result,
        }
        atomic_write_json(cache_path, cache_payload)

        if preserve_packet_as is not None:
            destination = out_dir / "artifacts" / preserve_packet_as
            custody_path = out_dir / "artifact_custody" / f"{preserve_packet_as}.json"
            if destination.exists():
                if not custody_path.is_file():
                    raise RuntimeError(
                        f"preserved packet {destination} lacks tree-custody receipt"
                    )
                recorded = load_json(custody_path)
                current = tree_custody(destination)
                if recorded.get("tree_sha256") != current["tree_sha256"]:
                    raise RuntimeError(f"preserved packet tree changed: {destination}")
                if current["files"] != recorded.get("files"):
                    raise RuntimeError(f"preserved packet file custody changed: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(full_packet, destination)
                atomic_write_json(custody_path, tree_custody(destination))

    d_seg = float(metrics["d_seg"])
    d_pose = float(metrics["d_pose"])
    return {
        "label": label,
        "status": "complete",
        "run_fingerprint": run_fingerprint,
        "full_blob_sha256": sha256_bytes(full_blob),
        "full_blob_bytes": len(full_blob),
        "archive_zip_sha256": archive_sha,
        "archive_zip_bytes": int(archive_bytes),
        "capped_receiver_blob_sha256": capped_sha,
        "capped_receiver_proof": cap_proof,
        "receiver_cache": receiver_cache,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "implied_score_advisory": compute_contest_score(d_seg, d_pose, archive_bytes),
        "raw_sha256_local_host": metrics["raw_sha256_local_host"],
        "oracle_receiver_check": oracle_result,
        "elapsed_seconds": time.perf_counter() - start,
        "axis": local_cpu_axis()["tag"],
    }


def verify_custody() -> dict[str, Any]:
    paths = [FIXTURE_PATH, SOURCE_VIDEO, SEGNET_PATH, POSENET_PATH]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required sealed input(s) missing: {missing}")
    actual = {
        "fixture_sha256": sha256_file(FIXTURE_PATH),
        "fixture_bytes": FIXTURE_PATH.stat().st_size,
        "source_video_sha256": sha256_file(SOURCE_VIDEO),
        "source_video_bytes": SOURCE_VIDEO.stat().st_size,
        "segnet_sha256": sha256_file(SEGNET_PATH),
        "posenet_sha256": sha256_file(POSENET_PATH),
    }
    if actual["fixture_sha256"] != FIXTURE_SHA256:
        raise RuntimeError("fixture fingerprint changed; refusing stale coefficient decisions")
    if (
        actual["source_video_sha256"] != SOURCE_VIDEO_SHA256
        or actual["source_video_bytes"] != SOURCE_VIDEO_BYTES
    ):
        raise RuntimeError("source-video fingerprint/bytes changed; refusing scorer decision")
    upstream_contract = build_upstream_eval_contract(repo_root=REPO)
    if not upstream_contract.get("contract_valid"):
        raise RuntimeError(
            f"canonical upstream scorer/evaluator custody failed: {upstream_contract['blockers']}"
        )
    model_records = {
        row["relative_path"]: row for row in upstream_contract["model_custody"]
    }
    if actual["segnet_sha256"] != model_records["models/segnet.safetensors"]["expected_sha256"]:
        raise RuntimeError("SegNet SHA-256 differs from the canonical frozen scorer")
    if actual["posenet_sha256"] != model_records["models/posenet.safetensors"]["expected_sha256"]:
        raise RuntimeError("PoseNet SHA-256 differs from the canonical frozen scorer")
    actual["upstream_eval_contract"] = upstream_contract
    return actual


def prepare_baseline_blob(lbc: Any) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    params, cfg = lbc._load_levelset_ckpt(FIXTURE_PATH.parent, FIXTURE_PATH.name)
    so = lbc.detect_self_orient(
        cfg,
        {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4},
    )
    blob, breakdown = lbc.build_levelset_blob(params, cfg, so, None)
    actual_blob_sha = sha256_bytes(blob)
    if actual_blob_sha != FIXTURE_BASELINE_BLOB_SHA256:
        raise RuntimeError(
            f"baseline LVLS1 bytes changed: {actual_blob_sha} != {FIXTURE_BASELINE_BLOB_SHA256}"
        )
    return blob, breakdown, {"cfg": cfg, "self_orient": so}


def measurement_from_row(row: dict[str, Any]) -> PrefixMeasurement:
    return PrefixMeasurement(
        section=str(row["section"]),
        family=str(row["family"]),  # type: ignore[arg-type]
        bits_removed=int(row["bits_removed"]),
        archive_bytes=int(row["archive_zip_bytes"]),
        d_seg=float(row["d_seg"]),
        d_pose=float(row["d_pose"]),
    )


def baseline_measurement(row: dict[str, Any], *, section: str, family: str) -> PrefixMeasurement:
    return PrefixMeasurement(
        section=section,
        family=family,  # type: ignore[arg-type]
        bits_removed=0,
        archive_bytes=int(row["archive_zip_bytes"]),
        d_seg=float(row["d_seg"]),
        d_pose=float(row["d_pose"]),
    )


def section_summary(
    *, section: CoefficientSection, rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    byte_candidates: list[PrefixMeasurement] = []
    for family in PREFIX_FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        measurements = [measurement_from_row(row) for row in family_rows]
        base = baseline_measurement(baseline, section=section.name, family=family)
        last_safe = select_last_safe_plane(
            measurements,
            base,
            seg_tolerance=SEG_TOLERANCE,
            pose_tolerance=POSE_TOLERANCE,
        )
        best_bytes = select_best_byte_safe(
            measurements,
            base,
            seg_tolerance=SEG_TOLERANCE,
            pose_tolerance=POSE_TOLERANCE,
        )
        if best_bytes is not None:
            byte_candidates.append(best_bytes)
        by_family[family] = {
            "histogram_fit": family_rows[0]["histogram_fit"] if family_rows else None,
            "last_safe_plane": None
            if last_safe is None
            else {
                **asdict(last_safe),
                "archive_bytes_saved": base.archive_bytes - last_safe.archive_bytes,
                "delta_d_seg": last_safe.d_seg - base.d_seg,
                "delta_d_pose": last_safe.d_pose - base.d_pose,
            },
            "best_byte_safe": None
            if best_bytes is None
            else {
                **asdict(best_bytes),
                "archive_bytes_saved": base.archive_bytes - best_bytes.archive_bytes,
                "delta_d_seg": best_bytes.d_seg - base.d_seg,
                "delta_d_pose": best_bytes.d_pose - base.d_pose,
            },
        }
    selected = (
        None
        if not byte_candidates
        else min(byte_candidates, key=lambda row: (row.archive_bytes, -row.bits_removed, row.family))
    )
    return {
        "section": section.name,
        "stream": section.stream,
        "shape": list(section.shape),
        "coefficient_count": section.count,
        "families": by_family,
        "individual_best_byte_safe": None
        if selected is None
        else {
            **asdict(selected),
            "archive_bytes_saved": int(baseline["archive_zip_bytes"]) - selected.archive_bytes,
            "raw_precision_bits_removed": section.count * selected.bits_removed,
        },
    }


def apply_prefix_choice(
    *,
    parts: BlobParts,
    section: CoefficientSection,
    family: str,
    bits_removed: int,
    base_raw: bytes,
    code_raw: bytes,
) -> tuple[bytes, bytes]:
    original = read_section(parts.base_raw, parts.code_raw, section)
    chain = generate_prefix_chain(original, family=family)  # type: ignore[arg-type]
    return replace_section(base_raw, code_raw, section, chain[bits_removed])


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def load_content_addressed_artifact(path: Path, *, expected_schema: str) -> dict[str, Any]:
    """Load a self-hashed JSON artifact and reject semantic identity drift."""

    payload = load_json(path)
    if payload.get("schema") != expected_schema:
        raise RuntimeError(f"artifact schema mismatch: {path}")
    claimed = payload.get("content_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "content_sha256"}
    actual = sha256_bytes(canonical_json_bytes(unsigned))
    if claimed != actual:
        raise RuntimeError(
            f"artifact content hash mismatch: {path} claims {claimed!r}, measured {actual}"
        )
    return payload


def _finite_nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{field} must be a numeric scalar")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise RuntimeError(f"{field} must be finite and non-negative")
    return numeric


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def require_derived_advisory_score(
    row: dict[str, Any], *, field: str
) -> float:
    expected = compute_contest_score(
        float(row["d_seg"]),
        float(row["d_pose"]),
        int(row["archive_zip_bytes"]),
    )
    claimed = row.get("implied_score_advisory")
    if (
        isinstance(claimed, bool)
        or not isinstance(claimed, (int, float))
        or not np.isfinite(float(claimed))
        or float(claimed) != expected
    ):
        raise RuntimeError(f"{field} advisory score does not re-derive from canonical formula")
    return expected


def load_response_curves_for_integration(
    path: Path, *, run_fingerprint: str, expected_sections: dict[str, int]
) -> dict[str, Any]:
    payload = load_content_addressed_artifact(
        path, expected_schema="jrd_section_precision_response_curves.v1"
    )
    rows = payload.get("rows")
    baseline = payload.get("baseline")
    pareto = payload.get("pareto_constraint")
    if (
        payload.get("status") != "complete"
        or payload.get("run_fingerprint") != run_fingerprint
        or not isinstance(rows, list)
        or not rows
        or not isinstance(baseline, dict)
        or not {"archive_zip_bytes", "d_seg", "d_pose"} <= baseline.keys()
        or not isinstance(pareto, dict)
        or pareto.get("score_compensation_allowed") is not False
    ):
        raise RuntimeError("response-curve artifact is incomplete or outside its exact Pareto law")
    baseline_bytes = _finite_nonnegative_number(
        baseline["archive_zip_bytes"], field="baseline.archive_zip_bytes"
    )
    baseline_seg = _finite_nonnegative_number(baseline["d_seg"], field="baseline.d_seg")
    baseline_pose = _finite_nonnegative_number(
        baseline["d_pose"], field="baseline.d_pose"
    )
    if not float(baseline_bytes).is_integer():
        raise RuntimeError("baseline.archive_zip_bytes must be an integer byte count")
    if (
        _finite_nonnegative_number(pareto.get("d_seg_max"), field="pareto.d_seg_max")
        != baseline_seg + SEG_TOLERANCE
        or _finite_nonnegative_number(
            pareto.get("d_pose_max"), field="pareto.d_pose_max"
        )
        != baseline_pose + POSE_TOLERANCE
    ):
        raise RuntimeError("response Pareto limits differ from the sealed baseline plus tolerance")
    if not expected_sections or any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for name, count in expected_sections.items()
    ):
        raise RuntimeError("expected response sections/counts must be valid and non-empty")
    observed_keys: list[tuple[str, str, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"response row {index} must be an object")
        if (
            not isinstance(row.get("section"), str)
            or not row["section"]
            or row.get("family") not in PREFIX_FAMILIES
            or isinstance(row.get("bits_removed"), bool)
            or not isinstance(row.get("bits_removed"), int)
            or not 1 <= row["bits_removed"] <= MAX_INT8_PREFIX_PLANES
            or isinstance(row.get("section_coefficient_count"), bool)
            or not isinstance(row.get("section_coefficient_count"), int)
            or row["section_coefficient_count"] <= 0
            or row.get("section_coefficient_count")
            != expected_sections.get(row.get("section"))
            or row.get("run_fingerprint") != run_fingerprint
        ):
            raise RuntimeError(f"response row {index} has invalid prefix identity")
        row_bytes = _finite_nonnegative_number(
            row.get("archive_zip_bytes"), field=f"rows[{index}].archive_zip_bytes"
        )
        if not float(row_bytes).is_integer():
            raise RuntimeError(f"response row {index} archive bytes must be an integer")
        _finite_nonnegative_number(row.get("d_seg"), field=f"rows[{index}].d_seg")
        _finite_nonnegative_number(row.get("d_pose"), field=f"rows[{index}].d_pose")
        observed_keys.append((row["section"], row["family"], row["bits_removed"]))
    expected_keys = {
        (section, family, bits_removed)
        for section in expected_sections
        for family in PREFIX_FAMILIES
        for bits_removed in range(1, MAX_INT8_PREFIX_PLANES + 1)
    }
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
        raise RuntimeError(
            "response curves do not cover every sealed section, family, and prefix plane exactly once"
        )
    return payload


def sealed_response_sections() -> dict[str, int]:
    """Re-derive exact response-section identities and counts from the sealed fixture."""

    lbc = load_byte_close_module()
    baseline_blob, _breakdown, _context = prepare_baseline_blob(lbc)
    parts = parse_blob(lbc, baseline_blob)
    return {
        section.name: section.count
        for section in coefficient_sections(
            parts.manifest,
            base_raw_len=len(parts.base_raw),
            code_raw_len=len(parts.code_raw),
            eval_pairs=EVAL_PAIRS,
        )
    }


def sealed_response_section_names() -> tuple[str, ...]:
    return tuple(sealed_response_sections())


def load_allocator_for_integration(
    path: Path, *, run_fingerprint: str, response_curves: dict[str, Any]
) -> dict[str, Any]:
    payload = load_content_addressed_artifact(
        path, expected_schema="jrd_prefix_allocator_planning_input.v1"
    )
    if (
        payload.get("run_fingerprint") != run_fingerprint
        or payload.get("research_only") is not True
        or payload.get("promotion_eligible") is not False
        or not isinstance(payload.get("proposed"), list)
        or not isinstance(payload.get("accepted"), list)
        or not isinstance(payload.get("rejected"), list)
    ):
        raise RuntimeError("allocator artifact is incomplete or has invalid authority scope")
    baseline = response_curves["baseline"]
    response_by_key = {
        (row["section"], row["family"], row["bits_removed"]): row
        for row in response_curves["rows"]
    }
    proposed = payload["proposed"]
    expected_proposed: list[dict[str, Any]] = []
    section_names = sorted({row["section"] for row in response_curves["rows"]})
    for section in section_names:
        byte_candidates: list[PrefixMeasurement] = []
        coefficient_counts: set[int] = set()
        for family in PREFIX_FAMILIES:
            family_rows = [
                row
                for row in response_curves["rows"]
                if row["section"] == section and row["family"] == family
            ]
            coefficient_counts.update(
                int(row["section_coefficient_count"]) for row in family_rows
            )
            best = select_best_byte_safe(
                [measurement_from_row(row) for row in family_rows],
                baseline_measurement(baseline, section=section, family=family),
                seg_tolerance=SEG_TOLERANCE,
                pose_tolerance=POSE_TOLERANCE,
            )
            if best is not None:
                byte_candidates.append(best)
        if len(coefficient_counts) != 1 or next(iter(coefficient_counts)) <= 0:
            raise RuntimeError(f"response section {section!r} coefficient count changed")
        if byte_candidates:
            selected = min(
                byte_candidates,
                key=lambda row: (row.archive_bytes, -row.bits_removed, row.family),
            )
            expected_proposed.append(
                {
                    **asdict(selected),
                    "archive_bytes_saved": baseline["archive_zip_bytes"]
                    - selected.archive_bytes,
                    "raw_precision_bits_removed": next(iter(coefficient_counts))
                    * selected.bits_removed,
                }
            )
    expected_proposed.sort(
        key=lambda row: (
            -int(row["archive_bytes_saved"]),
            str(row["section"]),
            str(row["family"]),
        )
    )
    if proposed != expected_proposed:
        raise RuntimeError("allocator proposal list does not re-derive from response curves")
    for index, choice in enumerate(proposed):
        if not isinstance(choice, dict):
            raise RuntimeError(f"allocator proposed[{index}] must be an object")
        key = (choice.get("section"), choice.get("family"), choice.get("bits_removed"))
        response = response_by_key.get(key)
        if response is None:
            raise RuntimeError(f"allocator proposed[{index}] is not a measured response row")
        if (
            choice.get("archive_bytes") != response["archive_zip_bytes"]
            or choice.get("d_seg") != response["d_seg"]
            or choice.get("d_pose") != response["d_pose"]
            or choice.get("archive_bytes_saved")
            != baseline["archive_zip_bytes"] - response["archive_zip_bytes"]
        ):
            raise RuntimeError(f"allocator proposed[{index}] changed measured response values")
    steps = [*payload["accepted"], *payload["rejected"]]
    if len(steps) != len(proposed) or any(not isinstance(step, dict) for step in steps):
        raise RuntimeError("allocator steps do not form a one-to-one proposed sequence")
    by_choice: dict[str, dict[str, Any]] = {}
    for step in steps:
        key = json.dumps(step.get("choice"), sort_keys=True, separators=(",", ":"))
        if key in by_choice:
            raise RuntimeError("allocator has duplicate steps for one proposed choice")
        by_choice[key] = step
    current_bytes = int(baseline["archive_zip_bytes"])
    derived_accepted: list[dict[str, Any]] = []
    derived_rejected: list[dict[str, Any]] = []
    for index, choice in enumerate(proposed):
        key = json.dumps(choice, sort_keys=True, separators=(",", ":"))
        step = by_choice.get(key)
        if step is None:
            raise RuntimeError(f"allocator proposed[{index}] has no exact combined step")
        expected_label = f"combined_step={index};section={choice['section']}"
        if (
            step.get("label") != expected_label
            or step.get("current_combined_bytes_before") != current_bytes
        ):
            raise RuntimeError("allocator combined sequence does not start from its prior bytes")
        gate = combined_step_gate(
            step, baseline=baseline, current_archive_bytes=current_bytes
        )
        if any(step.get(gate_key) is not value for gate_key, value in gate.items()):
            raise RuntimeError("allocator asserted a gate it did not derive")
        if gate["accepted"]:
            derived_accepted.append(step)
            current_bytes = step["archive_zip_bytes"]
        else:
            derived_rejected.append(step)
    if payload["accepted"] != derived_accepted or payload["rejected"] != derived_rejected:
        raise RuntimeError("allocator decision lists do not preserve derived sequence order")
    return payload


def load_measurement_receipt_for_integration(
    path: Path, *, run_fingerprint: str
) -> dict[str, Any]:
    payload = load_content_addressed_artifact(
        path, expected_schema="jrd_coefficient_prefix_measurement.v1"
    )
    baseline = payload.get("baseline")
    selected = payload.get("selected")
    delta = payload.get("delta")
    inventory = payload.get("payload_inventory")
    controls = payload.get("controls")
    boundaries = payload.get("boundaries")
    if (
        payload.get("status") != "complete"
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("task_id") != "v9_jrd_coeff_prefix_probe_20260712"
        or payload.get("task_verdict") != "NEEDS-MORE"
        or payload.get("research_only") is not True
        or payload.get("score_claim") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("pointer_moved") is not False
        or not isinstance(baseline, dict)
        or not isinstance(selected, dict)
        or not isinstance(delta, dict)
        or not isinstance(inventory, dict)
        or not isinstance(controls, dict)
        or not isinstance(boundaries, dict)
        or not isinstance(payload.get("accepted_combined_steps"), list)
        or not isinstance(payload.get("rejected_combined_steps"), list)
    ):
        raise RuntimeError("measurement receipt is incomplete or has invalid authority scope")
    measurements: dict[str, tuple[int, float, float, float]] = {}
    for name, row in (("baseline", baseline), ("selected", selected)):
        if row.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"measurement receipt {name} fingerprint changed")
        archive_bytes = _finite_nonnegative_number(
            row.get("archive_zip_bytes"), field=f"receipt.{name}.archive_zip_bytes"
        )
        if not archive_bytes.is_integer():
            raise RuntimeError(f"receipt {name} archive bytes must be an integer")
        d_seg = _finite_nonnegative_number(
            row.get("d_seg"), field=f"receipt.{name}.d_seg"
        )
        d_pose = _finite_nonnegative_number(
            row.get("d_pose"), field=f"receipt.{name}.d_pose"
        )
        _require_sha256(
            row.get("archive_zip_sha256"), field=f"receipt.{name}.archive_zip_sha256"
        )
        _require_sha256(
            row.get("full_blob_sha256"), field=f"receipt.{name}.full_blob_sha256"
        )
        _require_sha256(
            row.get("raw_sha256_local_host"),
            field=f"receipt.{name}.raw_sha256_local_host",
        )
        implied_score = require_derived_advisory_score(row, field=f"receipt.{name}")
        measurements[name] = (int(archive_bytes), d_seg, d_pose, implied_score)
    baseline_bytes, baseline_seg, baseline_pose, baseline_score = measurements["baseline"]
    selected_bytes, selected_seg, selected_pose, selected_score = measurements["selected"]
    bytes_saved = baseline_bytes - selected_bytes
    if (
        bytes_saved < 0
        or delta.get("archive_bytes_saved") != bytes_saved
        or delta.get("d_seg") != selected_seg - baseline_seg
        or delta.get("d_pose") != selected_pose - baseline_pose
        or delta.get("implied_score_advisory") != selected_score - baseline_score
    ):
        raise RuntimeError("measurement receipt deltas do not re-derive from measured rows")
    if (
        selected_seg > baseline_seg + SEG_TOLERANCE
        or selected_pose > baseline_pose + POSE_TOLERANCE
    ):
        raise RuntimeError("measurement receipt selected row violates its component guard")
    expected_fixture_verdict = "GO" if bytes_saved > 0 else "NO-GO"
    if payload.get("fixture_verdict") != expected_fixture_verdict:
        raise RuntimeError("measurement receipt fixture verdict does not re-derive from bytes")
    if (
        inventory.get("eligible_count") != 0
        or not isinstance(inventory.get("unclassified_lvls1_count"), int)
        or inventory["unclassified_lvls1_count"] < 0
        or not isinstance(inventory.get("path"), str)
        or not inventory["path"]
    ):
        raise RuntimeError("measurement receipt lost its unresolved V9/v8 inventory blocker")
    _require_sha256(inventory.get("sha256"), field="receipt.payload_inventory.sha256")
    expected_controls = {
        "positive_baseline": "controls/baseline.json",
        "positive_repeat": "controls/baseline_repeat.json",
        "negative_all_zero": "controls/all_zero_negative.json",
    }
    if controls != expected_controls:
        raise RuntimeError("measurement receipt control paths are incomplete")
    if (
        boundaries.get("eligible_v9_v8_payload") is not False
        or boundaries.get("eval_pairs") != EVAL_PAIRS
        or boundaries.get("upstream_evaluate_py_run") is not False
        or boundaries.get("contest_cpu_linux_x86_64") is not False
        or boundaries.get("contest_cuda") is not False
    ):
        raise RuntimeError("measurement receipt authority boundary changed")
    if inventory["path"] != "v9_v8_payload_inventory.json":
        raise RuntimeError("measurement receipt inventory path changed")
    inventory_path = path.parent / inventory["path"]
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise RuntimeError("measurement receipt inventory artifact is missing or symlinked")
    inventory_artifact = load_json(inventory_path)
    claimed_inventory_sha = inventory_artifact.get("inventory_sha256")
    unsigned_inventory = {
        key: value
        for key, value in inventory_artifact.items()
        if key != "inventory_sha256"
    }
    measured_inventory_sha = sha256_bytes(canonical_json_bytes(unsigned_inventory))
    records = inventory_artifact.get("records")
    if (
        claimed_inventory_sha != measured_inventory_sha
        or inventory["sha256"] != measured_inventory_sha
        or not isinstance(records, list)
        or inventory_artifact.get("eligible_count")
        != sum(bool(row.get("eligible_nonlive_byte_closed")) for row in records)
        or inventory_artifact.get("unclassified_lvls1_count")
        != sum(
            bool(row.get("lvls1_magic") and row.get("typed_vehicle_provenance") is None)
            for row in records
        )
        or inventory_artifact.get("eligible_count") != inventory["eligible_count"]
        or inventory_artifact.get("unclassified_lvls1_count")
        != inventory["unclassified_lvls1_count"]
    ):
        raise RuntimeError("measurement receipt inventory custody/counts do not re-derive")
    control_rows: dict[str, dict[str, Any]] = {}
    for name, relative in expected_controls.items():
        control_path = path.parent / relative
        if control_path.is_symlink() or not control_path.is_file():
            raise RuntimeError(f"measurement receipt control {name} is missing or symlinked")
        row = load_json(control_path)
        if row.get("status") != "complete" or row.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(f"measurement receipt control {name} identity changed")
        control_rows[name] = row
    if control_rows["positive_baseline"] != baseline:
        raise RuntimeError("measurement receipt baseline differs from its control artifact")
    repeat_fields = (
        "archive_zip_sha256",
        "archive_zip_bytes",
        "raw_sha256_local_host",
        "d_seg",
        "d_pose",
    )
    if any(
        control_rows["positive_repeat"].get(field) != baseline.get(field)
        for field in repeat_fields
    ):
        raise RuntimeError("measurement receipt positive repeat differs from baseline")
    validate_negative_control(baseline, control_rows["negative_all_zero"])
    return payload


def assert_existing_probe_outcome_matches(
    existing: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """Refuse stale idempotent reuse when any load-bearing candidate field changed."""

    for key, value in candidate.items():
        if existing.get(key) != value:
            raise RuntimeError(f"existing canonical probe outcome changed field {key!r}")


def assert_cross_artifact_baseline_identity(
    receipt: dict[str, Any], response_curves: dict[str, Any]
) -> None:
    fields = ("archive_zip_sha256", "archive_zip_bytes", "d_seg", "d_pose")
    if any(
        receipt["baseline"].get(field) != response_curves["baseline"].get(field)
        for field in fields
    ):
        raise RuntimeError("receipt and response curves name different sealed baselines")


def canonical_probe_outcome_fields(*, exact_bytes_saved: int) -> dict[str, Any]:
    return {
        "probe_id": "v9_jrd_coeff_prefix_probe_20260712",
        "substrate": "v9_v8_coefficient_payload",
        "probe_kind": "uniform_vs_laplace_dead_zone_exact_receiver_prefix",
        "verdict": "DEFER",
        "metric_name": "fixture_archive_bytes_saved_pair0_advisory",
        "metric_value": exact_bytes_saved,
        "threshold": 1.0,
        "threshold_token": "eligible_nonlive_v9_v8_payload_required",
        "next_action": (
            "rerun the exact prefix oracle on a sealed non-live V9/v8 LVLS1 payload and replay "
            "early boundary and late saved regimes"
        ),
        "reactivation_criteria": [
            "payload inventory contains one sealed non-live V9/v8 LVLS1 archive",
            "early boundary and late saved-regime payloads are content-addressed",
        ],
        "blocker_status": "blocking",
        "agent": "codex",
        "notes": "pair-0 fixture is planning-only and cannot adjudicate V9/v8",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }


def assert_probe_outcome_control_law(
    candidate: dict[str, Any], *, exact_bytes_saved: int
) -> None:
    expected = canonical_probe_outcome_fields(exact_bytes_saved=exact_bytes_saved)
    allowed_keys = set(expected) | {"recipe_path", "evidence_path"}
    if set(candidate) != allowed_keys or any(
        candidate.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("probe outcome metadata differs from the measured control law")


def integrate_system_intelligence(
    out_dir: Path,
    *,
    run_fingerprint: str,
    posterior_candidate: dict[str, Any],
    probe_outcome_candidate: dict[str, Any],
    task_hook_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Execute the research-only canonical consumers without granting promotion."""

    from cathedral_autopilot_autonomous_loop import (
        load_candidates_from_probe_disambiguator_output,
    )

    from tac.canonical_task_status.loader import latest_status_by_task_id
    from tac.canonical_task_status.writer import update_status
    from tac.component_sensitivity_artifact import (
        custody_metadata,
        materialize_component_sensitivity_manifest,
        write_component_sensitivity_manifest,
    )
    from tac.continual_learning import ContestResult, posterior_update_locked
    from tac.probe_outcomes_ledger import query_by_probe_id, register_probe_outcome

    receipt_path = out_dir / "measurement_receipt.json"
    receipt = load_measurement_receipt_for_integration(
        receipt_path, run_fingerprint=run_fingerprint
    )
    receipt_rel = _repo_relative(receipt_path)
    response_path = out_dir / "section_precision_response_curves.json"
    response_curves = load_response_curves_for_integration(
        response_path,
        run_fingerprint=run_fingerprint,
        expected_sections=sealed_response_sections(),
    )
    assert_cross_artifact_baseline_identity(receipt, response_curves)
    response_custody = custody_metadata(response_path)
    response_custody["path"] = response_path.name
    sensitivity_manifest = materialize_component_sensitivity_manifest(
        {
            "schema_version": 1,
            "format": "component_sensitivity_v1",
            "device": local_cpu_axis()["tag"],
            "promotion_eligible": False,
            "evidence_grade": "research_only_pair0",
            "diagnostic_response_curve": response_custody,
            "promotion_blockers": [
                {
                    "code": "pair0_macos_advisory_not_v9_v8",
                    "mathematical_explanation": (
                        "A one-pair macOS CPU component-response curve cannot establish the "
                        "600-pair contest CPU/CUDA distortion allocation or transfer to an "
                        "unresolved V9/v8 coefficient payload."
                    ),
                }
            ],
        },
        root=out_dir,
        promotion=False,
    )
    sensitivity_path = out_dir / "component_sensitivity_manifest.json"
    write_component_sensitivity_manifest(sensitivity_path, sensitivity_manifest)

    allocator = load_allocator_for_integration(
        out_dir / "allocator_planning_input.json",
        run_fingerprint=run_fingerprint,
        response_curves=response_curves,
    )
    accepted = allocator.get("accepted")
    rejected = allocator.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise RuntimeError("allocator execution artifact lost its accepted/rejected lists")
    if any(
        step.get("accepted") is not True
        or step.get("safe_vs_sealed_baseline") is not True
        or step.get("improves_vs_current_combined_bytes") is not True
        for step in accepted
    ):
        raise RuntimeError("allocator accepted a step outside its exact component/rate gate")
    if (
        receipt["accepted_combined_steps"] != accepted
        or receipt["rejected_combined_steps"] != rejected
    ):
        raise RuntimeError("receipt and allocator decisions differ")
    final_allocator_row = accepted[-1] if accepted else response_curves["baseline"]
    selected_identity_fields = (
        "archive_zip_sha256",
        "archive_zip_bytes",
        "d_seg",
        "d_pose",
    )
    if any(
        receipt["selected"].get(field) != final_allocator_row.get(field)
        for field in selected_identity_fields
    ):
        raise RuntimeError("receipt selected replay differs from the allocator terminus")
    contest_result = posterior_candidate.get("contest_result")
    if not isinstance(contest_result, dict) or any(
        contest_result.get(candidate_field) != receipt["selected"].get(receipt_field)
        for candidate_field, receipt_field in (
            ("archive_sha256", "archive_zip_sha256"),
            ("archive_bytes", "archive_zip_bytes"),
            ("cpu_seg", "d_seg"),
            ("cpu_pose", "d_pose"),
        )
    ):
        raise RuntimeError("continual-learning candidate differs from the selected replay")
    assert_probe_outcome_control_law(
        probe_outcome_candidate,
        exact_bytes_saved=receipt["delta"]["archive_bytes_saved"],
    )
    expected_task_fields = {
        "task_id": "v9_jrd_coeff_prefix_probe_20260712",
        "status": "blocked",
        "test_status": "green",
        "blocker": "eligible_nonlive_v9_v8_payload_missing_or_unresolved",
        "actual_delta_s": None,
    }
    if any(task_hook_candidate.get(key) != value for key, value in expected_task_fields.items()):
        raise RuntimeError("probe outcome or task hook differs from the measured receipt")
    expected_score = receipt["selected"].get("implied_score_advisory")
    if (
        not isinstance(expected_score, (int, float))
        or not np.isfinite(float(expected_score))
        or contest_result.get("score_value") != expected_score
        or contest_result.get("axis") != "cpu"
        or contest_result.get("hardware_substrate")
        != local_cpu_axis()["hardware_substrate"]
        or contest_result.get("evidence_tag") != local_cpu_axis()["evidence_tag"]
        or contest_result.get("architecture_class")
        != "jrd_coefficient_prefix_v75_fixture_pair0"
    ):
        raise RuntimeError("continual-learning advisory score or authority tags changed")

    autopilot_candidates = load_candidates_from_probe_disambiguator_output(
        out_dir / "probe_disambiguator_output.json"
    )
    if not autopilot_candidates:
        raise RuntimeError("canonical autopilot consumer returned no planning candidate")

    posterior_dir = out_dir / "system_intelligence" / "continual_learning"
    posterior_update_path = posterior_dir / "update.json"
    contest_result_sha256 = sha256_bytes(
        canonical_json_bytes(posterior_candidate["contest_result"])
    )
    if posterior_update_path.exists():
        posterior_update_record = load_json(posterior_update_path)
        if (
            posterior_update_record.get("run_fingerprint") != run_fingerprint
            or posterior_update_record.get("contest_result_sha256")
            != contest_result_sha256
            or posterior_update_record.get("custody_verdict", {}).get("accepted")
            is not False
            or posterior_update_record.get("posterior_update", {}).get("accepted")
            is not False
        ):
            raise RuntimeError("run-local continual-learning update fingerprint changed")
    else:
        result = ContestResult(**posterior_candidate["contest_result"])
        custody_verdict = result.validate_custody_verdict()
        update = posterior_update_locked(
            result,
            posterior_path=posterior_dir / "posterior.json",
            lock_path=posterior_dir / ".lock",
            forbid_macos_promotion=True,
        )
        posterior_update_record = {
            "schema": "jrd_continual_learning_integration.v1",
            "run_fingerprint": run_fingerprint,
            "contest_result_sha256": contest_result_sha256,
            "custody_verdict": asdict(custody_verdict),
            "posterior_update": asdict(update),
        }
        if update.accepted:
            raise RuntimeError("research-only local result was admitted to the posterior")
        atomic_write_json(posterior_update_path, posterior_update_record)

    probe_outcome_candidate = {
        **probe_outcome_candidate,
        "recipe_path": receipt_rel,
        "evidence_path": receipt_rel,
    }
    prior_probe_rows = query_by_probe_id(probe_outcome_candidate["probe_id"])
    matching_probe_rows = [
        row
        for row in prior_probe_rows
        if row.get("evidence_path") == receipt_rel
        and row.get("verdict") == probe_outcome_candidate["verdict"]
    ]
    if matching_probe_rows:
        probe_outcome_row = matching_probe_rows[-1]
        assert_existing_probe_outcome_matches(probe_outcome_row, probe_outcome_candidate)
    else:
        probe_outcome_row = register_probe_outcome(**probe_outcome_candidate)

    current_task = latest_status_by_task_id(task_hook_candidate["task_id"], REPO)
    if current_task is None:
        raise RuntimeError("canonical task hook names an unregistered task")
    if current_task.status == "blocked":
        if (
            task_hook_candidate["blocker"] not in current_task.blockers
            or receipt_rel not in current_task.event_notes
        ):
            raise RuntimeError(
                "existing canonical task blocker is not this exact JRD receipt"
            )
        task_row = current_task
    elif current_task.status in {"pending", "in_progress"}:
        task_row = update_status(
            task_hook_candidate["task_id"],
            "blocked",
            actor="codex",
            session_id=f"jrd_coeff_prefix_{run_fingerprint[:12]}",
            notes=f"research-only exact-R receipt: {receipt_rel}",
            test_status="green",
            blockers=(task_hook_candidate["blocker"],),
            actual_delta_s=None,
            repo_root=REPO,
        )
    else:
        raise RuntimeError(
            f"canonical task status {current_task.status!r} cannot accept this blocker"
        )

    integration = {
        "schema": "jrd_coefficient_prefix_system_integration.v1",
        "run_fingerprint": run_fingerprint,
        "score_claim": False,
        "promotion_eligible": False,
        "sensitivity_map": {
            "canonical_consumer": (
                "tac.component_sensitivity_artifact."
                "materialize_component_sensitivity_manifest"
            ),
            "path": sensitivity_path.name,
            "sha256": sha256_file(sensitivity_path),
            "source_response_sha256": response_custody["sha256"],
        },
        "pareto_constraint": {
            "canonical_consumer": (
                "tac.packet_compiler.jrd_coefficient_prefix.select_best_byte_safe"
            ),
            "componentwise_nonworse": True,
            "score_compensation_allowed": False,
        },
        "bit_allocator": {
            "control_law": allocator["control_law"],
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "content_sha256": allocator["content_sha256"],
        },
        "cathedral_autopilot": {
            "canonical_consumer": (
                "tools.cathedral_autopilot_autonomous_loop."
                "load_candidates_from_probe_disambiguator_output"
            ),
            "candidate_ids": [row.candidate_id for row in autopilot_candidates],
            "all_blocked": all(bool(row.blockers) for row in autopilot_candidates),
        },
        "continual_learning": posterior_update_record,
        "probe_outcome": probe_outcome_row,
        "canonical_task_status": task_row.to_json_obj(),
    }
    atomic_write_json(out_dir / "system_integration_receipt.json", integration)
    return integration


def run_probe(out_dir: Path) -> dict[str, Any]:
    refuse_transient_evidence_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_output_lock(out_dir)
    try:
        return _run_probe_locked(out_dir)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _run_probe_locked(out_dir: Path) -> dict[str, Any]:
    refuse_transient_evidence_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    lbc = load_byte_close_module()
    custody = verify_custody()
    fingerprint_payload = run_fingerprint_payload(
        lbc,
        upstream_contract=custody["upstream_eval_contract"],
    )
    run_fingerprint = sha256_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    run_manifest = {
        **fingerprint_payload,
        "run_fingerprint": run_fingerprint,
        "payload_scope": "v7.5.2 frozen fixture staged by V9 apply-pass dry run",
        "eligible_v9_v8_payload": False,
        "score_claim": False,
        "promotion_eligible": False,
        "review_status": "recovery-written-UNREVIEWED",
        "research_only": True,
        "git_head_at_start": git_head,
        "custody": custody,
        "governed_admission": os.environ.get("TAC_GOVERNED_ADMISSION") == "1",
        "launch_provenance": {
            "cwd": str(REPO),
            "child_argv": list(sys.argv),
            "parent_process": process_command(os.getppid()),
            "relevant_env": {key: os.environ.get(key) for key in RECEIVER_ENV_KEYS},
            "admission_boundary": (
                "TAC_GOVERNED_ADMISSION=1 plus captured safe_run parent command; the durable "
                "launch log records the system-memory governor decision"
            ),
        },
        "resume_contract": {
            "explicit_cli": "--resume-from <run_dir|run_manifest.json>",
            "registry": "tac.witness_control.resume_registry.ResumeRegistry",
            "state": "resume/jrd_probe_latest.npz",
            "preserved_stage_checkpoints": True,
            "atomic_candidate_checkpoints": True,
        },
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("existing run manifest has a different fingerprint; refuse mixed resume")
    else:
        atomic_write_json(manifest_path, run_manifest)
    resume_status = restore_or_initialize_resume(
        out_dir, run_fingerprint=run_fingerprint
    )
    verified_checkpoint_paths = set(
        resume_status["registered_checkpoint_paths"]
    )
    pending_checkpoint_paths = set(resume_status["pending_checkpoint_paths"])
    cleanup_scratch(out_dir, run_fingerprint=run_fingerprint)
    free = shutil.disk_usage(out_dir).free
    storage = {
        "path": str(out_dir),
        "free_bytes": free,
        "required_bytes": 64 * 1024 * 1024,
        "ok": free >= 64 * 1024 * 1024,
        "basis": "one 6.1MB pair raw plus archive/scripts, 10x bounded margin",
    }
    if not storage["ok"]:
        raise RuntimeError(f"storage preflight failed: {storage}")
    payload_inventory = build_payload_inventory()
    atomic_write_json(out_dir / "v9_v8_payload_inventory.json", payload_inventory)
    if payload_inventory["eligible_count"]:
        raise RuntimeError(
            "eligible non-live V9/v8 LVLS1 payload now exists; sealed fixture selection is stale"
        )
    atomic_write_json(
        out_dir / "stage_00_custody_and_storage.json",
        {
            **storage,
            "resume_status": resume_status,
            "payload_inventory_sha256": payload_inventory["inventory_sha256"],
            "eligible_nonlive_v9_v8_payloads": payload_inventory["eligible_count"],
        },
    )
    write_resume_state(
        out_dir,
        run_fingerprint=run_fingerprint,
        stage="custody_storage_inventory_complete",
        preserve_as="stage_00_custody_storage_inventory",
        verified_checkpoint_paths=verified_checkpoint_paths,
    )

    baseline_blob, breakdown, baseline_context = prepare_baseline_blob(lbc)
    parts = parse_blob(lbc, baseline_blob)
    repacked_baseline = repack_blob(
        lbc, parts, base_raw=parts.base_raw, code_raw=parts.code_raw
    )
    if repacked_baseline != baseline_blob:
        raise RuntimeError("parse/repack identity failed for sealed baseline LVLS1 bytes")
    sections = coefficient_sections(
        parts.manifest,
        base_raw_len=len(parts.base_raw),
        code_raw_len=len(parts.code_raw),
        eval_pairs=EVAL_PAIRS,
    )
    if int(parts.manifest["n_pairs"]) != 600:
        raise RuntimeError(
            f"sealed fixture has n_pairs={parts.manifest['n_pairs']}; expected the 600-pair payload"
        )
    atomic_write_json(
        out_dir / "stage_01_payload_layout.json",
        {
            "run_fingerprint": run_fingerprint,
            "baseline_blob_sha256": sha256_bytes(baseline_blob),
            "byte_close_breakdown": breakdown,
            "context": baseline_context,
            "sections": [asdict(section) for section in sections],
            "pair_matched_rate_debt_scope": {
                "eval_pairs": EVAL_PAIRS,
                "code_rows_mutated": 2 * EVAL_PAIRS,
                "unscored_code_rows_preserved_byte_identical": (
                    int(parts.manifest["code_shape"][0]) - 2 * EVAL_PAIRS
                ),
            },
            "parse_repack_identity": True,
        },
    )
    write_resume_state(
        out_dir,
        run_fingerprint=run_fingerprint,
        stage="payload_layout_complete",
        preserve_as="stage_01_payload_layout",
        verified_checkpoint_paths=verified_checkpoint_paths,
    )

    gt, segnet, posenet = lbc.twr.precompute_gt(EVAL_PAIRS)
    if gt.n_pairs != EVAL_PAIRS:
        raise RuntimeError(f"GT precompute returned {gt.n_pairs} pairs, expected {EVAL_PAIRS}")

    controls_dir = out_dir / "controls"
    controls_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = controls_dir / "baseline.json"
    if baseline_path.exists():
        baseline = load_checked_checkpoint(
            baseline_path,
            run_fingerprint=run_fingerprint,
            expected_fields={
                "label": "positive_control_baseline",
                "full_blob_sha256": FIXTURE_BASELINE_BLOB_SHA256,
            },
        )
    else:
        baseline = measure_blob(
            lbc,
            full_blob=baseline_blob,
            label="positive_control_baseline",
            run_fingerprint=run_fingerprint,
            out_dir=out_dir,
            gt=gt,
            segnet=segnet,
            posenet=posenet,
            oracle_check=True,
            preserve_packet_as="baseline_packet",
        )
        atomic_write_json(baseline_path, baseline)
    if (
        baseline["archive_zip_bytes"] != FIXTURE_BASELINE_ARCHIVE_BYTES
        or baseline["archive_zip_sha256"] != FIXTURE_BASELINE_ARCHIVE_SHA256
    ):
        raise RuntimeError("baseline archive bytes/hash differ from sealed prior byte-close receipt")

    repeat_path = controls_dir / "baseline_repeat.json"
    baseline_repeat = measure_blob(
        lbc,
        full_blob=baseline_blob,
        label="positive_control_baseline_repeat",
        run_fingerprint=run_fingerprint,
        out_dir=out_dir,
        gt=gt,
        segnet=segnet,
        posenet=posenet,
        oracle_check=True,
        force_receiver_replay=True,
        preserve_packet_as="baseline_packet",
    )
    atomic_write_json(repeat_path, baseline_repeat)
    replay_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_pid{os.getpid()}"
    atomic_write_json(controls_dir / "replays" / f"{replay_id}.json", baseline_repeat)
    repeat_fields = (
        "archive_zip_sha256",
        "archive_zip_bytes",
        "raw_sha256_local_host",
        "d_seg",
        "d_pose",
    )
    repeat_mismatch = {
        field: [baseline[field], baseline_repeat[field]]
        for field in repeat_fields
        if baseline[field] != baseline_repeat[field]
    }
    if repeat_mismatch:
        raise RuntimeError(f"positive control is not deterministic: {repeat_mismatch}")

    negative_path = controls_dir / "all_zero_negative.json"
    zero_blob = repack_blob(
        lbc,
        parts,
        base_raw=bytes(len(parts.base_raw)),
        code_raw=bytes(len(parts.code_raw)),
    )
    negative = measure_negative_control(
        lbc,
        negative_path=negative_path,
        zero_blob=zero_blob,
        run_fingerprint=run_fingerprint,
        out_dir=out_dir,
        gt=gt,
        segnet=segnet,
        posenet=posenet,
    )
    validate_negative_control(baseline, negative)
    atomic_write_json(
        out_dir / "stage_02_controls_complete.json",
        {
            "run_fingerprint": run_fingerprint,
            "positive_repeat_exact": True,
            "negative_control_separates_raw_and_components": True,
            "within_run_component_noise_floor": {"d_seg": 0.0, "d_pose": 0.0},
            "across_seed_variance": "UNKNOWN (single frozen payload)",
            "across_host_variance": "UNKNOWN; local raw hash is not portable authority",
        },
    )
    write_resume_state(
        out_dir,
        run_fingerprint=run_fingerprint,
        stage="controls_complete",
        preserve_as="stage_02_controls",
        verified_checkpoint_paths=verified_checkpoint_paths,
    )

    all_rows: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections):
        original = read_section(parts.base_raw, parts.code_raw, section)
        histogram_fit = asdict(fit_laplace_histogram(original))
        for family in PREFIX_FAMILIES:
            chain = generate_prefix_chain(original, family=family)
            for bits_removed in range(1, MAX_INT8_PREFIX_PLANES + 1):
                checkpoint = candidate_checkpoint_path(
                    out_dir,
                    section=section.name,
                    family=family,
                    bits_removed=bits_removed,
                )
                checkpoint_rel = checkpoint.relative_to(out_dir).as_posix()
                row = load_registered_checkpoint_or_none(
                    checkpoint,
                    out_dir=out_dir,
                    verified_checkpoint_paths=verified_checkpoint_paths,
                    run_fingerprint=run_fingerprint,
                    expected_fields={
                        "section": section.name,
                        "family": family,
                        "bits_removed": bits_removed,
                    },
                )
                if row is None:
                    base_new, code_new = replace_section(
                        parts.base_raw,
                        parts.code_raw,
                        section,
                        chain[bits_removed],
                    )
                    candidate_blob = repack_blob(
                        lbc, parts, base_raw=base_new, code_raw=code_new
                    )
                    if section.stream == "code" and code_new[section.count :] != parts.code_raw[section.count :]:
                        raise RuntimeError("unscored code suffix changed during pair-matched mutation")
                    measured = measure_blob(
                        lbc,
                        full_blob=candidate_blob,
                        label=(
                            f"section={section.name};family={family};"
                            f"bits_removed={bits_removed}"
                        ),
                        run_fingerprint=run_fingerprint,
                        out_dir=out_dir,
                        gt=gt,
                        segnet=segnet,
                        posenet=posenet,
                    )
                    row = {
                        **measured,
                        "section": section.name,
                        "section_index": section_index,
                        "stream": section.stream,
                        "section_shape": list(section.shape),
                        "section_coefficient_count": section.count,
                        "section_original_sha256": sha256_bytes(original.tobytes()),
                        "section_prefix_sha256": sha256_bytes(
                            chain[bits_removed].tobytes()
                        ),
                        "family": family,
                        "bits_removed": bits_removed,
                        "candidate_sequence_index": len(all_rows),
                        "pair_matched_code_suffix_sha256": (
                            None
                            if section.stream != "code"
                            else sha256_bytes(code_new[section.count :])
                        ),
                        "histogram_fit": histogram_fit,
                        "archive_bytes_saved_vs_baseline": (
                            int(baseline["archive_zip_bytes"])
                            - int(measured["archive_zip_bytes"])
                        ),
                        "delta_d_seg": float(measured["d_seg"])
                        - float(baseline["d_seg"]),
                        "delta_d_pose": float(measured["d_pose"])
                        - float(baseline["d_pose"]),
                    }
                    atomic_write_json(checkpoint, row)
                    verified_checkpoint_paths.add(checkpoint_rel)
                    pending_checkpoint_paths.discard(checkpoint_rel)
                    write_resume_state(
                        out_dir,
                        run_fingerprint=run_fingerprint,
                        stage=(
                            f"candidate_{len(all_rows):03d}_{section_index:02d}_"
                            f"{family}_k{bits_removed}"
                        ),
                        verified_checkpoint_paths=verified_checkpoint_paths,
                    )
                all_rows.append(row)
                print(
                    f"[{len(all_rows):03d}/{len(sections)*len(PREFIX_FAMILIES)*MAX_INT8_PREFIX_PLANES}] "
                    f"{section.name} {family} k={bits_removed} "
                    f"bytes={row['archive_zip_bytes']} dseg={row['d_seg']:.12g} "
                    f"dpose={row['d_pose']:.12g}",
                    flush=True,
                )
        atomic_write_json(
            out_dir / "section_checkpoints" / f"{section_index:02d}.json",
            {
                "run_fingerprint": run_fingerprint,
                "status": "complete",
                "section": section.name,
                "candidate_count": len(PREFIX_FAMILIES) * MAX_INT8_PREFIX_PLANES,
            },
        )
        write_resume_state(
            out_dir,
            run_fingerprint=run_fingerprint,
            stage=f"section_{section_index:02d}_{section.name}_complete",
            preserve_as=f"stage_03_section_{section_index:02d}",
            verified_checkpoint_paths=verified_checkpoint_paths,
        )

    summaries: list[dict[str, Any]] = []
    for section in sections:
        rows = [row for row in all_rows if row["section"] == section.name]
        summaries.append(section_summary(section=section, rows=rows, baseline=baseline))
    atomic_write_json(
        out_dir / "stage_03_per_section_summary.json",
        {
            "run_fingerprint": run_fingerprint,
            "status": "complete",
            "seg_tolerance": SEG_TOLERANCE,
            "pose_tolerance": POSE_TOLERANCE,
            "sections": summaries,
        },
    )
    response_curves = {
        "schema": "jrd_section_precision_response_curves.v1",
        "status": "complete",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "run_fingerprint": run_fingerprint,
        "baseline": {
            "archive_zip_bytes": baseline["archive_zip_bytes"],
            "archive_zip_sha256": baseline["archive_zip_sha256"],
            "d_seg": baseline["d_seg"],
            "d_pose": baseline["d_pose"],
        },
        "pareto_constraint": {
            "law": "constant componentwise non-worse gate",
            "d_seg_max": baseline["d_seg"] + SEG_TOLERANCE,
            "d_pose_max": baseline["d_pose"] + POSE_TOLERANCE,
            "score_compensation_allowed": False,
        },
        "allocator_scope": (
            "planning-only pair-0 response curves; no V9/v8 or promotion consumption"
        ),
        "rows": all_rows,
    }
    response_curves["content_sha256"] = sha256_bytes(canonical_json_bytes(response_curves))
    atomic_write_json(out_dir / "section_precision_response_curves.json", response_curves)
    write_resume_state(
        out_dir,
        run_fingerprint=run_fingerprint,
        stage="per_section_response_curves_complete",
        preserve_as="stage_03_response_curves",
        verified_checkpoint_paths=verified_checkpoint_paths,
    )

    proposed: list[dict[str, Any]] = [
        summary["individual_best_byte_safe"]
        for summary in summaries
        if summary["individual_best_byte_safe"] is not None
    ]
    proposed.sort(
        key=lambda row: (
            -int(row["archive_bytes_saved"]),
            str(row["section"]),
            str(row["family"]),
        )
    )
    section_by_name = {section.name: section for section in sections}
    combined_base = parts.base_raw
    combined_code = parts.code_raw
    combined_bytes = int(baseline["archive_zip_bytes"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    combo_dir = out_dir / "combined"
    for index, choice in enumerate(proposed):
        step_path = combo_dir / f"step_{index:02d}.json"
        step_rel = step_path.relative_to(out_dir).as_posix()
        section = section_by_name[str(choice["section"])]
        trial_base, trial_code = apply_prefix_choice(
            parts=parts,
            section=section,
            family=str(choice["family"]),
            bits_removed=int(choice["bits_removed"]),
            base_raw=combined_base,
            code_raw=combined_code,
        )
        step = load_registered_checkpoint_or_none(
            step_path,
            out_dir=out_dir,
            verified_checkpoint_paths=verified_checkpoint_paths,
            run_fingerprint=run_fingerprint,
            expected_fields={"choice": choice},
        )
        if step is None:
            trial_blob = repack_blob(
                lbc, parts, base_raw=trial_base, code_raw=trial_code
            )
            measured = measure_blob(
                lbc,
                full_blob=trial_blob,
                label=f"combined_step={index};section={section.name}",
                run_fingerprint=run_fingerprint,
                out_dir=out_dir,
                gt=gt,
                segnet=segnet,
                posenet=posenet,
            )
            gate = combined_step_gate(
                measured,
                baseline=baseline,
                current_archive_bytes=combined_bytes,
            )
            step = {
                **measured,
                "choice": choice,
                **gate,
                "current_combined_bytes_before": combined_bytes,
            }
            atomic_write_json(step_path, step)
            verified_checkpoint_paths.add(step_rel)
            pending_checkpoint_paths.discard(step_rel)
            write_resume_state(
                out_dir,
                run_fingerprint=run_fingerprint,
                stage=f"combined_step_{index:02d}_complete",
                verified_checkpoint_paths=verified_checkpoint_paths,
            )
        if step["accepted"]:
            combined_base, combined_code = trial_base, trial_code
            combined_bytes = int(step["archive_zip_bytes"])
            accepted.append(step)
        else:
            rejected.append(step)

    actual_checkpoint_inventory = _checkpoint_inventory(
        out_dir, run_fingerprint=run_fingerprint
    )
    actual_checkpoint_paths = {
        str(row["relative_path"])
        for row in actual_checkpoint_inventory["records"]
    }
    if (
        pending_checkpoint_paths
        or actual_checkpoint_paths != verified_checkpoint_paths
    ):
        raise ResumeIntegrityError(
            "unregistered checkpoint survived exact remeasurement; "
            f"pending={sorted(pending_checkpoint_paths)}, "
            f"unverified={sorted(actual_checkpoint_paths - verified_checkpoint_paths)}"
        )

    selected_blob = repack_blob(
        lbc, parts, base_raw=combined_base, code_raw=combined_code
    )
    selected = measure_blob(
        lbc,
        full_blob=selected_blob,
        label="selected_combined_exact_replay",
        run_fingerprint=run_fingerprint,
        out_dir=out_dir,
        gt=gt,
        segnet=segnet,
        posenet=posenet,
        oracle_check=True,
        preserve_packet_as="selected_packet",
        force_receiver_replay=True,
    )
    if selected["archive_zip_bytes"] != combined_bytes:
        raise RuntimeError("selected replay bytes differ from final accepted combined state")
    if (
        selected["d_seg"] > baseline["d_seg"] + SEG_TOLERANCE
        or selected["d_pose"] > baseline["d_pose"] + POSE_TOLERANCE
    ):
        raise RuntimeError("selected combined replay violates the exact component guard")

    raw_precision_bits = sum(
        section_by_name[str(step["choice"]["section"])].count
        * int(step["choice"]["bits_removed"])
        for step in accepted
    )
    exact_bytes_saved = int(baseline["archive_zip_bytes"]) - int(
        selected["archive_zip_bytes"]
    )
    fixture_verdict = "GO" if exact_bytes_saved > 0 else "NO-GO"
    task_verdict = "NEEDS-MORE"
    baseline_score = float(baseline["implied_score_advisory"])
    selected_score = float(selected["implied_score_advisory"])
    allocator_input = {
        "schema": "jrd_prefix_allocator_planning_input.v1",
        "research_only": True,
        "promotion_eligible": False,
        "run_fingerprint": run_fingerprint,
        "control_law": (
            "sort individually component-safe byte-saving choices by descending saved bytes; "
            "accept a choice only when exact combined replay remains component-safe and shrinks ZIP"
        ),
        "formulation_limit": "single forward greedy pass; no offsetting component debt",
        "proposed": proposed,
        "accepted": accepted,
        "rejected": rejected,
    }
    allocator_input["content_sha256"] = sha256_bytes(canonical_json_bytes(allocator_input))
    atomic_write_json(out_dir / "allocator_planning_input.json", allocator_input)

    autopilot_output = {
        "schema": "jrd_coefficient_prefix_probe_disambiguator.v1",
        "tool": "tools/probe_jrd_coefficient_prefix.py",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "autopilot_rows": [
            {
                "candidate_id": "v9_jrd_coeff_prefix_replay_on_eligible_payload",
                "family": "jrd_coefficient_prefix",
                "lane_class": "research_only",
                "predicted_score_delta": 0.0,
                "expected_information_gain": 0.0,
                "estimated_dispatch_cost_usd": 0.0,
                "blockers": [
                    "eligible_nonlive_v9_v8_payload_missing_or_unresolved",
                    "pair0_fixture_result_nontransferable",
                ],
                "notes": (
                    "reactivate only when a sealed non-live V9/v8 LVLS1 payload is inventoried; "
                    "no dispatch or score prior follows from this fixture"
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "license_ok": True,
                "literature_anchor": (
                    "Xie et al. 2026, The Last Byte: Learning Just Enough for Machine-Oriented "
                    "Image Compression, DOI 10.1609/aaai.v40i19.38635"
                ),
                "source_supports": "machine-oriented receiver-bound stopping",
                "paper_claim_scope": "not a Pact coefficient-plane theorem",
                "pact_must_prove": "exact component-safe byte savings on eligible V9/v8 payload",
            }
        ],
    }
    atomic_write_json(out_dir / "probe_disambiguator_output.json", autopilot_output)

    posterior_candidate = {
        "schema": "jrd_continual_learning_candidate.v1",
        "canonical_consumer": "tac.continual_learning.posterior_update_locked",
        "expected_acceptance": False,
        "expected_refusal_class": "macos_substrate",
        "contest_result": {
            "axis": "cpu",
            "hardware_substrate": local_cpu_axis()["hardware_substrate"],
            "architecture_class": "jrd_coefficient_prefix_v75_fixture_pair0",
            "score_value": selected_score,
            "evidence_tag": local_cpu_axis()["evidence_tag"],
            "archive_sha256": selected["archive_zip_sha256"],
            "archive_bytes": selected["archive_zip_bytes"],
            "cpu_pose": selected["d_pose"],
            "cpu_seg": selected["d_seg"],
            "notes": "research-only pair-0 fixture; canonical posterior must refuse promotion",
            "metadata": {
                "probe_run_fingerprint": run_fingerprint,
                "n_samples": EVAL_PAIRS,
                "receipt": str(out_dir / "measurement_receipt.json"),
            },
        },
    }
    atomic_write_json(out_dir / "continual_learning_candidate.json", posterior_candidate)

    probe_outcome_candidate = {
        **canonical_probe_outcome_fields(exact_bytes_saved=exact_bytes_saved),
        "recipe_path": str(out_dir / "measurement_receipt.json"),
        "evidence_path": str(out_dir / "measurement_receipt.json"),
    }
    atomic_write_json(out_dir / "probe_outcome_candidate.json", probe_outcome_candidate)

    task_hook_candidate = {
        "task_id": "v9_jrd_coeff_prefix_probe_20260712",
        "status": "blocked",
        "test_status": "green",
        "blocker": "eligible_nonlive_v9_v8_payload_missing_or_unresolved",
        "actual_delta_s": None,
        "evidence_path": str(out_dir / "measurement_receipt.json"),
    }
    atomic_write_json(out_dir / "canonical_task_hook_candidate.json", task_hook_candidate)

    final = {
        "schema": "jrd_coefficient_prefix_measurement.v1",
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_fingerprint": run_fingerprint,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "task_id": "v9_jrd_coeff_prefix_probe_20260712",
        "task_verdict": task_verdict,
        "task_verdict_scope": (
            "V9/v8 family; the content-addressed runtime inventory found no typed, hash-bound "
            "eligible sealed non-live V9/v8 LVLS1 payload; unclassified LVLS1 artifacts remain "
            "UNKNOWN rather than being promoted by path heuristics"
        ),
        "payload_inventory": {
            "path": "v9_v8_payload_inventory.json",
            "sha256": payload_inventory["inventory_sha256"],
            "eligible_count": payload_inventory["eligible_count"],
            "unclassified_lvls1_count": payload_inventory[
                "unclassified_lvls1_count"
            ],
        },
        "fixture_verdict": fixture_verdict,
        "fixture_verdict_scope": (
            "single frozen v7.5.2 checkpoint staged for V9 apply-pass; pair 0 only; "
            "individual-best descending-rate single-pass greedy formulation; local CPU advisory; "
            "no transfer to V9/v8, other pairs, or other saved regimes"
        ),
        "review_status": "recovery-written-UNREVIEWED",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "authority_axis": local_cpu_axis()["tag"],
        "stores_consulted": (
            "corpus_query research/equations/memory/DAG/council/tasks/docs; sealed fixture; "
            "canonical upstream scorer contract; local and connected-SSD payload inventory. "
            "Deliberately not loaded: protected live V9 run and full 5GB GT cache."
        ),
        "baseline": baseline,
        "selected": selected,
        "delta": {
            "archive_bytes_saved": exact_bytes_saved,
            "d_seg": float(selected["d_seg"]) - float(baseline["d_seg"]),
            "d_pose": float(selected["d_pose"]) - float(baseline["d_pose"]),
            "implied_score_advisory": selected_score - baseline_score,
            "raw_precision_bits_removed": raw_precision_bits,
            "raw_precision_byte_equivalents": raw_precision_bits / 8.0,
        },
        "component_guard": {
            "law": "constant exact non-worse guard",
            "seg_tolerance": SEG_TOLERANCE,
            "pose_tolerance": POSE_TOLERANCE,
            "derivation": (
                "positive-control repeat measured zero within-process component noise; no "
                "cross-component score compensation admitted"
            ),
            "named_recess_measurement": "controls/baseline_repeat.json",
        },
        "controls": {
            "positive_baseline": "controls/baseline.json",
            "positive_repeat": "controls/baseline_repeat.json",
            "negative_all_zero": "controls/all_zero_negative.json",
        },
        "per_section_summary": "stage_03_per_section_summary.json",
        "accepted_combined_steps": accepted,
        "rejected_combined_steps": rejected,
        "six_hook_wire_in": {
            "sensitivity_map": "component_sensitivity_manifest.json",
            "pareto_constraint": (
                "tac.packet_compiler.jrd_coefficient_prefix.component_safe plus response artifact"
            ),
            "bit_allocator": "allocator_planning_input.json",
            "cathedral_autopilot": "system_integration_receipt.json",
            "continual_learning": "system_integration_receipt.json",
            "probe_disambiguator": "system_integration_receipt.json",
            "canonical_task_status": "system_integration_receipt.json",
            "integration_boundary": (
                "canonical consumers execute locally; the posterior records refusal, the probe "
                "ledger records DEFER, and the task ledger records the unresolved-payload blocker"
            ),
        },
        "boundaries": {
            "eligible_v9_v8_payload": False,
            "eval_pairs": EVAL_PAIRS,
            "full_archive_retains_all_pair_codes": True,
            "unscored_pair_code_rows_changed": False,
            "full_600_exact_replay": False,
            "early_boundary_late_saved_regime_replay": False,
            "contest_cpu_linux_x86_64": False,
            "contest_cuda": False,
            "upstream_evaluate_py_run": False,
            "single_seed_spine": True,
            "across_seed_variance": "UNKNOWN",
        },
        "licensing": {
            "paper_code_imported": False,
            "mvr_net_imported": False,
            "implementation": "clean-room equations and existing Pact receiver/scorer harnesses",
        },
    }
    final["content_sha256"] = sha256_bytes(canonical_json_bytes(final))
    atomic_write_json(out_dir / "measurement_receipt.json", final)
    integration = integrate_system_intelligence(
        out_dir,
        run_fingerprint=run_fingerprint,
        posterior_candidate=posterior_candidate,
        probe_outcome_candidate=probe_outcome_candidate,
        task_hook_candidate=task_hook_candidate,
    )
    atomic_write_json(
        out_dir / "stage_04_selected_replay_complete.json",
        {
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "selected_archive_sha256": selected["archive_zip_sha256"],
            "selected_archive_bytes": selected["archive_zip_bytes"],
            "archive_bytes_saved": exact_bytes_saved,
            "fixture_verdict": fixture_verdict,
            "task_verdict": task_verdict,
            "system_integration_receipt_sha256": sha256_file(
                out_dir / "system_integration_receipt.json"
            ),
            "canonical_task_status": integration["canonical_task_status"]["status"],
        },
    )
    write_resume_state(
        out_dir,
        run_fingerprint=run_fingerprint,
        stage="selected_replay_and_receipt_complete",
        preserve_as="stage_04_selected_replay",
        verified_checkpoint_paths=verified_checkpoint_paths,
    )
    cleanup_scratch(out_dir, run_fingerprint=run_fingerprint)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "durable receipt directory under experiments/results; default is "
            "experiments/results/jrd_coeff_prefix_probe_<UTC>"
        ),
    )
    destination.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="existing run directory or its run_manifest.json; resumes atomic stage/candidate state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_governed_admission("probe_jrd_coefficient_prefix")
    if os.environ.get("TAC_GOVERNED_ADMISSION") != "1":
        raise PermissionError(
            "JRD receiver/scorer sweep must be launched through tools/safe_run.py"
        )
    if args.resume_from is not None:
        resume = args.resume_from.resolve()
        out_dir = resume.parent if resume.is_file() else resume
        if not (out_dir / "run_manifest.json").is_file():
            raise FileNotFoundError(
                f"--resume-from must name an existing JRD run or run_manifest.json: {resume}"
            )
    elif args.out_dir is not None:
        out_dir = args.out_dir.resolve()
        if (out_dir / "run_manifest.json").exists():
            raise RuntimeError("existing run requires explicit --resume-from")
    else:
        out_dir = REPO / "experiments/results" / f"jrd_coeff_prefix_probe_{utc_slug()}"
    result = run_probe(out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
