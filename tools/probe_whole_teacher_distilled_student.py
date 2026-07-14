#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed cached-n600 probe for the whole-teacher distilled student.

This probe is a throughput MEANS only.  It never calls or reconstructs the
frozen teacher, never accepts synthetic/source-video substitutes, and never
activates the witness trainer.  ``preflight`` is deliberately useful even when
bulk storage is unavailable: it aggregates read-only storage and cache-custody
debts into a small repo-local receipt without touching the blocked workload
root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LANE_ID = "lane_whole_teacher_distilled_student_20260713"
PROBE_SCHEMA = "whole_teacher_distilled_student_probe.v1"
BLOCKER_SCHEMA = "whole_teacher_distilled_student_preflight_receipt.v1"
RUN_CONTRACT_SCHEMA = "whole_teacher_distilled_student_run_contract.v1"
CACHE_SCHEMA = "whole_teacher_distilled_student_cache.v2"
FIT_RESULT_SCHEMA = "whole_teacher_distilled_student_fit_measurement.v2"
ADMISSION_STAGE_SCHEMA = "whole_teacher_distilled_student_admission_decision.v1"
PARAMETER_RECEIPT_SCHEMA = "whole_teacher_distilled_student_parameter_receipt.v1"
AXIS = "[n600 macOS-MLX advisory; NumPy-fp32 reference; no score authority]"
REQUIRED_N = 600
REQUIRED_SPLITS = {"train": 480, "heldout": 120}
REQUIRED_REPLAY_STATES = {"ep150", "ep251", "ep275"}
DEFAULT_OUTPUT = Path(
    "/Volumes/VertigoDataTier/pact/whole_teacher_distilled_student_20260713"
)
DEFAULT_RECEIPT_DIR = REPO / "experiments/results/whole_teacher_distilled_student_20260713"
DEFAULT_STORAGE_PLAN = (
    REPO / ".omx/research/whole_teacher_distilled_student_storage_preflight_20260713.json"
)
DEFAULT_CACHE_MANIFEST = DEFAULT_RECEIPT_DIR / "n600_cache_manifest.json"
VERDICT_SCOPE = (
    "INSTANCE x INPUT-CACHE; INSTANCE x STORAGE-PREFLIGHT; "
    "INSTANCE x CURRENT-HOST; FORMULATION x FIT-DRIVER"
)
REQ_R = (
    "provide a writable SSD-tier workload root and a content-bound manifest for exactly "
    "600 unique real V9 rendered-state assignments over ep150/ep251/ep275, each with "
    "rendered frame, teacher centered quotient, exact full teacher input-VJP, labels, "
    "and verified SHA-256 custody; provide a usable MLX Metal device and land a typed "
    "optimizer/stage driver with full optimizer-state resume parity; then rerun the "
    "preregistered fit and worst-pair gates"
)

VERDICT_SCOPE_BY_SURFACE = {
    "storage": "INSTANCE x STORAGE-PREFLIGHT",
    "cache": "INSTANCE x INPUT-CACHE",
    "backend": "INSTANCE x CURRENT-HOST",
    "implementation": "FORMULATION x FIT-DRIVER",
    "policy": "INSTANCE x POLICY-CONTRACT",
    "source": "INSTANCE x SOURCE-CUSTODY",
}
REQ_R_BY_SURFACE = {
    "storage": "select a writable SSD-tier workload root with the requested free-space reserve",
    "cache": (
        "seal the exact real n600 rendered-frame, quotient, full input-VJP, labels, and hash bundle"
    ),
    "backend": "rerun on a host/process with an evaluated-array-proven usable MLX Metal device",
    "implementation": (
        "land and seal a typed MLX optimizer/stage driver with complete optimizer-state resume parity"
    ),
    "policy": "repair the typed policy contract without weakening or post-hoc changing its gates",
    "source": "restore and hash-close every declared probe source file",
}

SOURCE_FILES = (
    "tools/probe_whole_teacher_distilled_student.py",
    "src/tac/scorer_surrogate/whole_teacher_distilled_student.py",
    "src/tac/witness_dsl/whole_teacher_distilled_student_policy.py",
    "src/tac/canonical_equations/whole_teacher_distilled_student_20260713.py",
    ".omx/research/whole_teacher_distilled_student_build_spec_20260713.md",
)


class ProbeError(RuntimeError):
    """The probe cannot proceed without weakening an authority boundary."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_bytes(payload: Any) -> bytes:
    normalized = _jsonable(payload)
    return (
        json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _semantic_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProbeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=reject_duplicate)


def _file_custody(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProbeError(f"required file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _block(surface: str, code: str, detail: str) -> dict[str, str]:
    return {"surface": surface, "code": code, "detail": detail}


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _source_custody() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    custody: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, str]] = []
    for relative in SOURCE_FILES:
        path = REPO / relative
        try:
            custody[relative] = _file_custody(path)
        except (OSError, ProbeError) as error:
            blockers.append(_block("source", "source_file_missing", str(error)))
    return custody, blockers


def _validate_storage_plan(
    path: Path, output_dir: Path
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Read and authenticate storage without creating the workload root."""

    blockers: list[dict[str, str]] = []
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "bulk_write_attempted": False,
    }
    if not path.is_file():
        blockers.append(
            _block("storage", "storage_plan_missing", f"storage plan does not exist: {path}")
        )
        return result, blockers
    try:
        payload = _load_json(path)
    except (OSError, ValueError, ProbeError) as error:
        blockers.append(_block("storage", "storage_plan_invalid_json", str(error)))
        return result, blockers
    result["custody"] = _file_custody(path)
    result["schema"] = payload.get("schema")
    result["selected_tier"] = payload.get("selected_tier")
    result["selected_workload_root"] = payload.get("selected_workload_root")
    result["requested_bytes"] = payload.get("requested_bytes")
    result["plan_blockers"] = list(payload.get("blockers") or [])
    for item in result["plan_blockers"]:
        blockers.append(_block("storage", "storage_plan_blocked", str(item)))

    selected = payload.get("selected_workload_root")
    if not isinstance(selected, str) or not selected.strip():
        blockers.append(
            _block("storage", "selected_workload_root_missing", "plan selected no workload root")
        )
    elif Path(selected).resolve() != output_dir.resolve():
        blockers.append(
            _block(
                "storage",
                "selected_workload_root_mismatch",
                f"plan selected {selected!r}, requested {str(output_dir)!r}",
            )
        )

    selected_tier = payload.get("selected_tier")
    local_enabled = bool(
        payload.get("operator_storage_policy", {}).get("local_disk_enabled", False)
    )
    result["explicit_local_opt_in"] = local_enabled
    if selected_tier == "local" and not local_enabled:
        blockers.append(
            _block(
                "storage",
                "local_storage_not_authorized",
                "local bulk storage requires explicit operator opt-in",
            )
        )

    requested = payload.get("requested_bytes")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        blockers.append(
            _block("storage", "requested_bytes_invalid", "requested_bytes must be positive")
        )

    # Only inspect free space when the plan has selected an existing root.  Do
    # not mkdir or issue a write probe here: the canonical storage planner owns
    # that authority and the blocked preflight must remain read-only.
    if selected and Path(selected).exists():
        try:
            usage = shutil.disk_usage(Path(selected))
            result["free_bytes_at_probe"] = usage.free
            if isinstance(requested, int) and usage.free < requested:
                blockers.append(
                    _block(
                        "storage",
                        "free_bytes_below_request",
                        f"free={usage.free}, requested={requested}",
                    )
                )
        except OSError as error:
            blockers.append(_block("storage", "free_bytes_unavailable", str(error)))
    return result, blockers


def _manifest_containment_guards(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Reject forbidden provenance before delegating exact schema/hash checks."""

    blockers: list[dict[str, str]] = []
    forbidden_truthy = (
        "synthetic",
        "synthetic_data",
        "teacher_regeneration_used",
        "teacher_fallback_used",
        "source_video_cache",
        "source_video_substitute",
    )
    for field in forbidden_truthy:
        if payload.get(field):
            blockers.append(
                _block(
                    "cache",
                    "forbidden_cache_provenance",
                    f"manifest field {field!r} is truthy",
                )
            )
    if payload.get("schema") != CACHE_SCHEMA:
        blockers.append(
            _block(
                "cache",
                "cache_schema_mismatch",
                f"expected {CACHE_SCHEMA!r}, observed {payload.get('schema')!r}",
            )
        )
    return blockers


def _validate_cache_manifest(
    path: Path, bundle_root: Path | None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Use the canonical student validator; no teacher fallback exists here."""

    blockers: list[dict[str, str]] = []
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bundle_root": str(bundle_root.resolve()) if bundle_root else None,
        "teacher_calls_attempted": 0,
        "synthetic_fallback_available": False,
        "regeneration_fallback_available": False,
    }
    if not path.is_file():
        blockers.append(
            _block("cache", "cache_manifest_missing", f"cache manifest does not exist: {path}")
        )
        return result, blockers
    try:
        payload = _load_json(path)
    except (OSError, ValueError, ProbeError) as error:
        blockers.append(_block("cache", "cache_manifest_invalid_json", str(error)))
        return result, blockers
    if not isinstance(payload, Mapping):
        blockers.append(
            _block("cache", "cache_manifest_not_object", "top-level manifest must be an object")
        )
        return result, blockers
    result["custody"] = _file_custody(path)
    blockers.extend(_manifest_containment_guards(payload))
    try:
        from tac.scorer_surrogate.whole_teacher_distilled_student import (
            validate_n600_cache_manifest,
        )
    except Exception as error:  # fail closed even for optional backend import errors
        blockers.append(_block("cache", "cache_validator_import_failed", repr(error)))
        return result, blockers
    try:
        validated = validate_n600_cache_manifest(
            payload,
            bundle_root=(bundle_root or path.parent),
        )
    except Exception as error:
        blockers.append(_block("cache", "cache_manifest_validation_failed", str(error)))
        return result, blockers
    normalized = _jsonable(validated)
    result["validated"] = normalized
    result["validated_sha256"] = _payload_sha256(normalized)
    result["schema"] = payload.get("schema")
    result["assignment_count"] = len(payload.get("rows", ()))
    return result, blockers


def _construct_policy(
    *,
    student_size: str,
    anchor_cadence: int,
    exact_costate_reuse_kmax: int | None,
    seed: int,
) -> Any:
    """Construct the one exact typed policy used by preflight and admission."""

    from tac.witness_dsl.whole_teacher_distilled_student_policy import (
        EvidenceTier,
        StudentSize,
        WholeTeacherDistilledStudentPolicy,
    )

    return WholeTeacherDistilledStudentPolicy(
        selected_size=StudentSize(student_size),
        student_anchor_cadence=anchor_cadence,
        exact_costate_reuse_kmax=exact_costate_reuse_kmax,
        tier=EvidenceTier.TRAINING_GRADIENT,
        seed=seed,
    )


def _load_policy_contract(
    *,
    student_size: str,
    anchor_cadence: int,
    exact_costate_reuse_kmax: int | None,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    try:
        policy = _construct_policy(
            student_size=student_size,
            anchor_cadence=anchor_cadence,
            exact_costate_reuse_kmax=exact_costate_reuse_kmax,
            seed=seed,
        )
        contract = policy.compile_measurement_contract()
        if contract.get("student_anchor_cadence") != anchor_cadence:
            raise ProbeError("typed policy changed the requested student anchor cadence")
        if contract.get("exact_costate_reuse_kmax") != exact_costate_reuse_kmax:
            raise ProbeError("typed policy changed the independent #487 Kmax composition field")
        return _jsonable(contract), blockers
    except Exception as error:
        blockers.append(_block("policy", "policy_contract_invalid", repr(error)))
        return {}, blockers


def _validate_mlx_backend() -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Prove a usable Metal device in a disposable process.

    ``mx.metal.is_available()`` is only a build/platform signal: it can be true
    while ``load_device`` fails.  The subprocess contains both failed MLX
    initialization and its atexit handlers, while the evaluated-array canary
    establishes that a real device is usable.
    """

    result: dict[str, Any] = {
        "requested_backend": "mlx_metal",
        "cpu_fallback_allowed": False,
        "torch_debug_relabel_allowed": False,
    }
    blockers: list[dict[str, str]] = []
    probe_source = r'''
import json
import os
import sys

payload = {"metal_usable": False}
try:
    import mlx.core as mx
    payload["reported_metal_available"] = bool(mx.metal.is_available())
    if not payload["reported_metal_available"]:
        raise RuntimeError("mx.metal.is_available() returned false")
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    payload["device_info"] = info
    canary = mx.array([0.0], dtype=mx.float32)
    mx.eval(canary)
    payload["canary_value"] = float(canary[0].item())
    payload["metal_usable"] = True
except BaseException as error:
    payload["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(payload, sort_keys=True, default=str))
    sys.stdout.flush()
    os._exit(7)
print(json.dumps(payload, sort_keys=True, default=str))
sys.stdout.flush()
os._exit(0)
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_source],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        result["probe_returncode"] = completed.returncode
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise ProbeError(
                "MLX usability canary emitted no JSON; "
                f"stderr={completed.stderr[-1000:]!r}"
            )
        payload = json.loads(lines[-1])
        result.update(_jsonable(payload))
        usable = completed.returncode == 0 and payload.get("metal_usable") is True
        result["metal_usable"] = usable
        if not usable:
            blockers.append(
                _block(
                    "backend",
                    "mlx_metal_device_unusable",
                    str(payload.get("error") or completed.stderr[-1000:] or "unknown device failure"),
                )
            )
    except (OSError, ValueError, ProbeError, subprocess.TimeoutExpired) as error:
        result["metal_usable"] = False
        result["canary_error"] = repr(error)
        blockers.append(_block("backend", "mlx_usability_canary_failed", repr(error)))
    return result, blockers


def _validate_fit_implementation() -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Surface the core's honest prepared-vs-executable fit-driver state."""

    result: dict[str, Any] = {}
    blockers: list[dict[str, str]] = []
    try:
        from tac.scorer_surrogate.whole_teacher_distilled_student import FIT_DRIVER_STATUS

        result["fit_driver_status"] = FIT_DRIVER_STATUS
        result["expected_fit_driver_status"] = "EXECUTABLE_CACHED_ONLY_MLX_AUTOGRAD_V1"
        result["executable"] = result["expected_fit_driver_status"] == FIT_DRIVER_STATUS
        if not result["executable"]:
            blockers.append(
                _block(
                    "implementation",
                    "typed_mlx_optimizer_resume_driver_missing",
                    FIT_DRIVER_STATUS,
                )
            )
    except Exception as error:
        result["executable"] = False
        result["import_error"] = repr(error)
        blockers.append(_block("implementation", "fit_driver_status_unavailable", repr(error)))
    return result, blockers


def _deduplicate_blockers(blockers: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []
    for blocker in blockers:
        key = (blocker["surface"], blocker["code"], blocker["detail"])
        if key not in seen:
            seen.add(key)
            result.append(blocker)
    return result


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_existed_before = args.output_dir.exists()
    storage, storage_blockers = _validate_storage_plan(args.storage_plan, args.output_dir)
    cache, cache_blockers = _validate_cache_manifest(args.cache_manifest, args.bundle_root)
    backend, backend_blockers = _validate_mlx_backend()
    implementation, implementation_blockers = _validate_fit_implementation()
    policy, policy_blockers = _load_policy_contract(
        student_size=args.student_size,
        anchor_cadence=args.anchor_cadence,
        exact_costate_reuse_kmax=args.exact_costate_reuse_kmax,
        seed=args.seed,
    )
    sources, source_blockers = _source_custody()
    blockers = _deduplicate_blockers(
        [
            *storage_blockers,
            *cache_blockers,
            *backend_blockers,
            *implementation_blockers,
            *policy_blockers,
            *source_blockers,
        ]
    )
    surfaces = sorted({item["surface"] for item in blockers})
    active_scopes = {
        surface: VERDICT_SCOPE_BY_SURFACE[surface]
        for surface in surfaces
        if surface in VERDICT_SCOPE_BY_SURFACE
    }
    active_requirements = {
        surface: REQ_R_BY_SURFACE[surface]
        for surface in surfaces
        if surface in REQ_R_BY_SURFACE
    }
    primary_state = (
        "BLOCKED_DATA_CUSTODY"
        if "cache" in surfaces
        else "BLOCKED_STORAGE_PREFLIGHT"
        if "storage" in surfaces
        else "BLOCKED_MLX_BACKEND"
        if "backend" in surfaces
        else "UNMEASURED_BLOCKED_IMPLEMENTATION"
        if "implementation" in surfaces
        else "BLOCKED_PREFLIGHT"
        if blockers
        else "PREFLIGHT_VALIDATED"
    )
    receipt = {
        "schema": BLOCKER_SCHEMA,
        "probe_schema": PROBE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "lane_id": LANE_ID,
        "mode": args.mode,
        "status": "blocked" if blockers else "validated",
        "primary_state": primary_state,
        "blockers": blockers,
        "blocker_surfaces": surfaces,
        "verdict_scope": (
            " | ".join(active_scopes.values())
            if active_scopes
            else "NONE_PREFLIGHT_VALIDATED"
        ),
        "verdict_scope_by_blocker_surface": active_scopes,
        "req_R": (
            " | ".join(active_requirements.values())
            if active_requirements
            else "NONE_PREFLIGHT_VALIDATED"
        ),
        "req_R_by_blocker_surface": active_requirements,
        "storage_preflight": storage,
        "cache_preflight": cache,
        "backend_preflight": backend,
        "fit_implementation_preflight": implementation,
        "typed_policy": policy,
        "source_custody": sources,
        "source_custody_sha256": _payload_sha256(sources),
        "git_head": _git_head(),
        "output_dir": str(args.output_dir.resolve()),
        "receipt_dir": str(args.receipt_dir.resolve()),
        "output_existed_before": output_existed_before,
        "bulk_output_root_touched": False,
        "measurement": {
            "n_pairs": 0,
            "fit_steps": 0,
            "teacher_calls": 0,
            "forward_fidelity": "UNMEASURED_BLOCKED" if blockers else "PENDING",
            "vjp_fidelity": "UNMEASURED_BLOCKED" if blockers else "PENDING",
            "economics": "UNMEASURED_BLOCKED" if blockers else "PENDING",
        },
        "authority": {
            "axis": AXIS,
            "means_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "live_trainer_activation": False,
            "teacher_regeneration": False,
            "synthetic_data": False,
            "numpy_fp32_reference": True,
            "mlx_is_advisory": True,
            "mps_is_score_authority": False,
            "byte_closed_exact_row_required_for_score": True,
        },
        "family_boundary": (
            "whole-teacher RGB-to-quotient/value-and-VJP distillation is structurally "
            "different from the family-closed fixed-feature cheap localizer; this receipt "
            "does not reopen the localizer or close the student family"
        ),
    }
    return receipt


def _write_preflight_receipt(args: argparse.Namespace, receipt: Mapping[str, Any]) -> Path:
    if receipt["blockers"] and _path_is_within(args.receipt_dir, args.output_dir):
        raise ProbeError(
            "blocked preflight requires --receipt-dir outside --output-dir so the bulk root remains untouched"
        )
    name = "blocker_receipt.json" if receipt["blockers"] else "preflight_receipt.json"
    path = args.receipt_dir / name
    _atomic_json(path, receipt)
    return path


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Preserve stage checkpoints; equivalent resume is read-only."""

    if path.exists():
        existing = _load_json(path)
        if existing != _jsonable(payload):
            raise ProbeError(f"preserved stage checkpoint drift: {path}")
        return
    _atomic_json(path, payload)


def _immutable_bytes(path: Path, payload: bytes) -> None:
    """Preserve a content-addressed binary artifact without overwrite authority."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ProbeError(f"preserved binary artifact drift: {path}")
        return
    _atomic_bytes(path, payload)


@contextmanager
def _success_only_scratch(output_dir: Path, *, run_contract_sha256: str) -> Any:
    scratch = output_dir / f".scratch_{os.getpid()}"
    scratch.mkdir(parents=False, exist_ok=False)
    success = False
    try:
        yield scratch
        success = True
    finally:
        if success:
            files = []
            for path in sorted(item for item in scratch.rglob("*") if item.is_file()):
                files.append(
                    {
                        "path": str(path.relative_to(scratch)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            cleanup = {
                "schema": "whole_teacher_student_cleanup_manifest.v1",
                "certified_at_utc": _utc_now(),
                "run_contract_sha256": run_contract_sha256,
                "original_path": str(scratch.resolve()),
                "files": files,
                "total_bytes": sum(row["bytes"] for row in files),
                "tree_sha256": _payload_sha256(files),
                "action": "success_only_delete_certified_task_scratch",
                "reason": (
                    "task-owned transient fit material is deterministically rebuildable from "
                    "the content-bound cache, run contract, seed, and preserved stage checkpoints"
                ),
                "command": [str(item) for item in sys.argv],
                "cold_store_destination": None,
                "false_authority_score_flags": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "pointer_moved": False,
                },
                "durable_stage_checkpoints_preserved": True,
            }
            _atomic_json(output_dir / "cleanup_manifest.json", cleanup)
            shutil.rmtree(scratch)


def _run_contract(args: argparse.Namespace, preflight: Mapping[str, Any]) -> dict[str, Any]:
    policy = _construct_policy(
        student_size=args.student_size,
        anchor_cadence=args.anchor_cadence,
        exact_costate_reuse_kmax=args.exact_costate_reuse_kmax,
        seed=args.seed,
    )
    if _jsonable(policy.compile_measurement_contract()) != preflight["typed_policy"]:
        raise ProbeError("typed policy drifted between preflight and run-contract sealing")
    body = {
        "schema": RUN_CONTRACT_SCHEMA,
        "probe_schema": PROBE_SCHEMA,
        "lane_id": LANE_ID,
        "cache_manifest": preflight["cache_preflight"].get("custody"),
        "cache_validated_sha256": preflight["cache_preflight"].get("validated_sha256"),
        "storage_plan": preflight["storage_preflight"].get("custody"),
        "source_custody_sha256": preflight["source_custody_sha256"],
        "typed_policy": preflight["typed_policy"],
        "measurement_contract_sha256": policy.measurement_contract_sha256(),
        "student_size": args.student_size,
        "student_anchor_cadence": args.anchor_cadence,
        "exact_costate_reuse_kmax": args.exact_costate_reuse_kmax,
        "seed": args.seed,
        "fit_epochs": args.fit_epochs,
        "authority": preflight["authority"],
    }
    return {"payload": body, "sha256": _payload_sha256(body)}


def _core_fit_entrypoint() -> Callable[..., Any]:
    """Resolve the isolated core's cache-only fit driver, never a teacher API."""

    try:
        from tac.scorer_surrogate import whole_teacher_distilled_student as core
    except Exception as error:
        raise ProbeError(f"student core import failed: {error!r}") from error
    fit = getattr(core, "fit_measure_cached_student", None)
    if not callable(fit):
        raise ProbeError(
            "student core lacks fit_measure_cached_student; refusing an ad-hoc or teacher-backed fallback"
        )
    return fit


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError(f"{name} must be a mapping")
    return value


def _require_exact_int(value: Any, expected: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ProbeError(f"{name} must equal {expected}, observed {value!r}")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbeError(f"{name} must be a non-negative integer")
    return value


def _require_finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ProbeError(f"{name} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ProbeError(f"{name} must be {qualifier}")
    return normalized


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProbeError(f"{name} must be a lowercase SHA-256")
    return value


def _timing_summary(values: Sequence[float], *, name: str) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
    ):
        raise ProbeError(f"{name} requires finite non-negative timing rows")
    return {
        "count": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "max_ms": float(np.max(array)),
    }


def _validate_nested_metric_assignment_ids(
    assignment_id: str, metric_rows: Mapping[str, Mapping[str, Any]]
) -> None:
    """Bind every per-pair metric surface to its outer cached assignment.

    The aggregate summaries are only meaningful when their constituent rows
    are tied to the exact cached replay assignment.  Do not accept a metric
    row merely because its values look valid: a swapped assignment id would
    silently corrupt the worst-pair evidence.
    """

    for metric_name, metric_row in metric_rows.items():
        if metric_row.get("assignment_id") != assignment_id:
            raise ProbeError(f"{metric_name} assignment_id drifted from outer/cache custody")


def _validate_fit_measurement_shape(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Reject any partial, boundary-only, synthetic, or relabeled result."""

    if result.get("schema") != FIT_RESULT_SCHEMA:
        raise ProbeError(f"fit result schema drifted: {result.get('schema')!r}")
    _require_exact_int(result.get("n_pairs"), REQUIRED_N, name="result.n_pairs")
    _require_exact_int(result.get("train_pairs"), REQUIRED_SPLITS["train"], name="result.train_pairs")
    _require_exact_int(
        result.get("heldout_pairs"), REQUIRED_SPLITS["heldout"], name="result.heldout_pairs"
    )
    _require_exact_int(result.get("teacher_calls"), 0, name="result.teacher_calls")
    if result.get("backend") != "mlx" or result.get("numerical_reference") != "numpy_fp32":
        raise ProbeError("fit result backend/numerical authority drifted")
    if result.get("measurement_axis") != AXIS:
        raise ProbeError("fit result is not the exact n600 non-score MLX measurement axis")
    if result.get("student_size") != args.student_size:
        raise ProbeError("fit result student size drifted from the typed run contract")
    if result.get("measured_tier") != "training_gradient":
        raise ProbeError("fit result tier drifted from the decisive training-gradient policy")
    if result.get("student_anchor_cadence") is not None:
        raise ProbeError("core fit result improperly claimed the DSL-owned student cadence")
    _require_exact_int(result.get("fit_epochs"), args.fit_epochs, name="result.fit_epochs")
    _require_nonnegative_int(result.get("fit_steps"), name="result.fit_steps")

    authority = _require_mapping(result.get("authority"), name="result.authority")
    required_false = (
        "score_claim",
        "promotion_eligible",
        "pointer_moved",
        "teacher_regeneration",
        "synthetic_fallback",
        "cpu_fallback",
    )
    if authority.get("research_only") is not True or authority.get("means_only") is not True:
        raise ProbeError("fit result lost research-only throughput-means containment")
    if any(authority.get(field) is not False for field in required_false):
        raise ProbeError("fit result reports a forbidden authority or fallback")
    if (
        authority.get("full_input_vjp_is_decisive") is not True
        or authority.get("boundary_input_vjp_is_diagnostic_only") is not True
        or authority.get("primary_teacher_fidelity_numerical_authority") != "numpy_fp32"
        or authority.get("mlx_outputs_used_for_primary_teacher_gate") is not False
    ):
        raise ProbeError("fit result did not preserve NumPy-primary full-vector VJP authority")

    cache = _require_mapping(preflight["cache_preflight"], name="cache_preflight")
    validated = _require_mapping(cache.get("validated"), name="cache_preflight.validated")
    manifest_custody = _require_mapping(cache.get("custody"), name="cache manifest custody")
    teacher_source = _require_mapping(
        validated.get("teacher_source_custody"), name="validated teacher source custody"
    )
    top_level_bindings = {
        "cache_manifest_sha256": validated.get("manifest_sha256"),
        "cache_manifest_file_sha256": manifest_custody.get("sha256"),
        "teacher_source_custody_sha256": teacher_source.get("custody_sha256"),
        "source_custody_sha256": teacher_source.get("custody_sha256"),
        "quotient_basis_sha256": teacher_source.get("helmert_basis_sha256"),
        "post_r_input_surface_sha256": teacher_source.get("post_r_input_surface_sha256"),
    }
    for name, expected in top_level_bindings.items():
        if result.get(name) != expected:
            raise ProbeError(f"fit result top-level custody drifted for {name}")
    hashes = _require_mapping(result.get("hashes"), name="result.hashes")
    try:
        from tac.scorer_surrogate import whole_teacher_distilled_student as core
    except Exception as error:
        raise ProbeError(f"student core import failed while binding fit policy: {error!r}") from error
    architecture = core.architecture_for_size(args.student_size)
    expected_fit_policy = core.cached_student_fit_policy(
        architecture=architecture,
        seed=args.seed,
        fit_epochs=args.fit_epochs,
    ).to_dict()
    if result.get("fit_policy") != expected_fit_policy:
        raise ProbeError("fit result optimizer/objective policy drifted from CLI authority")
    expected_fit_policy_sha = _semantic_payload_sha256(expected_fit_policy)
    if hashes.get("policy_sha256") != expected_fit_policy_sha:
        raise ProbeError("fit policy hash was not re-derived from the exact typed core policy")
    source_custody = _require_mapping(preflight.get("source_custody"), name="task source custody")
    core_source_custody = _require_mapping(
        source_custody.get("src/tac/scorer_surrogate/whole_teacher_distilled_student.py"),
        name="student core source custody",
    )
    if hashes.get("source_sha256") != core_source_custody.get("sha256"):
        raise ProbeError("fit source hash drifted from preflight-sealed student core bytes")
    expected_layout_sha = core.parameter_layout_sha256(architecture)
    if (
        hashes.get("parameter_layout_sha256") != expected_layout_sha
        or result.get("parameter_layout_sha256") != expected_layout_sha
    ):
        raise ProbeError("fit result parameter layout drifted from the selected core architecture")
    validated_rows = validated.get("rows")
    observed_rows = result.get("per_pair")
    if not isinstance(validated_rows, list) or len(validated_rows) != REQUIRED_N:
        raise ProbeError("validated cache row custody is not exact n600")
    if not isinstance(observed_rows, list) or len(observed_rows) != REQUIRED_N:
        raise ProbeError("fit result per-pair measurement is not exact n600")
    expected_by_index: dict[int, Mapping[str, Any]] = {}
    for raw in validated_rows:
        row = _require_mapping(raw, name="validated cache row")
        pair_index = _require_nonnegative_int(row.get("pair_index"), name="cache pair_index")
        if pair_index in expected_by_index:
            raise ProbeError("validated cache duplicated a pair_index")
        expected_by_index[pair_index] = row
        artifacts = _require_mapping(row.get("artifacts"), name="cache row artifacts")
        if set(artifacts) != {
            "rendered_frame",
            "teacher_quotient4",
            "teacher_input_costate",
            "labels",
        }:
            raise ProbeError("validated cache row lacks exact value/VJP custody")
    if set(expected_by_index) != set(range(REQUIRED_N)):
        raise ProbeError("validated cache pair indices do not cover 0..599 exactly")
    observed_indices: set[int] = set()
    observed_ids: set[str] = set()
    forward_rows: list[Mapping[str, Any]] = []
    vjp_rows: list[Mapping[str, Any]] = []
    boundary_rows: list[Mapping[str, Any]] = []
    parity_forward_rows: list[Mapping[str, Any]] = []
    parity_vjp_rows: list[Mapping[str, Any]] = []
    student_forward_timing_rows: list[float] = []
    student_forward_vjp_timing_rows: list[float] = []
    for raw in observed_rows:
        row = _require_mapping(raw, name="fit result per-pair row")
        expected_result_row_keys = {
            "assignment_id",
            "pair_index",
            "checkpoint_name",
            "checkpoint_epoch",
            "split",
            "forward_numpy_fp32_authority",
            "full_input_vjp_numpy_fp32_decisive",
            "boundary_input_vjp_diagnostic",
            "mlx_vs_numpy_forward",
            "mlx_vs_numpy_input_vjp",
            "timing_ms",
        }
        if set(row) != expected_result_row_keys:
            raise ProbeError("fit result per-pair keys drifted from NumPy-primary authority schema")
        pair_index = _require_nonnegative_int(row.get("pair_index"), name="result pair_index")
        expected = expected_by_index.get(pair_index)
        if expected is None or pair_index in observed_indices:
            raise ProbeError("fit result pair_index is absent from or duplicated against cache custody")
        assignment_id = row.get("assignment_id")
        if (
            not isinstance(assignment_id, str)
            or assignment_id != expected.get("assignment_id")
            or assignment_id in observed_ids
        ):
            raise ProbeError("fit result assignment identity drifted from cache custody")
        for field in ("checkpoint_name", "checkpoint_epoch", "split"):
            if row.get(field) != expected.get(field):
                raise ProbeError(f"fit result {field} drifted from cache custody")
        forward_row = _require_mapping(
            row.get("forward_numpy_fp32_authority"), name="per-pair NumPy-primary forward"
        )
        vjp_row = _require_mapping(
            row.get("full_input_vjp_numpy_fp32_decisive"), name="per-pair decisive VJP"
        )
        boundary_row = _require_mapping(
            row.get("boundary_input_vjp_diagnostic"), name="per-pair diagnostic VJP"
        )
        parity_forward_row = _require_mapping(
            row.get("mlx_vs_numpy_forward"), name="per-pair MLX/NumPy forward"
        )
        parity_vjp_row = _require_mapping(
            row.get("mlx_vs_numpy_input_vjp"), name="per-pair MLX/NumPy VJP"
        )
        _validate_nested_metric_assignment_ids(
            assignment_id,
            {
                "NumPy-primary forward": forward_row,
                "NumPy-primary decisive VJP": vjp_row,
                "boundary VJP diagnostic": boundary_row,
                "MLX/NumPy forward parity": parity_forward_row,
                "MLX/NumPy VJP parity": parity_vjp_row,
            },
        )
        timing_row = _require_mapping(row.get("timing_ms"), name="per-pair student timing")
        forward_rows.append(forward_row)
        vjp_rows.append(vjp_row)
        boundary_rows.append(boundary_row)
        parity_forward_rows.append(parity_forward_row)
        parity_vjp_rows.append(parity_vjp_row)
        student_forward_timing_rows.append(
            _require_finite_number(
                timing_row.get("student_forward"), name="per-pair student forward timing"
            )
        )
        student_forward_vjp_timing_rows.append(
            _require_finite_number(
                timing_row.get("student_forward_input_vjp"),
                name="per-pair student forward+VJP timing",
            )
        )
        observed_indices.add(pair_index)
        observed_ids.add(assignment_id)

    forward = _require_mapping(result.get("forward_fidelity"), name="forward_fidelity")
    forward_n600 = _require_mapping(forward.get("n600"), name="forward_fidelity.n600")
    vjp = _require_mapping(
        result.get("vjp_fidelity_decisive_full_vector"),
        name="vjp_fidelity_decisive_full_vector",
    )
    vjp_n600 = _require_mapping(vjp.get("n600"), name="decisive VJP n600")
    parity = _require_mapping(result.get("framework_parity"), name="framework_parity")
    if (
        parity.get("primary_teacher_fidelity_uses_mlx") is not False
        or parity.get("primary_teacher_fidelity_authority") != "numpy_fp32"
        or parity.get("mlx_role") != "advisory_parity_timing_and_repeat_only"
    ):
        raise ProbeError("framework parity attempted to replace NumPy-fp32 primary authority")
    parity_forward = _require_mapping(
        parity.get("mlx_vs_numpy_forward_n600"), name="MLX/NumPy forward parity n600"
    )
    parity_vjp = _require_mapping(
        parity.get("mlx_vs_numpy_input_vjp_n600"), name="MLX/NumPy VJP parity n600"
    )
    boundary = _require_mapping(
        result.get("vjp_fidelity_boundary_diagnostic"), name="boundary VJP diagnostic"
    )
    boundary_n600 = _require_mapping(boundary.get("n600"), name="boundary VJP n600")
    heldout_forward = [
        row for row, raw in zip(forward_rows, observed_rows, strict=True) if raw["split"] == "heldout"
    ]
    heldout_vjp = [
        row for row, raw in zip(vjp_rows, observed_rows, strict=True) if raw["split"] == "heldout"
    ]
    heldout_boundary = [
        row for row, raw in zip(boundary_rows, observed_rows, strict=True) if raw["split"] == "heldout"
    ]
    recomputed_summaries = {
        "forward n600": (forward_n600, core.aggregate_forward_pair_metrics(forward_rows)),
        "forward heldout n120": (
            _require_mapping(forward.get("heldout_n120"), name="forward heldout n120"),
            core.aggregate_forward_pair_metrics(heldout_forward),
        ),
        "decisive VJP n600": (vjp_n600, core.aggregate_vjp_pair_metrics(vjp_rows)),
        "decisive VJP heldout n120": (
            _require_mapping(vjp.get("heldout_n120"), name="decisive VJP heldout n120"),
            core.aggregate_vjp_pair_metrics(heldout_vjp),
        ),
        "boundary VJP n600": (
            boundary_n600,
            core.aggregate_vjp_pair_metrics(boundary_rows),
        ),
        "boundary VJP heldout n120": (
            _require_mapping(boundary.get("heldout_n120"), name="boundary VJP heldout n120"),
            core.aggregate_vjp_pair_metrics(heldout_boundary),
        ),
        "MLX/NumPy forward n600": (
            parity_forward,
            core.aggregate_forward_pair_metrics(parity_forward_rows),
        ),
        "MLX/NumPy VJP n600": (
            parity_vjp,
            core.aggregate_vjp_pair_metrics(parity_vjp_rows),
        ),
    }
    for name, (observed, recomputed) in recomputed_summaries.items():
        if observed != recomputed:
            raise ProbeError(f"{name} summary disagrees with exact per-pair rows")
    for name, summary in (
        ("forward", forward_n600),
        ("decisive full VJP", vjp_n600),
        ("forward parity", parity_forward),
        ("VJP parity", parity_vjp),
    ):
        _require_exact_int(summary.get("pair_count"), REQUIRED_N, name=f"{name}.pair_count")

    charged_timings = _require_mapping(result.get("charged_timings"), name="charged_timings")
    if charged_timings.get("warmup_excluded") is not True:
        raise ProbeError("charged student timing failed to exclude first-use warmup")
    expected_student_forward_timing = _timing_summary(
        student_forward_timing_rows, name="n600 student forward timing"
    )
    expected_student_vjp_timing = _timing_summary(
        student_forward_vjp_timing_rows, name="n600 student forward+VJP timing"
    )
    if charged_timings.get("student_forward") != expected_student_forward_timing:
        raise ProbeError("charged student forward summary disagrees with n600 raw timing rows")
    if charged_timings.get("student_forward_input_vjp") != expected_student_vjp_timing:
        raise ProbeError("charged student forward+VJP summary disagrees with n600 raw timing rows")

    return {
        "validated_cache": validated,
        "forward_n600": forward_n600,
        "vjp_n600": vjp_n600,
        "parity_forward_n600": parity_forward,
        "parity_vjp_n600": parity_vjp,
        "charged_timings": charged_timings,
        "hashes": hashes,
    }


def _materialize_best_parameter_blob(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the best parameter bytes from the preserved completion checkpoint."""

    try:
        from tac.scorer_surrogate import whole_teacher_distilled_student as core
    except Exception as error:
        raise ProbeError(f"student core import failed while sealing parameters: {error!r}") from error
    architecture = core.architecture_for_size(args.student_size)
    expected_layout = core.parameter_layout_sha256(architecture)
    if _require_sha256(hashes.get("parameter_layout_sha256"), name="parameter layout") != expected_layout:
        raise ProbeError("fit result parameter layout hash drifted from the sealed architecture")
    fit_steps = _require_nonnegative_int(result.get("fit_steps"), name="result.fit_steps")
    stem = f"stage_02_completion_step_{fit_steps:06d}"
    metadata_path = args.output_dir / f"{stem}.json"
    outer = _require_mapping(_load_json(metadata_path), name="completion checkpoint metadata")
    if outer.get("run_contract_sha256") != contract["sha256"]:
        raise ProbeError("completion checkpoint run-contract binding drifted")
    npz_name = outer.get("npz")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise ProbeError("completion checkpoint NPZ path is unsafe")
    npz_path = metadata_path.parent / npz_name
    if _sha256(npz_path) != outer.get("npz_sha256"):
        raise ProbeError("completion checkpoint NPZ custody drifted")
    inner = _require_mapping(outer.get("metadata"), name="core completion metadata")
    if inner.get("stage") != "completion" or inner.get("hashes") != dict(hashes):
        raise ProbeError("core completion metadata source/policy/cache binding drifted")
    if inner.get("terminal_result") != result:
        raise ProbeError("completion checkpoint terminal result differs from returned measurement")
    with np.load(npz_path, allow_pickle=False) as archive:
        parameters = {
            name.removeprefix("best_parameter__"): np.array(archive[name], copy=True)
            for name in archive.files
            if name.startswith("best_parameter__")
        }
        if "history__train_step_ms" not in archive.files:
            raise ProbeError("completion checkpoint lacks charged optimizer timing history")
        train_step_ms = np.asarray(archive["history__train_step_ms"], dtype=np.float64).copy()
    if train_step_ms.shape != (fit_steps,):
        raise ProbeError("completion checkpoint optimizer timing count drifted from fit_steps")
    anchor_update_timing = _timing_summary(
        train_step_ms.tolist(), name="completion checkpoint anchor-update timing"
    )
    charged_timings = _require_mapping(result.get("charged_timings"), name="charged timings")
    if charged_timings.get("anchor_update") != anchor_update_timing:
        raise ProbeError("charged anchor-update summary disagrees with preserved raw history")
    blob = core.serialize_student_parameters(architecture, parameters)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    if blob_sha256 != _require_sha256(
        result.get("best_parameters_sha256"), name="result.best_parameters_sha256"
    ):
        raise ProbeError("best parameter bytes drifted from the measured result")
    _require_exact_int(
        result.get("best_parameters_blob_bytes"),
        len(blob),
        name="result.best_parameters_blob_bytes",
    )
    durable_claim = _require_mapping(
        result.get("durable_best_parameters_blob"), name="durable parameter handoff claim"
    )
    if (
        durable_claim.get("status")
        != "PROBE_MUST_MATERIALIZE_FROM_COMPLETION_CHECKPOINT"
        or durable_claim.get("sha256") != blob_sha256
        or durable_claim.get("bytes") != len(blob)
        or durable_claim.get("path") is not None
    ):
        raise ProbeError("core durable-parameter handoff contract drifted")
    restored_architecture, _restored_parameters = core.deserialize_student_parameters(blob)
    if restored_architecture != architecture:
        raise ProbeError("serialized student architecture drifted on parse-back")
    blob_path = args.output_dir / "student_best_parameters.wtds"
    _immutable_bytes(blob_path, blob)
    receipt = {
        "schema": PARAMETER_RECEIPT_SCHEMA,
        "run_contract_sha256": contract["sha256"],
        "path": str(blob_path.resolve()),
        "bytes": blob_path.stat().st_size,
        "sha256": _sha256(blob_path),
        "parameter_layout_sha256": expected_layout,
        "student_size": args.student_size,
        "source_checkpoint": str(metadata_path.resolve()),
        "source_checkpoint_sha256": _sha256(metadata_path),
        "source_npz_sha256": _sha256(npz_path),
        "anchor_update_timing": anchor_update_timing,
        "parse_back_verified": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _immutable_json(args.output_dir / "student_best_parameters_receipt.json", receipt)
    return receipt


def _verify_bundle_file_custody(
    custody: Any, *, bundle_root: Path, name: str
) -> dict[str, Any]:
    value = _require_mapping(custody, name=name)
    if set(value) != {"path", "bytes", "sha256"}:
        raise ProbeError(f"{name} requires exact path/bytes/sha256 custody")
    relative = value.get("path")
    byte_count = value.get("bytes")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ProbeError(f"{name} path is unsafe")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise ProbeError(f"{name} byte count must be positive")
    expected_sha = _require_sha256(value.get("sha256"), name=f"{name}.sha256")
    path = (bundle_root / relative).resolve()
    if not _path_is_within(path, bundle_root) or not path.is_file():
        raise ProbeError(f"{name} is absent from the validated bundle root")
    if path.stat().st_size != byte_count or _sha256(path) != expected_sha:
        raise ProbeError(f"{name} byte custody drifted")
    return {
        "path": str(path),
        "bytes": byte_count,
        "sha256": expected_sha,
    }


def _repeat_stream_verified(
    value: Any,
    *,
    name: str,
    expected_scope: str,
    verification_field: str,
    extra_required_keys: frozenset[str] = frozenset(),
    expected_numerical_authority: str | None = None,
) -> bool:
    stream = _require_mapping(value, name=name)
    required = {
        "scope",
        "pair_count",
        "timed",
        "charged_student_timing_includes_repeat",
        "first_forward_sha256",
        "second_forward_sha256",
        "forward_equal",
        "first_input_vjp_sha256",
        "second_input_vjp_sha256",
        "input_vjp_equal",
        "first_combined_sha256",
        "second_combined_sha256",
        "combined_equal",
        verification_field,
    }
    required |= extra_required_keys
    if set(stream) != required:
        raise ProbeError(f"{name} keys drifted")
    if stream.get("scope") != expected_scope:
        raise ProbeError(f"{name} scope drifted")
    if (
        expected_numerical_authority is not None
        and stream.get("numerical_authority") != expected_numerical_authority
    ):
        raise ProbeError(f"{name} numerical authority drifted")
    _require_exact_int(stream.get("pair_count"), REQUIRED_N, name=f"{name}.pair_count")
    if stream.get("timed") is not False or stream.get(
        "charged_student_timing_includes_repeat"
    ) is not False:
        raise ProbeError(f"{name} leaked into charged student timing")
    first_forward = _require_sha256(
        stream.get("first_forward_sha256"), name=f"{name} first forward"
    )
    second_forward = _require_sha256(
        stream.get("second_forward_sha256"), name=f"{name} second forward"
    )
    first_vjp = _require_sha256(
        stream.get("first_input_vjp_sha256"), name=f"{name} first VJP"
    )
    second_vjp = _require_sha256(
        stream.get("second_input_vjp_sha256"), name=f"{name} second VJP"
    )
    first_combined = _require_sha256(
        stream.get("first_combined_sha256"), name=f"{name} first combined"
    )
    second_combined = _require_sha256(
        stream.get("second_combined_sha256"), name=f"{name} second combined"
    )
    derived = bool(
        stream.get("forward_equal") is True
        and stream.get("input_vjp_equal") is True
        and stream.get("combined_equal") is True
        and first_forward == second_forward
        and first_vjp == second_vjp
        and first_combined == second_combined
    )
    if stream.get(verification_field) is not derived:
        raise ProbeError(f"{name} verification flag was not derived from exact stream hashes")
    return derived


def _validate_deterministic_repeat(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Require independent deterministic repeats for authority and advisory paths.

    The NumPy-fp32 stream proves verdict-authority determinism.  The separate
    MLX stream proves that the execution candidate is repeatable too, without
    allowing its second pass to distort charged timing economics.
    """

    repeat = _require_mapping(result.get("deterministic_repeat"), name="deterministic_repeat")
    numpy_repeat_verified = _repeat_stream_verified(
        repeat,
        name="NumPy-fp32 authority deterministic repeat",
        expected_scope="full_ordered_n600_numpy_fp32_authority_forward_and_input_vjp_stream",
        verification_field="authority_verified",
        extra_required_keys=frozenset(
            {"numerical_authority", "mlx_advisory", "all_required_streams_equal"}
        ),
        expected_numerical_authority="numpy_fp32",
    )
    mlx_repeat_verified = _repeat_stream_verified(
        repeat.get("mlx_advisory"),
        name="MLX advisory deterministic repeat",
        expected_scope="full_ordered_n600_mlx_advisory_forward_and_input_vjp_stream",
        verification_field="advisory_verified",
    )
    repeat_verified = numpy_repeat_verified and mlx_repeat_verified
    if repeat.get("all_required_streams_equal") is not repeat_verified:
        raise ProbeError("deterministic-repeat aggregate verdict was not derived from both streams")
    if result.get("deterministic_repeat_verified") is not repeat_verified:
        raise ProbeError("top-level deterministic-repeat verdict was not derived from both streams")
    return repeat, repeat_verified


def _build_admission_evidence(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    result: Mapping[str, Any],
    surfaces: Mapping[str, Mapping[str, Any]],
    parameter_receipt: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Construct only custody-derived typed evidence and evaluate it once."""

    from tac.witness_dsl.whole_teacher_distilled_student_policy import (
        StudentAdmissionEvidence,
    )

    policy = _construct_policy(
        student_size=args.student_size,
        anchor_cadence=args.anchor_cadence,
        exact_costate_reuse_kmax=args.exact_costate_reuse_kmax,
        seed=args.seed,
    )
    validated = surfaces["validated_cache"]
    teacher_custody = _require_mapping(
        validated.get("teacher_source_custody"), name="validated teacher source custody"
    )
    result_source = _require_mapping(result.get("source_custody"), name="result.source_custody")
    hashes = surfaces["hashes"]
    cache_preflight = _require_mapping(preflight["cache_preflight"], name="cache_preflight")
    manifest_custody = _require_mapping(
        cache_preflight.get("custody"), name="cache manifest file custody"
    )

    manifest_sha = _require_sha256(
        validated.get("manifest_sha256"), name="validated manifest semantic hash"
    )
    manifest_file_sha = _require_sha256(
        manifest_custody.get("sha256"), name="cache manifest file hash"
    )
    teacher_source_sha = _require_sha256(
        teacher_custody.get("custody_sha256"), name="teacher source custody hash"
    )
    semantic_bindings = {
        "cache_manifest_sha256": manifest_sha,
        "cache_manifest_file_sha256": manifest_file_sha,
        "teacher_source_custody_sha256": teacher_source_sha,
        "r_operator_sha256": _require_sha256(
            teacher_custody.get("r_operator_sha256"), name="actual R operator hash"
        ),
        "scalar_objective_sha256": _require_sha256(
            teacher_custody.get("scalar_objective_sha256"), name="scalar objective hash"
        ),
        "quotient_basis_sha256": _require_sha256(
            teacher_custody.get("helmert_basis_sha256"), name="quotient basis hash"
        ),
        "post_r_input_surface_sha256": _require_sha256(
            teacher_custody.get("post_r_input_surface_sha256"),
            name="post-R input surface hash",
        ),
    }
    for name, expected in semantic_bindings.items():
        if result_source.get(name) != expected:
            raise ProbeError(f"result source custody drifted for {name}")
    for name in (
        "cache_manifest_sha256",
        "cache_manifest_file_sha256",
        "teacher_source_custody_sha256",
        "scalar_objective_sha256",
    ):
        if hashes.get(name) != semantic_bindings[name]:
            raise ProbeError(f"fit contract hash drifted for {name}")

    repository_files = teacher_custody.get("repository_files")
    if not isinstance(repository_files, list):
        raise ProbeError("validated teacher repository file custody is missing")
    weights = [
        _require_mapping(value, name="teacher repository source")
        for value in repository_files
        if isinstance(value, Mapping) and value.get("path") == "upstream/models/segnet.safetensors"
    ]
    if len(weights) != 1:
        raise ProbeError("validated teacher custody lacks exactly one SegNet weights source")
    frozen_teacher_weights_sha = _require_sha256(
        weights[0].get("sha256"), name="frozen teacher weights hash"
    )

    repeat, repeat_verified = _validate_deterministic_repeat(result)
    deterministic_repeat_sha = _payload_sha256(repeat)

    timings = surfaces["charged_timings"]
    student_forward = _require_mapping(
        timings.get("student_forward"), name="student forward timings"
    )
    student_forward_vjp = _require_mapping(
        timings.get("student_forward_input_vjp"), name="student forward+VJP timings"
    )
    anchor_update = _require_mapping(
        timings.get("anchor_update"), name="anchor update timings"
    )
    checkpoint_anchor_update = _require_mapping(
        parameter_receipt.get("anchor_update_timing"),
        name="completion-checkpoint anchor update timings",
    )
    if checkpoint_anchor_update != anchor_update:
        raise ProbeError("anchor update timing lost completion-checkpoint custody")
    student_forward_ms = _require_finite_number(
        student_forward.get("median_ms"), name="student forward median", positive=True
    )
    student_forward_vjp_ms = _require_finite_number(
        student_forward_vjp.get("median_ms"), name="student forward+VJP median", positive=True
    )
    anchor_update_ms = _require_finite_number(
        checkpoint_anchor_update.get("median_ms"),
        name="anchor update median",
        positive=True,
    )
    teacher = _require_mapping(
        timings.get("exact_teacher_inputs"), name="exact teacher timing inputs"
    )
    timing_ready = (
        timings.get("fully_charged_economics_ready") is True
        and teacher.get("status") == "CONTENT_BOUND_MATCHED_AXIS_RECEIPT_VERIFIED"
    )
    teacher_forward_ms: float | None = None
    teacher_forward_vjp_ms: float | None = None
    teacher_timing_sha: str | None = None
    teacher_axis = "UNMEASURED"
    timing_custody: dict[str, Any] | None = None
    if timing_ready:
        bundle_root = Path(str(validated.get("bundle_root"))).resolve()
        receipt_claim = _require_mapping(
            teacher.get("teacher_timing_receipt"), name="teacher timing receipt claim"
        )
        receipt_custody = _verify_bundle_file_custody(
            receipt_claim,
            bundle_root=bundle_root,
            name="teacher timing receipt",
        )
        raw_custody = _verify_bundle_file_custody(
            teacher.get("raw_observations"),
            bundle_root=bundle_root,
            name="teacher timing raw observations",
        )
        teacher_timing_sha = receipt_custody["sha256"]
        if teacher_timing_sha != teacher.get("teacher_timing_receipt_sha256"):
            raise ProbeError("teacher timing receipt hash drifted inside result")
        if (
            timings.get("teacher_timing_receipt_sha256") != teacher_timing_sha
            or timings.get("teacher_timing_receipt_path")
            != receipt_claim.get("path")
        ):
            raise ProbeError("charged timing receipt summary drifted from verified teacher inputs")
        if teacher.get("hardware_fingerprint_sha256") != result.get(
            "hardware_fingerprint_sha256"
        ):
            raise ProbeError("teacher/student timing hardware fingerprint drifted")
        teacher_axis_value = teacher.get("measurement_axis")
        if teacher_axis_value != result.get("measurement_axis"):
            raise ProbeError("teacher/student timing measurement axes drifted")
        if timings.get("student_teacher_axis_matched") is not True:
            raise ProbeError("charged timing summary denied the verified matched axis")
        teacher_axis = str(teacher_axis_value)
        teacher_forward_ms = _require_finite_number(
            teacher.get("exact_teacher_forward_ms"),
            name="exact teacher forward timing",
            positive=True,
        )
        teacher_forward_vjp_ms = _require_finite_number(
            teacher.get("exact_teacher_forward_input_vjp_ms"),
            name="exact teacher forward+VJP timing",
            positive=True,
        )
        timing_custody = {"receipt": receipt_custody, "raw_observations": raw_custody}
    else:
        absent_fields = (
            "exact_teacher_forward_ms",
            "exact_teacher_forward_input_vjp_ms",
            "teacher_timing_receipt",
            "teacher_timing_receipt_sha256",
            "raw_observations",
            "measurement_axis",
            "hardware_fingerprint_sha256",
        )
        if any(teacher.get(field) is not None for field in absent_fields):
            raise ProbeError("unverified teacher timing input carried partial authority fields")
        if (
            timings.get("teacher_timing_receipt_sha256") is not None
            or timings.get("teacher_timing_receipt_path") is not None
            or timings.get("student_teacher_axis_matched") is not False
        ):
            raise ProbeError("unverified charged timing summary carried partial authority")

    forward = surfaces["forward_n600"]
    vjp = surfaces["vjp_n600"]
    parity_forward = surfaces["parity_forward_n600"]
    parity_vjp = surfaces["parity_vjp_n600"]
    replay_states = tuple(
        f"ep{epoch}"
        for epoch in sorted(
            {
                _require_nonnegative_int(row.get("checkpoint_epoch"), name="replay epoch")
                for row in validated["rows"]
            }
        )
    )
    evidence = StudentAdmissionEvidence(
        n_pairs=REQUIRED_N,
        cache_manifest_valid=True,
        real_rendered_states=True,
        exact_teacher_quotient_custody=True,
        exact_teacher_input_vjp_custody=True,
        actual_r_custody_verified=True,
        frozen_teacher_custody_verified=True,
        scalar_objective_custody_verified=True,
        deterministic_repeat_verified=repeat_verified,
        replay_states=replay_states,
        measured_tier=policy.tier,
        measured_student_size=policy.selected_size,
        measured_student_anchor_cadence=policy.student_anchor_cadence,
        backend="mlx",
        teacher_calls=0,
        cache_manifest_sha256=manifest_sha,
        cache_manifest_file_sha256=manifest_file_sha,
        cache_validated_sha256=_require_sha256(
            cache_preflight.get("validated_sha256"), name="validated cache aggregate hash"
        ),
        source_custody_sha256=_require_sha256(
            preflight.get("source_custody_sha256"), name="task source custody hash"
        ),
        teacher_source_custody_sha256=teacher_source_sha,
        actual_r_operator_sha256=semantic_bindings["r_operator_sha256"],
        post_r_input_surface_sha256=semantic_bindings["post_r_input_surface_sha256"],
        frozen_teacher_weights_sha256=frozen_teacher_weights_sha,
        quotient_basis_sha256=semantic_bindings["quotient_basis_sha256"],
        scalar_objective_sha256=semantic_bindings["scalar_objective_sha256"],
        fit_policy_sha256=_require_sha256(hashes.get("policy_sha256"), name="fit policy hash"),
        parameter_layout_sha256=_require_sha256(
            parameter_receipt.get("parameter_layout_sha256"), name="parameter layout receipt hash"
        ),
        student_parameters_sha256=_require_sha256(
            parameter_receipt.get("sha256"), name="student parameter blob hash"
        ),
        deterministic_repeat_sha256=deterministic_repeat_sha,
        measurement_contract_sha256=policy.measurement_contract_sha256(),
        teacher_timing_receipt_sha256=teacher_timing_sha,
        forward_worst_pair_cosine=_require_finite_number(
            forward.get("worst_cosine"), name="forward worst-pair cosine"
        ),
        forward_worst_pair_relative_l2=_require_finite_number(
            forward.get("worst_relative_l2"), name="forward worst-pair relative L2"
        ),
        forward_worst_pair_argmax_disagreement=_require_finite_number(
            forward.get("worst_argmax_disagreement"),
            name="forward worst-pair argmax disagreement",
        ),
        vjp_worst_pair_cosine=_require_finite_number(
            vjp.get("worst_cosine"), name="decisive VJP worst-pair cosine"
        ),
        vjp_worst_pair_relative_l2=_require_finite_number(
            vjp.get("worst_relative_l2"), name="decisive VJP worst-pair relative L2"
        ),
        numpy_framework_forward_worst_pair_cosine=_require_finite_number(
            parity_forward.get("worst_cosine"), name="NumPy/MLX forward worst-pair cosine"
        ),
        numpy_framework_vjp_worst_pair_cosine=_require_finite_number(
            parity_vjp.get("worst_cosine"), name="NumPy/MLX VJP worst-pair cosine"
        ),
        charged_timing_measured=timing_ready,
        student_forward_cost_ms=student_forward_ms,
        student_forward_vjp_cost_ms=student_forward_vjp_ms,
        exact_teacher_forward_cost_ms=teacher_forward_ms,
        exact_teacher_forward_vjp_cost_ms=teacher_forward_vjp_ms,
        anchor_update_cost_ms=anchor_update_ms,
        student_timing_axis=str(result["measurement_axis"]),
        teacher_timing_axis=teacher_axis,
        measurement_axis=str(result["measurement_axis"]),
    )
    decision = policy.evaluate_evidence(evidence)
    diagnostics = {
        "evidence": _jsonable(evidence),
        "evidence_sha256": _payload_sha256(evidence),
        "derived_economics": evidence.derived_economics_for(policy),
        "deterministic_repeat": _jsonable(repeat),
        "teacher_timing_custody": timing_custody,
        "measurement_contract_sha256": policy.measurement_contract_sha256(),
    }
    return policy, decision, diagnostics


def _admission_disposition(decision: Any) -> str:
    """Translate one typed decision without changing its scope or requirements."""

    admitted = getattr(decision, "admitted", None)
    if admitted is True:
        return "ADMISSION_PREPARED_NOT_FIRED"
    if admitted is False:
        return "MEASUREMENT_COMPLETE_SCOPED_NO_GO"
    raise ProbeError("typed admission decision lacks a boolean admitted field")


def _fit_measure(args: argparse.Namespace, preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Run only after the complete storage/cache/source/policy gate is green.

    The model module owns math and framework parity.  This orchestration owns
    crash custody: callbacks persist complete, distinct stage/epoch states and
    never overwrite a prior checkpoint.
    """

    if preflight["blockers"]:
        raise ProbeError("fit refused because aggregated preflight is blocked")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = _run_contract(args, preflight)
    _immutable_json(args.output_dir / "run_contract.json", contract)
    _immutable_json(
        args.output_dir / "stage_00_storage_preflight.json",
        {
            "schema": "whole_teacher_student_stage.v1",
            "stage": "storage_preflight",
            "run_contract_sha256": contract["sha256"],
            "storage": preflight["storage_preflight"],
        },
    )
    _immutable_json(
        args.output_dir / "stage_01_cache_validation.json",
        {
            "schema": "whole_teacher_student_stage.v1",
            "stage": "cache_validation",
            "run_contract_sha256": contract["sha256"],
            "cache": preflight["cache_preflight"],
        },
    )

    fit_entrypoint = _core_fit_entrypoint()

    def checkpoint(stage: str, step: int, state: Mapping[str, Any]) -> dict[str, Any]:
        if not stage.replace("_", "").isalnum():
            raise ProbeError(f"invalid checkpoint stage name: {stage!r}")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ProbeError("checkpoint step must be a non-negative integer")
        arrays = state.get("arrays")
        metadata = state.get("metadata")
        if not isinstance(arrays, Mapping) or not isinstance(metadata, Mapping):
            raise ProbeError("checkpoint state requires complete arrays and metadata mappings")
        stem = f"stage_02_{stage}_step_{step:06d}"
        npz_path = args.output_dir / f"{stem}.npz"
        meta_path = args.output_dir / f"{stem}.json"
        if npz_path.exists() or meta_path.exists():
            if not (npz_path.is_file() and meta_path.is_file()):
                raise ProbeError(f"partial preserved checkpoint: {stem}")
            prior = _load_json(meta_path)
            if prior.get("run_contract_sha256") != contract["sha256"]:
                raise ProbeError(f"checkpoint contract drift: {stem}")
            if prior.get("npz_sha256") != _sha256(npz_path):
                raise ProbeError(f"checkpoint byte drift: {stem}")
            return prior
        _atomic_npz(npz_path, arrays)
        payload = {
            "schema": "whole_teacher_student_fit_checkpoint.v1",
            "stage": stage,
            "step": step,
            "run_contract_sha256": contract["sha256"],
            "npz": npz_path.name,
            "npz_bytes": npz_path.stat().st_size,
            "npz_sha256": _sha256(npz_path),
            "metadata": _jsonable(metadata),
        }
        _atomic_json(meta_path, payload)
        return payload

    resume_state: dict[str, Any] | None = None
    if args.resume_from is not None:
        meta = _load_json(args.resume_from)
        if meta.get("run_contract_sha256") != contract["sha256"]:
            raise ProbeError("--resume-from run contract drift")
        npz_path = args.resume_from.parent / meta["npz"]
        if _sha256(npz_path) != meta.get("npz_sha256"):
            raise ProbeError("--resume-from NPZ custody drift")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        resume_state = {"arrays": arrays, "metadata": meta["metadata"]}

    started = time.perf_counter()
    with _success_only_scratch(
        args.output_dir, run_contract_sha256=contract["sha256"]
    ) as scratch:
        result = fit_entrypoint(
            manifest_path=args.cache_manifest,
            bundle_root=args.bundle_root or args.cache_manifest.parent,
            student_size=args.student_size,
            seed=args.seed,
            fit_epochs=args.fit_epochs,
            checkpoint_callback=checkpoint,
            resume_state=resume_state,
            scratch_dir=scratch,
            backend="mlx",
            numerical_reference="numpy_fp32",
        )
    elapsed = time.perf_counter() - started
    normalized = _jsonable(result)
    if not isinstance(normalized, Mapping):
        raise ProbeError("fit_measure_cached_student must return a mapping or dataclass")
    surfaces = _validate_fit_measurement_shape(args, preflight, normalized)
    parameter_receipt = _materialize_best_parameter_blob(
        args, contract, normalized, surfaces["hashes"]
    )
    heldout = {
        "schema": "whole_teacher_student_stage.v1",
        "stage": "heldout_measurement",
        "run_contract_sha256": contract["sha256"],
        "result": normalized,
        "elapsed_seconds": elapsed,
        "axis": AXIS,
    }
    _immutable_json(args.output_dir / "stage_03_heldout_measurement.json", heldout)
    policy, decision, diagnostics = _build_admission_evidence(
        args,
        preflight,
        normalized,
        surfaces,
        parameter_receipt,
    )
    decision_payload = decision.to_dict()
    disposition = _admission_disposition(decision)
    admission = {
        "schema": ADMISSION_STAGE_SCHEMA,
        "disposition": disposition,
        "run_contract_sha256": contract["sha256"],
        "heldout_sha256": _sha256(args.output_dir / "stage_03_heldout_measurement.json"),
        "parameter_receipt": parameter_receipt,
        "policy": _jsonable(policy.compile_measurement_contract()),
        "policy_measurement_contract_sha256": policy.measurement_contract_sha256(),
        "typed_admission_decision": decision_payload,
        "typed_evidence": diagnostics["evidence"],
        "typed_evidence_sha256": diagnostics["evidence_sha256"],
        "derived_economics": diagnostics["derived_economics"],
        "deterministic_repeat": diagnostics["deterministic_repeat"],
        "teacher_timing_custody": diagnostics["teacher_timing_custody"],
        "verdict_scope": decision.verdict_scope,
        "req_R": decision.req_R,
        "n_pairs": REQUIRED_N,
        "fit_steps": normalized["fit_steps"],
        "teacher_calls": normalized["teacher_calls"],
        "admission_prepared": bool(decision.admitted),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "live_trainer_activation": False,
        "cleanup": {
            "status": "SUCCESS_ONLY_SCRATCH_REMOVED",
            "scratch_root": "task_output_context_managed",
            "durable_checkpoints_preserved": True,
            "manifest": "cleanup_manifest.json",
            "manifest_sha256": _sha256(args.output_dir / "cleanup_manifest.json"),
        },
    }
    _immutable_json(args.output_dir / "stage_04_admission_decision.json", admission)
    return {
        "run_contract": contract,
        "measurement": normalized,
        "admission": admission,
        "disposition": disposition,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "fit-measure"),
        default="preflight",
        help="preflight is read-only except for its small receipt; fit-measure requires every gate",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--storage-plan", type=Path, default=DEFAULT_STORAGE_PLAN)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        help="root for manifest-relative cached tensors; defaults to manifest parent",
    )
    parser.add_argument("--student-size", choices=("tiny", "small", "medium"), default="small")
    parser.add_argument(
        "--anchor-cadence",
        type=int,
        default=20,
        help="student exact-teacher anchor K; independent of optional #487 reuse Kmax",
    )
    parser.add_argument(
        "--exact-costate-reuse-kmax",
        type=int,
        choices=(2,),
        help="optional separate #487 composition field; never caps --anchor-cadence",
    )
    parser.add_argument("--seed", type=int, default=455)
    parser.add_argument("--fit-epochs", type=int, default=40)
    parser.add_argument("--resume-from", type=Path)
    return parser


def _partial_fit_observation(args: argparse.Namespace) -> dict[str, Any]:
    """Recover only counters proven by durable stage checkpoints after failure."""

    unknown = "UNKNOWN_AFTER_PARTIAL_EXECUTION"
    rows: list[dict[str, Any]] = []
    cleanup_path = args.output_dir / "cleanup_manifest.json"
    cleanup: Any = "FAILURE_RETAINED_CERTIFY_OR_BLOCK"
    if cleanup_path.is_file():
        cleanup = {
            "status": "SUCCESS_ONLY_SCRATCH_REMOVED_BEFORE_LATER_FAILURE",
            "path": str(cleanup_path.resolve()),
            "sha256": _sha256(cleanup_path),
        }
    if args.output_dir.is_dir():
        for path in args.output_dir.glob("stage_02_*_step_*.json"):
            try:
                outer = _require_mapping(_load_json(path), name="partial checkpoint")
                inner = _require_mapping(
                    outer.get("metadata"), name="partial core checkpoint metadata"
                )
                step = _require_nonnegative_int(
                    inner.get("optimizer_step"), name="partial optimizer step"
                )
                stage = inner.get("stage")
                if not isinstance(stage, str) or not stage:
                    raise ProbeError("partial checkpoint stage is absent")
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "step": step,
                        "stage": stage,
                        "teacher_calls": inner.get("teacher_calls", unknown),
                        "terminal_result": inner.get("terminal_result"),
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                )
            except (OSError, ValueError, ProbeError):
                continue
    if not rows:
        return {
            "n_pairs": unknown,
            "fit_steps": unknown,
            "teacher_calls": unknown,
            "last_durable_stage": unknown,
            "durable_checkpoint": None,
            "measured_terminal_result_present": False,
            "forward_fidelity": unknown,
            "vjp_fidelity": unknown,
            "economics": unknown,
            "scratch_cleanup": cleanup,
        }
    latest = max(rows, key=lambda row: (row["step"], row["mtime_ns"]))
    teacher_values = {row["teacher_calls"] for row in rows}
    teacher_calls: Any = next(iter(teacher_values)) if len(teacher_values) == 1 else unknown
    terminal_results = [
        row["terminal_result"]
        for row in rows
        if isinstance(row["terminal_result"], Mapping)
    ]
    n_values = {value.get("n_pairs") for value in terminal_results}
    n_pairs: Any = next(iter(n_values)) if len(n_values) == 1 else unknown
    terminal = terminal_results[-1] if terminal_results else None
    return {
        "n_pairs": n_pairs,
        "fit_steps": latest["step"],
        "teacher_calls": teacher_calls,
        "last_durable_stage": latest["stage"],
        "durable_checkpoint": {
            "path": latest["path"],
            "sha256": latest["sha256"],
        },
        "measured_terminal_result_present": terminal is not None,
        "forward_fidelity": (
            terminal.get("forward_fidelity") if terminal is not None else unknown
        ),
        "vjp_fidelity": (
            terminal.get("vjp_fidelity_decisive_full_vector")
            if terminal is not None
            else unknown
        ),
        "economics": (
            terminal.get("charged_timings") if terminal is not None else unknown
        ),
        "scratch_cleanup": cleanup,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.anchor_cadence < 1:
        raise ProbeError("--anchor-cadence must be positive")
    if args.seed < 0:
        raise ProbeError("--seed must be non-negative")
    if args.fit_epochs < 1:
        raise ProbeError("--fit-epochs must be positive")
    if args.resume_from is not None and args.mode != "fit-measure":
        raise ProbeError("--resume-from is valid only in fit-measure mode")

    preflight = _preflight(args)
    receipt_path = _write_preflight_receipt(args, preflight)
    if preflight["blockers"]:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "primary_state": preflight["primary_state"],
                    "blocker_count": len(preflight["blockers"]),
                    "blocker_surfaces": preflight["blocker_surfaces"],
                    "receipt": str(receipt_path),
                    "bulk_output_root_touched": False,
                },
                sort_keys=True,
            )
        )
        return 2
    if args.mode == "preflight":
        print(
            json.dumps(
                {
                    "status": "validated",
                    "n_pairs": REQUIRED_N,
                    "receipt": str(receipt_path),
                    "fit_executed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        result = _fit_measure(args, preflight)
    except Exception as error:
        partial = _partial_fit_observation(args)
        measured_terminal = partial["measured_terminal_result_present"] is True
        failure = {
            "schema": BLOCKER_SCHEMA,
            "probe_schema": PROBE_SCHEMA,
            "generated_at_utc": _utc_now(),
            "lane_id": LANE_ID,
            "status": "blocked",
            "primary_state": (
                "ADMISSION_BLOCKED_AFTER_MEASURED_RESULT"
                if measured_terminal
                else "MEASUREMENT_BLOCKED_AFTER_PARTIAL_EXECUTION"
            ),
            "blockers": [
                _block(
                    "fit",
                    "cache_only_fit_measure_failed",
                    f"{type(error).__name__}: {error}",
                )
            ],
            "preflight_receipt": str(receipt_path),
            "preflight_sha256": _sha256(receipt_path),
            "verdict_scope": (
                "INSTANCE x ADMISSION-BINDING x PRESERVED-MEASUREMENT"
                if measured_terminal
                else "INSTANCE x FIT-DRIVER x CURRENT-HOST"
            ),
            "req_R": (
                "repair the exact failed admission/custody seam and resume from the latest "
                "hash-verified stage checkpoint; do not rerun completed stages or relabel a "
                "preserved measured terminal result as unmeasured"
            ),
            "measurement": partial,
            "authority": preflight["authority"],
            "bulk_output_root_touched": True,
            "scratch_cleanup": partial["scratch_cleanup"],
        }
        failure_path = args.receipt_dir / "fit_blocker_receipt.json"
        _atomic_json(failure_path, failure)
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "primary_state": failure["primary_state"],
                    "receipt": str(failure_path),
                    "fit_steps": partial["fit_steps"],
                    "teacher_calls": partial["teacher_calls"],
                    "score_claim": False,
                },
                sort_keys=True,
            )
        )
        return 3
    _atomic_json(args.receipt_dir / "measurement_receipt.json", result)
    print(
        json.dumps(
            {
                "status": result["disposition"],
                "n_pairs": REQUIRED_N,
                "receipt": str(args.receipt_dir / "measurement_receipt.json"),
                "admission_prepared": result["admission"]["admission_prepared"],
                "typed_admission_state": result["admission"]["typed_admission_decision"][
                    "state"
                ],
                "score_claim": False,
                "live_trainer_activation": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
