#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable, local-only YOPO first-layer costate measurement receipt.

This is deliberately a *measurement* surface.  It never changes a trainer,
checkpoint, or scorer.  It accepts only the sealed legacy level-set snapshots
listed below and writes an atomic JSON receipt after every completed inner
step.  A missing exact renderer state or provider is a durable BLOCKED receipt,
never a synthetic substitute.

The implemented provider's pre-registered EfficientNet-B2 cut is
``conv_stem -> bn1 -> blocks[0]``.  Its provider bank owns the exact split
identity and validates the full topology before every costate use.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

REGIMES = {
    "early": "frozen_ep299_CEend.npz",
    "boundary": "frozen_ep726_MuonStart.npz",
    "late": "frozen_ep925_liveEMA.npz",
}
CHECKPOINT_DIR = REPO / "experiments/results/tau_crossover_trainflow_20260707"
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n6.npz"
SEGNET = REPO / "upstream/models/segnet.safetensors"
VIDEO = REPO / "upstream/videos/0.mkv"
VIDEO_SHA256 = "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9"
VIDEO_BYTES = 37_545_489
K_VALUES = (1, 2, 4)
SCHEMA = "yopo_first_layer_costate_probe.v2"
SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP = 4
SAME_FRAME_TEACHER_CE_PATH_FLOOR_ANCHOR = {
    "status": "MEASURED",
    "artifact": ".omx/research/yopo_same_frame_teacher_label_path_floor_measurement_20260712.json",
    "artifact_sha256": "f9cd10e263736319270f3d76387ffe78f17fc66f74bb1c322ab94f76a17d456d",
    "measurement": (
        "complete non-refresh CPU path replay: CE paths differed by 4 float32 ULP "
        "(3.725290298461914e-09 absolute); d_seg was exactly equal"
    ),
    "scope": "same-frame labeled SegNet inference path versus input-gradient path; not a costate admission threshold",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_ulp_distance(left: float, right: float) -> int | None:
    """Return the ordered float32 ULP distance, or None for nonfinite inputs."""
    left32 = np.float32(left)
    right32 = np.float32(right)
    if not np.isfinite(left32) or not np.isfinite(right32):
        return None
    if left32 == right32:
        return 0

    def ordered(value: np.float32) -> int:
        bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
        return ((~bits) & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000

    return abs(ordered(left32) - ordered(right32))


def _teacher_label_path_agreement(reference: dict[str, float], observed: dict[str, float]) -> dict[str, Any]:
    """Compare same-frame label paths against the registered numerical floor."""
    schema_ok = set(reference) == {"ce", "dseg"} and set(observed) == {"ce", "dseg"}
    ce_ulp_distance = _float32_ulp_distance(reference["ce"], observed["ce"]) if schema_ok else None
    dseg_finite = bool(
        schema_ok and np.isfinite(reference["dseg"]) and np.isfinite(observed["dseg"])
    )
    dseg_exact = bool(dseg_finite and reference["dseg"] == observed["dseg"])
    passed = bool(
        schema_ok
        and ce_ulp_distance is not None
        and ce_ulp_distance <= SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP
        and dseg_exact
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "reference": reference,
        "observed": observed,
        "ce_float32_ulp_distance": ce_ulp_distance,
        "ce_max_registered_float32_ulp_floor": SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP,
        "dseg_finite": dseg_finite,
        "dseg_exact_match": dseg_exact,
        "floor_anchor": SAME_FRAME_TEACHER_CE_PATH_FLOOR_ANCHOR,
    }


def _require_teacher_label_path_agreement(
    reference: dict[str, float], observed: dict[str, float]
) -> dict[str, Any]:
    record = _teacher_label_path_agreement(reference, observed)
    if record["status"] != "PASS":
        raise RuntimeError(f"same-frame teacher label paths exceed registered numerical floor: {record}")
    return record


def _teacher_label_path_floor_canary() -> dict[str, Any]:
    """Prove the meter admits its measured floor and rejects the next ULP."""
    start = np.float32(1.0)
    within = start
    outside = start
    for _ in range(SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP):
        within = np.nextafter(within, np.float32(np.inf), dtype=np.float32)
        outside = np.nextafter(outside, np.float32(np.inf), dtype=np.float32)
    outside = np.nextafter(outside, np.float32(np.inf), dtype=np.float32)
    positive = _teacher_label_path_agreement(
        {"ce": float(start), "dseg": 0.25}, {"ce": float(within), "dseg": 0.25}
    )
    ce_negative = _teacher_label_path_agreement(
        {"ce": float(start), "dseg": 0.25}, {"ce": float(outside), "dseg": 0.25}
    )
    dseg_negative = _teacher_label_path_agreement(
        {"ce": float(start), "dseg": 0.25}, {"ce": float(start), "dseg": 0.5}
    )
    anchor_path = REPO / SAME_FRAME_TEACHER_CE_PATH_FLOOR_ANCHOR["artifact"]
    anchor_payload = None
    anchor_error = None
    try:
        anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        anchor_error = f"{type(exc).__name__}: {exc}"
    anchor_sha256 = _sha256(anchor_path) if anchor_path.is_file() else None
    anchor_passed = bool(
        anchor_payload
        and anchor_sha256 == SAME_FRAME_TEACHER_CE_PATH_FLOOR_ANCHOR["artifact_sha256"]
        and anchor_payload.get("status") == "MEASURED"
        and anchor_payload.get("comparison", {}).get("ce_float32_ulp_distance")
        == SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP
        and anchor_payload.get("comparison", {}).get("dseg_exact_match") is True
        and anchor_payload.get("inputs", {}).get("source_video_sha256") == VIDEO_SHA256
        and anchor_payload.get("inputs", {}).get("source_video_bytes") == VIDEO_BYTES
    )
    passed = (
        positive["status"] == "PASS"
        and positive["ce_float32_ulp_distance"] == SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP
        and ce_negative["status"] == "FAIL"
        and ce_negative["ce_float32_ulp_distance"] == SAME_FRAME_TEACHER_CE_PATH_MAX_FLOAT32_ULP + 1
        and dseg_negative["status"] == "FAIL"
        and anchor_passed
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "positive_control_at_registered_floor": positive,
        "negative_control_one_ulp_beyond_floor": ce_negative,
        "negative_control_dseg_mismatch": dseg_negative,
        "measured_anchor": {
            "status": "PASS" if anchor_passed else "FAIL",
            "path": str(anchor_path),
            "sha256": anchor_sha256,
            "error": anchor_error,
        },
    }


def _source_custody() -> dict[str, dict[str, Any]]:
    sources = (
        REPO / "tools/probe_yopo_first_layer_costate.py",
        REPO / "src/tac/cuda_levelset_training.py",
        REPO / "src/tac/boundary_math/seg_core.py",
        REPO / "src/tac/boundary_math/segnet_gradient_replacement.py",
        REPO / "src/tac/local_acceleration/torch_levelset_inflate.py",
        REPO / "src/tac/witness_annulus_metrics.py",
        REPO / "tools/dash_comb_probe_n600.py",
    )
    return {str(path.relative_to(REPO)): {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in sources}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


_CLEANUP_REPRODUCIBILITY_CONTEXT = (
    "receipt.config + receipt.runtime_provenance.argv + receipt.inputs + receipt.source_custody"
)
_CLEANUP_STATUSES = {
    "CERTIFIED_REBUILDABLE_SCRATCH_PENDING_REMOVAL",
    "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED",
}


def _validate_cleanup_certificate(record: dict[str, Any], arm: dict[str, Any], scratch_path: Path) -> None:
    """Require the full machine-readable cleanup proof and active-bank binding."""
    required = {
        "status",
        "path",
        "bytes",
        "sha256",
        "rebuild",
        "reason",
        "cold_store_destination",
        "reproducibility_context",
        "false_authority",
    }
    if set(record) != required:
        raise RuntimeError("scratch cleanup certificate schema is incomplete or ambiguous")
    sha256 = record.get("sha256")
    valid_sha256 = (
        isinstance(sha256, str) and len(sha256) == 64 and all(character in "0123456789abcdef" for character in sha256)
    )
    if (
        record.get("status") not in _CLEANUP_STATUSES
        or record.get("path") != str(scratch_path)
        or not isinstance(record.get("bytes"), int)
        or isinstance(record.get("bytes"), bool)
        or int(record["bytes"]) < 0
        or not valid_sha256
        or not isinstance(record.get("rebuild"), str)
        or not record["rebuild"]
        or not isinstance(record.get("reason"), str)
        or not record["reason"]
        or record.get("cold_store_destination") is not None
        or record.get("reproducibility_context") != _CLEANUP_REPRODUCIBILITY_CONTEXT
        or record.get("false_authority") != {"score_claim": False, "promotion_eligible": False}
    ):
        raise RuntimeError("scratch cleanup certificate content is invalid")
    active = arm.get("active_bank")
    if active is not None and (
        active.get("path") != str(scratch_path)
        or active.get("sha256") != record["sha256"]
        or active.get("rebuild") != record["rebuild"]
    ):
        raise RuntimeError("scratch cleanup certificate does not match the active bank custody")


def _certify_then_remove_scratch(
    *,
    receipt_path: Path,
    receipt: dict[str, Any],
    arm: dict[str, Any],
    scratch_path: Path,
    reason: str,
) -> dict[str, Any]:
    """Durably certify rebuildable scratch before unlinking it."""
    cleanup = arm.setdefault("cleanup", [])
    candidates = [
        item for item in cleanup if item.get("path") == str(scratch_path) and item.get("status") in _CLEANUP_STATUSES
    ]
    if len(candidates) > 1:
        raise RuntimeError("multiple cleanup certificates exist for one scratch bank")
    record = candidates[0] if candidates else None
    active = arm.get("active_bank") or {}
    rebuild = active.get("rebuild")
    if record is None:
        if not scratch_path.is_file():
            raise RuntimeError("scratch bank is missing before cleanup certification")
        if not isinstance(rebuild, str) or not rebuild:
            raise RuntimeError("scratch bank lacks deterministic rebuild proof; refusing removal")
        record = {
            "status": "CERTIFIED_REBUILDABLE_SCRATCH_PENDING_REMOVAL",
            "path": str(scratch_path),
            "bytes": scratch_path.stat().st_size,
            "sha256": _sha256(scratch_path),
            "rebuild": rebuild,
            "reason": reason,
            "cold_store_destination": None,
            "reproducibility_context": _CLEANUP_REPRODUCIBILITY_CONTEXT,
            "false_authority": {"score_claim": False, "promotion_eligible": False},
        }
        _validate_cleanup_certificate(record, arm, scratch_path)
        cleanup.append(record)
        _atomic_write(receipt_path, receipt)
    else:
        _validate_cleanup_certificate(record, arm, scratch_path)
    if scratch_path.exists():
        if scratch_path.stat().st_size != int(record["bytes"]) or _sha256(scratch_path) != record["sha256"]:
            raise RuntimeError("scratch bank changed after cleanup certification; refusing removal")
        scratch_path.unlink()
    record["status"] = "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED"
    _atomic_write(receipt_path, receipt)
    return record


def _require_active_scratch_custody(arm: dict[str, Any], scratch_path: Path) -> tuple[bool, bool]:
    """Block a missing active bank unless cleanup was already certified."""
    candidates = [
        item
        for item in arm.get("cleanup", [])
        if item.get("path") == str(scratch_path) and item.get("status") in _CLEANUP_STATUSES
    ]
    if len(candidates) > 1:
        raise RuntimeError("multiple cleanup certificates exist for one scratch bank")
    for record in candidates:
        _validate_cleanup_certificate(record, arm, scratch_path)
    pending_cleanup = bool(candidates and candidates[0]["status"] == "CERTIFIED_REBUILDABLE_SCRATCH_PENDING_REMOVAL")
    removed_cleanup = bool(candidates and candidates[0]["status"] == "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED")
    if arm.get("active_bank") is not None and not scratch_path.exists() and not pending_cleanup and not removed_cleanup:
        raise RuntimeError("active scratch bank is missing without a durable cleanup certificate")
    return pending_cleanup, removed_cleanup


def _reconcile_terminal_arm_cleanup(
    *,
    receipt_path: Path,
    receipt: dict[str, Any],
    arm: dict[str, Any],
    scratch_path: Path,
) -> bool:
    """Finish certified scratch cleanup before skipping a terminal arm."""
    if arm.get("status") not in {"NO_GO_NON_DESCENT", "NO_GO_PROVIDER_FALLBACK"}:
        return False
    pending_cleanup, _removed_cleanup = _require_active_scratch_custody(arm, scratch_path)
    if scratch_path.exists() or pending_cleanup:
        _certify_then_remove_scratch(
            receipt_path=receipt_path,
            receipt=receipt,
            arm=arm,
            scratch_path=scratch_path,
            reason="adjudicated arm fell back to its already-custodied exact teacher",
        )
    arm["active_bank"] = None
    _atomic_write(receipt_path, receipt)
    return True


def _fsync_directory(directory: Path) -> None:
    """Persist the replacement directory entry, not only the replacement file."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_custody(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        fields = (
            "__epoch",
            "__cfg_softmax_temp",
            "__cfg_curriculum",
            "__cfg_self_orient",
            "__cfg_n_hidden",
            "__cfg_hidden_dim",
            "__cfg_in_feat",
            "__cfg_chroma",
        )
        result = {
            key.removeprefix("__"): archive[key].item() for key in fields if key in archive and archive[key].size == 1
        }
        result["keys"] = sorted(archive.files)
    return result


def _objective_metadata(regime: str, checkpoint: Path) -> dict[str, Any]:
    metadata = _checkpoint_metadata(checkpoint)
    return {
        "loss_name": "frozen_segnet_cross_entropy_gt_n6_lstars_pair0",
        "common_objective_across_regimes": True,
        "pair_index": 0,
        "regime": regime,
        "checkpoint_epoch": metadata.get("epoch"),
        "checkpoint_softmax_temp": metadata.get("cfg_softmax_temp"),
        "checkpoint_curriculum": metadata.get("cfg_curriculum"),
        "target": "gt_n6.npz:lstars[0]",
        "receiver": "tac.cuda_levelset_training.contest_r",
        "scorer_split": "encoder.model.conv_stem -> encoder.model.bn1 -> encoder.model.blocks[0]",
        "costate_shape": [1, 16, 192, 256],
    }


def _peak_rss_bytes() -> int:
    # Darwin ru_maxrss is bytes; Linux reports KiB. This environment is macOS,
    # but retaining the platform branch makes the receipt portable.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _metrics(reference: Any, candidate: Any, *, mask: Any | None = None) -> dict[str, Any]:
    """Use the canonical NumPy costate metrics without importing a live trainer."""
    from tac.boundary_math.segnet_gradient_replacement import measure_costate_agreement

    return measure_costate_agreement(reference, candidate, mask=mask).to_dict()


def _cosine(a: Any, b: Any) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if not np.isfinite(denom) or denom == 0.0:
        return None
    return float(np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))


def _require_complete_horizon(inner_steps: int) -> None:
    if inner_steps < max(K_VALUES):
        raise ValueError(
            f"inner_steps must be >= {max(K_VALUES)} so K={max(K_VALUES)} reaches its pre-registered age horizon"
        )


def _provider_or_full_teacher_fallback(
    *,
    provider: Any,
    provider_kwargs: dict[str, Any] | None,
    exact_costate: Any,
    preflight_failure: str | None = None,
) -> tuple[Any, dict[str, Any], float, bool]:
    """Invoke the provider once and make every provider rejection an exact fallback."""
    if preflight_failure is not None or provider_kwargs is None:
        return (
            exact_costate,
            {
                "selected_mode": "full_teacher_fallback",
                "provider_failure": preflight_failure or "RuntimeError: provider arguments are unavailable",
            },
            0.0,
            True,
        )
    started = time.perf_counter()
    try:
        candidate, metadata = provider(**provider_kwargs)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return (
            exact_costate,
            {
                "selected_mode": "full_teacher_fallback",
                "provider_failure": f"{type(exc).__name__}: {exc}",
            },
            time.perf_counter() - started,
            True,
        )
    return candidate, metadata, time.perf_counter() - started, False


def _operational_timing_record(
    *,
    step_started: float,
    operational_completed: float,
    component_seconds: dict[str, float],
    path: str,
) -> dict[str, Any]:
    """Keep whole-step wall time authoritative and component sums diagnostic."""
    wall_seconds = operational_completed - step_started
    if not np.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise RuntimeError("operational wall interval is negative or nonfinite")
    if any(not np.isfinite(value) or value < 0.0 for value in component_seconds.values()):
        raise RuntimeError("operational timing component is negative or nonfinite")
    component_sum = float(sum(component_seconds.values()))
    return {
        "path": path,
        "wall_seconds": wall_seconds,
        "component_sum_seconds_diagnostic_only": component_sum,
        "unattributed_overhead_seconds": wall_seconds - component_sum,
        "authority": "wall interval; component sum is diagnostic only",
    }


def _arm_ready_for_success_finalization(arm: dict[str, Any], *, expected_steps: int) -> bool:
    """Exclude terminal/fallback rows from the successful-arm finalizer."""
    steps = arm.get("steps", [])
    return bool(
        arm.get("status") == "RUNNING"
        and len(steps) == expected_steps
        and all(
            step.get("status") == "MEASURED"
            and step.get("candidate_non_descent") is False
            and step.get("provider_fallback") is False
            and isinstance(step.get("controls"), dict)
            for step in steps
        )
    )


def _admission(regime_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """No invented cosine threshold: descent and matching measured K knees decide."""
    knees: dict[str, int] = {}
    blockers: list[str] = []
    falsified_cadences: dict[str, list[int]] = {}
    falsifier_reasons: dict[str, list[str]] = {}
    for regime in REGIMES:
        row = regime_rows.get(regime)
        if not row or row.get("status") != "MEASURED":
            blockers.append(f"{regime}: missing measured regime receipt")
            continue
        observed_k = {int(arm["K"]) for arm in row.get("arms", [])}
        if observed_k != set(K_VALUES):
            blockers.append(f"{regime}: measured K coverage {sorted(observed_k)} != {list(K_VALUES)}")
            continue
        for arm in row.get("arms", []):
            k_value = int(arm["K"])
            steps = arm.get("steps", [])
            if arm.get("status") in {"NO_GO_NON_DESCENT", "NO_GO_PROVIDER_FALLBACK"}:
                falsified_cadences.setdefault(regime, []).append(k_value)
                falsifier_reasons.setdefault(regime, []).append(f"K={k_value} terminated as {arm['status']}")
                continue
            if any(step.get("candidate_non_descent") is True for step in steps):
                falsified_cadences.setdefault(regime, []).append(k_value)
                falsifier_reasons.setdefault(regime, []).append(f"K={k_value} recorded non-descent")
                continue
            if any(step.get("provider_fallback") is True for step in steps):
                falsified_cadences.setdefault(regime, []).append(k_value)
                falsifier_reasons.setdefault(regime, []).append(f"K={k_value} recorded full-teacher fallback")
                continue
            if arm.get("status") != "MEASURED":
                blockers.append(f"{regime}: K={k_value} is not a completed MEASURED arm")
                continue
            if arm.get("controls", {}).get("status") != "PASS":
                blockers.append(f"{regime}: K={k_value} measurement controls did not pass")
                continue
            if arm.get("controls", {}).get("full_age_horizon_exercised") is not True:
                blockers.append(f"{regime}: K={k_value} did not reach bank age K-1")
        controls = row.get("controls", {})
        if controls.get("status") != "PASS":
            blockers.append(f"{regime}: positive/negative measurement controls did not pass")
            continue
        pareto = row.get("pareto", {})
        if not pareto.get("pareto_nondominated_K") or pareto.get("pareto_knee_K") is None:
            reasons = pareto.get("reason") or ["no measured non-dominated speed/regret arm"]
            if isinstance(reasons, str):
                reasons = [reasons]
            detail = "; ".join(reasons)
            blockers.append(f"{regime}: {detail}")
            continue
        knees[regime] = int(pareto["pareto_knee_K"])
    fully_falsified_regimes = [regime for regime in REGIMES if set(falsified_cadences.get(regime, [])) == set(K_VALUES)]
    if fully_falsified_regimes:
        return {
            "status": "NO-GO",
            "reason": [
                f"{regime}: every registered cadence recorded non-descent or provider fallback"
                for regime in fully_falsified_regimes
            ],
            "pareto_knee_by_regime": knees,
            "falsified_cadences_by_regime": falsified_cadences,
            "verdict_scope": "the measured first-block split and K={1,2,4} cadence set only",
        }
    if blockers:
        return {
            "status": "NEEDS-MORE",
            "reason": blockers,
            "pareto_knee_by_regime": knees,
            "falsified_cadences_by_regime": falsified_cadences,
            "verdict_scope": "n=1 pair0 saved-regime replay; no family-level verdict",
        }
    if len(set(knees.values())) != 1:
        return {
            "status": "NO-GO",
            "reason": ["Pareto knee differs across regimes"],
            "pareto_knee_by_regime": knees,
            "falsified_cadences_by_regime": falsified_cadences,
            "verdict_scope": "the measured cadence-invariance formulation only",
        }
    selected_k = next(iter(knees.values()))
    if selected_k <= 1:
        return {
            "status": "NO-GO",
            "reason": ["the shared measured Pareto knee is K=1, which performs no deep-costate reuse"],
            "pareto_knee_by_regime": knees,
            "selected_K": selected_k,
            "falsified_cadences_by_regime": falsified_cadences,
            "verdict_scope": "the measured first-block split and K={1,2,4} cadence set only",
        }
    selected_failures = [
        f"{regime}: {reason}"
        for regime, reasons in falsifier_reasons.items()
        for reason in reasons
        if reason.startswith(f"K={selected_k} ")
    ]
    if selected_failures:
        return {
            "status": "NO-GO",
            "reason": selected_failures,
            "pareto_knee_by_regime": knees,
            "selected_K": selected_k,
            "falsified_cadences_by_regime": falsified_cadences,
            "verdict_scope": "the selected cadence at the measured first-block split only",
        }
    speeds: dict[str, float] = {}
    speed_lower_bounds: dict[str, float] = {}
    for regime, row in regime_rows.items():
        pareto_row = next(item for item in row["pareto"]["rows"] if int(item["K"]) == selected_k)
        speeds[regime] = float(pareto_row["measured_operational_speedup_vs_K1"])
        speed_lower_bounds[regime] = float(pareto_row["conservative_speedup_lower_bound_vs_K1"])
    if any(speed <= 1.0 for speed in speed_lower_bounds.values()):
        return {
            "status": "NO-GO",
            "reason": [
                "the shared K>1 knee does not beat K=1 operational time above the measured timing floor in every regime"
            ],
            "pareto_knee_by_regime": knees,
            "selected_K": selected_k,
            "measured_operational_speedup_vs_K1_by_regime": speeds,
            "conservative_speedup_lower_bound_vs_K1_by_regime": speed_lower_bounds,
            "falsified_cadences_by_regime": falsified_cadences,
            "verdict_scope": "the measured operational component path and within-run timing floor only",
        }
    return {
        "status": "GO",
        "reason": [
            "The selected cadence has no registered non-descent/fallback, has a matching regime knee, "
            "and beats K=1 above the measured timing floor"
        ],
        "pareto_knee_by_regime": knees,
        "selected_K": selected_k,
        "measured_operational_speedup_vs_K1_by_regime": speeds,
        "conservative_speedup_lower_bound_vs_K1_by_regime": speed_lower_bounds,
        "falsified_cadences_by_regime": falsified_cadences,
        "verdict_scope": "n=1 pair0 early/boundary/late saved-regime replay only",
    }


def _pareto_knee(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a dimensionless knee on the measured non-dominated front.

    Dominance is measured, not guessed: higher operational-cycle speedup is better;
    both CE and d_seg regret versus the exact-reference recess are lower-is-better.
    A non-descent arm is excluded before the comparison.  The knee minimizes
    normalized Euclidean distance to the observed utopia; ties choose smaller K.
    """
    completed = [arm for arm in arms if arm.get("status") == "MEASURED"]
    if not completed:
        return {"status": "NEEDS_MORE", "reason": "no completed measured arms"}
    control = next((arm for arm in completed if int(arm["K"]) == 1), None)
    if control is None:
        return {"status": "NEEDS_MORE", "reason": "missing K=1 exact control"}
    control_steps = control.get("steps", [])
    if (
        not control_steps
        or control.get("controls", {}).get("status") != "PASS"
        or any(step.get("candidate_non_descent") is not False for step in control_steps)
    ):
        return {
            "status": "NEEDS_MORE",
            "reason": "K=1 exact measurement control is invalid or contains non-descent",
        }
    clock_floor = 2.0 * time.get_clock_info("perf_counter").resolution

    def timing_support(arm: dict[str, Any]) -> tuple[float, float, float]:
        values = [float(step["timing_measured_seconds"]["operational_cycle"]) for step in arm["steps"]]
        mean = float(np.mean(values))
        lower = max(min(values) - clock_floor, 0.0)
        upper = max(values) + clock_floor
        return mean, lower, upper

    control_time, control_lower, control_upper = timing_support(control)
    candidates: list[dict[str, Any]] = []
    excluded_non_descent: list[int] = []
    excluded_underexercised: list[int] = []
    excluded_control_failure: list[int] = []
    for arm in completed:
        steps = arm.get("steps", [])
        k_value = int(arm["K"])
        if not steps or max(int(step.get("bank_age_steps", -1)) for step in steps) < k_value - 1:
            excluded_underexercised.append(k_value)
            continue
        if arm.get("controls", {}).get("status") != "PASS":
            excluded_control_failure.append(k_value)
            continue
        if not steps or any(step.get("candidate_non_descent") is not False for step in steps):
            excluded_non_descent.append(k_value)
            continue
        summary = arm["summary"]
        arm_time, arm_lower, arm_upper = timing_support(arm)
        speedup = control_time / arm_time
        conservative_speedup_lower_bound = control_lower / arm_upper
        candidates.append(
            {
                "K": int(arm["K"]),
                "measured_operational_speedup_vs_K1": speedup,
                "conservative_speedup_lower_bound_vs_K1": conservative_speedup_lower_bound,
                "K1_observed_timing_support_seconds": [control_lower, control_upper],
                "arm_observed_timing_support_seconds": [arm_lower, arm_upper],
                "timing_noise_rule": (
                    "observed extrema expanded by two perf-counter ticks; "
                    "conservative lower bound is K1 lower / arm upper; across-seed variance UNKNOWN"
                ),
                "mean_ce_regret": float(summary["mean_ce_regret_vs_exact_reference"]),
                "mean_dseg_regret": float(summary["mean_dseg_regret_vs_exact_reference"]),
            }
        )
    front: list[dict[str, Any]] = []
    for row in candidates:
        dominated = any(
            other is not row
            and other["conservative_speedup_lower_bound_vs_K1"] >= row["conservative_speedup_lower_bound_vs_K1"]
            and other["mean_ce_regret"] <= row["mean_ce_regret"]
            and other["mean_dseg_regret"] <= row["mean_dseg_regret"]
            and (
                other["conservative_speedup_lower_bound_vs_K1"] > row["conservative_speedup_lower_bound_vs_K1"]
                or other["mean_ce_regret"] < row["mean_ce_regret"]
                or other["mean_dseg_regret"] < row["mean_dseg_regret"]
            )
            for other in candidates
        )
        if not dominated:
            front.append(row)
    speed_values = [row["conservative_speedup_lower_bound_vs_K1"] for row in front]
    ce_values = [row["mean_ce_regret"] for row in front]
    dseg_values = [row["mean_dseg_regret"] for row in front]

    def normalize(value: float, values: list[float], *, higher_is_better: bool) -> float:
        lo, hi = min(values), max(values)
        if hi == lo:
            return 0.0
        return ((hi - value) if higher_is_better else (value - lo)) / (hi - lo)

    for row in front:
        row["utopia_normalized_distance"] = float(
            np.sqrt(
                normalize(row["conservative_speedup_lower_bound_vs_K1"], speed_values, higher_is_better=True) ** 2
                + normalize(row["mean_ce_regret"], ce_values, higher_is_better=False) ** 2
                + normalize(row["mean_dseg_regret"], dseg_values, higher_is_better=False) ** 2
            )
        )
    knee = min(front, key=lambda row: (row["utopia_normalized_distance"], row["K"])) if front else None
    return {
        "status": "MEASURED" if front else "NEEDS_MORE",
        "reason": ([] if front else ["every completed arm has a non-descent or incomplete step"]),
        "excluded_non_descent_K": excluded_non_descent,
        "excluded_underexercised_K": excluded_underexercised,
        "excluded_control_failure_K": excluded_control_failure,
        "dimensions": [
            "conservative_speedup_lower_bound_vs_K1 higher",
            "mean_ce_regret lower",
            "mean_dseg_regret lower",
        ],
        "pareto_nondominated_K": [row["K"] for row in front],
        "rows": front,
        "pareto_knee_K": knee["K"] if knee else None,
        "rule": "minimum normalized Euclidean distance to measured utopia; equal distance selects smaller K",
    }


def _base_receipt(args: argparse.Namespace) -> dict[str, Any]:
    _require_complete_horizon(args.steps)
    checkpoints = {name: CHECKPOINT_DIR / filename for name, filename in REGIMES.items()}
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": {
            "axis": "[macOS-CPU advisory; local-only; torch-fp32]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "verdict_scope": "n=1 pair0 saved-regime costate measurement only",
            "review_status": "pre-registered-only",
            "noise_floor": {
                "within_run_timing": "MEASURED per arm and regime",
                "across_seed_variance": "UNKNOWN; single-seed spine",
                "cross_hardware_transfer": "UNKNOWN; macOS CPU advisory axis only",
            },
            "peak_memory_authority": (
                "MEASURED validation-harness process RSS high-water; operational-only peak is UNKNOWN"
            ),
        },
        "config": {
            "pair_index": 0,
            "K_values": list(K_VALUES),
            "inner_steps": args.steps,
            "seed": args.seed,
            "step_control": (
                "event-conditioned candidate-teacher backtracking: fraction=1e-2, anneal=0.5, "
                "accept CE decrease with d_seg non-worsening, terminate at fp32 bit identity"
            ),
            "falsifier": (
                "NO-GO for a cadence on any provider fallback, candidate non-descent, failed measurement canary, "
                "underexercised K-1 age horizon, non-surviving early/boundary/late knee, or measured whole-step "
                "operational speedup lower bound <=1 against K=1"
            ),
            "economics_definition": (
                "MEASURED whole wall interval from step start through operational candidate validation, including "
                "renderer, live custody, control-flow, exact refresh/bank write or current-frame prefix VJP, renderer "
                "pullback, and all candidate CE/d_seg validation trials including labels. Component sums are diagnostic "
                "only. Fresh non-refresh teacher F/B, exact-reference validation, reverse-control validation, and their "
                "fresh diagnostic re-render are measurement-only and separately timed."
            ),
        },
        "runtime_provenance": {
            "argv": list(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "deterministic_algorithms": True,
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            ).stdout.strip()
            or None,
        },
        "inputs": {
            "video": _file_custody(VIDEO),
            "segnet": _file_custody(SEGNET),
            "gt_cache": _file_custody(GT_CACHE),
            "checkpoints": {name: _file_custody(path) for name, path in checkpoints.items()},
        },
        "source_custody": _source_custody(),
        "split": {
            "name": "efficientnet_b2_block0_provider_cut",
            "prefix": ["encoder.model.conv_stem", "encoder.model.bn1", "encoder.model.blocks[0]"],
            "output_shape": [1, 16, 192, 256],
            "p1_bytes_fp32": 3_145_728,
            "provider": "tac.boundary_math.segnet_gradient_replacement.YopoFirstLayerBank",
        },
        "objective": "common frozen SegNet CE against gt_n6 lstars[0]; checkpoint stage/tau bound per regime",
        "measurement_canaries": {
            "same_frame_teacher_label_path_float32_floor": _teacher_label_path_floor_canary(),
        },
        "regimes": {},
        "admission": {"status": "PENDING"},
    }


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Atomically persist small resumability state under the caller output tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _terminal_status_for_step(step_row: dict[str, Any]) -> str | None:
    """Use one deterministic terminal precedence in live and resumed paths."""
    if step_row.get("provider_fallback") is True:
        return "NO_GO_PROVIDER_FALLBACK"
    if step_row.get("candidate_non_descent") is True:
        return "NO_GO_NON_DESCENT"
    return None


def _recover_pending_step(arm: dict[str, Any], state_path: Path) -> tuple[np.ndarray, bool]:
    """Recover an atomically staged inner-step before continuing a resumed arm.

    The small state file is written first with the entire JSON step record.  If a
    process stops before the receipt replacement, this function promotes that
    staged record exactly once.  It refuses every other receipt/state divergence.
    """
    with np.load(state_path, allow_pickle=False) as state:
        theta = np.asarray(state["theta"], np.float32)
        next_step = int(state["next_step"].item())
        pending_raw = str(state["pending_step_json"].item()) if "pending_step_json" in state.files else ""
        active_bank_raw = str(state["active_bank_json"].item()) if "active_bank_json" in state.files else ""
    staged_active_bank = json.loads(active_bank_raw) if active_bank_raw else arm.get("active_bank")
    steps = arm["steps"]
    if pending_raw:
        pending = json.loads(pending_raw)
        if next_step == len(steps) + 1 and int(pending["step"]) == len(steps):
            steps.append(pending)
            arm["active_bank"] = staged_active_bank
            terminal_status = _terminal_status_for_step(pending)
            if terminal_status is not None:
                arm["status"] = terminal_status
            return theta, True
        if next_step == len(steps) and steps and pending == steps[-1] and staged_active_bank == arm.get("active_bank"):
            terminal_status = _terminal_status_for_step(pending)
            if terminal_status is not None:
                arm["status"] = terminal_status
            return theta, False
        raise RuntimeError("state pending-step record does not match receipt")
    if next_step != len(steps) or staged_active_bank != arm.get("active_bank"):
        raise RuntimeError("state/receipt step or active-bank metadata mismatch")
    return theta, False


def _stage_step_state(
    state_path: Path,
    theta: Any,
    next_step: int,
    step_row: dict[str, Any],
    active_bank: dict[str, Any] | None,
) -> None:
    _atomic_savez(
        state_path,
        theta=np.asarray(theta, np.float32),
        next_step=np.asarray(next_step),
        pending_step_json=np.asarray(json.dumps(step_row, sort_keys=True, separators=(",", ":"))),
        active_bank_json=np.asarray(json.dumps(active_bank, sort_keys=True, separators=(",", ":"))),
    )


def _clear_staged_step_state(state_path: Path, theta: Any, next_step: int, active_bank: dict[str, Any] | None) -> None:
    _atomic_savez(
        state_path,
        theta=np.asarray(theta, np.float32),
        next_step=np.asarray(next_step),
        pending_step_json=np.asarray(""),
        active_bank_json=np.asarray(json.dumps(active_bank, sort_keys=True, separators=(",", ":"))),
    )


def _acquire_output_lock(output_dir: Path) -> int:
    """Fail closed on a second writer; receipts are single-writer transactions."""
    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".probe_single_writer.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"output directory already has an active probe writer: {output_dir}") from exc
    return fd


def _release_output_lock(fd: int) -> None:
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def _import_dash_renderer() -> Any:
    """Load the settled legacy renderer rather than recreate its fixed point."""
    path = REPO / "tools/dash_comb_probe_n600.py"
    spec = importlib.util.spec_from_file_location("_settled_dash_comb_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load settled renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_renderer(checkpoint: Path) -> tuple[Any, np.ndarray, dict[str, Any], Any]:
    """Instantiate the settled Renderer directly from one sealed snapshot."""
    dash = _import_dash_renderer()
    with np.load(checkpoint, allow_pickle=False) as archive:
        params = {key: np.asarray(archive[key], np.float32) for key in archive.files if not key.startswith("__")}
        cfg = {key: archive[key].item() for key in archive.files if key.startswith("__") and archive[key].size == 1}
    code = params.pop("code")
    model = {
        "render_h": 384,
        "render_w": 512,
        "n_pairs": code.shape[0] // 2,
        "bank_n_scales": int(cfg["__bank_n_scales"]),
        "bank_n_orient0": int(cfg["__bank_n_orient0"]),
        "bank_f0": float(cfg["__bank_f0"]),
        "bank_base": float(cfg["__bank_base"]),
        "bank_n_iso": int(cfg["__bank_n_iso"]),
        "max_bank_freq": float(cfg["__cfg_max_bank_freq"]),
        "self_orient": bool(int(cfg["__cfg_self_orient"])),
        "n_dir_freqs": int(cfg["__cfg_n_dir_freqs"]),
        "so_iters": 4,
        "so_freq_along": float(cfg["__cfg_freq_along"]),
        "so_freq_across": float(cfg["__cfg_freq_across"]),
        "so_tau": 4.0,
        "activation": str(cfg["__cfg_activation"]),
        "wire_w0": float(cfg["__cfg_wire_w0"]),
        "wire_s0": float(cfg["__cfg_wire_s0"]),
        "hosc_beta": float(cfg["__cfg_hosc_beta"]),
        "hosc_omega": float(cfg["__cfg_hosc_omega"]),
        "n_hidden": int(cfg["__cfg_n_hidden"]),
        "hidden_dim": int(cfg["__cfg_hidden_dim"]),
        "softmax_temp": float(cfg["__cfg_softmax_temp"]),
        "chroma": bool(int(cfg["__cfg_chroma"])),
    }
    return dash.Renderer(params, code, model), code, model, dash


def _render_chart(renderer: Any, theta: Any) -> Any:
    """Exact conditional code-row chart through the settled detached orientation branch.

    The state-dependent orientation is deliberately computed from ``theta.detach()``
    with ``Renderer._self_orient_native``.  The following renderer equations retain
    autograd to the actual last-frame code row and terminate in canonical contest_r.
    """
    import torch

    from tac.cuda_levelset_training import contest_r
    from tac.local_acceleration import torch_levelset_inflate as tli
    from tac.local_acceleration.torch_levelset_inflate import _torch_act

    renderer.code[1] = theta.detach()
    feats_np = renderer._self_orient_native(0) if renderer.m["self_orient"] else renderer.curv_n
    feats = torch.as_tensor(feats_np, dtype=torch.float32)
    m, p = renderer.m, renderer.P
    h = tli.torch_in_proj_h0(p, feats, m)
    film = (theta @ p["film.weight"].T + p["film.bias"]).reshape(renderer.nH, 2, renderer.hd)
    kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
    for li in range(renderer.nH):
        h = _torch_act(
            (h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * (1.0 + film[li, 0]) + film[li, 1],
            *kw,
        )
    phi = h @ p["out_sdf.weight"].T + p["out_sdf.bias"]
    tex = h @ p["out_tex.weight"].T + p["out_tex.bias"]
    logits = phi / float(m["softmax_temp"])
    logits = logits - logits.max(-1, keepdim=True).values
    soft = torch.exp(logits)
    soft = soft / soft.sum(-1, keepdim=True)
    rgb = torch.sigmoid(soft @ p["palette"] + tex) * 255.0
    if not m["chroma"]:
        luma = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        rgb = torch.cat((luma, luma, luma), dim=-1)
    return contest_r(rgb.reshape(1, m["render_h"], m["render_w"], 3))


def _renderer_parity_canary(renderer: Any, theta: Any) -> dict[str, Any]:
    """Confirm the chart forward is byte-close to the settled Renderer at start."""
    import torch

    from tac.cuda_levelset_training import contest_r

    renderer.code[1] = theta.detach()
    expected_bulk, _ = renderer.render_pair(0)
    expected = contest_r(torch.as_tensor(expected_bulk, dtype=torch.float32).unsqueeze(0))
    actual = _render_chart(renderer, theta)
    diff = (actual.detach() - expected).abs()
    return {
        "status": "MEASURED",
        "max_abs": float(diff.max().item()),
        "different_elements": int((diff != 0).sum().item()),
        "theta_requires_grad": bool(theta.requires_grad),
        "render_requires_grad": bool(actual.requires_grad),
        "settled_renderer": "tools/dash_comb_probe_n600.py::Renderer",
        "receiver": "tac.cuda_levelset_training.contest_r",
    }


def _live_decision_custody(regime: str) -> dict[str, Any]:
    source = _source_custody()
    return {
        "gt_sha256": _sha256(GT_CACHE),
        "segnet_sha256": _sha256(SEGNET),
        "checkpoint_sha256": _sha256(CHECKPOINT_DIR / REGIMES[regime]),
        "source_sha256": {name: row["sha256"] for name, row in source.items()},
    }


def _objective_fingerprint(
    metadata: dict[str, Any], receipt: dict[str, Any], live_custody: dict[str, Any] | None = None
) -> str:
    live = live_custody or _live_decision_custody(str(metadata["regime"]))
    payload = {
        "objective_metadata": metadata,
        "live_custody": live,
        "expected_gt_sha256": receipt["inputs"]["gt_cache"]["sha256"],
        "expected_segnet_sha256": receipt["inputs"]["segnet"]["sha256"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _decision_custody_changed(receipt: dict[str, Any], regime: str, live: dict[str, Any]) -> bool:
    expected = {
        "gt_sha256": receipt["inputs"]["gt_cache"]["sha256"],
        "segnet_sha256": receipt["inputs"]["segnet"]["sha256"],
        "checkpoint_sha256": receipt["inputs"]["checkpoints"][regime]["sha256"],
        "source_sha256": {name: row["sha256"] for name, row in receipt["source_custody"].items()},
    }
    return live != expected


def _evaluate_teacher(segnet: Any, frame_nhwc: Any, labels: Any) -> tuple[float, float]:
    """One labeled full scorer forward used for exact reference/candidate validation."""
    import torch.nn.functional as F

    logits = segnet(frame_nhwc.permute(0, 3, 1, 2).contiguous())
    loss = F.cross_entropy(logits, labels)
    dseg = (logits.argmax(1) != labels).float().mean()
    return float(loss.detach().item()), float(dseg.detach().item())


def _capture_labeled_teacher(
    *,
    capture: Any,
    segnet: Any,
    frame_nchw: Any,
    labels: Any,
    objective_context: str,
    scorer_fingerprint: str,
    step_index: int,
) -> tuple[Any, Any, dict[str, float], float]:
    """Time the fresh costate together with its exact CE/d_seg label work."""
    import torch.nn.functional as F

    holder: dict[str, float] = {}

    def teacher_loss_fn(logits: Any) -> Any:
        loss = F.cross_entropy(logits, labels)
        holder["ce"] = float(loss.detach().item())
        holder["dseg"] = float((logits.argmax(1) != labels).float().mean().detach().item())
        return loss

    started = time.perf_counter()
    bank, exact_costate = capture(
        segnet=segnet,
        anchor_frame=frame_nchw,
        teacher_loss_fn=teacher_loss_fn,
        objective_context_fingerprint=objective_context,
        scorer_fingerprint=scorer_fingerprint,
        evaluated_at_step=step_index,
    )
    elapsed = time.perf_counter() - started
    if set(holder) != {"ce", "dseg"}:
        raise RuntimeError("fresh teacher capture did not execute the bound CE/d_seg label work")
    return bank, exact_costate, holder, elapsed


def _capture_exact_teacher_costate(*, segnet: Any, frame_nchw: Any, labels: Any) -> tuple[Any, dict[str, float], float]:
    """Measure the ordinary full-teacher baseline without YOPO bank overhead."""
    import torch
    import torch.nn.functional as F

    frame = frame_nchw.detach().requires_grad_(True)
    started = time.perf_counter()
    logits = segnet(frame)
    loss = F.cross_entropy(logits, labels)
    exact_costate = torch.autograd.grad(loss, frame)[0].detach()
    elapsed = time.perf_counter() - started
    holder = {
        "ce": float(loss.detach().item()),
        "dseg": float((logits.argmax(1) != labels).float().mean().detach().item()),
    }
    if not bool(torch.isfinite(exact_costate).all()):
        raise RuntimeError("ordinary exact teacher produced a nonfinite input costate")
    return exact_costate, holder, elapsed


def _select_candidate_recess(
    *,
    renderer: Any,
    theta: Any,
    candidate_grad: Any,
    segnet: Any,
    labels: Any,
    current_loss: float,
    current_dseg: float,
) -> tuple[Any | None, Any | None, list[dict[str, Any]], int, float]:
    """Choose an event-conditioned candidate step with a terminating anneal."""
    import torch

    theta_norm = float(torch.linalg.vector_norm(theta.detach()).item())
    grad_norm = float(torch.linalg.vector_norm(candidate_grad.detach()).item())
    base_norm = max(theta_norm, 1.0)
    # Control law: fraction starts at the pre-registered 1e-2 default and halves
    # until the candidate decreases exact teacher CE without worsening d_seg.
    # The fp32 bit-identical predicate guarantees completion.
    fraction, anneal = 1e-2, 0.5
    trials: list[dict[str, Any]] = []
    if not np.isfinite(grad_norm) or grad_norm == 0.0:
        return (
            None,
            None,
            [{"status": "BLOCKED", "reason": "zero/nonfinite candidate renderer gradient"}],
            0,
            0.0,
        )
    validation_started = time.perf_counter()
    while True:
        step_norm = fraction * base_norm
        candidate = theta.detach() - (step_norm / grad_norm) * candidate_grad.detach()
        if torch.equal(candidate, theta.detach()):
            trials.append(
                {
                    "fraction_of_max_theta_norm": fraction,
                    "target_parameter_step_norm": step_norm,
                    "status": "BIT_IDENTICAL_TERMINATION",
                    "accepted": False,
                }
            )
            return None, None, trials, len(trials), time.perf_counter() - validation_started
        with torch.inference_mode():
            loss, dseg = _evaluate_teacher(segnet, _render_chart(renderer, candidate), labels)
        accepted = bool(loss < current_loss and dseg <= current_dseg)
        trials.append(
            {
                "fraction_of_max_theta_norm": fraction,
                "target_parameter_step_norm": step_norm,
                "candidate_ce": loss,
                "candidate_dseg": dseg,
                "current_ce": current_loss,
                "current_dseg": current_dseg,
                "accepted": accepted,
            }
        )
        if accepted:
            return candidate, step_norm, trials, len(trials), time.perf_counter() - validation_started
        fraction *= anneal


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_resume_custody(receipt: dict[str, Any], template: dict[str, Any]) -> None:
    """Reject changed artifacts or source code before any resume mutation."""
    for key in (
        "schema",
        "authority",
        "config",
        "inputs",
        "source_custody",
        "split",
        "objective",
        "measurement_canaries",
    ):
        if receipt.get(key) != template.get(key):
            raise RuntimeError(f"resume immutable field {key!r} differs from the current probe")
    stable_runtime_keys = (
        "python",
        "platform",
        "machine",
        "deterministic_algorithms",
        "git_head",
        "numpy",
        "torch",
        "torch_num_threads",
    )
    receipt_runtime = receipt.get("runtime_provenance", {})
    template_runtime = template.get("runtime_provenance", {})
    if any(receipt_runtime.get(key) != template_runtime.get(key) for key in stable_runtime_keys):
        raise RuntimeError("resume runtime provenance differs from the current probe")
    if receipt.get("completed_at_utc"):
        raise RuntimeError("receipt is terminal; choose a fresh content-bound output directory")


def _validate_regime_resume_custody(
    row: dict[str, Any], *, checkpoint_metadata: dict[str, Any], objective_metadata: dict[str, Any]
) -> None:
    """Re-derive saved-regime objective inputs before any decision fingerprint."""
    if row.get("checkpoint_metadata") != checkpoint_metadata:
        raise RuntimeError("resume checkpoint metadata differs from the current checkpoint bytes")
    if row.get("objective_metadata") != objective_metadata:
        raise RuntimeError("resume objective metadata differs from the current checkpoint-derived objective")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Measure all sealed regimes through the exact conditional renderer/R path."""
    import torch

    from tac.boundary_math.seg_core import load_real_segnet
    from tac.boundary_math.segnet_gradient_replacement import (
        array_content_sha256,
        capture_yopo_first_layer_bank,
        write_yopo_first_layer_bank,
        yopo_first_layer_costate_torch,
        yopo_first_layer_split_identity,
    )
    from tac.witness_annulus_metrics import annulus_mask_bottom_k

    _require_complete_horizon(args.steps)
    permitted_root = (REPO / "experiments" / "results").resolve()
    try:
        args.output_dir.resolve().relative_to(permitted_root)
    except ValueError as exc:
        raise RuntimeError("output directory must stay under experiments/results") from exc
    if VIDEO.stat().st_size != VIDEO_BYTES or _sha256(VIDEO) != VIDEO_SHA256:
        raise RuntimeError("source video custody mismatch; refusing measurement")
    lock_fd = _acquire_output_lock(args.output_dir)
    atexit.register(_release_output_lock, lock_fd)
    receipt_path = args.output_dir / "receipt.json"
    if receipt_path.exists() and not args.resume:
        _release_output_lock(lock_fd)
        raise RuntimeError("output receipt already exists; choose a fresh content-bound output directory")
    receipt = _load_existing(receipt_path) if args.resume else None
    template = _base_receipt(args)
    if template["measurement_canaries"]["same_frame_teacher_label_path_float32_floor"]["status"] != "PASS":
        _release_output_lock(lock_fd)
        raise RuntimeError("same-frame teacher label-path floor canary failed before measurement")
    template["runtime_provenance"].update(
        {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        }
    )
    if receipt is None:
        receipt = template
    try:
        _validate_resume_custody(receipt, template)
    except RuntimeError:
        _release_output_lock(lock_fd)
        raise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    segnet = load_real_segnet("cpu").eval()
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)
    with np.load(GT_CACHE, allow_pickle=False) as gt:
        labels = torch.as_tensor(np.asarray(gt["lstars"][0], np.int64))[None]
        margin = np.asarray(gt["margins"][0], np.float32)
    annulus = annulus_mask_bottom_k(margin, k=0.05)[None, None]
    split_identity = yopo_first_layer_split_identity(segnet)

    for regime, filename in REGIMES.items():
        checkpoint = CHECKPOINT_DIR / filename
        checkpoint_metadata = _checkpoint_metadata(checkpoint)
        objective_metadata = _objective_metadata(regime, checkpoint)
        existing_row = receipt["regimes"].get(regime)
        if existing_row is not None:
            _validate_regime_resume_custody(
                existing_row,
                checkpoint_metadata=checkpoint_metadata,
                objective_metadata=objective_metadata,
            )
        row = receipt["regimes"].setdefault(
            regime,
            {
                "status": "RUNNING",
                "checkpoint_metadata": checkpoint_metadata,
                "objective_metadata": objective_metadata,
                "arms": [],
            },
        )
        if row.get("status") == "MEASURED":
            continue
        try:
            renderer, code, _model, _dash = _load_renderer(checkpoint)
            initial_theta = torch.as_tensor(code[1], dtype=torch.float32)
            row.setdefault(
                "renderer_parity_canary", _renderer_parity_canary(renderer, initial_theta.clone().requires_grad_(True))
            )
            if row["renderer_parity_canary"]["max_abs"] != 0.0:
                raise RuntimeError("conditional chart differs from settled Renderer through contest_r")
            # Negative control: provider rejects a forged split identity before it
            # may evaluate the scorer prefix. This establishes that the bank is
            # topology-bound rather than a reusable anonymous activation blob.
            try:
                live_custody = _live_decision_custody(regime)
                objective_context = _objective_fingerprint(row["objective_metadata"], receipt, live_custody)
                scorer_fingerprint = live_custody["segnet_sha256"]
                yopo_first_layer_costate_torch(
                    segnet=segnet,
                    current_frame=torch.zeros((1, 3, 384, 512), dtype=torch.float32),
                    bank_path=args.output_dir / "scratch" / "nonexistent_negative_canary.npz",
                    expected_bank_sha256="0" * 64,
                    objective_context_fingerprint=objective_context,
                    scorer_fingerprint=scorer_fingerprint,
                    current_step=1,
                    expected_split_identity_sha256="f" * 64,
                    expected_anchor_frame_sha256="0" * 64,
                    expected_source_step=0,
                    max_staleness_steps=1,
                )
            except ValueError as exc:
                if "split identity mismatch" not in str(exc):
                    raise RuntimeError(f"forged-split canary rejected for the wrong reason: {exc}") from exc
                row.setdefault(
                    "provider_negative_canary",
                    {
                        "status": "MEASURED_REJECTED",
                        "reason": str(exc),
                        "control": "forged split identity; no bank/scorer-prefix fallback permitted",
                    },
                )
            else:
                raise RuntimeError("YOPO provider accepted forged split identity")
            for k_value in K_VALUES:
                arm = next((item for item in row["arms"] if int(item["K"]) == k_value), None)
                if arm is None:
                    arm = {"K": k_value, "status": "RUNNING", "steps": [], "cleanup": []}
                    row["arms"].append(arm)
                state_path = args.output_dir / "state" / f"{regime}_K{k_value}.npz"
                scratch_path = args.output_dir / "scratch" / f"{regime}_K{k_value}_active_bank.npz"
                if arm.get("status") in {"MEASURED", "BLOCKED"}:
                    continue
                if _reconcile_terminal_arm_cleanup(
                    receipt_path=receipt_path,
                    receipt=receipt,
                    arm=arm,
                    scratch_path=scratch_path,
                ):
                    continue
                if state_path.is_file():
                    theta_np, promoted_pending = _recover_pending_step(arm, state_path)
                    theta = torch.as_tensor(theta_np).clone()
                    if promoted_pending:
                        receipt["peak_rss_bytes"] = _peak_rss_bytes()
                        receipt["admission"] = _admission(receipt["regimes"])
                        _atomic_write(receipt_path, receipt)
                    _clear_staged_step_state(state_path, theta_np, len(arm["steps"]), arm.get("active_bank"))
                    if _reconcile_terminal_arm_cleanup(
                        receipt_path=receipt_path,
                        receipt=receipt,
                        arm=arm,
                        scratch_path=scratch_path,
                    ):
                        continue
                else:
                    theta = initial_theta.clone()
                for step_index in range(len(arm["steps"]), args.steps):
                    step_started = time.perf_counter()
                    theta = theta.detach().clone().requires_grad_(True)
                    frame = _render_chart(renderer, theta)
                    frame_nchw = frame.permute(0, 3, 1, 2).contiguous()
                    refresh = step_index % k_value == 0
                    live_custody = _live_decision_custody(regime)
                    objective_context = _objective_fingerprint(row["objective_metadata"], receipt, live_custody)
                    scorer_fingerprint = live_custody["segnet_sha256"]
                    decision_custody_changed = _decision_custody_changed(receipt, regime, live_custody)
                    operational_shared_seconds = time.perf_counter() - step_started
                    bank_write_seconds = 0.0
                    refresh_bank_capture_seconds = 0.0
                    exact_baseline_seconds = 0.0
                    exact_renderer_vjp_seconds = 0.0
                    candidate_renderer_vjp_seconds = 0.0
                    fallback_renderer_rerender_seconds = 0.0
                    measurement_only_renderer_rerender_seconds = 0.0
                    current_baseline_validation_seconds = 0.0
                    current_baseline_validation_forwards = 0
                    exact_costate = None
                    exact_grad = None
                    label_path_agreement = {
                        "status": "NOT_APPLICABLE_PROVIDER_FALLBACK",
                        "reason": "a rejected provider uses one operational exact-teacher label path and terminates NO-GO",
                    }
                    active = arm.get("active_bank")
                    provider_preflight_failure = None
                    provider_kwargs = None
                    if decision_custody_changed:
                        provider_preflight_failure = f"ValueError: live objective/scorer custody changed: {json.dumps(live_custody, sort_keys=True)}"
                    elif refresh:
                        # The refresh creates the bank; no prior bank is required.
                        pass
                    elif active:
                        provider_kwargs = {
                            "segnet": segnet,
                            "current_frame": frame_nchw,
                            "bank_path": scratch_path,
                            "expected_bank_sha256": active["sha256"],
                            "objective_context_fingerprint": objective_context,
                            "scorer_fingerprint": scorer_fingerprint,
                            "current_step": step_index,
                            "expected_split_identity_sha256": split_identity,
                            "expected_anchor_frame_sha256": active["anchor_frame_sha256"],
                            "expected_source_step": int(active["source_step"]),
                            "max_staleness_steps": k_value - 1,
                        }
                    else:
                        provider_preflight_failure = "RuntimeError: active YOPO bank custody is missing"
                    if refresh and provider_preflight_failure is None and k_value == 1:
                        # The K=1 control measures the ordinary full teacher first.
                        # YOPO bank capture below is a measurement-only positive
                        # canary and cannot warm this operational path.
                        exact_costate, holder, exact_baseline_seconds = _capture_exact_teacher_costate(
                            segnet=segnet, frame_nchw=frame_nchw, labels=labels
                        )
                        exact_grad_started = time.perf_counter()
                        exact_grad = torch.autograd.grad((frame_nchw * exact_costate).sum(), theta)[0]
                        exact_renderer_vjp_seconds = time.perf_counter() - exact_grad_started
                        candidate_costate = exact_costate
                        candidate_grad = exact_grad
                        candidate_renderer_vjp_seconds = exact_renderer_vjp_seconds
                        provider_seconds = 0.0
                        provider_fallback = False
                        provider_metadata = {"selected_mode": "ordinary_exact_K1_control"}
                    elif refresh and provider_preflight_failure is None:
                        # The K>1 refresh path is measured before any ordinary
                        # teacher/reference control, so its timing is not
                        # systematically cache-warmed by excluded diagnostics.
                        _fresh_bank, exact_costate, holder, refresh_bank_capture_seconds = _capture_labeled_teacher(
                            capture=capture_yopo_first_layer_bank,
                            segnet=segnet,
                            frame_nchw=frame_nchw,
                            labels=labels,
                            objective_context=objective_context,
                            scorer_fingerprint=scorer_fingerprint,
                            step_index=step_index,
                        )
                        bank_write_started = time.perf_counter()
                        bank_sha256 = write_yopo_first_layer_bank(scratch_path, _fresh_bank)
                        bank_write_seconds = time.perf_counter() - bank_write_started
                        arm["active_bank"] = {
                            "path": str(scratch_path),
                            "sha256": bank_sha256,
                            "source_step": step_index,
                            "anchor_frame_sha256": _fresh_bank.anchor_frame_sha256,
                            "rebuild": "fresh exact teacher CE + blocks[0] adjoint at recorded code-row state",
                        }
                        active = arm["active_bank"]
                        candidate_costate = exact_costate
                        candidate_grad_started = time.perf_counter()
                        candidate_grad = torch.autograd.grad((frame_nchw * candidate_costate).sum(), theta)[0]
                        candidate_renderer_vjp_seconds = time.perf_counter() - candidate_grad_started
                        exact_grad = candidate_grad
                        provider_seconds = 0.0
                        provider_fallback = False
                        provider_metadata = {"selected_mode": "exact_refresh_from_banked_teacher"}
                    else:
                        # Non-refresh provider work is timed before the fresh
                        # exact-teacher diagnostic.  On provider rejection the
                        # exact teacher is then run as the operational fallback.
                        candidate_costate, provider_metadata, provider_seconds, provider_fallback = (
                            _provider_or_full_teacher_fallback(
                                provider=yopo_first_layer_costate_torch,
                                provider_kwargs=provider_kwargs,
                                exact_costate=None,
                                preflight_failure=provider_preflight_failure,
                            )
                        )
                        if not provider_fallback:
                            candidate_grad_started = time.perf_counter()
                            candidate_grad = torch.autograd.grad((frame_nchw * candidate_costate).sum(), theta)[0]
                            candidate_renderer_vjp_seconds = time.perf_counter() - candidate_grad_started
                            current_validation_started = time.perf_counter()
                            with torch.inference_mode():
                                current_ce, current_dseg = _evaluate_teacher(segnet, frame, labels)
                            current_baseline_validation_seconds = time.perf_counter() - current_validation_started
                            current_baseline_validation_forwards = 1
                            holder = {"ce": current_ce, "dseg": current_dseg}
                        else:
                            exact_costate, holder, exact_baseline_seconds = _capture_exact_teacher_costate(
                                segnet=segnet, frame_nchw=frame_nchw, labels=labels
                            )
                            exact_grad_started = time.perf_counter()
                            exact_grad = torch.autograd.grad((frame_nchw * exact_costate).sum(), theta)[0]
                            exact_renderer_vjp_seconds = time.perf_counter() - exact_grad_started
                            candidate_costate = exact_costate
                            candidate_grad = exact_grad
                    bank_age_steps = 0 if refresh else step_index - int(active["source_step"]) if active else -1
                    candidate_norm = float(torch.linalg.vector_norm(candidate_grad).item())
                    if candidate_norm == 0.0 or not np.isfinite(candidate_norm):
                        provider_fallback = True
                        provider_metadata = {
                            "selected_mode": "full_teacher_fallback",
                            "provider_failure": "ValueError: YOPO renderer gradient is zero/nonfinite",
                        }
                        if exact_costate is None or exact_grad is None:
                            exact_costate, holder, exact_baseline_seconds = _capture_exact_teacher_costate(
                                segnet=segnet, frame_nchw=frame_nchw, labels=labels
                            )
                            fallback_rerender_started = time.perf_counter()
                            fallback_frame = _render_chart(renderer, theta)
                            fallback_frame_nchw = fallback_frame.permute(0, 3, 1, 2).contiguous()
                            fallback_renderer_rerender_seconds = time.perf_counter() - fallback_rerender_started
                            if not torch.equal(fallback_frame_nchw, frame_nchw):
                                raise RuntimeError("fallback renderer re-evaluation changed the current frame")
                            exact_grad_started = time.perf_counter()
                            exact_grad = torch.autograd.grad((fallback_frame_nchw * exact_costate).sum(), theta)[0]
                            exact_renderer_vjp_seconds = time.perf_counter() - exact_grad_started
                        candidate_costate = exact_costate
                        candidate_grad = exact_grad
                        candidate_norm = float(torch.linalg.vector_norm(exact_grad).item())
                    (
                        candidate_theta,
                        target_norm,
                        recess_trials,
                        recess_validation_forwards,
                        candidate_validation_seconds,
                    ) = _select_candidate_recess(
                        renderer=renderer,
                        theta=theta,
                        candidate_grad=candidate_grad,
                        segnet=segnet,
                        labels=labels,
                        current_loss=holder["ce"],
                        current_dseg=holder["dseg"],
                    )
                    operational_components_completed = time.perf_counter()
                    if provider_fallback:
                        operational_components = {
                            "shared_renderer_and_live_custody": operational_shared_seconds,
                            "provider_attempt": provider_seconds,
                            "candidate_renderer_vjp_attempt": candidate_renderer_vjp_seconds,
                            "current_baseline_validation": current_baseline_validation_seconds,
                            "fallback_renderer_rerender": fallback_renderer_rerender_seconds,
                            "exact_teacher_fallback": exact_baseline_seconds,
                            "exact_renderer_vjp": exact_renderer_vjp_seconds,
                            "candidate_validation": candidate_validation_seconds,
                        }
                        operational_path = "provider_attempt_then_exact_teacher_fallback_plus_candidate_validation"
                    elif refresh:
                        operational_components = {
                            "shared_renderer_and_live_custody": operational_shared_seconds,
                            "refresh_teacher": (
                                exact_baseline_seconds if k_value == 1 else refresh_bank_capture_seconds
                            ),
                            "refresh_bank_write": bank_write_seconds if k_value > 1 else 0.0,
                            "candidate_renderer_vjp": candidate_renderer_vjp_seconds,
                            "candidate_validation": candidate_validation_seconds,
                        }
                        operational_path = "refresh_teacher_renderer_vjp_plus_candidate_validation"
                    else:
                        operational_components = {
                            "shared_renderer_and_live_custody": operational_shared_seconds,
                            "provider": provider_seconds,
                            "candidate_renderer_vjp": candidate_renderer_vjp_seconds,
                            "current_baseline_validation": current_baseline_validation_seconds,
                            "candidate_validation": candidate_validation_seconds,
                        }
                        operational_path = (
                            "current_frame_prefix_vjp_renderer_vjp_current_labels_plus_candidate_validation"
                        )
                    operational_timing = _operational_timing_record(
                        step_started=step_started,
                        operational_completed=operational_components_completed,
                        component_seconds=operational_components,
                        path=operational_path,
                    )
                    operational_seconds = float(operational_timing["wall_seconds"])
                    measurement_controls_started = time.perf_counter()
                    timing_order_canary = {
                        "status": (
                            "PASS" if measurement_controls_started >= operational_components_completed else "FAIL"
                        ),
                        "operational_components_completed_offset_seconds": (
                            operational_components_completed - step_started
                        ),
                        "measurement_controls_started_offset_seconds": measurement_controls_started - step_started,
                        "contract": "operational provider/renderer/label/candidate path precedes excluded diagnostics",
                    }
                    if refresh and provider_preflight_failure is None and k_value == 1:
                        _fresh_bank, refresh_exact_costate, refresh_holder, refresh_bank_capture_seconds = (
                            _capture_labeled_teacher(
                                capture=capture_yopo_first_layer_bank,
                                segnet=segnet,
                                frame_nchw=frame_nchw,
                                labels=labels,
                                objective_context=objective_context,
                                scorer_fingerprint=scorer_fingerprint,
                                step_index=step_index,
                            )
                        )
                        bank_write_started = time.perf_counter()
                        bank_sha256 = write_yopo_first_layer_bank(scratch_path, _fresh_bank)
                        bank_write_seconds = time.perf_counter() - bank_write_started
                        arm["active_bank"] = {
                            "path": str(scratch_path),
                            "sha256": bank_sha256,
                            "source_step": step_index,
                            "anchor_frame_sha256": _fresh_bank.anchor_frame_sha256,
                            "rebuild": "fresh exact teacher CE + blocks[0] adjoint at recorded code-row state",
                        }
                        active = arm["active_bank"]
                        label_path_agreement = _require_teacher_label_path_agreement(holder, refresh_holder)
                        if array_content_sha256(refresh_exact_costate) != array_content_sha256(exact_costate):
                            raise RuntimeError("YOPO refresh capture differs from the ordinary exact-teacher baseline")
                    elif refresh and provider_preflight_failure is None:
                        diagnostic_costate, diagnostic_holder, exact_baseline_seconds = _capture_exact_teacher_costate(
                            segnet=segnet, frame_nchw=frame_nchw, labels=labels
                        )
                        diagnostic_rerender_started = time.perf_counter()
                        diagnostic_frame = _render_chart(renderer, theta)
                        diagnostic_frame_nchw = diagnostic_frame.permute(0, 3, 1, 2).contiguous()
                        measurement_only_renderer_rerender_seconds = time.perf_counter() - diagnostic_rerender_started
                        if not torch.equal(diagnostic_frame_nchw, frame_nchw):
                            raise RuntimeError("diagnostic renderer re-evaluation changed the current frame")
                        diagnostic_grad_started = time.perf_counter()
                        diagnostic_grad = torch.autograd.grad(
                            (diagnostic_frame_nchw * diagnostic_costate).sum(), theta
                        )[0]
                        exact_renderer_vjp_seconds = time.perf_counter() - diagnostic_grad_started
                        label_path_agreement = _require_teacher_label_path_agreement(holder, diagnostic_holder)
                        if array_content_sha256(diagnostic_costate) != array_content_sha256(exact_costate):
                            raise RuntimeError("YOPO refresh capture differs from the ordinary exact-teacher baseline")
                        exact_costate = diagnostic_costate
                        exact_grad = diagnostic_grad
                    elif not provider_fallback:
                        # Fresh-teacher agreement is a measurement-only control
                        # and deliberately follows the complete operational
                        # provider/VJP/current-label/candidate-validation path.
                        exact_costate, exact_holder, exact_baseline_seconds = _capture_exact_teacher_costate(
                            segnet=segnet, frame_nchw=frame_nchw, labels=labels
                        )
                        diagnostic_rerender_started = time.perf_counter()
                        diagnostic_frame = _render_chart(renderer, theta)
                        diagnostic_frame_nchw = diagnostic_frame.permute(0, 3, 1, 2).contiguous()
                        measurement_only_renderer_rerender_seconds = time.perf_counter() - diagnostic_rerender_started
                        if not torch.equal(diagnostic_frame_nchw, frame_nchw):
                            raise RuntimeError("diagnostic renderer re-evaluation changed the current frame")
                        exact_grad_started = time.perf_counter()
                        exact_grad = torch.autograd.grad((diagnostic_frame_nchw * exact_costate).sum(), theta)[0]
                        exact_renderer_vjp_seconds = time.perf_counter() - exact_grad_started
                        label_path_agreement = _require_teacher_label_path_agreement(holder, exact_holder)
                    exact_norm = float(torch.linalg.vector_norm(exact_grad).item())
                    if exact_norm == 0.0 or not np.isfinite(exact_norm):
                        raise RuntimeError("exact-teacher renderer gradient is zero/nonfinite")
                    if target_norm is None or candidate_theta is None:
                        step_row = {
                            "step": step_index,
                            "refresh": refresh,
                            "status": "MEASURED",
                            "candidate_non_descent": True,
                            "provider_fallback": provider_fallback,
                            "fallback_to_full_teacher": True,
                            "blocker": "candidate anneal reached fp32 completion without CE+d_seg descent",
                            "exact_teacher_ce": holder["ce"],
                            "exact_teacher_dseg": holder["dseg"],
                            "recess_trials": recess_trials,
                            "bank_age_steps": bank_age_steps,
                            "teacher_label_path_agreement": label_path_agreement,
                            "operational_timing": operational_timing,
                            "timing_order_canary": timing_order_canary,
                            "timing_measured_seconds": {
                                "validation_harness_whole_step": time.perf_counter() - step_started,
                                "operational_cycle": operational_seconds,
                            },
                        }
                        _stage_step_state(
                            state_path,
                            theta.detach().cpu().numpy(),
                            len(arm["steps"]) + 1,
                            step_row,
                            arm.get("active_bank"),
                        )
                        arm["steps"].append(step_row)
                        terminal_status = _terminal_status_for_step(step_row)
                        assert terminal_status is not None
                        arm["status"] = terminal_status
                        receipt["peak_rss_bytes"] = _peak_rss_bytes()
                        receipt["admission"] = _admission(receipt["regimes"])
                        _atomic_write(receipt_path, receipt)
                        _clear_staged_step_state(
                            state_path,
                            theta.detach().cpu().numpy(),
                            len(arm["steps"]),
                            arm.get("active_bank"),
                        )
                        break
                    selected_candidate = next(trial for trial in recess_trials if trial.get("accepted"))
                    candidate_ce = float(selected_candidate["candidate_ce"])
                    candidate_dseg = float(selected_candidate["candidate_dseg"])
                    exact_reference_theta = theta.detach() - (target_norm / exact_norm) * exact_grad.detach()
                    exact_reference_started = time.perf_counter()
                    with torch.inference_mode():
                        exact_reference_ce, exact_reference_dseg = _evaluate_teacher(
                            segnet, _render_chart(renderer, exact_reference_theta), labels
                        )
                    exact_reference_validation_seconds = time.perf_counter() - exact_reference_started
                    reverse_started = time.perf_counter()
                    with torch.inference_mode():
                        reverse_theta = theta.detach() + (target_norm / candidate_norm) * candidate_grad.detach()
                        reverse_ce, reverse_dseg = _evaluate_teacher(
                            segnet, _render_chart(renderer, reverse_theta), labels
                        )
                    reverse_validation_seconds = time.perf_counter() - reverse_started
                    t_exact = exact_baseline_seconds + exact_renderer_vjp_seconds
                    t_approx = provider_seconds + candidate_renderer_vjp_seconds
                    ceiling = k_value * t_exact / (t_exact + (k_value - 1) * t_approx) if t_approx > 0.0 else None
                    exact_candidate_max_abs = float((exact_costate - candidate_costate).abs().max().item())
                    if refresh and exact_candidate_max_abs != 0.0:
                        raise RuntimeError("exact-refresh provider canary differs from fresh teacher costate")
                    candidate_non_descent = bool(candidate_ce >= holder["ce"] or candidate_dseg > holder["dseg"])
                    reverse_non_descent = bool(reverse_ce >= holder["ce"] or reverse_dseg > holder["dseg"])
                    step_row = {
                        "step": step_index,
                        "status": "MEASURED",
                        "refresh": refresh,
                        "bank_age_steps": bank_age_steps,
                        "exact_teacher_ce": holder["ce"],
                        "exact_teacher_dseg": holder["dseg"],
                        "candidate_ce": candidate_ce,
                        "candidate_dseg": candidate_dseg,
                        "exact_reference_ce": exact_reference_ce,
                        "exact_reference_dseg": exact_reference_dseg,
                        "candidate_non_descent": candidate_non_descent,
                        "provider_fallback": provider_fallback,
                        "fallback_to_full_teacher": provider_fallback or candidate_non_descent,
                        "ce_regret_vs_exact_reference": candidate_ce - exact_reference_ce,
                        "dseg_regret_vs_exact_reference": candidate_dseg - exact_reference_dseg,
                        "costate_metrics_global": _metrics(exact_costate, candidate_costate),
                        "costate_metrics_gt_boundary_annulus_bottom_k_0p05": _metrics(
                            exact_costate, candidate_costate, mask=annulus
                        ),
                        "renderer_gradient_cosine": _cosine(
                            exact_grad.detach().numpy(), candidate_grad.detach().numpy()
                        ),
                        "parameter_step_norm": target_norm,
                        "exact_teacher_gradient_norm": exact_norm,
                        "yopo_gradient_norm": candidate_norm,
                        "recess_trials": recess_trials,
                        "provider_metadata": provider_metadata,
                        "operational_path": operational_path,
                        "operational_timing": operational_timing,
                        "timing_order_canary": timing_order_canary,
                        "controls": {
                            "teacher_label_path_agreement": label_path_agreement,
                            "refresh_exact_positive_canary": (
                                {
                                    "status": "MEASURED",
                                    "costate_max_abs": exact_candidate_max_abs,
                                    "candidate_is_same_refresh_provider_vjp": True,
                                }
                                if refresh
                                else None
                            ),
                            "K1_exact_refresh_positive_canary": (
                                {
                                    "status": "MEASURED",
                                    "costate_max_abs": exact_candidate_max_abs,
                                    "candidate_is_same_refresh_provider_vjp": True,
                                }
                                if k_value == 1
                                else None
                            ),
                            "sign_reversed_behavioral_negative": {
                                "status": "PASS" if reverse_non_descent else "FAIL",
                                "candidate_ce": candidate_ce,
                                "candidate_dseg": candidate_dseg,
                                "sign_reversed_ce": reverse_ce,
                                "sign_reversed_dseg": reverse_dseg,
                                "reverse_non_descent_from_current": reverse_non_descent,
                            },
                        },
                        "timing_measured_seconds": {
                            "operational_shared_renderer_and_live_custody": operational_shared_seconds,
                            "refresh_bank_write": bank_write_seconds,
                            "ordinary_exact_teacher_forward_backward": exact_baseline_seconds,
                            "refresh_yopo_bank_capture_forward_backward": refresh_bank_capture_seconds,
                            "exact_renderer_vjp": exact_renderer_vjp_seconds,
                            "yopo_provider": provider_seconds,
                            "candidate_renderer_vjp": candidate_renderer_vjp_seconds,
                            "operational_fallback_renderer_rerender": fallback_renderer_rerender_seconds,
                            "current_baseline_validation_including_labels": current_baseline_validation_seconds,
                            "candidate_validation_including_labels": candidate_validation_seconds,
                            "measurement_only_exact_renderer_rerender": measurement_only_renderer_rerender_seconds,
                            "measurement_only_exact_reference_validation": exact_reference_validation_seconds,
                            "measurement_only_reverse_control_validation": reverse_validation_seconds,
                            "operational_cycle": operational_seconds,
                            "validation_harness_whole_step": time.perf_counter() - step_started,
                        },
                        "teacher_work_counts": {
                            "actual_probe_teacher_forward_backward_including_labels": 1 + int(refresh),
                            "operational_teacher_forward_backward_including_labels": int(refresh or provider_fallback),
                            "measurement_only_teacher_forward_backward_including_labels": (
                                1 + int(refresh) - int(refresh or provider_fallback)
                            ),
                            "operational_validation_forwards_including_labels": (
                                current_baseline_validation_forwards + recess_validation_forwards
                            ),
                            "measurement_only_control_forwards_including_labels": 2,
                            "actual_probe_teacher_forwards_total_including_labels": (
                                1 + int(refresh) + current_baseline_validation_forwards + recess_validation_forwards + 2
                            ),
                            "full_teacher_fallbacks": int(provider_fallback or candidate_non_descent),
                        },
                        "algebraic_speed_ceiling_derived": {
                            "formula": "K*t_exact/(t_exact+(K-1)*t_approx)",
                            "K": k_value,
                            "t_exact_measured_seconds": t_exact,
                            "t_approx_measured_seconds": t_approx,
                            "ceiling": ceiling,
                            "assumptions": "idealized refresh/VJP-only model; excludes labels, recession trials, candidate/reference/reverse validation, receipt IO, and renderer chart work",
                        },
                    }
                    # A failed provider step never advances the approximate trajectory.
                    # The already-measured exact reference is the automatic full-teacher fallback.
                    theta = (
                        theta.detach() - (target_norm / exact_norm) * exact_grad.detach()
                        if provider_fallback or candidate_non_descent
                        else candidate_theta.detach()
                    )
                    _stage_step_state(
                        state_path,
                        theta.cpu().numpy(),
                        len(arm["steps"]) + 1,
                        step_row,
                        arm.get("active_bank"),
                    )
                    arm["steps"].append(step_row)
                    terminal_status = _terminal_status_for_step(step_row)
                    if terminal_status is not None:
                        arm["status"] = terminal_status
                    receipt["peak_rss_bytes"] = _peak_rss_bytes()
                    receipt["admission"] = _admission(receipt["regimes"])
                    _atomic_write(receipt_path, receipt)
                    _clear_staged_step_state(
                        state_path,
                        theta.cpu().numpy(),
                        len(arm["steps"]),
                        arm.get("active_bank"),
                    )
                    if provider_fallback or candidate_non_descent:
                        break
                if _arm_ready_for_success_finalization(arm, expected_steps=args.steps):
                    pending_cleanup, _removed_cleanup = _require_active_scratch_custody(arm, scratch_path)
                    if scratch_path.exists() or pending_cleanup:
                        _certify_then_remove_scratch(
                            receipt_path=receipt_path,
                            receipt=receipt,
                            arm=arm,
                            scratch_path=scratch_path,
                            reason="completed arm has durable state and receipt metadata",
                        )
                    arm["active_bank"] = None
                    arm["status"] = "MEASURED"
                    positive_pass = bool(
                        all(
                            step["controls"]["refresh_exact_positive_canary"] is None
                            or step["controls"]["refresh_exact_positive_canary"]["costate_max_abs"] == 0.0
                            for step in arm["steps"]
                        )
                    )
                    negative_pass = all(
                        step["controls"]["sign_reversed_behavioral_negative"]["status"] == "PASS"
                        for step in arm["steps"]
                    )
                    full_horizon_pass = max(int(step["bank_age_steps"]) for step in arm["steps"]) >= k_value - 1
                    timing_order_pass = all(step["timing_order_canary"]["status"] == "PASS" for step in arm["steps"])
                    label_path_floor_pass = all(
                        step["controls"]["teacher_label_path_agreement"]["status"] == "PASS"
                        for step in arm["steps"]
                    )
                    arm["controls"] = {
                        "status": (
                            "PASS"
                            if (
                                positive_pass
                                and negative_pass
                                and full_horizon_pass
                                and timing_order_pass
                                and label_path_floor_pass
                            )
                            else "FAIL"
                        ),
                        "refresh_exact_positive_all_refreshes": positive_pass,
                        "K1_exact_refresh_positive": positive_pass if k_value == 1 else "NOT_APPLICABLE",
                        "sign_reversed_negative_all_steps": negative_pass,
                        "full_age_horizon_exercised": full_horizon_pass,
                        "operational_before_measurement_controls": timing_order_pass,
                        "same_frame_teacher_label_path_floor_all_steps": label_path_floor_pass,
                        "required_max_bank_age_steps": k_value - 1,
                        "observed_max_bank_age_steps": max(int(step["bank_age_steps"]) for step in arm["steps"]),
                    }
                    arm["summary"] = {
                        "mean_operational_cycle_seconds": float(
                            np.mean([step["timing_measured_seconds"]["operational_cycle"] for step in arm["steps"]])
                        ),
                        "mean_validation_harness_whole_step_seconds": float(
                            np.mean(
                                [
                                    step["timing_measured_seconds"]["validation_harness_whole_step"]
                                    for step in arm["steps"]
                                ]
                            )
                        ),
                        "mean_ce_regret_vs_exact_reference": float(
                            np.mean([step["ce_regret_vs_exact_reference"] for step in arm["steps"]])
                        ),
                        "mean_dseg_regret_vs_exact_reference": float(
                            np.mean([step["dseg_regret_vs_exact_reference"] for step in arm["steps"]])
                        ),
                        "mean_algebraic_speed_ceiling_derived": float(
                            np.mean([step["algebraic_speed_ceiling_derived"]["ceiling"] for step in arm["steps"]])
                        ),
                        "teacher_work_total": {
                            "actual_probe_teacher_forward_backward_including_labels": int(
                                sum(
                                    step["teacher_work_counts"][
                                        "actual_probe_teacher_forward_backward_including_labels"
                                    ]
                                    for step in arm["steps"]
                                )
                            ),
                            "operational_teacher_forward_backward_including_labels": int(
                                sum(
                                    step["teacher_work_counts"]["operational_teacher_forward_backward_including_labels"]
                                    for step in arm["steps"]
                                )
                            ),
                            "operational_validation_forwards_including_labels": int(
                                sum(
                                    step["teacher_work_counts"]["operational_validation_forwards_including_labels"]
                                    for step in arm["steps"]
                                )
                            ),
                            "measurement_only_teacher_forwards_including_labels": int(
                                sum(
                                    step["teacher_work_counts"][
                                        "measurement_only_teacher_forward_backward_including_labels"
                                    ]
                                    + step["teacher_work_counts"]["measurement_only_control_forwards_including_labels"]
                                    for step in arm["steps"]
                                )
                            ),
                            "full_teacher_fallbacks": int(
                                sum(step["teacher_work_counts"]["full_teacher_fallbacks"] for step in arm["steps"])
                            ),
                        },
                    }
                    _atomic_write(receipt_path, receipt)
                if arm.get("status") in {"NO_GO_NON_DESCENT", "NO_GO_PROVIDER_FALLBACK"}:
                    _reconcile_terminal_arm_cleanup(
                        receipt_path=receipt_path,
                        receipt=receipt,
                        arm=arm,
                        scratch_path=scratch_path,
                    )
            terminal_arm_statuses = {
                "MEASURED",
                "NO_GO_NON_DESCENT",
                "NO_GO_PROVIDER_FALLBACK",
            }
            row["status"] = (
                "MEASURED" if all(a.get("status") in terminal_arm_statuses for a in row["arms"]) else "BLOCKED"
            )
            if row["status"] == "MEASURED":
                k1 = next(arm for arm in row["arms"] if int(arm["K"]) == 1)
                measured_arms = [arm for arm in row["arms"] if arm.get("status") == "MEASURED"]
                positive_pass = k1.get("controls", {}).get("K1_exact_refresh_positive") is True
                eligible_controls_pass = bool(measured_arms) and all(
                    arm.get("controls", {}).get("status") == "PASS" for arm in measured_arms
                )
                row["controls"] = {
                    "status": "PASS" if positive_pass and eligible_controls_pass else "FAIL",
                    "K1_exact_refresh_positive": positive_pass,
                    "eligible_arm_controls": {
                        str(arm["K"]): arm.get("controls", {"status": "MISSING"}) for arm in measured_arms
                    },
                    "provider_forged_split_negative": row.get("provider_negative_canary", {}).get("status")
                    == "MEASURED_REJECTED",
                }
                if not row["controls"]["provider_forged_split_negative"]:
                    row["controls"]["status"] = "FAIL"
                row["pareto"] = _pareto_knee(row["arms"])
        except Exception as exc:  # A receipt is more useful than an orphaned local traceback.
            row["status"] = "BLOCKED"
            row["blocker"] = f"{type(exc).__name__}: {exc}"
        receipt["peak_rss_bytes"] = _peak_rss_bytes()
        receipt["admission"] = _admission(receipt["regimes"])
        _atomic_write(receipt_path, receipt)
    receipt["completed_at_utc"] = datetime.now(UTC).isoformat()
    receipt["admission"] = _admission(receipt["regimes"])
    _atomic_write(receipt_path, receipt)
    _release_output_lock(lock_fd)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=4, choices=(4,))
    parser.add_argument("--seed", type=int, default=20260712)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO / "experiments/results" / f"yopo_first_layer_costate_probe_{stamp}"
    args.output_dir = args.output_dir.resolve()
    receipt = run(args)
    print(
        json.dumps(
            {"receipt": str(args.output_dir / "receipt.json"), "admission": receipt["admission"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
