#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable local-n600 probe for round-4 support-ranking formulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
TOOLS = REPO / "tools"
for entry in (SRC, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import probe_frozen_replay_convex_head as round2  # noqa: E402

from tac.causal_manifest import check_fore_support  # noqa: E402
from tac.scorer_surrogate.frozen_replay_convex_head import (  # noqa: E402
    ReplayAssignment,
    deterministic_replay_assignments,
)
from tac.scorer_surrogate.replace_round3_fidelity_wall import (  # noqa: E402
    ConvMacLedger,
    local_prefix_feature_snapshot,
)
from tac.scorer_surrogate.replace_round4_support_ranking import (  # noqa: E402
    AUTHORITY_SCOPE,
    BLOCK_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    ORDERED_PAIR_COUNT,
    SCHEMA,
    ConvexScalarFit,
    IsotonicCalibrator,
    QuadraticStatistics,
    aggregate_mass_localization,
    array_sha256,
    block_scores,
    calibrated_block_scores,
    capture_exact_support_teacher,
    exact_support_target,
    fit_block_calibrators,
    fit_exact_quadratic,
    fit_isotonic_calibrator,
    global_scores,
    mass_localization_metrics,
    pair_id_to_classes,
    pairwise_rank_block_statistics,
    support_feature_matrices,
    weighted_topk_block_statistics,
    weighted_topk_statistics,
)
from tac.witness_dsl.replace_round4_support_ranking_policy import (  # noqa: E402
    ReplaceRound4SupportRankingPolicy,
)

LANE_ID = "lane_replace_round4_support_ranking_20260713"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex fit; frozen CPU SegNet exact costates]"
DEFAULT_OUTPUT = REPO / "experiments/results/replace_round4_support_ranking_20260713"
PREREGISTRATION = DEFAULT_OUTPUT / "preregistration.json"
PREREGISTRATION_SHA256 = "ec90adf96b0ec8f239409971f55bb9f5d3f8e442365df772a2ce983d9521c8ff"
STORAGE_PREFLIGHT = REPO / ".omx/research/replace_round4_support_ranking_storage_preflight_20260713.json"
ROUND3_RECEIPT = REPO / "experiments/results/replace_round3_fidelity_wall_20260713/measurement_receipt.json"

SOURCE_FILES = (
    "src/tac/scorer_surrogate/replace_round4_support_ranking.py",
    "src/tac/witness_dsl/replace_round4_support_ranking_policy.py",
    "tools/probe_replace_round4_support_ranking.py",
    "src/tac/scorer_surrogate/replace_round3_fidelity_wall.py",
    "tools/probe_replace_round3_fidelity_wall.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_yopo_first_layer_costate.py",
    "src/tac/causal_manifest.py",
)
SOURCE_AMENDMENT_ID = "resume-accumulator-schema-v2"
SOURCE_AMENDMENT_CHANGED_PATHS = frozenset(
    {
        "tools/probe_replace_round4_support_ranking.py",
    }
)
SOURCE_AMENDMENT_OLD_SHA256 = {
    "tools/probe_replace_round4_support_ranking.py": (
        "c023dba24c6a0103c955d59c7b4411fe5c82cf7a372525a5327ea4e8032c5815"
    ),
}


class ProbeError(RuntimeError):
    """Measurement or evidence custody failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _json_normalized(payload: Any) -> Any:
    """Return the exact JSON-domain value used by durable run contracts."""

    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


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


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    result = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing measurement source {relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _source_bundle(
    output_dir: Path, *, bundle_relative: Path = Path("source_bundle")
) -> dict[str, dict[str, Any]]:
    if bundle_relative.is_absolute() or ".." in bundle_relative.parts:
        raise ProbeError("source bundle path must remain inside the run directory")
    result = {}
    for relative, custody in _source_fingerprints().items():
        source = REPO / relative
        destination = output_dir / bundle_relative / relative
        if destination.is_file():
            if destination.stat().st_size != custody["bytes"] or _sha256(destination) != custody["sha256"]:
                raise ProbeError(f"source bundle drift for {relative}")
        else:
            _atomic_bytes(destination, source.read_bytes())
        result[relative] = {**custody, "path": str(destination.relative_to(output_dir))}
    return result


def _custody_fingerprints(
    custody: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        relative: {"bytes": row["bytes"], "sha256": row["sha256"]}
        for relative, row in custody.items()
    }


def _verify_bundled_sources(
    output_dir: Path, custody: dict[str, dict[str, Any]]
) -> None:
    for relative, row in custody.items():
        path = output_dir / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or _sha256(path) != row["sha256"]
        ):
            raise ProbeError(f"bundled source custody drift for {relative}")


def _verify_source_amendment_boundary(
    output_dir: Path, *, require_preheldout: bool
) -> dict[str, Any]:
    """Certify an outcome-blind repair after train labels and before heldout."""

    stage_path = output_dir / "stage_train_complete.json"
    if not stage_path.is_file():
        raise ProbeError("source amendment requires the completed training stage")
    stage = json.loads(stage_path.read_text())
    targets = stage.get("compact_targets", {})
    if stage.get("state_count") != 480 or len(targets) != 480:
        raise ProbeError("source-amendment training target cardinality drift")
    target_manifest = []
    for pair_index, custody in sorted(targets.items(), key=lambda row: int(row[0])):
        path = output_dir / custody["path"]
        if (
            not path.is_file()
            or path.stat().st_size != custody["bytes"]
            or _sha256(path) != custody["sha256"]
        ):
            raise ProbeError(f"source-amendment target custody drift for pair {pair_index}")
        target_manifest.append(
            {
                "pair_index": int(pair_index),
                "bytes": custody["bytes"],
                "sha256": custody["sha256"],
            }
        )
    accumulator = stage["accumulator"]
    accumulator_path = output_dir / accumulator["path"]
    if (
        not accumulator_path.is_file()
        or accumulator_path.stat().st_size != accumulator["bytes"]
        or _sha256(accumulator_path) != accumulator["sha256"]
    ):
        raise ProbeError("source-amendment accumulator custody drift")

    ledger_path = output_dir / "teacher_calls.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    starts = [
        row
        for row in rows
        if row.get("stage") == "round4_train_target"
        and row.get("event") == "exact_teacher_state_call_started"
    ]
    completions = [
        row
        for row in rows
        if row.get("stage") == "round4_train_target"
        and row.get("event") == "exact_teacher_state_call_completed"
    ]
    batches = [
        row
        for row in rows
        if row.get("stage") == "round4_train_target"
        and row.get("event") == "exact_teacher_batch_completed"
    ]
    if not (len(starts) == len(completions) == len(batches) == 480):
        raise ProbeError("source-amendment train teacher-call cardinality drift")
    train_rows = starts + completions + batches
    if require_preheldout:
        heldout = [
            row for row in rows if row.get("stage") == "round4_heldout_validation"
        ]
        heldout_records = list((output_dir / "heldout").glob("pair_*.json"))
        if (
            heldout
            or heldout_records
            or (output_dir / "stage_heldout_complete.json").exists()
        ):
            raise ProbeError("source amendment is forbidden after heldout observation")
        if (output_dir / "stage_fit_complete.json").exists():
            raise ProbeError("source amendment expected the pre-fit failure boundary")
    return {
        "status": "MEASURED",
        "outcome_blind": True,
        "training_stage": {
            "path": str(stage_path.relative_to(output_dir)),
            "bytes": stage_path.stat().st_size,
            "sha256": _sha256(stage_path),
            "target_count": len(target_manifest),
            "target_tree_sha256": hashlib.sha256(
                json.dumps(target_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "accumulator": accumulator,
        },
        "teacher_ledger": {
            "path": str(ledger_path.relative_to(output_dir)),
            "train_event_tree_sha256": hashlib.sha256(
                json.dumps(train_rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "train_calls_started": len(starts),
            "train_calls_completed": len(completions),
            "train_batches_completed": len(batches),
            "teacher_calls_recomputed_by_amendment": 0,
        },
    }


def _resolve_source_custody(
    output_dir: Path,
    *,
    prior_contract: dict[str, Any],
    requested_amendment: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    prior_sources = prior_contract.get("effective_sources", prior_contract["sources"])
    current = _source_fingerprints()
    if current == _custody_fingerprints(prior_sources):
        if requested_amendment is not None:
            raise ProbeError("source amendment requested without source drift")
        _verify_bundled_sources(output_dir, prior_sources)
        return prior_sources, None

    amendment_path = output_dir / f"source_amendment_{SOURCE_AMENDMENT_ID}.json"
    if amendment_path.is_file():
        amendment = json.loads(amendment_path.read_text())
        if amendment.get("amendment_id") != SOURCE_AMENDMENT_ID:
            raise ProbeError("source amendment identity drift")
        if amendment.get("old_sources") != prior_sources:
            raise ProbeError("source amendment old-source custody drift")
        if current != _custody_fingerprints(amendment["new_sources"]):
            raise ProbeError("current sources drifted from the sealed amendment")
        if (
            _verify_source_amendment_boundary(output_dir, require_preheldout=False)
            != amendment["recovery_boundary"]
        ):
            raise ProbeError("source amendment recovery boundary drift")
        effective = _source_bundle(
            output_dir,
            bundle_relative=Path("source_bundle_amendments") / SOURCE_AMENDMENT_ID,
        )
        if effective != amendment["new_sources"]:
            raise ProbeError("source amendment bundle custody drift")
        return effective, {
            "path": str(amendment_path.relative_to(output_dir)),
            "bytes": amendment_path.stat().st_size,
            "sha256": _sha256(amendment_path),
            "amendment_id": SOURCE_AMENDMENT_ID,
        }

    if requested_amendment != SOURCE_AMENDMENT_ID:
        raise ProbeError("source drift requires the explicit outcome-blind amendment")
    changed = {
        relative
        for relative in SOURCE_FILES
        if current[relative] != _custody_fingerprints(prior_sources)[relative]
    }
    if changed != SOURCE_AMENDMENT_CHANGED_PATHS:
        raise ProbeError(f"source amendment has unapproved deltas: {sorted(changed)}")
    for relative, expected in SOURCE_AMENDMENT_OLD_SHA256.items():
        if prior_sources[relative]["sha256"] != expected:
            raise ProbeError(f"source amendment old hash drift for {relative}")
    boundary = _verify_source_amendment_boundary(output_dir, require_preheldout=True)
    effective = _source_bundle(
        output_dir,
        bundle_relative=Path("source_bundle_amendments") / SOURCE_AMENDMENT_ID,
    )
    amendment = {
        "schema": "replace_round4_source_amendment.v1",
        "amendment_id": SOURCE_AMENDMENT_ID,
        "created_at_utc": _utc_now(),
        "reason": (
            "repair the resume-only accumulator schema check so the completed-pair vector may "
            "grow from its empty template while every sufficient-statistic array retains its "
            "sealed shape; no label, feature, target, objective, fit, calibration, or decision "
            "rule changes"
        ),
        "verdict_scope": (
            "resume accumulator validator only; the v1 rank-truncated MP amendment and all "
            "measurement semantics remain unchanged"
        ),
        "old_sources": prior_sources,
        "new_sources": effective,
        "source_delta": {
            relative: {"old": prior_sources[relative], "new": effective[relative]}
            for relative in sorted(changed)
        },
        "recovery_boundary": boundary,
        "teacher_calls_reused": 480,
        "teacher_calls_recomputed": 0,
        "pointer_moved": False,
    }
    _atomic_json(amendment_path, amendment)
    return effective, {
        "path": str(amendment_path.relative_to(output_dir)),
        "bytes": amendment_path.stat().st_size,
        "sha256": _sha256(amendment_path),
        "amendment_id": SOURCE_AMENDMENT_ID,
    }


def _validate_preregistration(policy: ReplaceRound4SupportRankingPolicy) -> dict[str, Any]:
    if not PREREGISTRATION.is_file() or _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ProbeError("round-4 preregistration is missing or drifted")
    payload = json.loads(PREREGISTRATION.read_text())
    contract = policy.compile_measurement_contract()
    registered = payload["measurement_contract"]
    if registered["primary_gate"] != contract["retained_mass_bar"]:
        raise ProbeError("typed policy and preregistered mass bar disagree")
    if [row["name"] for row in payload["rungs"]] != list(contract["rung_order"]):
        raise ProbeError("typed policy and preregistered rung order disagree")
    if registered["selected_prefix_cells"] != contract["selected_prefix_cells"]:
        raise ProbeError("typed policy and preregistered selection area disagree")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO)),
        "bytes": PREREGISTRATION.stat().st_size,
        "sha256": _sha256(PREREGISTRATION),
        "payload": payload,
    }


def _storage_custody(output_dir: Path) -> dict[str, Any]:
    if not STORAGE_PREFLIGHT.is_file():
        raise ProbeError("missing storage preflight")
    payload = json.loads(STORAGE_PREFLIGHT.read_text())
    if payload.get("blockers"):
        raise ProbeError(f"storage preflight blocked: {payload['blockers']}")
    if Path(payload.get("selected_workload_root", "")).resolve() != output_dir.resolve():
        raise ProbeError("storage preflight selected a different workload root")
    return {
        "path": str(STORAGE_PREFLIGHT.relative_to(REPO)),
        "bytes": STORAGE_PREFLIGHT.stat().st_size,
        "sha256": _sha256(STORAGE_PREFLIGHT),
        "selected_tier": payload["selected_tier"],
        "requested_bytes": payload["requested_bytes"],
        "explicit_local_opt_in": payload["operator_storage_policy"]["local_disk_enabled"],
        "blockers": [],
    }


def _runtime_custody(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _train_target_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "train_targets" / f"pair_{pair_index:04d}.npz"


def _heldout_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout" / f"pair_{pair_index:04d}.json"


def _empty_accumulator() -> dict[str, np.ndarray]:
    return {
        "completed_pairs": np.empty(0, dtype=np.int64),
        "global_gram": np.zeros((GLOBAL_FEATURE_COUNT, GLOBAL_FEATURE_COUNT), dtype=np.float64),
        "global_rhs": np.zeros(GLOBAL_FEATURE_COUNT, dtype=np.float64),
        "global_target_square": np.zeros(1, dtype=np.float64),
        "global_row_count": np.zeros(1, dtype=np.int64),
        "global_state_count": np.zeros(1, dtype=np.int64),
        "weighted_gram": np.zeros((ORDERED_PAIR_COUNT, BLOCK_FEATURE_COUNT, BLOCK_FEATURE_COUNT), dtype=np.float64),
        "weighted_rhs": np.zeros((ORDERED_PAIR_COUNT, BLOCK_FEATURE_COUNT), dtype=np.float64),
        "weighted_target_square": np.zeros(ORDERED_PAIR_COUNT, dtype=np.float64),
        "weighted_row_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
        "weighted_state_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
        "pairwise_gram": np.zeros((ORDERED_PAIR_COUNT, BLOCK_FEATURE_COUNT, BLOCK_FEATURE_COUNT), dtype=np.float64),
        "pairwise_rhs": np.zeros((ORDERED_PAIR_COUNT, BLOCK_FEATURE_COUNT), dtype=np.float64),
        "pairwise_target_square": np.zeros(ORDERED_PAIR_COUNT, dtype=np.float64),
        "pairwise_row_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
        "pairwise_state_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
    }


def _load_accumulator(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return _empty_accumulator()
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = _empty_accumulator()
    fixed_shape_keys = set(expected) - {"completed_pairs"}
    if (
        result.keys() != expected.keys()
        or result["completed_pairs"].ndim != 1
        or any(result[key].shape != expected[key].shape for key in fixed_shape_keys)
    ):
        raise ProbeError("training accumulator schema drift")
    completed = result["completed_pairs"]
    if (
        completed.dtype != np.int64
        or completed.size > 480
        or not np.array_equal(completed, np.unique(completed))
        or np.any((completed < 0) | (completed >= 600))
    ):
        raise ProbeError("training accumulator completion index drift")
    return result


def _save_accumulator(path: Path, accumulator: dict[str, np.ndarray]) -> None:
    _atomic_npz(path, **accumulator)


def _add_record(
    accumulator: dict[str, np.ndarray],
    *,
    assignment: ReplayAssignment,
    global_record: QuadraticStatistics,
    weighted_records: Sequence[QuadraticStatistics],
    pairwise_records: Sequence[QuadraticStatistics],
) -> None:
    accumulator["global_gram"] += global_record.gram
    accumulator["global_rhs"] += global_record.rhs
    accumulator["global_target_square"][0] += global_record.target_square
    accumulator["global_row_count"][0] += global_record.row_count
    accumulator["global_state_count"][0] += global_record.state_count
    for name, records in (("weighted", weighted_records), ("pairwise", pairwise_records)):
        for block, record in enumerate(records):
            accumulator[f"{name}_gram"][block] += record.gram
            accumulator[f"{name}_rhs"][block] += record.rhs
            accumulator[f"{name}_target_square"][block] += record.target_square
            accumulator[f"{name}_row_count"][block] += record.row_count
            accumulator[f"{name}_state_count"][block] += record.state_count
    accumulator["completed_pairs"] = np.sort(
        np.append(accumulator["completed_pairs"], assignment.pair_index).astype(np.int64)
    )


def _stats_from_accumulator(
    accumulator: dict[str, np.ndarray], name: str, block: int | None = None
) -> QuadraticStatistics:
    if block is None:
        return QuadraticStatistics(
            gram=accumulator[f"{name}_gram"],
            rhs=accumulator[f"{name}_rhs"],
            target_square=float(accumulator[f"{name}_target_square"][0]),
            row_count=int(accumulator[f"{name}_row_count"][0]),
            state_count=int(accumulator[f"{name}_state_count"][0]),
        )
    return QuadraticStatistics(
        gram=accumulator[f"{name}_gram"][block],
        rhs=accumulator[f"{name}_rhs"][block],
        target_square=float(accumulator[f"{name}_target_square"][block]),
        row_count=int(accumulator[f"{name}_row_count"][block]),
        state_count=int(accumulator[f"{name}_state_count"][block]),
    )


def _teacher_call(
    *,
    ledger: Path,
    assignment: ReplayAssignment,
    stage: str,
    frame_nchw: Any,
    labels_t: Any,
    segnet: Any,
    measure_cost: bool,
) -> tuple[Any, Any, np.ndarray, dict[str, float], float, dict[str, Any] | None]:
    batch_id = f"{stage}-p{assignment.pair_index:04d}-{time.time_ns()}"
    round2._teacher_start(ledger, assignment, stage=stage, batch_id=batch_id)
    cost_model = None
    if measure_cost:
        with ConvMacLedger(segnet) as mac_ledger:
            prefix, costate, pair_ids, metrics, elapsed = capture_exact_support_teacher(
                segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
            )
        cost_model = mac_ledger.summary()
    else:
        prefix, costate, pair_ids, metrics, elapsed = capture_exact_support_teacher(
            segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
        )
    round2._teacher_complete(
        ledger,
        assignment,
        stage=stage,
        batch_id=batch_id,
        teacher_metrics=metrics,
        elapsed_seconds=elapsed,
    )
    round2._append_jsonl(
        ledger,
        {
            "event": "exact_teacher_batch_completed",
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "batch_id": batch_id,
            "state_count": 1,
            "elapsed_seconds": elapsed,
        },
    )
    return prefix, costate, pair_ids, metrics, elapsed, cost_model


def _save_train_target(
    path: Path,
    *,
    assignment: ReplayAssignment,
    support: np.ndarray,
    pair_ids: np.ndarray,
    frame_sha256: str,
    prefix_sha256: str,
    costate_sha256: str,
    exact_mass_square: float,
    metrics: dict[str, float],
    elapsed: float,
) -> None:
    _atomic_npz(
        path,
        pair_index=np.asarray(assignment.pair_index, dtype=np.int64),
        checkpoint_index=np.asarray(assignment.checkpoint_index, dtype=np.int64),
        checkpoint_name=np.asarray(assignment.checkpoint_name),
        split=np.asarray(assignment.split),
        support=np.asarray(support, dtype=np.uint8),
        pair_ids=np.asarray(pair_ids, dtype=np.int16),
        frame_sha256=np.asarray(frame_sha256),
        prefix_sha256=np.asarray(prefix_sha256),
        exact_costate_sha256=np.asarray(costate_sha256),
        exact_mass_square=np.asarray(exact_mass_square, dtype=np.float64),
        teacher_ce=np.asarray(metrics["ce"], dtype=np.float64),
        teacher_dseg=np.asarray(metrics["dseg"], dtype=np.float64),
        teacher_elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
    )


def _load_train_target(path: Path, assignment: ReplayAssignment) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            int(archive["pair_index"].item()) != assignment.pair_index
            or int(archive["checkpoint_index"].item()) != assignment.checkpoint_index
            or str(archive["checkpoint_name"].item()) != assignment.checkpoint_name
            or str(archive["split"].item()) != "train"
        ):
            raise ProbeError(f"training target assignment drift at pair {assignment.pair_index}")
        support = np.array(archive["support"], dtype=np.bool_, copy=True)
        pair_ids = np.array(archive["pair_ids"], dtype=np.int16, copy=True)
    if support.shape != (48, 64) or pair_ids.shape != support.shape:
        raise ProbeError("training target lattice drift")
    return support, pair_ids


def _training_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound4SupportRankingPolicy,
    segnet: Any,
    yopo: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stage_path = output_dir / "stage_train_complete.json"
    accumulator_path = output_dir / "train_accumulator_current.npz"
    train = [row for row in assignments if row.split == "train"]
    accumulator = _load_accumulator(accumulator_path)
    completed = {int(value) for value in accumulator["completed_pairs"]}
    ledger = output_dir / "teacher_calls.jsonl"
    cost_path = output_dir / "prefix_cost_model.json"
    parity_rows = []
    stage_checkpoints = []
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.n_pairs or code.shape[0] != 2 * policy.n_pairs:
            raise ProbeError(f"checkpoint {checkpoint_name} is not the registered n600 renderer")
        parity = round2._checkpoint_parity(renderer, checkpoint_index)
        if parity["status"] != "MEASURED_PASS":
            raise ProbeError(f"renderer parity failed for {checkpoint_name}")
        parity_rows.append({"checkpoint_name": checkpoint_name, **parity})
        cohort = [row for row in train if row.checkpoint_index == checkpoint_index]
        for assignment in cohort:
            if assignment.pair_index in completed:
                _load_train_target(_train_target_path(output_dir, assignment.pair_index), assignment)
                continue
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            target_path = _train_target_path(output_dir, assignment.pair_index)
            if target_path.is_file():
                sampled_support, sampled_pairs = _load_train_target(target_path, assignment)
                prefix = local_prefix_feature_snapshot(segnet, frame)
            else:
                torch = __import__("torch")
                prefix, costate, pair_ids, metrics, elapsed, cost_model = _teacher_call(
                    ledger=ledger,
                    assignment=assignment,
                    stage="round4_train_target",
                    frame_nchw=frame,
                    labels_t=torch.as_tensor(label[None], dtype=torch.long),
                    segnet=segnet,
                    measure_cost=not cost_path.exists(),
                )
                if cost_model is not None:
                    _atomic_json(cost_path, cost_model)
                mass, support, count = exact_support_target(
                    costate.detach().cpu().numpy(), area_fraction=policy.requested_area_fraction
                )
                if count != policy.selected_prefix_cells:
                    raise ProbeError("training target selected-area drift")
                sampled_support = support[:: policy.train_lattice_stride_on_prefix, :: policy.train_lattice_stride_on_prefix]
                sampled_pairs = pair_ids[::2, ::2][
                    :: policy.train_lattice_stride_on_prefix,
                    :: policy.train_lattice_stride_on_prefix,
                ]
                _save_train_target(
                    target_path,
                    assignment=assignment,
                    support=sampled_support,
                    pair_ids=sampled_pairs,
                    frame_sha256=array_sha256(frame.detach().cpu().numpy()),
                    prefix_sha256=array_sha256(prefix.detach().cpu().numpy()),
                    costate_sha256=array_sha256(costate.detach().cpu().numpy()),
                    exact_mass_square=float(mass.sum(dtype=np.float64)),
                    metrics=metrics,
                    elapsed=elapsed,
                )
                del costate, mass, support
            global_x, block_x, pair_rows = support_feature_matrices(
                prefix.detach().cpu().numpy(),
                label,
                margin,
                sampled_pairs,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.train_lattice_stride_on_prefix,
            )
            support_rows = sampled_support.reshape(-1)
            if not np.array_equal(pair_rows, sampled_pairs.reshape(-1)):
                raise ProbeError("cached and reconstructed pair rows disagree")
            global_record = weighted_topk_statistics(global_x, support_rows)
            weighted_records = weighted_topk_block_statistics(block_x, pair_rows, support_rows)
            pairwise_records = pairwise_rank_block_statistics(block_x, pair_rows, support_rows)
            _add_record(
                accumulator,
                assignment=assignment,
                global_record=global_record,
                weighted_records=weighted_records,
                pairwise_records=pairwise_records,
            )
            _save_accumulator(accumulator_path, accumulator)
            completed.add(assignment.pair_index)
            round2._append_jsonl(
                ledger,
                {
                    "event": "round4_training_state_checkpointed",
                    "timestamp_utc": _utc_now(),
                    "pair_index": assignment.pair_index,
                    "checkpoint_name": assignment.checkpoint_name,
                    "target_sha256": _sha256(target_path),
                    "accumulator_sha256": _sha256(accumulator_path),
                },
            )
        preserved = output_dir / "stage_checkpoints" / f"train_{checkpoint_name}_complete.npz"
        if not preserved.exists():
            _atomic_npz(preserved, **accumulator)
        stage_checkpoints.append(
            {
                "checkpoint_name": checkpoint_name,
                "path": str(preserved.relative_to(output_dir)),
                "bytes": preserved.stat().st_size,
                "sha256": _sha256(preserved),
                "completed_train_states_cumulative": int(accumulator["completed_pairs"].size),
            }
        )
    if len(completed) != len(train):
        raise ProbeError("training stage did not cover every registered state")
    records = {
        str(row.pair_index): {
            "path": str(_train_target_path(output_dir, row.pair_index).relative_to(output_dir)),
            "bytes": _train_target_path(output_dir, row.pair_index).stat().st_size,
            "sha256": _sha256(_train_target_path(output_dir, row.pair_index)),
        }
        for row in train
    }
    stage = {
        "schema": "replace_round4_train_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(train),
        "raw_exact_costates_preserved": False,
        "compact_targets": records,
        "accumulator": {
            "path": str(accumulator_path.relative_to(output_dir)),
            "bytes": accumulator_path.stat().st_size,
            "sha256": _sha256(accumulator_path),
        },
        "preserved_stage_checkpoints": stage_checkpoints,
        "checkpoint_parity": parity_rows,
        "prefix_cost_model": {
            "path": str(cost_path.relative_to(output_dir)),
            "bytes": cost_path.stat().st_size,
            "sha256": _sha256(cost_path),
        },
    }
    _atomic_json(stage_path, stage)
    return accumulator, stage


def _fit_stage(
    output_dir: Path, accumulator: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, ConvexScalarFit], dict[str, Any]]:
    stage_path = output_dir / "stage_fit_complete.json"
    weights_path = output_dir / "fit" / "round4_weights.npz"
    global_fit = fit_exact_quadratic(_stats_from_accumulator(accumulator, "global"))
    weighted_fits = tuple(
        fit_exact_quadratic(_stats_from_accumulator(accumulator, "weighted", block))
        for block in range(ORDERED_PAIR_COUNT)
    )
    pairwise_fits = tuple(
        fit_exact_quadratic(_stats_from_accumulator(accumulator, "pairwise", block))
        for block in range(ORDERED_PAIR_COUNT)
    )
    weights = {
        "global": global_fit.weights,
        "weighted_blocks": np.stack([fit.weights for fit in weighted_fits]),
        "pairwise_blocks": np.stack([fit.weights for fit in pairwise_fits]),
    }
    if not weights_path.exists():
        _atomic_npz(weights_path, **weights)
    else:
        with np.load(weights_path, allow_pickle=False) as archive:
            for name, value in weights.items():
                if not np.array_equal(archive[name], value):
                    raise ProbeError("fit weights drifted on resume")
    fits = {
        "weighted-topk-global-84": global_fit,
        "weighted-topk-pair-block-44": ConvexScalarFit(
            weights=weights["weighted_blocks"],
            certificate={"blocks": [fit.certificate for fit in weighted_fits]},
        ),
        "pairwise-rank-pair-block-44": ConvexScalarFit(
            weights=weights["pairwise_blocks"],
            certificate={"blocks": [fit.certificate for fit in pairwise_fits]},
        ),
    }
    stage = {
        "schema": "replace_round4_fit_stage.v1",
        "completed_at_utc": _utc_now(),
        "exact_optimum_class": "float64 convex quadratic normal equations",
        "rungs": {name: fit.certificate for name, fit in fits.items()},
        "weights": {
            "path": str(weights_path.relative_to(output_dir)),
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
            "arrays": {name: array_sha256(value) for name, value in weights.items()},
        },
    }
    _atomic_json(stage_path, stage)
    return weights, fits, stage


def _calibrator_from_dict(row: dict[str, Any]) -> IsotonicCalibrator:
    return IsotonicCalibrator(
        x=np.asarray(row["x"], dtype=np.float64),
        probability=np.asarray(row["probability"], dtype=np.float64),
        valid=bool(row["valid"]),
        reason=str(row["reason"]),
        sample_count=int(row["sample_count"]),
        positive_count=int(row["positive_count"]),
    )


def _calibration_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound4SupportRankingPolicy,
    weights: dict[str, np.ndarray],
    segnet: Any,
    yopo: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_path = output_dir / "stage_calibration_complete.json"
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text())
        objects = {}
        for rung, row in stage["calibrators"].items():
            objects[rung] = {
                "global": _calibrator_from_dict(row["global"]),
                "blocks": tuple(_calibrator_from_dict(block) for block in row.get("blocks", [])),
            }
        return objects, stage
    all_scores = {name: [] for name in policy.rung_order}
    all_support = []
    all_pairs = []
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        cohort = [
            row for row in assignments if row.split == "train" and row.checkpoint_index == checkpoint_index
        ]
        for assignment in cohort:
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            prefix = local_prefix_feature_snapshot(segnet, frame)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            support, pair_grid = _load_train_target(
                _train_target_path(output_dir, assignment.pair_index), assignment
            )
            global_x, block_x, pair_rows = support_feature_matrices(
                prefix.detach().cpu().numpy(),
                label,
                margin,
                pair_grid,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.train_lattice_stride_on_prefix,
            )
            all_scores[policy.rung_order[0]].append(global_scores(global_x, weights["global"]))
            all_scores[policy.rung_order[1]].append(
                block_scores(block_x, pair_rows, weights["weighted_blocks"])
            )
            all_scores[policy.rung_order[2]].append(
                block_scores(block_x, pair_rows, weights["pairwise_blocks"])
            )
            all_support.append(support.reshape(-1))
            all_pairs.append(pair_rows)
    support_rows = np.concatenate(all_support)
    pair_rows = np.concatenate(all_pairs)
    objects: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    for rung in policy.rung_order:
        raw = np.concatenate(all_scores[rung])
        if rung == policy.rung_order[0]:
            global_cal = fit_isotonic_calibrator(
                raw, support_rows, bin_count=policy.calibration_bin_count
            )
            blocks: tuple[IsotonicCalibrator, ...] = ()
        else:
            global_cal, blocks = fit_block_calibrators(
                raw, support_rows, pair_rows, bin_count=policy.calibration_bin_count
            )
        objects[rung] = {"global": global_cal, "blocks": blocks}
        payload[rung] = {
            "global": global_cal.to_dict(),
            "blocks": [block.to_dict() for block in blocks],
            "raw_score_sha256": array_sha256(raw),
        }
    stage = {
        "schema": "replace_round4_calibration_stage.v1",
        "completed_at_utc": _utc_now(),
        "fit_split": "train",
        "state_count": policy.train_state_count,
        "row_count": int(support_rows.size),
        "support_prevalence": float(support_rows.mean()),
        "method": "16-quantile Jeffreys bins plus monotone PAV and linear interpolation",
        "calibrators": payload,
    }
    _atomic_json(stage_path, stage)
    return objects, stage


def _calibration_sufficient(probability: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    prob = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(support, dtype=np.bool_).reshape(-1)
    bins = np.minimum((prob * 10.0).astype(np.int64), 9)
    return {
        "count": np.bincount(bins, minlength=10).astype(int).tolist(),
        "probability_sum": np.bincount(bins, weights=prob, minlength=10).tolist(),
        "support_sum": np.bincount(bins, weights=y.astype(np.float64), minlength=10).tolist(),
        "squared_error_sum": float(np.square(prob - y.astype(np.float64)).sum()),
        "sample_count": int(prob.size),
        "support_count": int(y.sum()),
    }


def _pair_mass_rows(
    mask: np.ndarray, pair_rows: np.ndarray, mass: np.ndarray
) -> list[dict[str, Any]]:
    flat_mask = mask.reshape(-1)
    flat_pair = pair_rows.reshape(-1)
    flat_mass = mass.reshape(-1)
    result = []
    for block in range(ORDERED_PAIR_COUNT):
        source, competitor = pair_id_to_classes(block)
        pair_mask = flat_pair == block
        result.append(
            {
                "pair_id": block,
                "source_class": source,
                "competitor_class": competitor,
                "cell_count": int(pair_mask.sum()),
                "selected_count": int(np.sum(pair_mask & flat_mask)),
                "selected_exact_mass_square": float(flat_mass[pair_mask & flat_mask].sum()),
                "total_exact_mass_square": float(flat_mass[pair_mask].sum()),
            }
        )
    return result


def _heldout_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound4SupportRankingPolicy,
    weights: dict[str, np.ndarray],
    calibrators: dict[str, Any],
    segnet: Any,
    yopo: Any,
) -> dict[str, Any]:
    stage_path = output_dir / "stage_heldout_complete.json"
    if stage_path.is_file():
        return json.loads(stage_path.read_text())
    ledger = output_dir / "teacher_calls.jsonl"
    heldout = [row for row in assignments if row.split == "heldout"]
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        cohort = [row for row in heldout if row.checkpoint_index == checkpoint_index]
        for assignment in cohort:
            record_path = _heldout_record_path(output_dir, assignment.pair_index)
            if record_path.is_file():
                continue
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            torch = __import__("torch")
            prefix, costate, pair_ids, teacher_metrics, elapsed, _cost = _teacher_call(
                ledger=ledger,
                assignment=assignment,
                stage="round4_heldout_validation",
                frame_nchw=frame,
                labels_t=torch.as_tensor(label[None], dtype=torch.long),
                segnet=segnet,
                measure_cost=False,
            )
            costate_np = costate.detach().cpu().numpy()
            mass, oracle_support, count = exact_support_target(
                costate_np, area_fraction=policy.requested_area_fraction
            )
            if count != policy.selected_prefix_cells:
                raise ProbeError("heldout target selected-area drift")
            global_x, block_x, pair_rows = support_feature_matrices(
                prefix.detach().cpu().numpy(),
                label,
                margin,
                pair_ids,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.heldout_lattice_stride_on_prefix,
            )
            raw = {
                policy.rung_order[0]: global_scores(global_x, weights["global"]),
                policy.rung_order[1]: block_scores(
                    block_x, pair_rows, weights["weighted_blocks"]
                ),
                policy.rung_order[2]: block_scores(
                    block_x, pair_rows, weights["pairwise_blocks"]
                ),
            }
            probability = {}
            fallback = {}
            masks = {}
            rung_rows = {}
            for rung in policy.rung_order:
                if rung == policy.rung_order[0]:
                    probability[rung] = calibrators[rung]["global"].predict(raw[rung])
                    fallback[rung] = np.zeros(raw[rung].size, dtype=np.bool_)
                else:
                    probability[rung], fallback[rung] = calibrated_block_scores(
                        raw[rung],
                        pair_rows,
                        global_calibrator=calibrators[rung]["global"],
                        block_calibrators=calibrators[rung]["blocks"],
                    )
                masks[rung] = np.zeros(raw[rung].size, dtype=np.bool_)
                selected = np.lexsort(
                    (np.arange(raw[rung].size, dtype=np.int64), -probability[rung])
                )[:count]
                masks[rung][selected] = True
                metrics = mass_localization_metrics(
                    costate_np,
                    probability[rung].reshape(mass.shape),
                    area_fraction=policy.requested_area_fraction,
                )
                rung_rows[rung] = {
                    "mass_localization": metrics,
                    "calibration_sufficient": _calibration_sufficient(
                        probability[rung], oracle_support
                    ),
                    "selected_calibration_fallback_count": int(fallback[rung][masks[rung]].sum()),
                    "pair_mass": _pair_mass_rows(masks[rung], pair_rows, mass),
                }
            overlap = {}
            for left_index, left in enumerate(policy.rung_order):
                for right in policy.rung_order[left_index + 1 :]:
                    intersection = int(np.sum(masks[left] & masks[right]))
                    union = int(np.sum(masks[left] | masks[right]))
                    overlap[f"{left}__{right}"] = {
                        "intersection": intersection,
                        "union": union,
                        "jaccard": intersection / union,
                    }
            consensus_count = sum(masks[rung].astype(np.int8) for rung in policy.rung_order)
            row = {
                "schema": "replace_round4_heldout_state.v1",
                "completed_at_utc": _utc_now(),
                "assignment": assignment.to_dict(),
                "teacher_metrics": teacher_metrics,
                "teacher_elapsed_seconds": elapsed,
                "frame_sha256": array_sha256(frame.detach().cpu().numpy()),
                "prefix_sha256": array_sha256(prefix.detach().cpu().numpy()),
                "exact_costate_sha256": array_sha256(costate_np),
                "rungs": rung_rows,
                "selection_overlap": overlap,
                "cells_selected_by_at_least_two_rungs": int(np.sum(consensus_count >= 2)),
                "cells_selected_by_all_three_rungs": int(np.sum(consensus_count == 3)),
                "authority": AXIS,
            }
            _atomic_json(record_path, row)
            del costate, costate_np, mass, oracle_support, prefix
    rows = [
        json.loads(_heldout_record_path(output_dir, row.pair_index).read_text()) for row in heldout
    ]
    rungs = []
    for rung in policy.rung_order:
        mass_summary = aggregate_mass_localization(
            [row["rungs"][rung]["mass_localization"] for row in rows]
        )
        sufficient = [row["rungs"][rung]["calibration_sufficient"] for row in rows]
        count = np.sum([row["count"] for row in sufficient], axis=0)
        prob_sum = np.sum([row["probability_sum"] for row in sufficient], axis=0)
        support_sum = np.sum([row["support_sum"] for row in sufficient], axis=0)
        total_samples = int(sum(row["sample_count"] for row in sufficient))
        reliability = []
        ece = 0.0
        for index in range(10):
            if count[index] == 0:
                continue
            mean_probability = float(prob_sum[index] / count[index])
            observed = float(support_sum[index] / count[index])
            gap = abs(mean_probability - observed)
            ece += float(count[index] / total_samples) * gap
            reliability.append(
                {
                    "bin": index,
                    "count": int(count[index]),
                    "mean_probability": mean_probability,
                    "observed_support_rate": observed,
                    "absolute_gap": gap,
                }
            )
        rungs.append(
            {
                "rung": rung,
                "mass_localization": mass_summary,
                "calibration": {
                    "sample_count": total_samples,
                    "support_prevalence": sum(row["support_count"] for row in sufficient)
                    / total_samples,
                    "expected_calibration_error_10bin": ece,
                    "brier_score": sum(row["squared_error_sum"] for row in sufficient)
                    / total_samples,
                    "reliability": reliability,
                },
                "selected_calibration_fallback_count": sum(
                    int(row["rungs"][rung]["selected_calibration_fallback_count"])
                    for row in rows
                ),
            }
        )
    overlap_keys = rows[0]["selection_overlap"]
    overlap = {}
    for key in overlap_keys:
        intersection = sum(int(row["selection_overlap"][key]["intersection"]) for row in rows)
        union = sum(int(row["selection_overlap"][key]["union"]) for row in rows)
        overlap[key] = {"intersection": intersection, "union": union, "jaccard": intersection / union}
    stage = {
        "schema": "replace_round4_heldout_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(rows),
        "rungs": rungs,
        "selection_overlap": overlap,
        "cells_selected_by_at_least_two_rungs": sum(
            int(row["cells_selected_by_at_least_two_rungs"]) for row in rows
        ),
        "cells_selected_by_all_three_rungs": sum(
            int(row["cells_selected_by_all_three_rungs"]) for row in rows
        ),
        "records": {
            str(row["assignment"]["pair_index"]): {
                "path": str(
                    _heldout_record_path(output_dir, int(row["assignment"]["pair_index"])).relative_to(
                        output_dir
                    )
                ),
                "sha256": _sha256(
                    _heldout_record_path(output_dir, int(row["assignment"]["pair_index"]))
                ),
            }
            for row in rows
        },
    }
    _atomic_json(stage_path, stage)
    return stage


def _teacher_accounting(
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    policy: ReplaceRound4SupportRankingPolicy,
) -> dict[str, Any]:
    ledger = output_dir / "teacher_calls.jsonl"
    rows = round2._canonicalize_event_ledger(ledger)
    starts = [row for row in rows if row["event"] == "exact_teacher_state_call_started"]
    completions = [row for row in rows if row["event"] == "exact_teacher_state_call_completed"]
    required = {(row.split, row.pair_index) for row in assignments}
    completed = {(row["split"], int(row["pair_index"])) for row in completions}
    if not required.issubset(completed):
        raise ProbeError("teacher ledger does not cover every registered round-4 state")
    round3 = json.loads(ROUND3_RECEIPT.read_text())["teacher_call_accounting"]
    return {
        "teacher_call_unit": "one batch-size-1 exact labeled SegNet forward plus input backward",
        "round4_started_calls_campaign_charged": len(starts),
        "round4_completed_call_events": len(completions),
        "round4_unique_states_completed": len(completed),
        "round4_required_unique_states": len(required),
        "round4_train_started_calls": sum(row["split"] == "train" for row in starts),
        "round4_heldout_started_calls": sum(row["split"] == "heldout" for row in starts),
        "round4_retries_charged": max(0, len(starts) - len(required)),
        "round3_lineage_started_calls": int(round3["all_exact_labeled_calls_observed_conservative"]),
        "round3_plus_round4_campaign_calls": int(
            round3["all_exact_labeled_calls_observed_conservative"] + len(starts)
        ),
        "conditional_composed_label_coefficient": policy.conditional_composed_label_coefficient,
        "conditional_variable_cost_reduction_x": policy.conditional_variable_cost_reduction_x,
        "current_exact_teacher_wall_cost_remains_dense": True,
        "teacher_ledger": {
            "path": str(ledger.relative_to(output_dir)),
            "bytes": ledger.stat().st_size,
            "sha256": _sha256(ledger),
            "rows": len(rows),
        },
    }


def _decision_stage(
    *, output_dir: Path, heldout: dict[str, Any], policy: ReplaceRound4SupportRankingPolicy
) -> dict[str, Any]:
    stage_path = output_dir / "stage_decision_complete.json"
    if stage_path.is_file():
        return json.loads(stage_path.read_text())
    by_order = {name: index for index, name in enumerate(policy.rung_order)}
    winner = max(
        heldout["rungs"],
        key=lambda row: (
            float(row["mass_localization"]["retained_exact_costate_l2_mass_fraction"]),
            -by_order[row["rung"]],
        ),
    )
    retained = float(winner["mass_localization"]["retained_exact_costate_l2_mass_fraction"])
    passed = retained >= policy.retained_mass_bar
    ece = float(winner["calibration"]["expected_calibration_error_10bin"])
    stage = {
        "schema": "replace_round4_decision_stage.v1",
        "completed_at_utc": _utc_now(),
        "verdict": (
            "GO_SUPPORT_LOCALIZATION_RESEARCH_ONLY"
            if passed
            else "NO_GO_SHALLOW_CHEAP_FEATURE_CONVEX_LOCALIZATION"
        ),
        "winner": winner["rung"],
        "winner_retained_mass_fraction": retained,
        "retained_mass_bar": policy.retained_mass_bar,
        "realized_input_area_fraction": winner["mass_localization"][
            "realized_input_area_fraction"
        ],
        "primary_gate_passed": passed,
        "winner_heldout_ece": ece,
        "calibration_live_guard_passed": ece <= policy.heldout_ece_refusal_bar,
        "finite_rung_order": list(policy.rung_order),
        "stop_rule_honored": True,
        "pointer_delta": "NONE",
        "family_verdict": (
            "FAMILY_OPEN__SHALLOW_CONVEX_LOCALIZER_PASSES"
            if passed
            else "FAMILY_LEVEL_NEGATIVE_SIGNAL__SHALLOW_PRESE_CHEAP_FEATURE_CONVEX_LOCALIZERS_ONLY"
        ),
        "verdict_scope": (
            "FAMILY x FIXED REPLAY: shallow pre-SE cheap-feature convex support localizers, "
            "n600 seed455 three-stage replay; deeper/nonconvex/on-policy families remain open"
        ),
    }
    _atomic_json(stage_path, stage)
    return stage


def _cleanup_manifest(output_dir: Path, run_contract: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "cleanup_manifest.json"
    if path.is_file():
        return json.loads(path.read_text())
    compact_paths = sorted((output_dir / "train_targets").glob("*.npz"))
    manifest = {
        "schema": "replace_round4_cleanup_manifest.v1",
        "completed_at_utc": _utc_now(),
        "policy": "exact costates are success-only in-memory scratch; persist compact targets and convex sufficient statistics",
        "raw_exact_costate_files_created": 0,
        "raw_exact_costate_bytes_deleted": 0,
        "compact_target_file_count": len(compact_paths),
        "compact_target_bytes": sum(item.stat().st_size for item in compact_paths),
        "compact_target_tree_sha256": hashlib.sha256(
            "".join(f"{item.relative_to(output_dir)}:{_sha256(item)}\n" for item in compact_paths).encode()
        ).hexdigest(),
        "preserved": [
            "train_targets",
            "train_accumulator_current.npz",
            "stage_checkpoints",
            "fit",
            "heldout",
            "teacher_calls.jsonl",
            "source_bundle",
        ],
        "deterministic_rebuild_command": (
            "PYTHONPATH=src .venv/bin/python tools/probe_replace_round4_support_ranking.py --resume"
        ),
        "run_contract_sha256": _sha256(output_dir / "run_contract.json"),
        "source_bundle_tree_identity": run_contract.get(
            "effective_sources", run_contract["sources"]
        ),
        "cold_store_destination": None,
        "reason": "all preserved evidence is compact; no bulky rebuildable scratch remains",
        "false_authority_score_flags": {"score_claim": False, "promotion_eligible": False},
        "blockers": [],
    }
    _atomic_json(path, manifest)
    return manifest


def _complete_receipt_or_none(output_dir: Path, *, resume: bool) -> dict[str, Any] | None:
    complete_path = output_dir / "complete.json"
    if not complete_path.exists():
        return None
    if not resume:
        raise ProbeError("completed result exists; pass --resume to verify it")
    complete = json.loads(complete_path.read_text())
    receipt_path = output_dir / complete["receipt"]
    if receipt_path.stat().st_size != complete["bytes"] or _sha256(receipt_path) != complete["sha256"]:
        raise ProbeError("completed receipt custody drift")
    return json.loads(receipt_path.read_text())


def run(
    *,
    output_dir: Path,
    resume: bool,
    validate_only: bool = False,
    source_amendment: str | None = None,
) -> dict[str, Any]:
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ProbeError("the preregistered instance has one sealed output directory")
    completed = _complete_receipt_or_none(output_dir, resume=resume)
    if completed is not None:
        return completed
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = ReplaceRound4SupportRankingPolicy()
    contract = policy.compile_measurement_contract()
    prereg = _validate_preregistration(policy)
    storage = _storage_custody(output_dir)
    inputs = round2._verify_input_custody()
    if validate_only:
        return {
            "schema": "replace_round4_validate_only.v1",
            "compiled_policy": contract,
            "preregistration": prereg,
            "storage": storage,
            "inputs": inputs,
            "sources": _source_fingerprints(),
            "teacher_calls": 0,
        }
    assignments = deterministic_replay_assignments(
        n_pairs=policy.n_pairs,
        checkpoint_names=tuple(row[0] for row in round2.CHECKPOINTS),
        holdout_period=policy.holdout_period,
        seed=policy.seed,
    )
    descriptor = round2._acquire_lock(output_dir)
    try:
        completed = _complete_receipt_or_none(output_dir, resume=resume)
        if completed is not None:
            return completed
        contract_path = output_dir / "run_contract.json"
        if contract_path.is_file():
            if not resume:
                raise ProbeError("existing run contract requires --resume")
            run_contract = json.loads(contract_path.read_text())
            if run_contract["compiled_policy"] != _json_normalized(contract):
                raise ProbeError("resume policy drift")
            sources, amendment_custody = _resolve_source_custody(
                output_dir,
                prior_contract=run_contract,
                requested_amendment=source_amendment,
            )
            if run_contract["inputs"] != inputs:
                raise ProbeError("resume input custody drift")
            if run_contract["storage_preflight"] != storage:
                raise ProbeError("resume storage custody drift")
            if amendment_custody is not None:
                prior_effective = run_contract.get("effective_sources")
                amendments = list(run_contract.get("source_amendments", []))
                if amendment_custody in amendments and prior_effective != sources:
                    raise ProbeError("resume effective-source custody drift")
                run_contract["effective_sources"] = sources
                if amendment_custody not in amendments:
                    amendments.append(amendment_custody)
                run_contract["source_amendments"] = amendments
                _atomic_json(contract_path, run_contract)
        else:
            if source_amendment is not None:
                raise ProbeError("source amendment is only valid at a sealed resume boundary")
            sources = _source_bundle(output_dir)
            run_contract = {
                "schema": "replace_round4_run_contract.v1",
                "created_at_utc": _utc_now(),
                "lane_id": LANE_ID,
                "compiled_policy": contract,
                "preregistration": prereg,
                "inputs": inputs,
                "sources": sources,
                "runtime": _runtime_custody(torch),
                "storage_preflight": storage,
                "git_head_at_measurement": _git_head(),
                "git_status_at_measurement_start": _git_status(),
                "source_runs_read_only": True,
                "paid_or_heavy_launch": False,
                "authority": AXIS,
            }
            _atomic_json(contract_path, run_contract)
        labels = round2._stored_npy_memmap(round2.GT_CACHE, "lstars.npy")
        margins = round2._stored_npy_memmap(round2.GT_CACHE, "margins.npy")
        if labels.shape != (600, 384, 512) or margins.shape != labels.shape:
            raise ProbeError("GT cache geometry drift")
        yopo = round2._load_tool_module(
            "_round4_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
        )
        segnet = round2._load_cpu_segnet()
        accumulator, train_stage = _training_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        weights, fits, fit_stage = _fit_stage(output_dir, accumulator)
        calibrators, calibration_stage = _calibration_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            weights=weights,
            segnet=segnet,
            yopo=yopo,
        )
        heldout_stage = _heldout_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            weights=weights,
            calibrators=calibrators,
            segnet=segnet,
            yopo=yopo,
        )
        decision = _decision_stage(output_dir=output_dir, heldout=heldout_stage, policy=policy)
        policy_sha = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        fore = check_fore_support(
            (), target_policy_sha256=policy_sha, target_arms=policy.rung_order
        )
        accounting = _teacher_accounting(output_dir, assignments, policy)
        cleanup = _cleanup_manifest(output_dir, run_contract)
        cleanup_path = output_dir / "cleanup_manifest.json"
        receipt = {
            "schema": SCHEMA,
            "completed_at_utc": _utc_now(),
            "verdict": decision,
            "authority": {
                "axis": AXIS,
                "module_scope": AUTHORITY_SCOPE,
                "research_only": True,
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "MPS_used": False,
            },
            "run_contract": run_contract,
            "train_stage": train_stage,
            "fit_stage": fit_stage,
            "calibration_stage": calibration_stage,
            "heldout_stage": heldout_stage,
            "teacher_call_accounting": accounting,
            "economics": {
                "form": "C_teacher = A + c_label * D",
                "prefix_fraction": contract["economics"]["prefix_fraction"],
                "selected_area_fraction": contract["realized_area_fraction"],
                "conditional_composed_label_coefficient": policy.conditional_composed_label_coefficient,
                "conditional_variable_cost_reduction_x": policy.conditional_variable_cost_reduction_x,
                "status": "DERIVED_CONDITIONAL_NOT_REALIZED_WALL_CLOCK",
                "blocker": (
                    "current exact EfficientNet-B2 teacher has global squeeze-excite dependence "
                    "and no sparse exact-kernel receipt"
                ),
                "campaign_retries_charged": accounting["round4_retries_charged"],
            },
            "FORE_composition": {**asdict(fore), "weights_applied": False},
            "query_policy": {
                "status": "REFUSE_LIVE__RESEARCH_ONLY_FIXED_REPLAY",
                "trust": "winner-selected plus at least two-of-three selector agreement and no calibration fallback",
                "query": "winner-selected with selector disagreement or calibration fallback",
                "refuse": "FORE not admissible, ECE guard fails, or custody drifts",
                "ticket": "DIG-S1-QUERY-REAL-CALIBRATION",
            },
            "cleanup_custody": {
                "path": str(cleanup_path.relative_to(output_dir)),
                "bytes": cleanup_path.stat().st_size,
                "sha256": _sha256(cleanup_path),
                "blockers": cleanup["blockers"],
            },
            "triality": {
                "dsl": "tac.witness_dsl.replace_round4_support_ranking_policy",
                "equation": "tac.canonical_equations.replace_round4_support_ranking_20260713",
                "dag_feed": ".omx/research/replace_round4_support_ranking_DAG_FEED_20260713.md",
            },
            "verdict_scope": decision["verdict_scope"],
            "reformulation_queue": [
                "queue-4 deeper local pre-SE features with a separately measured cost fraction",
                "queue-5 transition-complete FORE and disagreement-audited on-policy queries",
                "queue-6 dense-label or nonlinear support learner under the same heldout gate",
            ],
            "pointer_delta": "NONE",
        }
        receipt_path = output_dir / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _atomic_json(
            output_dir / "complete.json",
            {
                "schema": "replace_round4_completion.v1",
                "completed_at_utc": _utc_now(),
                "receipt": str(receipt_path.relative_to(output_dir)),
                "bytes": receipt_path.stat().st_size,
                "sha256": _sha256(receipt_path),
                "verdict": decision["verdict"],
            },
        )
        return receipt
    finally:
        round2._release_lock(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--source-amendment", choices=(SOURCE_AMENDMENT_ID,))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if REPO not in output_dir.parents:
        raise ProbeError("durable evidence must remain under the repository")
    receipt = run(
        output_dir=output_dir,
        resume=args.resume,
        validate_only=args.validate_only,
        source_amendment=args.source_amendment,
    )
    if args.validate_only:
        print(json.dumps({"schema": receipt["schema"], "teacher_calls": 0}, sort_keys=True))
    else:
        decision = receipt["verdict"]
        print(
            json.dumps(
                {
                    "verdict": decision["verdict"],
                    "winner": decision["winner"],
                    "retained_mass_fraction": decision["winner_retained_mass_fraction"],
                    "retained_mass_bar": decision["retained_mass_bar"],
                    "receipt": str(output_dir / "receipt.json"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
