#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable $0 local probe for REPLACE round-5 deeper/nonlinear localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
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
from tac.scorer_surrogate.replace_round4_support_ranking import (  # noqa: E402
    ORDERED_PAIR_COUNT,
    QuadraticStatistics,
    array_sha256,
    block_scores,
    exact_support_target,
    fit_exact_quadratic,
    pairwise_rank_block_statistics,
)
from tac.scorer_surrogate.replace_round5_deeper_nonlinear import (  # noqa: E402
    AUTHORITY_SCOPE,
    DEEP_FEATURE_COUNT,
    SCHEMA,
    DeepCutCostLedger,
    PairGatedMLPWeights,
    capture_round5_teacher,
    deep_feature_snapshot,
    deeper_pair_block_features,
    disagreement_query_audit,
    pair_gated_logits_numpy,
    sigmoid_probabilities,
)
from tac.witness_dsl.replace_round5_deeper_nonlinear_policy import (  # noqa: E402
    ReplaceRound5DeeperNonlinearPolicy,
)

LANE_ID = "lane_replace_round5_deeper_nonlinear_20260713"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex fit; CPU-Torch nonlinear]"
DEFAULT_OUTPUT = REPO / "experiments/results/replace_round5_deeper_nonlinear_20260713"
PREREGISTRATION = DEFAULT_OUTPUT / "preregistration.json"
STORAGE_PREFLIGHT = (
    REPO / ".omx/research/replace_round5_deeper_nonlinear_storage_preflight_20260713.json"
)

SOURCE_FILES = (
    "src/tac/scorer_surrogate/replace_round5_deeper_nonlinear.py",
    "src/tac/witness_dsl/replace_round5_deeper_nonlinear_policy.py",
    "tools/probe_replace_round5_deeper_nonlinear.py",
    "src/tac/scorer_surrogate/replace_round4_support_ranking.py",
    "src/tac/scorer_surrogate/replace_round3_fidelity_wall.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_yopo_first_layer_costate.py",
    "src/tac/causal_manifest.py",
)


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


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
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


def _source_bundle(output_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for relative, custody in _source_fingerprints().items():
        source = REPO / relative
        destination = output_dir / "source_bundle" / relative
        if destination.exists():
            if destination.stat().st_size != custody["bytes"] or _sha256(destination) != custody[
                "sha256"
            ]:
                raise ProbeError(f"source bundle drift for {relative}")
        else:
            _atomic_bytes(destination, source.read_bytes())
        result[relative] = {
            **custody,
            "path": str(destination.relative_to(output_dir)),
        }
    return result


def _write_preregistration(policy: ReplaceRound5DeeperNonlinearPolicy) -> dict[str, Any]:
    payload = {
        "schema": "replace_round5_preregistration.v1",
        "sealed_at_utc": _utc_now(),
        "sealed_before_any_round5_teacher_call": True,
        "measurement_contract": policy.compile_measurement_contract(),
        "rungs": [
            {
                "name": "convex-deeper-pair-block-mp",
                "target": "exact top-area input-costate L2-square support",
                "features": "base-42 plus bilinear block2-24 plus block3-48 plus sensitivity-2",
                "fit": "twenty exact float64 pair-block RankRLS Moore-Penrose optima",
            },
            {
                "name": "nonlinear-pair-gated-mlp-ensemble",
                "target": "same exact top-area support",
                "features": "same sealed 116-column deeper chart",
                "fit": "three deterministic 116-to-32-to-20 pair-gated ReLU MLP seeds",
            },
        ],
        "no_post_heldout_rung": True,
        "number_labels": "MEASURED DERIVED INFERRED ASSUMED",
        "verdict_ladder": "INSTANCE < FORMULATION < FAMILY < PARADIGM",
    }
    if PREREGISTRATION.exists():
        existing = json.loads(PREREGISTRATION.read_text())
        existing_without_time = {k: v for k, v in existing.items() if k != "sealed_at_utc"}
        payload_without_time = {k: v for k, v in payload.items() if k != "sealed_at_utc"}
        if existing_without_time != payload_without_time:
            raise ProbeError("existing preregistration differs from the typed policy")
        return existing
    _atomic_json(PREREGISTRATION, payload)
    return payload


def _validate_preregistration(policy: ReplaceRound5DeeperNonlinearPolicy) -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise ProbeError("round-5 preregistration is missing")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload["measurement_contract"] != _json_normalized(policy.compile_measurement_contract()):
        raise ProbeError("typed policy and preregistered contract disagree")
    ledger = DEFAULT_OUTPUT / "teacher_calls.jsonl"
    if (
        ledger.exists()
        and any(
        json.loads(line).get("event") == "exact_teacher_state_call_started"
        for line in ledger.read_text().splitlines()
        if line.strip()
        )
        and payload.get("sealed_before_any_round5_teacher_call") is not True
    ):
        raise ProbeError("preregistration did not precede teacher calls")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO)),
        "bytes": PREREGISTRATION.stat().st_size,
        "sha256": _sha256(PREREGISTRATION),
        "payload": payload,
    }


def _storage_custody(output_dir: Path) -> dict[str, Any]:
    if not STORAGE_PREFLIGHT.is_file():
        raise ProbeError("missing round-5 storage waterfall preflight")
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


def _target_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "train_targets" / f"pair_{pair_index:04d}.npz"


def _heldout_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout" / f"pair_{pair_index:04d}.json"


def _empty_accumulator() -> dict[str, np.ndarray]:
    return {
        "completed_pairs": np.empty(0, dtype=np.int64),
        "gram": np.zeros(
            (ORDERED_PAIR_COUNT, DEEP_FEATURE_COUNT, DEEP_FEATURE_COUNT), dtype=np.float64
        ),
        "rhs": np.zeros((ORDERED_PAIR_COUNT, DEEP_FEATURE_COUNT), dtype=np.float64),
        "target_square": np.zeros(ORDERED_PAIR_COUNT, dtype=np.float64),
        "row_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
        "state_count": np.zeros(ORDERED_PAIR_COUNT, dtype=np.int64),
    }


def _load_accumulator(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return _empty_accumulator()
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = _empty_accumulator()
    if result.keys() != expected.keys() or any(
        key != "completed_pairs" and result[key].shape != expected[key].shape
        for key in expected
    ):
        raise ProbeError("round-5 convex accumulator schema drift")
    completed = result["completed_pairs"]
    if completed.dtype != np.int64 or not np.array_equal(completed, np.unique(completed)):
        raise ProbeError("round-5 completed-pair index drift")
    return result


def _stats(accumulator: dict[str, np.ndarray], block: int) -> QuadraticStatistics:
    return QuadraticStatistics(
        gram=accumulator["gram"][block],
        rhs=accumulator["rhs"][block],
        target_square=float(accumulator["target_square"][block]),
        row_count=int(accumulator["row_count"][block]),
        state_count=int(accumulator["state_count"][block]),
    )


def _add_pairwise(
    accumulator: dict[str, np.ndarray],
    assignment: ReplayAssignment,
    records: Sequence[QuadraticStatistics],
) -> None:
    for block, record in enumerate(records):
        accumulator["gram"][block] += record.gram
        accumulator["rhs"][block] += record.rhs
        accumulator["target_square"][block] += record.target_square
        accumulator["row_count"][block] += record.row_count
        accumulator["state_count"][block] += record.state_count
    accumulator["completed_pairs"] = np.sort(
        np.append(accumulator["completed_pairs"], assignment.pair_index).astype(np.int64)
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
) -> tuple[Any, Any, Any, Any, np.ndarray, dict[str, float], float, dict[str, Any] | None]:
    batch_id = f"{stage}-p{assignment.pair_index:04d}-{time.time_ns()}"
    round2._teacher_start(ledger, assignment, stage=stage, batch_id=batch_id)
    cost_model = None
    if measure_cost:
        with DeepCutCostLedger(segnet) as cost_ledger:
            result = capture_round5_teacher(
                segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
            )
        cost_model = cost_ledger.summary()
    else:
        result = capture_round5_teacher(segnet=segnet, frame_nchw=frame_nchw, labels=labels_t)
    prefix, block2, block3, costate, pair_ids, metrics, elapsed = result
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
    return prefix, block2, block3, costate, pair_ids, metrics, elapsed, cost_model


def _save_target(
    path: Path,
    *,
    assignment: ReplayAssignment,
    support: np.ndarray,
    mass: np.ndarray,
    pair_ids: np.ndarray,
    feature_shas: dict[str, str],
    costate_sha: str,
) -> None:
    _atomic_npz(
        path,
        pair_index=np.asarray(assignment.pair_index, dtype=np.int64),
        checkpoint_index=np.asarray(assignment.checkpoint_index, dtype=np.int64),
        checkpoint_name=np.asarray(assignment.checkpoint_name),
        split=np.asarray(assignment.split),
        support=np.asarray(support, dtype=np.bool_),
        mass=np.asarray(mass, dtype=np.float32),
        pair_ids=np.asarray(pair_ids, dtype=np.int16),
        prefix_sha256=np.asarray(feature_shas["prefix"]),
        block2_sha256=np.asarray(feature_shas["block2"]),
        block3_sha256=np.asarray(feature_shas["block3"]),
        costate_sha256=np.asarray(costate_sha),
    )


def _load_target(path: Path, assignment: ReplayAssignment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            int(archive["pair_index"].item()) != assignment.pair_index
            or int(archive["checkpoint_index"].item()) != assignment.checkpoint_index
            or str(archive["checkpoint_name"].item()) != assignment.checkpoint_name
            or str(archive["split"].item()) != "train"
        ):
            raise ProbeError(f"training target assignment drift at pair {assignment.pair_index}")
        support = np.array(archive["support"], dtype=np.bool_, copy=True)
        mass = np.array(archive["mass"], dtype=np.float32, copy=True)
        pair_ids = np.array(archive["pair_ids"], dtype=np.int16, copy=True)
    if support.shape != (48, 64) or mass.shape != support.shape or pair_ids.shape != support.shape:
        raise ProbeError("round-5 compact target geometry drift")
    return support, mass, pair_ids


def _training_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound5DeeperNonlinearPolicy,
    segnet: Any,
    yopo: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stage_path = output_dir / "stage_train_complete.json"
    accumulator_path = output_dir / "train_accumulator_current.npz"
    cost_path = output_dir / "deep_cut_cost_model.json"
    accumulator = _load_accumulator(accumulator_path)
    completed = {int(value) for value in accumulator["completed_pairs"]}
    train = [row for row in assignments if row.split == "train"]
    ledger = output_dir / "teacher_calls.jsonl"
    parity_rows = []
    stage_checkpoints = []
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(
        round2.CHECKPOINTS
    ):
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.n_pairs or code.shape[0] != 2 * policy.n_pairs:
            raise ProbeError(f"checkpoint {checkpoint_name} is not the sealed n600 renderer")
        parity = round2._checkpoint_parity(renderer, checkpoint_index)
        if parity["status"] != "MEASURED_PASS":
            raise ProbeError(f"renderer parity failed for {checkpoint_name}")
        parity_rows.append({"checkpoint_name": checkpoint_name, **parity})
        for assignment in (
            row for row in train if row.checkpoint_index == checkpoint_index
        ):
            target_path = _target_path(output_dir, assignment.pair_index)
            if assignment.pair_index in completed:
                _load_target(target_path, assignment)
                continue
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            if target_path.exists():
                support, _mass, pair_ids = _load_target(target_path, assignment)
                prefix, block2, block3 = deep_feature_snapshot(segnet, frame)
            else:
                torch = __import__("torch")
                prefix, block2, block3, costate, full_pairs, _metrics, _elapsed, cost_model = (
                    _teacher_call(
                        ledger=ledger,
                        assignment=assignment,
                        stage="round5_train_target",
                        frame_nchw=frame,
                        labels_t=torch.as_tensor(label[None], dtype=torch.long),
                        segnet=segnet,
                        measure_cost=not cost_path.exists(),
                    )
                )
                if cost_model is not None:
                    _atomic_json(cost_path, cost_model)
                manual = deep_feature_snapshot(segnet, frame)
                if not all(
                    bool(torch.equal(left, right))
                    for left, right in zip(manual, (prefix, block2, block3), strict=True)
                ):
                    raise ProbeError("hooked and direct post-SE feature snapshots differ")
                mass, full_support, selected = exact_support_target(
                    costate.detach().cpu().numpy(), area_fraction=policy.requested_area_fraction
                )
                if selected != policy.selected_prefix_cells:
                    raise ProbeError("training support area drift")
                support = full_support[:: policy.train_lattice_stride_on_prefix, :: policy.train_lattice_stride_on_prefix]
                sampled_mass = mass[:: policy.train_lattice_stride_on_prefix, :: policy.train_lattice_stride_on_prefix]
                pair_ids = full_pairs[::2, ::2][
                    :: policy.train_lattice_stride_on_prefix,
                    :: policy.train_lattice_stride_on_prefix,
                ]
                _save_target(
                    target_path,
                    assignment=assignment,
                    support=support,
                    mass=sampled_mass,
                    pair_ids=pair_ids,
                    feature_shas={
                        "prefix": array_sha256(prefix.cpu().numpy()),
                        "block2": array_sha256(block2.cpu().numpy()),
                        "block3": array_sha256(block3.cpu().numpy()),
                    },
                    costate_sha=array_sha256(costate.cpu().numpy()),
                )
                del costate, mass, full_support, sampled_mass
            features, reconstructed_pairs = deeper_pair_block_features(
                prefix.cpu().numpy(),
                block2.cpu().numpy(),
                block3.cpu().numpy(),
                label,
                margin,
                pair_ids,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.train_lattice_stride_on_prefix,
            )
            if not np.array_equal(reconstructed_pairs, pair_ids.reshape(-1)):
                raise ProbeError("cached and reconstructed pair rows differ")
            records = pairwise_rank_block_statistics(
                features.astype(np.float64), reconstructed_pairs, support.reshape(-1)
            )
            _add_pairwise(accumulator, assignment, records)
            _atomic_npz(accumulator_path, **accumulator)
            completed.add(assignment.pair_index)
            round2._append_jsonl(
                ledger,
                {
                    "event": "round5_training_state_checkpointed",
                    "timestamp_utc": _utc_now(),
                    "pair_index": assignment.pair_index,
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
    stage = {
        "schema": "replace_round5_train_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(train),
        "compact_target_count": len(train),
        "raw_exact_costates_preserved": False,
        "accumulator": {
            "path": str(accumulator_path.relative_to(output_dir)),
            "bytes": accumulator_path.stat().st_size,
            "sha256": _sha256(accumulator_path),
        },
        "preserved_stage_checkpoints": stage_checkpoints,
        "checkpoint_parity": parity_rows,
        "deep_cut_cost_model": {
            "path": str(cost_path.relative_to(output_dir)),
            "bytes": cost_path.stat().st_size,
            "sha256": _sha256(cost_path),
        },
    }
    _atomic_json(stage_path, stage)
    return accumulator, stage


def _fit_convex_stage(
    output_dir: Path, accumulator: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    stage_path = output_dir / "stage_convex_fit_complete.json"
    fits = tuple(fit_exact_quadratic(_stats(accumulator, block)) for block in range(20))
    weights = np.stack([fit.weights for fit in fits]).astype(np.float64)
    weights_path = output_dir / "fit" / "convex_deeper_weights.npz"
    if weights_path.exists():
        with np.load(weights_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["weights"], weights):
                raise ProbeError("convex deeper weights drifted on resume")
    else:
        _atomic_npz(weights_path, weights=weights)
    certificates = [fit.certificate for fit in fits]
    if not all(row["normal_equation_optimum_certified"] for row in certificates):
        raise ProbeError("a populated convex deeper block lacks its optimum certificate")
    stage = {
        "schema": "replace_round5_convex_fit_stage.v1",
        "completed_at_utc": _utc_now(),
        "exact_optimum_class": "float64 pair-block RankRLS Moore-Penrose",
        "block_certificates": certificates,
        "weights": {
            "path": str(weights_path.relative_to(output_dir)),
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
            "array_sha256": array_sha256(weights),
        },
    }
    _atomic_json(stage_path, stage)
    return weights, stage


def _balanced_core_indices(support: np.ndarray, *, seed: int) -> np.ndarray:
    positives = np.flatnonzero(support)
    negatives = np.flatnonzero(~support)
    if positives.size == 0 or negatives.size < positives.size:
        raise ProbeError("nonlinear balanced core requires positive and negative rows")
    generator = np.random.default_rng(seed)
    chosen_negative = np.sort(generator.choice(negatives, size=positives.size, replace=False))
    return np.sort(np.concatenate((positives, chosen_negative)))


def _build_nonlinear_data(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound5DeeperNonlinearPolicy,
    segnet: Any,
    yopo: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_path = output_dir / "nonlinear" / "train_dev_data.npz"
    stage_path = output_dir / "stage_nonlinear_data_complete.json"
    if data_path.exists() and stage_path.exists():
        with np.load(data_path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}, json.loads(
                stage_path.read_text()
            )
    core_x: list[np.ndarray] = []
    core_pair: list[np.ndarray] = []
    core_y: list[np.ndarray] = []
    dev_x: list[np.ndarray] = []
    dev_pair: list[np.ndarray] = []
    dev_y: list[np.ndarray] = []
    dev_mass: list[np.ndarray] = []
    dev_offsets = [0]
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        for assignment in (
            row
            for row in assignments
            if row.split == "train" and row.checkpoint_index == checkpoint_index
        ):
            support, mass, pair_ids = _load_target(
                _target_path(output_dir, assignment.pair_index), assignment
            )
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            prefix, block2, block3 = deep_feature_snapshot(segnet, frame)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            features, pairs = deeper_pair_block_features(
                prefix.cpu().numpy(),
                block2.cpu().numpy(),
                block3.cpu().numpy(),
                label,
                margin,
                pair_ids,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.train_lattice_stride_on_prefix,
            )
            support_rows = support.reshape(-1)
            if assignment.pair_index % policy.nonlinear_dev_modulus == policy.nonlinear_dev_remainder:
                dev_x.append(features)
                dev_pair.append(pairs)
                dev_y.append(support_rows)
                dev_mass.append(mass.reshape(-1))
                dev_offsets.append(dev_offsets[-1] + features.shape[0])
            else:
                keep = _balanced_core_indices(support_rows, seed=policy.seed + assignment.pair_index)
                core_x.append(features[keep])
                core_pair.append(pairs[keep])
                core_y.append(support_rows[keep])
    data = {
        "core_x": np.concatenate(core_x).astype(np.float32),
        "core_pair": np.concatenate(core_pair).astype(np.int16),
        "core_y": np.concatenate(core_y).astype(np.float32),
        "dev_x": np.concatenate(dev_x).astype(np.float32),
        "dev_pair": np.concatenate(dev_pair).astype(np.int16),
        "dev_y": np.concatenate(dev_y).astype(np.bool_),
        "dev_mass": np.concatenate(dev_mass).astype(np.float32),
        "dev_offsets": np.asarray(dev_offsets, dtype=np.int64),
    }
    mean = data["core_x"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = data["core_x"].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    data["feature_mean"] = mean
    data["feature_std"] = std
    data["core_x"] = np.ascontiguousarray((data["core_x"] - mean) / std, dtype=np.float32)
    data["dev_x"] = np.ascontiguousarray((data["dev_x"] - mean) / std, dtype=np.float32)
    _atomic_npz(data_path, **data)
    stage = {
        "schema": "replace_round5_nonlinear_data_stage.v1",
        "completed_at_utc": _utc_now(),
        "core_states": policy.nonlinear_core_state_count,
        "dev_states": policy.nonlinear_dev_state_count,
        "core_balanced_rows": int(data["core_x"].shape[0]),
        "dev_full_rows": int(data["dev_x"].shape[0]),
        "feature_count": DEEP_FEATURE_COUNT,
        "artifact": {
            "path": str(data_path.relative_to(output_dir)),
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path),
        },
    }
    _atomic_json(stage_path, stage)
    return data, stage


def _dev_retained_mass(
    scores: np.ndarray, mass: np.ndarray, offsets: np.ndarray, area_fraction: float
) -> float:
    retained = 0.0
    total = 0.0
    for start, end in pairwise(offsets):
        state_scores = scores[start:end]
        state_mass = mass[start:end].astype(np.float64)
        count = max(1, math.ceil(area_fraction * state_scores.size))
        selected = np.lexsort((np.arange(state_scores.size), -state_scores))[:count]
        retained += float(state_mass[selected].sum(dtype=np.float64))
        total += float(state_mass.sum(dtype=np.float64))
    return retained / total


def _new_mlp(policy: ReplaceRound5DeeperNonlinearPolicy) -> Any:
    import torch

    class PairGatedMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(DEEP_FEATURE_COUNT, policy.nonlinear_hidden_width)
            self.output = torch.nn.Linear(policy.nonlinear_hidden_width, ORDERED_PAIR_COUNT)

        def forward(self, features: Any, pair_ids: Any) -> Any:
            hidden = torch.relu(self.input(features))
            all_logits = self.output(hidden)
            return all_logits.gather(1, pair_ids[:, None]).squeeze(1)

    return PairGatedMLP()


def _export_mlp_weights(model: Any) -> PairGatedMLPWeights:
    state = model.state_dict()
    weights = PairGatedMLPWeights(
        input_weight=state["input.weight"].detach().cpu().numpy().astype(np.float32),
        input_bias=state["input.bias"].detach().cpu().numpy().astype(np.float32),
        output_weight=state["output.weight"].detach().cpu().numpy().astype(np.float32),
        output_bias=state["output.bias"].detach().cpu().numpy().astype(np.float32),
    )
    weights.validate()
    return weights


def _fit_one_seed(
    *,
    output_dir: Path,
    data: dict[str, np.ndarray],
    policy: ReplaceRound5DeeperNonlinearPolicy,
    seed: int,
) -> tuple[PairGatedMLPWeights, dict[str, Any]]:
    import torch

    seed_dir = output_dir / "nonlinear" / f"seed_{seed}"
    complete_path = seed_dir / "stage_complete.json"
    weights_path = seed_dir / "weights.npz"
    if complete_path.exists() and weights_path.exists():
        with np.load(weights_path, allow_pickle=False) as archive:
            weights = PairGatedMLPWeights(
                *(np.array(archive[name], copy=True) for name in ("w1", "b1", "w2", "b2"))
            )
        weights.validate()
        return weights, json.loads(complete_path.read_text())
    torch.manual_seed(seed)
    model = _new_mlp(policy)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=policy.nonlinear_learning_rate,
        weight_decay=policy.nonlinear_weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    resume_path = seed_dir / "resume.pt"
    start_epoch = 0
    best_metric = -math.inf
    best_epoch = -1
    best_state = None
    wait = 0
    history: list[dict[str, float | int]] = []
    if resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        generator.set_state(resume["generator_state"])
        start_epoch = int(resume["next_epoch"])
        best_metric = float(resume["best_metric"])
        best_epoch = int(resume["best_epoch"])
        best_state = resume["best_state"]
        wait = int(resume["wait"])
        history = list(resume["history"])
    core_x = torch.from_numpy(data["core_x"])
    core_pair = torch.from_numpy(data["core_pair"].astype(np.int64))
    core_y = torch.from_numpy(data["core_y"])
    dev_x = torch.from_numpy(data["dev_x"])
    dev_pair = torch.from_numpy(data["dev_pair"].astype(np.int64))
    criterion = torch.nn.BCEWithLogitsLoss()
    stopped_reason = "max-epochs"
    for epoch in range(start_epoch, policy.nonlinear_max_epochs):
        model.train()
        permutation = torch.randperm(core_x.shape[0], generator=generator)
        epoch_loss = 0.0
        for start in range(0, core_x.shape[0], policy.nonlinear_batch_size):
            index = permutation[start : start + policy.nonlinear_batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(core_x[index], core_pair[index])
            loss = criterion(logits, core_y[index])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * index.numel()
        model.eval()
        with torch.no_grad():
            dev_scores = model(dev_x, dev_pair).cpu().numpy()
        metric = _dev_retained_mass(
            dev_scores,
            data["dev_mass"],
            data["dev_offsets"],
            policy.requested_area_fraction,
        )
        history.append(
            {
                "epoch": epoch,
                "train_balanced_bce": epoch_loss / core_x.shape[0],
                "dev_retained_mass": metric,
            }
        )
        if metric >= best_metric + policy.nonlinear_min_delta:
            best_metric = metric
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        payload = {
            "schema": "replace_round5_mlp_resume.v1",
            "seed": seed,
            "next_epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "wait": wait,
            "history": history,
        }
        _atomic_torch_save(resume_path, payload)
        if (epoch + 1) % 5 == 0:
            periodic = seed_dir / "stage_checkpoints" / f"epoch_{epoch + 1:04d}.pt"
            if not periodic.exists():
                _atomic_torch_save(periodic, payload)
        if wait >= policy.nonlinear_patience:
            stopped_reason = "train-only-dev-patience"
            break
    if best_state is None:
        raise ProbeError(f"nonlinear seed {seed} produced no finite best state")
    model.load_state_dict(best_state)
    weights = _export_mlp_weights(model)
    _atomic_npz(
        weights_path,
        w1=weights.input_weight,
        b1=weights.input_bias,
        w2=weights.output_weight,
        b2=weights.output_bias,
    )
    final_checkpoint = seed_dir / "stage_checkpoints" / "seed_complete.pt"
    _atomic_torch_save(
        final_checkpoint,
        {
            "schema": "replace_round5_mlp_seed_complete.v1",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "model_state": best_state,
            "history": history,
        },
    )
    stage = {
        "schema": "replace_round5_mlp_seed_stage.v1",
        "completed_at_utc": _utc_now(),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_retained_mass": best_metric,
        "epochs_executed": len(history),
        "stopped_reason": stopped_reason,
        "early_stop_used_heldout": False,
        "weights": {
            "path": str(weights_path.relative_to(output_dir)),
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
        },
        "complete_checkpoint": {
            "path": str(final_checkpoint.relative_to(output_dir)),
            "bytes": final_checkpoint.stat().st_size,
            "sha256": _sha256(final_checkpoint),
        },
        "history": history,
    }
    _atomic_json(complete_path, stage)
    return weights, stage


def _fit_nonlinear_stage(
    *,
    output_dir: Path,
    data: dict[str, np.ndarray],
    policy: ReplaceRound5DeeperNonlinearPolicy,
) -> tuple[tuple[PairGatedMLPWeights, ...], dict[str, Any]]:
    weights = []
    seed_stages = []
    for seed in policy.nonlinear_seeds:
        fitted, stage = _fit_one_seed(
            output_dir=output_dir, data=data, policy=policy, seed=seed
        )
        weights.append(fitted)
        seed_stages.append(stage)
    stage_path = output_dir / "stage_nonlinear_fit_complete.json"
    stage = {
        "schema": "replace_round5_nonlinear_fit_stage.v1",
        "completed_at_utc": _utc_now(),
        "seed_count": len(weights),
        "seeds": seed_stages,
        "selection": "train-only dev retained mass; untouched heldout used once after all seeds",
    }
    _atomic_json(stage_path, stage)
    return tuple(weights), stage


def _selection_metrics(mass: np.ndarray, scores: np.ndarray, selected: int) -> dict[str, float]:
    flat_mass = mass.reshape(-1).astype(np.float64)
    flat_scores = scores.reshape(-1).astype(np.float64)
    order = np.lexsort((np.arange(flat_scores.size), -flat_scores))[:selected]
    return {
        "total_mass": float(flat_mass.sum(dtype=np.float64)),
        "retained_mass": float(flat_mass[order].sum(dtype=np.float64)),
    }


def _ece_sums(probability: np.ndarray, support: np.ndarray, bins: int = 10) -> list[dict[str, float | int]]:
    rows = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        rows.append(
            {
                "count": int(mask.sum()),
                "probability_sum": float(probability[mask].sum(dtype=np.float64)),
                "support_sum": float(support[mask].sum(dtype=np.float64)),
            }
        )
    return rows


def _heldout_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound5DeeperNonlinearPolicy,
    convex_weights: np.ndarray,
    nonlinear_weights: Sequence[PairGatedMLPWeights],
    data: dict[str, np.ndarray],
    segnet: Any,
    yopo: Any,
) -> dict[str, Any]:
    stage_path = output_dir / "stage_heldout_complete.json"
    heldout = [row for row in assignments if row.split == "heldout"]
    ledger = output_dir / "teacher_calls.jsonl"
    records = []
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        for assignment in (
            row for row in heldout if row.checkpoint_index == checkpoint_index
        ):
            record_path = _heldout_path(output_dir, assignment.pair_index)
            if record_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            import torch

            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            prefix, block2, block3, costate, pair_ids, metrics, elapsed, _cost = _teacher_call(
                ledger=ledger,
                assignment=assignment,
                stage="round5_heldout",
                frame_nchw=frame,
                labels_t=torch.as_tensor(label[None], dtype=torch.long),
                segnet=segnet,
                measure_cost=False,
            )
            mass, support, selected = exact_support_target(
                costate.cpu().numpy(), area_fraction=policy.requested_area_fraction
            )
            features, pairs = deeper_pair_block_features(
                prefix.cpu().numpy(),
                block2.cpu().numpy(),
                block3.cpu().numpy(),
                label,
                margin,
                pair_ids,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.heldout_lattice_stride_on_prefix,
            )
            convex_scores = block_scores(features.astype(np.float64), pairs, convex_weights)
            standardized = np.ascontiguousarray(
                (features - data["feature_mean"]) / data["feature_std"], dtype=np.float32
            )
            seed_logits = np.stack(
                [pair_gated_logits_numpy(standardized, pairs, weights) for weights in nonlinear_weights]
            )
            seed_probabilities = sigmoid_probabilities(seed_logits)
            ensemble = seed_probabilities.mean(axis=0, dtype=np.float64).astype(np.float32)
            selections = {
                "convex-deeper-pair-block-mp": _selection_metrics(
                    mass, convex_scores, selected
                ),
                "nonlinear-pair-gated-mlp-ensemble": _selection_metrics(
                    mass, ensemble, selected
                ),
                "oracle": {
                    "total_mass": float(mass.sum(dtype=np.float64)),
                    "retained_mass": float(
                        np.sort(mass.reshape(-1).astype(np.float64))[-selected:].sum(
                            dtype=np.float64
                        )
                    ),
                },
            }
            for index, seed in enumerate(policy.nonlinear_seeds):
                selections[f"nonlinear-seed-{seed}"] = _selection_metrics(
                    mass, seed_probabilities[index], selected
                )
            query = disagreement_query_audit(
                seed_probabilities,
                support.reshape(-1),
                seed=policy.seed + assignment.pair_index,
                targeted_fraction=policy.query_targeted_fraction,
                random_audit_fraction=policy.query_random_audit_fraction,
            )
            record = {
                "schema": "replace_round5_heldout_state.v1",
                "pair_index": assignment.pair_index,
                "checkpoint_index": assignment.checkpoint_index,
                "checkpoint_name": assignment.checkpoint_name,
                "selections": selections,
                "ensemble_ece_sums": _ece_sums(ensemble, support.reshape(-1)),
                "query_real_calibration": query,
                "teacher_metrics": metrics,
                "teacher_elapsed_seconds": elapsed,
                "costate_sha256": array_sha256(costate.cpu().numpy()),
            }
            _atomic_json(record_path, record)
            records.append(record)
    if len(records) != policy.heldout_state_count:
        raise ProbeError("heldout state count drift")
    names = [
        "convex-deeper-pair-block-mp",
        "nonlinear-pair-gated-mlp-ensemble",
        *(f"nonlinear-seed-{seed}" for seed in policy.nonlinear_seeds),
        "oracle",
    ]
    aggregate = {}
    for name in names:
        total = sum(row["selections"][name]["total_mass"] for row in records)
        retained = sum(row["selections"][name]["retained_mass"] for row in records)
        aggregate[name] = {
            "total_mass": total,
            "retained_mass": retained,
            "retained_mass_fraction": retained / total,
            "conditional_masked_exact_cosine": math.sqrt(retained / total),
        }
    bins = []
    for index in range(10):
        count = sum(row["ensemble_ece_sums"][index]["count"] for row in records)
        probability_sum = sum(
            row["ensemble_ece_sums"][index]["probability_sum"] for row in records
        )
        support_sum = sum(row["ensemble_ece_sums"][index]["support_sum"] for row in records)
        bins.append(
            {
                "count": count,
                "mean_probability": probability_sum / count if count else None,
                "mean_support": support_sum / count if count else None,
            }
        )
    ece = sum(
        row["count"]
        * abs(float(row["mean_probability"]) - float(row["mean_support"]))
        for row in bins
        if row["count"]
    ) / sum(row["count"] for row in bins)
    query_numeric = (
        "high_to_low_error_ratio",
        "spearman_disagreement_vs_absolute_error",
        "queried_absolute_error_fraction",
        "realized_query_fraction",
        "random_audit_positive_propensity",
    )
    query_summary = {
        key: float(np.mean([row["query_real_calibration"][key] for row in records]))
        for key in query_numeric
    }
    query_summary["state_gate_pass_fraction"] = float(
        np.mean(
            [row["query_real_calibration"]["disagreement_rank_gate_pass"] for row in records]
        )
    )
    query_summary["aggregate_gate_pass"] = bool(
        query_summary["high_to_low_error_ratio"] >= policy.disagreement_error_ratio_bar
        and query_summary["spearman_disagreement_vs_absolute_error"] > 0.0
    )
    stage = {
        "schema": "replace_round5_heldout_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(records),
        "aggregate": aggregate,
        "ensemble_ece": ece,
        "ensemble_reliability": bins,
        "query_real_calibration": query_summary,
    }
    _atomic_json(stage_path, stage)
    return stage


def _teacher_accounting(output_dir: Path, policy: ReplaceRound5DeeperNonlinearPolicy) -> dict[str, Any]:
    ledger = output_dir / "teacher_calls.jsonl"
    starts = []
    completions = []
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "exact_teacher_state_call_started":
            starts.append(row)
        elif row.get("event") == "exact_teacher_state_call_completed":
            completions.append(row)
    unique_started = {int(row["pair_index"]) for row in starts}
    unique_completed = {int(row["pair_index"]) for row in completions}
    return {
        "schema": "replace_round5_teacher_accounting.v1",
        "campaign_honest_started_calls": len(starts),
        "completed_calls": len(completions),
        "unique_started_states": len(unique_started),
        "unique_completed_states": len(unique_completed),
        "retry_starts_charged": len(starts) - len(unique_started),
        "started_call_budget": policy.teacher_started_call_budget,
        "budget_gate_pass": len(starts) <= policy.teacher_started_call_budget,
        "all_n600_states_completed": len(unique_completed) == policy.n_pairs,
        "ledger": {
            "path": str(ledger.relative_to(output_dir)),
            "bytes": ledger.stat().st_size,
            "sha256": _sha256(ledger),
        },
    }


def _decision(
    *,
    heldout: dict[str, Any],
    accounting: dict[str, Any],
    policy: ReplaceRound5DeeperNonlinearPolicy,
) -> dict[str, Any]:
    aggregate = heldout["aggregate"]
    convex = aggregate["convex-deeper-pair-block-mp"]["retained_mass_fraction"]
    nonlinear = aggregate["nonlinear-pair-gated-mlp-ensemble"]["retained_mass_fraction"]
    seed_values = np.asarray(
        [
            aggregate[f"nonlinear-seed-{seed}"]["retained_mass_fraction"]
            for seed in policy.nonlinear_seeds
        ],
        dtype=np.float64,
    )
    seed_std = float(seed_values.std(ddof=0))
    convex_pass = bool(convex >= policy.retained_mass_bar)
    nonlinear_pass = bool(
        nonlinear >= policy.retained_mass_bar
        and seed_std <= policy.nonlinear_seed_std_bar
        and accounting["budget_gate_pass"]
        and accounting["all_n600_states_completed"]
    )
    if convex_pass or nonlinear_pass:
        verdict = "GO-LOCALIZATION-RESEARCH-ONLY"
        scope = "FORMULATION x FIXED REPLAY"
    else:
        verdict = "KILL-CHEAP-LOCALIZATION-FAMILY-BY-FEATURE-SOURCE"
        scope = "FAMILY x FEATURE-SOURCE x FIXED REPLAY"
    rungs = {
        "convex-deeper-pair-block-mp": {
            "retained_mass_fraction": convex,
            "primary_mass_gate_pass": convex_pass,
            "exact_optimum_certificates": True,
        },
        "nonlinear-pair-gated-mlp-ensemble": {
            "retained_mass_fraction": nonlinear,
            "primary_mass_gate_pass": nonlinear >= policy.retained_mass_bar,
            "seed_retained_mass": seed_values.tolist(),
            "seed_population_std": seed_std,
            "stability_gate_pass": seed_std <= policy.nonlinear_seed_std_bar,
            "teacher_economics_gate_pass": accounting["budget_gate_pass"],
            "admitted": nonlinear_pass,
        },
    }
    winner = max(policy.rung_order, key=lambda name: rungs[name]["retained_mass_fraction"])
    return {
        "verdict": verdict,
        "verdict_scope": scope,
        "retained_mass_bar": policy.retained_mass_bar,
        "realized_area_fraction": policy.realized_area_fraction,
        "rungs": rungs,
        "winner": winner,
        "winner_retained_mass_fraction": rungs[winner]["retained_mass_fraction"],
        "oracle_retained_mass_fraction": aggregate["oracle"]["retained_mass_fraction"],
        "pointer_moved": False,
    }


def _cleanup_manifest(output_dir: Path, run_contract: dict[str, Any]) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and ".tmp" not in path.name:
            artifacts.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "action": "PRESERVED-DURABLE",
                    "rebuildable": True,
                    "source_bundle_sha256": hashlib.sha256(
                        json.dumps(run_contract["sources"], sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
    payload = {
        "schema": "replace_round5_cleanup_manifest.v1",
        "created_at_utc": _utc_now(),
        "policy": "certify-or-block",
        "artifacts": artifacts,
        "deleted": [],
        "moved": [],
        "blockers": [],
    }
    _atomic_json(output_dir / "cleanup_manifest.json", payload)
    return payload


def run(*, output_dir: Path, resume: bool, validate_only: bool) -> dict[str, Any]:
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ProbeError("the preregistered instance has one sealed output directory")
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        if not resume:
            raise ProbeError("completed receipt exists; pass --resume to verify")
        complete = json.loads(complete_path.read_text())
        receipt = output_dir / complete["receipt"]
        if receipt.stat().st_size != complete["bytes"] or _sha256(receipt) != complete["sha256"]:
            raise ProbeError("completed receipt custody drift")
        return json.loads(receipt.read_text())
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = ReplaceRound5DeeperNonlinearPolicy()
    contract = policy.compile_measurement_contract()
    preregistration = _validate_preregistration(policy)
    storage = _storage_custody(output_dir)
    inputs = round2._verify_input_custody()
    if validate_only:
        return {
            "schema": "replace_round5_validate_only.v1",
            "compiled_policy": contract,
            "preregistration": preregistration,
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
        contract_path = output_dir / "run_contract.json"
        if contract_path.exists():
            if not resume:
                raise ProbeError("existing run contract requires --resume")
            run_contract = json.loads(contract_path.read_text())
            if run_contract["compiled_policy"] != _json_normalized(contract):
                raise ProbeError("resume policy drift")
            if run_contract["inputs"] != inputs or run_contract["storage_preflight"] != storage:
                raise ProbeError("resume input or storage custody drift")
            for relative, row in run_contract["sources"].items():
                path = output_dir / row["path"]
                if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
                    raise ProbeError(f"bundled source drift for {relative}")
        else:
            sources = _source_bundle(output_dir)
            run_contract = {
                "schema": "replace_round5_run_contract.v1",
                "created_at_utc": _utc_now(),
                "lane_id": LANE_ID,
                "compiled_policy": contract,
                "preregistration": preregistration,
                "inputs": inputs,
                "sources": sources,
                "runtime": _runtime_custody(torch),
                "storage_preflight": storage,
                "git_head_at_measurement": _git_head(),
                "git_status_at_measurement_start": _git_status(),
                "source_rounds_2_to_4_read_only": True,
                "paid_or_remote_launch": False,
                "authority": AXIS,
            }
            _atomic_json(contract_path, run_contract)
        labels = round2._stored_npy_memmap(round2.GT_CACHE, "lstars.npy")
        margins = round2._stored_npy_memmap(round2.GT_CACHE, "margins.npy")
        if labels.shape != (600, 384, 512) or margins.shape != labels.shape:
            raise ProbeError("GT cache geometry drift")
        yopo = round2._load_tool_module(
            "_round5_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
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
        convex_weights, convex_stage = _fit_convex_stage(output_dir, accumulator)
        data, nonlinear_data_stage = _build_nonlinear_data(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        nonlinear_weights, nonlinear_fit_stage = _fit_nonlinear_stage(
            output_dir=output_dir, data=data, policy=policy
        )
        heldout_stage = _heldout_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            convex_weights=convex_weights,
            nonlinear_weights=nonlinear_weights,
            data=data,
            segnet=segnet,
            yopo=yopo,
        )
        accounting = _teacher_accounting(output_dir, policy)
        decision = _decision(heldout=heldout_stage, accounting=accounting, policy=policy)
        cost_model = json.loads((output_dir / "deep_cut_cost_model.json").read_text())
        cut = cost_model["cuts"]["block3-post-se"]
        c_label = cut["fraction_of_full_teacher_conv_flops"] + (
            1.0 - cut["fraction_of_full_teacher_conv_flops"]
        ) * policy.realized_area_fraction
        policy_sha = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        fore = check_fore_support(
            (), target_policy_sha256=policy_sha, target_arms=policy.rung_order
        )
        cleanup = _cleanup_manifest(output_dir, run_contract)
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
            "convex_fit_stage": convex_stage,
            "nonlinear_data_stage": nonlinear_data_stage,
            "nonlinear_fit_stage": nonlinear_fit_stage,
            "heldout_stage": heldout_stage,
            "deep_cut_cost_model": cost_model,
            "teacher_call_accounting": accounting,
            "economics": {
                "form": "C_teacher = A + c_label * D",
                "anchor_calls_A_campaign_honest": accounting["campaign_honest_started_calls"],
                "feature_cut_fraction_p": cut["fraction_of_full_teacher_conv_flops"],
                "selected_area_fraction_q": policy.realized_area_fraction,
                "conditional_c_label": c_label,
                "conditional_variable_cost_reduction_x": 1.0 / c_label,
                "break_even_future_steps_D": accounting["campaign_honest_started_calls"]
                / (1.0 - c_label),
                "wall_clock_claim": False,
                "pay_only_on_support_admitted": decision["verdict"].startswith("GO-"),
            },
            "FORE_composition": {**asdict(fore), "weights_applied": False},
            "branch_horizon_ticket": {
                "ticket": "DIG-S1-BRANCH-AUDIT-HORIZON",
                "status": "blocked-not-identified",
                "reason": "fixed replay has no Z,A,R,Z-prime transitions or branch propensities",
                "next_run_supply": "src/tac/causal_manifest.py boundary records",
                "candidate_horizons": list(policy.branch_horizons),
                "equal_exact_call_budgets": True,
            },
            "query_real_ticket": {
                "ticket": "DIG-S1-QUERY-REAL-CALIBRATION",
                "status": (
                    "research-only-calibrated"
                    if heldout_stage["query_real_calibration"]["aggregate_gate_pass"]
                    else "kill-disagreement-formulation"
                ),
                "live_status": "refuse-live-research-only-fixed-replay",
                **heldout_stage["query_real_calibration"],
            },
            "cleanup_custody": {
                "path": "cleanup_manifest.json",
                "blockers": cleanup["blockers"],
            },
            "triality": {
                "dsl": "tac.witness_dsl.replace_round5_deeper_nonlinear_policy",
                "equation": "tac.canonical_equations.replace_round5_deeper_nonlinear_20260713",
                "dag_feed": ".omx/research/replace_round5_deeper_nonlinear_DAG_FEED_20260713.md",
            },
            "verdict_scope": decision["verdict_scope"],
            "reformulation_queue": [
                "dense-label localizer on the same exact support target",
                "transition-complete FORE successor with Z,A,R,Z-prime custody",
                "on-policy query/refuse controller with randomized audit propensities",
                "evaluator-equivalent witness successor if localization family is closed",
            ],
            "pointer_delta": "NONE",
        }
        receipt_path = output_dir / "receipt.json"
        _atomic_json(receipt_path, receipt)
        _atomic_json(
            complete_path,
            {
                "schema": "replace_round5_completion.v1",
                "completed_at_utc": _utc_now(),
                "receipt": "receipt.json",
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
    parser.add_argument("--write-preregistration", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if REPO not in output_dir.parents:
        raise ProbeError("durable evidence must remain under the repository")
    if args.write_preregistration:
        if args.resume or args.validate_only:
            raise ProbeError("preregistration write cannot be combined with run modes")
        payload = _write_preregistration(ReplaceRound5DeeperNonlinearPolicy())
        print(
            json.dumps(
                {
                    "path": str(PREREGISTRATION),
                    "bytes": PREREGISTRATION.stat().st_size,
                    "sha256": _sha256(PREREGISTRATION),
                    "sealed_at_utc": payload["sealed_at_utc"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = run(output_dir=output_dir, resume=args.resume, validate_only=args.validate_only)
    if args.validate_only:
        print(json.dumps({"schema": receipt["schema"], "teacher_calls": 0}, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "verdict": receipt["verdict"]["verdict"],
                    "convex_retained_mass": receipt["verdict"]["rungs"][
                        "convex-deeper-pair-block-mp"
                    ]["retained_mass_fraction"],
                    "nonlinear_retained_mass": receipt["verdict"]["rungs"][
                        "nonlinear-pair-gated-mlp-ensemble"
                    ]["retained_mass_fraction"],
                    "receipt": str(output_dir / "receipt.json"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
