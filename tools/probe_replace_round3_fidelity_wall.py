#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable local-n600 probe for the round-3 fidelity-wall formulations."""

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

from tac.scorer_surrogate.frozen_replay_convex_head import (  # noqa: E402
    ReplayAssignment,
    array_sha256,
    deterministic_replay_assignments,
    vector_fidelity,
)
from tac.scorer_surrogate.replace_round3_fidelity_wall import (  # noqa: E402
    AUTHORITY_SCOPE,
    ConvMacLedger,
    MultiTargetStateStatistics,
    aggregate_multi_target_statistics,
    cache_multi_target_sufficient_statistics,
    capture_exact_teacher_with_prefix_adjoint,
    direction_admission,
    exact_prefix_vjp,
    fit_multi_target_ridge,
    local_prefix_feature_snapshot,
    log_costate_mass_target_rows,
    mass_localization_metrics,
    predict_prefix_adjoint,
    prefix_feature_matrix,
    rff_lift,
    sampled_prefix_target_rows,
    source_margin_risk_scores,
)
from tac.witness_dsl.replace_round3_fidelity_wall_policy import (  # noqa: E402
    ReplaceRound3FidelityWallPolicy,
)

SCHEMA = "replace_round3_fidelity_wall_probe.v1"
LANE_ID = "lane_replace_round3_fidelity_wall_20260713"
AXIS = "[macOS-CPU advisory; fp32 training-gradient evidence; no score authority]"
DEFAULT_OUTPUT = REPO / "experiments/results/replace_round3_fidelity_wall_20260713"
PREREGISTRATION = DEFAULT_OUTPUT / "preregistration.json"
STORAGE_PREFLIGHT = REPO / ".omx/research/replace_round3_fidelity_wall_storage_preflight_20260713.json"

SOURCE_FILES = (
    "src/tac/scorer_surrogate/replace_round3_fidelity_wall.py",
    "src/tac/witness_dsl/replace_round3_fidelity_wall_policy.py",
    "src/tac/scorer_surrogate/frozen_replay_convex_head.py",
    "src/tac/scorer_surrogate/onpolicy_matched_verdict.py",
    "src/tac/cuda_levelset_training.py",
    "src/tac/local_acceleration/torch_levelset_inflate.py",
    "tools/probe_replace_round3_fidelity_wall.py",
    "tools/probe_frozen_replay_convex_head.py",
    "tools/probe_yopo_first_layer_costate.py",
    "tools/probe_onpolicy_costate_matched_window.py",
    "tools/dash_comb_probe_n600.py",
    "upstream/modules.py",
)


class ProbeError(RuntimeError):
    """The preregistration, measurement, or custody contract failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_status() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    return output.splitlines()


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise ProbeError(f"missing measurement source {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return rows


def _source_bundle(output_dir: Path) -> dict[str, dict[str, Any]]:
    bundle = output_dir / "source_bundle"
    rows: dict[str, dict[str, Any]] = {}
    for relative, custody in _source_fingerprints().items():
        source = REPO / relative
        destination = bundle / relative
        if destination.exists():
            if destination.stat().st_size != custody["bytes"] or _sha256(destination) != custody["sha256"]:
                raise ProbeError(f"source bundle drift for {relative}")
        else:
            _atomic_bytes(destination, source.read_bytes())
        rows[relative] = {
            **custody,
            "path": str(destination.relative_to(output_dir)),
        }
    return rows


def _validate_preregistration(policy: ReplaceRound3FidelityWallPolicy) -> dict[str, Any]:
    if not PREREGISTRATION.is_file():
        raise ProbeError("missing machine-readable preregistration")
    payload = json.loads(PREREGISTRATION.read_text())
    policy_path = REPO / payload["policy_path"]
    if _sha256(policy_path) != payload["policy_sha256_before_measurement"]:
        raise ProbeError("typed policy drifted after preregistration")
    contract = policy.compile_measurement_contract()
    if payload["primary_direction_decision_rule"]["bar"] != contract["input_costate_cosine_bar"]:
        raise ProbeError("preregistered direction bar and typed policy disagree")
    if payload["rung_order"] != list(contract["rung_order"]):
        raise ProbeError("preregistered rung order and typed policy disagree")
    return {
        "path": str(PREREGISTRATION.relative_to(REPO)),
        "bytes": PREREGISTRATION.stat().st_size,
        "sha256": _sha256(PREREGISTRATION),
        "payload": payload,
    }


def _storage_custody(output_dir: Path) -> dict[str, Any]:
    if not STORAGE_PREFLIGHT.is_file():
        raise ProbeError("missing round-3 storage preflight")
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


def _train_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "train_cache" / f"pair_{pair_index:04d}.npz"


def _exact_record_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout_exact_scratch" / f"pair_{pair_index:04d}.npz"


def _direction_record_path(output_dir: Path, rung: str, pair_index: int) -> Path:
    return output_dir / "heldout_direction" / rung / f"pair_{pair_index:04d}.json"


def _localizer_record_path(output_dir: Path, rung: str, pair_index: int) -> Path:
    return output_dir / "heldout_localizer" / rung / f"pair_{pair_index:04d}.json"


def _save_state_stats(
    path: Path,
    assignment: ReplayAssignment,
    linear: MultiTargetStateStatistics,
    rff: MultiTargetStateStatistics,
    *,
    frame_sha256: str,
    prefix_sha256: str,
    exact_input_costate_sha256: str,
    teacher_metrics: dict[str, float],
    teacher_elapsed_seconds: float,
) -> None:
    _atomic_npz(
        path,
        pair_index=np.asarray(assignment.pair_index, dtype=np.int64),
        checkpoint_index=np.asarray(assignment.checkpoint_index, dtype=np.int64),
        checkpoint_name=np.asarray(assignment.checkpoint_name),
        split=np.asarray(assignment.split),
        linear_gram=linear.gram,
        linear_rhs=linear.rhs,
        linear_target_square_by_channel=linear.target_square_by_channel,
        linear_feature_sha256=np.asarray(linear.feature_sha256),
        linear_target_sha256=np.asarray(linear.target_sha256),
        rff_gram=rff.gram,
        rff_rhs=rff.rhs,
        rff_target_square_by_channel=rff.target_square_by_channel,
        rff_feature_sha256=np.asarray(rff.feature_sha256),
        rff_target_sha256=np.asarray(rff.target_sha256),
        row_count=np.asarray(linear.row_count, dtype=np.int64),
        frame_sha256=np.asarray(frame_sha256),
        prefix_sha256=np.asarray(prefix_sha256),
        exact_input_costate_sha256=np.asarray(exact_input_costate_sha256),
        teacher_ce=np.asarray(teacher_metrics["ce"], dtype=np.float64),
        teacher_dseg=np.asarray(teacher_metrics["dseg"], dtype=np.float64),
        teacher_elapsed_seconds=np.asarray(teacher_elapsed_seconds, dtype=np.float64),
    )


def _load_state_stats(
    path: Path, assignment: ReplayAssignment
) -> tuple[MultiTargetStateStatistics, MultiTargetStateStatistics]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            int(archive["pair_index"].item()) != assignment.pair_index
            or int(archive["checkpoint_index"].item()) != assignment.checkpoint_index
            or str(archive["checkpoint_name"].item()) != assignment.checkpoint_name
            or str(archive["split"].item()) != assignment.split
        ):
            raise ProbeError(f"training-cache assignment drift at pair {assignment.pair_index}")
        row_count = int(archive["row_count"].item())
        linear = MultiTargetStateStatistics(
            gram=np.array(archive["linear_gram"], dtype=np.float32, copy=True),
            rhs=np.array(archive["linear_rhs"], dtype=np.float32, copy=True),
            target_square_by_channel=np.array(
                archive["linear_target_square_by_channel"], dtype=np.float64, copy=True
            ),
            row_count=row_count,
            feature_sha256=str(archive["linear_feature_sha256"].item()),
            target_sha256=str(archive["linear_target_sha256"].item()),
        )
        rff = MultiTargetStateStatistics(
            gram=np.array(archive["rff_gram"], dtype=np.float32, copy=True),
            rhs=np.array(archive["rff_rhs"], dtype=np.float32, copy=True),
            target_square_by_channel=np.array(
                archive["rff_target_square_by_channel"], dtype=np.float64, copy=True
            ),
            row_count=row_count,
            feature_sha256=str(archive["rff_feature_sha256"].item()),
            target_sha256=str(archive["rff_target_sha256"].item()),
        )
    linear.validate()
    rff.validate()
    return linear, rff


def _teacher_call(
    *,
    round2_ledger: Path,
    assignment: ReplayAssignment,
    stage: str,
    frame_nchw: Any,
    labels_t: Any,
    segnet: Any,
    mac_ledger_needed: bool,
) -> tuple[Any, Any, Any, dict[str, float], float, dict[str, Any] | None]:
    batch_id = f"{stage}-p{assignment.pair_index:04d}-{time.time_ns()}"
    round2._teacher_start(round2_ledger, assignment, stage=stage, batch_id=batch_id)
    mac_summary = None
    if mac_ledger_needed:
        with ConvMacLedger(segnet) as ledger:
            prefix, prefix_adjoint, input_costate, metrics, elapsed = (
                capture_exact_teacher_with_prefix_adjoint(
                    segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
                )
            )
        mac_summary = ledger.summary()
    else:
        prefix, prefix_adjoint, input_costate, metrics, elapsed = (
            capture_exact_teacher_with_prefix_adjoint(
                segnet=segnet, frame_nchw=frame_nchw, labels=labels_t
            )
        )
    round2._teacher_complete(
        round2_ledger,
        assignment,
        stage=stage,
        batch_id=batch_id,
        teacher_metrics=metrics,
        elapsed_seconds=elapsed,
    )
    round2._append_jsonl(
        round2_ledger,
        {
            "event": "exact_teacher_batch_completed",
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "batch_id": batch_id,
            "state_count": 1,
            "elapsed_seconds": elapsed,
        },
    )
    return prefix, prefix_adjoint, input_costate, metrics, elapsed, mac_summary


def _training_cache_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound3FidelityWallPolicy,
    segnet: Any,
    yopo: Any,
) -> dict[str, Any]:
    stage_path = output_dir / "stage_train_cache_complete.json"
    train = [row for row in assignments if row.split == "train"]
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text())
        if stage["state_count"] != len(train):
            raise ProbeError("completed training-cache cardinality drift")
        for row in train:
            _load_state_stats(_train_record_path(output_dir, row.pair_index), row)
        return stage

    ledger_path = output_dir / "teacher_calls.jsonl"
    cost_model_path = output_dir / "prefix_cost_model.json"
    parity_rows: list[dict[str, Any]] = []
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        cohort = [
            row
            for row in train
            if row.checkpoint_index == checkpoint_index
            and not _train_record_path(output_dir, row.pair_index).is_file()
        ]
        if not cohort:
            continue
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != policy.n_pairs or code.shape[0] != 2 * policy.n_pairs:
            raise ProbeError(f"checkpoint {checkpoint_name} is not an n600 renderer")
        parity = round2._checkpoint_parity(renderer, checkpoint_index)
        if parity["status"] != "MEASURED_PASS":
            raise ProbeError(f"renderer parity failed for {checkpoint_name}")
        parity_rows.append({"checkpoint_name": checkpoint_name, **parity})
        for assignment in cohort:
            frame = round2._render_state_nchw(renderer, assignment.pair_index)
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            labels_t = __import__("torch").as_tensor(label[None], dtype=__import__("torch").long)
            prefix, prefix_adjoint, input_costate, metrics, elapsed, mac_summary = _teacher_call(
                round2_ledger=ledger_path,
                assignment=assignment,
                stage="train_cache",
                frame_nchw=frame,
                labels_t=labels_t,
                segnet=segnet,
                mac_ledger_needed=not cost_model_path.exists(),
            )
            if mac_summary is not None:
                _atomic_json(cost_model_path, mac_summary)
            # Match the exact teacher's autograd-enabled CPU execution path.
            # The no-grad/no-requires-grad convolution path may select a
            # numerically distinct kernel even though the graph is identical.
            direct_prefix = local_prefix_feature_snapshot(segnet, frame)
            if not bool(__import__("torch").equal(prefix, direct_prefix)):
                raise ProbeError("hooked and direct local-prefix activations differ")
            base = prefix_feature_matrix(
                prefix.detach().cpu().numpy(),
                label,
                margin,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=policy.train_lattice_stride_on_prefix,
            )
            lifted = rff_lift(base, seed=policy.rff_seed)
            prefix_targets = sampled_prefix_target_rows(
                prefix_adjoint.detach().cpu().numpy(),
                stride=policy.train_lattice_stride_on_prefix,
            )
            mass_targets = log_costate_mass_target_rows(
                input_costate.detach().cpu().numpy(),
                stride=policy.train_lattice_stride_on_prefix,
            )
            linear_stats = cache_multi_target_sufficient_statistics(base, prefix_targets)
            rff_stats = cache_multi_target_sufficient_statistics(
                lifted, np.concatenate((prefix_targets, mass_targets), axis=1)
            )
            _save_state_stats(
                _train_record_path(output_dir, assignment.pair_index),
                assignment,
                linear_stats,
                rff_stats,
                frame_sha256=array_sha256(frame.detach().cpu().numpy()),
                prefix_sha256=array_sha256(prefix.detach().cpu().numpy()),
                exact_input_costate_sha256=array_sha256(input_costate.detach().cpu().numpy()),
                teacher_metrics=metrics,
                teacher_elapsed_seconds=elapsed,
            )

    records = {
        str(row.pair_index): {
            "path": str(_train_record_path(output_dir, row.pair_index).relative_to(output_dir)),
            "bytes": _train_record_path(output_dir, row.pair_index).stat().st_size,
            "sha256": _sha256(_train_record_path(output_dir, row.pair_index)),
        }
        for row in train
    }
    stage = {
        "schema": "replace_round3_train_cache_stage.v1",
        "completed_at_utc": _utc_now(),
        "state_count": len(train),
        "linear_target_channels": 32,
        "rff_target_channels": 33,
        "raw_training_costates_preserved": False,
        "prefix_cost_model": {
            "path": str(cost_model_path.relative_to(output_dir)),
            "bytes": cost_model_path.stat().st_size,
            "sha256": _sha256(cost_model_path),
        },
        "checkpoint_parity": parity_rows,
        "records": records,
    }
    _atomic_json(stage_path, stage)
    return stage


def _fit_stage(
    *,
    output_dir: Path,
    assignments: Sequence[ReplayAssignment],
    policy: ReplaceRound3FidelityWallPolicy,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stage_path = output_dir / "stage_fit_complete.json"
    weights_path = output_dir / "fit" / "round3_weights.npz"
    if stage_path.is_file():
        with np.load(weights_path, allow_pickle=False) as archive:
            linear_weights = np.array(archive["linear_weights"], dtype=np.float32, copy=True)
            rff_weights = np.array(archive["rff_weights"], dtype=np.float32, copy=True)
        return linear_weights, rff_weights, json.loads(stage_path.read_text())
    linear_records: list[MultiTargetStateStatistics] = []
    rff_records: list[MultiTargetStateStatistics] = []
    for row in assignments:
        if row.split != "train":
            continue
        linear, rff = _load_state_stats(_train_record_path(output_dir, row.pair_index), row)
        linear_records.append(linear)
        rff_records.append(rff)
    linear_fit = fit_multi_target_ridge(
        aggregate_multi_target_statistics(linear_records), epochs=policy.fit_epochs
    )
    rff_fit = fit_multi_target_ridge(
        aggregate_multi_target_statistics(rff_records), epochs=policy.fit_epochs
    )
    _atomic_npz(
        weights_path,
        linear_weights=linear_fit.weights,
        rff_weights=rff_fit.weights,
    )
    stage = {
        "schema": "replace_round3_fit_stage.v1",
        "completed_at_utc": _utc_now(),
        "linear": linear_fit.summary(),
        "rff": rff_fit.summary(),
        "weights": {
            "path": str(weights_path.relative_to(output_dir)),
            "bytes": weights_path.stat().st_size,
            "sha256": _sha256(weights_path),
            "linear_array_sha256": array_sha256(linear_fit.weights),
            "rff_array_sha256": array_sha256(rff_fit.weights),
        },
    }
    _atomic_json(stage_path, stage)
    return linear_fit.weights, rff_fit.weights, stage


def _load_exact_cache(path: Path) -> tuple[np.ndarray, dict[str, float], float]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.array(archive["input_costate"], dtype=np.float32, copy=True),
            {
                "ce": float(archive["teacher_ce"].item()),
                "dseg": float(archive["teacher_dseg"].item()),
            },
            float(archive["teacher_elapsed_seconds"].item()),
        )


def _aggregate_fidelity(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [row[key] for row in rows]
    dot = sum(float(row["dot"]) for row in metrics)
    reference_square = sum(float(row["reference_norm"]) ** 2 for row in metrics)
    candidate_square = sum(float(row["candidate_norm"]) ** 2 for row in metrics)
    delta_square = sum(
        (float(row["relative_l2_error"]) * float(row["reference_norm"])) ** 2
        for row in metrics
    )
    cosine = dot / math.sqrt(reference_square * candidate_square) if reference_square and candidate_square else None
    return {
        "state_count": len(rows),
        "compared_elements": sum(int(row["compared_elements"]) for row in metrics),
        "dot": dot,
        "cosine_similarity": cosine,
        "relative_l2_error": math.sqrt(delta_square / reference_square) if reference_square else None,
        "reference_norm": math.sqrt(reference_square),
        "candidate_norm": math.sqrt(candidate_square),
        "mean_per_state_cosine": float(
            np.mean([float(row["cosine_similarity"]) for row in metrics])
        ),
        "positive_dot_state_fraction": float(np.mean([float(row["dot"]) > 0.0 for row in metrics])),
        "reduction_dtype": "float64",
    }


def _direction_stage(
    *,
    output_dir: Path,
    rung: str,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound3FidelityWallPolicy,
    weights: np.ndarray,
    segnet: Any,
    yopo: Any,
    matched: Any,
    allow_teacher: bool,
) -> dict[str, Any]:
    stage_path = output_dir / f"stage_{rung}_complete.json"
    if stage_path.is_file():
        return json.loads(stage_path.read_text())
    heldout = [row for row in assignments if row.split == "heldout"]
    ledger_path = output_dir / "teacher_calls.jsonl"
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        cohort = [row for row in heldout if row.checkpoint_index == checkpoint_index]
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        for assignment in cohort:
            record_path = _direction_record_path(output_dir, rung, assignment.pair_index)
            if record_path.is_file():
                continue
            label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            theta_exact = renderer.code[2 * assignment.pair_index + 1].detach().clone().requires_grad_(True)
            frame_nhwc = round2._render_chart_for_pair(renderer, theta_exact, assignment.pair_index)
            frame_nchw = frame_nhwc.permute(0, 3, 1, 2).contiguous()
            settled = round2._render_state_nchw(renderer, assignment.pair_index)
            if not bool(__import__("torch").equal(frame_nchw.detach(), settled)):
                raise ProbeError(f"heldout render parity failed at pair {assignment.pair_index}")
            exact_path = _exact_record_path(output_dir, assignment.pair_index)
            if exact_path.is_file():
                exact_np, teacher_metrics, teacher_elapsed = _load_exact_cache(exact_path)
                prefix = local_prefix_feature_snapshot(segnet, frame_nchw)
            else:
                if not allow_teacher:
                    raise ProbeError("later rung cannot regenerate a missing heldout exact cache")
                prefix, _prefix_adjoint, exact_t, teacher_metrics, teacher_elapsed, _mac = _teacher_call(
                    round2_ledger=ledger_path,
                    assignment=assignment,
                    stage="heldout_validation",
                    frame_nchw=frame_nchw,
                    labels_t=__import__("torch").as_tensor(label[None], dtype=__import__("torch").long),
                    segnet=segnet,
                    mac_ledger_needed=False,
                )
                exact_np = exact_t.detach().cpu().numpy()
                _atomic_npz(
                    exact_path,
                    input_costate=exact_np,
                    teacher_ce=np.asarray(teacher_metrics["ce"], dtype=np.float64),
                    teacher_dseg=np.asarray(teacher_metrics["dseg"], dtype=np.float64),
                    teacher_elapsed_seconds=np.asarray(teacher_elapsed, dtype=np.float64),
                    frame_sha256=np.asarray(array_sha256(frame_nchw.detach().cpu().numpy())),
                )
            total_started = time.perf_counter()
            base = prefix_feature_matrix(
                prefix.detach().cpu().numpy(),
                label,
                margin,
                checkpoint_index=assignment.checkpoint_index,
                checkpoint_count=policy.checkpoint_count,
                stride=1,
            )
            features = base if rung == "pre_se_prefix_linear" else rff_lift(base, seed=policy.rff_seed)
            prefix_prediction = predict_prefix_adjoint(
                features,
                weights[:, :32],
                height=prefix.shape[2],
                width=prefix.shape[3],
            )
            predicted_t, prefix_vjp_seconds = exact_prefix_vjp(
                segnet=segnet,
                frame_nchw=frame_nchw,
                prefix_adjoint_nchw=prefix_prediction,
            )
            surrogate_total_seconds = time.perf_counter() - total_started
            exact_t = __import__("torch").as_tensor(exact_np, dtype=__import__("torch").float32)
            costate_fidelity = vector_fidelity(exact_np, predicted_t.detach().cpu().numpy())
            exact_gradient, exact_vjp_seconds = matched._renderer_gradient(
                frame_nhwc, theta_exact, exact_t
            )
            theta_predicted = (
                renderer.code[2 * assignment.pair_index + 1].detach().clone().requires_grad_(True)
            )
            repeated_frame = round2._render_chart_for_pair(
                renderer, theta_predicted, assignment.pair_index
            )
            predicted_gradient, predicted_renderer_vjp_seconds = matched._renderer_gradient(
                repeated_frame, theta_predicted, predicted_t
            )
            row = {
                "schema": "replace_round3_direction_state.v1",
                "completed_at_utc": _utc_now(),
                "rung": rung,
                "assignment": assignment.to_dict(),
                "teacher_metrics": teacher_metrics,
                "teacher_elapsed_seconds": teacher_elapsed,
                "surrogate_total_seconds": surrogate_total_seconds,
                "prefix_vjp_seconds": prefix_vjp_seconds,
                "costate_fidelity": costate_fidelity,
                "renderer_gradient_fidelity": vector_fidelity(
                    exact_gradient.detach().cpu().numpy(),
                    predicted_gradient.detach().cpu().numpy(),
                ),
                "exact_renderer_gradient": exact_gradient.detach().cpu().numpy().astype(float).tolist(),
                "exact_renderer_vjp_seconds": exact_vjp_seconds,
                "predicted_renderer_vjp_seconds": predicted_renderer_vjp_seconds,
                "exact_costate_cache": {
                    "path": str(exact_path.relative_to(output_dir)),
                    "bytes": exact_path.stat().st_size,
                    "sha256": _sha256(exact_path),
                },
                "authority": AXIS,
            }
            _atomic_json(record_path, row)
    rows = [
        json.loads(_direction_record_path(output_dir, rung, row.pair_index).read_text())
        for row in heldout
    ]
    costate = _aggregate_fidelity(rows, "costate_fidelity")
    renderer = _aggregate_fidelity(rows, "renderer_gradient_fidelity")
    admission = direction_admission(
        cosine=float(costate["cosine_similarity"]),
        positive_dot_state_fraction=float(costate["positive_dot_state_fraction"]),
        cosine_bar=policy.input_costate_cosine_bar,
        fraction_bar=policy.positive_dot_state_fraction_bar,
    )
    exact_seconds = sum(float(row["teacher_elapsed_seconds"]) for row in rows)
    surrogate_seconds = sum(float(row["surrogate_total_seconds"]) for row in rows)
    stage = {
        "schema": "replace_round3_direction_stage.v1",
        "completed_at_utc": _utc_now(),
        "rung": rung,
        "state_count": len(rows),
        "costate_fidelity": costate,
        "renderer_gradient_fidelity": renderer,
        "admission": admission,
        "timing": {
            "exact_teacher_total_seconds": exact_seconds,
            "surrogate_total_seconds": surrogate_seconds,
            "measured_surrogate_fraction_of_exact_teacher": surrogate_seconds / exact_seconds,
            "measured_speedup_at_registered_seam_x": exact_seconds / surrogate_seconds,
            "scope": "heldout feature chart plus convex head plus exact local-prefix VJP",
        },
        "records": {
            str(row["assignment"]["pair_index"]): {
                "path": str(
                    _direction_record_path(
                        output_dir, rung, int(row["assignment"]["pair_index"])
                    ).relative_to(output_dir)
                ),
                "sha256": _sha256(
                    _direction_record_path(
                        output_dir, rung, int(row["assignment"]["pair_index"])
                    )
                ),
            }
            for row in rows
        },
    }
    _atomic_json(stage_path, stage)
    return stage


def _aggregate_localizer(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row["mass_localization"] for row in rows]
    total = sum(float(row["exact_costate_l2_square"]) for row in metrics)
    retained = sum(float(row["retained_exact_costate_l2_square"]) for row in metrics)
    oracle = sum(float(row["oracle_retained_exact_costate_l2_square"]) for row in metrics)
    area = sum(
        int(row["selected_prefix_cells"]) for row in metrics
    ) / sum(int(row["prefix_cell_count"]) for row in metrics)
    fraction = retained / total
    return {
        "state_count": len(rows),
        "realized_area_fraction": area,
        "retained_exact_costate_l2_mass_fraction": fraction,
        "oracle_retained_exact_costate_l2_mass_fraction": oracle / total,
        "uplift_over_uniform_area": fraction / area,
        "conditional_masked_exact_costate_cosine": math.sqrt(fraction),
        "mean_per_state_mass_fraction": float(
            np.mean([float(row["retained_exact_costate_l2_mass_fraction"]) for row in metrics])
        ),
    }


def _localizer_stage(
    *,
    output_dir: Path,
    rung: str,
    assignments: Sequence[ReplayAssignment],
    labels: np.memmap,
    margins: np.memmap,
    policy: ReplaceRound3FidelityWallPolicy,
    rff_weights: np.ndarray,
    segnet: Any,
    yopo: Any,
) -> dict[str, Any]:
    stage_path = output_dir / f"stage_{rung}_complete.json"
    if stage_path.is_file():
        return json.loads(stage_path.read_text())
    heldout = [row for row in assignments if row.split == "heldout"]
    for checkpoint_index, (_name, checkpoint_path, _epoch) in enumerate(round2.CHECKPOINTS):
        cohort = [row for row in heldout if row.checkpoint_index == checkpoint_index]
        renderer, _code, _model, _dash = yopo._load_renderer(checkpoint_path)
        for assignment in cohort:
            record_path = _localizer_record_path(output_dir, rung, assignment.pair_index)
            if record_path.is_file():
                continue
            exact_path = _exact_record_path(output_dir, assignment.pair_index)
            if not exact_path.is_file():
                raise ProbeError("localizer stage lost the heldout exact cache")
            exact_np, _metrics, _elapsed = _load_exact_cache(exact_path)
            margin = np.array(margins[assignment.pair_index], dtype=np.float32, copy=True)
            if rung == "source_margin_risk":
                scores = source_margin_risk_scores(margin)
            else:
                label = np.array(labels[assignment.pair_index], dtype=np.int64, copy=True)
                frame = round2._render_state_nchw(renderer, assignment.pair_index)
                prefix = local_prefix_feature_snapshot(segnet, frame)
                base = prefix_feature_matrix(
                    prefix.cpu().numpy(),
                    label,
                    margin,
                    checkpoint_index=assignment.checkpoint_index,
                    checkpoint_count=policy.checkpoint_count,
                    stride=1,
                )
                lifted = rff_lift(base, seed=policy.rff_seed)
                scores = np.ascontiguousarray(
                    (lifted @ rff_weights[:, 32]).reshape(prefix.shape[2], prefix.shape[3]),
                    dtype=np.float32,
                )
            _atomic_json(
                record_path,
                {
                    "schema": "replace_round3_localizer_state.v1",
                    "completed_at_utc": _utc_now(),
                    "rung": rung,
                    "assignment": assignment.to_dict(),
                    "mass_localization": mass_localization_metrics(
                        exact_np, scores, area_fraction=policy.flip_risk_area_fraction
                    ),
                },
            )
    rows = [
        json.loads(_localizer_record_path(output_dir, rung, row.pair_index).read_text())
        for row in heldout
    ]
    aggregate = _aggregate_localizer(rows)
    stage = {
        "schema": "replace_round3_localizer_stage.v1",
        "completed_at_utc": _utc_now(),
        "rung": rung,
        "mass_localization": aggregate,
        "admission": {
            "verdict": (
                "PASS"
                if aggregate["retained_exact_costate_l2_mass_fraction"]
                >= policy.localizer_mass_fraction_bar
                else "FAIL"
            ),
            "mass_fraction_bar": policy.localizer_mass_fraction_bar,
            "direction_surrogate_claim": False,
            "exact_dense_teacher_still_required_for_conditional_cosine": True,
        },
    }
    _atomic_json(stage_path, stage)
    return stage


def _teacher_accounting(
    output_dir: Path,
    policy: ReplaceRound3FidelityWallPolicy,
    assignments: Sequence[ReplayAssignment],
) -> dict[str, Any]:
    ledger = output_dir / "teacher_calls.jsonl"
    rows = round2._canonicalize_event_ledger(ledger)
    starts = [row for row in rows if row["event"] == "exact_teacher_state_call_started"]
    completions = [row for row in rows if row["event"] == "exact_teacher_state_call_completed"]
    required = {(row.split, row.pair_index) for row in assignments}
    completed = {(row["split"], int(row["pair_index"])) for row in completions}
    if not required.issubset(completed):
        raise ProbeError("teacher ledger does not cover every registered state")
    train_starts = sum(row["split"] == "train" for row in starts)
    validation_starts = sum(row["split"] == "heldout" for row in starts)
    effective = policy.effective_cached_label_uses
    return {
        "teacher_call_unit": "one batch-size-1 exact labeled SegNet forward plus input backward",
        "training_label_calls_observed_conservative": train_starts,
        "validation_calls_observed_conservative": validation_starts,
        "all_exact_labeled_calls_observed_conservative": len(starts),
        "completed_call_events": len(completions),
        "required_unique_states": len(required),
        "completed_unique_states": len(completed),
        "effective_cached_training_label_uses": effective,
        "label_only_teacher_amortization_x": effective / train_starts,
        "inclusive_teacher_amortization_x": effective / len(starts),
        "validation_calls_do_not_amortize_labels": True,
        "label_difference_teacher_coefficient": 0,
        "FORE_weight_fit_teacher_calls": 0,
        "teacher_ledger": {
            "path": str(ledger.relative_to(output_dir)),
            "bytes": ledger.stat().st_size,
            "sha256": _sha256(ledger),
            "rows": len(rows),
        },
    }


def _decision_stage(
    *,
    output_dir: Path,
    linear: dict[str, Any],
    rff: dict[str, Any] | None,
    localizers: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    stage_path = output_dir / "stage_decision_complete.json"
    if stage_path.is_file():
        return json.loads(stage_path.read_text())
    direction_rows = [linear, *([] if rff is None else [rff])]
    passing_direction = [row for row in direction_rows if row["admission"]["verdict"] == "PASS"]
    passing_localizer = [row for row in localizers if row["admission"]["verdict"] == "PASS"]
    if passing_direction:
        winner = passing_direction[0]["rung"]
        verdict = "GO_DIRECTION_RESEARCH_ONLY"
        winning_cosine = passing_direction[0]["costate_fidelity"]["cosine_similarity"]
    elif passing_localizer:
        winner = passing_localizer[0]["rung"]
        verdict = "GO_TARGET_LOCALIZATION_ONLY__DENSE_TEACHER_ECONOMICS_UNCHANGED"
        winning_cosine = passing_localizer[0]["mass_localization"][
            "conditional_masked_exact_costate_cosine"
        ]
    else:
        winner_row = max(
            direction_rows,
            key=lambda row: float(row["costate_fidelity"]["cosine_similarity"]),
        )
        winner = winner_row["rung"]
        verdict = "NO_GO_REGISTERED_ROUND3_RUNGS"
        winning_cosine = winner_row["costate_fidelity"]["cosine_similarity"]
    stage = {
        "schema": "replace_round3_decision_stage.v1",
        "completed_at_utc": _utc_now(),
        "verdict": verdict,
        "winning_rung": winner,
        "winning_reported_cosine": winning_cosine,
        "preregistered_input_costate_cosine_bar": linear["admission"]["cosine_bar"],
        "direction_rungs": [row["rung"] for row in direction_rows],
        "localizer_rungs": [row["rung"] for row in localizers],
        "stop_rule_honored": True,
        "pointer_delta": "NONE",
    }
    _atomic_json(stage_path, stage)
    return stage


def _cleanup_exact_scratch(output_dir: Path, *, run_contract: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_dir / "cleanup_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())
    entries = []
    for path in sorted((output_dir / "heldout_exact_scratch").glob("*.npz")):
        custody = {
            "original_path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "deterministic_rebuild_command": (
                "PYTHONPATH=src .venv/bin/python tools/probe_replace_round3_fidelity_wall.py "
                f"--output-dir {output_dir.relative_to(REPO)} --resume"
            ),
            "run_contract_sha256": _sha256(output_dir / "run_contract.json"),
            "source_bundle_tree_identity": run_contract["sources"],
            "reason": "success-only heldout exact-costate scratch; every admitted reduction is sealed",
            "rebuildable": True,
            "cold_store_destination": None,
            "deleted_after_certification": True,
            "false_authority_score_flags": {
                "score_claim": False,
                "promotion_eligible": False,
            },
        }
        entries.append(custody)
        path.unlink()
    directory = output_dir / "heldout_exact_scratch"
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()
    manifest = {
        "schema": "replace_round3_cleanup_manifest.v1",
        "completed_at_utc": _utc_now(),
        "policy": "certify deterministic rebuildability before deleting success-only scratch",
        "deleted_entries": entries,
        "deleted_bytes": sum(int(row["bytes"]) for row in entries),
        "preserved_compact_evidence": [
            "train_cache",
            "fit",
            "heldout_direction",
            "heldout_localizer",
            "teacher_calls.jsonl",
        ],
        "blockers": [],
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _complete_receipt_or_none(output_dir: Path, *, resume: bool) -> dict[str, Any] | None:
    complete_path = output_dir / "complete.json"
    if not complete_path.exists():
        return None
    if not resume:
        raise ProbeError("completed result exists; pass --resume to verify/read it")
    complete = json.loads(complete_path.read_text())
    receipt_path = output_dir / complete["receipt"]
    if receipt_path.stat().st_size != complete["bytes"] or _sha256(receipt_path) != complete["sha256"]:
        raise ProbeError("completed receipt custody drift")
    return json.loads(receipt_path.read_text())


def run(*, output_dir: Path, resume: bool) -> dict[str, Any]:
    completed = _complete_receipt_or_none(output_dir, resume=resume)
    if completed is not None:
        return completed
    if output_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise ProbeError("the preregistered instance has one sealed output directory")

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = ReplaceRound3FidelityWallPolicy()
    contract = policy.compile_measurement_contract()
    prereg = _validate_preregistration(policy)
    storage = _storage_custody(output_dir)
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
        runtime = _runtime_custody(torch)
        inputs = round2._verify_input_custody()
        if contract_path.exists():
            if not resume:
                raise ProbeError("existing run contract requires --resume")
            run_contract = json.loads(contract_path.read_text())
            if run_contract["compiled_policy"] != contract:
                raise ProbeError("resume policy drift")
            if run_contract["sources"] != _source_bundle(output_dir):
                raise ProbeError("resume source custody drift")
            if run_contract["inputs"] != inputs or run_contract["storage_preflight"] != storage:
                raise ProbeError("resume input/storage custody drift")
        else:
            run_contract = {
                "schema": "replace_round3_run_contract.v1",
                "created_at_utc": _utc_now(),
                "lane_id": LANE_ID,
                "compiled_policy": contract,
                "preregistration": prereg,
                "inputs": inputs,
                "sources": _source_bundle(output_dir),
                "runtime": runtime,
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
            "_round3_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
        )
        matched = round2._load_tool_module(
            "_round3_committed_matched", "tools/probe_onpolicy_costate_matched_window.py"
        )
        segnet = round2._load_cpu_segnet()

        train_stage = _training_cache_stage(
            output_dir=output_dir,
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            segnet=segnet,
            yopo=yopo,
        )
        linear_weights, rff_weights, fit_stage = _fit_stage(
            output_dir=output_dir, assignments=assignments, policy=policy
        )
        linear_stage = _direction_stage(
            output_dir=output_dir,
            rung="pre_se_prefix_linear",
            assignments=assignments,
            labels=labels,
            margins=margins,
            policy=policy,
            weights=linear_weights,
            segnet=segnet,
            yopo=yopo,
            matched=matched,
            allow_teacher=True,
        )
        rff_stage = None
        localizers: list[dict[str, Any]] = []
        if linear_stage["admission"]["verdict"] != "PASS":
            rff_stage = _direction_stage(
                output_dir=output_dir,
                rung="pre_se_prefix_rff",
                assignments=assignments,
                labels=labels,
                margins=margins,
                policy=policy,
                weights=rff_weights,
                segnet=segnet,
                yopo=yopo,
                matched=matched,
                allow_teacher=False,
            )
        if rff_stage is not None and rff_stage["admission"]["verdict"] != "PASS":
            margin_stage = _localizer_stage(
                output_dir=output_dir,
                rung="source_margin_risk",
                assignments=assignments,
                labels=labels,
                margins=margins,
                policy=policy,
                rff_weights=rff_weights,
                segnet=segnet,
                yopo=yopo,
            )
            localizers.append(margin_stage)
            if margin_stage["admission"]["verdict"] != "PASS":
                localizers.append(
                    _localizer_stage(
                        output_dir=output_dir,
                        rung="rff_costate_mass_ridge",
                        assignments=assignments,
                        labels=labels,
                        margins=margins,
                        policy=policy,
                        rff_weights=rff_weights,
                        segnet=segnet,
                        yopo=yopo,
                    )
                )
        decision = _decision_stage(
            output_dir=output_dir,
            linear=linear_stage,
            rff=rff_stage,
            localizers=localizers,
        )
        accounting = _teacher_accounting(output_dir, policy, assignments)
        cleanup = _cleanup_exact_scratch(output_dir, run_contract=run_contract)
        cleanup_path = output_dir / "cleanup_manifest.json"
        cost_model = json.loads((output_dir / "prefix_cost_model.json").read_text())
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
            "train_cache_stage": train_stage,
            "fit_stage": fit_stage,
            "direction_stages": [
                linear_stage,
                *([] if rff_stage is None else [rff_stage]),
            ],
            "target_reformulation_stages": localizers,
            "prefix_cost_model": cost_model,
            "teacher_call_accounting": accounting,
            "FORE_composition": {
                "status": "NO_GO_CURRENT_INSTANCE__CONDITIONAL_FORMULATION_OPEN",
                "weights_applied": False,
                "reason": "isolated replay states are not transition-complete and have no support receipt",
            },
            "cleanup_custody": {
                "path": str(cleanup_path.relative_to(output_dir)),
                "bytes": cleanup_path.stat().st_size,
                "sha256": _sha256(cleanup_path),
                "deleted_exact_scratch_bytes": cleanup["deleted_bytes"],
                "blockers": cleanup["blockers"],
            },
            "triality": {
                "dsl": "tac.witness_dsl.replace_round3_fidelity_wall_policy",
                "equation": "tac.canonical_equations.replace_round3_fidelity_wall_20260713",
                "dag_feed": ".omx/research/replace_round3_fidelity_wall_DAG_FEED_20260713.md",
            },
            "verdict_scope": (
                "FORMULATION x INSTANCE: local pre-SE prefix-adjoint chart, one RFF lift, "
                "fixed V9 n600 replay, seed455, macOS-CPU advisory"
            ),
            "reformulation_queue": [
                "deeper local pre-SE prefix with separately derived cost fraction",
                "class-pair block heads on the same prefix",
                "transition-complete stage-frozen FORE successor",
                "on-policy dense-label learner under a matched exact controller",
            ],
            "pointer_delta": "NONE",
        }
        receipt_path = output_dir / "measurement_receipt.json"
        _atomic_json(receipt_path, receipt)
        _atomic_json(
            output_dir / "complete.json",
            {
                "schema": "replace_round3_completion.v1",
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
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if REPO not in output_dir.parents:
        raise ProbeError("durable evidence must remain under the repository")
    receipt = run(output_dir=output_dir, resume=args.resume)
    decision = receipt["verdict"]
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "winning_rung": decision["winning_rung"],
                "winning_reported_cosine": decision["winning_reported_cosine"],
                "preregistered_input_costate_cosine_bar": decision[
                    "preregistered_input_costate_cosine_bar"
                ],
                "label_only_teacher_amortization_x": receipt["teacher_call_accounting"][
                    "label_only_teacher_amortization_x"
                ],
                "inclusive_teacher_amortization_x": receipt["teacher_call_accounting"][
                    "inclusive_teacher_amortization_x"
                ],
                "receipt": str(output_dir / "measurement_receipt.json"),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
