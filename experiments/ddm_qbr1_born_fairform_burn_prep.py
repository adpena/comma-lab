#!/usr/bin/env python3
"""Build, seal, run, and adjudicate the QBR1 born/QBT fair-form A/B.

Build is scorer-free.  ``run-config`` is MAIN-only and fail-closes unless both
the Metal and scorer lanes have been bound into the sealed cell config.  Every
trained cell starts from the same retained r10 EMA state, runs exactly 5,000
updates on the preregistered n32 population, and retains checkpoints, milestone
scorer payloads, and deterministic QBF1 coder payloads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments import ddm_qbt1_qbflow_trainer as qbt

REPO = Path(__file__).resolve().parents[1]
AP_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep")
CONFIG_ROOT = AP_ROOT / "sealed_configs"
RUN_ROOT = AP_ROOT / "runs"
R10_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r10")
R10_CHECKPOINT = R10_ROOT / "stage_03_joint_boundary_interior_birth/checkpoints/stage_03_end.pt"
R10_STATUS = R10_ROOT / "resource_safe_run_status.json"
R10_RESULT = R10_ROOT / "RESULT.json"
R10_CONFIG = R10_ROOT.parent / "AUTHORIZED_N32_R10_10020_20260829.json"
QBR_INITIAL_STATE = AP_ROOT / "initialized/qbr1_from_r10_ema_state.pt"

SCHEMA = "ddm_qbr1_fairform_config.v1"
CHECKPOINT_SCHEMA = "ddm_qbr1_fairform_checkpoint.v1"
RESULT_SCHEMA = "ddm_qbr1_fairform_result.v1"
SEEDS = (20260902, 20260903, 20260904)
TOTAL_STEPS = 5_000
CHECKPOINT_EVERY = TOTAL_STEPS // qbt.CHECKPOINT_CRASH_LOSS_DENOMINATOR
MILESTONES = (0, 1_000, 2_000, 3_000, 4_000, 5_000)
RATE_DENOMINATOR = 37_545_489
TARGET_SCORE = 0.12
ARMS: dict[str, dict[str, Any]] = {
    "control_native100": {
        "role": "control",
        "realized_weight": 100.0,
        "native_interface_weight": 100.0,
    },
    "treatment_zero_native": {
        "role": "treatment",
        "realized_weight": 100.0,
        "native_interface_weight": 0.0,
    },
}

EXPECTED_SHA256 = {
    "qbt_trainer": "5da466af9f64295d6bc9d1242ed427724faec586c53d450d3e358e7f1c7492c8",
    "ce1_target_margin": "ffdf098801863ff8bffe8bd818ce101928dd75b4937cbbffb2e225bddbc12f4b",
    "w96b_law_module": "053bd12e198bb74a44036e497a1277d9d36638c96acdabba278a2c72f2234923",
    "r10_checkpoint": "09fd416531c74f69ca7033cf3f13b23c9e0472486a97ce9973f62f2fb86c138f",
    "r10_config": "87eff6e8cc0339c8b669de9f714e8c666d13a9a8f406a396245540e774c200e9",
    "r10_result": "9d769f0dd95e76e40ed817aece5b3d608b8b72c3e4bd643793a59ccf0e31354d",
    "r10_status": "ef65dc03210ac38e3c0a69e8264ef28d593d1567e9fe0213d58548aeb628a8cb",
}
INPUT_PATHS = {
    "qbt_trainer": REPO / "experiments/ddm_qbt1_qbflow_trainer.py",
    "ce1_target_margin": REPO / "src/tac/pr130_lift/lifted/semantic_renderer_oracle.py",
    "w96b_law_module": REPO / "src/tac/witness_dsl/w96b_aligned_loss_levers_20260826.py",
    "r10_checkpoint": R10_CHECKPOINT,
    "r10_config": R10_CONFIG,
    "r10_result": R10_RESULT,
    "r10_status": R10_STATUS,
}
_VERIFIED_INPUT_CACHE: dict[str, dict[str, Any]] | None = None


class QBR1Error(RuntimeError):
    """Fail-closed QBR1 contract error."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def source_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def verify_inputs() -> dict[str, dict[str, Any]]:
    global _VERIFIED_INPUT_CACHE
    if _VERIFIED_INPUT_CACHE is not None:
        return copy.deepcopy(_VERIFIED_INPUT_CACHE)
    rows = {name: qbt.file_fact(path) for name, path in INPUT_PATHS.items()}
    for name, expected in EXPECTED_SHA256.items():
        if rows[name]["sha256"] != expected:
            raise QBR1Error(f"pinned input drifted: {name}")
    qbt_rows = qbt.verify_pins()
    rows.update({f"qbt_{name}": value for name, value in qbt_rows.items()})
    _VERIFIED_INPUT_CACHE = copy.deepcopy(rows)
    return copy.deepcopy(rows)


def schedule_for_seed(seed: int, steps: int = TOTAL_STEPS) -> tuple[int, ...]:
    """Return a balanced, seeded order over the two fixed 16-pair chunks."""

    if steps < 2 or steps % 2:
        raise QBR1Error("fair-form schedule needs a positive even update count")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    schedule: list[int] = []
    for _epoch in range(steps // 2):
        first = int(rng.integers(0, 2))
        schedule.extend((first, 1 - first))
    return tuple(schedule)


def schedule_receipt(seed: int) -> dict[str, Any]:
    schedule = schedule_for_seed(seed)
    raw = np.asarray(schedule, dtype="u1").tobytes()
    return {
        "algorithm": "numpy.PCG64(seed); per two-update epoch choose [0,1] or [1,0]",
        "seed": int(seed),
        "updates": len(schedule),
        "chunk_update_counts": [schedule.count(0), schedule.count(1)],
        "sha256_u8": hashlib.sha256(raw).hexdigest(),
        "first_32": list(schedule[:32]),
        "last_32": list(schedule[-32:]),
    }


def config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "action",
        "output",
        "resume_from",
        "launch_authorized",
        "scorer_lane",
        "metal_lane",
    }
    return {key: copy.deepcopy(value) for key, value in config.items() if key not in ignored}


def build_initial_state() -> dict[str, Any]:
    checkpoint = torch.load(R10_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "stage_03_joint_boundary_interior_birth_end":
        raise QBR1Error("r10 source is not the stage-03 end checkpoint")
    reference = qbt.load_initial_model(torch.device("cpu"))
    shadow = checkpoint.get("ema", {}).get("shadow", {})
    if set(shadow) != set(reference.state_dict()):
        raise QBR1Error("r10 EMA tensor set differs from the exact QBF1 twin")
    payload = {
        "schema": "ddm_qbt2b_initialized_qbf1_state.v1",
        "state_dict": {name: value.detach().cpu().clone() for name, value in shadow.items()},
        "provenance": {
            "source_checkpoint": qbt.file_fact(R10_CHECKPOINT),
            "source_stage": checkpoint["stage"],
            "source_step": int(checkpoint["step"]),
            "basis": "ema_shadow",
            "same_start_for_all_six_cells": True,
            "birth_gate_revision": "qbt2b_r6_existence_majority_reviewed",
        },
    }
    fact = qbt.atomic_torch(QBR_INITIAL_STATE, payload)
    roundtrip = torch.load(QBR_INITIAL_STATE, map_location="cpu", weights_only=False)
    if set(roundtrip["state_dict"]) != set(reference.state_dict()):
        raise QBR1Error("QBR initialized state failed strict tensor-key round trip")
    return fact


def compile_cell(seed: int, arm_name: str, initial_state: Mapping[str, Any]) -> dict[str, Any]:
    arm = ARMS[arm_name]
    ema = qbt.resolve_ema_law(TOTAL_STEPS)
    config = {
        "schema": SCHEMA,
        "action": "train",
        "cell_id": f"seed_{seed}_{arm_name}",
        "arm_name": arm_name,
        "arm_role": arm["role"],
        "output": str((RUN_ROOT / f"seed_{seed}" / arm_name).resolve()),
        "seed": int(seed),
        "device": "mps",
        "pair_ids": list(qbt.SELECTION_IDS),
        "selection_weights": list(qbt.SELECTION_WEIGHTS),
        "population_n": qbt.N,
        "chunk_pairs": qbt.REAL_TRAIN_CHUNK_PAIRS,
        "total_steps": TOTAL_STEPS,
        "checkpoint_every_steps": CHECKPOINT_EVERY,
        "milestones": list(MILESTONES),
        "learning_rate": 2.0e-4,
        "expected_flip_tau_start": 0.15,
        "expected_flip_tau_end": 0.05,
        "pose_start_step": 0,
        "objective": {
            "law": "expected_flip_target_vs_best_other_margin_v1",
            "realized_weight": arm["realized_weight"],
            "native_interface_weight": arm["native_interface_weight"],
            "pose_term": "sqrt(10*weighted_pose_mse)",
            "realization_path": "QBF1 RGB -> bicubic camera grid -> uint8 STE -> frozen scorers",
            "closed_form_note": (
                "Exact argmax is piecewise constant with zero derivative almost everywhere; "
                "the registered CE1 expected-flip margin is the exact differentiable "
                "continuation, while milestones use exact argmax after the full round trip."
            ),
        },
        "ema": ema,
        "schedule": schedule_receipt(seed),
        "initial_state": dict(initial_state),
        "birth_gate_revision": {
            "mode": "existence_majority",
            "within_class_error_max": 0.5,
            "source": "qbt2b_r6_reviewed_revision",
            "note": "inherited born start; raw r5 balanced-CE start is forbidden",
        },
        "margin_constraints": {
            "mode": qbt.MARGIN_CONSTRAINT_LANE_MOVABLE,
            "bounds": copy.deepcopy(qbt.MARGIN_CONSTRAINT_MODE_PINS[qbt.MARGIN_CONSTRAINT_LANE_MOVABLE]["bounds"]),
            "eta_lambda": float(qbt.MARGIN_CONSTRAINT_MODE_PINS[qbt.MARGIN_CONSTRAINT_LANE_MOVABLE]["eta_lambda"]),
        },
        "minimum_free_bytes": 8 * 1024**3,
        "retain_all_payloads": True,
        "resume_from": None,
        "launch_authorized": False,
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
        "source_pins": verify_inputs(),
        "source_revision": source_revision(),
        "score_claim": False,
        "promotion_eligible": False,
    }
    validate_config(config, require_launch_authority=False)
    return config


def validate_config(config: Mapping[str, Any], *, require_launch_authority: bool = True) -> None:
    if config.get("schema") != SCHEMA or config.get("arm_name") not in ARMS:
        raise QBR1Error("QBR config schema or arm differs")
    arm = ARMS[str(config["arm_name"])]
    if (
        config.get("arm_role") != arm["role"]
        or float(config["objective"]["realized_weight"]) != arm["realized_weight"]
        or float(config["objective"]["native_interface_weight"]) != arm["native_interface_weight"]
    ):
        raise QBR1Error("QBR single treatment variable differs")
    if (
        tuple(map(int, config["pair_ids"])) != qbt.SELECTION_IDS
        or tuple(map(int, config["selection_weights"])) != qbt.SELECTION_WEIGHTS
        or int(config["chunk_pairs"]) != 16
        or int(config["total_steps"]) != TOTAL_STEPS
        or tuple(map(int, config["milestones"])) != MILESTONES
        or int(config["checkpoint_every_steps"]) > CHECKPOINT_EVERY
        or config.get("retain_all_payloads") is not True
    ):
        raise QBR1Error("QBR preregistered population/schedule/retention differs")
    expected_schedule = schedule_receipt(int(config["seed"]))
    if config.get("schedule") != expected_schedule:
        raise QBR1Error("QBR seeded schedule differs")
    expected_ema = qbt.stable_ema_law_identity(qbt.resolve_ema_law(TOTAL_STEPS))
    if qbt.stable_ema_law_identity(config["ema"]) != expected_ema:
        raise QBR1Error("QBR EMA is not the derived 5,000-update LawRef")
    initial_fact = qbt.file_fact(Path(config["initial_state"]["path"]))
    if initial_fact != config["initial_state"]:
        raise QBR1Error("QBR same-start state drifted")
    if config["source_pins"] != verify_inputs():
        raise QBR1Error("QBR source pins differ from live exact inputs")
    if require_launch_authority:
        if config.get("device") != "mps" or config.get("launch_authorized") is not True:
            raise QBR1Error("MAIN has not authorized this Metal burn")
        for lane in ("scorer_lane", "metal_lane"):
            claim = config.get(lane, {})
            if claim.get("claimed") is not True or not claim.get("claim_id"):
                raise QBR1Error(f"MAIN has not bound a live {lane} claim")


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    lookup = {int(row["pair_id"]): float(row[key]) for row in rows}
    if tuple(lookup) != qbt.SELECTION_IDS:
        raise QBR1Error("realized milestone pair order differs")
    return (
        sum(weight * lookup[pair_id] for pair_id, weight in zip(qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS, strict=True))
        / qbt.N
    )


def fairform_objective(
    config: Mapping[str, Any],
    outputs: Mapping[str, torch.Tensor],
    camera: torch.Tensor,
    pose6: torch.Tensor,
    logits: torch.Tensor,
    target_argmax: torch.Tensor,
    target_pose6: torch.Tensor,
    tau: float,
    sample_weights: torch.Tensor,
    lambdas: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    realized = qbt.expected_flip_margin_loss(logits, target_argmax, tau, sample_weights)
    native = qbt.expected_flip_margin_loss(
        outputs["class_logits"].permute(0, 3, 1, 2), target_argmax, tau, sample_weights
    )
    pose_per_sample = (pose6 - target_pose6).square().mean(dim=1)
    weights = sample_weights.to(pose_per_sample)
    pose_mse = (pose_per_sample * weights).sum() / weights.sum()
    pose_score = torch.sqrt(torch.clamp(10.0 * pose_mse, min=1.0e-20))
    total = (
        float(config["objective"]["realized_weight"]) * realized
        + float(config["objective"]["native_interface_weight"]) * native
        + pose_score
    )
    components: dict[str, torch.Tensor] = {
        "loss_total": total,
        "seg_expected_flip_realized": realized,
        "seg_expected_flip_native_interface": native,
        "pose_mse_realized": pose_mse,
        "pose_score_realized": pose_score,
        "tau": total.new_tensor(tau),
        "camera_min": camera.detach().amin(),
        "camera_max": camera.detach().amax(),
    }
    penalty = total.new_zeros(())
    for class_name, class_id in (("Lane", 1), ("Movable", 3)):
        class_flip = qbt.per_class_expected_flip_margin_loss(logits, target_argmax, tau, class_id, sample_weights)
        class_penalty = 100.0 * float(lambdas[class_name]) * class_flip
        penalty = penalty + class_penalty
        components[f"margin_constraint_expected_flip_{class_name}"] = class_flip
        components[f"margin_constraint_penalty_score_{class_name}"] = class_penalty
    total = total + penalty
    components["margin_constraint_penalty_score"] = penalty
    components["loss_total"] = total
    return total, components


def _append_history(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: qbt.EMA,
    completed_steps: int,
    lambdas: Mapping[str, float],
    history_path: Path,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "stage_01_fairform_finish",
        "completed_steps": int(completed_steps),
        "config_identity": config_identity(config),
        "config_identity_sha256": canonical_sha256(config_identity(config)),
        "live_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "ema": qbt._ema_payload(ema),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng": qbt._rng_payload(),
        "margin_constraint_lambdas": dict(lambdas),
        "history_prefix": qbt.file_fact(history_path),
        "schedule_completion_semantics": "range(completed_steps,total_steps); never add total_steps",
    }
    return qbt.atomic_torch(path, payload)


def _load_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, qbt.EMA, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise QBR1Error("resume checkpoint schema differs")
    if payload.get("config_identity") != config_identity(config):
        raise QBR1Error("resume checkpoint scientific identity differs")
    model.load_state_dict(payload["live_state_dict"], strict=True)
    ema = qbt._restore_ema(model, payload["ema"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    qbt._restore_rng(payload["rng"])
    history_path = Path(payload["history_prefix"]["path"])
    prefix_bytes = int(payload["history_prefix"]["bytes"])
    live = history_path.read_bytes()
    if len(live) < prefix_bytes:
        raise QBR1Error("resume history sidecar is shorter than its checkpoint prefix")
    prefix = live[:prefix_bytes]
    if hashlib.sha256(prefix).hexdigest() != payload["history_prefix"]["sha256"]:
        raise QBR1Error("resume history sidecar checkpoint prefix drifted")
    if len(live) != prefix_bytes:
        qbt.atomic_bytes(history_path, prefix)
    return (
        int(payload["completed_steps"]),
        ema,
        {name: float(value) for name, value in payload["margin_constraint_lambdas"].items()},
    )


def _evaluate_milestone(
    root: Path,
    *,
    config: Mapping[str, Any],
    model: torch.nn.Module,
    ema: qbt.EMA,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    step: int,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    with qbt.ema_scope(model, ema), torch.no_grad():
        for chunk_ids in qbt.pair_chunks(qbt.SELECTION_IDS, 16):
            ids = torch.tensor(chunk_ids, dtype=torch.long, device=device)
            target_argmax, target_pose6 = qbt._target_arrays(chunk_ids, device)
            outputs = model(ids, height=qbt.EVAL_H, width=qbt.EVAL_W)
            camera = qbt.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
            retained = qbt._retain_eval_outputs(
                root / "realized",
                pair_ids=chunk_ids,
                camera=camera,
                pose6=pose6,
                logits=logits,
                target_argmax=target_argmax,
                target_pose6=target_pose6,
            )
            pair_rows.extend(retained["rows"])
    reencode = qbt.reencode_inference_state(
        root / "reencoded",
        model=model,
        state=ema.shadow,
        selected_pair_ids=qbt.SELECTION_IDS,
        consolidate=True,
    )
    d_seg = _weighted_mean(pair_rows, "d_seg")
    d_pose = _weighted_mean(pair_rows, "d_pose")
    archive_bytes = int(reencode["archive"]["bytes"])
    rate = 25.0 * archive_bytes / RATE_DENOMINATOR
    pose_allowance = TARGET_SCORE - rate - 100.0 * d_seg
    pose_corner_max = pose_allowance**2 / 10.0 if pose_allowance > 0 else None
    row = {
        "schema": "ddm_qbr1_realized_milestone.v1",
        "axis": "[macOS-MPS n32 stratified advisory; not contest authority]",
        "score_claim": False,
        "step": int(step),
        "selection_n": 32,
        "population_n": 600,
        "selection_mode": "fixed no2 stratified Horvitz-Thompson",
        "d_seg_hat": d_seg,
        "d_pose_hat": d_pose,
        "archive_bytes_exact": archive_bytes,
        "rate_exact": rate,
        "S_hat": 100.0 * d_seg + math.sqrt(10.0 * d_pose) + rate,
        "pose_corner_max_d_pose": pose_corner_max,
        "pose_corner_pass": pose_corner_max is not None and d_pose < pose_corner_max,
        "pair_rows": pair_rows,
        "reencode": reencode,
        "all_materialized_frames_scorer_outputs_and_coder_payloads_retained": True,
    }
    qbt.atomic_json(root / "MILESTONE.json", row)
    return row


def run_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, require_launch_authority=True)
    output = Path(config["output"])
    storage = qbt.storage_preflight(output, int(config["minimum_free_bytes"]))
    qbt.seed_everything(int(config["seed"]))
    device = torch.device(str(config["device"]))
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    model = qbt.load_initial_model(device)
    initialization = torch.load(Path(config["initial_state"]["path"]), map_location="cpu", weights_only=False)
    model.load_state_dict(
        {name: value.detach().clone().to(device) for name, value in initialization["state_dict"].items()},
        strict=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    ema = qbt.EMA(model, decay=float(config["ema"]["value"]), warmup=True)
    completed = 0
    bounds = {name: float(value) for name, value in config["margin_constraints"]["bounds"].items()}
    lambdas = dict.fromkeys(bounds, 0.0)
    history_path = output / "history.jsonl"
    if config.get("resume_from"):
        completed, ema, lambdas = _load_checkpoint(
            Path(config["resume_from"]), config=config, model=model, optimizer=optimizer
        )
    else:
        qbt.atomic_bytes(history_path, b"")
    schedule = schedule_for_seed(int(config["seed"]))
    chunks = qbt.training_chunks(qbt.SELECTION_IDS, 16)
    milestones: list[dict[str, Any]] = []
    if completed == 0:
        milestones.append(
            _evaluate_milestone(
                output / "milestones/step_000000",
                config=config,
                model=model,
                ema=ema,
                posenet=posenet,
                segnet=segnet,
                step=0,
            )
        )
    started = time.monotonic()
    for current in range(completed, TOTAL_STEPS):
        chunk_ids = chunks[schedule[current]]
        ids = torch.tensor(chunk_ids, dtype=torch.long, device=device)
        target_argmax, target_pose6 = qbt._target_arrays(chunk_ids, device)
        sample_weights = qbt.no2_sample_weights(chunk_ids, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(ids, height=qbt.EVAL_H, width=qbt.EVAL_W)
        camera = qbt.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
        pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
        tau = qbt.tau_for_step(
            current,
            TOTAL_STEPS,
            float(config["expected_flip_tau_start"]),
            float(config["expected_flip_tau_end"]),
        )
        realized_werr = {
            "Lane": qbt.realized_within_class_error(logits, target_argmax, 1),
            "Movable": qbt.realized_within_class_error(logits, target_argmax, 3),
        }
        lambdas = qbt.dual_ascent_margin_constraints(
            lambdas,
            realized_werr,
            bounds,
            eta_lambda=float(config["margin_constraints"]["eta_lambda"]),
        )
        total, components = fairform_objective(
            config,
            outputs,
            camera,
            pose6,
            logits,
            target_argmax,
            target_pose6,
            tau,
            sample_weights,
            lambdas,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        completed = current + 1
        _append_history(
            history_path,
            {
                "completed_steps": completed,
                "chunk_index": int(schedule[current]),
                "pair_ids": list(chunk_ids),
                "objective": {name: float(value.detach().cpu()) for name, value in components.items()},
                "realized_within_class_error": realized_werr,
                "margin_constraint_lambdas": dict(lambdas),
                "ema_effective_decay": ema.effective_decay(),
            },
        )
        if completed % int(config["checkpoint_every_steps"]) == 0:
            _save_checkpoint(
                output / "stage_01_fairform_finish/checkpoints" / f"periodic_{completed:06d}.pt",
                config=config,
                model=model,
                optimizer=optimizer,
                ema=ema,
                completed_steps=completed,
                lambdas=lambdas,
                history_path=history_path,
            )
        if completed in MILESTONES:
            milestones.append(
                _evaluate_milestone(
                    output / "milestones" / f"step_{completed:06d}",
                    config=config,
                    model=model,
                    ema=ema,
                    posenet=posenet,
                    segnet=segnet,
                    step=completed,
                )
            )
    stage_end = _save_checkpoint(
        output / "stage_01_fairform_finish/checkpoints/stage_01_end.pt",
        config=config,
        model=model,
        optimizer=optimizer,
        ema=ema,
        completed_steps=completed,
        lambdas=lambdas,
        history_path=history_path,
    )
    complete_milestones = []
    for milestone in MILESTONES:
        milestone_path = output / "milestones" / f"step_{milestone:06d}" / "MILESTONE.json"
        if not milestone_path.is_file():
            raise QBR1Error(f"completed run lacks preregistered milestone {milestone}")
        complete_milestones.append(json.loads(milestone_path.read_text(encoding="utf-8")))
    result = {
        "schema": RESULT_SCHEMA,
        "complete": completed == TOTAL_STEPS,
        "cell_id": config["cell_id"],
        "axis": "[macOS-MPS n32 stratified advisory; not contest authority]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "completed_steps": completed,
        "schedule_completion_semantics": "completed exactly total_steps; resume never extends",
        "elapsed_seconds_this_process": time.monotonic() - started,
        "storage_preflight": storage,
        "stage_end_checkpoint": stage_end,
        "milestones": complete_milestones,
        "history": qbt.file_fact(history_path),
        "all_payloads_retained": True,
        "metal_invocations": 1,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
    }
    qbt.atomic_json(output / "RESULT.json", result)
    return result


def _r10_timing() -> dict[str, Any]:
    status = json.loads(R10_STATUS.read_text(encoding="utf-8"))
    config = json.loads(R10_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(R10_RESULT.read_text(encoding="utf-8"))
    elapsed = float(status["elapsed_s"])
    updates = len(result["history"])
    if updates != 10_010 or int(config["steps"]) != 10_020:
        raise QBR1Error("r10 realized-versus-authorized update geometry drifted")
    seconds_per_update = elapsed / updates
    finish_seconds = seconds_per_update * TOTAL_STEPS * len(SEEDS) * len(ARMS)
    scorer_seconds = 484.769 * len(SEEDS) * len(ARMS)
    return {
        "source_status": qbt.file_fact(R10_STATUS),
        "source_config": qbt.file_fact(R10_CONFIG),
        "source_result": qbt.file_fact(R10_RESULT),
        "r10_elapsed_seconds": elapsed,
        "r10_optimizer_updates": updates,
        "r10_authorized_step_cap": int(config["steps"]),
        "seconds_per_optimizer_update": seconds_per_update,
        "six_finish_seconds": finish_seconds,
        "six_finish_hours": finish_seconds / 3600.0,
        "br2_realization_seconds_each": 484.769,
        "six_realization_seconds": scorer_seconds,
        "six_realization_hours": scorer_seconds / 3600.0,
        "total_hours_before_build_overhead": (finish_seconds + scorer_seconds) / 3600.0,
    }


def _storage_projection() -> dict[str, Any]:
    status = json.loads(R10_STATUS.read_text(encoding="utf-8"))
    measured_peak = int(float(status["peak_rss_mib"]) * 1024**2)
    eval_payload_per_milestone = 121_825_988
    checkpoint_bytes = int(R10_CHECKPOINT.stat().st_size)
    checkpoints_per_cell = math.ceil(TOTAL_STEPS / CHECKPOINT_EVERY) + 1
    milestones_per_cell = len(MILESTONES)
    coder_payload_allowance = 8 * 1024**2
    projected_per_cell = (
        checkpoint_bytes * checkpoints_per_cell
        + eval_payload_per_milestone * milestones_per_cell
        + coder_payload_allowance * milestones_per_cell
    )
    projected_all = projected_per_cell * len(SEEDS) * len(ARMS)
    free = os.statvfs(AP_ROOT.parent)
    available = int(free.f_bavail * free.f_frsize)
    reserve = 8 * 1024**3
    return {
        "schema": "ddm_qbr1_real_config_preflight.v1",
        "memory": {
            "status": "PASS_MEASURED_REAL_B16_PARENT_CONFIG",
            "source": qbt.file_fact(R10_STATUS),
            "measured_chunk_pairs": 16,
            "measured_peak_rss_bytes": measured_peak,
            "host_ceiling_bytes": 116 * 1024**3,
            "control_graph_matches_r10_100_100": True,
            "treatment_graph_removes_one weighted term": True,
            "fresh_scorer_smoke_run_by_arm": False,
            "reason": "the scorer lane belongs to MAIN; retained r10 is the exact B=16 parent graph",
        },
        "storage": {
            "status": "PASS" if available - projected_all >= reserve else "BLOCKED",
            "tier": str(AP_ROOT),
            "available_bytes": available,
            "projected_checkpoint_bytes_each": checkpoint_bytes,
            "projected_checkpoints_per_cell": checkpoints_per_cell,
            "measured_n32_eval_payload_bytes_per_milestone": eval_payload_per_milestone,
            "milestones_per_cell": milestones_per_cell,
            "projected_bytes_per_cell": projected_per_cell,
            "projected_bytes_all_six_cells": projected_all,
            "required_post_projection_reserve_bytes": reserve,
        },
    }


def build(review_receipt: Path) -> dict[str, Any]:
    inputs = verify_inputs()
    review = json.loads(review_receipt.read_text(encoding="utf-8"))
    if review.get("two_genuine_passes") is not True:
        raise QBR1Error("two-pass review receipt is absent")
    AP_ROOT.mkdir(parents=True, exist_ok=True)
    initial = build_initial_state()
    configs = []
    for seed in SEEDS:
        for arm_name in ARMS:
            config = compile_cell(seed, arm_name, initial)
            path = CONFIG_ROOT / f"{config['cell_id']}.json"
            fact = qbt.atomic_json(path, config)
            if json.loads(path.read_text(encoding="utf-8")) != config:
                raise QBR1Error("sealed config JSON round trip differs")
            configs.append({"cell_id": config["cell_id"], "config": fact})
    preflight = _storage_projection()
    preflight_fact = qbt.atomic_json(AP_ROOT / "REAL_CONFIG_PREFLIGHT.json", preflight)
    if preflight["storage"]["status"] != "PASS":
        raise QBR1Error("six-cell storage projection does not preserve the 8 GiB reserve")
    timing = _r10_timing()
    timing_fact = qbt.atomic_json(AP_ROOT / "REDERIVED_TIMING.json", timing)
    cells = []
    for order, row in enumerate(configs, start=1):
        config_path = row["config"]["path"]
        launch_root = AP_ROOT / "launch" / row["cell_id"]
        cells.append(
            {
                "order": order,
                "cell_id": row["cell_id"],
                "config": row["config"],
                "disposition": "SEALED_AWAITING_MAIN_LIVE_CLAIMS",
                "owner": "MAIN",
                "consumer_store": str(Path(config_path).parent.parent / "runs"),
                "fire_trigger": (
                    "MAIN has consumed qxr1/QXO1 realized distortion, confirms this row "
                    "remains non-dominated, then binds unique live scorer and Metal claims"
                ),
                "claim_mutation": (
                    "copy config to an authorized path; set launch_authorized=true and bind "
                    "scorer_lane.claimed/claim_id and metal_lane.claimed/claim_id; retain copy"
                ),
                "launcher_argv": [
                    str(REPO / ".venv/bin/python"),
                    str(REPO / "tools/launch_detached_process.py"),
                    "--output-dir",
                    str(launch_root),
                    "--cwd",
                    str(REPO),
                    "--purpose",
                    f"QBR1 governed fair-form cell {row['cell_id']}",
                    "--authority",
                    "MAIN",
                    "--derive-resource-budgets",
                    "--measured-peak-rss-gib",
                    str(preflight["memory"]["measured_peak_rss_bytes"] / 1024**3),
                    "--measured-thread-need",
                    "4",
                    "--walltime-cap-s",
                    "18000",
                    "--done-receipt",
                    "DONE.json",
                    "--",
                    str(REPO / ".venv/bin/python"),
                    str(Path(__file__).resolve()),
                    "run-config",
                    "AUTHORIZED_CONFIG_PATH",
                ],
            }
        )
    adjudication = {
        "schema": "ddm_qbr1_preregistered_adjudication.v1",
        "endpoint_step": TOTAL_STEPS,
        "primary_metric": "n32 stratified HT S_hat using each cell's exact QBF1 archive bytes",
        "per_seed_treatment_win": "treatment_zero_native.S_hat < control_native100.S_hat",
        "optimization_live": "treatment wins at least 2 of 3 seeds AND pose corner passes at least 2 of 3 treatment cells",
        "optimization_closed": "treatment wins 0 of 3 seeds OR pose corner passes 0 of 3 treatment cells",
        "mixed": "all remaining outcomes are INCONCLUSIVE_MIXED; no family closure",
        "pose_corner": "d_pose_hat < (0.12-rate_exact-100*d_seg_hat)^2/10 with positive numerator",
        "no_n600_buy_before_sign_repeats": True,
        "milestones": list(MILESTONES),
        "selection_n": 32,
        "population_n": 600,
        "selection_mode": "fixed no2 stratified Horvitz-Thompson",
    }
    adjudication_fact = qbt.atomic_json(AP_ROOT / "ADJUDICATION_SCHEMA.json", adjudication)
    fire_order = {
        "schema": "ddm_qbr1_sealed_main_fire_order.v1",
        "status": "SEALED_AWAITING_MAIN_LIVE_CLAIMS_AND_RESUME_SMOKE",
        "score_claim": False,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "source_revision": source_revision(),
        "source": qbt.file_fact(Path(__file__)),
        "review_receipt": qbt.file_fact(review_receipt),
        "inputs": inputs,
        "initial_state": initial,
        "real_config_preflight": preflight_fact,
        "timing": timing_fact,
        "adjudication": adjudication_fact,
        "bounded_resume_smoke": {
            "status": "OWED_TO_MAIN_SCORER_LANE",
            "disposition": "SEALED_BLOCKED_ON_MAIN_SCORER_LANE",
            "owner": "MAIN",
            "consumer_store": str(AP_ROOT / "resume_smoke"),
            "fire_trigger": "MAIN binds the scorer lane before the first six-cell burn",
            "acceptance": (
                "interrupt after one update, resume to total_steps=2, compare against an "
                "uninterrupted two-update run; final live/EMA/archive hashes equal and cursor=2"
            ),
            "real_chunk_pairs": 16,
        },
        "cells": cells,
    }
    fire_fact = qbt.atomic_json(AP_ROOT / "SEALED_MAIN_FIRE_ORDER.json", fire_order)
    receipt = {
        "schema": "ddm_qbr1_build_receipt.v1",
        "complete": False,
        "status": "BUILD_COMPLETE_SEAL_CONDITIONAL_ON_MAIN_RESUME_SMOKE_BURN_NOT_FIRED",
        "score_claim": False,
        "training_launched": False,
        "configs": configs,
        "fire_order": fire_fact,
        "real_config_preflight": preflight_fact,
        "timing": timing_fact,
        "adjudication": adjudication_fact,
        "all_materialized_build_payloads_retained": True,
    }
    qbt.atomic_json(AP_ROOT / "BUILD_RECEIPT.json", receipt)
    return receipt


def adjudicate(result_paths: Sequence[Path], output: Path) -> dict[str, Any]:
    if len(result_paths) != 6:
        raise QBR1Error("adjudication needs exactly six cell results")
    cells: dict[tuple[int, str], Mapping[str, Any]] = {}
    for path in result_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != RESULT_SCHEMA or row.get("complete") is not True:
            raise QBR1Error("adjudication input is not a complete QBR cell")
        parts = str(row["cell_id"]).split("_", 2)
        key = (int(parts[1]), parts[2])
        cells[key] = row
    seed_rows = []
    wins = 0
    pose_passes = 0
    for seed in SEEDS:
        control = cells[(seed, "control_native100")]["milestones"][-1]
        treatment = cells[(seed, "treatment_zero_native")]["milestones"][-1]
        win = float(treatment["S_hat"]) < float(control["S_hat"])
        pose_pass = bool(treatment["pose_corner_pass"])
        wins += int(win)
        pose_passes += int(pose_pass)
        seed_rows.append(
            {
                "seed": seed,
                "control_S_hat": control["S_hat"],
                "treatment_S_hat": treatment["S_hat"],
                "delta_treatment_minus_control": treatment["S_hat"] - control["S_hat"],
                "treatment_win": win,
                "treatment_pose_corner_pass": pose_pass,
            }
        )
    if wins >= 2 and pose_passes >= 2:
        disposition = "OPTIMIZATION_LIVE_DISTORTION_ROUTE"
    elif wins == 0 or pose_passes == 0:
        disposition = "OPTIMIZATION_CLOSED_CHANGED_CAPACITY_OBJECT_ONLY"
    else:
        disposition = "INCONCLUSIVE_MIXED_NO_FAMILY_CLOSURE"
    result = {
        "schema": "ddm_qbr1_adjudication_result.v1",
        "axis": "[macOS-MPS n32 stratified advisory; not contest authority]",
        "score_claim": False,
        "disposition": disposition,
        "treatment_wins": wins,
        "treatment_pose_corner_passes": pose_passes,
        "seed_rows": seed_rows,
        "source_results": [qbt.file_fact(path) for path in result_paths],
    }
    qbt.atomic_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--review-receipt", type=Path, required=True)
    validate_cmd = sub.add_parser("validate-config")
    validate_cmd.add_argument("config", type=Path)
    run_cmd = sub.add_parser("run-config")
    run_cmd.add_argument("config", type=Path)
    adjudicate_cmd = sub.add_parser("adjudicate")
    adjudicate_cmd.add_argument("--output", type=Path, required=True)
    adjudicate_cmd.add_argument("results", nargs=6, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "build":
        result = build(args.review_receipt)
    elif args.action == "validate-config":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config, require_launch_authority=False)
        result = {"status": "PASS", "config_sha256": canonical_sha256(config)}
    elif args.action == "run-config":
        result = run_config(args.config)
    else:
        result = adjudicate(args.results, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
