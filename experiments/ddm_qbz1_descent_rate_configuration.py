# SPDX-License-Identifier: MIT
"""Full-n supervised capacity probe for the frozen QBF1/qbt2b schema.

The fit is deliberately scorer-free.  It teaches the landed QBF1 field from
the registered DALI partition and the inherited, counted FP1 RGB palette, then
retains the quantized packet and every native field it materializes.  A
separate, fail-closed ``realize`` action is the only path that loads the frozen
scorers; MAIN must own an active local scorer claim before invoking it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1
from tac.gt_lineage import AUTHORITY_LINEAGE, assert_gt_lineage
from tac.scorer import load_differentiable_scorers

SCHEMA = "ddm_qbz1_supervised_capacity_fit.v1"
CHECKPOINT_SCHEMA = "ddm_qbz1_supervised_capacity_checkpoint.v1"
REALIZATION_SCHEMA = "ddm_qbz1_frozen_scorer_realization.v1"
SEED = 20260829
N = 600
H, W = 384, 512
RATE_DENOMINATOR = 37_545_489
OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbz1_descent_rate_configuration")
QB_ROOT = Path("/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer")
R10_CHECKPOINT = QB_ROOT / "governed_n32_r10/stage_05_same_budget_admission/checkpoints/stage_05_end.pt"
GT_ARGMAX = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")
GT_POSE6 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy"
)
FP1_PALETTE = Path("/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/prototypes.npz")
ACTIVE_CLAIMS = REPO / ".omx/state/active_lane_dispatch_claims.md"
R10_CHECKPOINT_SHA256 = "bf0a3a64b3f9ff59e1662d1c9676aa8c249f1d32738a8ed9cf967625e08c2f75"
GT_ARGMAX_SHA256 = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
GT_POSE6_SHA256 = "8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff"
FP1_PALETTE_SHA256 = "19e6524b75724f0b19f0e2e49a827d9f28b40d087b1e5504c3a85577a9e76f0b"
SOURCE_RUNS = (7, 8, 9, 10)
PAIR_HOLDOUT_COUNT = 120
PAIR_HOLDOUT_EPOCHS = 2
SPATIAL_HOLDOUT_EPOCHS = 10
SPATIAL_HOLDOUT_MODULUS = 5
DEFAULT_LR = 2.0e-4
PALETTE_MSE_WEIGHT = 8.0
MINIMUM_FREE_BYTES = 5_000_000_000
EVAL_CHUNK = 30


class QBZ1Error(RuntimeError):
    """Fail-closed refusal for custody, lineage, resume, or lane violations."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QBZ1Error(f"required file is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def source_facts() -> dict[str, Any]:
    assert_gt_lineage(
        GT_ARGMAX,
        required=AUTHORITY_LINEAGE,
        instrument="ddm_qbz1 supervised partition fit",
    )
    expected = {
        "r10_checkpoint": (R10_CHECKPOINT, R10_CHECKPOINT_SHA256),
        "gt_argmax": (GT_ARGMAX, GT_ARGMAX_SHA256),
        "gt_pose6": (GT_POSE6, GT_POSE6_SHA256),
        "fp1_palette": (FP1_PALETTE, FP1_PALETTE_SHA256),
    }
    facts: dict[str, Any] = {}
    for name, (path, digest) in expected.items():
        fact = file_fact(path)
        if fact["sha256"] != digest:
            raise QBZ1Error(f"frozen source drifted: {name}")
        facts[name] = fact
    facts["trainer_module"] = file_fact(Path(qbt1.__file__).resolve())
    facts["packet_module"] = file_fact(Path(qbf1.__file__).resolve())
    facts["instrument"] = file_fact(Path(__file__).resolve())
    return facts


def storage_preflight(output: Path, required: int = MINIMUM_FREE_BYTES) -> dict[str, Any]:
    resolved = output.resolve()
    authorized = OUTPUT_ROOT.resolve()
    if resolved != authorized and authorized not in resolved.parents:
        raise QBZ1Error(f"output is outside qbz1 AP custody: {resolved}")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < int(required):
        raise QBZ1Error(f"AP storage preflight refused: free={usage.free} required={required}")
    return {
        "root": str(resolved),
        "free_bytes": usage.free,
        "required_free_bytes": int(required),
        "status": "PASS",
        "cleanup": "certify-or-block; fitted packets, checkpoints, and materialized fields are retained",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_r10_model() -> qbt1.QBFLOWTorch:
    payload = torch.load(R10_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("schema") != qbt1.CHECKPOINT_SCHEMA:
        raise QBZ1Error("r10 checkpoint schema differs")
    shadow = payload.get("ema", {}).get("shadow")
    if not isinstance(shadow, Mapping):
        raise QBZ1Error("r10 EMA shadow is absent")
    params = {
        name: shadow[f"params.{name}"].detach().cpu().numpy().astype(np.float32, copy=True)
        for name in qbf1.expected_param_shapes()
    }
    boundary = shadow["boundary_latents"].detach().cpu().numpy().astype(np.float32, copy=True)
    interior = shadow["interior_latents"].detach().cpu().numpy().astype(np.float32, copy=True)
    model = qbt1.QBFLOWTorch(params, boundary, interior)
    model.load_state_dict({name: tensor.detach().cpu().clone() for name, tensor in shadow.items()}, strict=True)
    return model


def load_palette() -> torch.Tensor:
    with np.load(FP1_PALETTE, allow_pickle=False) as payload:
        palette = np.asarray(payload["proto_solved"], dtype=np.float32)
    if palette.shape != (qbf1.N_CLASSES, 3) or not np.isfinite(palette).all():
        raise QBZ1Error("FP1 palette geometry differs")
    return torch.from_numpy(palette / np.float32(255.0))


def split_pair_ids(seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = np.random.default_rng(seed).permutation(N)
    holdout = tuple(sorted(map(int, order[:PAIR_HOLDOUT_COUNT])))
    train = tuple(sorted(map(int, order[PAIR_HOLDOUT_COUNT:])))
    if len(train) != 480 or len(holdout) != 120 or set(train) & set(holdout):
        raise AssertionError("deterministic pair split differs")
    return train, holdout


def spatial_holdout_mask(pair_id: int) -> torch.Tensor:
    y = torch.arange(H, dtype=torch.int64)[:, None]
    x = torch.arange(W, dtype=torch.int64)[None, :]
    mixed = (x * 73_856_093) ^ (y * 19_349_663) ^ (int(pair_id) * 83_492_791) ^ SEED
    return torch.remainder(mixed, SPATIAL_HOLDOUT_MODULUS) == 0


def schedule(seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_ids, _holdout_ids = split_pair_ids(seed)
    rows: list[dict[str, Any]] = []
    boundaries: dict[str, int] = {}
    for phase, ids, epochs in (
        ("pair_holdout", train_ids, PAIR_HOLDOUT_EPOCHS),
        ("spatial_holdout", tuple(range(N)), SPATIAL_HOLDOUT_EPOCHS),
    ):
        for epoch in range(epochs):
            ordered = np.random.default_rng(seed + len(rows) + 1009 * (epoch + 1)).permutation(ids)
            for pair_id in ordered:
                rows.append({"phase": phase, "epoch": epoch + 1, "pair_id": int(pair_id)})
            boundaries[f"{phase}_epoch_{epoch + 1:02d}"] = len(rows)
    return rows, boundaries


def fit_config(output: Path, learning_rate: float) -> dict[str, Any]:
    train_ids, holdout_ids = split_pair_ids(SEED)
    rows, boundaries = schedule(SEED)
    return {
        "schema": SCHEMA,
        "seed": SEED,
        "output": str(output.resolve()),
        "device": "cpu",
        "n": N,
        "height": H,
        "width": W,
        "learning_rate": float(learning_rate),
        "optimizer": "AdamW(weight_decay=0)",
        "pair_holdout_epochs": PAIR_HOLDOUT_EPOCHS,
        "spatial_holdout_epochs": SPATIAL_HOLDOUT_EPOCHS,
        "spatial_holdout_modulus": SPATIAL_HOLDOUT_MODULUS,
        "pair_train_ids": list(train_ids),
        "pair_holdout_ids": list(holdout_ids),
        "updates": len(rows),
        "stage_boundaries": boundaries,
        "loss": {
            "partition": "unweighted CE(native class logits, DALI GT partition)",
            "renderer": "MSE(last-frame RGB, inherited FP1 palette indexed by DALI GT partition)",
            "palette_mse_weight": PALETTE_MSE_WEIGHT,
            "heldout": "all GT-derived losses exclude the deterministic spatial holdout in phase 2",
        },
        "scope": "full frozen QBF1 schema, real n600, no tensor/section/receiver change",
        "scorers_loaded_during_fit": False,
    }


def _rng_payload() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])


def save_checkpoint(
    path: Path,
    *,
    model: qbt1.QBFLOWTorch,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    cursor: int,
    history: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    return qbt1.atomic_torch(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "config": copy.deepcopy(dict(config)),
            "config_sha256": canonical_sha256(config),
            "cursor": int(cursor),
            "stage": stage,
            "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "rng": _rng_payload(),
            "history": copy.deepcopy(list(history)),
        },
    )


def load_resume(
    path: Path,
    *,
    model: qbt1.QBFLOWTorch,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("config") != dict(config):
        raise QBZ1Error("resume checkpoint schema/config differs")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng"])
    return int(payload["cursor"]), list(payload["history"])


def read_packet_model(container: Path) -> qbt1.QBFLOWTorch:
    with tarfile.open(container, mode="r") as archive:
        member = archive.getmember("packet.qbf")
        stream = archive.extractfile(member)
        if stream is None:
            raise QBZ1Error("retained packet member is unreadable")
        packet_payload = stream.read()
    decoded = qbf1.decode_packet(packet_payload)
    params = qbf1.decode_model(decoded.sections[qbf1.SECTION_MODEL])
    meta = qbf1.decode_latent_meta(decoded.sections[qbf1.SECTION_LATENT_META])
    records = qbf1.decode_latent_table(decoded.sections[qbf1.SECTION_LATENTS])
    if set(records) != set(range(N)):
        raise QBZ1Error("parsed packet does not carry all n600 latent records")
    boundary = np.stack(
        [qbf1.dequantize(records[index][0], meta["boundary_scale"], (qbf1.BOUNDARY_LATENT_DIM,)) for index in range(N)]
    )
    interior = np.stack(
        [qbf1.dequantize(records[index][1], meta["interior_scale"], (qbf1.INTERIOR_LATENT_DIM,)) for index in range(N)]
    )
    return qbt1.QBFLOWTorch(params, boundary, interior)


def reencode_model(root: Path, model: qbt1.QBFLOWTorch) -> tuple[dict[str, Any], qbt1.QBFLOWTorch]:
    manifest = qbt1.reencode_inference_state(
        root,
        model=model,
        state=model.state_dict(),
        selected_pair_ids=qbt1.SELECTION_IDS,
        consolidate=True,
    )
    container = Path(manifest["retention_container"]["path"])
    parsed = read_packet_model(container)
    return manifest, parsed


def evaluate_native(
    root: Path,
    *,
    model: qbt1.QBFLOWTorch,
    pair_ids: Sequence[int],
    gt: np.ndarray,
    spatial_metrics: bool,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    total_errors = total_pixels = 0
    heldout_errors = heldout_pixels = 0
    train_errors = train_pixels = 0
    retained: list[dict[str, Any]] = []
    with torch.no_grad():
        for offset in range(0, len(pair_ids), EVAL_CHUNK):
            chunk_ids = list(map(int, pair_ids[offset : offset + EVAL_CHUNK]))
            logits_parts: list[np.ndarray] = []
            argmax_parts: list[np.ndarray] = []
            target_parts: list[np.ndarray] = []
            rgb_parts: list[np.ndarray] = []
            for pair_id in chunk_ids:
                outputs = model(torch.tensor([pair_id], dtype=torch.long), height=H, width=W)
                logits = outputs["class_logits"][0]
                predicted = logits.argmax(dim=-1).cpu().numpy().astype(np.uint8)
                target = np.asarray(gt[pair_id], dtype=np.uint8)
                errors = predicted != target
                total_errors += int(errors.sum())
                total_pixels += errors.size
                if spatial_metrics:
                    mask = spatial_holdout_mask(pair_id).numpy()
                    heldout_errors += int(errors[mask].sum())
                    heldout_pixels += int(mask.sum())
                    train_errors += int(errors[~mask].sum())
                    train_pixels += int((~mask).sum())
                logits_parts.append(logits.cpu().numpy().astype("<f2"))
                argmax_parts.append(predicted)
                target_parts.append(target.copy())
                rgb_parts.append(
                    outputs["rgb_pair_01"][0].mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
            fact = qbt1.atomic_npz(
                root / f"native_field_pairs_{chunk_ids[0]:04d}_{chunk_ids[-1]:04d}.npz",
                pair_ids_i64=np.asarray(chunk_ids, dtype=np.int64),
                native_class_logits_f16=np.stack(logits_parts),
                native_argmax_u8=np.stack(argmax_parts),
                target_argmax_u8=np.stack(target_parts),
                rgb_pair_evalgrid_u8=np.stack(rgb_parts),
            )
            retained.append(fact)
    result = {
        "axis": "[macOS-CPU scorer-free native-field advisory]",
        "score_claim": False,
        "pair_count": len(pair_ids),
        "pixel_count": total_pixels,
        "native_d_seg": total_errors / total_pixels,
        "retained_chunks": retained,
        "all_materialized_native_logits_argmax_targets_and_rgb_retained": True,
    }
    if spatial_metrics:
        result["spatial_holdout_d_seg"] = heldout_errors / heldout_pixels
        result["spatial_train_d_seg"] = train_errors / train_pixels
        result["spatial_holdout_pixels"] = heldout_pixels
        result["spatial_train_pixels"] = train_pixels
    return result


def rederive_source_series() -> dict[str, Any]:
    rows = []
    alignment_rows = []
    for run in SOURCE_RUNS:
        root = QB_ROOT / f"governed_n32_r{run}"
        gate_path = root / "stage_05_same_budget_admission/GATE.json"
        result_path = root / "RESULT.json"
        gate = json.loads(gate_path.read_text())
        d_seg = float(gate["d_seg_hat"])
        d_pose = float(gate["d_pose_hat"])
        archive_bytes = int(gate["B_hat"])
        distortion = 100.0 * d_seg + math.sqrt(10.0 * d_pose)
        rate = 25.0 * archive_bytes / RATE_DENOMINATOR
        recomputed = distortion + rate
        rows.append(
            {
                "run": f"r{run}",
                "gate": file_fact(gate_path),
                "B_hat": archive_bytes,
                "d_seg_hat": d_seg,
                "d_pose_hat": d_pose,
                "distortion_recomputed": distortion,
                "rate_recomputed": rate,
                "S_recomputed": recomputed,
                "S_stored": float(gate["S_hat"]),
                "S_reproduces_abs_error": abs(recomputed - float(gate["S_hat"])),
                "estimator_status": gate["estimator_status"],
                "selection_count": int(gate["selection_count"]),
                "control_status": gate["control_status"],
                "unmet_gates": sorted(name for name, passed in gate["gates"].items() if not passed),
            }
        )
        result = json.loads(result_path.read_text())
        birth = [row for row in result["history"] if row["stage"] == "stage_03a_ce_class_birth"]
        margin = [row for row in result["history"] if row["stage"] == "stage_03_joint_boundary_interior_birth"]
        integrated = {
            "realized_seg_100x": sum(100.0 * float(row["objective"]["seg_expected_flip_realized"]) for row in margin),
            "native_interface_seg_100x": sum(
                100.0 * float(row["objective"]["seg_expected_flip_native_interface"]) for row in margin
            ),
            "pose_score": sum(float(row["objective"]["pose_score_realized"]) for row in margin),
            "constraint_penalty": sum(float(row["objective"].get("margin_constraint_penalty_score", 0.0)) for row in margin),
        }
        denominator = sum(integrated.values())
        alignment_rows.append(
            {
                "run": f"r{run}",
                "result": file_fact(result_path),
                "birth_updates": len(birth),
                "margin_updates": len(margin),
                "birth_update_share": len(birth) / (len(birth) + len(margin)),
                "integrated_margin_objective_components": integrated,
                "integrated_margin_objective_shares": {
                    name: value / denominator for name, value in integrated.items()
                },
                "nominal_seg_coefficients": {"realized_seg": 100.0, "native_interface_seg": 100.0},
            }
        )
    return {
        "schema": "ddm_qbz1_source_rederivation.v1",
        "axis": "[macOS-CPU advisory re-derivation of retained n32 HT rows]",
        "score_claim": False,
        "rows": rows,
        "alignment_rows": alignment_rows,
        "level_comparison_refused": True,
        "refusal_reason": "all four n32 HT rows lack three admission gates, including a real same-budget QBW1 control",
    }


def run_fit(output: Path, *, learning_rate: float, resume_from: Path | None) -> dict[str, Any]:
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise QBZ1Error("learning rate must be finite and positive")
    started = time.time()
    storage = storage_preflight(output)
    facts = source_facts()
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    gt = np.load(GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    if gt.shape != (N, H, W) or gt.dtype != np.uint8:
        raise QBZ1Error("DALI GT partition geometry differs")
    palette = load_palette()
    model = load_r10_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    config = fit_config(output, learning_rate)
    qbt1.atomic_json(output / "FIT_CONFIG.json", config)
    plan, boundaries = schedule(SEED)
    cursor = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        cursor, history = load_resume(resume_from, model=model, optimizer=optimizer, config=config)
    boundary_by_cursor = {value: name for name, value in boundaries.items()}
    recent_losses: list[float] = []

    pair_phase_end = boundaries[f"pair_holdout_epoch_{PAIR_HOLDOUT_EPOCHS:02d}"]
    phase_receipt = output / "PAIR_HOLDOUT_EVAL.json"
    if cursor > pair_phase_end and not phase_receipt.exists():
        raise QBZ1Error(
            "resume is past the pair-holdout boundary but its retained evaluation is absent"
        )
    if cursor == pair_phase_end and not phase_receipt.exists():
        manifest, parsed = reencode_model(output / "pair_holdout_reencode", model)
        pair_eval = evaluate_native(
            output / "pair_holdout_native_field",
            model=parsed,
            pair_ids=split_pair_ids(SEED)[1],
            gt=gt,
            spatial_metrics=False,
        )
        qbt1.atomic_json(
            phase_receipt,
            {
                "schema": "ddm_qbz1_pair_holdout_control.v1",
                "scope": "120 deterministic pair holdouts after training shared field and 480 train-pair latents",
                "reencode": manifest,
                "evaluation": pair_eval,
            },
        )
    for index in range(cursor, len(plan)):
        row = plan[index]
        pair_id = int(row["pair_id"])
        target = torch.from_numpy(np.asarray(gt[pair_id], dtype=np.int64).copy()).unsqueeze(0)
        outputs = model(torch.tensor([pair_id], dtype=torch.long), height=H, width=W)
        logits = outputs["class_logits"].permute(0, 3, 1, 2)
        ce_map = F.cross_entropy(logits, target, reduction="none")[0]
        target_rgb = palette[target[0]].permute(2, 0, 1)
        palette_map = (outputs["rgb_pair_01"][0, 1] - target_rgb).square().mean(dim=0)
        if row["phase"] == "spatial_holdout":
            train_mask = ~spatial_holdout_mask(pair_id)
            ce = ce_map[train_mask].mean()
            palette_mse = palette_map[train_mask].mean()
        else:
            ce = ce_map.mean()
            palette_mse = palette_map.mean()
        loss = ce + PALETTE_MSE_WEIGHT * palette_mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent_losses.append(float(loss.detach()))
        next_cursor = index + 1
        if next_cursor % 100 == 0 or next_cursor in boundary_by_cursor:
            history.append(
                {
                    "cursor": next_cursor,
                    "phase": row["phase"],
                    "epoch": int(row["epoch"]),
                    "loss_mean_since_last_receipt": float(np.mean(recent_losses)),
                    "loss_last": float(loss.detach()),
                    "ce_last": float(ce.detach()),
                    "palette_mse_last": float(palette_mse.detach()),
                    "elapsed_seconds": time.time() - started,
                }
            )
            recent_losses.clear()
        checkpoint_due = next_cursor % 300 == 0
        stage_name = boundary_by_cursor.get(next_cursor)
        if checkpoint_due or stage_name is not None:
            name = f"periodic_step_{next_cursor:06d}.pt" if stage_name is None else f"{stage_name}_end.pt"
            checkpoint = save_checkpoint(
                output / "checkpoints" / name,
                model=model,
                optimizer=optimizer,
                config=config,
                cursor=next_cursor,
                history=history,
                stage=stage_name or str(row["phase"]),
            )
            if history:
                history[-1]["checkpoint"] = checkpoint
        print(
            json.dumps(
                {
                    "cursor": next_cursor,
                    "updates": len(plan),
                    "phase": row["phase"],
                    "epoch": row["epoch"],
                    "loss": float(loss.detach()),
                }
            ),
            flush=True,
        ) if next_cursor % 100 == 0 or stage_name is not None else None

        if next_cursor == pair_phase_end and not phase_receipt.exists():
            manifest, parsed = reencode_model(output / "pair_holdout_reencode", model)
            pair_eval = evaluate_native(
                output / "pair_holdout_native_field",
                model=parsed,
                pair_ids=split_pair_ids(SEED)[1],
                gt=gt,
                spatial_metrics=False,
            )
            qbt1.atomic_json(
                phase_receipt,
                {
                    "schema": "ddm_qbz1_pair_holdout_control.v1",
                    "scope": "120 deterministic pair holdouts after training shared field and 480 train-pair latents",
                    "reencode": manifest,
                    "evaluation": pair_eval,
                },
            )

    final_checkpoint = save_checkpoint(
        output / "checkpoints/final_end.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        cursor=len(plan),
        history=history,
        stage="supervised_capacity_fit_end",
    )
    manifest, parsed = reencode_model(output / "final_reencode", model)
    native = evaluate_native(
        output / "final_native_field",
        model=parsed,
        pair_ids=tuple(range(N)),
        gt=gt,
        spatial_metrics=True,
    )
    pair_control = json.loads((output / "PAIR_HOLDOUT_EVAL.json").read_text())
    result = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU scorer-free native-field advisory]",
        "score_claim": False,
        "promotable": False,
        "complete": True,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "source_facts": facts,
        "storage_preflight": storage,
        "source_series_rederivation": rederive_source_series(),
        "pair_holdout_control": pair_control,
        "final_checkpoint": final_checkpoint,
        "fitted_packet": manifest,
        "native_capacity_measurement": native,
        "elapsed_seconds": time.time() - started,
        "scorers_loaded": False,
        "realized_capacity_ceiling": None,
        "realized_capacity_status": "QUEUED_REQUIRES_MAIN_SCORER_CLAIM",
        "all_fitted_fields_and_materialized_outputs_retained": True,
        "boundaries": [
            "native partition/palette supervision is not a frozen-scorer realization",
            "single seed; family fork is forbidden until the queued real render/R/uint8/scorer terminal lands",
            "r7-r10 levels remain non-comparable n32 HT estimates with three unmet gates",
        ],
        "modal_invocations": 0,
        "metal_invocations": 0,
        "contest_eval_invocations": 0,
        "pointer_moved": False,
    }
    qbt1.atomic_json(output / "FIT_RESULT.json", result)
    return result


def assert_active_scorer_claim(claim_id: str) -> dict[str, Any]:
    if not claim_id.startswith("ddm_qbz1_"):
        raise QBZ1Error("scorer claim must be a qbz1-owned lane id")
    rows: list[dict[str, str]] = []
    for line in ACTIVE_CLAIMS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 8 or not fields[0].startswith("20"):
            continue
        rows.append(
            {
                "timestamp": fields[0],
                "lane_id": fields[2],
                "platform": fields[3],
                "status": fields[6],
                "raw": line,
            }
        )
    newest_by_lane: dict[str, dict[str, str]] = {}
    for row in rows:  # registry contract is newest-first
        newest_by_lane.setdefault(row["lane_id"], row)
    own = newest_by_lane.get(claim_id)
    if (
        own is None
        or not own["status"].startswith("active_")
        or own["platform"] != "local_macos_cpu"
    ):
        raise QBZ1Error("newest qbz1 lane row must be an active local_macos_cpu scorer claim")
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    conflicts = []
    for lane_id, row in newest_by_lane.items():
        if lane_id == claim_id or "scorer" not in lane_id or not row["status"].startswith("active_"):
            continue
        try:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise QBZ1Error(f"active scorer claim timestamp is malformed: {row['timestamp']}") from exc
        if timestamp >= cutoff:
            conflicts.append(row["raw"])
    if conflicts:
        raise QBZ1Error(f"another live scorer claim remains active: {conflicts}")
    return {"claim_id": claim_id, "registry": file_fact(ACTIVE_CLAIMS), "row": own["raw"]}


def run_realize(output: Path, *, claim_id: str, launch_authorized: bool) -> dict[str, Any]:
    if not launch_authorized:
        raise QBZ1Error("frozen-scorer realization requires explicit MAIN launch authorization")
    claim = assert_active_scorer_claim(claim_id)
    storage = storage_preflight(output)
    facts = source_facts()
    assert_gt_lineage(
        GT_POSE6,
        required=AUTHORITY_LINEAGE,
        instrument="ddm_qbz1 frozen-scorer pose realization",
    )
    fit_result = json.loads((output / "FIT_RESULT.json").read_text())
    container = Path(fit_result["fitted_packet"]["retention_container"]["path"])
    model = read_packet_model(container)
    model.eval()
    gt = np.load(GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    pose_target = np.load(GT_POSE6, mmap_mode="r", allow_pickle=False)
    if pose_target.shape != (N, 6):
        raise QBZ1Error("DALI pose target geometry differs")
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=torch.device("cpu"))
    posenet.eval()
    segnet.eval()
    retained: list[dict[str, Any]] = []
    seg_errors = seg_pixels = 0
    pose_squared_error = 0.0
    pose_values = 0
    started = time.time()
    with torch.no_grad():
        for start in range(0, N, EVAL_CHUNK):
            ids = list(range(start, min(N, start + EVAL_CHUNK)))
            camera_parts: list[np.ndarray] = []
            logits_parts: list[np.ndarray] = []
            argmax_parts: list[np.ndarray] = []
            pose_parts: list[np.ndarray] = []
            for pair_id in ids:
                outputs = model(torch.tensor([pair_id], dtype=torch.long), height=H, width=W)
                camera = qbt1.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
                pose6, logits = qbt1.scorer_forward(camera, posenet, segnet)
                predicted = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                target = np.asarray(gt[pair_id], dtype=np.uint8)
                pose_np = pose6[0].cpu().numpy().astype("<f4")
                seg_errors += int((predicted != target).sum())
                seg_pixels += target.size
                pose_squared_error += float(np.square(pose_np.astype(np.float64) - pose_target[pair_id]).sum())
                pose_values += 6
                camera_parts.append(camera[0].round().clamp(0, 255).to(torch.uint8).cpu().numpy())
                logits_parts.append(logits[0].cpu().numpy().astype("<f2"))
                argmax_parts.append(predicted)
                pose_parts.append(pose_np)
            retained.append(
                qbt1.atomic_npz(
                    output / "realized_n600" / f"scorer_pairs_{ids[0]:04d}_{ids[-1]:04d}.npz",
                    pair_ids_i64=np.asarray(ids, dtype=np.int64),
                    camera_pair_u8=np.stack(camera_parts),
                    segnet_logits_f16=np.stack(logits_parts),
                    segnet_argmax_u8=np.stack(argmax_parts),
                    target_argmax_u8=np.asarray(gt[ids], dtype=np.uint8),
                    posenet_pose6_f32=np.stack(pose_parts),
                    target_pose6_f32=np.asarray(pose_target[ids], dtype="<f4"),
                )
            )
            print(json.dumps({"realized_pairs": ids[-1] + 1, "n": N}), flush=True)
    d_seg = seg_errors / seg_pixels
    d_pose = pose_squared_error / pose_values
    distortion = 100.0 * d_seg + math.sqrt(10.0 * d_pose)
    archive_bytes = int(fit_result["fitted_packet"]["B_hat"])
    rate = 25.0 * archive_bytes / RATE_DENOMINATOR
    score = distortion + rate
    if distortion <= 0.067:
        fork = "OPTIMIZATION_LIMITED"
    elif distortion >= 0.30:
        fork = "CAPACITY_LIMITED_FAMILY_CLOSE"
    else:
        fork = "INTERMEDIATE_EXACT_NO_ROUNDED_FORK"
    result = {
        "schema": REALIZATION_SCHEMA,
        "axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "promotable": False,
        "n": N,
        "seed": SEED,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "distortion_recomputed": distortion,
        "B_hat": archive_bytes,
        "rate_recomputed": rate,
        "S_recomputed": score,
        "fork_verdict": fork,
        "fork_scope": "single-seed full-schema qbt2b capacity probe",
        "claim": claim,
        "storage_preflight": storage,
        "source_facts": facts,
        "retained_chunks": retained,
        "all_materialized_camera_frames_and_scorer_outputs_retained": True,
        "elapsed_seconds": time.time() - started,
        "contest_eval_invocations": 0,
        "pointer_moved": False,
    }
    qbt1.atomic_json(output / "REALIZED_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("rederive", help="recompute retained r7-r10 score/alignment facts")
    inspect.add_argument("--output", type=Path, default=OUTPUT_ROOT / "SOURCE_REDERIVATION.json")
    fit = sub.add_parser("fit", help="run/resume the scorer-free full-n supervised capacity fit")
    fit.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    fit.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    fit.add_argument("--resume-from", type=Path)
    realize = sub.add_parser("realize", help="MAIN-only frozen-scorer terminal")
    realize.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    realize.add_argument("--scorer-claim-id", required=True)
    realize.add_argument("--launch-authorized", action="store_true")
    args = parser.parse_args()
    if args.action == "rederive":
        storage_preflight(args.output.parent)
        result = rederive_source_series()
        qbt1.atomic_json(args.output, result)
    elif args.action == "fit":
        result = run_fit(args.output, learning_rate=args.learning_rate, resume_from=args.resume_from)
    else:
        result = run_realize(
            args.output,
            claim_id=args.scorer_claim_id,
            launch_authorized=args.launch_authorized,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
