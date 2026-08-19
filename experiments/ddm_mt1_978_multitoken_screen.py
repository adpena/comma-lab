#!/usr/bin/env python3
"""Resumable n64 receiver-native multi-token representative screen for #978.

This is a bounded advisory screen, not an exact-score evaluator.  It trains on
32 fixed-seed stratified-random pairs and evaluates a disjoint 32-pair heldout
set against both CP135 hard tokens and the direct C1 representative.  Every
materialized receiver/scorer payload is retained on the SSD.  The candidate is
the parsed counted module, and all three arms use the same exact CP135 frame-0
carrier so Pose differences come only from their semantic frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_ec2_oriented_adapter_trainer_worker import (
    load_exact_semantic,
)
from experiments.ddm_mt1_runtime import multitoken_representative as mt1
from experiments.ddm_qs1_frame0_schur_coupled_solve import (
    CP135Surface,
    load_posenet,
    pose_vectors,
)
from tac.differentiable_eval_roundtrip import (
    CameraLiftKernel,
    apply_camera_uint8_lift_during_training,
)
from tac.training import EMA

RUN_ID: Final = "ddm_mt1_978_multitoken_screen_20260814"
OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained"
)
CP135_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"
)
CP135_ARCHIVE: Final = CP135_RUNTIME / "archive.zip"
BASE_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/retained/decoded_tokens_n600.npy"
)
C1_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/"
    "c1_solved_tokens_n600.u8"
)
BG2_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/"
    "bg2_postmortem_r3/decomposition"
)
GT_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1b_20260813b/retained/fields/gt_argmax_n600.npy"
)
BASE_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1b_20260813b/retained/fields/"
    "cp135_base_argmax_n600.npy"
)
# GT LINEAGE (#1142 cure, 2026-08-19): DALI table = the shipping-axis objective; the prior
# gt_first6_n600.npy is the PyAV/advisory lineage (additive gap C = 1.406151e-04).
GT_POSE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy"
)

SEED: Final = 2_026_081_4978
PAIR_COUNT: Final = 600
WORK_HW: Final = (384, 512)
CAMERA_HW: Final = (874, 1164)
CLASSES: Final = 5
TRAIN_PER_STRATUM: Final = 4
HELDOUT_PER_STRATUM: Final = 4
TOP_CELLS: Final = ((5, 8), (5, 9), (6, 12), (5, 7), (6, 3))
HIDDEN: Final = 4
MAX_SUPPORT_MASS: Final = 0.25
STAGES: Final = (
    {
        "name": "10_error_birth",
        "learning_rate": 1.0e-3,
        "correct_weight": 0.25,
    },
    {
        "name": "20_collateral_finish",
        "learning_rate": 3.0e-4,
        "correct_weight": 1.0,
    },
)
TOTAL_STEPS: Final = len(STAGES) * 8 * TRAIN_PER_STRATUM
EMA_DECAY: Final = EMA.decay_from_total_steps(TOTAL_STEPS)
EXPECTED_RETAINED_BYTES: Final = 4 * 1024**3
STORAGE_RESERVE_BYTES: Final = 8 * 1024**3
AXIS: Final = (
    "[macOS-CPU advisory frozen CPU-torch SegNet/PoseNet; "
    "stratified-random n32 heldout] NON-PROMOTABLE"
)


class MT1ScreenError(RuntimeError):
    """A source, receiver, retention, resume, or comparison invariant failed."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _partial(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.partial")


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if path.is_file():
        if file_record(path) != expected:
            raise MT1ScreenError(f"retained payload differs: {path}")
        return expected
    partial = _partial(path)
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def retain_json(path: Path, value: Any) -> dict[str, Any]:
    return retain_bytes(path, canonical_json(value))


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = canonical_json(value)
    partial = _partial(path)
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def retain_npy(path: Path, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    partial = _partial(path)
    try:
        with partial.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        candidate = file_record(partial)
        expected = {**candidate, "path": str(path.resolve())}
        if path.is_file():
            if file_record(path) != expected:
                raise MT1ScreenError(f"retained array differs: {path}")
            return expected
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def atomic_torch(path: Path, value: Any) -> dict[str, Any]:
    partial = _partial(path)
    try:
        torch.save(value, partial)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(path)


def storage_preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    required = max(0, EXPECTED_RETAINED_BYTES - retained) + STORAGE_RESERVE_BYTES
    usage = shutil.disk_usage(root)
    receipt = {
        "schema": "ddm_mt1_storage_preflight.v1",
        "tier": str(root.resolve()),
        "free_bytes": usage.free,
        "already_retained_bytes": retained,
        "expected_total_retained_bytes": EXPECTED_RETAINED_BYTES,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; keep all materialized payloads",
    }
    atomic_json(root / "STORAGE_PREFLIGHT.json", receipt)
    if not receipt["passed"]:
        raise MT1ScreenError(f"storage preflight failed: {receipt}")
    return receipt


def configure_reproducibility() -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    return {
        "seed": SEED,
        "torch_deterministic_algorithms": True,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def load_raw_tokens(path: Path) -> np.ndarray:
    value = np.memmap(path, mode="r", dtype=np.uint8)
    expected = PAIR_COUNT * WORK_HW[0] * WORK_HW[1]
    if value.size != expected:
        raise MT1ScreenError(f"raw token geometry differs: {path} has {value.size}")
    return value.reshape(PAIR_COUNT, *WORK_HW)


def source_records() -> dict[str, Any]:
    sources = {
        "cp135_archive": CP135_ARCHIVE,
        "base_tokens": BASE_TOKENS,
        "c1_tokens": C1_TOKENS,
        "gt_field": GT_FIELD,
        "base_field": BASE_FIELD,
        "bg2_per_frame": BG2_ROOT / "per_frame_counts.int64.npy",
        "bg2_spatial": BG2_ROOT / "spatial_introduced.int64.npy",
        "gt_pose": GT_POSE,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise MT1ScreenError(f"required source files are missing: {missing}")
    return {name: file_record(path) for name, path in sources.items()}


def stratified_split(
    introduced_counts: np.ndarray,
    spatial_introduced: np.ndarray,
) -> dict[str, Any]:
    """Fixed-seed non-prefix 32/32 split over eight BG2 failure strata."""
    per_frame = np.asarray(introduced_counts, dtype=np.int64)
    if per_frame.shape == (PAIR_COUNT, 6):
        if not np.array_equal(per_frame[:, 0], np.arange(PAIR_COUNT)):
            raise MT1ScreenError("BG2 per-frame pair IDs differ")
        counts = per_frame[:, 4]
    else:
        counts = per_frame
    spatial = np.asarray(spatial_introduced, dtype=np.int64)
    if counts.shape != (PAIR_COUNT,) or spatial.shape != (PAIR_COUNT, 12, 16):
        raise MT1ScreenError("BG2 stratification geometry differs")
    cuts = np.quantile(counts, (0.25, 0.5, 0.75), method="linear")
    quartile = np.searchsorted(cuts, counts, side="right").astype(np.int8)
    top_exposure = sum(spatial[:, row, column] for row, column in TOP_CELLS)
    median = float(np.median(top_exposure))
    high = (top_exposure > median).astype(np.int8)
    strata = quartile * 2 + high
    rng = np.random.default_rng(SEED)
    train: list[int] = []
    heldout: list[int] = []
    rows: list[dict[str, Any]] = []
    for stratum in range(8):
        eligible = np.flatnonzero(strata == stratum)
        if len(eligible) < TRAIN_PER_STRATUM + HELDOUT_PER_STRATUM:
            raise MT1ScreenError(f"stratum {stratum} is too small: {len(eligible)}")
        selected = rng.choice(
            eligible,
            size=TRAIN_PER_STRATUM + HELDOUT_PER_STRATUM,
            replace=False,
        )
        train.extend(int(value) for value in selected[:TRAIN_PER_STRATUM])
        heldout.extend(int(value) for value in selected[TRAIN_PER_STRATUM:])
        rows.append(
            {
                "stratum": stratum,
                "introduced_quartile": stratum // 2,
                "top_cell_exposure": "above_median" if stratum % 2 else "at_or_below_median",
                "eligible_n": len(eligible),
                "train": [int(value) for value in selected[:TRAIN_PER_STRATUM]],
                "heldout": [int(value) for value in selected[TRAIN_PER_STRATUM:]],
            }
        )
    if set(train) & set(heldout) or len(train) != 32 or len(heldout) != 32:
        raise MT1ScreenError("stratified train/heldout partition differs")
    if train == list(range(32)) or heldout == list(range(32)):
        raise MT1ScreenError("prefix selection is forbidden")
    return {
        "schema": "ddm_mt1_stratified_random_split.v1",
        "seed": SEED,
        "selection_mode": "stratified-random without replacement; never contiguous-prefix",
        "strata_definition": (
            "BG2 introduced-error count quartile x above-median count in the "
            "pre-registered top-five BG2 spatial cells"
        ),
        "quartile_cuts": [float(value) for value in cuts],
        "top_cells": [list(value) for value in TOP_CELLS],
        "top_cell_exposure_median": median,
        "train": train,
        "heldout": heldout,
        "strata": rows,
    }


def initialize_model() -> mt1.MultiTokenRepresentative:
    torch.manual_seed(SEED)
    model = mt1.MultiTokenRepresentative(HIDDEN, MAX_SUPPORT_MASS)
    nn.init.kaiming_uniform_(model.context.weight, a=math.sqrt(5))
    nn.init.zeros_(model.context.bias)
    nn.init.dirac_(model.depthwise.weight)
    nn.init.zeros_(model.depthwise.bias)
    nn.init.zeros_(model.mass_head.weight)
    nn.init.constant_(model.mass_head.bias, -5.0)
    nn.init.zeros_(model.support_head.weight)
    nn.init.zeros_(model.support_head.bias)
    return model.train()


def load_local_segnet() -> nn.Module:
    from safetensors.torch import load_file

    upstream = REPO / "upstream"
    sys.path.insert(0, str(upstream))
    try:
        from modules import SegNet, segnet_sd_path
    finally:
        sys.path.pop(0)
    scorer = SegNet().eval().cpu()
    scorer.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return scorer


def semantic_receiver(
    semantic: nn.Module,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
    model: mt1.MultiTokenRepresentative | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model is None:
        pre_r = semantic(tokens, pair_indices)
    else:
        probability = model.probability_state(tokens)
        value = torch.einsum("bkhw,kd->bdhw", probability, semantic.token_embed.weight)
        value = semantic.coord_mix(
            torch.cat(
                [
                    value,
                    semantic.coordinates(value.shape[0], value.device, value.dtype),
                ],
                dim=1,
            )
        )
        frame = semantic.frame_embed(pair_indices)
        for block in semantic.blocks:
            value = block(value, frame)
        pre_r = torch.sigmoid(semantic.head(functional.gelu(value))) * 255.0
    camera = apply_camera_uint8_lift_during_training(
        pre_r,
        lift_kernel=CameraLiftKernel.BILINEAR,
        simulate_uint8=True,
        ste_round=True,
    )
    scorer_input = functional.interpolate(
        camera,
        size=WORK_HW,
        mode="bilinear",
        align_corners=False,
    )
    return pre_r, camera, scorer_input


def weighted_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    base_field: torch.Tensor,
    stage: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    base_error = base_field != target
    error_count = max(1, int(torch.count_nonzero(base_error).item()))
    error_weight = (WORK_HW[0] * WORK_HW[1]) / error_count
    weights = torch.where(
        base_error,
        torch.full_like(target, error_weight, dtype=torch.float32),
        torch.full_like(target, float(stage["correct_weight"]), dtype=torch.float32),
    )
    predicted_before_update = logits.detach().argmax(dim=1)
    road_to_lane = (target == 0) & (predicted_before_update == 1)
    weights = torch.where(road_to_lane, weights * 4.0, weights)
    cell_h = WORK_HW[0] // 12
    cell_w = WORK_HW[1] // 16
    hotspot = torch.zeros_like(target, dtype=torch.bool)
    for row, column in TOP_CELLS:
        hotspot[:, row * cell_h : (row + 1) * cell_h, column * cell_w : (column + 1) * cell_w] = True
    weights = torch.where((base_error | road_to_lane) & hotspot, weights * 2.0, weights)
    pixel = functional.cross_entropy(logits, target, reduction="none")
    loss = (weights * pixel).mean()
    predicted = logits.argmax(dim=1)
    return loss, {
        "loss": float(loss.detach()),
        "errors": int(torch.count_nonzero(predicted != target)),
        "fixed_base_errors": int(torch.count_nonzero(base_error & (predicted == target))),
        "introduced_errors": int(torch.count_nonzero((~base_error) & (predicted != target))),
        "road_to_lane_errors_before_update": int(torch.count_nonzero(road_to_lane)),
        "error_weight_derived_from_pair": error_weight,
    }


def retain_train_step(
    root: Path,
    *,
    probability: torch.Tensor,
    pre_r: torch.Tensor,
    camera: torch.Tensor,
    scorer_input: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    base_field: torch.Tensor,
    receipt: dict[str, Any],
) -> None:
    payloads = {
        "probability_state.float32.npy": probability.detach().cpu().numpy(),
        "pre_r.float32.npy": pre_r.detach().cpu().numpy(),
        "camera.uint8.npy": camera.detach().cpu().numpy().astype(np.uint8, copy=False),
        "scorer_input.float32.npy": scorer_input.detach().cpu().numpy(),
        "logits.float32.npy": logits.detach().cpu().numpy(),
        "argmax.uint8.npy": logits.detach().argmax(1).cpu().numpy().astype(np.uint8),
        "target.uint8.npy": target.detach().cpu().numpy().astype(np.uint8),
        "base_field.uint8.npy": base_field.detach().cpu().numpy().astype(np.uint8),
    }
    records = {name: retain_npy(root / name, value) for name, value in payloads.items()}
    retain_json(root / "STEP_RECEIPT.json", {**receipt, "payloads": records})


def checkpoint(
    root: Path,
    model: mt1.MultiTokenRepresentative,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    *,
    stage_index: int,
    stage_step: int,
    global_step: int,
    label: str,
) -> dict[str, Any]:
    config = {
        "schema": "ddm_mt1_training_config.v1",
        "seed": SEED,
        "hidden": HIDDEN,
        "max_support_mass": MAX_SUPPORT_MASS,
        "stages": list(STAGES),
        "ema_decay": EMA_DECAY,
        "ema_warmup": True,
        "selection": "fixed retained stratified split",
    }
    checkpoint_root = root / "checkpoints" / label
    live = {
        "config": config,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    shadow = {
        "config": config,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "global_step": global_step,
        "decay": ema.decay,
        "warmup": ema.warmup,
        "num_updates": ema._num_updates,
        "shadow": ema.state_dict(),
    }
    pointer = {
        "schema": "ddm_mt1_checkpoint_pointer.v1",
        "stage_index": stage_index,
        "stage_step": stage_step,
        "global_step": global_step,
        "live": atomic_torch(checkpoint_root / "live.pt", live),
        "ema": atomic_torch(checkpoint_root / "ema.pt", shadow),
    }
    atomic_json(root / "checkpoints/LATEST.json", pointer)
    return pointer


def resume(
    root: Path,
    model: mt1.MultiTokenRepresentative,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
) -> dict[str, int]:
    path = root / "checkpoints/LATEST.json"
    if not path.is_file():
        return {"stage_index": 0, "stage_step": 0, "global_step": 0}
    pointer = json.loads(path.read_text())
    live = torch.load(pointer["live"]["path"], map_location="cpu", weights_only=False)
    shadow = torch.load(pointer["ema"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(live["model"], strict=True)
    optimizer.load_state_dict(live["optimizer"])
    torch.set_rng_state(live["torch_rng"])
    np.random.set_state(live["numpy_rng"])
    random.setstate(live["python_rng"])
    ema.shadow = {name: value.clone() for name, value in shadow["shadow"].items()}
    ema._num_updates = int(shadow["num_updates"])
    return {
        "stage_index": int(live["stage_index"]),
        "stage_step": int(live["stage_step"]),
        "global_step": int(live["global_step"]),
    }


def package_model(
    root: Path,
    model: mt1.MultiTokenRepresentative,
    ema: EMA,
    stage_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind, state in (("live", model.state_dict()), ("ema", ema.state_dict())):
        snapshot = mt1.MultiTokenRepresentative(HIDDEN, MAX_SUPPORT_MASS)
        snapshot.load_state_dict({name: value.cpu() for name, value in state.items()})
        coded, receipt = mt1.serialize_model(snapshot)
        coded_record = retain_bytes(root / f"stages/{stage_name}/{kind}.mt1.br", coded)
        repeat, _ = mt1.serialize_model(snapshot)
        repeat_record = retain_bytes(
            root / f"stages/{stage_name}/{kind}.repeat.mt1.br", repeat
        )
        if coded != repeat:
            raise MT1ScreenError(f"module serialization repeat differs: {stage_name}/{kind}")
        parsed = mt1.load_module(coded, torch.device("cpu"))
        for name, expected in snapshot.state_dict().items():
            actual = parsed.state_dict()[name]
            if name.endswith("weight"):
                maximum = max(float(expected.abs().amax()) / 127.0, 1.0e-8)
                expected = torch.clamp(torch.round(expected / maximum), -127, 127) * maximum
            else:
                expected = expected.to(torch.float16).float()
            if not torch.equal(actual, expected):
                raise MT1ScreenError(f"module parseback differs: {kind}/{name}")
        result[kind] = {
            "module": coded_record,
            "repeat": repeat_record,
            "serialization": receipt,
            "parseback_exact": True,
        }
    retain_json(root / f"stages/{stage_name}/STAGE_PACKAGE.json", result)
    return result


def train(
    root: Path,
    semantic: nn.Module,
    scorer: nn.Module,
    tokens: np.ndarray,
    target: np.ndarray,
    base_field: np.ndarray,
    train_pairs: list[int],
) -> tuple[Path, dict[str, Any]]:
    model = initialize_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(STAGES[0]["learning_rate"]), weight_decay=0.0
    )
    ema = EMA(model, decay=EMA_DECAY, warmup=True)
    state = resume(root, model, optimizer, ema)
    final_package: dict[str, Any] | None = None
    for stage_index, stage in enumerate(STAGES):
        if stage_index < state["stage_index"]:
            continue
        start = state["stage_step"] if stage_index == state["stage_index"] else 0
        for group in optimizer.param_groups:
            group["lr"] = float(stage["learning_rate"])
        order = np.random.default_rng(SEED + stage_index + 1).permutation(train_pairs)
        for stage_step in range(start, len(order)):
            pair = int(order[stage_step])
            pair_index = torch.tensor([pair], dtype=torch.long)
            token = torch.from_numpy(np.asarray(tokens[pair]).copy())[None].long()
            target_tensor = torch.from_numpy(np.asarray(target[pair]).copy())[None].long()
            base_tensor = torch.from_numpy(np.asarray(base_field[pair]).copy())[None].long()
            optimizer.zero_grad(set_to_none=True)
            probability = model.probability_state(token)
            pre_r, camera, scorer_input = semantic_receiver(
                semantic, token, pair_index, model
            )
            logits = scorer(scorer_input)
            loss, metrics = weighted_loss(logits, target_tensor, base_tensor, stage)
            retain_train_step(
                root / f"training/{stage['name']}/step_{stage_step:03d}_pair_{pair:03d}",
                probability=probability,
                pre_r=pre_r,
                camera=camera,
                scorer_input=scorer_input,
                logits=logits,
                target=target_tensor,
                base_field=base_tensor,
                receipt={
                    "schema": "ddm_mt1_training_step.v1",
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    "global_step_before_update": state["global_step"],
                    "pair": pair,
                    "metrics_before_update": metrics,
                },
            )
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            ema.update(model)
            state["global_step"] += 1
            state["stage_index"] = stage_index
            state["stage_step"] = stage_step + 1
            retain_json(
                root / f"training/receipts/global_{state['global_step']:03d}.json",
                {
                    "schema": "ddm_mt1_optimizer_step.v1",
                    "global_step": state["global_step"],
                    "stage_index": stage_index,
                    "stage_step": stage_step + 1,
                    "pair": pair,
                    "gradient_norm": gradient_norm,
                },
            )
            checkpoint(
                root,
                model,
                optimizer,
                ema,
                stage_index=stage_index,
                stage_step=stage_step + 1,
                global_step=state["global_step"],
                label=f"periodic_g{state['global_step']:03d}",
            )
        final_package = package_model(root, model, ema, str(stage["name"]))
        checkpoint(
            root,
            model,
            optimizer,
            ema,
            stage_index=stage_index + 1,
            stage_step=0,
            global_step=state["global_step"],
            label=f"stage_{stage['name']}_complete",
        )
        state["stage_index"] = stage_index + 1
        state["stage_step"] = 0
    if final_package is None:
        package_path = root / f"stages/{STAGES[-1]['name']}/STAGE_PACKAGE.json"
        if not package_path.is_file():
            raise MT1ScreenError("completed resume lacks final stage package")
        final_package = json.loads(package_path.read_text())
    selected = Path(final_package["ema"]["module"]["path"])
    return selected, final_package


def endpoint_arm(
    root: Path,
    *,
    name: str,
    semantic: nn.Module,
    scorer: nn.Module,
    posenet: nn.Module,
    surface: CP135Surface,
    tokens: np.ndarray,
    target: np.ndarray,
    gt_pose: np.ndarray,
    pairs: list[int],
    model: mt1.MultiTokenRepresentative | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    argmax_parts: list[np.ndarray] = []
    pose_parts: list[np.ndarray] = []
    for batch_index, first in enumerate(range(0, len(pairs), 4)):
        chosen = pairs[first : first + 4]
        batch_root = root / f"endpoint/{name}/batch_{batch_index:02d}"
        token = torch.from_numpy(np.asarray(tokens[chosen]).copy()).long()
        indices = torch.tensor(chosen, dtype=torch.long)
        with torch.inference_mode():
            pre_r, camera, scorer_input = semantic_receiver(
                semantic, token, indices, model
            )
            logits = scorer(scorer_input)
        camera_u8 = camera.cpu().numpy().astype(np.uint8, copy=False)
        masters = camera_u8.transpose(0, 2, 3, 1)
        slaves = np.concatenate(
            [surface.render(surface.codes[pair : pair + 1], pair) for pair in chosen],
            axis=0,
        )
        pose_input = np.stack((slaves, masters), axis=1)
        vectors = pose_vectors(posenet, pose_input)
        argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        payloads = {
            "pairs.int16.npy": np.asarray(chosen, dtype=np.int16),
            "tokens.uint8.npy": np.asarray(tokens[chosen], dtype=np.uint8),
            "pre_r.float32.npy": pre_r.cpu().numpy(),
            "camera.uint8.npy": camera_u8,
            "scorer_input.float32.npy": scorer_input.cpu().numpy(),
            "seg_logits.float32.npy": logits.cpu().numpy(),
            "seg_argmax.uint8.npy": argmax,
            "seg_target.uint8.npy": np.asarray(target[chosen], dtype=np.uint8),
            "pose_slave.uint8.npy": slaves,
            "pose_input.uint8.npy": pose_input,
            "pose_first6.float32.npy": vectors,
            "pose_target_first6.float32.npy": np.asarray(gt_pose[chosen], dtype=np.float32),
        }
        records = {
            payload_name: retain_npy(batch_root / payload_name, value)
            for payload_name, value in payloads.items()
        }
        row = {
            "schema": "ddm_mt1_endpoint_batch.v1",
            "arm": name,
            "pairs": chosen,
            "payloads": records,
        }
        retain_json(batch_root / "BATCH_RECEIPT.json", row)
        rows.append(row)
        argmax_parts.append(argmax)
        pose_parts.append(vectors)
    argmax_all = np.concatenate(argmax_parts)
    pose_all = np.concatenate(pose_parts)
    target_all = np.asarray(target[pairs])
    gt_pose_all = np.asarray(gt_pose[pairs])
    errors = argmax_all != target_all
    result = {
        "schema": "ddm_mt1_endpoint_arm.v1",
        "arm": name,
        "axis": AXIS,
        "n_pairs": len(pairs),
        "selection_mode": "fixed-seed stratified-random heldout",
        "seg_errors": int(np.count_nonzero(errors)),
        "seg_denominator_pixels": int(errors.size),
        "d_seg": float(np.mean(errors)),
        "d_pose": float(np.mean((pose_all - gt_pose_all) ** 2)),
        "retained_argmax": retain_npy(root / f"endpoint/{name}/argmax_n32.npy", argmax_all),
        "retained_pose": retain_npy(root / f"endpoint/{name}/pose_n32.npy", pose_all),
        "batch_receipts": rows,
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(root / f"endpoint/{name}/RESULT.json", result)
    return result


def run(root: Path) -> dict[str, Any]:
    storage = storage_preflight(root)
    reproducibility = configure_reproducibility()
    sources = source_records()
    introduced_counts = np.load(BG2_ROOT / "per_frame_counts.int64.npy", allow_pickle=False)
    spatial_introduced = np.load(BG2_ROOT / "spatial_introduced.int64.npy", allow_pickle=False)
    selection = stratified_split(introduced_counts, spatial_introduced)
    retain_json(root / "inputs/SELECTION.json", selection)
    base_tokens = np.load(BASE_TOKENS, mmap_mode="r", allow_pickle=False)
    c1_tokens = load_raw_tokens(C1_TOKENS)
    target = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    base_field = np.load(BASE_FIELD, mmap_mode="r", allow_pickle=False)
    gt_pose = np.load(GT_POSE, mmap_mode="r", allow_pickle=False)
    if (
        base_tokens.shape != (PAIR_COUNT, *WORK_HW)
        or c1_tokens.shape != base_tokens.shape
        or target.shape != base_tokens.shape
        or base_field.shape != base_tokens.shape
        or gt_pose.shape != (PAIR_COUNT, 6)
    ):
        raise MT1ScreenError("source array geometry differs")
    selected_pairs = selection["train"] + selection["heldout"]
    selected_inputs = {
        "base_tokens_n64.npy": np.asarray(base_tokens[selected_pairs], dtype=np.uint8),
        "c1_tokens_n64.npy": np.asarray(c1_tokens[selected_pairs], dtype=np.uint8),
        "gt_argmax_n64.npy": np.asarray(target[selected_pairs], dtype=np.uint8),
        "base_argmax_n64.npy": np.asarray(base_field[selected_pairs], dtype=np.uint8),
        "gt_pose_n64.npy": np.asarray(gt_pose[selected_pairs], dtype=np.float32),
    }
    selected_records = {
        name: retain_npy(root / f"inputs/{name}", value)
        for name, value in selected_inputs.items()
    }
    retain_json(
        root / "inputs/INPUT_MANIFEST.json",
        {
            "schema": "ddm_mt1_input_manifest.v1",
            "sources": sources,
            "selected_payloads": selected_records,
            "selection": file_record(root / "inputs/SELECTION.json"),
        },
    )
    semantic = load_exact_semantic(CP135_RUNTIME, CP135_ARCHIVE, torch.device("cpu"))
    scorer = load_local_segnet()
    selected_model, packages = train(
        root,
        semantic,
        scorer,
        base_tokens,
        target,
        base_field,
        selection["train"],
    )
    parsed = mt1.load_module(selected_model.read_bytes(), torch.device("cpu"))
    surface, surface_pins = CP135Surface.load()
    posenet = load_posenet()
    arms = {
        "cp135_hard": endpoint_arm(
            root,
            name="cp135_hard",
            semantic=semantic,
            scorer=scorer,
            posenet=posenet,
            surface=surface,
            tokens=base_tokens,
            target=target,
            gt_pose=gt_pose,
            pairs=selection["heldout"],
            model=None,
        ),
        "hc1_direct_c1": endpoint_arm(
            root,
            name="hc1_direct_c1",
            semantic=semantic,
            scorer=scorer,
            posenet=posenet,
            surface=surface,
            tokens=c1_tokens,
            target=target,
            gt_pose=gt_pose,
            pairs=selection["heldout"],
            model=None,
        ),
        "mt1_multitoken": endpoint_arm(
            root,
            name="mt1_multitoken",
            semantic=semantic,
            scorer=scorer,
            posenet=posenet,
            surface=surface,
            tokens=base_tokens,
            target=target,
            gt_pose=gt_pose,
            pairs=selection["heldout"],
            model=parsed,
        ),
    }
    base = arms["cp135_hard"]
    direct = arms["hc1_direct_c1"]
    candidate = arms["mt1_multitoken"]
    pose_delta = float(candidate["d_pose"] - base["d_pose"])
    positive = bool(
        candidate["seg_errors"] < direct["seg_errors"]
        and candidate["seg_errors"] < base["seg_errors"]
        and pose_delta <= 0.0
    )
    comparison = {
        "schema": "ddm_mt1_sign_comparison.v1",
        "axis": AXIS,
        "n_train": 32,
        "n_heldout": 32,
        "heldout_denominator_pixels": 32 * WORK_HW[0] * WORK_HW[1],
        "arms": arms,
        "candidate_minus_base_pose_mse": pose_delta,
        "gates": {
            "stronger_than_direct_c1_seg": candidate["seg_errors"] < direct["seg_errors"],
            "improves_cp135_seg": candidate["seg_errors"] < base["seg_errors"],
            "zero_pose_damage": pose_delta <= 0.0,
            "parsed_counted_model_consumed": True,
            "no_changed_site_list": True,
            "same_exact_cp135_frame0_carrier": True,
        },
        "positive_local_sign": positive,
        "verdict_scope": (
            "single fixed-seed stratified-random n32 heldout screen of the hidden-4, "
            "max-support-mass-0.25 local simplex formulation"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    retain_json(root / "COMPARISON.json", comparison)
    train_order: dict[str, Any] | None = None
    if positive:
        train_order = {
            "schema": "ddm_mt1_second_train_fire_order.v1",
            "disposition": "SEALED_NOT_FIRED",
            "owner": "MAIN",
            "consumer_store": "main_hot_state.md and lane registry",
            "fire_trigger": "T4 sign gate reproduces all three positive local gates",
            "input_model": file_record(selected_model),
            "requested_train": "stratified-random n120 train plus disjoint n120 heldout",
            "budget_usd": 0.75,
            "score_claim": False,
        }
        retain_json(root / "SECOND_TRAIN_FIRE_ORDER.json", train_order)
    result = {
        "schema": "ddm_mt1_screen_result.v1",
        "run_id": RUN_ID,
        "axis": AXIS,
        "storage_preflight": storage,
        "reproducibility": reproducibility,
        "selection": selection,
        "selected_model": file_record(selected_model),
        "stage_packages": packages,
        "surface_pins": surface_pins,
        "comparison": comparison,
        "second_train_fire_order": train_order,
        "pointer_moved": False,
        "exact_score_measured": False,
        "score_claim": False,
    }
    retain_json(root / "FINAL_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        receipt = {
            "storage": storage_preflight(args.output),
            "sources": source_records(),
        }
        retain_json(args.output / "PREFLIGHT_ONLY.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    result = run(args.output)
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
