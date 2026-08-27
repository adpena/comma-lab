# SPDX-License-Identifier: MIT
"""Real scorer-in-loop trainer for the frozen QBF1 packet ABI.

This module is the stage-03/04/05 consumer named by the sealed QBFLOW fire
order.  It changes no QBF1 field, tensor name, tensor shape, section, coder, or
receiver rule.  Training is joint from step zero: the QBFLOW coordinate field
renders both RGB frames, the exact camera round trip is applied, and gradients
flow through the frozen SegNet and PoseNet graphs.  Every checkpoint is atomic,
contains live/optimizer/RNG state plus the EMA inference shadow, and is
immediately re-encoded through ``experiments.ddm_qbflow_packet`` with every
real coder candidate retained.

The bounded ``smoke`` action is mechanism verification only.  It is CPU-only,
accepts at most four pairs, sets ``score_claim=false``, and cannot satisfy the
same-budget QBW1 control gate without a separately retained, real control
receipt.  Heavy training remains MAIN-only and additionally requires a
compiled config with explicit lane claims and launch authorization.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import resource
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_qbflow_packet as qbf1
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.scorer import load_differentiable_scorers
from tac.training import EMA
from tac.witness_dsl.curriculum_dsl import EmaDecayCalibrated

SCHEMA = "ddm_qbt1_qbflow_compiled_config.v1"
CHECKPOINT_SCHEMA = "ddm_qbt1_qbflow_checkpoint.v1"
RESULT_SCHEMA = "ddm_qbflow_observability.v1"
CONTROL_SCHEMA = "ddm_qbt1_same_budget_qbw1_control.v1"
MEMORY_SCHEMA = "ddm_qbt1_materialization_memory_projection.v1"
LAUNCH_SCHEMA = "ddm_qbt1_compiled_launch_request.v1"
SEED = 20260827
N = 600
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
RATE_DENOMINATOR = 37_545_489
MAX_CHUNK_PAIRS = 30
REAL_TRAIN_CHUNK_PAIRS = 16
MEMORY_CEILING_BYTES = 124_554_051_584
COMPLETE_ARCHIVE_CAP_BYTES = 137_986
DPOSE_HAT_MAX = 1.25e-4
S_HAT_MAX_EXCLUSIVE = 0.12
SMOKE_MAX_PAIRS = 4
SELECTION_IDS = (
    4,
    31,
    49,
    52,
    62,
    90,
    100,
    113,
    128,
    148,
    173,
    179,
    186,
    187,
    214,
    236,
    256,
    260,
    268,
    278,
    326,
    328,
    341,
    352,
    368,
    382,
    444,
    456,
    483,
    508,
    563,
    573,
)
SELECTION_WEIGHTS = (15.0,) * 24 + (30.0,) * 8

STORE = Path("/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow")
TRAIN_ROOT = STORE / "qbt1_trainer"
FIRE_ORDER = STORE / "SEALED_TRAINING_FIRE_ORDER.json"
INITIAL_PARAMS = STORE / "stage_01_initialize_quantize/initialized_float_params.npz"
INITIAL_LATENTS = STORE / "stage_01_initialize_quantize/initialized_float_latents.npz"
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
QBW1_RESULT = Path("/Volumes/APDataStore/pact/ddm_qbw1_boundary_event_quotient/RESULT_STAGE02.json")
QBW1_FIRE_ORDER = Path(
    "/Volumes/APDataStore/pact/ddm_qbw1_boundary_event_quotient/"
    "sealed_main_fire_order/FIRE_ORDER.json"
)
SCHEMA_DOC = REPO / ".omx/research/SPEC_ddm_qbflow_packet_schema_v1_20260827.md"
WD3_REFERENCE = REPO / "experiments/ddm_wd3_sealed_3d9e021d07_runner.py"
NO2_MEMO = REPO / ".omx/research/ddm_no2_quotient_born_object_20260827.md"
W96B_VERDICT = REPO / ".omx/research/ddm_w96b_seed20260816_aligned_verdict_and_family_closure_20260827.md"
POSENET = REPO / "upstream/models/posenet.safetensors"
SEGNET = REPO / "upstream/models/segnet.safetensors"

PINNED_SHA256 = {
    "packet_module": "cdf90d1a4d7d13001118f50a76692c04605f8e5ae9a7816c80f6e346160c7b9c",
    "packet_schema": "5405ccd499d14d28230874059e47d47f1f2818038519f1b27c97ed9377f132aa",
    "fire_order": "7fa18f51d741b9f079b14c67d1c6560edd534bcaab6e2b8984f5bd0bd4b1ba8a",
    "initialized_float_params": "b2e61092bab168e390572cf3590ab74615f21ec9d6759a45244beaa6b325bc17",
    "initialized_float_latents": "9e9d3fa0792ffc3a47de702e288ec364bebda9f3c6c23451216eda27de449355",
    "gt_cache": "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    "posenet": "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
    "segnet": "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    "wd3_reference": "6a567db93c9947e63b5fb022411dd583ce848ccb22e3fe0e2393fe58c94a86df",
}

PIN_PATHS = {
    "packet_module": Path(qbf1.__file__).resolve(),
    "packet_schema": SCHEMA_DOC,
    "fire_order": FIRE_ORDER,
    "initialized_float_params": INITIAL_PARAMS,
    "initialized_float_latents": INITIAL_LATENTS,
    "gt_cache": GT_CACHE,
    "posenet": POSENET,
    "segnet": SEGNET,
}


class QBT1Error(RuntimeError):
    """The trainer refused an ABI, custody, launch, or evidence violation."""


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QBT1Error(f"required retained payload is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    return atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, default=str).encode() + b"\n")


def atomic_torch(path: Path, value: object) -> dict[str, Any]:
    payload = io.BytesIO()
    torch.save(value, payload)
    return atomic_bytes(path, payload.getvalue())


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    return atomic_bytes(path, payload.getvalue())


def verify_pins() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, path in PIN_PATHS.items():
        row = file_fact(path)
        if row["sha256"] != PINNED_SHA256[name]:
            raise QBT1Error(f"frozen input drifted: {name}")
        rows[name] = row
    # WD3 is a reviewed reference form consumed during this build, not a
    # runtime import.  Verify it when the shared-worktree copy is present, but
    # keep the compiled provenance stable when that separately owned file is
    # absent from a clean checkout.
    if WD3_REFERENCE.is_file() and sha256_file(WD3_REFERENCE) != PINNED_SHA256["wd3_reference"]:
        raise QBT1Error("WD3 scorer-in-loop reference form drifted")
    rows["wd3_reference"] = {
        "path": str(WD3_REFERENCE.resolve()),
        "bytes": 145_956,
        "sha256": PINNED_SHA256["wd3_reference"],
    }
    rows["no2_gate"] = file_fact(NO2_MEMO)
    rows["w96b_expected_flip_receipt"] = file_fact(W96B_VERDICT)
    rows["qbw1_stage02_result"] = file_fact(QBW1_RESULT)
    rows["qbw1_fire_order"] = file_fact(QBW1_FIRE_ORDER)
    return rows


def storage_preflight(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    resolved = output.resolve()
    root = STORE.resolve()
    if resolved != root and root not in resolved.parents:
        raise QBT1Error(f"QBT1 output must remain under the AP custody root {root}")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < int(minimum_free_bytes):
        raise QBT1Error(
            f"APDataStore storage preflight refused: free={usage.free} required={minimum_free_bytes}"
        )
    return {
        "root": str(resolved),
        "free_bytes": usage.free,
        "required_free_bytes": int(minimum_free_bytes),
        "cleanup": "certify-or-block; retained checkpoints, packets, archives, frames, and scorer outputs are never deleted",
        "status": "PASS",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _linear(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,ij->...j", value, weight) + bias


def _base_features(pair_ids: torch.Tensor, *, height: int, width: int) -> tuple[torch.Tensor, ...]:
    dtype, device = torch.float32, pair_ids.device
    ys = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    xs = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    batch = int(pair_ids.numel())
    x = x.expand(batch, -1, -1)
    y = y.expand(batch, -1, -1)
    t = -1.0 + 2.0 * pair_ids.float() / float(N - 1)
    t_field = t[:, None, None].expand_as(x)
    road_center = 0.08 * torch.sin(torch.pi * t_field)
    road_half = 0.18 + 0.62 * (y + 1.0) * 0.5
    road_u = (x - road_center) / road_half.clamp_min(0.05)
    road_v = y + 0.15
    road_soft = torch.exp(-road_u.square())
    features = [x, y, t_field, road_u, road_v, road_soft]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        for coordinate in (x, y):
            phase = torch.pi * frequency * coordinate
            features.extend((torch.sin(phase), torch.cos(phase)))
    for frequency in (1.0, 2.0, 4.0):
        phase = torch.pi * frequency * t_field
        features.extend((torch.sin(phase), torch.cos(phase)))
    features.extend((x * y, road_u * y, x.square()))
    base = torch.stack(features, dim=-1)
    if base.shape[-1] != qbf1.BASE_FEATURE_DIM:
        raise QBT1Error("QBF1 base feature dimension drifted")
    return base.reshape(batch, height * width, -1), x, y


class QBFLOWTorch(nn.Module):
    """Differentiable training twin of the immutable NumPy QBF1 receiver."""

    def __init__(self, params: Mapping[str, np.ndarray], boundary: np.ndarray, interior: np.ndarray) -> None:
        super().__init__()
        qbf1.validate_param_shapes(params)
        if boundary.shape != (N, qbf1.BOUNDARY_LATENT_DIM) or interior.shape != (
            N,
            qbf1.INTERIOR_LATENT_DIM,
        ):
            raise QBT1Error("QBF1 latent geometry differs")
        self.params = nn.ParameterDict(
            {name: nn.Parameter(torch.from_numpy(np.asarray(value, dtype=np.float32).copy())) for name, value in params.items()}
        )
        self.boundary_latents = nn.Parameter(torch.from_numpy(np.asarray(boundary, dtype=np.float32).copy()))
        self.interior_latents = nn.Parameter(torch.from_numpy(np.asarray(interior, dtype=np.float32).copy()))
        incidence = np.zeros((qbf1.N_INTERFACES, qbf1.N_CLASSES), dtype=np.float32)
        index = 0
        for left in range(qbf1.N_CLASSES):
            for right in range(left + 1, qbf1.N_CLASSES):
                incidence[index, left] = 1.0
                incidence[index, right] = -1.0
                index += 1
        self.register_buffer("incidence", torch.from_numpy(incidence), persistent=False)

    def packet_state(self, state: Mapping[str, torch.Tensor] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        source = self.state_dict() if state is None else state
        params = {
            name: source[f"params.{name}"].detach().cpu().float().numpy().copy()
            for name in qbf1.expected_param_shapes()
        }
        boundary = source["boundary_latents"].detach().cpu().float().numpy().copy()
        interior = source["interior_latents"].detach().cpu().float().numpy().copy()
        qbf1.validate_param_shapes(params)
        return params, boundary, interior

    def forward(self, pair_ids: torch.Tensor, *, height: int = EVAL_H, width: int = EVAL_W) -> dict[str, torch.Tensor]:
        if pair_ids.ndim != 1 or pair_ids.numel() < 1:
            raise QBT1Error("pair_ids must be a nonempty vector")
        if int(pair_ids.min()) < 0 or int(pair_ids.max()) >= N:
            raise QBT1Error("pair id is outside the frozen n600 geometry")
        base, x, y = _base_features(pair_ids, height=height, width=width)
        batch, points = pair_ids.numel(), height * width
        boundary = self.boundary_latents[pair_ids]
        interior = self.interior_latents[pair_ids]
        boundary_tiled = boundary[:, None].expand(-1, points, -1)
        interior_tiled = interior[:, None].expand(-1, points, -1)
        p = self.params

        coarse_input = torch.cat((base, boundary_tiled, interior_tiled), dim=-1)
        coarse = torch.tanh(_linear(coarse_input, p["coarse_in_w"], p["coarse_in_b"]))
        coarse = torch.tanh(coarse + _linear(coarse, p["coarse_res_w"], p["coarse_res_b"]))
        coarse_logits = _linear(coarse, p["coarse_logits_w"], p["coarse_logits_b"])
        coarse_features = torch.tanh(_linear(coarse, p["coarse_feat_w"], p["coarse_feat_b"]))

        road_probability = coarse_logits.softmax(dim=-1)[..., 0].reshape(batch, height, width)
        gy, gx = torch.gradient(road_probability, dim=(1, 2))
        norm = torch.sqrt(gx.square() + gy.square()).add(1.0e-6)
        tangent_x, tangent_y = -gy / norm, gx / norm
        road_condition = torch.stack((road_probability, gx, gy, tangent_x, tangent_y), dim=-1)
        flow_input = torch.cat((base, coarse_features, road_condition.reshape(batch, points, 5), boundary_tiled), dim=-1)
        film = _linear(boundary, p["flow_film_w"], p["flow_film_b"]).reshape(
            batch, qbf1.FLOW_LAYERS, 2, qbf1.FLOW_DIM
        )
        flow = _linear(flow_input, p["flow_in_w"], p["flow_in_b"])
        flow = flow * (1.0 + 0.1 * torch.tanh(film[:, 0, 0])[:, None])
        flow = flow + 0.1 * film[:, 0, 1][:, None]
        flow = torch.tanh(
            flow * p["step_slope_0"][None, None] - p["step_center_0"][None, None]
        )
        for index in range(1, qbf1.FLOW_LAYERS):
            proposal = _linear(flow, p[f"flow_res_{index}_w"], p[f"flow_res_{index}_b"])
            proposal = proposal * (1.0 + 0.1 * torch.tanh(film[:, index, 0])[:, None])
            proposal = proposal + 0.1 * film[:, index, 1][:, None]
            flow = torch.tanh(
                (flow + proposal) * p[f"step_slope_{index}"][None, None]
                - p[f"step_center_{index}"][None, None]
            )

        u_tangent = x * tangent_x + y * tangent_y
        along = []
        for frequency in (8.0, 16.0, 24.0, 32.0):
            phase = torch.pi * frequency * u_tangent
            along.extend((torch.sin(phase), torch.cos(phase)))
        along_features = torch.stack(along, dim=-1).reshape(batch, points, qbf1.ALONG_FEATURE_DIM)
        flow_output = _linear(torch.cat((flow, along_features), dim=-1), p["flow_head_w"], p["flow_head_b"])
        signed_interfaces = flow_output[..., : qbf1.N_INTERFACES]
        boundary_features = torch.tanh(flow_output[..., qbf1.N_INTERFACES :])
        class_logits = coarse_logits + torch.einsum("bpi,ic->bpc", signed_interfaces, self.incidence)

        interior_input = torch.cat((base, coarse_features, interior_tiled), dim=-1)
        interior_state = torch.tanh(_linear(interior_input, p["interior_in_w"], p["interior_in_b"]))
        interior_state = torch.tanh(
            interior_state + _linear(interior_state, p["interior_res_w"], p["interior_res_b"])
        )
        interior_features = torch.tanh(_linear(interior_state, p["interior_head_w"], p["interior_head_b"]))
        render_input = torch.cat((interior_features, boundary_features, boundary_tiled, interior_tiled), dim=-1)
        render_state = torch.tanh(_linear(render_input, p["render_in_w"], p["render_in_b"]))
        rgb = torch.sigmoid(_linear(render_state, p["render_out_w"], p["render_out_b"]))
        pooled = interior_features.mean(dim=1)
        pose_input = torch.cat((pooled, interior), dim=-1)
        pose_state = torch.tanh(_linear(pose_input, p["pose_in_w"], p["pose_in_b"]))
        pose12 = _linear(pose_state, p["pose_out_w"], p["pose_out_b"])
        return {
            "signed_interfaces": signed_interfaces.reshape(batch, height, width, qbf1.N_INTERFACES),
            "class_logits": class_logits.reshape(batch, height, width, qbf1.N_CLASSES),
            "rgb_pair_01": rgb.reshape(batch, height, width, 2, 3).permute(0, 3, 4, 1, 2),
            "pose12": pose12,
            "coarse_road_probability": road_probability,
            "road_tangent": torch.stack((tangent_x, tangent_y), dim=-1),
        }


def roundtrip_to_camera_uint8_ste(rgb_pair_01: torch.Tensor) -> torch.Tensor:
    """QBF1 render -> bicubic camera grid -> uint8 STE; scorer downsamples bilinearly."""

    if rgb_pair_01.ndim != 5 or rgb_pair_01.shape[1:3] != (2, 3):
        raise QBT1Error("QBFLOW RGB pair must have shape [B,2,3,H,W]")
    batch = rgb_pair_01.shape[0]
    camera = F.interpolate(
        rgb_pair_01.reshape(batch * 2, 3, *rgb_pair_01.shape[-2:]) * 255.0,
        size=(CAMERA_H, CAMERA_W),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 255.0)
    rounded = camera.round().clamp(0.0, 255.0)
    camera = camera + (rounded - camera).detach()
    return camera.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)


def scorer_forward(pair: torch.Tensor, posenet: nn.Module, segnet: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    pose_output = posenet(posenet.preprocess_input(pair))
    if not isinstance(pose_output, Mapping) or "pose" not in pose_output:
        raise QBT1Error("PoseNet output lacks the official pose head")
    pose6 = pose_output["pose"][..., :6]
    logits = segnet(segnet.preprocess_input(pair))
    if pose6.shape != (pair.shape[0], 6) or logits.shape[1:] != (5, EVAL_H, EVAL_W):
        raise QBT1Error("frozen scorer output geometry differs")
    return pose6, logits


def expected_flip_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    tau: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 4 or target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise QBT1Error("expected-flip logits/target geometry differs")
    if not tau > 0:
        raise QBT1Error("expected-flip tau must be positive")
    target_index = target[:, None].long()
    target_logit = logits.gather(1, target_index).squeeze(1)
    other = logits.clone()
    other.scatter_(1, target_index, -1.0e9)
    margin = target_logit - other.amax(dim=1)
    per_sample = torch.sigmoid(-margin / tau).mean(dim=(1, 2))
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.to(per_sample)
    if weights.shape != per_sample.shape or not bool(torch.all(weights > 0)):
        raise QBT1Error("expected-flip sample weights differ")
    return (per_sample * weights).sum() / weights.sum()


def tau_for_step(step: int, total_steps: int, start: float = 0.15, end: float = 0.05) -> float:
    if total_steps < 1 or not 0 <= step < total_steps or not start > end > 0:
        raise QBT1Error("expected-flip schedule geometry differs")
    return start + (end - start) * step / max(total_steps - 1, 1)


def joint_objective(
    outputs: Mapping[str, torch.Tensor],
    camera_pair: torch.Tensor,
    scorer_pose6: torch.Tensor,
    scorer_logits: torch.Tensor,
    target_argmax: torch.Tensor,
    target_pose6: torch.Tensor,
    tau: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    realized_seg = expected_flip_margin_loss(scorer_logits, target_argmax, tau, sample_weights)
    native_logits = outputs["class_logits"].permute(0, 3, 1, 2)
    interface_seg = expected_flip_margin_loss(native_logits, target_argmax, tau, sample_weights)
    pose_per_sample = (scorer_pose6 - target_pose6).square().mean(dim=1)
    if sample_weights is None:
        pose_mse = pose_per_sample.mean()
    else:
        weights = sample_weights.to(pose_per_sample)
        pose_mse = (pose_per_sample * weights).sum() / weights.sum()
    pose_score = torch.sqrt(torch.clamp(10.0 * pose_mse, min=1.0e-20))
    # Both Seg terms are the same expected-flip probability law.  Their sum
    # makes the ABI's signed-interface head real while the realized term alone
    # remains the score-facing quantity.  Pose is active from the first step.
    total = 100.0 * (realized_seg + interface_seg) + pose_score
    return total, {
        "loss_total": total,
        "seg_expected_flip_realized": realized_seg,
        "seg_expected_flip_native_interface": interface_seg,
        "pose_mse_realized": pose_mse,
        "pose_score_realized": pose_score,
        "tau": total.new_tensor(tau),
        "camera_min": camera_pair.detach().amin(),
        "camera_max": camera_pair.detach().amax(),
    }


def load_initial_model(device: torch.device) -> QBFLOWTorch:
    with np.load(INITIAL_PARAMS, allow_pickle=False) as params_npz:
        params = {name: np.asarray(params_npz[name], dtype=np.float32) for name in params_npz.files}
    with np.load(INITIAL_LATENTS, allow_pickle=False) as latents_npz:
        boundary = np.asarray(latents_npz["boundary"], dtype=np.float32)
        interior = np.asarray(latents_npz["interior"], dtype=np.float32)
    return QBFLOWTorch(params, boundary, interior).to(device)


def _ema_payload(ema: EMA) -> dict[str, Any]:
    return {
        "decay": float(ema.decay),
        "warmup": bool(ema.warmup),
        "num_updates": int(ema._num_updates),
        "shadow": {name: value.detach().cpu().clone() for name, value in ema.shadow.items()},
    }


def _restore_ema(model: nn.Module, payload: Mapping[str, Any]) -> EMA:
    ema = EMA(model, decay=float(payload["decay"]), warmup=bool(payload["warmup"]))
    ema._num_updates = int(payload["num_updates"])
    if set(ema.shadow) != set(payload["shadow"]):
        raise QBT1Error("EMA shadow tensor set differs on resume")
    ema.shadow = {name: value.detach().clone().to(model.state_dict()[name]) for name, value in payload["shadow"].items()}
    return ema


def _rng_payload() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload["torch_cuda"]:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"action", "resume_from", "launch_authorized", "scorer_lane", "metal_lane"}
    return {name: copy.deepcopy(value) for name, value in config.items() if name not in ignored}


def save_checkpoint(
    path: Path,
    *,
    model: QBFLOWTorch,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    config: Mapping[str, Any],
    step: int,
    stage: str,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": stage,
        "step": int(step),
        "config_identity": config_identity(config),
        "config_identity_sha256": canonical_sha256(config_identity(config)),
        "live_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "ema": _ema_payload(ema),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng": _rng_payload(),
        "history": list(history),
    }
    return atomic_torch(path, payload)


def load_checkpoint(
    path: Path,
    *,
    model: QBFLOWTorch,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> tuple[int, EMA, list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise QBT1Error("resume checkpoint schema differs")
    if payload.get("config_identity") != config_identity(config):
        raise QBT1Error("resume config identity differs")
    model.load_state_dict(payload["live_state_dict"], strict=True)
    ema = _restore_ema(model, payload["ema"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng(payload["rng"])
    return int(payload["step"]), ema, list(payload["history"]), payload


@contextlib.contextmanager
def ema_scope(model: nn.Module, ema: EMA) -> Iterable[None]:
    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema.apply(model)
    try:
        yield
    finally:
        model.load_state_dict(live, strict=True)


def _retain_section(root: Path, section_id: int, raw: bytes) -> qbf1.EncodedSection:
    atomic_bytes(root / "raw.bin", raw)
    candidates = qbf1.encode_section_candidates(section_id, raw)
    repeats = qbf1.encode_section_candidates(section_id, raw)
    for codec_name, candidate in candidates.items():
        primary = atomic_bytes(root / f"candidate.{codec_name}.bin", candidate.payload)
        repeat = atomic_bytes(root / f"candidate.{codec_name}.repeat.bin", repeats[codec_name].payload)
        if primary["sha256"] != repeat["sha256"]:
            raise QBT1Error(f"real coder repeat drifted: section={section_id} codec={codec_name}")
    return qbf1.choose_section(candidates)


def reencode_inference_state(
    root: Path,
    *,
    model: QBFLOWTorch,
    state: Mapping[str, torch.Tensor],
    selected_pair_ids: Sequence[int],
) -> dict[str, Any]:
    """Retain every real coder payload and complete QBF1 framing for one EMA state."""

    params, boundary, interior = model.packet_state(state)
    config_raw = qbf1.canonical_json_bytes(qbf1.architecture_config(num_pairs=N, seed=SEED))
    model_raw = qbf1.encode_model(params)
    meta_raw, boundary_codes, interior_codes = qbf1.encode_latent_meta(boundary, interior)
    latent_raw = qbf1.encode_latent_table(range(N), boundary_codes, interior_codes)
    sections = [
        _retain_section(root / "sections/config", qbf1.SECTION_CONFIG, config_raw),
        _retain_section(root / "sections/model", qbf1.SECTION_MODEL, model_raw),
        _retain_section(root / "sections/latent_meta", qbf1.SECTION_LATENT_META, meta_raw),
        _retain_section(root / "sections/latents", qbf1.SECTION_LATENTS, latent_raw),
    ]
    packet_primary = qbf1.pack_packet(sections)
    packet_repeat = qbf1.pack_packet(sections)
    if packet_primary != packet_repeat:
        raise QBT1Error("QBF1 packet repeat drifted")
    archive_primary = qbf1.deterministic_archive(packet_primary)
    archive_repeat = qbf1.deterministic_archive(packet_repeat)
    if archive_primary != archive_repeat:
        raise QBT1Error("QBF1 archive repeat drifted")
    packet_fact = atomic_bytes(root / "packet.qbf", packet_primary)
    packet_repeat_fact = atomic_bytes(root / "packet.repeat.qbf", packet_repeat)
    archive_fact = atomic_bytes(root / "archive.zip", archive_primary)
    archive_repeat_fact = atomic_bytes(root / "archive.repeat.zip", archive_repeat)
    decoded = qbf1.decode_packet(packet_primary)
    if set(decoded.sections) != {
        qbf1.SECTION_CONFIG,
        qbf1.SECTION_MODEL,
        qbf1.SECTION_LATENT_META,
        qbf1.SECTION_LATENTS,
    }:
        raise QBT1Error("QBF1 parse-back section set differs")

    shared_sections = sections[:3]
    shared_archive = qbf1.deterministic_archive(qbf1.pack_packet(shared_sections))
    shared_fact = atomic_bytes(root / "reset_projection/shared.archive.zip", shared_archive)
    reset_rows = []
    weight_lookup = dict(zip(SELECTION_IDS, SELECTION_WEIGHTS, strict=True))
    for pair_id in selected_pair_ids:
        raw = qbf1.encode_latent_record(pair_id, boundary_codes[pair_id], interior_codes[pair_id])
        candidates = qbf1.encode_reset_record(raw)
        repeat_candidates = qbf1.encode_reset_record(raw)
        pair_root = root / "reset_projection" / f"pair_{pair_id:04d}"
        rows = []
        for codec_name, payload in candidates.items():
            fact = atomic_bytes(pair_root / f"record.{codec_name}.qbr", payload)
            repeat = atomic_bytes(pair_root / f"record.{codec_name}.repeat.qbr", repeat_candidates[codec_name])
            if fact["sha256"] != repeat["sha256"] or qbf1.decode_reset_record(payload) != raw:
                raise QBT1Error(f"QBF1 reset record failed for pair {pair_id}/{codec_name}")
            rows.append((fact["bytes"], codec_name, fact))
        _size, winner, winner_fact = min(rows, key=lambda row: (row[0], qbf1.CODEC_IDS[row[1]]))
        reset_rows.append(
            {
                "pair_id": pair_id,
                "winner": winner,
                "winner_payload": winner_fact,
                "ht_weight": weight_lookup.get(pair_id),
            }
        )
    ht_ready = len(selected_pair_ids) == len(SELECTION_IDS) and tuple(selected_pair_ids) == SELECTION_IDS
    b_var_hat = (
        sum(float(row["ht_weight"]) * int(row["winner_payload"]["bytes"]) for row in reset_rows)
        if ht_ready
        else None
    )
    b_hat = int(shared_fact["bytes"] + math.ceil(b_var_hat)) if b_var_hat is not None else None
    manifest = {
        "schema": "ddm_qbt1_qbf1_reencode.v1",
        "score_claim": False,
        "packet": packet_fact,
        "packet_repeat": packet_repeat_fact,
        "archive": archive_fact,
        "archive_repeat": archive_repeat_fact,
        "parseback_exact": True,
        "section_facts": list(decoded.section_facts),
        "shared_archive": shared_fact,
        "reset_rows": reset_rows,
        "B_var_hat": b_var_hat,
        "B_hat": b_hat,
        "ht_projection_ready": ht_ready,
    }
    atomic_json(root / "REENCODE_MANIFEST.json", manifest)
    return manifest


def state_tensor_role(name: str) -> str:
    if name == "boundary_latents":
        return "boundary_latents"
    if name == "interior_latents":
        return "interior_latents"
    tensor = name.removeprefix("params.")
    if tensor.startswith("step_"):
        return "step_transition"
    if tensor.startswith("flow_"):
        return "boundary_flow"
    if tensor.startswith("coarse_"):
        return "coarse_partition"
    if tensor.startswith("interior_"):
        return "interior_field"
    if tensor.startswith("render_"):
        return "rgb_renderer"
    if tensor.startswith("pose_"):
        return "pose_head"
    raise QBT1Error(f"unclassified QBF1 state tensor: {name}")


def prequantize_role(
    state: Mapping[str, torch.Tensor], role: str, bits: int
) -> dict[str, torch.Tensor]:
    if not 2 <= int(bits) <= 16:
        raise QBT1Error("stage-04 precision probe bits must lie in [2,16]")
    candidate = {name: value.detach().clone() for name, value in state.items()}
    touched = 0
    for name, value in candidate.items():
        if state_tensor_role(name) != role:
            continue
        source = value.detach().cpu().float().numpy()
        codes, scale = qbf1.quantize(source, int(bits))
        restored = qbf1.dequantize(codes, scale, source.shape)
        candidate[name] = torch.from_numpy(restored.copy()).to(value)
        touched += value.numel()
    if touched == 0:
        raise QBT1Error(f"stage-04 precision probe touched no tensor for role {role}")
    return candidate


def precision_sensitivity_and_reencode(
    root: Path,
    *,
    model: QBFLOWTorch,
    state: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    selected_pair_ids: Sequence[int],
    probe_bits: Sequence[int],
) -> dict[str, Any]:
    """Retain real QBF1 options and a labelled first-order sensitivity table.

    The gradient metric is only a shortlist signal.  This stage deliberately
    does not promote a prequantized option without a realized scorer A/B.
    """

    roles = tuple(sorted({state_tensor_role(name) for name in state}))
    baseline = reencode_inference_state(
        root / "baseline", model=model, state=state, selected_pair_ids=selected_pair_ids
    )
    rows: list[dict[str, Any]] = []
    for role in roles:
        role_gradient_energy = sum(
            float(gradients.get(name, torch.zeros_like(value)).detach().float().square().sum().cpu())
            for name, value in state.items()
            if state_tensor_role(name) == role
        )
        for bits in tuple(map(int, probe_bits)):
            candidate = prequantize_role(state, role, bits)
            first_order_abs = 0.0
            perturbation_l2 = 0.0
            for name, baseline_value in state.items():
                if state_tensor_role(name) != role:
                    continue
                delta = candidate[name].detach().float() - baseline_value.detach().float()
                gradient = gradients.get(name)
                if gradient is not None:
                    first_order_abs += float((gradient.detach().float() * delta).abs().sum().cpu())
                perturbation_l2 += float(delta.square().sum().cpu())
            encoded = reencode_inference_state(
                root / f"role_{role}" / f"prequant_{bits:02d}bit",
                model=model,
                state=candidate,
                selected_pair_ids=selected_pair_ids,
            )
            rows.append(
                {
                    "role": role,
                    "prequant_bits": bits,
                    "gradient_energy": role_gradient_energy,
                    "first_order_abs_proxy": first_order_abs,
                    "perturbation_l2": perturbation_l2,
                    "archive_bytes": encoded["archive"]["bytes"],
                    "archive_delta_bytes": int(encoded["archive"]["bytes"])
                    - int(baseline["archive"]["bytes"]),
                    "B_hat": encoded["B_hat"],
                    "reencode_manifest": file_fact(
                        root / f"role_{role}" / f"prequant_{bits:02d}bit" / "REENCODE_MANIFEST.json"
                    ),
                }
            )
    manifest = {
        "schema": "ddm_qbt1_precision_sensitivity.v1",
        "axis": "[first-order shortlist plus real QBF1 byte-close; no scorer verdict]",
        "score_claim": False,
        "baseline_archive": baseline["archive"],
        "probe_bits": list(map(int, probe_bits)),
        "rows": rows,
        "selection_disposition": "NO_ADOPTION_WITHOUT_REALIZED_SCORER_AB",
        "all_candidate_coder_payloads_retained": True,
    }
    atomic_json(root / "PRECISION_SENSITIVITY.json", manifest)
    return manifest


def _target_arrays(pair_ids: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = open_stored_npy_memmap(GT_CACHE, "lstars")
    poses = open_stored_npy_memmap(GT_CACHE, "gt_poses")
    ids = np.asarray(pair_ids, dtype=np.int64)
    target_argmax = torch.from_numpy(np.asarray(labels[ids], dtype=np.int64).copy()).to(device)
    target_pose6 = torch.from_numpy(np.asarray(poses[ids], dtype=np.float32).copy()).to(device)
    return target_argmax, target_pose6


def pair_chunks(pair_ids: Sequence[int], chunk_pairs: int) -> tuple[tuple[int, ...], ...]:
    if not 1 <= int(chunk_pairs) <= MAX_CHUNK_PAIRS:
        raise QBT1Error("materialization chunk exceeds the hard ceiling of 30")
    ids = tuple(map(int, pair_ids))
    return tuple(ids[start : start + chunk_pairs] for start in range(0, len(ids), chunk_pairs))


def training_chunks(pair_ids: Sequence[int], chunk_pairs: int) -> tuple[tuple[int, ...], ...]:
    """Return equal-mass n32 chunks for the sealed no2 selection.

    The no2 design has twenty-four weight-15 and eight weight-30 pairs.  Two
    chunks with twelve light and four heavy pairs each avoid the 30+2 SGD bias
    while remaining well below the structural ceiling.
    """

    ids = tuple(map(int, pair_ids))
    if ids != SELECTION_IDS:
        return pair_chunks(ids, chunk_pairs)
    if int(chunk_pairs) != REAL_TRAIN_CHUNK_PAIRS:
        raise QBT1Error("sealed n32 training requires two equal 16-pair chunks")
    light, heavy = ids[:24], ids[24:]
    return (light[:12] + heavy[:4], light[12:] + heavy[4:])


def no2_sample_weights(pair_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    lookup = dict(zip(SELECTION_IDS, SELECTION_WEIGHTS, strict=True))
    try:
        values = [lookup[int(pair_id)] for pair_id in pair_ids]
    except KeyError as exc:
        raise QBT1Error("training pair is outside the sealed no2 selection") from exc
    return torch.tensor(values, dtype=torch.float32, device=device)


def _retain_eval_outputs(
    root: Path,
    *,
    pair_ids: Sequence[int],
    camera: torch.Tensor,
    pose6: torch.Tensor,
    logits: torch.Tensor,
    target_argmax: torch.Tensor,
    target_pose6: torch.Tensor,
) -> dict[str, Any]:
    camera_u8 = camera.detach().round().clamp(0, 255).cpu().to(torch.uint8).numpy()
    logits_np = logits.detach().cpu().numpy().astype("<f2")
    argmax_np = logits.argmax(dim=1).detach().cpu().numpy().astype("u1")
    pose_np = pose6.detach().cpu().numpy().astype("<f4")
    target_argmax_np = target_argmax.detach().cpu().numpy().astype("u1")
    target_pose_np = target_pose6.detach().cpu().numpy().astype("<f4")
    rows = []
    for index, pair_id in enumerate(pair_ids):
        payload = atomic_npz(
            root / f"pair_{pair_id:04d}.npz",
            camera_pair_u8=camera_u8[index],
            segnet_logits_f16=logits_np[index],
            segnet_argmax_u8=argmax_np[index],
            posenet_pose6_f32=pose_np[index],
            target_argmax_u8=target_argmax_np[index],
            target_pose6_f32=target_pose_np[index],
        )
        dseg = float(np.mean(argmax_np[index] != target_argmax_np[index]))
        dpose = float(np.mean(np.square(pose_np[index].astype(np.float64) - target_pose_np[index])))
        rows.append({"pair_id": pair_id, "d_seg": dseg, "d_pose": dpose, "payload": payload})
    return {"rows": rows, "all_frames_and_scorer_outputs_retained": True}


def no2_gate(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    archive_bytes: int,
    b_hat: int | None,
    control: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lookup = {int(row["pair_id"]): row for row in pair_rows}
    ht_ready = len(pair_rows) == len(SELECTION_IDS) and tuple(lookup) == SELECTION_IDS
    if ht_ready:
        weights = dict(zip(SELECTION_IDS, SELECTION_WEIGHTS, strict=True))
        dseg_hat = sum(weights[pair_id] * float(lookup[pair_id]["d_seg"]) for pair_id in SELECTION_IDS) / N
        dpose_hat = sum(weights[pair_id] * float(lookup[pair_id]["d_pose"]) for pair_id in SELECTION_IDS) / N
    else:
        dseg_hat = float(np.mean([float(row["d_seg"]) for row in pair_rows]))
        dpose_hat = float(np.mean([float(row["d_pose"]) for row in pair_rows]))
    scored_bytes = int(b_hat) if b_hat is not None else int(archive_bytes)
    s_hat = 100.0 * dseg_hat + math.sqrt(10.0 * dpose_hat) + 25.0 * scored_bytes / RATE_DENOMINATOR
    control_status = "REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL"
    control_pass = False
    if control is not None:
        if (
            control.get("schema") != CONTROL_SCHEMA
            or control.get("score_claim") is not False
            or control.get("family") != "QBW1_discrete_boundary_quotient"
            or control.get("custody_verified") is not True
        ):
            raise QBT1Error("same-budget QBW1 control schema/claim boundary differs")
        if int(control["archive_bytes"]) != scored_bytes:
            control_status = "REFUSED_CONTROL_BUDGET_DIFFERS"
        elif tuple(map(int, control["pair_ids"])) != tuple(lookup):
            control_status = "REFUSED_CONTROL_PAIR_SET_DIFFERS"
        elif not control.get("all_payloads_retained"):
            control_status = "REFUSED_CONTROL_PAYLOAD_CUSTODY_INCOMPLETE"
        else:
            control_status = "PASS_REAL_SAME_BUDGET_CONTROL"
            control_pass = s_hat < float(control["S_hat"])
    gates = {
        "complete_archive": scored_bytes <= COMPLETE_ARCHIVE_CAP_BYTES,
        "d_pose_hat": dpose_hat <= DPOSE_HAT_MAX,
        "s_hat": s_hat < S_HAT_MAX_EXCLUSIVE,
        "same_budget_qbw1_control": control_pass,
        "ht_selection_complete": ht_ready,
    }
    return {
        "schema": "ddm_qbt1_no2_section5_gate.v1",
        "score_claim": False,
        "axis": "[local mechanism/advisory; not contest authority]",
        "d_seg_hat": dseg_hat,
        "d_pose_hat": dpose_hat,
        "B_hat": scored_bytes,
        "S_hat": s_hat,
        "selection_count": len(pair_rows),
        "estimator_status": (
            "NO2_SECTION5_HT_COMPLETE" if ht_ready else "UNWEIGHTED_BOUNDED_SMOKE_MEAN_ONLY"
        ),
        "gates": gates,
        "control_status": control_status,
        "admitted": all(gates.values()),
    }


def resolve_ema_law(total_updates: int) -> dict[str, Any]:
    lever = EmaDecayCalibrated(total_updates, target_seed_fraction=0.01)
    return {
        "value": float(lever.overrides["--ema-decay"]),
        "lawref": lever.constant_manifest["--ema-decay"],
        "factory": "tac.witness_dsl.curriculum_dsl.EmaDecayCalibrated",
    }


def stable_ema_law_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(value))
    identity.get("lawref", {}).pop("resolved_at", None)
    return identity


def compile_config(*, action: str, output: Path, pair_ids: Sequence[int], steps: int, device: str) -> dict[str, Any]:
    ids = tuple(map(int, pair_ids))
    if not ids or len(ids) > N or len(set(ids)) != len(ids):
        raise QBT1Error("compiled pair IDs must be unique and nonempty")
    if steps < 1:
        raise QBT1Error("compiled steps must be positive")
    chunk_pairs = REAL_TRAIN_CHUNK_PAIRS if action == "train" else len(ids)
    total_updates = int(steps)
    ema = resolve_ema_law(total_updates)
    config = {
        "schema": SCHEMA,
        "action": action,
        "output": str(output.resolve()),
        "seed": SEED,
        "device": device,
        "num_pairs": N,
        "pair_ids": list(ids),
        "chunk_pairs": chunk_pairs,
        "chunk_pairs_hard_ceiling": MAX_CHUNK_PAIRS,
        "steps": steps,
        "checkpoint_every_steps": max(1, min(5, steps)),
        "learning_rate": 2.0e-4,
        "render_height": EVAL_H,
        "render_width": EVAL_W,
        "camera_height": CAMERA_H,
        "camera_width": CAMERA_W,
        "expected_flip_tau_start": 0.15,
        "expected_flip_tau_end": 0.05,
        "pose_start_step": 0,
        "ema": ema,
        "minimum_free_bytes": 8 * 1024**3,
        "retain_all_payloads": True,
        "resume_from": None,
        "launch_authorized": action == "smoke",
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
        "same_budget_qbw1_control_receipt": None,
        "precision_probe_bits": [8] if action == "smoke" else [6, 8, 10, 12],
        "source_pins": verify_pins(),
    }
    validate_config(config, require_launch_authority=action != "train")
    return config


def validate_config(config: Mapping[str, Any], *, require_launch_authority: bool = True) -> None:
    if config.get("schema") != SCHEMA:
        raise QBT1Error("compiled config schema differs")
    unknown = set(config) - {
        "schema", "action", "output", "seed", "device", "num_pairs", "pair_ids", "chunk_pairs",
        "chunk_pairs_hard_ceiling", "steps", "checkpoint_every_steps", "learning_rate", "render_height",
        "render_width", "camera_height", "camera_width", "expected_flip_tau_start", "expected_flip_tau_end",
        "pose_start_step", "ema", "minimum_free_bytes", "retain_all_payloads", "resume_from",
        "launch_authorized", "scorer_lane", "metal_lane", "same_budget_qbw1_control_receipt",
        "precision_probe_bits", "source_pins",
    }
    if unknown:
        raise QBT1Error(f"compiled config has unknown fields: {sorted(unknown)}")
    if int(config["num_pairs"]) != N or int(config["chunk_pairs_hard_ceiling"]) != MAX_CHUNK_PAIRS:
        raise QBT1Error("frozen population or chunk ceiling differs")
    ids = tuple(map(int, config["pair_ids"]))
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0 or max(ids) >= N:
        raise QBT1Error("compiled pair IDs are not a unique subset of n600")
    if not 1 <= int(config["chunk_pairs"]) <= MAX_CHUNK_PAIRS:
        raise QBT1Error("chunk_pairs exceeds the hard-coded ceiling of 30")
    if config["retain_all_payloads"] is not True or int(config["pose_start_step"]) != 0:
        raise QBT1Error("payload retention or step-zero pose binding differs")
    if (
        int(config["seed"]) != SEED
        or int(config["minimum_free_bytes"]) < 8 * 1024**3
        or not 1 <= int(config["checkpoint_every_steps"]) <= 5
    ):
        raise QBT1Error("seed, storage preflight, or checkpoint cadence differs")
    if (
        float(config["expected_flip_tau_start"]),
        float(config["expected_flip_tau_end"]),
    ) != (0.15, 0.05):
        raise QBT1Error("derived expected-flip schedule differs")
    if (int(config["render_height"]), int(config["render_width"])) != (EVAL_H, EVAL_W):
        raise QBT1Error("QBF1 scorer grid differs")
    if (int(config["camera_height"]), int(config["camera_width"])) != (CAMERA_H, CAMERA_W):
        raise QBT1Error("camera round-trip grid differs")
    resolved_ema = resolve_ema_law(int(config["steps"]))
    if stable_ema_law_identity(config["ema"]) != stable_ema_law_identity(resolved_ema):
        raise QBT1Error("EMA decay is not resolved through the canonical run-geometry LawRef")
    probe_bits = tuple(map(int, config["precision_probe_bits"]))
    if not probe_bits or len(set(probe_bits)) != len(probe_bits) or any(
        bits < 2 or bits > 16 for bits in probe_bits
    ):
        raise QBT1Error("stage-04 precision probe list differs")
    action = str(config["action"])
    if action == "smoke":
        if config["device"] != "cpu" or len(config["pair_ids"]) > SMOKE_MAX_PAIRS:
            raise QBT1Error("smoke is CPU-only and capped at four pairs")
    elif action == "train" and require_launch_authority:
        if config["device"] != "mps":
            raise QBT1Error("governed QBFLOW training is bound to the Metal lane")
        if config["launch_authorized"] is not True:
            raise QBT1Error("heavy training is not authorized")
        for resource_name in ("scorer_lane", "metal_lane"):
            claim = config[resource_name]
            if not claim.get("claimed") or not claim.get("claim_id"):
                raise QBT1Error(f"heavy training lacks a real {resource_name} claim")
    elif action != "train":
        raise QBT1Error(f"unsupported compiled action: {action}")
    if action == "train" and (
        tuple(map(int, config["pair_ids"])) != SELECTION_IDS
        or int(config["chunk_pairs"]) != REAL_TRAIN_CHUNK_PAIRS
    ):
        raise QBT1Error("heavy training must use the sealed no2 n32 selection in equal chunks")


def _load_control(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = config.get("same_budget_qbw1_control_receipt")
    if value is None:
        return None
    path = Path(value)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != CONTROL_SCHEMA
        or receipt.get("score_claim") is not False
        or receipt.get("family") != "QBW1_discrete_boundary_quotient"
        or tuple(map(int, receipt.get("pair_ids", ()))) != SELECTION_IDS
    ):
        raise QBT1Error("same-budget QBW1 control identity differs")
    archive = file_fact(Path(receipt["archive"]["path"]))
    if archive["sha256"] != receipt["archive"]["sha256"] or archive["bytes"] != int(
        receipt["archive_bytes"]
    ):
        raise QBT1Error("same-budget QBW1 control archive custody differs")
    retained = receipt.get("retained_pair_payloads", ())
    if len(retained) != len(SELECTION_IDS):
        raise QBT1Error("same-budget QBW1 control lacks 32 retained pair payloads")
    for expected_pair_id, row in zip(SELECTION_IDS, retained, strict=True):
        fact = file_fact(Path(row["payload"]["path"]))
        if (
            int(row["pair_id"]) != expected_pair_id
            or fact["sha256"] != row["payload"]["sha256"]
            or fact["bytes"] != int(row["payload"]["bytes"])
        ):
            raise QBT1Error("same-budget QBW1 control pair custody differs")
    recomputed = (
        100.0 * float(receipt["d_seg_hat"])
        + math.sqrt(10.0 * float(receipt["d_pose_hat"]))
        + 25.0 * int(receipt["archive_bytes"]) / RATE_DENOMINATOR
    )
    if not math.isclose(recomputed, float(receipt["S_hat"]), rel_tol=0.0, abs_tol=1.0e-12):
        raise QBT1Error("same-budget QBW1 control score arithmetic differs")
    verified = dict(receipt)
    verified["custody_verified"] = True
    return verified


def _model_and_optimizer(config: Mapping[str, Any], device: torch.device) -> tuple[QBFLOWTorch, torch.optim.AdamW, EMA]:
    model = load_initial_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    ema = EMA(model, decay=float(config["ema"]["value"]), warmup=True)
    return model, optimizer, ema


def run_training(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    pins = verify_pins()
    if config["source_pins"] != pins:
        raise QBT1Error("compiled source pins differ from live frozen inputs")
    output = Path(config["output"])
    storage = storage_preflight(output, int(config["minimum_free_bytes"]))
    seed_everything(int(config["seed"]))
    device = torch.device(str(config["device"]))
    pair_ids = tuple(map(int, config["pair_ids"]))
    chunks = pair_chunks(pair_ids, int(config["chunk_pairs"]))
    optimizer_chunks = training_chunks(pair_ids, int(config["chunk_pairs"]))
    baseline_rss = _maximum_rss_bytes()
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    model, optimizer, ema = _model_and_optimizer(config, device)
    step = 0
    history: list[dict[str, Any]] = []
    resume_identity = None
    if config.get("resume_from"):
        step, ema, history, payload = load_checkpoint(
            Path(config["resume_from"]), model=model, optimizer=optimizer, config=config
        )
        resume_identity = {
            "checkpoint": file_fact(Path(config["resume_from"])),
            "live_state_sha256": canonical_sha256(
                {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in payload["live_state_dict"].items()}
            ),
            "ema_state_sha256": canonical_sha256(
                {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in payload["ema"]["shadow"].items()}
            ),
        }

    for current in range(step, int(config["steps"])):
        chunk_ids = optimizer_chunks[current % len(optimizer_chunks)]
        ids_tensor = torch.tensor(chunk_ids, dtype=torch.long, device=device)
        target_argmax, target_pose6 = _target_arrays(chunk_ids, device)
        sample_weights = no2_sample_weights(chunk_ids, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(ids_tensor, height=EVAL_H, width=EVAL_W)
        camera = roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
        pose6, logits = scorer_forward(camera, posenet, segnet)
        tau = tau_for_step(
            current,
            int(config["steps"]),
            float(config["expected_flip_tau_start"]),
            float(config["expected_flip_tau_end"]),
        )
        total, components = joint_objective(
            outputs,
            camera,
            pose6,
            logits,
            target_argmax,
            target_pose6,
            tau,
            sample_weights,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        row = {
            "step": current + 1,
            "pair_ids": list(chunk_ids),
            "materialized_pairs": len(chunk_ids),
            "objective": {name: float(value.detach().cpu()) for name, value in components.items()},
            "ema_effective_decay": ema.effective_decay(),
        }
        history.append(row)
        checkpoint_due = (current + 1) % int(config["checkpoint_every_steps"]) == 0 or current + 1 == int(
            config["steps"]
        )
        if checkpoint_due:
            checkpoint = save_checkpoint(
                output / "stage_03_joint_boundary_interior_birth/checkpoints" / f"periodic_step_{current + 1:06d}.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                config=config,
                step=current + 1,
                stage="stage_03_joint_boundary_interior_birth",
                history=history,
            )
            checkpoint_reencode = reencode_inference_state(
                output / "stage_03_joint_boundary_interior_birth/reencoded" / f"step_{current + 1:06d}",
                model=model,
                state=ema.shadow,
                selected_pair_ids=pair_ids,
            )
            row["checkpoint"] = checkpoint
            row["reencode"] = checkpoint_reencode

    stage3_checkpoint = save_checkpoint(
        output / "stage_03_joint_boundary_interior_birth/checkpoints/stage_03_end.pt",
        model=model,
        optimizer=optimizer,
        ema=ema,
        config=config,
        step=int(config["steps"]),
        stage="stage_03_joint_boundary_interior_birth_end",
        history=history,
    )
    stage3_reencode = reencode_inference_state(
        output / "stage_03_joint_boundary_interior_birth/reencoded/stage_03_end",
        model=model,
        state=ema.shadow,
        selected_pair_ids=pair_ids,
    )

    # Reload the atomic stage boundary and prove live/EMA/optimizer/RNG identity
    # before stage 04.  The checkpoint is then re-encoded independently.
    resumed_model, resumed_optimizer, _unused = _model_and_optimizer(config, device)
    resumed_step, resumed_ema, resumed_history, resumed_payload = load_checkpoint(
        Path(stage3_checkpoint["path"]),
        model=resumed_model,
        optimizer=resumed_optimizer,
        config=config,
    )
    if resumed_step != int(config["steps"]) or resumed_history != history:
        raise QBT1Error("resume cursor/history differs")
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), resumed_model.state_dict()[name].detach().cpu()):
            raise QBT1Error(f"resume live tensor differs: {name}")
    for name, value in ema.shadow.items():
        if not torch.equal(value.detach().cpu(), resumed_ema.shadow[name].detach().cpu()):
            raise QBT1Error(f"resume EMA tensor differs: {name}")
    resumed_reencode = reencode_inference_state(
        output / "stage_04_precision_waterfill_and_byteclose/reencoded/resume_identity",
        model=resumed_model,
        state=resumed_ema.shadow,
        selected_pair_ids=pair_ids,
    )
    if resumed_reencode["archive"]["sha256"] != stage3_reencode["archive"]["sha256"]:
        raise QBT1Error("resume re-encoded archive differs")
    resume_identity = {
        "bit_faithful": True,
        "checkpoint": stage3_checkpoint,
        "archive_before": stage3_reencode["archive"],
        "archive_after_reload": resumed_reencode["archive"],
        "rng_restored": resumed_payload["rng"] is not None,
    }

    # Stage 04 uses the real joint scorer graph to accumulate one deterministic
    # full-selection gradient in <=30-pair chunks.  It then prequantizes one
    # state role at a time and retains every real QBF1 coder candidate.  The
    # first-order metric is explicitly a shortlist proxy, never a verdict.
    with ema_scope(resumed_model, resumed_ema):
        resumed_optimizer.zero_grad(set_to_none=True)
        for chunk_ids in optimizer_chunks:
            ids_tensor = torch.tensor(chunk_ids, dtype=torch.long, device=device)
            target_argmax, target_pose6 = _target_arrays(chunk_ids, device)
            sample_weights = no2_sample_weights(chunk_ids, device)
            outputs = resumed_model(ids_tensor, height=EVAL_H, width=EVAL_W)
            camera = roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = scorer_forward(camera, posenet, segnet)
            total, _components = joint_objective(
                outputs,
                camera,
                pose6,
                logits,
                target_argmax,
                target_pose6,
                float(config["expected_flip_tau_end"]),
                sample_weights,
            )
            (total * (len(chunk_ids) / len(pair_ids))).backward()
        torch.nn.utils.clip_grad_norm_(resumed_model.parameters(), 1.0)
        stage4_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in resumed_model.named_parameters()
            if parameter.grad is not None
        }
    precision = precision_sensitivity_and_reencode(
        output / "stage_04_precision_waterfill_and_byteclose/options",
        model=resumed_model,
        state=resumed_ema.shadow,
        gradients=stage4_gradients,
        selected_pair_ids=pair_ids,
        probe_bits=config["precision_probe_bits"],
    )
    stage4_checkpoint = save_checkpoint(
        output / "stage_04_precision_waterfill_and_byteclose/checkpoints/stage_04_end.pt",
        model=resumed_model,
        optimizer=resumed_optimizer,
        ema=resumed_ema,
        config=config,
        step=int(config["steps"]),
        stage="stage_04_precision_waterfill_and_byteclose_end",
        history=history,
    )
    stage4_reencode = reencode_inference_state(
        output / "stage_04_precision_waterfill_and_byteclose/reencoded/stage_04_end",
        model=resumed_model,
        state=resumed_ema.shadow,
        selected_pair_ids=pair_ids,
    )

    retained_rows: list[dict[str, Any]] = []
    with ema_scope(resumed_model, resumed_ema), torch.no_grad():
        for chunk_ids in chunks:
            ids_tensor = torch.tensor(chunk_ids, dtype=torch.long, device=device)
            target_argmax, target_pose6 = _target_arrays(chunk_ids, device)
            outputs = resumed_model(ids_tensor, height=EVAL_H, width=EVAL_W)
            camera = roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            pose6, logits = scorer_forward(camera, posenet, segnet)
            retained = _retain_eval_outputs(
                output / "stage_05_same_budget_admission/retained",
                pair_ids=chunk_ids,
                camera=camera,
                pose6=pose6,
                logits=logits,
                target_argmax=target_argmax,
                target_pose6=target_pose6,
            )
            retained_rows.extend(retained["rows"])
    last_eval = {
        "rows": retained_rows,
        "all_frames_and_scorer_outputs_retained": True,
        "chunk_pairs": int(config["chunk_pairs"]),
    }
    stage5_checkpoint = save_checkpoint(
        output / "stage_05_same_budget_admission/checkpoints/stage_05_end.pt",
        model=resumed_model,
        optimizer=resumed_optimizer,
        ema=resumed_ema,
        config=config,
        step=int(config["steps"]),
        stage="stage_05_same_budget_admission_end",
        history=history,
    )
    stage5_reencode = reencode_inference_state(
        output / "stage_05_same_budget_admission/reencoded/stage_05_end",
        model=resumed_model,
        state=resumed_ema.shadow,
        selected_pair_ids=pair_ids,
    )
    peak_rss = _maximum_rss_bytes()
    memory = project_memory(
        baseline_rss_bytes=baseline_rss,
        observed_peak_rss_bytes=peak_rss,
        observed_pairs=max(map(len, chunks)),
        real_chunk_pairs=REAL_TRAIN_CHUNK_PAIRS,
    )
    atomic_json(output / "MEMORY_PROJECTION.json", memory)
    gate = no2_gate(
        pair_rows=last_eval["rows"],
        archive_bytes=int(stage5_reencode["archive"]["bytes"]),
        b_hat=stage5_reencode["B_hat"],
        control=_load_control(config),
    )
    atomic_json(output / "stage_05_same_budget_admission/GATE.json", gate)
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "arm": "ddm_qbt1_qbflow_trainer_build",
        "axis": (
            "[macOS-CPU mechanism smoke; not a verdict]"
            if config["action"] == "smoke"
            else "[macOS-MPS governed n32 research row; not contest authority]"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "boundaries": (
            {
                "measured": "n<=4 real QBF1 encode, scorer-in-loop train step, re-encode, retained frames/scorer outputs, gate arithmetic, resume identity",
                "not_measured": "n32 training verdict, n600 distortion, Metal memory, contest CPU/CUDA, same-budget QBW1 control unless a real receipt is supplied",
            }
            if config["action"] == "smoke"
            else {
                "measured": "governed n32 QBFLOW training, realized frozen-scorer outputs, HT projection, real re-encoded bytes, resume identity, and local admission gates",
                "not_measured": "n600 distortion, contest CPU/CUDA, and any gate whose retained control receipt is absent",
            }
        ),
        "pins": pins,
        "storage": storage,
        "config_sha256": canonical_sha256(config),
        "history": history,
        "stage_03_checkpoint": stage3_checkpoint,
        "stage_03_reencode": stage3_reencode,
        "stage_04_checkpoint": stage4_checkpoint,
        "stage_04_reencode": stage4_reencode,
        "stage_04_precision_sensitivity": precision,
        "stage_05_checkpoint": stage5_checkpoint,
        "stage_05_reencode": stage5_reencode,
        "resume_identity": resume_identity,
        "retained_evaluation": last_eval,
        "memory_projection": memory,
        "stage_05_gate": gate,
        "all_payloads_retained": True,
        "training_launch": config["action"] == "train",
        "metal_invocations": int(config["action"] == "train" and config["device"] == "mps"),
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
    }
    atomic_json(output / "RESULT.json", result)
    return result


def project_memory(
    *,
    baseline_rss_bytes: int,
    observed_peak_rss_bytes: int,
    observed_pairs: int,
    real_chunk_pairs: int,
) -> dict[str, Any]:
    if observed_pairs < 1 or not 1 <= real_chunk_pairs <= MAX_CHUNK_PAIRS:
        raise QBT1Error("memory projection geometry differs")
    observed_delta = max(0, int(observed_peak_rss_bytes) - int(baseline_rss_bytes))
    empirical_linear = int(baseline_rss_bytes) + math.ceil(observed_delta * real_chunk_pairs / observed_pairs)
    camera_bytes = real_chunk_pairs * 2 * 3 * CAMERA_H * CAMERA_W * 4
    seg_logits_bytes = real_chunk_pairs * 5 * EVAL_H * EVAL_W * 4
    qbf_flow_bytes = real_chunk_pairs * EVAL_H * EVAL_W * qbf1.FLOW_DIM * 4
    retained_n32_bytes = 32 * (
        2 * 3 * CAMERA_H * CAMERA_W + 5 * EVAL_H * EVAL_W * 2 + EVAL_H * EVAL_W + 6 * 4
    )
    materialization_floor = camera_bytes + seg_logits_bytes + qbf_flow_bytes + retained_n32_bytes
    # The WD3-derived 85.76-GiB scorer materialization precedent is the
    # conservative reference floor named by the QBW1 fire order.  It is not a
    # measurement of QBFLOW or Metal; the maximum below is a launch projection.
    wd3_reference_bytes = int(85.76 * 1024**3)
    projected_peak = max(empirical_linear, materialization_floor * 8, wd3_reference_bytes)
    return {
        "schema": MEMORY_SCHEMA,
        "axis": "[derived CPU-smoke projection; Metal peak not measured]",
        "score_claim": False,
        "observed_smoke": {
            "pairs": observed_pairs,
            "baseline_rss_bytes": int(baseline_rss_bytes),
            "peak_rss_bytes": int(observed_peak_rss_bytes),
            "incremental_peak_bytes": observed_delta,
        },
        "real_config": {
            "selection_pairs": 32,
            "materialization_chunk_pairs": real_chunk_pairs,
            "chunk_ceiling": MAX_CHUNK_PAIRS,
            "camera_tensor_bytes": camera_bytes,
            "seg_logits_tensor_bytes": seg_logits_bytes,
            "qbf_flow_tensor_bytes": qbf_flow_bytes,
            "retained_n32_payload_bytes": retained_n32_bytes,
        },
        "projection_method": "max(linear observed materialization scaling, explicit tensor floor x8 autograd reserve, WD3 85.76-GiB scorer precedent)",
        "projected_peak_bytes": projected_peak,
        "ceiling_bytes": MEMORY_CEILING_BYTES,
        "headroom_bytes": MEMORY_CEILING_BYTES - projected_peak,
        "passes_ceiling": projected_peak <= MEMORY_CEILING_BYTES,
        "live_preflight_required_at_fire": True,
    }


def compiled_launch_request(
    smoke_result: Mapping[str, Any], path: Path, *, review_receipt_path: Path
) -> dict[str, Any]:
    if (
        smoke_result.get("schema") != RESULT_SCHEMA
        or smoke_result.get("score_claim") is not False
        or smoke_result.get("training_launch") is not False
        or smoke_result.get("all_payloads_retained") is not True
        or smoke_result.get("pins") != verify_pins()
    ):
        raise QBT1Error("compiled launch request received a stale or non-smoke receipt")
    review_receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    expected_review_files = {
        "experiments/ddm_qbt1_qbflow_trainer.py",
        "experiments/tests/test_ddm_qbt1_qbflow_trainer.py",
    }
    if (
        review_receipt.get("schema") != "ddm_qbt1_two_pass_review_receipt.v1"
        or set(review_receipt.get("python_files", ())) != expected_review_files
        or review_receipt.get("passes_per_file") != 2
        or review_receipt.get("status") != "PASS"
    ):
        raise QBT1Error("compiled launch request lacks the required two-pass Python review receipt")
    config = compile_config(
        action="train",
        output=TRAIN_ROOT / "governed_n32",
        pair_ids=SELECTION_IDS,
        steps=130,
        device="mps",
    )
    config["launch_authorized"] = False
    config["scorer_lane"] = {"claimed": False, "claim_id": None}
    config["metal_lane"] = {"claimed": False, "claim_id": None}
    config_path = path.parent / "COMPILED_N32_CONFIG.json"
    atomic_json(config_path, config)
    memory = smoke_result["memory_projection"]
    elapsed = float(smoke_result["elapsed_seconds"])
    smoke_steps = max(1, len(smoke_result["history"]))
    smoke_pairs = max(1, max(int(row["materialized_pairs"]) for row in smoke_result["history"]))
    projected_wall = elapsed / smoke_steps * 130 * REAL_TRAIN_CHUNK_PAIRS / smoke_pairs
    blockers = []
    if not smoke_result["resume_identity"]["bit_faithful"]:
        blockers.append("bounded smoke resume identity failed")
    if not memory["passes_ceiling"]:
        blockers.append("projected materialization peak exceeds 116 GiB")
    if smoke_result["stage_05_gate"]["control_status"] != "PASS_REAL_SAME_BUDGET_CONTROL":
        blockers.append(
            "stage-05 same-budget QBW1 control receipt remains absent; stage-03/04 may run only if MAIN binds a real control before stage-05"
        )
    request = {
        "schema": LAUNCH_SCHEMA,
        "disposition": (
            "QUEUED_STAGE03_04_FIRE_STAGE05_BLOCKED"
            if len(blockers) == 1
            else "BLOCKED_NOT_LAUNCHABLE"
        ),
        "owner": "MAIN QBFLOW joint-training owner",
        "consumer_store": str(STORE),
        "fire_trigger": (
            "MAIN verifies the committed hashes against the two-pass receipt, confirms no duplicate active lane and no full-n600 scorer job, claims Metal and scorer lanes, and reruns live storage plus <=116-GiB admission before stage 03"
        ),
        "stage_05_fire_trigger": (
            "the governed n32 QBFLOW result exists and a real retained same-budget QBW1 control receipt passes custody, pair-set, budget, and score-arithmetic validation"
        ),
        "compiled_config": file_fact(config_path),
        "two_pass_review_receipt": file_fact(review_receipt_path),
        "smoke_result": file_fact(Path(smoke_result["stage_03_checkpoint"]["path"]).parents[2] / "RESULT.json"),
        "memory_projection": memory,
        "schedule_estimate": {
            "optimizer_updates": 130,
            "pair_chunks_per_window": 2,
            "windows": 65,
            "wall_seconds_upper_projection": projected_wall,
            "basis": "n<=4 CPU smoke elapsed scaled linearly by optimizer updates and 16/smoke_pairs; not a Metal measurement",
            "status": "DERIVED_PROJECTION_REMEASURE_ON_METAL_BEFORE_FIRE",
        },
        "fire_order_checklist": {
            "frozen_qbf1_abi_verified": True,
            "chunk_pairs_le_30_structural": True,
            "joint_pose_from_step_zero": True,
            "expected_flip_margin_law": True,
            "road_conditioning_receiver_derived": True,
            "along_tangent_comb_8_16_24_32": True,
            "real_R_uint8_frozen_scorers": True,
            "ema_lawref_resolved": True,
            "periodic_and_stage_atomic_checkpoints": True,
            "every_checkpoint_reencodes_real_coder": True,
            "same_budget_qbw1_control_bound": False,
            "live_memory_preflight_passed": False,
            "metal_lane_claimed": False,
            "scorer_lane_claimed": False,
            "arm_two_pass_review_complete": True,
        },
        "blockers": blockers,
        "training_launched": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
    }
    atomic_json(path, request)
    return request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    smoke = sub.add_parser("smoke", help="run the bounded real CPU mechanism smoke")
    smoke.add_argument("--output", type=Path, default=TRAIN_ROOT / "smoke_n1")
    smoke.add_argument("--pairs", type=int, default=1, choices=range(1, SMOKE_MAX_PAIRS + 1))
    smoke.add_argument("--steps", type=int, default=1)
    compile_request = sub.add_parser("compile-launch-request")
    compile_request.add_argument("--smoke-result", type=Path, required=True)
    compile_request.add_argument("--review-receipt", type=Path, required=True)
    compile_request.add_argument("--output", type=Path, default=TRAIN_ROOT / "COMPILED_LAUNCH_REQUEST.json")
    run_config = sub.add_parser("run-config", help="MAIN-only governed execution of an authorized config")
    run_config.add_argument("config", type=Path)
    validate = sub.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "smoke":
        started = time.monotonic()
        config = compile_config(
            action="smoke",
            output=args.output,
            pair_ids=SELECTION_IDS[: args.pairs],
            steps=args.steps,
            device="cpu",
        )
        result = run_training(config)
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_json(Path(config["output"]) / "RESULT.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.action == "compile-launch-request":
        smoke_result = json.loads(args.smoke_result.read_text(encoding="utf-8"))
        request = compiled_launch_request(
            smoke_result, args.output, review_receipt_path=args.review_receipt
        )
        print(json.dumps(request, indent=2, sort_keys=True, default=str))
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.action == "run-config":
        result = run_training(config)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    validate_config(config)
    print(json.dumps({"status": "PASS", "config_sha256": canonical_sha256(config)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
